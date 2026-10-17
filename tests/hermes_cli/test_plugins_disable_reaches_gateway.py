"""``hermes plugins disable`` must reach the running gateway, or say a restart is needed (LC-3).

Under one gateway the daemon holds the plugin manager for every session, so "takes effect on next
session" was false: new sessions kept the disabled plugin's hooks and tools until a restart. The
command now asks the gateway to re-discover (the same ``reload-plugins`` verb enable uses).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import plugins_cmd
from hermes_cli.config import load_config, save_config


@pytest.fixture
def home(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    plugin_dir = hermes_home / "plugins" / "probe"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text("name: probe\nversion: '1.0'\n", encoding="utf-8")
    (plugin_dir / "__init__.py").write_text("def register(ctx):\n    pass\n", encoding="utf-8")
    (hermes_home / "config.yaml").write_text("plugins:\n  enabled: [probe]\n  disabled: []\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    def admit(enabled, disabled, **_kwargs):
        cfg = load_config()
        cfg["plugins"] = {"enabled": sorted(enabled), "disabled": sorted(disabled)}
        save_config(cfg)

    monkeypatch.setattr("hermes_cli.plugins_admission.admit_plugin_set_change", admit)
    return hermes_home


def _gateway(monkeypatch, answer, *, shared=None):
    """``answer`` from the profile's own socket; ``shared`` (when given) from the multiplexer's."""
    calls: list[tuple[Path, object]] = []

    def reload_gateway_plugins(target, *, profile_home=None, timeout=30.0):
        calls.append((Path(target), profile_home))
        return answer if profile_home is None else shared

    monkeypatch.setattr("gateway.control_socket.reload_gateway_plugins", reload_gateway_plugins)
    return calls


def test_disable_unloads_the_plugin_from_the_running_gateway(home, monkeypatch, capsys):
    calls = _gateway(monkeypatch, {"reloaded": True, "plugins": []})

    plugins_cmd.cmd_disable("probe")

    out = capsys.readouterr().out
    assert calls and calls[0][0].resolve() == home.resolve()
    assert "next session" not in out
    assert "running gateway reloaded" in out


def test_the_gateways_forced_rediscovery_drops_a_disabled_plugins_hook(home):
    """What the ``reload-plugins`` verb runs: the reloaded manager no longer offers the plugin's hook."""
    from hermes_cli import plugins

    (home / "plugins" / "probe" / "__init__.py").write_text(
        "def _probe_hook(**kwargs):\n    return None\n\n"
        "def register(ctx):\n    ctx.register_hook('on_session_start', _probe_hook)\n", encoding="utf-8")

    def _probe_hooked() -> bool:
        return any(cb.__name__ == "_probe_hook" for cb in plugins.iter_hook_callbacks("on_session_start"))

    plugins._reset_plugin_managers_for_tests()
    try:
        plugins.discover_plugins(force=True)
        assert _probe_hooked()
        cfg = load_config()
        cfg["plugins"] = {"enabled": [], "disabled": ["probe"]}
        save_config(cfg)
        plugins.discover_plugins(force=True)
        assert not _probe_hooked()
    finally:
        plugins._reset_plugin_managers_for_tests()


_UNSERVED = {"reloaded": False, "error": "home is not served by this gateway"}


def test_an_explicit_refusal_is_reported_not_called_silence(home, monkeypatch, capsys):
    _gateway(monkeypatch, _UNSERVED)

    plugins_cmd.cmd_disable("probe")

    out = " ".join(capsys.readouterr().out.split())
    assert "refused" in out and "home is not served by this gateway" in out
    assert "saved config" in out and "No running gateway answered" not in out


@pytest.fixture
def shared_root(tmp_path, monkeypatch):
    root = tmp_path / "shared-root"
    root.mkdir()
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    return root


def test_a_profile_without_its_own_gateway_reaches_the_shared_one(home, shared_root, monkeypatch, capsys):
    calls = _gateway(monkeypatch, None, shared={"reloaded": True, "plugins": []})

    plugins_cmd.cmd_disable("probe")

    assert [(t.resolve(), p and Path(str(p)).resolve()) for t, p in calls] == [
        (home.resolve(), None), (shared_root.resolve(), home.resolve())]
    assert "running gateway reloaded" in capsys.readouterr().out


def test_a_shared_gateway_refusing_an_unserved_home_is_reported(home, shared_root, monkeypatch, capsys):
    calls = _gateway(monkeypatch, None, shared=_UNSERVED)

    plugins_cmd.cmd_disable("probe")

    out = " ".join(capsys.readouterr().out.split())
    assert len(calls) == 2 and "home is not served by this gateway" in out
    assert "saved config" in out and "No running gateway answered" not in out


def test_disable_without_an_answering_gateway_says_a_restart_is_needed(home, monkeypatch, capsys):
    _gateway(monkeypatch, None)

    plugins_cmd.cmd_disable("probe")

    out = capsys.readouterr().out
    assert "next session" not in out
    assert "hermes gateway restart" in out


def test_disable_says_the_running_gateway_unloaded_it(home, monkeypatch, capsys):
    """Live, an open session stops running the disabled plugin's commands at once: not only new ones."""
    _gateway(monkeypatch, {"reloaded": True, "plugins": []})

    plugins_cmd.cmd_disable("probe")

    out = " ".join(capsys.readouterr().out.split())
    assert "running gateway reloaded its plugins and unloaded probe" in out, out
    assert "new sessions" not in out, out
    # callbacks the plugin already wired into platform adapters outlive the reload (plugin guide)
    assert "platform hooks it already wired up stay until" in out and "hermes gateway restart" in out, out


@pytest.mark.parametrize("reloaded", [True, False])
def test_enable_prints_only_the_outcome_that_applies(home, monkeypatch, capsys, reloaded):
    """``enable`` knows whether the running gateway took the plugin: it prints that outcome, never
    "Takes effect on next session." above a line saying it is active in the running gateway now."""
    from hermes_cli import plugins_activation
    save_config({**load_config(), "plugins": {"enabled": [], "disabled": ["probe"]}})
    outcome = {"gateway_reloaded": reloaded, "restart_required": not reloaded,
               "activation": {"activated_now": {"hooks": 1}} if reloaded else None}
    monkeypatch.setattr(plugins_activation, "activate_plugin_now", lambda *_a, **_k: outcome)

    plugins_cmd.cmd_enable("probe", allow_tool_override=False)

    out = " ".join(capsys.readouterr().out.split())
    assert "Plugin probe enabled." in out and "next session" not in out, out
    assert ("active in the running gateway now" in out) is reloaded, out
    assert ("hermes gateway restart" in out) is (not reloaded), out
