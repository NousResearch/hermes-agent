"""CLI-5 (#134889): shell hooks under the one-gateway runtime.

The gateway has no TTY and outlives every client, so (a) the classic CLI asks for first-use
consent on its own terminal before it opens its session, and (b) the gateway re-syncs the
profile's hooks on every local session open: added / approved hooks register, removed / revoked
ones stop firing, and the gateway itself never prompts.
"""
import json
import sys
from types import SimpleNamespace

import pytest

from agent import shell_hooks


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    for name in ("HERMES_ACCEPT_HOOKS", "HERMES_TUI_GATEWAY_URL", "HERMES_SAFE_MODE", "HERMES_IGNORE_USER_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    from hermes_cli import plugins
    monkeypatch.setattr(plugins.get_plugin_manager(), "_hooks", {})
    shell_hooks.reset_for_tests()
    yield h
    shell_hooks.reset_for_tests()


def _script(tmp_path, name):
    path = tmp_path / name
    path.write_text("#!/bin/sh\nprintf '{}'\n")
    path.chmod(0o755)
    return str(path)


def _cfg(*commands):
    return {"hooks": {"pre_tool_call": [{"command": c} for c in commands]}}


def _wired(event="pre_tool_call"):
    from hermes_cli.plugins import get_plugin_manager
    return sorted(cb.__name__ for cb in get_plugin_manager()._hooks.get(event, []))


def test_gateway_resync_follows_config_and_allowlist_without_prompting(home, tmp_path, monkeypatch):
    from gateway import run
    from gateway.session_policy import accept_launch_hooks, build_policy, refresh_profile_hooks
    a, b = _script(tmp_path, "a.sh"), _script(tmp_path, "b.sh")
    cfg = _cfg(a)
    monkeypatch.setattr(run, "_load_gateway_config", lambda: cfg)
    # A foreground gateway on a terminal still never asks: input() would hang the session open.
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("the gateway prompted"))
    policy = build_policy({"source": "cli", "cwd": str(tmp_path)}, {})

    refresh_profile_hooks(policy)
    assert _wired() == []  # not approved yet: skipped
    accept_launch_hooks(build_policy({"source": "cli", "cwd": str(tmp_path), "accept_hooks": True}, {}))
    refresh_profile_hooks(policy)  # a creation launch's --accept-hooks, then the open's re-sync
    assert _wired() == [f"shell_hook[pre_tool_call:{a}]"]

    # Approved on a client terminal after the gateway started (the allowlist is the hand-off).
    cfg["hooks"]["pre_tool_call"].append({"command": b})
    shell_hooks._record_approval("pre_tool_call", b)
    refresh_profile_hooks(policy)
    refresh_profile_hooks(policy)  # idempotent
    assert _wired() == [f"shell_hook[pre_tool_call:{a}]", f"shell_hook[pre_tool_call:{b}]"]

    # Unreadable inputs never disarm a guard: an empty config (the loader fails open to {}), or a
    # corrupt allowlist.
    shell_hooks.sync_from_config(dict)
    assert len(_wired()) == 2
    allowlist = home / shell_hooks.ALLOWLIST_FILENAME
    saved = allowlist.read_text()
    allowlist.write_text("{corrupt")
    refresh_profile_hooks(policy)
    assert len(_wired()) == 2
    allowlist.write_text(saved)
    # An opt-in in force approves every configured hook, as on main: kept even with no allowlist
    # entry (an approval whose write failed).
    c = _script(tmp_path, "c.sh")
    cfg["hooks"]["pre_tool_call"].append({"command": c})
    cfg["hooks_auto_accept"] = True
    refresh_profile_hooks(policy)
    assert f"shell_hook[pre_tool_call:{c}]" in _wired() and shell_hooks.allowlist_entry_for("pre_tool_call", c) is None
    cfg["hooks"]["pre_tool_call"].pop()
    del cfg["hooks_auto_accept"]
    shell_hooks.revoke(a)  # revoked: stops firing at the next session open
    refresh_profile_hooks(policy)
    assert _wired() == [f"shell_hook[pre_tool_call:{b}]"]
    cfg["hooks"]["pre_tool_call"] = []  # removed from config.yaml
    refresh_profile_hooks(policy)
    assert _wired() == []

    # A bypass (--safe-mode / --ignore-user-config) session leaves the registry alone.
    cfg["hooks"]["pre_tool_call"] = [{"command": b}]
    bypass = build_policy({"source": "cli", "cwd": str(tmp_path), "model": "m", "safe_mode": True,
                           "accept_hooks": True}, {})
    accept_launch_hooks(bypass)
    refresh_profile_hooks(bypass)
    assert _wired() == []


def test_resync_keeps_in_memory_consent_and_never_disarms_on_bad_input(home, tmp_path, monkeypatch):
    """Review of #139067: a launch's --accept-hooks survives a failed allowlist write; the gateway's
    own --accept-hooks covers hooks added later; a malformed hooks: entry never disarms a guard and
    a malformed allowlist record never breaks the valid ones; an older snapshot never lands after a
    newer one."""
    import threading
    from gateway import run
    from gateway.session_policy import accept_launch_hooks, build_policy, refresh_profile_hooks
    a, b = _script(tmp_path, "a.sh"), _script(tmp_path, "b.sh")
    cfg = _cfg(a)
    monkeypatch.setattr(run, "_load_gateway_config", lambda: cfg)
    policy = build_policy({"source": "cli", "cwd": str(tmp_path)}, {})
    # Unwritable allowlist: the launch's consent is kept in memory, as main registered it.
    with monkeypatch.context() as unwritable:
        unwritable.setattr(shell_hooks, "atomic_json_write", lambda *a, **k: (_ for _ in ()).throw(OSError("ro")))
        accept_launch_hooks(build_policy({"source": "cli", "cwd": str(tmp_path), "accept_hooks": True}, {}))
        refresh_profile_hooks(policy)
        refresh_profile_hooks(policy)
    assert _wired() == [f"shell_hook[pre_tool_call:{a}]"]

    # A malformed entry (a list written as a mapping, or a bad record) adds but never removes.
    allowlist = home / shell_hooks.ALLOWLIST_FILENAME
    allowlist.write_text(json.dumps({"approvals": [{"event": "pre_tool_call", "command": []},
                                                   {"event": "pre_tool_call", "command": b}]}))
    cfg["hooks"]["pre_tool_call"] = {"command": a}
    refresh_profile_hooks(policy)
    assert _wired() == [f"shell_hook[pre_tool_call:{a}]"]
    cfg["hooks"]["pre_tool_call"] = [{"command": a}, {"command": b}]
    refresh_profile_hooks(policy)
    assert len(_wired()) == 2

    # `gateway run --accept-hooks`: a hook added after start registers with no allowlist entry.
    c = _script(tmp_path, "c.sh")
    cfg["hooks"]["pre_tool_call"].append({"command": c})
    refresh_profile_hooks(policy)
    assert f"shell_hook[pre_tool_call:{c}]" not in _wired()
    shell_hooks.accept_for_process()
    refresh_profile_hooks(policy)
    assert f"shell_hook[pre_tool_call:{c}]" in _wired()
    shell_hooks.reset_for_tests()
    from hermes_cli import plugins
    plugins.get_plugin_manager()._hooks = {}

    # Ordering: a sync that read the old config first still applies before the newer one.
    old_read, release = threading.Event(), threading.Event()

    def stale():
        old_read.set()
        release.wait(5)
        return _cfg(a)

    first = threading.Thread(target=shell_hooks.sync_from_config, args=(stale,))
    first.start()
    old_read.wait(5)
    second = threading.Thread(target=shell_hooks.sync_from_config, args=(lambda: _cfg(b),))
    second.start()
    release.set()
    first.join(5)
    second.join(5)
    assert _wired() == [f"shell_hook[pre_tool_call:{b}]"]


def test_saved_launch_approval_stays_revocable_and_bad_records_are_skipped(home, tmp_path):
    a, b = _script(tmp_path, "a.sh"), _script(tmp_path, "b.sh")
    shell_hooks.accept_for_launch(_cfg(a))  # saved, so not held in memory
    shell_hooks.revoke(a)  # before any session open
    shell_hooks.sync_from_config(lambda: _cfg(a))
    assert _wired() == []
    (home / shell_hooks.ALLOWLIST_FILENAME).write_text(json.dumps({"approvals": [
        {"event": "pre_tool_call", "command": []}, {"event": "pre_tool_call", "command": b}]}))
    shell_hooks.sync_from_config(lambda: _cfg(b))
    assert _wired() == [f"shell_hook[pre_tool_call:{b}]"]


def test_unsaved_launch_consent_ends_with_the_config_entry(home, tmp_path, monkeypatch):
    a = _script(tmp_path, "a.sh")
    with monkeypatch.context() as unwritable:
        unwritable.setattr(shell_hooks, "atomic_json_write", lambda *a, **k: (_ for _ in ()).throw(OSError("ro")))
        shell_hooks.accept_for_launch(_cfg(a))
    shell_hooks.sync_from_config(lambda: _cfg(a))
    assert _wired() == [f"shell_hook[pre_tool_call:{a}]"]
    shell_hooks.sync_from_config(lambda: {"hooks": {}})  # removed from config.yaml
    shell_hooks.sync_from_config(lambda: _cfg(a))  # added back: needs consent again
    assert _wired() == []


def test_refused_plugin_only_entry_does_not_suspend_removal(home, tmp_path):
    a = _script(tmp_path, "a.sh")
    shell_hooks.register_from_config(_cfg(a), accept_hooks=True)
    shell_hooks.revoke(a)
    cfg = _cfg(a)
    cfg["hooks"]["transform_api_error_classification"] = [{"command": a}]
    shell_hooks.sync_from_config(lambda: cfg)
    assert _wired() == []


def test_register_never_reasks_for_a_wired_hook(home, tmp_path, monkeypatch):
    a = _script(tmp_path, "a.sh")
    shell_hooks.register_from_config(_cfg(a), accept_hooks=True)
    shell_hooks.revoke(a)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("re-prompted for a wired hook"))
    assert shell_hooks.register_from_config(_cfg(a)) == []


def test_classic_cli_asks_on_its_terminal_before_opening_the_session(home, tmp_path, monkeypatch):
    from hermes_cli import cli_init_mixin
    from hermes_cli.gateway_session_store import GatewaySessionStore
    import hermes_cli.config as config_mod
    hook = _script(tmp_path, "guard.sh")
    monkeypatch.setattr(config_mod, "load_config", lambda: _cfg(hook))
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
    answers = iter(["n", "y"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    opened = []

    class Store(GatewaySessionStore):
        def __init__(self):
            pass

        def open_session(self, cli, launch=None):
            # The gateway reads the allowlist when this lands: consent is recorded first.
            opened.append(shell_hooks.allowlist_entry_for("pre_tool_call", hook) is not None)
            return "local-x"

    monkeypatch.setattr("hermes_cli.gateway_tools.bind_session", lambda *a: None)
    for _ in range(2):
        cli = SimpleNamespace(_session_db=Store(), session_id="local-x", _resumed=False)
        cli_init_mixin.CLIInitMixin._ensure_gateway_session(cli)
    assert opened == [False, True]  # "n" records nothing, "y" lands before the second open

    # --accept-hooks / HERMES_ACCEPT_HOOKS approves without asking; a remote gateway is not asked for.
    other = _script(tmp_path, "other.sh")
    monkeypatch.setattr(config_mod, "load_config", lambda: _cfg(other))
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("prompted"))
    monkeypatch.setenv("HERMES_TUI_GATEWAY_URL", "ws://remote.example/api/ws")
    cli_init_mixin._record_shell_hook_consent()
    assert shell_hooks.allowlist_entry_for("pre_tool_call", other) is None
    monkeypatch.delenv("HERMES_TUI_GATEWAY_URL")
    monkeypatch.setenv("HERMES_IGNORE_USER_CONFIG", "1")
    cli_init_mixin._record_shell_hook_consent()
    assert shell_hooks.allowlist_entry_for("pre_tool_call", other) is None
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG")
    monkeypatch.setenv("HERMES_ACCEPT_HOOKS", "1")
    cli_init_mixin._record_shell_hook_consent()
    assert json.loads((home / shell_hooks.ALLOWLIST_FILENAME).read_text())["approvals"][-1]["command"] == other
