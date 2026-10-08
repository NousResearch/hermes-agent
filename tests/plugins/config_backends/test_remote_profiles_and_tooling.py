"""Remote config: file tooling, doctor, profiles, the TUI and RPC surfaces."""
from __future__ import annotations

import pytest

from hermes_cli.config_backend import (
    ConfigBackendUnavailable,
    get_config_backend,
)
from plugins.config_backends import remote as remote_pkg


def test_file_tooling_refused(plane):
    from hermes_cli.config_backend import require_file_tooling
    with pytest.raises(ConfigBackendUnavailable, match="Profile clone"):
        require_file_tooling("Profile clone")


def test_doctor_reports_backend_status(plane, capsys):
    from hermes_cli.doctor_config import _check_config_backend
    from hermes_cli.doctor_report import Finding
    f = Finding()
    assert _check_config_backend(get_config_backend(), plane.home, f) is True
    assert "Remote Config" in capsys.readouterr().out
    plane.fail_status = 503
    remote_pkg._reset_for_tests()
    f2 = Finding()
    assert _check_config_backend(get_config_backend(), plane.home, f2) is False
    assert f2.issues


def test_general_plugin_manager_never_loads_config_backends(plane, monkeypatch):
    from hermes_cli.plugins import PluginManager
    monkeypatch.setenv("HERMES_CONFIG_BACKEND", "file")
    mgr = PluginManager()
    mgr.discover_and_load()
    assert not any("config_backends" in key or key == "remote" for key in mgr._plugins)


def test_fresh_profile_creation_succeeds_against_a_healthy_plane(profile_plane):
    """F6: the staging home is never read through the remote backend."""
    from hermes_cli.profiles import create_profile
    before = len(profile_plane.requests)

    path = create_profile("fresh", no_alias=True, no_skills=True)

    assert path.is_dir() and (path / ".env").exists()
    assert not list(path.parent.glob(".fresh.staging-*"))
    assert not any(r["profile"].startswith(".") for r in profile_plane.requests[before:])


def test_rename_is_refused_before_anything_moves(profile_plane, monkeypatch):
    """F7: the plane keys settings by profile name and has no rename."""
    from hermes_cli import profiles
    for name in ("_check_gateway_running", "_cleanup_gateway_service", "_maybe_unregister_gateway_service",
                 "_stop_bot_desktop", "_live_default_multiplexer", "_maybe_register_gateway_service"):
        monkeypatch.setattr(profiles, name, lambda *a, **k: False)
    old = profile_plane.home / "profiles" / "old"
    old.mkdir(parents=True)
    profile_plane.profile("old").update(values={"model": {"default": "profile-model"}}, version=1)
    profile_plane.profile("new").update(values={"model": {"default": "target-model"}}, version=1)

    with pytest.raises(ValueError, match="not supported"):
        profiles.rename_profile("old", "new")

    assert old.is_dir() and not (profile_plane.home / "profiles" / "new").exists()
    assert profile_plane.profile("old")["values"] == {"model": {"default": "profile-model"}}
    assert profile_plane.patches() == []


def test_roster_model_follows_the_remote_layer_not_the_local_file(profile_plane):
    """F9: a poll changes the profile's model while an ignored local config.yaml stays put."""
    from hermes_cli.profiles import list_profiles
    named = profile_plane.home / "profiles" / "roster"
    named.mkdir(parents=True)
    (named / "config.yaml").write_text("model:\n  default: ignored-local\n")
    profile_plane.profile("roster").update(values={"model": {"default": "model-A", "provider": "nous"}}, version=1)

    def roster():
        return {p.name: (p.model, p.provider) for p in list_profiles()}["roster"]

    assert roster() == ("model-A", "nous")
    profile_plane.profile("roster").update(values={"model": {"default": "model-B", "provider": "openrouter"}}, version=2)
    backend = get_config_backend()
    assert backend.poll_one(backend._state(named))
    assert roster() == ("model-B", "openrouter")


def _tui_server(monkeypatch, home):
    import hermes_cli.banner as banner
    monkeypatch.setattr(banner, "prefetch_update_check", lambda: None)
    from tui_gateway import server
    monkeypatch.setattr(server, "_hermes_home", home)
    server._cfg_cache = server._cfg_sig = server._cfg_path = None
    return server


def test_tui_config_set_of_a_locked_key_is_refused_not_reported_saved(plane, monkeypatch):
    """F10: a keyed TUI edit under a lock answers with the lock, sends nothing, and the raw cache
    keeps the accepted value."""
    plane.upper = {"display": {"tui_theme": "dark"}}
    plane.upper_locks = [{"path": "display.tui_theme", "level": "tenant"}]
    server = _tui_server(monkeypatch, plane.home)

    answer = server._methods["config.set"](1, {"key": "theme", "value": "light"})

    assert answer["error"]["code"] == 4002 and "locked" in answer["error"]["message"]
    assert plane.patches() == []
    assert server._load_cfg_raw()["display"]["tui_theme"] == "dark"

    plane.upper_locks = []
    backend = get_config_backend()
    assert backend.poll_one(backend._state(plane.home))
    assert server._methods["config.set"](2, {"key": "theme", "value": "light"})["result"]["value"] == "light"
    assert server._load_cfg_raw()["display"]["tui_theme"] == "light"
    assert plane.profile("default")["values"] == {"display": {"tui_theme": "light"}}


def test_rpc_refused_by_the_config_backend_still_gets_its_one_answer(plane, monkeypatch):
    """F11: ConfigBackendUnavailable is a SystemExit; an admitted request (inline or pooled) still
    answers once, with its own id."""
    import copy
    import threading

    server = _tui_server(monkeypatch, plane.home)

    def refused(rid, params):
        raise ConfigBackendUnavailable("Remote Config: cannot load the config for profile 'lazy'")

    monkeypatch.setitem(server._methods, "cc.refused", refused)
    inline = server.handle_request({"jsonrpc": "2.0", "id": "inline-1", "method": "cc.refused", "params": {}})
    assert inline["id"] == "inline-1" and "cannot load the config" in inline["error"]["message"]

    monkeypatch.setattr(server, "_LONG_HANDLERS", server._LONG_HANDLERS | {"cc.refused"})
    written = threading.Event()

    class Recorder:
        frames = []

        def write(self, obj):
            self.frames.append(copy.deepcopy(obj))
            written.set()
            return True

        def close(self):
            pass

    transport = Recorder()
    assert server.dispatch({"jsonrpc": "2.0", "id": "pooled-1", "method": "cc.refused", "params": {}}, transport) is None
    assert written.wait(10)
    (frame,) = transport.frames
    assert frame["id"] == "pooled-1" and "cannot load the config" in frame["error"]["message"]


def test_tui_section_and_prompt_setters_refuse_locked_edits(plane, monkeypatch):
    """Round 4 #6: prompt / reasoning / details_mode / details_mode.<section> under an upper lock
    answer with the lock, send nothing, and leave the session as it was. Unlocked, they write."""
    plane.upper = {"custom_prompt": "tenant-prompt",
                   "display": {"show_reasoning": False, "sections": {"thinking": "hidden"}, "details_mode": "collapsed"}}
    plane.upper_locks = [{"path": "custom_prompt", "level": "tenant"}, {"path": "display", "level": "group"}]
    server = _tui_server(monkeypatch, plane.home)
    session = {"show_reasoning": False, "session_key": "k"}
    server._sessions["s1"] = session
    try:
        _locked_then_unlocked_tui_setters(plane, server, session)
    finally:
        server._sessions.pop("s1", None)  # never torn down by the TUI fixture after the plane stops


def _locked_then_unlocked_tui_setters(plane, server, session):
    calls = [("prompt", "synthetic-new-prompt"), ("prompt", "clear"), ("reasoning", "show"),
             ("details_mode", "expanded"), ("details_mode.thinking", "expanded"), ("details_mode.thinking", "")]

    for rid, (key, value) in enumerate(calls):
        answer = server._methods["config.set"](rid, {"key": key, "value": value, "session_id": "s1"})
        assert answer.get("error", {}).get("code") == 4002, (key, value, answer)
        assert "locked" in answer["error"]["message"]
    assert plane.patches() == []
    assert session["show_reasoning"] is False
    raw = server._load_cfg_raw()
    assert raw["custom_prompt"] == "tenant-prompt" and raw["display"]["show_reasoning"] is False

    plane.upper_locks = []
    backend = get_config_backend()
    assert backend.poll_one(backend._state(plane.home))
    for rid, (key, value) in enumerate(calls[:1] + calls[2:5]):
        assert "result" in server._methods["config.set"](100 + rid, {"key": key, "value": value, "session_id": "s1"})
    stored = plane.profile("default")["values"]
    assert stored["custom_prompt"] == "synthetic-new-prompt"
    assert stored["display"]["show_reasoning"] is True and stored["display"]["details_mode"] == "expanded"
    assert stored["display"]["sections"]["thinking"] == "expanded"
    assert session["show_reasoning"] is True
    assert all("unset" not in p["body"] for p in plane.patches())


def test_tui_shared_metrics_set_refuses_a_locked_consent(plane, monkeypatch):
    """Round 4 #6 (same class): the consent setter is explicit too; a locked answer is refused,
    not saved-minus-the-lock while the consent bookkeeping records it."""
    plane.upper = {"telemetry": {"shared_metrics": {"enabled": False, "send": False}}}
    plane.upper_locks = [{"path": "telemetry.shared_metrics", "level": "tenant"}]
    server = _tui_server(monkeypatch, plane.home)
    recorded = []
    from hermes_cli import setup as setup_mod
    monkeypatch.setattr(setup_mod, "_record_send_consent_change", lambda **kw: recorded.append(kw))

    answer = server._methods["shared_metrics.set"](1, {"enabled": True, "send": True})

    assert answer["error"]["code"] == 4002 and "locked" in answer["error"]["message"]
    assert plane.patches() == [] and recorded == []
