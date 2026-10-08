"""Remote config review round 7: grouped edits are one write, and readers see only completed
in-memory migrations.

1. A grouped edit (a model switch's ``model.*`` keys, the TUI ``/focus`` and ``thinking_mode``
   setters) is ONE backend write: a lock on any of its keys refuses all of them, so a refused
   switch never leaves a half-applied route. Unlocked, the whole group lands.
2. While one reader migrates a legacy profile in memory (D12), a concurrent reader or writer
   never observes the unmigrated document: it gets completed migration output.
"""
from __future__ import annotations

import threading

import pytest

from hermes_cli.config_backend import ConfigLockedError, get_config_backend
from plugins.config_backends import remote as remote_pkg
from plugins.config_backends.remote import backend as backend_mod


def _tui_server(monkeypatch, home):
    import hermes_cli.banner as banner
    monkeypatch.setattr(banner, "prefetch_update_check", lambda: None)
    from tui_gateway import server
    monkeypatch.setattr(server, "_hermes_home", home)
    server._cfg_cache = server._cfg_sig = server._cfg_path = None
    return server


def _switch(model="new-model", provider="new-provider"):
    from hermes_cli.model_switch import ModelSwitchResult
    return ModelSwitchResult(success=True, new_model=model, target_provider=provider, provider_changed=True,
                             api_key="", base_url="", api_mode="", is_global=True)


# --- 1. grouped edits ----------------------------------------------------------------------

def test_locked_provider_refuses_the_whole_model_switch(plane):
    """A tenant lock on model.provider alone: the switch raises and model.default stays put."""
    from hermes_cli.config import read_raw_config
    from hermes_cli.model_switch import persist_model_selection
    plane.upper = {"model": {"default": "old-model", "provider": "old-provider"}}
    plane.upper_locks = [{"path": "model.provider", "level": "tenant"}]

    with pytest.raises(ConfigLockedError):
        persist_model_selection(_switch(), plane.home / "config.yaml")

    assert plane.patches() == []
    assert plane.profile("default") == {"values": {}, "version": 0, "writer": None}
    model = read_raw_config()["model"]
    assert (model["default"], model["provider"]) == ("old-model", "old-provider")


def test_unlocked_model_switch_is_one_patch_and_keeps_siblings(plane):
    from hermes_cli.model_switch import persist_model_selection
    plane.profile("default").update(values={"model": {"default": "old-model", "provider": "old-provider",
                                                      "model_slots": {"fast": "mini"}}}, version=1)

    persist_model_selection(_switch(), plane.home / "config.yaml")

    (patch,) = plane.patches()
    assert patch["body"]["expectedVersion"] == 1
    model = plane.profile("default")["values"]["model"]
    assert (model["default"], model["provider"]) == ("new-model", "new-provider")
    assert model["model_slots"] == {"fast": "mini"}


def test_file_backend_model_switch_still_writes_every_key(tmp_path, monkeypatch):
    """Control: the file backend keeps its key-by-key round-trip (comments, siblings intact)."""
    import hermes_yaml as yaml
    from hermes_cli.model_switch import persist_model_selection
    monkeypatch.delenv("HERMES_CONFIG_BACKEND", raising=False)
    remote_pkg._reset_for_tests()
    cfg = tmp_path / "config.yaml"
    cfg.write_text("# keep me\nmodel:\n  default: old\n  provider: old-p\n  api_mode: x\n  model_slots:\n"
                   "    fast: mini\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    persist_model_selection(_switch(), cfg)

    text = cfg.read_text(encoding="utf-8")
    model = yaml.safe_load(text)["model"]
    assert "# keep me" in text
    assert model == {"default": "new-model", "provider": "new-provider", "model_slots": {"fast": "mini"}}


def test_tui_focus_on_under_a_partial_lock_changes_nothing(plane, monkeypatch):
    """/focus on writes focus_saved_tool_progress + tool_progress + focus_view; a lock on
    focus_view alone refuses all three (no stash or mode left behind), and the session is unchanged."""
    plane.upper = {"display": {"tool_progress": "verbose", "focus_view": False}}
    plane.upper_locks = [{"path": "display.focus_view", "level": "tenant"}]
    monkeypatch.delenv("HERMES_TUI_TOOL_PROGRESS", raising=False)
    server = _tui_server(monkeypatch, plane.home)
    session = {"session_key": "k", "tool_progress_mode": "verbose", "focus_view": False}
    server._sessions["s1"] = session
    try:
        answer = server._methods["config.set"](1, {"key": "focus", "value": "on", "session_id": "s1"})

        assert answer["error"]["code"] == 4002 and "locked" in answer["error"]["message"]
        assert plane.patches() == []
        assert session == {"session_key": "k", "tool_progress_mode": "verbose", "focus_view": False}
        display = server._load_cfg_raw()["display"]
        assert display["tool_progress"] == "verbose" and "focus_saved_tool_progress" not in display

        plane.upper_locks = []
        backend = get_config_backend()
        assert backend.poll_one(backend._state(plane.home))
        answer = server._methods["config.set"](2, {"key": "focus", "value": "on", "session_id": "s1"})
    finally:
        server._sessions.pop("s1", None)
    assert answer["result"]["value"] == "on"
    assert session["focus_view"] is True and session["tool_progress_mode"] == "off"
    (patch,) = plane.patches()
    stored = plane.profile("default")["values"]["display"]
    assert stored == {"focus_saved_tool_progress": "verbose", "tool_progress": "off", "focus_view": True}
    assert "unset" not in patch["body"]


def test_tui_thinking_mode_under_a_partial_lock_changes_nothing(plane, monkeypatch):
    plane.upper = {"display": {"thinking_mode": "collapsed", "details_mode": "collapsed"}}
    plane.upper_locks = [{"path": "display.details_mode", "level": "group"}]
    server = _tui_server(monkeypatch, plane.home)

    answer = server._methods["config.set"](1, {"key": "thinking_mode", "value": "full"})

    assert answer["error"]["code"] == 4002 and "locked" in answer["error"]["message"]
    assert plane.patches() == []
    assert server._load_cfg_raw()["display"]["thinking_mode"] == "collapsed"

    plane.upper_locks = []
    backend = get_config_backend()
    assert backend.poll_one(backend._state(plane.home))
    assert "result" in server._methods["config.set"](2, {"key": "thinking_mode", "value": "full"})
    assert len(plane.patches()) == 1
    assert plane.profile("default")["values"]["display"] == {"thinking_mode": "full", "details_mode": "expanded"}


# --- 2. readers see only completed migrations -----------------------------------------------

def _legacy_profile(plane):
    latest = backend_mod._latest_config_version()
    assert latest > 46
    plane.profile("default").update(values={"compression": {"threshold_tokens": 256000}}, version=1, writer=46)
    return latest


def _pause_first_migration(backend, monkeypatch, name="migrator"):
    """Pause the named thread's private migration run after its steps ran, before it publishes."""
    entered, resume = threading.Event(), threading.Event()
    real = backend._run_private

    def paused(*args, **kwargs):
        out = real(*args, **kwargs)
        if threading.current_thread().name == name:
            entered.set()
            assert resume.wait(10), "the migrating thread was never released"
        return out

    monkeypatch.setattr(backend, "_run_private", paused)
    return entered, resume


def _start(target, name):
    failures, results = [], []

    def run():
        try:
            results.append(target())
        except BaseException as exc:  # noqa: BLE001 — surfaced by the caller
            failures.append(exc)

    t = threading.Thread(target=run, name=name)
    t.start()
    return t, results, failures


def test_concurrent_reader_never_sees_the_unmigrated_doc(plane, monkeypatch):
    from hermes_cli.config_backend import read_config_doc
    latest = _legacy_profile(plane)
    backend = get_config_backend()
    backend._state(plane.home)
    entered, resume = _pause_first_migration(backend, monkeypatch)
    cfg = plane.home / "config.yaml"

    migrator, first, failures = _start(lambda: read_config_doc(cfg), "migrator")
    try:
        assert entered.wait(10)
        second = read_config_doc(cfg)  # while the first migration is paused, unpublished
    finally:
        resume.set()
        migrator.join(10)
    assert not migrator.is_alive() and not failures, failures

    for doc in (second, first[0]):
        assert doc["_config_version"] == latest
        assert "threshold_tokens" not in (doc.get("compression") or {})
    assert plane.patches() == []


def test_concurrent_readonly_reader_never_sees_the_unmigrated_doc(plane, monkeypatch):
    from hermes_cli.config_backend import read_config_doc_readonly
    latest = _legacy_profile(plane)
    backend = get_config_backend()
    backend._state(plane.home)
    entered, resume = _pause_first_migration(backend, monkeypatch)
    cfg = plane.home / "config.yaml"

    migrator, _, failures = _start(lambda: read_config_doc_readonly(cfg), "migrator")
    try:
        assert entered.wait(10)
        doc = read_config_doc_readonly(cfg)
        assert doc["_config_version"] == latest and "threshold_tokens" not in (doc.get("compression") or {})
    finally:
        resume.set()
        migrator.join(10)
    assert not migrator.is_alive() and not failures, failures


def test_writer_during_a_migration_sends_only_its_edit(plane, monkeypatch):
    """Concurrent-writer control: the edit diffs against completed migration output, so the
    migration's removal is never sent (D12) and the stored legacy value survives."""
    from hermes_cli.config import read_raw_config, set_config_value
    from hermes_cli.config_backend import read_config_doc
    _legacy_profile(plane)
    backend = get_config_backend()
    backend._state(plane.home)
    entered, resume = _pause_first_migration(backend, monkeypatch)

    migrator, _, failures = _start(lambda: read_config_doc(plane.home / "config.yaml"), "migrator")
    try:
        assert entered.wait(10)
        set_config_value("display.personality", "pirate")
    finally:
        resume.set()
        migrator.join(10)
    assert not migrator.is_alive() and not failures, failures

    (patch,) = plane.patches()
    assert "unset" not in patch["body"] and "compression" not in str(patch["body"])
    assert "writerConfigVersion" not in patch["body"]  # option 1: the old stamp is kept
    assert plane.profile("default")["values"]["compression"] == {"threshold_tokens": 256000}
    later = read_raw_config()
    assert later["display"]["personality"] == "pirate"
    assert "threshold_tokens" not in (later.get("compression") or {})


def test_poll_installing_a_newer_doc_mid_read_is_migrated_before_it_is_returned(plane, monkeypatch):
    """A poll installs a fresh legacy doc between a reader completing the old one and taking its
    snapshot: the reader goes round again and still returns only migrated output."""
    from hermes_cli.config_backend import read_config_doc
    latest = _legacy_profile(plane)
    backend = get_config_backend()
    st = backend._state(plane.home)
    real = backend._completed
    polled = []

    def poll_after_completing(state):
        ready = real(state)
        if not polled:
            polled.append(1)
            plane.profile("default").update(values={"compression": {"threshold_tokens": 256000},
                                                    "display": {"personality": "pirate"}}, version=2)
            assert backend.poll_one(st) and st.postprocessed is False
        return ready

    monkeypatch.setattr(backend, "_completed", poll_after_completing)
    doc = read_config_doc(plane.home / "config.yaml")
    assert polled and doc["_config_version"] == latest
    assert doc["display"]["personality"] == "pirate"  # the newer doc, not the first one
    assert "threshold_tokens" not in (doc.get("compression") or {})


def test_migration_recursion_still_sees_its_private_copy(plane, monkeypatch):
    """Control: a migration step's own reads get its private copy without starting another pass."""
    from hermes_cli import config as config_mod
    from hermes_cli import config_migrations
    latest = backend_mod._latest_config_version()
    plane.profile("default").update(values={"old_key": 1}, version=1, writer=latest - 1)
    seen = []

    def step(current, results, quiet, **kwargs):
        doc = config_mod.read_raw_config()
        seen.append(doc.get("_config_version"))
        doc.pop("old_key", None)
        doc["new_key"] = 2
        config_mod._persist_migration(doc)

    monkeypatch.setattr(config_migrations, "run_migrations", step)
    doc = config_mod.read_raw_config()

    assert seen == [latest - 1]
    assert doc["new_key"] == 2 and "old_key" not in doc
    assert plane.patches() == []
