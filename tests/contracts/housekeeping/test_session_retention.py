"""``session_retention``: the work after a delete, prune or archive commits, and the automatic archive and prune passes, over the reference backend.

The cases pin the hook payload, the order hook, media collection, transcript-file removal, the result dicts and the log lines.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pytest

import session_retention
from hermes_constants import get_hermes_home
from persistence import SessionId
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession
from persistence.housekeeping.maintenance import PruneCriteria
from tests.contracts._scopes import MEDIA_ROOT

NOW = time.time()
DAY = 86400.0
SPILLOVER_PREFIX = "gateway.spillover.retired.v1."


def _payload(sid, started, ended=None, end_reason=None, parent=None, source="cli"):
    return {"id": sid, "source": source, "started_at": started, "ended_at": ended, "end_reason": end_reason, "model": "model-a",
            "parent_session_id": parent, "messages": [{"role": "user", "content": "hello", "timestamp": started},
                                                      {"role": "assistant", "content": "hi", "timestamp": started + 1}]}


def _seed(store, *payloads):
    assert store.maintenance.import_sessions(list(payloads)).ok


def _transcript_files(sessions_dir: Path, sid: str) -> None:
    sessions_dir.mkdir(parents=True, exist_ok=True)
    for name in (f"{sid}.json", f"{sid}.jsonl", f"session_{sid}.json", f"request_dump_{sid}_1.json"):
        (sessions_dir / name).write_text("{}")


def _files(sessions_dir: Path) -> list[str]:
    return sorted(p.name for p in sessions_dir.iterdir()) if sessions_dir.is_dir() else []


@pytest.fixture
def sessions_dir(tmp_path):
    return tmp_path / "sessions"


@pytest.fixture
def events(monkeypatch, sessions_dir):
    """The hook dispatches and media collections in call order, each with the transcript files present at that moment."""
    seen: list = []

    def hook(name, **payload):
        seen.append(["hook", name, payload, str(get_hermes_home()), _files(sessions_dir)])

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", hook)
    monkeypatch.setattr(session_retention, "collect_retired_media", lambda store, home: seen.append(["collect", _files(sessions_dir)]))
    return seen


class _Admin:
    """A physical admin handle recording its calls; ``lock=None`` answers as a lock held elsewhere."""

    def __init__(self, *, ratio=0.0, blockers=(), lock="lock-1", vacuum_error=None):
        self.ratio, self.blockers, self.lock, self.vacuum_error = ratio, tuple(blockers), lock, vacuum_error
        self.calls: list = []

    def try_acquire_auto_maintenance_lock(self):
        self.calls.append("acquire")
        return self.lock

    def release_auto_maintenance_lock(self, handle):
        self.calls.append(("release", handle))

    def freelist_ratio(self):
        self.calls.append("freelist_ratio")
        return self.ratio

    def vacuum_blockers(self):
        self.calls.append("vacuum_blockers")
        return self.blockers

    def vacuum(self):
        self.calls.append("vacuum")
        if self.vacuum_error:
            raise self.vacuum_error
        return 0


def _messages(caplog, *, debug=False):
    return [(r.levelname, r.getMessage()) for r in caplog.records if r.name == "hermes_state" and (debug or r.levelname != "DEBUG")]


# ── after_sessions_removed and emit_session_deleted ─────────────────────────


def test_delete_runs_hook_then_media_collection_then_transcript_removal(backend, scope_a, sessions_dir, events):
    store = backend.open(scope_a)
    _seed(store, _payload("s-del", NOW), _payload("s-other", NOW))
    _transcript_files(sessions_dir, "s-del")
    _transcript_files(sessions_dir, "s-other")
    removed = store.sessions.delete(SessionId("s-del"), media_root=MEDIA_ROOT)
    session_retention.after_sessions_removed(store, scope_a.profile_dir, removed, reason="delete", primary="s-del", sessions_dir=sessions_dir)
    every_file = ["request_dump_s-del_1.json", "request_dump_s-other_1.json", "s-del.json", "s-del.jsonl", "s-other.json", "s-other.jsonl",
                  "session_s-del.json", "session_s-other.json"]
    # CONTRACT: the hook names the requested session, runs under the store's own profile home, and both it and the media collection see the transcript files still present.
    assert events == [
        ["hook", "on_session_delete", {"session_id": "s-del", "reason": "delete", "profile": "profile-a", "sessions_dir": str(sessions_dir)},
         str(scope_a.profile_dir), every_file],
        ["collect", every_file],
    ]
    # CONTRACT: every file of the removed session goes afterwards; other sessions' files stay.
    assert _files(sessions_dir) == ["request_dump_s-other_1.json", "s-other.json", "s-other.jsonl", "session_s-other.json"]


def test_nothing_removed_still_collects_media(backend, scope_a, sessions_dir, events):
    store = backend.open(scope_a)
    _transcript_files(sessions_dir, "nope")
    removed = store.sessions.delete(SessionId("nope"), media_root=MEDIA_ROOT)
    session_retention.after_sessions_removed(store, scope_a.profile_dir, removed, reason="delete", primary="nope", sessions_dir=sessions_dir)
    # CONTRACT: no hook without a removed id, the collection runs anyway, no file is touched.
    assert removed == ()
    assert events == [["collect", ["nope.json", "nope.jsonl", "request_dump_nope_1.json", "session_nope.json"]]]
    assert len(_files(sessions_dir)) == 4


def test_prune_payload_carries_every_removed_id(backend, scope_a, sessions_dir, events):
    store = backend.open(scope_a)
    _seed(store, _payload("s-p1", NOW - 100 * DAY, NOW - 100 * DAY, "done"), _payload("s-p2", NOW - 100 * DAY, NOW - 100 * DAY, "done"))
    _transcript_files(sessions_dir, "s-p1")
    outcome = store.maintenance.prune_sessions(older_than_days=30, media_root=MEDIA_ROOT)
    session_retention.after_sessions_removed(store, scope_a.profile_dir, outcome.deleted_ids, reason="prune", sessions_dir=sessions_dir)
    files = ["request_dump_s-p1_1.json", "s-p1.json", "s-p1.jsonl", "session_s-p1.json"]
    # CONTRACT: without a primary the first sorted id stands in, and ``deleted_ids`` lists them all.
    assert outcome.count == 2
    assert events == [
        ["hook", "on_session_delete", {"session_id": "s-p1", "reason": "prune", "profile": "profile-a", "deleted_ids": ["s-p1", "s-p2"],
                                       "sessions_dir": str(sessions_dir)}, str(scope_a.profile_dir), files],
        ["collect", files],
    ]
    assert _files(sessions_dir) == []


def test_a_failing_hook_never_breaks_the_delete(monkeypatch, caplog, scope_a):
    def boom(name, **payload):
        raise RuntimeError("hook boom")

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", boom)
    caplog.set_level(logging.DEBUG, logger="hermes_state")
    session_retention.emit_session_deleted(scope_a.profile_dir, ["s-j"], reason="delete", primary="s-j")
    # CONTRACT: the dispatch fails open with one warning.
    assert _messages(caplog) == [("WARNING", "on_session_delete dispatch failed")]


# ── archive_sessions ──────────────────────────────────────────────────────────


def test_archive_sessions_archives_matching_tips_and_notifies_each(backend, scope_a, events):
    store = backend.open(scope_a)
    _seed(store, _payload("s-e1", NOW - 10 * DAY, NOW - 10 * DAY, "done"), _payload("s-e2", NOW - 9 * DAY, NOW - 9 * DAY, "done"),
          _payload("s-e3", NOW, NOW, "done"), _payload("s-e4", NOW - 10 * DAY, NOW - 10 * DAY, "done", source="tui"))
    # CONTRACT: the count of matches; one archive notification per newly hidden lineage, primary the matched tip.
    assert session_retention.archive_sessions(store, scope_a.profile_dir, 5, "cli") == 2
    assert events == [
        ["hook", "on_session_delete", {"session_id": sid, "reason": "archive", "profile": "profile-a"}, str(scope_a.profile_dir), []]
        for sid in ("s-e1", "s-e2")
    ]
    events.clear()
    # CONTRACT: idempotent: archived rows no longer match, nothing is notified.
    assert session_retention.archive_sessions(store, scope_a.profile_dir, 5, "cli") == 0
    assert events == []
    assert {sid: store.maintenance.export_session(SessionId(sid)).session.archived for sid in ("s-e1", "s-e2", "s-e3", "s-e4")} == {
        "s-e1": 1, "s-e2": 1, "s-e3": 0, "s-e4": 0}
    # CONTRACT: an explicit archived filter is kept; archiving an archived row again notifies nobody.
    assert session_retention.archive_sessions(store, scope_a.profile_dir, 5, "cli", criteria=PruneCriteria(archived=True)) == 2
    assert events == []


# ── run_auto_archive ──────────────────────────────────────────────────────────


def test_run_auto_archive_gates_on_the_interval_and_never_raises(backend, scope_a, caplog, monkeypatch, events):
    store = backend.open(scope_a)
    _seed(store, _payload("s-idle", NOW - 10 * DAY, NOW - 10 * DAY, "done"), _payload("s-fresh", NOW))
    caplog.set_level(logging.DEBUG, logger="hermes_state")
    # CONTRACT: the result dicts: a run, a gated rerun, a rerun past a corrupt stamp.
    assert session_retention.run_auto_archive(store, idle_days=3) == {"skipped": False, "archived": 1}
    assert float(store.state_meta.get("last_auto_archive")) >= NOW
    assert session_retention.run_auto_archive(store, idle_days=3) == {"skipped": True, "archived": 0}
    store.state_meta.set("last_auto_archive", "garbage")
    assert session_retention.run_auto_archive(store, idle_days=3) == {"skipped": False, "archived": 0}
    rows = {sid: store.maintenance.export_session(SessionId(sid)).session for sid in ("s-idle", "s-fresh")}
    # CONTRACT: only the idle lineage is hidden, stamped as the sweep's.
    assert {sid: row.archived for sid, row in rows.items()} == {"s-idle": 1, "s-fresh": 0}
    assert rows["s-idle"].auto_archived == 1
    store.state_meta.set("last_auto_archive", "")

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(store.maintenance, "list_idle_archive_candidates", boom)
    # CONTRACT: a failure is reported in the dict, never raised, and leaves the stamp alone.
    assert session_retention.run_auto_archive(store, idle_days=3) == {"skipped": False, "archived": 0, "error": "boom"}
    assert store.state_meta.get("last_auto_archive") == ""
    assert _messages(caplog, debug=True) == [("INFO", "state.db auto-archive: archived 1 session(s) idle >= 3 days"),
                                             ("WARNING", "state.db auto-archive failed: boom")]
    # CONTRACT: the idle sweep notifies no plugin.
    assert events == []


# ── run_auto_prune_and_vacuum ─────────────────────────────────────────────────


def _prune(store, scope, admin, **kwargs):
    return session_retention.run_auto_prune_and_vacuum(store, admin, store_home=scope.profile_dir, media_root=MEDIA_ROOT, **kwargs)


def test_run_auto_prune_refuses_a_negative_retention(backend, scope_a, caplog):
    admin = _Admin()
    caplog.set_level(logging.DEBUG, logger="hermes_state")
    assert _prune(backend.open(scope_a), scope_a, admin, retention_days=-1) == {"skipped": True, "pruned": 0, "closed": 0, "vacuumed": False}
    assert _messages(caplog) == [("WARNING", "state.db auto-maintenance skipped: sessions.retention_days=-1 is outside the allowed range "
                                             "(a whole number of days >= 0); set sessions.auto_prune: false to disable pruning")]
    # CONTRACT: refused before the maintenance lock is touched.
    assert admin.calls == []


def test_run_auto_prune_prunes_closes_and_vacuums(backend, scope_a, sessions_dir, events, caplog):
    store = backend.open(scope_a)
    _seed(store, _payload("s-old1", NOW - 100 * DAY, NOW - 100 * DAY, "done"), _payload("s-old2", NOW - 100 * DAY, NOW - 100 * DAY, "done"),
          _payload("s-open-cli", NOW - 100 * DAY), _payload("s-open-tui", NOW - 100 * DAY, source="tui"), _payload("s-recent", NOW, NOW, "done"))
    _transcript_files(sessions_dir, "s-old1")
    admin = _Admin(ratio=0.0)
    caplog.set_level(logging.DEBUG, logger="hermes_state")
    result = _prune(store, scope_a, admin, retention_days=90, sessions_dir=sessions_dir, min_vacuum_freelist_ratio=-1.0)
    assert result == {"skipped": False, "pruned": 2, "closed": 1, "vacuumed": True, "freelist_ratio": 0.0}
    assert _messages(caplog) == [("INFO", "state.db auto-maintenance: closed 1 stale open session(s), pruned 2 session(s) inactive for 90 days + VACUUM")]
    files = ["request_dump_s-old1_1.json", "s-old1.json", "s-old1.jsonl", "session_s-old1.json"]
    # CONTRACT: the prune's post-commit work in order, the transcript files last.
    assert events == [
        ["hook", "on_session_delete", {"session_id": "s-old1", "reason": "prune", "profile": "profile-a", "deleted_ids": ["s-old1", "s-old2"],
                                       "sessions_dir": str(sessions_dir)}, str(scope_a.profile_dir), files],
        ["collect", files],
    ]
    assert _files(sessions_dir) == []
    rows = {sid: store.sessions.get(SessionId(sid)) for sid in ("s-old1", "s-old2", "s-open-cli", "s-open-tui", "s-recent")}
    # CONTRACT: the stale open CLI row is closed as an orphan; the UI row is left open; the recent row is untouched.
    assert {sid: (row.ended_at is not None, row.end_reason) if row else None for sid, row in rows.items()} == {
        "s-old1": None, "s-old2": None, "s-open-cli": (True, "startup_orphan_reap"), "s-open-tui": (False, None), "s-recent": (True, "done")}
    assert {key: store.state_meta.get(key) is not None for key in ("last_auto_prune", "last_vacuum")} == {"last_auto_prune": True, "last_vacuum": True}
    assert admin.calls == ["acquire", "freelist_ratio", "vacuum_blockers", "vacuum", ("release", "lock-1")]
    # CONTRACT: a second pass inside the interval is skipped, still under the lock.
    admin.calls.clear()
    assert _prune(store, scope_a, admin, retention_days=90, sessions_dir=sessions_dir) == {"skipped": True, "pruned": 0, "closed": 0, "vacuumed": False}
    assert admin.calls == ["acquire", ("release", "lock-1")]


def test_run_auto_prune_without_pruned_rows_never_considers_vacuum(backend, scope_a, events, caplog):
    store = backend.open(scope_a)
    _seed(store, _payload("s-young", NOW, NOW, "done"))
    admin = _Admin()
    caplog.set_level(logging.DEBUG, logger="hermes_state")
    assert _prune(store, scope_a, admin, retention_days=90) == {"skipped": False, "pruned": 0, "closed": 0, "vacuumed": False}
    assert _messages(caplog, debug=True) == []
    assert events == [["collect", []]]
    assert admin.calls == ["acquire", ("release", "lock-1")]


@pytest.mark.parametrize(("admin", "expected", "debug_line"), [
    (_Admin(ratio=0.1), {"freelist_ratio": 0.1},
     "state.db auto-maintenance: skipping VACUUM, only 10.0% of pages reclaimable (threshold 25%)"),
    (_Admin(blockers=[(4242, "state.db-wal")]), {"freelist_ratio": 0.0, "vacuum_skipped_holders": 1},
     "state.db auto-maintenance: skipping VACUUM, 1 other process(es) hold the store or a WAL sidecar (4242:state.db-wal)"),
], ids=["sparse_freelist", "holder"])
def test_run_auto_prune_skips_vacuum(backend, scope_a, events, caplog, admin, expected, debug_line):
    store = backend.open(scope_a)
    _seed(store, _payload("s-o", NOW - 100 * DAY, NOW - 100 * DAY, "done"))
    caplog.set_level(logging.DEBUG, logger="hermes_state")
    ratio = {"min_vacuum_freelist_ratio": -1.0} if admin.blockers else {}
    assert _prune(store, scope_a, admin, retention_days=90, **ratio) == {"skipped": False, "pruned": 1, "closed": 0, "vacuumed": False, **expected}
    assert _messages(caplog, debug=True) == [("DEBUG", debug_line),
                                             ("INFO", "state.db auto-maintenance: closed 0 stale open session(s), pruned 1 session(s) inactive for 90 days")]
    assert "vacuum" not in admin.calls
    assert store.state_meta.get("last_vacuum") is None


def test_run_auto_prune_skips_while_the_lock_is_held(backend, scope_a, events):
    store = backend.open(scope_a)
    _seed(store, _payload("s-o", NOW - 100 * DAY, NOW - 100 * DAY, "done"))
    admin = _Admin(lock=None)
    assert _prune(store, scope_a, admin, retention_days=90) == {"skipped": True, "pruned": 0, "closed": 0, "vacuumed": False}
    # CONTRACT: nothing ran and nothing is released.
    assert admin.calls == ["acquire"]
    assert store.sessions.get(SessionId("s-o")) is not None
    assert events == []


def test_run_auto_prune_reports_a_failing_prune_and_releases_the_lock(backend, scope_a, caplog, monkeypatch):
    store = backend.open(scope_a)
    admin = _Admin()

    def boom(*args, **kwargs):
        raise RuntimeError("prune boom")

    monkeypatch.setattr(store.maintenance, "prune_sessions", boom)
    caplog.set_level(logging.DEBUG, logger="hermes_state")
    assert _prune(store, scope_a, admin, retention_days=90) == {"skipped": False, "pruned": 0, "closed": 0, "vacuumed": False, "error": "prune boom"}
    assert _messages(caplog) == [("WARNING", "state.db auto-maintenance failed: prune boom")]
    assert store.state_meta.get("last_auto_prune") is None
    assert admin.calls == ["acquire", ("release", "lock-1")]


def test_run_auto_prune_survives_a_failing_vacuum(backend, scope_a, events, caplog):
    store = backend.open(scope_a)
    _seed(store, _payload("s-o", NOW - 100 * DAY, NOW - 100 * DAY, "done"))
    caplog.set_level(logging.DEBUG, logger="hermes_state")
    result = _prune(store, scope_a, _Admin(vacuum_error=RuntimeError("vac boom")), retention_days=90, min_vacuum_freelist_ratio=-1.0)
    assert result == {"skipped": False, "pruned": 1, "closed": 0, "vacuumed": False, "freelist_ratio": 0.0}
    assert _messages(caplog) == [("WARNING", "state.db VACUUM failed: vac boom"),
                                 ("INFO", "state.db auto-maintenance: closed 0 stale open session(s), pruned 1 session(s) inactive for 90 days")]
    assert {key: store.state_meta.get(key) is not None for key in ("last_auto_prune", "last_vacuum")} == {"last_auto_prune": True, "last_vacuum": False}


def test_run_auto_prune_without_a_physical_admin_skips_lock_and_vacuum(backend, scope_a, events):
    store = backend.open(scope_a)
    _seed(store, _payload("s-o", NOW - 100 * DAY, NOW - 100 * DAY, "done"))
    # A backend without a physical admin prunes and closes; the lock and VACUUM steps do not exist there.
    assert _prune(store, scope_a, None, retention_days=90, min_vacuum_freelist_ratio=-1.0) == {
        "skipped": False, "pruned": 1, "closed": 0, "vacuumed": False}
    assert store.state_meta.get("last_vacuum") is None
    assert store.state_meta.get("last_auto_prune") is not None


# ── collect_retired_media and collect_retired_spillover ──────────────────────


def test_collect_retired_media_releases_unheld_media_and_drops_collected_candidates(backend, scope_a, tmp_path):
    from gateway.session_ingress_media import native_inputs_root
    store = backend.open(scope_a)
    # CONTRACT: no candidate, nothing to do.
    session_retention.collect_retired_media(store, tmp_path)
    root = native_inputs_root()
    refs = {}
    for name, digest in (("unheld.png", "a" * 64), ("mentioned.png", "b" * 64)):
        path = root / digest / name
        path.parent.mkdir(parents=True)
        path.write_bytes(b"x")
        refs[name] = {"path": str(path), "sha256": digest, "size": 1}
    store.sessions.create(NewSession(session_id=SessionId("s-kept"), source="cli"))
    store.messages.append(SessionId("s-kept"), NewMessage(role="user", content=f"see {'b' * 64}/mentioned.png"))
    store.admissions.record_retired_media(list(refs.values()))
    session_retention.collect_retired_media(store, tmp_path)
    # CONTRACT: the candidate nothing holds goes with its digest folder and is dropped; the one a retained transcript row mentions stays, and so does its candidate.
    assert not Path(refs["unheld.png"]["path"]).parent.exists()
    assert Path(refs["mentioned.png"]["path"]).read_bytes() == b"x"
    assert [value["path"] for _key, value in store.admissions.retired_candidates("media")] == [refs["mentioned.png"]["path"]]


def test_collect_retired_spillover_unlinks_unreferenced_archives(backend, scope_a, tmp_path, monkeypatch):
    store = backend.open(scope_a)
    spill = tmp_path / "cache" / "spillover"
    spill.mkdir(parents=True)
    for name in ("gone.txt", "kept.txt"):
        (spill / name).write_text("x")
    # The deleted session points at both archives; a surviving sibling still points at one of them.
    for sid in ("s-spill", "s-sibling"):
        store.sessions.create(NewSession(session_id=SessionId(sid), source="cli"))
    store.messages.append(SessionId("s-spill"), NewMessage(role="tool", content="output in ~/.hermes/cache/spillover/gone.txt", tool_call_id="c1", tool_name="terminal"))
    store.messages.append(SessionId("s-spill"), NewMessage(role="assistant", content="short", api_content="see ~/.hermes/cache/spillover/kept.txt"))
    store.messages.append(SessionId("s-sibling"), NewMessage(role="user", content="~/.hermes/cache/spillover/kept.txt"))
    assert store.sessions.delete(SessionId("s-spill"), media_root=MEDIA_ROOT) == (SessionId("s-spill"),)
    scanned: list = []

    def state():
        return {"files": sorted(p.name for p in spill.iterdir()),
                "candidates": [key[len(SPILLOVER_PREFIX):] for key, _value in store.admissions.retired_candidates("spillover")]}

    # CONTRACT: the delete recorded both archives its rows named, the shared one included.
    assert state() == {"files": ["gone.txt", "kept.txt"], "candidates": ["gone.txt", "kept.txt"]}
    monkeypatch.setattr("tools.tool_result_storage.referenced_spillover_names", lambda d: scanned.append(d))
    session_retention.collect_retired_spillover(store, tmp_path)
    # CONTRACT: an unknown reference set keeps files and candidates.
    assert state() == {"files": ["gone.txt", "kept.txt"], "candidates": ["gone.txt", "kept.txt"]}
    monkeypatch.setattr("tools.tool_result_storage.referenced_spillover_names", lambda d: scanned.append(d) or {"kept.txt"})
    session_retention.collect_retired_spillover(store, tmp_path)
    # CONTRACT: the referenced archive stays on disk, the other goes, and both candidates are dropped.
    assert state() == {"files": ["kept.txt"], "candidates": []}
    assert scanned == [spill, spill]
