"""``hermes sessions repair-profiles``: the backward-looking half of the per-profile store model.

The forward-only fixes (#88734 store routing, the #88381 inheritance fence, #76423 topic labels,
#75198 voice keys) put NEW state in the right place; nothing settled what earlier releases left
crossed. One fixture carries all six kinds of crossing; the contracts are that a dry run mutates
nothing, that apply settles every repairable finding, and that a second apply finds nothing.
Part of #88715.
"""
from __future__ import annotations

import json
import sqlite3
from argparse import Namespace
from pathlib import Path

import pytest

from hermes_state import SessionDB

ROOT_KEY = "agent:main:telegram:dm:100"
ACME_KEY = "agent:acme:telegram:dm:200"
BETA_KEY = "agent:beta:telegram:dm:300"
GHOST_KEY = "agent:ghost:telegram:dm:400"


@pytest.fixture
def homes(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    from hermes_cli.profiles import create_profile
    for name in ("acme", "beta"):
        create_profile(name, no_alias=True, no_skills=True)
    return {"default": root, "acme": root / "profiles" / "acme", "beta": root / "profiles" / "beta"}


def _session(db: SessionDB, sid: str, key: str, *, profile: str, parent=None, messages=2, chat_id=None,
             source="telegram"):
    db.create_session(sid, source, session_key=key, chat_id=chat_id or key.rsplit(":", 1)[-1],
                      chat_type="dm", profile_name=profile, parent_session_id=parent, system_prompt="sys")
    for i in range(messages):
        db.append_message(sid, "user" if i % 2 == 0 else "assistant", f"{sid} m{i}")


def _seed_crossings(homes):
    """Every defect the command knows, spread over the three stores."""
    root = SessionDB(homes["default"] / "state.db")
    acme = SessionDB(homes["acme"] / "state.db")
    beta = SessionDB(homes["beta"] / "state.db")
    scope = str((homes["default"] / "sessions").resolve())

    # healthy controls — must survive untouched
    _session(root, "ok-root", ROOT_KEY, profile="default")
    _session(acme, "ok-acme", ACME_KEY, profile="acme")
    root.save_gateway_routing_entry(ACME_KEY, json.dumps({"session_key": ACME_KEY, "session_id": "ok-acme"}), scope=scope)

    # 1. label ≠ key namespace (row in the right store)
    _session(acme, "mislabelled", "agent:acme:telegram:dm:201", profile="default")
    # 2. wrong store: acme's key in the root store, with a compressed parent it points at (moves whole)
    _session(root, "stray-parent", "agent:acme:telegram:dm:202", profile="acme", messages=3)
    _session(root, "stray-child", "agent:acme:telegram:dm:202", profile="acme", parent="stray-parent")
    #    …and a legacy agent:main row inside acme's store (reported, not repaired by default)
    _session(acme, "legacy-main", "agent:main:telegram:dm:203", profile="default")
    #    …and a row keyed to a profile that does not exist
    _session(root, "ghost", GHOST_KEY, profile="ghost")
    # 3. parent crossing namespaces inside beta's store: beta's own child forked from an acme row
    #    (which is itself a wrong-store row and moves out — the child must not follow a pointer
    #    into another profile's store)
    _session(beta, "beta-parent", BETA_KEY, profile="beta")
    _session(beta, "acme-in-beta", "agent:acme:telegram:dm:204", profile="acme")
    _session(beta, "beta-child", "agent:beta:telegram:dm:301", profile="beta", parent="acme-in-beta")
    # 4. routing rows: beta's index row copied into beta's store (#66887) + a ghost row in the root
    beta.save_gateway_routing_entry(BETA_KEY, json.dumps({"session_key": BETA_KEY, "session_id": "beta-parent"}), scope=scope)
    root.save_gateway_routing_entry(GHOST_KEY, json.dumps({"session_key": GHOST_KEY, "session_id": "ghost"}), scope=scope)
    # 5. profile-less topic binding + voice-mode entry for a chat only acme's bot holds
    acme.bind_telegram_topic(chat_id="200", thread_id="7", user_id="200", session_key=ACME_KEY,
                             session_id="ok-acme", profile_name="default")
    (homes["default"] / "gateway_voice_mode.json").write_text(json.dumps({
        "telegram:200": "all",        # acme's chat, unprefixed
        "telegram:100": "voice_only",  # the default bot's chat — legitimately unprefixed
    }))
    # 6. sessions.json mirror entry for the ghost namespace
    (homes["default"] / "sessions").mkdir(exist_ok=True)
    (homes["default"] / "sessions" / "sessions.json").write_text(json.dumps({
        "_README": "x", GHOST_KEY: {"session_key": GHOST_KEY, "session_id": "ghost"},
        ROOT_KEY: {"session_key": ROOT_KEY, "session_id": "ok-root"},
    }))
    for db in (root, acme, beta):
        db.close()


def _dump(path: Path) -> dict:
    """Every user table as sorted row tuples — the whole-file mutation oracle for the dry run."""
    conn = sqlite3.connect(path)
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "AND name NOT LIKE '%_fts%' ORDER BY name")]
        return {t: sorted(map(repr, conn.execute(f"SELECT * FROM {t}").fetchall())) for t in tables}
    finally:
        conn.close()


def _run(**kw) -> int:
    from hermes_cli.sessions_cmd import cmd_sessions
    args = Namespace(sessions_action="repair-profiles", apply=False, json=False, yes=True, legacy_main="report")
    for k, v in kw.items():
        setattr(args, k, v)
    return cmd_sessions(args) or 0


def _report() -> dict:
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert _run(json=True) == 0
    return json.loads(buf.getvalue())


def test_dry_run_names_every_crossing_and_changes_nothing(homes, capsys):
    _seed_crossings(homes)
    before = {name: _dump(home / "state.db") for name, home in homes.items()}
    files_before = {p.name: p.read_text() for p in (homes["default"] / "gateway_voice_mode.json",
                                                    homes["default"] / "sessions" / "sessions.json")}

    report = _report()

    kinds = {(f["kind"], f["subject"]) for f in report["findings"]}
    assert {
        ("mislabelled", "mislabelled"),
        ("wrong_store", "stray-parent"), ("wrong_store", "stray-child"),
        ("legacy_main", "legacy-main"),
        ("unclaimed_namespace", "ghost"),
        ("wrong_store", "acme-in-beta"), ("crossed_parent", "beta-child"),
        ("routing_unclaimed", f"{GHOST_KEY} (scope {str((homes['default'] / 'sessions').resolve())!r})"),
        ("topic_profile_less", "chat 200 thread 7"),
        ("voice_profile_less", "telegram:200"),
        ("sessions_json_unclaimed", GHOST_KEY),
    } <= kinds
    # beta's own routing row: a standalone gateway's index lives in its own store; nothing recorded a
    # multiplexing verdict here, so it is NOT reported as stray.
    assert not any(f["kind"] == "routing_stray" for f in report["findings"])
    # the healthy controls and the default bot's own voice key are not findings
    assert not any(f["subject"] in {"ok-root", "ok-acme", "telegram:100", ROOT_KEY} for f in report["findings"])
    # the two report-only kinds say why
    by_kind = {f["kind"]: f for f in report["findings"]}
    assert by_kind["legacy_main"]["action"] is None and "--legacy-main" in by_kind["legacy_main"]["reason"]
    assert by_kind["unclaimed_namespace"]["action"] is None and "ghost" in by_kind["unclaimed_namespace"]["reason"]

    assert {name: _dump(home / "state.db") for name, home in homes.items()} == before
    assert {p.name: p.read_text() for p in (homes["default"] / "gateway_voice_mode.json",
                                            homes["default"] / "sessions" / "sessions.json")} == files_before


def test_apply_settles_every_repairable_crossing_and_is_idempotent(homes, monkeypatch, capsys):
    _seed_crossings(homes)
    snapshots = []
    monkeypatch.setattr("hermes_cli.sessions_cmd_repair_profiles.default_snapshot",
                        lambda store: snapshots.append(store.profile) or f"snap-{store.profile}")

    assert _run(apply=True) == 0
    assert set(snapshots) == {"default", "acme", "beta"}

    root = SessionDB(homes["default"] / "state.db")
    acme = SessionDB(homes["acme"] / "state.db")
    beta = SessionDB(homes["beta"] / "state.db")
    try:
        # 1. relabelled from the key
        assert acme.get_session("mislabelled")["profile_name"] == "acme"
        # 2. moved whole: parent link and every message intact in the target, gone from the source
        assert root.get_session("stray-parent") is None and root.get_session("stray-child") is None
        moved_child = acme.get_session("stray-child")
        assert moved_child["profile_name"] == "acme" and moved_child["parent_session_id"] == "stray-parent"
        assert len(acme.get_messages("stray-parent")) == 3 and len(acme.get_messages("stray-child")) == 2
        assert acme.get_session("stray-parent")["system_prompt"] == "sys"
        #    legacy agent:main row and the ghost row are left exactly as they were
        assert acme.get_session("legacy-main")["session_key"] == "agent:main:telegram:dm:203"
        assert root.get_session("ghost")["session_key"] == GHOST_KEY
        # 3. severed, own identity kept; the acme row it pointed at moved to acme's store
        child = beta.get_session("beta-child")
        assert child["parent_session_id"] is None and child["profile_name"] == "beta"
        assert beta.get_session("acme-in-beta") is None and acme.get_session("acme-in-beta") is not None
        # 4. ghost routing row dropped, acme's healthy row kept, beta's own row kept (standalone)
        scope = str((homes["default"] / "sessions").resolve())
        assert set(root.load_gateway_routing_entries(scope=scope)) == {ACME_KEY}
        assert set(beta.load_gateway_routing_entries(scope=scope)) == {BETA_KEY}
        # 5. topic binding carries acme; voice key prefixed only for acme's chat
        assert acme.get_telegram_topic_binding(chat_id="200", thread_id="7", profile_name="acme") is not None
        assert acme.get_telegram_topic_binding(chat_id="200", thread_id="7", profile_name="default") is None
        voice = json.loads((homes["default"] / "gateway_voice_mode.json").read_text())
        assert voice == {"acme:telegram:200": "all", "telegram:100": "voice_only"}
        # 6. ghost mirror entry dropped, the default's kept
        mirror = json.loads((homes["default"] / "sessions" / "sessions.json").read_text())
        assert set(mirror) == {"_README", ROOT_KEY}
        # controls untouched
        assert root.get_session("ok-root")["profile_name"] == "default"
        assert acme.get_session("ok-acme")["profile_name"] == "acme"
    finally:
        for db in (root, acme, beta):
            db.close()

    # idempotent: only the two report-only findings remain, and nothing is repairable
    report = _report()
    assert {f["kind"] for f in report["findings"]} == {"legacy_main", "unclaimed_namespace"}
    assert report["repairable"] == 0


def test_apply_refuses_while_a_gateway_owns_a_store(homes, monkeypatch, capsys):
    _seed_crossings(homes)
    monkeypatch.setattr("hermes_cli.sessions_cmd_repair_profiles.live_gateway_homes",
                        lambda stores: [("default", 4242)])
    before = _dump(homes["default"] / "state.db")

    assert _run(apply=True) == 1

    assert "pid 4242" in capsys.readouterr().err
    assert _dump(homes["default"] / "state.db") == before


def test_legacy_main_rekey_adopts_a_standalone_gateways_history(homes, monkeypatch):
    """The #113884 incident: multiplexing switched on, every existing key of the named profile's own
    gateway stopped resolving. ``--legacy-main rekey`` gives them the namespace the multiplexer reads."""
    _seed_crossings(homes)
    monkeypatch.setattr("hermes_cli.sessions_cmd_repair_profiles.default_snapshot", lambda store: "snap")

    assert _run(apply=True, legacy_main="rekey") == 0

    acme = SessionDB(homes["acme"] / "state.db")
    try:
        row = acme.get_session("legacy-main")
        assert row["session_key"] == "agent:acme:telegram:dm:203" and row["profile_name"] == "acme"
    finally:
        acme.close()
    assert not any(f["kind"] == "legacy_main" for f in _report()["findings"])


def test_move_into_a_store_that_already_holds_the_title(homes, monkeypatch):
    """Titles are unique per store only. A stranded row whose title an unrelated session of the
    target profile already holds used to raise on the unique-title index, fail every row of the
    batch, and leave the rows copied before it in both stores, on every run. The clash is the
    FIRST row of the batch and its title is at the length cap, so the suffix must still fit."""
    title = "G" * SessionDB.MAX_TITLE_LENGTH
    monkeypatch.setattr("hermes_cli.sessions_cmd_repair_profiles.default_snapshot", lambda store: "snap")
    root = SessionDB(homes["default"] / "state.db")
    acme = SessionDB(homes["acme"] / "state.db")
    try:
        _session(root, "s-a", "agent:acme:telegram:dm:210", profile="acme")
        _session(root, "s-b", "agent:acme:telegram:dm:211", profile="acme", messages=3)
        _session(acme, "w-1", ACME_KEY, profile="acme")
        clash, other = "s-a", "s-b"
        root.set_session_title(clash, title)
        root.set_session_title(other, "Weekend plans")
        acme.set_session_title("w-1", title)
    finally:
        root.close()
        acme.close()

    assert _run(apply=True) == 0

    root = SessionDB(homes["default"] / "state.db")
    acme = SessionDB(homes["acme"] / "state.db")
    try:
        assert root.get_session("s-a") is None and root.get_session("s-b") is None
        assert len(acme.get_messages("s-a")) == 2 and len(acme.get_messages("s-b")) == 3
        assert acme.get_session(other)["title"] == "Weekend plans"
        # the resident row keeps the name, so resolving it by title still finds it
        assert acme.get_session("w-1")["title"] == title
        assert acme.resolve_session_by_title(title) == "w-1"
        moved = acme.get_session(clash)["title"]
        assert moved not in (None, title) and len(moved) <= SessionDB.MAX_TITLE_LENGTH
    finally:
        root.close()
        acme.close()
    assert not any(f["kind"] == "wrong_store" for f in _report()["findings"])


def test_a_failing_row_moves_alone_and_its_lineage_waits_with_it(homes, monkeypatch, capsys):
    """One row that cannot move is that row's failure: an unrelated row of the same batch still
    moves, the batch is not re-run per finding, and the failed row's lineage stays linked so the
    next run moves it whole. The failure is at import (the shape a unique-index clash takes)."""
    stage = "import"
    monkeypatch.setattr("hermes_cli.sessions_cmd_repair_profiles.default_snapshot", lambda store: "snap")
    root = SessionDB(homes["default"] / "state.db")
    try:
        _session(root, "gp", "agent:acme:telegram:dm:220", profile="acme")
        _session(root, "par", "agent:acme:telegram:dm:220", profile="acme", parent="gp")
        _session(root, "kid", "agent:acme:telegram:dm:220", profile="acme", parent="par")
        _session(root, "lone", "agent:acme:telegram:dm:221", profile="acme")
    finally:
        root.close()

    method = "import_moved_session" if stage == "import" else "delete_moved_session"
    real = getattr(SessionDB, method)
    calls = []

    def flaky(self, arg, **kw):
        sid = arg["session"]["id"] if stage == "import" else arg
        calls.append(sid)
        if sid == "par":
            raise sqlite3.OperationalError("disk I/O error")
        return real(self, arg, **kw)

    monkeypatch.setattr(SessionDB, method, flaky)
    assert _run(apply=True) == 1
    out = capsys.readouterr().out
    assert "wrong_store par: OperationalError: disk I/O error" in out
    assert sorted(calls) == sorted(set(calls)), "the batch was re-run for a later finding"

    root = SessionDB(homes["default"] / "state.db")
    acme = SessionDB(homes["acme"] / "state.db")
    try:
        assert root.get_session("lone") is None and acme.get_session("lone") is not None
        if stage == "import":
            assert "wrong_store kid:" in out
            # the child never lands parentless; the grandparent stays for the parent that points at it
            assert acme.get_session("kid") is None and acme.get_session("par") is None
            assert root.get_session("kid")["parent_session_id"] == "par"
            assert root.get_session("par")["parent_session_id"] == "gp"
    finally:
        root.close()
        acme.close()

    monkeypatch.setattr(SessionDB, method, real)
    assert _run(apply=True) == 0
    root = SessionDB(homes["default"] / "state.db")
    acme = SessionDB(homes["acme"] / "state.db")
    try:
        assert all(root.get_session(sid) is None for sid in ("gp", "par", "kid", "lone"))
        assert acme.get_session("par")["parent_session_id"] == "gp"
        assert acme.get_session("kid")["parent_session_id"] == "par"
    finally:
        root.close()
        acme.close()
    assert not any(f["kind"] == "wrong_store" for f in _report()["findings"])
