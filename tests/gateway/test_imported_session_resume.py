"""An imported foreign transcript resumes through the canonical path (#142091).

``hermes_cli.foreign_sessions.import_foreign_session`` writes a LOCAL row whose provenance lives in
``origin_json.imported_from``; the row carries no authority receipt, and the importer's label
(``claude-code`` / ``codex-cli``) is not a local surface. ``session.resume`` therefore fell through
to the messaging/as-API resolution and answered ``not_found`` for a transcript the legacy TUI opened
fine. These tests pin the adoption and the guards that keep it from becoming a bypass.
"""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.session_contract import SessionRef


def _claude_fixture(tmp_path):
    work = tmp_path / "proj"
    work.mkdir()
    proj = tmp_path / ".claude" / "projects" / "-home-user-proj"
    proj.mkdir(parents=True)
    lines = [
        {"type": "summary", "summary": "Fix the flaky test"},
        {"type": "user", "sessionId": "abc-123", "cwd": str(work),
         "message": {"role": "user", "content": "Please fix the flaky test."}},
        {"type": "assistant", "sessionId": "abc-123", "cwd": str(work),
         "message": {"role": "assistant", "content": [{"type": "text", "text": "Looking."}]}},
        {"type": "user", "sessionId": "abc-123", "cwd": str(work),
         "message": {"role": "user", "content": "Great, thanks!"}},
        {"type": "assistant", "sessionId": "abc-123", "cwd": str(work),
         "message": {"role": "assistant", "content": [{"type": "text", "text": "Anytime."}]}},
    ]
    path = proj / "abc-123.jsonl"
    path.write_text("\n".join(json.dumps(entry) for entry in lines) + "\n", encoding="utf-8")
    return path


def _imported_row(**over):
    # The shape both importers write: provenance in origin_json, no route columns at all.
    row = {"id": "imported-1", "source": "claude-code", "chat_id": None,
           "session_key": None, "profile_name": None, "user_id": None,
           "origin_json": json.dumps({"imported_from": {
               "tool": "claude-code", "path": "/tmp/abc-123.jsonl", "foreign_session_id": "abc-123"}})}
    row.update(over)
    return row


def _attach_snapshot(owner_ref):
    return SimpleNamespace(
        subscription_id="sub", history=(), pending=(), prompts=(), replay_epoch="e", last_sequence=0,
        handle=SimpleNamespace(ref=owner_ref, execution_state="idle", authority_epoch=1, revision=1,
                               execution_generation=0),
    )


@pytest.mark.asyncio
async def test_session_resume_remaps_an_imported_foreign_row(monkeypatch):
    """``session.resume`` hands an imported row to the local adoption path, not the messaging one."""
    from gateway.session_controls import AuthorityConnection

    resolved = []
    monkeypatch.setattr("gateway.session_local_migration.resolve_local_target",
                        lambda authority, actor, sid: resolved.append(sid) or SessionRef("fixture", sid))
    authority = SimpleNamespace(
        profile_id="fixture", instance_id="one", events={},
        db=SimpleNamespace(get_session=lambda sid: _imported_row() if sid == "imported-1" else None,
                           get_compression_tip=lambda sid: None),
        sessions={"imported-1": SimpleNamespace(source=SimpleNamespace(platform=None),
                                                route="agent:main:cli:dm:imported-1")},
        authorize=lambda *args, **kwargs: None,
        attach=AsyncMock(return_value=_attach_snapshot(SessionRef("fixture", "imported-1"))),
    )
    connection = AuthorityConnection(
        authority, object(),
        {"user_id": "owner", "capabilities": ["session:create", "session:read", "session:control"]},
    )
    result = await connection.resume(SessionRef("fixture", "imported-1"), {})
    assert resolved == ["imported-1"]
    assert result["session_id"] == "imported-1"


@pytest.mark.asyncio
async def test_an_imported_row_is_adopted_and_reopened_by_the_canonical_owner(tmp_path, monkeypatch):
    """The imported transcript becomes a canonical local session whose history reads back intact."""
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Principal
    from gateway.session_local_migration import resolve_local_target
    from hermes_cli.foreign_sessions import import_foreign_session
    from hermes_state_local import local_receipt

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = {"platform_toolsets": {"cli": []}, "model": {"model": "demo-model", "provider": "custom"}}
    monkeypatch.setattr(run, "_load_gateway_config", lambda: config)
    monkeypatch.setattr(run, "_resolve_gateway_model", lambda _config: "demo-model")
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    db = store._db
    sid = import_foreign_session("claude", _claude_fixture(tmp_path), db=db)
    assert db.get_session(sid)["source"] == "claude-code"

    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id="first")
    native = Principal("native", str(tmp_path), frozenset(
        {"session:create", "session:read", "session:submit", "session:control"}), "native-socket")
    authority._native_legacy_transports = {native.transport_id: native}

    ref = resolve_local_target(authority, native, sid)
    assert ref.session_id == sid
    snapshot = await authority.attach(native, ref)
    assert [message["content"] for message in snapshot.history] == [
        "Please fix the flaky test.", "Looking.", "Great, thanks!", "Anytime."]
    receipt = local_receipt(db, sid)
    # Adopted as the surface that imported it, while the row keeps the importer's label.
    assert receipt["policy"]["source"] == "cli"
    assert db.get_session(sid)["source"] == "claude-code"

    # A second open finds the committed receipt instead of adopting (and rewriting) the row again.
    authority.sessions.clear()
    again = resolve_local_target(authority, native, sid)
    assert again.session_id == sid
    assert local_receipt(db, sid) == receipt


def test_validate_legacy_row_still_refuses_routed_and_foreign_import_labels(tmp_path):
    """Only the importer's own unrouted, unowned row is adoptable under a foreign-tool label."""
    from hermes_state import SessionDB
    from hermes_state_local_migration import validate_legacy_row
    from hermes_state_runtime import RuntimeStoreError

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        # Provenance, but a canonical writer already routed the row.
        with pytest.raises(RuntimeStoreError, match="not_found"):
            validate_legacy_row(db, _imported_row(session_key="agent:main:cli:dm:imported-1"))
        # Provenance, but a chat_id names a route that is not this row's own id.
        with pytest.raises(RuntimeStoreError, match="not_found"):
            validate_legacy_row(db, _imported_row(chat_id="local-other-route"))
        # Provenance, but a login owns the row.
        with pytest.raises(RuntimeStoreError, match="not_found"):
            validate_legacy_row(db, _imported_row(user_id="nous:someone-else"))
        # A foreign-tool label with no importer provenance at all.
        with pytest.raises(RuntimeStoreError, match="not_found"):
            validate_legacy_row(db, _imported_row(origin_json=None))
        # Provenance, but the row belongs to another profile.
        with pytest.raises(RuntimeStoreError, match="not_found"):
            validate_legacy_row(db, _imported_row(profile_name="someone-else"))
        # A local surface label that carries the same stamp keeps today's veto.
        with pytest.raises(RuntimeStoreError, match="not_found"):
            validate_legacy_row(db, _imported_row(source="cli", session_key="agent:main:cli:dm:x"))
        # The row the importer itself writes is adoptable.
        validate_legacy_row(db, _imported_row())
    finally:
        db.close()
