from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from hermes_state import SessionDB


def _db(tmp_path: Path) -> SessionDB:
    return SessionDB(db_path=tmp_path / "state.db")


def _payload(label: str, *, api_call_count: int = 1, retry_count: int = 0) -> dict:
    return {
        "schema_version": 1,
        "kind": "subagent_context_payload",
        "raw_unredacted_by_viewer": True,
        "child_session_id": "child-1",
        "parent_session_id": "parent-1",
        "subagent_id": "sa-0-test",
        "api_call_count": api_call_count,
        "retry_count": retry_count,
        "captured_at": 123.0 + api_call_count,
        "finalized_at": None,
        "status": "capturing",
        "role": "leaf",
        "profile": "coder",
        "model": "test/model",
        "provider": "openrouter",
        "api_mode": "chat_completions",
        "base_url": "https://example.invalid/v1",
        "toolsets": ["file", "terminal"],
        "canonical_messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": label},
        ],
        "provider_request": {"model": "test/model", "messages": [{"role": "user", "content": label}]},
        "provider_request_keys": ["model", "messages"],
        "serialization_warnings": [],
    }


def test_create_read_update_finalize_and_delete_before_child_session_exists(tmp_path: Path):
    from agent.subagent_context_artifacts import (
        create_subagent_context_artifact_pointer,
        delete_subagent_context_artifacts_for_sessions,
        finalize_subagent_context_artifact,
        get_subagent_context_artifact,
        update_subagent_context_artifact_capture,
    )

    db = _db(tmp_path)
    try:
        pointer = create_subagent_context_artifact_pointer(
            child_session_id="child-1",
            parent_session_id="parent-1",
            subagent_id="sa-0-test",
            role="leaf",
            profile="coder",
            model="test/model",
            provider="openrouter",
            api_mode="chat_completions",
            base_url="https://example.invalid/v1",
            toolsets=["file", "terminal"],
            session_db=db,
        )

        assert pointer["child_session_id"] == "child-1"
        assert pointer["parent_session_id"] == "parent-1"
        assert pointer["capture_sequence"] == 0
        assert pointer["status"] == "capturing"
        artifact_path = Path(pointer["latest_artifact_path"])
        assert artifact_path.name == "latest.json"
        assert artifact_path.exists()
        # No sessions row exists yet; the pointer must still be readable.
        assert db._conn.execute("SELECT COUNT(*) FROM sessions WHERE id = 'child-1'").fetchone()[0] == 0

        initial = get_subagent_context_artifact("child-1", session_db=db)
        assert initial["ok"] is True
        assert initial["pointer"]["child_session_id"] == "child-1"
        assert initial["artifact"]["status"] == "capturing"

        first = update_subagent_context_artifact_capture(
            "child-1",
            _payload("first", api_call_count=1, retry_count=0),
            session_db=db,
        )
        assert first["ok"] is True
        assert first["capture_sequence"] == 1
        assert first["artifact_size_bytes"] > 0
        assert first["artifact_sha256"]

        second = update_subagent_context_artifact_capture(
            "child-1",
            _payload("second", api_call_count=2, retry_count=1),
            session_db=db,
        )
        assert second["ok"] is True
        assert second["capture_sequence"] == 2

        latest = get_subagent_context_artifact("child-1", session_db=db)
        assert latest["ok"] is True
        assert latest["pointer"]["capture_sequence"] == 2
        assert latest["pointer"]["latest_api_call_count"] == 2
        assert latest["pointer"]["latest_retry_count"] == 1
        assert latest["artifact"]["capture_sequence"] == 2
        assert latest["artifact"]["canonical_messages"][1]["content"] == "second"
        assert "first" not in artifact_path.read_text(encoding="utf-8")

        finalized = finalize_subagent_context_artifact("child-1", session_db=db)
        assert finalized["ok"] is True
        assert finalized["status"] == "finalized"
        frozen = get_subagent_context_artifact("child-1", session_db=db)
        frozen_artifact = frozen["artifact"]
        assert frozen_artifact["status"] == "finalized"
        assert frozen_artifact["finalized_at"] is not None

        ignored = update_subagent_context_artifact_capture(
            "child-1",
            _payload("after-finalize", api_call_count=3),
            session_db=db,
        )
        assert ignored["ok"] is False
        assert ignored["ignored"] is True
        assert ignored["status"] == "finalized"
        assert get_subagent_context_artifact("child-1", session_db=db)["artifact"] == frozen_artifact

        deleted = delete_subagent_context_artifacts_for_sessions(["parent-1"], session_db=db)
        assert deleted["deleted"] == 1
        assert not artifact_path.exists()
        assert get_subagent_context_artifact("child-1", session_db=db)["ok"] is False
    finally:
        db.close()


def test_json_safe_copy_tags_non_serializable_values(tmp_path: Path):
    from agent.subagent_context_artifacts import (
        create_subagent_context_artifact_pointer,
        get_subagent_context_artifact,
        update_subagent_context_artifact_capture,
    )

    db = _db(tmp_path)
    try:
        create_subagent_context_artifact_pointer(
            child_session_id="child-json",
            parent_session_id="parent-json",
            subagent_id="sa-json",
            session_db=db,
        )
        payload = _payload("json-safe")
        payload["child_session_id"] = "child-json"
        payload["parent_session_id"] = "parent-json"
        payload["provider_request"]["non_serializable"] = object()
        payload["provider_request"][42] = "non-string key"

        result = update_subagent_context_artifact_capture("child-json", payload, session_db=db)
        assert result["ok"] is True

        artifact = get_subagent_context_artifact("child-json", session_db=db)["artifact"]
        assert "non_serializable" in artifact["provider_request"]
        assert artifact["provider_request"]["non_serializable"].startswith("[non_serializable:")
        assert artifact["provider_request"]["42"] == "non-string key"
        warnings = artifact["serialization_warnings"]
        assert any("non-serializable" in warning for warning in warnings)
        assert any("non-string dict key" in warning for warning in warnings)
    finally:
        db.close()


def test_session_delete_removes_delegate_context_artifact(tmp_path: Path):
    from agent.subagent_context_artifacts import (
        create_subagent_context_artifact_pointer,
        get_subagent_context_artifact,
        update_subagent_context_artifact_capture,
    )

    db = _db(tmp_path)
    try:
        db.create_session(session_id="parent-delete", source="cli", model="test")
        db.create_session(
            session_id="child-delete",
            source="delegate",
            model="test",
            model_config={"_delegate_from": "parent-delete"},
            parent_session_id="parent-delete",
        )
        pointer = create_subagent_context_artifact_pointer(
            child_session_id="child-delete",
            parent_session_id="parent-delete",
            subagent_id="sa-delete",
            session_db=db,
        )
        artifact_path = Path(pointer["latest_artifact_path"])
        payload = _payload("delete me")
        payload["child_session_id"] = "child-delete"
        payload["parent_session_id"] = "parent-delete"
        update_subagent_context_artifact_capture("child-delete", payload, session_db=db)

        assert artifact_path.exists()
        assert db.delete_session("parent-delete") is True
        assert not artifact_path.exists()
        assert get_subagent_context_artifact("child-delete", session_db=db)["ok"] is False
    finally:
        db.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX modes are not enforced on Windows")
def test_posix_artifact_modes_are_private(tmp_path: Path):
    from agent.subagent_context_artifacts import (
        create_subagent_context_artifact_pointer,
        update_subagent_context_artifact_capture,
    )

    db = _db(tmp_path)
    try:
        pointer = create_subagent_context_artifact_pointer(
            child_session_id="child/mode..check",
            parent_session_id="parent-mode",
            subagent_id="sa-mode",
            session_db=db,
        )
        artifact_path = Path(pointer["latest_artifact_path"])
        update_subagent_context_artifact_capture("child/mode..check", _payload("modes"), session_db=db)

        assert stat.S_IMODE(artifact_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(artifact_path.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(artifact_path.parent.parent.stat().st_mode) == 0o700
    finally:
        db.close()


def test_missing_and_corrupt_artifact_return_structured_errors(tmp_path: Path):
    from agent.subagent_context_artifacts import (
        create_subagent_context_artifact_pointer,
        get_subagent_context_artifact,
    )

    db = _db(tmp_path)
    try:
        pointer = create_subagent_context_artifact_pointer(
            child_session_id="child-corrupt",
            parent_session_id="parent-corrupt",
            subagent_id="sa-corrupt",
            session_db=db,
        )
        artifact_path = Path(pointer["latest_artifact_path"])
        artifact_path.write_text("{not-json", encoding="utf-8")

        corrupt = get_subagent_context_artifact("child-corrupt", session_db=db)
        assert corrupt["ok"] is False
        assert corrupt["error"]["code"] == "artifact_corrupt"
        assert corrupt["pointer"]["child_session_id"] == "child-corrupt"

        artifact_path.unlink()
        missing = get_subagent_context_artifact("child-corrupt", session_db=db)
        assert missing["ok"] is False
        assert missing["error"]["code"] == "artifact_missing"
        assert missing["pointer"]["child_session_id"] == "child-corrupt"
    finally:
        db.close()
