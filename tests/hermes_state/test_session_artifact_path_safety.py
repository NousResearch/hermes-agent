"""Session IDs from import/external boundaries never become filesystem paths."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.agent_runtime_helpers import dump_api_request_debug
from hermes_state import SessionDB, divert_session_transcript_jsonl
from hermes_state_ids import session_id_storage_name


def _path_shaped_id(tmp_path, shape: str) -> str:
    if shape == "absolute":
        return str(tmp_path / "outside" / "absolute-session")
    return "../../outside/traversal-session"


def _dump_request(sessions_dir, session_id: str):
    agent = SimpleNamespace(
        api_mode="chat_completions",
        base_url="https://provider.invalid/v1",
        client=SimpleNamespace(),
        session_id=session_id,
        logs_dir=sessions_dir,
        log_prefix="",
        verbose_logging=False,
        _mask_api_key_for_logs=lambda _key: "masked",
        _vprint=lambda _message: None,
    )
    return dump_api_request_debug(agent, {"model": "test", "messages": []}, reason="test")


@pytest.mark.parametrize("shape", ("absolute", "traversal"))
def test_import_rejects_path_shaped_session_ids_atomically(tmp_path, shape):
    session_id = _path_shaped_id(tmp_path, shape)
    safe_id = f"safe-{shape}"
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        result = db.import_sessions([
            {"id": session_id, "source": "import", "messages": [{"role": "user", "content": "hello"}]},
            {"id": safe_id, "source": "import", "messages": []},
        ])

        assert result["ok"] is False
        assert result["imported"] == 0
        assert result["errors"][0]["session_id"] == session_id
        assert "session id" in result["errors"][0]["error"]
        assert db.get_session(session_id) is None
        assert db.get_session(safe_id) is None
    finally:
        db.close()


@pytest.mark.parametrize("shape", ("absolute", "traversal"))
def test_external_session_artifacts_write_and_delete_only_inside_sessions(tmp_path, monkeypatch, shape):
    home = tmp_path / "home"
    sessions_dir = home / "sessions"
    sessions_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    session_id = _path_shaped_id(tmp_path, shape)

    # These are the paths the historical raw interpolation reached. Keep one sentinel for the
    # delete boundary and separately observe whether diversion wrote the other.
    escaped_json = (sessions_dir / f"{session_id}.json").resolve()
    escaped_jsonl = (sessions_dir / f"{session_id}.jsonl").resolve()
    escaped_json.parent.mkdir(parents=True, exist_ok=True)
    escaped_json.write_text("outside-sentinel", encoding="utf-8")
    escaped_jsonl.unlink(missing_ok=True)

    db = SessionDB(db_path=home / "state.db")
    try:
        db.create_session(session_id, source="external")
        diverted = divert_session_transcript_jsonl(
            session_id, [{"role": "user", "content": "diverted"}],
        )
        escaped_write_happened = escaped_jsonl.exists()
        request_dump = _dump_request(sessions_dir, session_id)
        assert diverted is not None and request_dump is not None

        storage_name = session_id_storage_name(session_id)
        canonical_artifacts = [
            sessions_dir / f"{storage_name}.json",
            sessions_dir / f"{storage_name}.jsonl",
            sessions_dir / f"session_{storage_name}.json",
            request_dump,
        ]
        for artifact in canonical_artifacts[:-1]:
            if not artifact.exists():
                artifact.write_text("contained", encoding="utf-8")

        artifact_paths = [diverted, *canonical_artifacts]
        root = sessions_dir.resolve()
        contained_before_delete = all(
            artifact.resolve().is_relative_to(root) for artifact in artifact_paths
        )
        assert db.delete_session(session_id, sessions_dir=sessions_dir)
    finally:
        db.close()

    assert contained_before_delete
    assert all(artifact.resolve().is_relative_to(root) for artifact in artifact_paths)
    assert request_dump.name.startswith(f"request_dump_{storage_name}_")
    assert not escaped_write_happened
    assert escaped_json.is_file()
    assert escaped_json.read_text(encoding="utf-8-sig") == "outside-sentinel"
    assert all(not artifact.exists() for artifact in canonical_artifacts)
