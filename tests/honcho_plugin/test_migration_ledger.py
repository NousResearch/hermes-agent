"""Behavior contracts for one-time Honcho memory-file migration."""

from __future__ import annotations

import json
import threading
from unittest.mock import MagicMock, patch

from plugins.memory.honcho.client import HonchoClientConfig
from plugins.memory.honcho.session import (
    HonchoSession,
    HonchoSessionManager,
    _MIGRATION_THREAD_LOCK,
)


def _manager(
    tmp_path,
    monkeypatch,
    *,
    session_key="cli:test",
    base_url="http://127.0.0.1:8000/v3",
    workspace_id="workspace",
    user_peer_id="user",
    ai_peer_id="assistant",
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = HonchoClientConfig(
        api_key="key",
        base_url=base_url,
        workspace_id=workspace_id,
        write_frequency="turn",
    )
    manager = HonchoSessionManager(honcho=MagicMock(), config=config)
    session = HonchoSession(
        key=session_key,
        user_peer_id=user_peer_id,
        assistant_peer_id=ai_peer_id,
        honcho_session_id=session_key.replace(":", "-"),
    )
    remote_session = MagicMock()
    manager._cache[session.key] = session
    manager._sessions_cache[session.honcho_session_id] = remote_session
    manager._peers_cache[user_peer_id] = MagicMock(name=f"peer-{user_peer_id}")
    manager._peers_cache[ai_peer_id] = MagicMock(name=f"peer-{ai_peer_id}")
    return manager, remote_session


def _write_memory_files(memory_dir, *filenames):
    memory_dir.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        (memory_dir / filename).write_text(f"contents of {filename}", encoding="utf-8")


def _uploaded_names(remote_session):
    return [call.kwargs["file"][0] for call in remote_session.upload_file.call_args_list]


def test_same_destination_uploads_each_file_once(tmp_path, monkeypatch):
    memory_dir = tmp_path / "memories"
    _write_memory_files(memory_dir, "MEMORY.md", "USER.md", "SOUL.md")
    first, first_remote = _manager(tmp_path, monkeypatch, session_key="cli:first")
    second, second_remote = _manager(tmp_path, monkeypatch, session_key="cli:second")

    assert first.migrate_memory_files("cli:first", str(memory_dir)) is True
    assert second.migrate_memory_files("cli:second", str(memory_dir)) is False

    assert _uploaded_names(first_remote) == [
        "consolidated_memory.md",
        "user_profile.md",
        "agent_soul.md",
    ]
    second_remote.upload_file.assert_not_called()


def test_partial_failure_retries_only_failed_file(tmp_path, monkeypatch):
    memory_dir = tmp_path / "memories"
    _write_memory_files(memory_dir, "MEMORY.md", "USER.md", "SOUL.md")
    first, first_remote = _manager(tmp_path, monkeypatch, session_key="cli:first")

    def fail_user_file(*, file, **kwargs):
        if file[0] == "user_profile.md":
            raise RuntimeError("upload failed")

    first_remote.upload_file.side_effect = fail_user_file
    assert first.migrate_memory_files("cli:first", str(memory_dir)) is True

    second, second_remote = _manager(tmp_path, monkeypatch, session_key="cli:second")
    assert second.migrate_memory_files("cli:second", str(memory_dir)) is True
    assert _uploaded_names(second_remote) == ["user_profile.md"]


def test_destination_changes_get_independent_state(tmp_path, monkeypatch):
    memory_dir = tmp_path / "memories"
    _write_memory_files(memory_dir, "MEMORY.md")
    first, _ = _manager(tmp_path, monkeypatch, session_key="cli:first")
    assert first.migrate_memory_files("cli:first", str(memory_dir)) is True

    normalized, normalized_remote = _manager(
        tmp_path,
        monkeypatch,
        session_key="cli:normalized",
        base_url="http://127.0.0.1:8000/",
    )
    assert normalized.migrate_memory_files("cli:normalized", str(memory_dir)) is False
    normalized_remote.upload_file.assert_not_called()

    changed_workspace, workspace_remote = _manager(
        tmp_path,
        monkeypatch,
        session_key="cli:workspace",
        workspace_id="other-workspace",
    )
    assert changed_workspace.migrate_memory_files("cli:workspace", str(memory_dir)) is True
    assert _uploaded_names(workspace_remote) == ["consolidated_memory.md"]

    changed_peer, peer_remote = _manager(
        tmp_path,
        monkeypatch,
        session_key="cli:peer",
        user_peer_id="other-user",
    )
    assert changed_peer.migrate_memory_files("cli:peer", str(memory_dir)) is True
    assert _uploaded_names(peer_remote) == ["consolidated_memory.md"]


def test_missing_file_remains_retryable(tmp_path, monkeypatch):
    memory_dir = tmp_path / "memories"
    _write_memory_files(memory_dir, "MEMORY.md")
    first, _ = _manager(tmp_path, monkeypatch, session_key="cli:first")
    assert first.migrate_memory_files("cli:first", str(memory_dir)) is True

    _write_memory_files(memory_dir, "USER.md")
    second, second_remote = _manager(tmp_path, monkeypatch, session_key="cli:second")
    assert second.migrate_memory_files("cli:second", str(memory_dir)) is True
    assert _uploaded_names(second_remote) == ["user_profile.md"]


def test_concurrent_initializations_upload_once(tmp_path, monkeypatch):
    memory_dir = tmp_path / "memories"
    _write_memory_files(memory_dir, "MEMORY.md", "USER.md", "SOUL.md")
    first, shared_remote = _manager(tmp_path, monkeypatch, session_key="cli:first")
    second, _ = _manager(tmp_path, monkeypatch, session_key="cli:second")
    second._sessions_cache["cli-second"] = shared_remote
    results = []

    threads = [
        threading.Thread(
            target=lambda manager=manager, key=key: results.append(
                manager.migrate_memory_files(key, str(memory_dir))
            )
        )
        for manager, key in ((first, "cli:first"), (second, "cli:second"))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert sorted(results) == [False, True]
    assert _uploaded_names(shared_remote) == [
        "consolidated_memory.md",
        "user_profile.md",
        "agent_soul.md",
    ]


def test_corrupt_state_fails_closed(tmp_path, monkeypatch, caplog):
    memory_dir = tmp_path / "memories"
    _write_memory_files(memory_dir, "MEMORY.md")
    state_path = tmp_path / "state" / "honcho_migration.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text("not-json", encoding="utf-8")
    manager, remote = _manager(tmp_path, monkeypatch)

    assert manager.migrate_memory_files("cli:test", str(memory_dir)) is False
    remote.upload_file.assert_not_called()
    assert "migration state is unreadable" in caplog.text


def test_thread_lock_timeout_skips_migration(tmp_path, monkeypatch, caplog):
    memory_dir = tmp_path / "memories"
    _write_memory_files(memory_dir, "MEMORY.md")
    manager, remote = _manager(tmp_path, monkeypatch)

    _MIGRATION_THREAD_LOCK.acquire()
    try:
        with patch(
            "plugins.memory.honcho.session._MIGRATION_LOCK_TIMEOUT_SECONDS",
            0.01,
        ):
            assert manager.migrate_memory_files("cli:test", str(memory_dir)) is False
    finally:
        _MIGRATION_THREAD_LOCK.release()

    remote.upload_file.assert_not_called()
    assert "timed out acquiring thread lock" in caplog.text


def test_ledger_records_files_independently(tmp_path, monkeypatch):
    memory_dir = tmp_path / "memories"
    _write_memory_files(memory_dir, "MEMORY.md", "SOUL.md")
    manager, _ = _manager(tmp_path, monkeypatch)

    assert manager.migrate_memory_files("cli:test", str(memory_dir)) is True

    state = json.loads(
        (tmp_path / "state" / "honcho_migration.json").read_text(encoding="utf-8")
    )
    target = next(iter(state["targets"].values()))
    assert target["files"] == {"MEMORY.md": True, "SOUL.md": True}


def test_state_write_failure_reports_remote_upload(tmp_path, monkeypatch, caplog):
    memory_dir = tmp_path / "memories"
    _write_memory_files(memory_dir, "MEMORY.md")
    manager, remote = _manager(tmp_path, monkeypatch)

    with patch(
        "plugins.memory.honcho.session.atomic_json_write",
        side_effect=OSError("disk full"),
    ):
        assert manager.migrate_memory_files("cli:test", str(memory_dir)) is True

    assert _uploaded_names(remote) == ["consolidated_memory.md"]
    assert "Failed to record Honcho migration" in caplog.text
