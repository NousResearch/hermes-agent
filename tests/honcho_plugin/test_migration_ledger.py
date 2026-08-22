"""Behavior contracts for one-time Honcho memory-file migration."""

from __future__ import annotations

import hashlib
import json
import threading
from unittest.mock import MagicMock, patch

import pytest

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
    base_url: str | None = "http://127.0.0.1:8000/v3",
    environment="production",
    workspace_id="workspace",
    user_peer_id="user",
    ai_peer_id="assistant",
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = HonchoClientConfig(
        api_key="key",
        base_url=base_url,
        environment=environment,
        workspace_id=workspace_id,
        write_frequency="turn",
    )
    honcho = MagicMock()
    monkeypatch.setattr(
        "plugins.memory.honcho.session.get_honcho_client",
        lambda *args, **kwargs: honcho,
    )
    manager = HonchoSessionManager(honcho=honcho, config=config)
    session = HonchoSession(
        key=session_key,
        user_peer_id=user_peer_id,
        assistant_peer_id=ai_peer_id,
        honcho_session_id=session_key.replace(":", "-"),
    )
    remote_session = MagicMock()
    remote_marker_session = MagicMock()
    def resolve_remote_session(session_id, **kwargs):
        if session_id == session.honcho_session_id:
            return remote_session
        remote_marker_session.creation_kwargs = kwargs
        return remote_marker_session

    honcho.session.side_effect = resolve_remote_session
    manager._cache[session.key] = session
    manager._sessions_cache[session.honcho_session_id] = remote_session
    manager._peers_cache[user_peer_id] = MagicMock(name=f"peer-{user_peer_id}")
    manager._peers_cache[ai_peer_id] = MagicMock(name=f"peer-{ai_peer_id}")
    return manager, remote_marker_session


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
    assert first.migrate_memory_files("cli:first", str(memory_dir)) is True

    second, second_remote = _manager(tmp_path, monkeypatch, session_key="cli:second")
    assert second.migrate_memory_files("cli:second", str(memory_dir)) is False

    assert _uploaded_names(first_remote) == [
        "consolidated_memory.md",
        "user_profile.md",
        "agent_soul.md",
    ]
    marker_peer_configs = first_remote.creation_kwargs["peers"]
    assert first_remote.creation_kwargs["metadata"]["source"] == (
        "hermes_memory_migration"
    )
    assert {peer_id for peer_id, _ in marker_peer_configs} == {"user", "assistant"}
    assert all(
        config.observe_me is True and config.observe_others is False
        for _, config in marker_peer_configs
    )
    second_remote.upload_file.assert_not_called()


def test_same_named_files_from_different_sources_each_upload(tmp_path, monkeypatch):
    first_dir = tmp_path / "workspace"
    second_dir = tmp_path / ".openclaw"
    _write_memory_files(first_dir, "MEMORY.md", "USER.md")
    _write_memory_files(second_dir, "MEMORY.md", "USER.md")
    manager, remote = _manager(tmp_path, monkeypatch)

    assert manager.migrate_memory_files("cli:test", str(first_dir)) is True
    assert manager.migrate_memory_files("cli:test", str(second_dir)) is True
    assert manager.migrate_memory_files("cli:test", str(first_dir)) is False
    assert manager.migrate_memory_files("cli:test", str(second_dir)) is False

    assert _uploaded_names(remote) == [
        "consolidated_memory.md",
        "user_profile.md",
        "consolidated_memory.md",
        "user_profile.md",
    ]


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


def test_environment_and_equivalent_url_share_destination_state(
    tmp_path, monkeypatch
):
    memory_dir = tmp_path / "memories"
    _write_memory_files(memory_dir, "MEMORY.md")
    environment_manager, _ = _manager(
        tmp_path,
        monkeypatch,
        session_key="cli:environment",
        base_url=None,
        environment="production",
    )
    assert (
        environment_manager.migrate_memory_files(
            "cli:environment", str(memory_dir)
        )
        is True
    )

    explicit_manager, explicit_remote = _manager(
        tmp_path,
        monkeypatch,
        session_key="cli:explicit",
        base_url="https://api.honcho.dev/v3",
    )
    assert explicit_manager.migrate_memory_files("cli:explicit", str(memory_dir)) is False
    explicit_remote.upload_file.assert_not_called()


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
    target = {
        "endpoint": "http://127.0.0.1:8000",
        "workspace_id": "workspace",
        "user_peer_id": "user",
        "ai_peer_id": "assistant",
    }
    target_key = hashlib.sha256(
        json.dumps(target, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    marker_session_id = f"hermes-memory-migration-{target_key}"
    assert len(marker_session_id) <= HonchoClientConfig._HONCHO_SESSION_ID_MAX_LEN
    first._sessions_cache[marker_session_id] = shared_remote
    second._sessions_cache[marker_session_id] = shared_remote
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
    source = next(iter(target["sources"].values()))
    assert source == {
        "path": str(memory_dir),
        "files": {"MEMORY.md": True, "SOUL.md": True},
    }


@pytest.mark.parametrize(
    ("filename", "upload_name"),
    [
        ("MEMORY.md", "consolidated_memory.md"),
        ("USER.md", "user_profile.md"),
        ("SOUL.md", "agent_soul.md"),
    ],
)
def test_state_write_failure_reconciles_remote_upload_after_session_changes(
    tmp_path, monkeypatch, caplog, filename, upload_name
):
    memory_dir = tmp_path / "memories"
    _write_memory_files(memory_dir, filename)
    first, first_remote = _manager(tmp_path, monkeypatch)
    uploaded_migration_ids: set[str] = set()

    def remember_upload(*, metadata, **kwargs):
        uploaded_migration_ids.add(metadata["migration_id"])

    first_remote.upload_file.side_effect = remember_upload
    first_remote.messages.side_effect = lambda **kwargs: (
        [MagicMock()]
        if kwargs["filters"]["metadata"]["migration_id"] in uploaded_migration_ids
        else []
    )

    with patch(
        "plugins.memory.honcho.session.atomic_json_write",
        side_effect=OSError("disk full"),
    ):
        assert first.migrate_memory_files("cli:test", str(memory_dir)) is True

    retry, retry_remote = _manager(
        tmp_path,
        monkeypatch,
        session_key="cli:retry",
    )
    marker_session_id = next(
        session_id
        for session_id in first._sessions_cache
        if session_id.startswith("hermes-memory-migration-")
    )
    retry._sessions_cache[marker_session_id] = first_remote

    assert retry.migrate_memory_files("cli:retry", str(memory_dir)) is False
    assert _uploaded_names(first_remote) == [upload_name]
    retry_remote.upload_file.assert_not_called()
    assert "Failed to record Honcho migration" in caplog.text

    upload_metadata = first_remote.upload_file.call_args.kwargs["metadata"]
    retry_filter = first_remote.messages.call_args_list[-1].kwargs["filters"]
    assert retry_filter["metadata"]["migration_id"] == upload_metadata["migration_id"]

    state = json.loads(
        (tmp_path / "state" / "honcho_migration.json").read_text(encoding="utf-8")
    )
    source = next(iter(next(iter(state["targets"].values()))["sources"].values()))
    assert source["files"] == {filename: True}


def test_remote_reconciliation_continues_with_remaining_files(
    tmp_path, monkeypatch
):
    memory_dir = tmp_path / "memories"
    _write_memory_files(memory_dir, "MEMORY.md", "USER.md", "SOUL.md")
    first, shared_remote = _manager(tmp_path, monkeypatch)
    uploaded_migration_ids: set[str] = set()

    def remember_upload(*, metadata, **kwargs):
        uploaded_migration_ids.add(metadata["migration_id"])

    shared_remote.upload_file.side_effect = remember_upload
    shared_remote.messages.side_effect = lambda **kwargs: (
        [MagicMock()]
        if kwargs["filters"]["metadata"]["migration_id"] in uploaded_migration_ids
        else []
    )

    with patch(
        "plugins.memory.honcho.session.atomic_json_write",
        side_effect=OSError("disk full"),
    ):
        assert first.migrate_memory_files("cli:test", str(memory_dir)) is True

    retry, retry_remote = _manager(tmp_path, monkeypatch, session_key="cli:retry")
    marker_session_id = next(
        session_id
        for session_id in first._sessions_cache
        if session_id.startswith("hermes-memory-migration-")
    )
    retry._sessions_cache[marker_session_id] = shared_remote

    assert retry.migrate_memory_files("cli:retry", str(memory_dir)) is True
    assert _uploaded_names(shared_remote) == [
        "consolidated_memory.md",
        "user_profile.md",
        "agent_soul.md",
    ]
    retry_remote.upload_file.assert_not_called()

    state = json.loads(
        (tmp_path / "state" / "honcho_migration.json").read_text(encoding="utf-8")
    )
    source = next(iter(next(iter(state["targets"].values()))["sources"].values()))
    assert source["files"] == {
        "MEMORY.md": True,
        "USER.md": True,
        "SOUL.md": True,
    }


def test_remote_reconciliation_failure_does_not_risk_duplicate_upload(
    tmp_path, monkeypatch, caplog
):
    memory_dir = tmp_path / "memories"
    _write_memory_files(memory_dir, "MEMORY.md")
    manager, remote = _manager(tmp_path, monkeypatch)
    remote.messages.side_effect = RuntimeError("remote unavailable")

    assert manager.migrate_memory_files("cli:test", str(memory_dir)) is False
    remote.upload_file.assert_not_called()
    assert "skipping upload" in caplog.text
