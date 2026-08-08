"""Behavioral tests for Honcho's saveMessages persistence boundary."""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from plugins.memory.honcho import HonchoMemoryProvider
from plugins.memory.honcho.client import HonchoClientConfig


def _initialize_provider(monkeypatch, tmp_path, *, save_messages: bool):
    config = HonchoClientConfig(
        api_key="test-key",
        enabled=True,
        recall_mode="tools",
        init_on_session_start=True,
        save_messages=save_messages,
        session_strategy="per-directory",
    )
    remote_session = MagicMock(name="remote-session")
    manager = MagicMock(name="manager")
    manager.get_or_create.return_value = SimpleNamespace(messages=[])

    def migrate_memory_files(*_args, **_kwargs):
        remote_session.upload_file(file=("consolidated_memory.md", b"memory", "text/plain"))
        return True

    manager.migrate_memory_files.side_effect = migrate_memory_files

    monkeypatch.setattr(
        "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
        lambda: config,
    )
    monkeypatch.setattr(
        "plugins.memory.honcho.client.get_honcho_client",
        lambda _config: MagicMock(name="honcho-client"),
    )
    monkeypatch.setattr(
        "plugins.memory.honcho.session.HonchoSessionManager",
        lambda **_kwargs: manager,
    )
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)

    provider = HonchoMemoryProvider()
    provider.initialize("test-session")
    return provider, manager, remote_session


def _ready_provider(*, save_messages: bool):
    provider = HonchoMemoryProvider()
    provider._config = HonchoClientConfig(
        save_messages=save_messages,
        message_max_chars=25000,
    )
    provider._session_key = "test-session"
    provider._session_initialized = True

    remote_session = MagicMock(name="remote-session")
    local_session = MagicMock(name="local-session")
    manager = MagicMock(name="manager")
    manager.get_or_create.return_value = local_session
    manager._flush_session.side_effect = lambda _session: remote_session.add_messages(
        ["persisted-turn"]
    )
    manager.flush_all.side_effect = lambda: remote_session.add_messages(
        ["persisted-session"]
    )
    provider._manager = manager
    return provider, manager, local_session, remote_session


def test_save_messages_false_skips_startup_memory_file_migration(
    monkeypatch, tmp_path
):
    provider, manager, remote_session = _initialize_provider(
        monkeypatch, tmp_path, save_messages=False
    )

    assert provider._session_initialized is True
    manager.get_or_create.assert_called_once_with(provider._session_key)
    manager.migrate_memory_files.assert_not_called()
    remote_session.upload_file.assert_not_called()
    remote_session.add_messages.assert_not_called()


def test_save_messages_true_preserves_startup_memory_file_migration(
    monkeypatch, tmp_path
):
    provider, manager, remote_session = _initialize_provider(
        monkeypatch, tmp_path, save_messages=True
    )

    assert provider._session_initialized is True
    manager.migrate_memory_files.assert_called_once()
    remote_session.upload_file.assert_called_once()


def test_save_messages_false_skips_turn_persistence():
    provider, manager, local_session, remote_session = _ready_provider(
        save_messages=False
    )

    provider.sync_turn("private user message", "private assistant response")

    assert provider._sync_thread is None
    manager.get_or_create.assert_not_called()
    local_session.add_message.assert_not_called()
    manager._flush_session.assert_not_called()
    remote_session.add_messages.assert_not_called()


def test_save_messages_true_preserves_turn_persistence():
    provider, manager, local_session, remote_session = _ready_provider(
        save_messages=True
    )

    provider.sync_turn("user message", "assistant response")
    sync_thread = provider._sync_thread
    assert sync_thread is not None
    sync_thread.join(timeout=1)

    assert not sync_thread.is_alive()
    assert local_session.add_message.call_args_list[0].args == ("user", "user message")
    assert local_session.add_message.call_args_list[1].args == (
        "assistant",
        "assistant response",
    )
    manager._flush_session.assert_called_once_with(local_session)
    remote_session.add_messages.assert_called_once_with(["persisted-turn"])


def test_save_messages_false_skips_automatic_memory_mirroring():
    provider, manager, _local_session, remote_session = _ready_provider(
        save_messages=False
    )

    provider.on_memory_write("add", "user", "private profile fact")

    manager.create_conclusion.assert_not_called()
    remote_session.add_messages.assert_not_called()
    remote_session.upload_file.assert_not_called()


def test_save_messages_true_preserves_automatic_memory_mirroring():
    provider, manager, _local_session, _remote_session = _ready_provider(
        save_messages=True
    )

    provider.on_memory_write("add", "user", "profile fact")

    deadline = time.time() + 1
    while time.time() < deadline and not manager.create_conclusion.called:
        time.sleep(0.01)
    manager.create_conclusion.assert_called_once_with("test-session", "profile fact")


def test_save_messages_false_skips_session_end_and_shutdown_flushes():
    provider, manager, _local_session, remote_session = _ready_provider(
        save_messages=False
    )

    provider.on_session_end([])
    provider.shutdown()

    manager.flush_all.assert_not_called()
    remote_session.add_messages.assert_not_called()
    remote_session.upload_file.assert_not_called()


def test_save_messages_true_preserves_session_end_flush():
    provider, manager, _local_session, remote_session = _ready_provider(
        save_messages=True
    )

    provider.on_session_end([])

    manager.flush_all.assert_called_once_with()
    remote_session.add_messages.assert_called_once_with(["persisted-session"])


def test_save_messages_false_does_not_disable_explicit_conclusions():
    provider, manager, _local_session, remote_session = _ready_provider(
        save_messages=False
    )
    manager.create_conclusion.return_value = True

    result = provider.handle_tool_call(
        "honcho_conclude",
        {"conclusion": "User explicitly approved this durable fact"},
    )

    assert "Conclusion saved" in result
    manager.create_conclusion.assert_called_once_with(
        "test-session",
        "User explicitly approved this durable fact",
        peer="user",
    )
    remote_session.add_messages.assert_not_called()
    remote_session.upload_file.assert_not_called()
