from __future__ import annotations

from datetime import datetime

from gateway.config import GatewayConfig, Platform
from gateway.session import (
    SessionEntry,
    SessionSource,
    SessionStore,
    build_session_context,
    build_session_context_prompt,
)
from gateway.session_context import clear_session_vars, get_session_env, set_session_vars
from gateway.workspace_state import normalize_workspace_path
from hermes_cli.commands import resolve_command


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        user_id="user-1",
        user_name="Test User",
    )


def test_workspace_command_registered_with_aliases():
    cmd = resolve_command("workspace")
    assert cmd is not None
    assert cmd.name == "workspace"
    assert cmd.gateway_only is True
    assert resolve_command("ws") is cmd
    assert resolve_command("workspaces") is cmd


def test_session_entry_persists_current_workspace_roundtrip(tmp_path):
    entry = SessionEntry(
        session_key="telegram:chat-1:user-1",
        session_id="sess-1",
        created_at=datetime(2026, 1, 1, 12, 0),
        updated_at=datetime(2026, 1, 1, 12, 1),
        current_workspace=str(tmp_path),
        origin=_source(),
        platform=Platform.TELEGRAM,
    )
    restored = SessionEntry.from_dict(entry.to_dict())
    assert restored.current_workspace == str(tmp_path)


def test_session_store_set_current_workspace(tmp_path):
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    store._db = None
    entry = store.get_or_create_session(_source())

    assert store.set_current_workspace(entry.session_key, str(tmp_path)) is True

    reloaded = SessionStore(tmp_path / "sessions", GatewayConfig())
    reloaded._db = None
    restored = reloaded.get_or_create_session(_source())
    assert restored.current_workspace == str(tmp_path)


def test_build_session_context_includes_workspace(tmp_path):
    entry = SessionEntry(
        session_key="telegram:chat-1:user-1",
        session_id="sess-1",
        created_at=datetime(2026, 1, 1, 12, 0),
        updated_at=datetime(2026, 1, 1, 12, 1),
        current_workspace=str(tmp_path),
    )
    context = build_session_context(_source(), GatewayConfig(), entry)
    assert context.current_workspace == str(tmp_path)
    prompt = build_session_context_prompt(context)
    assert f"**Workspace:** {tmp_path}" in prompt


def test_session_context_terminal_cwd_is_task_local(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINAL_CWD", "/env/fallback")
    tokens = set_session_vars(
        platform="telegram",
        chat_id="chat-1",
        session_key="telegram:chat-1:user-1",
        terminal_cwd=str(tmp_path),
    )
    try:
        assert get_session_env("TERMINAL_CWD") == str(tmp_path)
    finally:
        clear_session_vars(tokens)
    assert get_session_env("TERMINAL_CWD") == ""


def test_session_context_terminal_cwd_preserves_env_fallback_for_unrelated_sessions(monkeypatch):
    monkeypatch.setenv("TERMINAL_CWD", "/env/fallback")
    tokens = set_session_vars(session_key="acp-session")
    try:
        assert get_session_env("TERMINAL_CWD") == "/env/fallback"
    finally:
        clear_session_vars(tokens)


def test_normalize_workspace_requires_existing_directory(tmp_path):
    ok, err = normalize_workspace_path(str(tmp_path))
    assert err is None
    assert ok == str(tmp_path.resolve())

    missing, err = normalize_workspace_path(str(tmp_path / "missing"))
    assert missing is None
    assert "does not exist" in (err or "")
