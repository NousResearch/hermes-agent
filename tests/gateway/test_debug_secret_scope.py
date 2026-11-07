"""BJS-3738: chat /debug scrubs neutral-named secrets before uploading its report."""

from unittest.mock import patch

import pytest

from agent.secret_scope import build_profile_secret_scope, reset_secret_scope, set_secret_scope
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.mark.asyncio
async def test_chat_debug_scrubs_profile_values_before_upload(tmp_path, monkeypatch):
    from gateway.run import GatewayRunner

    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "launch"))
    value = "opaque-chat-diagnostic-abcdefgh"
    (home / ".env").write_text(f"NEUTRAL_FIELD={value}\n")
    logs = home / "logs"
    logs.mkdir()
    (logs / "gateway.log").write_text(f"NEUTRAL_FIELD={value} | useful log detail\n")
    home_token = set_hermes_home_override(home)
    scope_token = set_secret_scope(build_profile_secret_scope(home), profile_home=str(home))
    uploaded = []
    runner = object.__new__(GatewayRunner)
    # Exercise the real context-preserving thread hop with asyncio's default executor.
    runner._get_executor = lambda: None
    event = MessageEvent(text="/debug", source=SessionSource(
        platform=Platform.TELEGRAM, chat_id="test-chat", user_id="test-user", chat_type="dm"))
    try:
        with patch("hermes_cli.dump.run_dump", side_effect=lambda *a, **kw: print(
                f"NEUTRAL_FIELD={value} | useful dump detail")), \
                patch("hermes_cli.debug._best_effort_sweep_expired_pastes"), \
                patch("hermes_cli.debug._schedule_auto_delete"), \
                patch("hermes_cli.debug.upload_to_pastebin", side_effect=lambda report:
                      uploaded.append(report) or "https://paste.rs/synthetic"):
            response = await runner._handle_debug_command(event)
        assert len(uploaded) == 1
        assert "useful log detail" in uploaded[0]
        assert "useful dump detail" in uploaded[0]
        assert value not in uploaded[0]
        assert "https://paste.rs/synthetic" in response
    finally:
        reset_secret_scope(scope_token)
        reset_hermes_home_override(home_token)
