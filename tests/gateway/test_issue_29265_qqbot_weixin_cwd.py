"""Issue #29265 QQBot/Weixin smoke coverage.

The full cwd behavior is covered by the common gateway/tool contract tests.
This file keeps a narrow regression label for the platforms named in #29265
without pretending to exercise live QQBot/Weixin network adapters.
"""

from __future__ import annotations

import pytest

from agent import prompt_builder
from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, build_session_context


@pytest.mark.parametrize("platform", [Platform.QQBOT, Platform.WEIXIN])
def test_qqbot_weixin_sessions_inherit_gateway_terminal_cwd(monkeypatch, tmp_path, platform):
    """QQBot/Weixin session sources should not alter the shared cwd contract."""
    workspace = tmp_path / "hermes-gateway-workspace"
    workspace.mkdir()

    monkeypatch.setattr(prompt_builder, "is_wsl", lambda: False)
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    monkeypatch.setenv("TERMINAL_CWD", str(workspace))
    prompt_builder._clear_backend_probe_cache()

    source = SessionSource(
        platform=platform,
        chat_id="chat-1",
        chat_name="DM",
        chat_type="dm",
        user_id="user-1",
        user_name="Test User",
    )

    context = build_session_context(source, config=GatewayConfig(), session_entry=None)
    hints = prompt_builder.build_environment_hints()

    assert context.source.platform == platform
    assert f"Current working directory: {workspace}" in hints
