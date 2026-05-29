from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.verify_command import (
    parse_verify_command,
    redeem_verify_code,
    verify_ack_text,
)


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("verify 123456", "123456"),
        ("/verify 123456", "123456"),
        ("Verify 123456", "123456"),
    ],
)
def test_parse_verify_command_accepts_valid_forms(text, code):
    assert parse_verify_command(text) == {"code": code}


@pytest.mark.parametrize(
    "text",
    [
        "verify abc",
        "verifying 123456",
        "/verify 12345",
        "hello",
    ],
)
def test_parse_verify_command_rejects_non_commands(text):
    assert parse_verify_command(text) is None


@pytest.mark.parametrize(
    ("result", "text"),
    [
        ({"ok": True}, "✅ 已將你加為 owner，現在可以開始對話"),
        ({"ok": False, "reason": "already_owner"}, "ℹ️ 你已經是 owner"),
        ({"ok": False, "reason": "rate_limited"}, "❌ 嘗試太多次，請稍後再試"),
        (
            {"ok": False, "reason": "invalid_or_expired"},
            "❌ 驗證碼無效或已過期，請聯絡管理員重新產生",
        ),
        ({"ok": False, "reason": "network_error"}, "❌ 驗證失敗，請稍後再試"),
    ],
)
def test_verify_ack_text_maps_results(result, text):
    assert verify_ack_text(result) == text


@pytest.mark.asyncio
async def test_redeem_verify_code_returns_network_error_when_env_unset(monkeypatch):
    monkeypatch.delenv("HERMES_SAAS_URL", raising=False)
    monkeypatch.delenv("HERMES_INTERNAL_TOKEN", raising=False)

    result = await redeem_verify_code(
        platform="discord",
        code="123456",
        user_id="U123",
    )

    assert result == {"ok": False, "reason": "network_error"}


@pytest.mark.asyncio
async def test_redeem_verify_code_posts_to_saas_redeem_endpoint(monkeypatch):
    monkeypatch.setenv("HERMES_SAAS_URL", "https://saas.example.test/")
    monkeypatch.setenv("HERMES_INTERNAL_TOKEN", "token-ref")

    response = AsyncMock()
    response.status = 200
    response.json = AsyncMock(return_value={"ok": True})
    response_cm = AsyncMock()
    response_cm.__aenter__.return_value = response

    session = AsyncMock()
    session.post = MagicMock(return_value=response_cm)
    session_cm = AsyncMock()
    session_cm.__aenter__.return_value = session

    client_session = MagicMock(return_value=session_cm)
    monkeypatch.setattr("gateway.verify_command.aiohttp.ClientSession", client_session)

    result = await redeem_verify_code(
        platform="telegram",
        code="123456",
        user_id="U123",
    )

    assert result == {"ok": True}
    session.post.assert_called_once()
    url = session.post.call_args.args[0]
    kwargs = session.post.call_args.kwargs
    assert url == "https://saas.example.test/api/integrations/pairing-code/redeem"
    assert kwargs["json"] == {
        "platform": "telegram",
        "code": "123456",
        "user_id": "U123",
    }
    assert kwargs["headers"] == {"X-Hermes-Token": "token-ref"}
