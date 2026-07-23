from types import SimpleNamespace

import pytest

from gateway.user_friendly_status import (
    LatestStatusGeneration,
    UserFriendlyStatusFilter,
    resolve_user_friendly_status_config,
)


def _response(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


def test_resolves_platform_override_and_defaults():
    config = {
        "display": {
            "user_friendly_status": {"enabled": False},
            "platforms": {
                "whatsapp": {
                    "user_friendly_status": {
                        "enabled": True,
                        "timeout": 1.5,
                        "max_chars": 100,
                    }
                }
            },
        }
    }

    resolved = resolve_user_friendly_status_config(config, "whatsapp")

    assert resolved.enabled is True
    assert resolved.timeout == 1.5
    assert resolved.max_chars == 100
    assert resolved.fallback == "suppress"


@pytest.mark.asyncio
async def test_rewrites_tool_activity_as_one_user_friendly_line():
    calls = []

    async def fake_llm(**kwargs):
        calls.append(kwargs)
        return _response("Checking the latest meeting details now.\nExtra detail")

    status_filter = UserFriendlyStatusFilter(enabled=True, llm_call=fake_llm)
    result = await status_filter.rewrite(
        kind="tool_progress",
        text='terminal: cli-anything-echo get abc --transcript',
    )

    assert result == "Checking the latest meeting details now."
    assert calls[0]["task"] == "status_filter"
    assert calls[0]["max_tokens"] <= 64


@pytest.mark.asyncio
async def test_skip_token_suppresses_internal_noise():
    async def fake_llm(**kwargs):
        return _response("SKIP")

    status_filter = UserFriendlyStatusFilter(enabled=True, llm_call=fake_llm)

    assert await status_filter.rewrite(
        kind="lifecycle",
        text="Compression threshold auto-lowered to 272000 tokens",
    ) is None


@pytest.mark.asyncio
async def test_failure_suppresses_raw_message_by_default():
    async def failing_llm(**kwargs):
        raise TimeoutError("too slow")

    status_filter = UserFriendlyStatusFilter(enabled=True, llm_call=failing_llm)

    assert await status_filter.rewrite(
        kind="tool_progress",
        text="terminal: secret internal command",
    ) is None


@pytest.mark.asyncio
async def test_disabled_filter_preserves_existing_message():
    status_filter = UserFriendlyStatusFilter(enabled=False)

    assert await status_filter.rewrite(kind="status", text="still working") == "still working"


@pytest.mark.asyncio
async def test_raw_fallback_never_returns_internal_diagnostics(monkeypatch):
    calls = []

    def fake_redact(text, *, force=False):
        calls.append(force)
        return "safe text"

    async def failing_llm(**kwargs):
        raise TimeoutError("too slow")

    monkeypatch.setattr("gateway.user_friendly_status.redact_sensitive_text", fake_redact)
    status_filter = UserFriendlyStatusFilter(
        enabled=True,
        fallback="raw",
        llm_call=failing_llm,
    )

    assert await status_filter.rewrite(
        kind="tool_progress",
        text="command with secret",
    ) == "Still working."
    assert calls == [True]


def test_latest_status_generation_rejects_superseded_update():
    generations = LatestStatusGeneration()

    older = generations.next()
    newer = generations.next()

    assert generations.is_current(older) is False
    assert generations.is_current(newer) is True
