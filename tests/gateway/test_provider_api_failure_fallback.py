"""Gateway fallback copy for provider/API failures."""

from gateway.run import _format_provider_api_failure_for_gateway


def test_openai_request_id_failure_is_hidden_from_user_response():
    raw = (
        "API call failed after 3 retries: An error occurred while processing your request. "
        "You can retry your request, or contact us through our help center at help.openai.com "
        "if the error persists. Please include the request ID "
        "45c0c992-18ae-44c0-a62a-2caf0563b98b in your message."
    )

    visible = _format_provider_api_failure_for_gateway(
        {"failed": True, "error": raw},
        raw,
    )

    assert visible == "⚠️ モデル応答が一時的に失敗しました。少し待って再実行してください。\n詳細はログに記録しました。"
    assert "API call failed" not in visible
    assert "help.openai.com" not in visible
    assert "45c0c992" not in visible
    assert "request ID" not in visible


def test_rate_limit_failure_gets_wait_and_retry_copy():
    raw = "API call failed after 3 retries: 429 Too Many Requests — rate limit exceeded"

    visible = _format_provider_api_failure_for_gateway(
        {"failed": True, "error": "429 Too Many Requests"},
        raw,
    )

    assert "混み合っています" in visible
    assert "再実行" in visible
    assert "429" not in visible


def test_context_provider_failure_keeps_actionable_compact_guidance():
    raw = "API call failed after 3 retries: prompt is too long: context window exceeded"

    visible = _format_provider_api_failure_for_gateway(
        {"failed": True, "error": raw},
        raw,
    )

    assert "/compact" in visible
    assert "/reset" in visible
    assert "prompt is too long" not in visible


def test_non_provider_failure_response_is_unchanged():
    response = "普通の応答です。"

    visible = _format_provider_api_failure_for_gateway(
        {"completed": True},
        response,
    )

    assert visible == response
