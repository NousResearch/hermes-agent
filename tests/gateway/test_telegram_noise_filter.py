"""Telegram-specific gateway filtering for noisy status/error output."""

from gateway.config import Platform
from gateway.run import (
    _prepare_gateway_status_message,
    _sanitize_gateway_final_response,
)


def test_telegram_status_suppresses_auxiliary_and_retry_noise():
    """Auxiliary failures and retry backoff chatter should not hit Telegram."""
    noisy_messages = [
        "⚠ Auxiliary title generation failed: HTTP 400: Operation contains cybersecurity risk",
        "⚠ context summary 失败：upstream error。已插入 fallback context marker。",
        "⚠ Compression summary failed: upstream error. Inserted a fallback context marker.",
        "🗜️ Compacting context — summarizing earlier conversation so I can continue...",
        "ℹ Configured compression model 'small-model' failed (timeout). Recovered using main model — check auxiliary.compression.model in config.yaml.",
        "⏳ Retrying in 4.2s (attempt 1/3)...",
        "⏱️ Rate limited. Waiting 30.0s (attempt 2/3)...",
        "⚠️ Max retries (3) exhausted — trying fallback...",
    ]

    for message in noisy_messages:
        assert _prepare_gateway_status_message(Platform.TELEGRAM, "warn", message) is None


def test_non_chatty_status_is_unchanged():
    """The chat-specific policy must not hide CLI/Discord diagnostics."""
    message = "⏳ Retrying in 4.2s (attempt 1/3)..."

    assert _prepare_gateway_status_message(Platform.DISCORD, "lifecycle", message) == message
    assert _prepare_gateway_status_message("local", "lifecycle", message) == message


def test_qqbot_status_localizes_retry_backoff():
    """QQ should see Chinese gateway retry status, not English chatter."""
    message = "⏳ Retrying in 4.2s (attempt 1/3)..."

    localized = _prepare_gateway_status_message(Platform.QQBOT, "lifecycle", message)

    assert localized == "⏳ 4.2 秒后重试（第 1/3 次）..."
    assert "Retrying" not in localized
    assert "attempt" not in localized


def test_qqbot_status_localizes_common_runtime_statuses():
    """QQ gets Chinese Hermes runtime envelopes while preserving technical tokens."""
    cases = {
        "⚠️ Empty/malformed response — switching to fallback...": "⚠️ 模型返回空或格式异常，正在切换到备用模型...",
        "⚠️ Rate limited — switching to fallback provider...": "⚠️ 已被限流，正在切换到备用提供商...",
        "⚠️  Request payload too large (413) — compression attempt 1/3...": "⚠️ 请求内容过大 (413)，正在压缩（第 1/3 次）...",
        "🗜️ Context too large (~12,345 tokens) — compressing (1/3)...": "🗜️ 上下文过大（约 12,345 tokens），正在压缩（第 1/3 次）...",
        "🗜️ Compressed 99 → 10 messages, retrying...": "🗜️ 已压缩 99 → 10 条消息，正在重试...",
        "⚠️ Non-retryable error (HTTP 400) — trying fallback...": "⚠️ 不可重试错误 (HTTP 400)，正在尝试备用模型...",
        "❌ Rate limited after 3 retries — HTTP 429 too many": "❌ 重试 3 次后仍被限流 — HTTP 429 too many",
        "❌ API failed after 3 retries — timeout": "❌ API 重试 3 次后仍失败 — timeout",
        "❌ API failed after 3 retries — Responses create(stream=True) fallback did not emit a terminal response.": "❌ API 重试 3 次后仍失败 — Responses create(stream=True) fallback 未产生终止响应。",
        "↻ Empty response after tool calls — using earlier content as final answer": "↻ 工具调用后模型返回空内容，改用较早内容作为最终回复",
        "⚠️ Model returned empty after tool calls — nudging to continue": "⚠️ 工具调用后模型返回空内容，正在提示模型继续",
        "↻ Thinking-only response — prefilling to continue (1/2)": "↻ 模型只返回思考内容，正在预填继续（第 1/2 次）",
        "⚠️ Empty response from model — retrying (1/3)": "⚠️ 模型返回空内容，正在重试（第 1/3 次）",
        "⚠️ Model returning empty responses — switching to fallback provider...": "⚠️ 模型持续返回空内容，正在切换到备用提供商...",
        "↻ Switched to fallback: gpt-4 (openai)": "↻ 已切换到备用模型：gpt-4 (openai)",
        "⚠️ Iteration budget exhausted (90/90) — asking model to summarise": "⚠️ 轮次预算已用尽（90/90），正在请求模型总结",
    }

    for original, expected in cases.items():
        localized = _prepare_gateway_status_message(Platform.QQBOT, "lifecycle", original)
        assert localized == expected
        assert "retrying" not in localized.lower()
        assert "switching" not in localized.lower()
        assert "fallback provider" not in localized.lower()
        assert "empty response" not in localized.lower()
        assert "model returned" not in localized.lower()


def test_qqbot_final_response_localizes_provider_failure_envelope():
    """QQ final replies should hide English provider failure envelopes."""
    raw = (
        "API call failed after 3 retries: Responses create(stream=True) "
        "fallback did not emit a terminal response."
    )

    sanitized = _sanitize_gateway_final_response(Platform.QQBOT, raw)

    assert sanitized == "⚠️ 模型提供商多次重试后仍失败。原始细节未发送到聊天；可查看 gateway logs。"
    assert "API call failed" not in sanitized
    assert "did not emit" not in sanitized


def test_qqbot_localizes_stream_stalled_tool_call_warning():
    """QQ should not receive the English dropped tool-call warning envelope."""
    raw = (
        "⚠ Stream stalled mid tool-call (write_file); the action was not "
        "executed. Ask me to retry if you want to continue."
    )

    expected = "⚠ 流式响应在工具调用中途停住（write_file）；操作未执行。需要继续的话，请让我重试。"

    status = _prepare_gateway_status_message(Platform.QQBOT, "lifecycle", raw)
    final = _sanitize_gateway_final_response(Platform.QQBOT, raw)

    assert status == expected
    assert final == expected
    for localized in (status, final):
        assert "Stream stalled" not in localized
        assert "action was not executed" not in localized
        assert "Ask me to retry" not in localized


def test_telegram_status_sanitizes_raw_provider_security_errors():
    """Provider policy/security bodies should be replaced before chat delivery."""
    raw = (
        "❌ API failed after 3 retries — HTTP 400: request blocked because "
        "Operation contains cybersecurity risk. request_id=req_123"
    )

    sanitized = _prepare_gateway_status_message(Platform.TELEGRAM, "lifecycle", raw)

    assert sanitized is not None
    assert "provider rejected" in sanitized.lower()
    assert "cybersecurity risk" not in sanitized.lower()
    assert "HTTP 400" not in sanitized
    assert "req_123" not in sanitized


def test_telegram_final_response_sanitizes_raw_provider_errors():
    """Final Telegram replies should not expose raw provider/security details."""
    raw = (
        "API call failed after 3 retries: HTTP 400: This request was blocked "
        "under the provider cybersecurity risk policy. request_id=req_abc"
    )

    sanitized = _sanitize_gateway_final_response(Platform.TELEGRAM, raw)

    assert "provider rejected" in sanitized.lower()
    assert "cybersecurity risk" not in sanitized.lower()
    assert "HTTP 400" not in sanitized
    assert "req_abc" not in sanitized


def test_telegram_final_response_redacts_auth_secrets():
    """Authentication errors should be useful without leaking key material."""
    raw = (
        "⚠️ Provider authentication failed: Incorrect API key provided: "
        "sk-live_abcdefghijklmnopqrstuvwxyz1234567890"
    )

    sanitized = _sanitize_gateway_final_response(Platform.TELEGRAM, raw)

    assert "authentication failed" in sanitized.lower()
    assert "check the configured credentials" in sanitized.lower()
    assert "sk-live" not in sanitized


def test_telegram_final_response_keeps_normal_answers():
    """Normal assistant content should not be rewritten."""
    answer = "Here is the clean summary you asked for."

    assert _sanitize_gateway_final_response(Platform.TELEGRAM, answer) == answer
