"""Cron + openai-codex resilience: retry budget and plain-English failure notices."""

from agent.openai_codex_resilience import (
    classify_openai_codex_error,
    format_openai_codex_failure_notice,
    maybe_format_codex_cron_failure,
    resolve_openai_codex_retry_budget,
)


def test_cron_openai_codex_retry_budget_defaults_to_five():
    assert (
        resolve_openai_codex_retry_budget(
            platform="cron",
            provider="openai-codex",
            default_retries=3,
        )
        == 5
    )


def test_classify_live_503_signature():
    err = RuntimeError(
        "HTTP 503: upstream connect error or disconnect/reset before headers. "
        "delayed connect error: Connection refused"
    )
    err.status_code = 503  # type: ignore[attr-defined]
    classified = classify_openai_codex_error(err, status_code=503)
    assert classified.error_class == "upstream_503"
    assert classified.transient is True


def test_maybe_format_codex_cron_failure_plain_english():
    agent = type(
        "Agent",
        (),
        {
            "platform": "cron",
            "provider": "openai-codex",
            "model": "gpt-5.5",
        },
    )()
    notice = maybe_format_codex_cron_failure(
        agent,
        "HTTP 503: upstream connect error",
        attempts=5,
        job_name="watchdog",
    )
    assert notice is not None
    assert "gpt-5.5" in notice
    assert "not a Kynver harness failure" in notice
    assert "not the Kynver runtime" in notice


def test_interactive_cli_keeps_default_retries():
    assert (
        resolve_openai_codex_retry_budget(
            platform="cli",
            provider="openai-codex",
            default_retries=3,
        )
        == 3
    )


def test_failure_notice_mentions_provider_classification():
    notice = format_openai_codex_failure_notice(
        job_name="watchdog",
        provider="openai-codex",
        model="gpt-5.5",
        error_class="sse_no_event",
        summary="Codex stream produced no bytes within 12s",
        attempts=5,
        degraded=True,
    )
    assert "Codex stream never started" in notice
