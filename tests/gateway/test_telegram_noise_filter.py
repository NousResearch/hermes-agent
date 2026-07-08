"""Gateway noise/secret filtering across chat surfaces (Telegram + siblings)."""

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import (
    _prepare_gateway_status_message,
    _sanitize_gateway_final_response,
)

# Every human-facing chat surface that must receive noise-filtered,
# secret-redacted, provider-error-sanitized output (not just Telegram).
CHAT_PLATFORMS = [
    "telegram",
    "whatsapp",
    "discord",
    "slack",
    "signal",
    "matrix",
    "mattermost",
    "dingtalk",
    "feishu",
    "wecom",
    "weixin",
    "bluebubbles",
    "qqbot",
    "homeassistant",
    "sms",
]

NOISY_STATUS_MESSAGES = [
    "🗜️ Preflight compression check before sending...",
    "🗜️ Compacting context — summarizing earlier conversation so I can continue...",
    "⚠️  Session compressed 12 times — accuracy may degrade. Consider /new to start fresh.",
    "⚠ Compression summary failed: upstream error. Inserted a fallback context marker.",
    "⏱️ Rate limited. Waiting 30.0s (attempt 2/3)...",
    "⏳ Retrying in 4.2s (attempt 1/3)...",
]


def test_telegram_status_suppresses_auxiliary_and_retry_noise():
    """Auxiliary failures and retry backoff chatter should not hit Telegram."""
    noisy_messages = [
        "⚠ Auxiliary title generation failed: HTTP 400: Operation contains cybersecurity risk",
        "⚠ Compression summary failed: upstream error. Inserted a fallback context marker.",
        "🗜️ Compacting context — summarizing earlier conversation so I can continue...",
        "ℹ Configured compression model 'small-model' failed (timeout). Recovered using main model — check auxiliary.compression.model in config.yaml.",
        "⏳ Retrying in 4.2s (attempt 1/3)...",
        "⏱️ Rate limited. Waiting 30.0s (attempt 2/3)...",
        "⚠️ Max retries (3) exhausted — trying fallback...",
    ]

    for message in noisy_messages:
        assert _prepare_gateway_status_message(Platform.TELEGRAM, "warn", message) is None


def test_programmatic_surfaces_keep_raw_status():
    """Programmatic surfaces (local/api/webhook) must keep raw diagnostics.

    Negative case for the invariant: the chat-noise filter must not touch
    CLI/TUI diagnostics, API JSON, or webhook payloads.
    """
    message = "⏳ Retrying in 4.2s (attempt 1/3)..."

    for platform in ("local", "api_server", "webhook", "msgraph_webhook"):
        assert (
            _prepare_gateway_status_message(platform, "lifecycle", message) == message
        )


@pytest.mark.parametrize("platform", CHAT_PLATFORMS)
@pytest.mark.parametrize("message", NOISY_STATUS_MESSAGES)
def test_all_chat_gateways_suppress_noise(platform, message):
    """Operational lifecycle/retry noise must be suppressed on every chat surface."""
    assert _prepare_gateway_status_message(platform, "warn", message) is None


@pytest.mark.parametrize("platform", ["whatsapp", "slack", "signal", "matrix"])
def test_chat_gateways_redact_secret_in_provider_error(platform):
    """Provider-error bodies carrying secrets must never reach chat users.

    THE security invariant being widened from Telegram (#28533) to all chat
    surfaces (#39293): a leaked bearer token in a provider error body must be
    redacted/replaced before delivery on any chat platform.
    """
    raw = (
        "API call failed after 3 retries: HTTP 401 Unauthorized — "
        "Authorization: Bearer sk-ABCDEF0123456789abcdef0123"
    )

    sanitized = _sanitize_gateway_final_response(platform, raw)

    assert "sk-ABCDEF0123456789abcdef0123" not in sanitized
    assert "sk-ABCDEF" not in sanitized
    assert "HTTP 401" not in sanitized
    # The user gets the safe provider-error category instead of the raw body.
    assert "provider" in sanitized.lower()


@pytest.mark.parametrize("platform", ["whatsapp", "slack", "signal", "matrix"])
def test_chat_gateways_redact_secret_in_non_error_body(platform):
    """Secrets must be redacted even when no provider-error rewrite fires.

    The provider-error case above is rewritten wholesale to a generic
    category string, so it cannot, on its own, prove the secret-redaction
    layer works — the rewrite would strip the body regardless. This case
    feeds ordinary assistant prose that merely *echoes* a bearer token (not
    a provider-error envelope), so `_redact_gateway_user_facing_secrets` is
    the only thing standing between the token and the user. Removing the
    redaction patterns makes this fail (genuine regression guard); the
    surrounding prose must survive intact.
    """
    raw = (
        "Sure — here is the example request you asked for: "
        "curl -H 'Authorization: Bearer sk-ABCDEF0123456789abcdef0123' "
        "https://api.example.com/v1/models"
    )

    sanitized = _sanitize_gateway_final_response(platform, raw)

    assert "sk-ABCDEF0123456789abcdef0123" not in sanitized
    assert "sk-ABCDEF" not in sanitized
    # The secret body is gone — assert the invariant, not the specific mask
    # marker. The outbound redactor delegates to redact_sensitive_text (#23810),
    # which masks as `***`/partial; the local pattern fallback uses `[REDACTED]`.
    assert "***" in sanitized or "[REDACTED]" in sanitized
    # Non-secret prose is preserved — redaction is surgical, not a wholesale
    # rewrite, on bodies that are not provider-error envelopes.
    assert "here is the example request you asked for" in sanitized


def test_plugin_platform_string_suppresses_noise():
    """Unknown/plugin chat platforms fail closed to the chat-filter path."""
    message = "⏳ Retrying in 4.2s (attempt 1/3)..."

    assert _prepare_gateway_status_message("irc", "warn", message) is None


@pytest.mark.parametrize("platform", CHAT_PLATFORMS)
def test_chat_gateways_keep_normal_answers(platform):
    """Normal assistant content must pass through unchanged on chat surfaces."""
    answer = "Here is the clean summary you asked for."

    assert _sanitize_gateway_final_response(platform, answer) == answer


@pytest.mark.parametrize("platform", CHAT_PLATFORMS)
def test_chat_gateways_drop_interrupt_sentinel(platform):
    """The interrupt-while-waiting sentinel is metadata, not a reply (#7921)."""
    sentinel = "Operation interrupted: waiting for model response (1.7s elapsed)."

    assert _sanitize_gateway_final_response(platform, sentinel) == ""
    assert _sanitize_gateway_final_response("local", sentinel) == sentinel


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


# Synthetic credential shapes from #23810. Bodies are placeholder gibberish —
# never real tokens — but they match the canonical redaction patterns. The
# outbound gateway redactor previously used a narrow local pattern subset that
# leaked the GitHub fine-grained PAT and Telegram bot-token shapes; it now
# delegates to agent.redact.redact_sensitive_text, the authoritative redactor
# already used for logs/tool-output/approval prompts.
_ISSUE_23810_SECRET_SHAPES = {
    "openai_sk": "sk-" + "a1b2c3d4e5f6a7b8c9d0",
    "github_fine_grained_pat": "github_pat_" + "1A" * 41,
    "github_classic_pat": "ghp_" + "Ab3Cd4Ef5Gh6Ij7Kl8Mn9Op0Qr1St2Uv3Wx",
    "telegram_bot_token": "bot1234567890:" + "AAH" * 13 + "x",
    "openrouter_v1": "sk-or-v1-" + "Z9" * 36 + "q",
}


@pytest.mark.parametrize("platform", CHAT_PLATFORMS)
@pytest.mark.parametrize("shape_name", sorted(_ISSUE_23810_SECRET_SHAPES))
def test_chat_gateways_redact_all_issue_23810_credential_shapes(platform, shape_name):
    """Outbound chat must mask every credential shape the banner promises.

    Regression guard for #23810: the gateway claimed "chat responses are
    scrubbed before delivery", but the outbound redactor used a divergent
    narrow pattern set that leaked the GitHub fine-grained PAT and Telegram
    bot-token shapes verbatim. Feed each shape as ordinary assistant prose
    (not a provider-error envelope, so no wholesale rewrite fires) and assert
    the secret body never reaches the user while surrounding prose survives.
    """
    secret = _ISSUE_23810_SECRET_SHAPES[shape_name]
    raw = f"Sure, here is the token you asked me to echo: {secret} — done."

    sanitized = _sanitize_gateway_final_response(platform, raw)

    assert secret not in sanitized, f"{shape_name} leaked verbatim on {platform}"
    # Prose around the secret is preserved — redaction is surgical.
    assert "here is the token you asked me to echo" in sanitized
    assert sanitized.endswith("done.")


# ---------------------------------------------------------------------------
# Operator-configured outbound suppression (suppress_outbound)
# ---------------------------------------------------------------------------


@pytest.fixture
def suppress_config(monkeypatch):
    """Install a GatewayConfig into the suppress_outbound resolution path.

    Returns a setter: call it with (global_patterns, per_platform) to make the
    gateway resolve those patterns. Clears the mtime-keyed config cache and
    the compiled-pattern cache before and after each test so tests cannot
    leak into each other (or into the unrelated tests above).
    """
    import gateway.run as run

    def _clear():
        run._SUPPRESS_OUTBOUND_CFG_CACHE.clear()
        run._SUPPRESS_OUTBOUND_COMPILED.clear()

    def _set(global_patterns=None, per_platform=None):
        platforms = {}
        for platform, patterns in (per_platform or {}).items():
            platforms[platform] = PlatformConfig(
                enabled=True, extra={"suppress_outbound": list(patterns)}
            )
        cfg = GatewayConfig(
            platforms=platforms,
            suppress_outbound=list(global_patterns or []),
        )
        _clear()
        monkeypatch.setattr(run, "load_gateway_config", lambda: cfg)
        return cfg

    _clear()
    yield _set
    _clear()


def test_suppress_outbound_drops_matching_final_response(suppress_config):
    """A configured pattern drops the final reply on a chat surface."""
    suppress_config(global_patterns=[r"^Liked it\.$"])

    assert _sanitize_gateway_final_response(Platform.TELEGRAM, "Liked it.") == ""
    # re.search semantics: unanchored patterns match anywhere.
    suppress_config(global_patterns=[r"Interrupting current task"])
    assert (
        _sanitize_gateway_final_response(
            Platform.TELEGRAM, "Interrupting current task to handle your message..."
        )
        == ""
    )


def test_suppress_outbound_drops_matching_status_message(suppress_config):
    """The same patterns cover the status/notice send path."""
    suppress_config(global_patterns=[r"Interrupting current task"])

    assert (
        _prepare_gateway_status_message(
            Platform.TELEGRAM, "lifecycle", "Interrupting current task..."
        )
        is None
    )


def test_suppress_outbound_exempts_raw_platforms(suppress_config):
    """Programmatic surfaces must never be muted by operator patterns."""
    suppress_config(global_patterns=[r".*"])  # suppress everything

    text = "Liked it."
    for platform in ("local", "api_server", "webhook", "msgraph_webhook"):
        assert _sanitize_gateway_final_response(platform, text) == text
        assert _prepare_gateway_status_message(platform, "warn", text) == text


def test_suppress_outbound_per_platform_extends_global(suppress_config):
    """platforms.<name>.suppress_outbound extends (not replaces) the global list."""
    suppress_config(
        global_patterns=[r"^Liked it\.$"],
        per_platform={Platform.TELEGRAM: [r"^Gateway restarted"]},
    )

    # Telegram gets global + its own pattern.
    assert _sanitize_gateway_final_response(Platform.TELEGRAM, "Liked it.") == ""
    assert (
        _sanitize_gateway_final_response(Platform.TELEGRAM, "Gateway restarted (v2)")
        == ""
    )
    # Other chat platforms only get the global pattern.
    assert _sanitize_gateway_final_response(Platform.DISCORD, "Liked it.") == ""
    assert (
        _sanitize_gateway_final_response(Platform.DISCORD, "Gateway restarted (v2)")
        == "Gateway restarted (v2)"
    )


def test_suppress_outbound_invalid_regex_warns_and_skips(suppress_config, caplog):
    """An invalid regex warns once, is skipped, and never crashes or over-drops."""
    import logging

    suppress_config(global_patterns=[r"[unclosed", r"^Liked it\.$"])

    with caplog.at_level(logging.WARNING):
        # Valid pattern still enforced despite the broken sibling.
        assert _sanitize_gateway_final_response(Platform.TELEGRAM, "Liked it.") == ""
        # Non-matching text passes through untouched.
        answer = "Here is the clean summary you asked for."
        assert _sanitize_gateway_final_response(Platform.TELEGRAM, answer) == answer

    assert any(
        "invalid suppress_outbound pattern" in record.getMessage()
        and "[unclosed" in record.getMessage()
        for record in caplog.records
    )


def test_suppress_outbound_empty_config_is_passthrough(suppress_config):
    """No configured patterns = zero behavior change."""
    suppress_config(global_patterns=[])

    answer = "Liked it."
    assert _sanitize_gateway_final_response(Platform.TELEGRAM, answer) == answer
    assert (
        _prepare_gateway_status_message(Platform.TELEGRAM, "info", answer) == answer
    )


def test_suppress_outbound_non_matching_text_untouched(suppress_config):
    """Patterns only drop matches; everything else flows through unchanged."""
    suppress_config(global_patterns=[r"^Liked it\.$"])

    answer = "I liked it. Here is the longer review you asked for."
    assert _sanitize_gateway_final_response(Platform.TELEGRAM, answer) == answer


def test_suppress_outbound_case_sensitive_as_written(suppress_config):
    """Patterns compile as written; operators opt into (?i) themselves."""
    suppress_config(global_patterns=[r"^liked it\.$"])
    assert _sanitize_gateway_final_response(Platform.TELEGRAM, "Liked it.") == "Liked it."

    suppress_config(global_patterns=[r"(?i)^liked it\.$"])
    assert _sanitize_gateway_final_response(Platform.TELEGRAM, "Liked it.") == ""


def test_get_suppress_outbound_resolution_order():
    """GatewayConfig.get_suppress_outbound: global first, then platform, deduped."""
    cfg = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                enabled=True,
                extra={"suppress_outbound": [r"^B$", r"^A$"]},
            )
        },
        suppress_outbound=[r"^A$"],
    )

    assert cfg.get_suppress_outbound(Platform.TELEGRAM) == [r"^A$", r"^B$"]
    assert cfg.get_suppress_outbound(Platform.DISCORD) == [r"^A$"]
    assert cfg.get_suppress_outbound(None) == [r"^A$"]
