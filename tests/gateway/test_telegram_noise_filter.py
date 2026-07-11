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


def _clear_suppress_caches():
    import gateway.run as run

    run._SUPPRESS_OUTBOUND_CFG_CACHE.clear()
    run._SUPPRESS_OUTBOUND_COMPILED.clear()


@pytest.fixture
def suppress_config(monkeypatch, tmp_path):
    """Write a real config.yaml under a temp HERMES_HOME for suppress_outbound.

    Returns a setter: call it with (global_patterns, per_platform) to write
    the YAML and point config resolution at it. Every resolution then runs
    the real ``load_gateway_config()`` YAML bridges (top-level
    ``suppress_outbound`` plus ``platforms.<name>.suppress_outbound``) — no
    stubbing of the loader. Clears the config and compiled-pattern caches
    around each write so tests cannot leak into each other (or into the
    unrelated tests above).
    """
    import yaml

    import gateway.run as run

    home = tmp_path / "hermes-home"
    home.mkdir()
    # load_gateway_config() resolves via get_hermes_home() (env), while
    # gateway.run's path helpers use the module-level _hermes_home snapshot —
    # point both at the temp home so the cache key and the loaded file agree.
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(run, "_hermes_home", home)

    def _set(global_patterns=None, per_platform=None):
        cfg_doc = {}
        if global_patterns is not None:
            cfg_doc["suppress_outbound"] = list(global_patterns)
        if per_platform:
            cfg_doc["platforms"] = {
                platform.value: {"suppress_outbound": list(patterns)}
                for platform, patterns in per_platform.items()
            }
        (home / "config.yaml").write_text(
            yaml.safe_dump(cfg_doc), encoding="utf-8"
        )
        # Two writes can land within one mtime tick, so drop the resolved
        # config rather than trusting the stamp within a single test.
        _clear_suppress_caches()
        return home

    _clear_suppress_caches()
    yield _set
    _clear_suppress_caches()


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


def test_suppress_outbound_loaded_from_real_config_yaml(suppress_config):
    """load_gateway_config() bridges suppress_outbound from a real config.yaml.

    Exercises the actual YAML loader (temp HERMES_HOME) for both the global
    key and the platforms.<name>.suppress_outbound per-platform extension.
    """
    from gateway.config import load_gateway_config

    suppress_config(
        global_patterns=[r"^Liked it\.$"],
        per_platform={Platform.TELEGRAM: [r"^Gateway restarted"]},
    )

    cfg = load_gateway_config()
    assert cfg.suppress_outbound == [r"^Liked it\.$"]
    telegram_cfg = cfg.platforms.get(Platform.TELEGRAM)
    assert telegram_cfg is not None
    assert telegram_cfg.extra.get("suppress_outbound") == [r"^Gateway restarted"]
    assert cfg.get_suppress_outbound(Platform.TELEGRAM) == [
        r"^Liked it\.$",
        r"^Gateway restarted",
    ]
    assert cfg.get_suppress_outbound(Platform.DISCORD) == [r"^Liked it\.$"]


def test_suppress_outbound_routed_profiles_do_not_share_cache(tmp_path):
    """Context-local profile homes must never reuse each other's rules.

    Regression for the mtime-only cache key: two profile config files with
    identical mtimes are distinct cache entries because the resolved config
    path is part of the identity. The cache is deliberately NOT cleared
    between the profile switches below — that reuse is what's under test.
    """
    import os

    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    home_a.mkdir()
    home_b.mkdir()
    (home_a / "config.yaml").write_text(
        "suppress_outbound:\n  - '^From profile A$'\n", encoding="utf-8"
    )
    (home_b / "config.yaml").write_text(
        "suppress_outbound:\n  - '^From profile B$'\n", encoding="utf-8"
    )
    # Force identical mtimes so an mtime-only cache key would alias them.
    stat_a = (home_a / "config.yaml").stat()
    os.utime(home_b / "config.yaml", ns=(stat_a.st_atime_ns, stat_a.st_mtime_ns))

    _clear_suppress_caches()
    try:
        token = set_hermes_home_override(home_a)
        try:
            assert (
                _sanitize_gateway_final_response(Platform.TELEGRAM, "From profile A")
                == ""
            )
            assert (
                _sanitize_gateway_final_response(Platform.TELEGRAM, "From profile B")
                == "From profile B"
            )
        finally:
            reset_hermes_home_override(token)

        token = set_hermes_home_override(home_b)
        try:
            # With the old mtime-only key this resolved profile A's rules.
            assert (
                _sanitize_gateway_final_response(Platform.TELEGRAM, "From profile B")
                == ""
            )
            assert (
                _sanitize_gateway_final_response(Platform.TELEGRAM, "From profile A")
                == "From profile A"
            )
        finally:
            reset_hermes_home_override(token)
    finally:
        _clear_suppress_caches()


@pytest.mark.asyncio
async def test_suppress_outbound_covers_active_session_shutdown_notice(suppress_config):
    """The direct shutdown-notification send honors suppress_outbound."""
    from unittest.mock import MagicMock

    from gateway.session import build_session_key
    from tests.gateway.restart_test_helpers import (
        make_restart_runner,
        make_restart_source,
    )

    suppress_config(global_patterns=[r"Gateway (restarting|shutting down)"])

    runner, adapter = make_restart_runner()
    source = make_restart_source()
    session_key = build_session_key(source)
    runner._running_agents = {session_key: MagicMock()}
    runner._cache_session_source(session_key, source)

    await runner._notify_active_sessions_of_shutdown()
    assert adapter.sent_calls == []

    # Control: with no matching pattern the same rail delivers the notice.
    suppress_config(global_patterns=[])
    await runner._notify_active_sessions_of_shutdown()
    assert len(adapter.sent_calls) == 1
    assert "Gateway shutting down" in adapter.sent_calls[0][1]


@pytest.mark.asyncio
async def test_suppress_outbound_covers_home_channel_shutdown_broadcast(suppress_config):
    """The home-channel shutdown broadcast honors suppress_outbound too."""
    from gateway.config import HomeChannel
    from tests.gateway.restart_test_helpers import make_restart_runner

    suppress_config(global_patterns=[r"Gateway (restarting|shutting down)"])

    runner, adapter = make_restart_runner()
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM,
        chat_id="home-chat",
        name="Telegram Home",
    )

    await runner._notify_active_sessions_of_shutdown()
    assert adapter.sent_calls == []

    suppress_config(global_patterns=[])
    await runner._notify_active_sessions_of_shutdown()
    assert len(adapter.sent_calls) == 1
    assert adapter.sent_calls[0][0] == "home-chat"
    assert "Gateway shutting down" in adapter.sent_calls[0][1]
