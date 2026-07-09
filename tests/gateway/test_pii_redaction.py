"""Tests for PII redaction in gateway session context prompts."""

from gateway.session import (
    SessionContext,
    SessionSource,
    build_session_context_prompt,
    _hash_id,
    _hash_sender_id,
    _hash_chat_id,
)
from gateway.config import Platform, HomeChannel


PROVIDER_REPORT_FIXTURES = {
    "set_cookie": "Set-Cookie: CF_Authorization=fixture_gateway_cookie; Path=/; Secure; HttpOnly",
    "cookie": "Cookie: CF_Authorization=fixture_gateway_cookie; session=fixture_gateway_session",
    "authorization": "Authorization: Bearer fixture_gateway_authorization",
    "signed_redirect": (
        "https://access.example.invalid/cdn-cgi/access/login/github?"
        "redirect_url=https%3A%2F%2Fcrm.example.invalid%2Fdashboard"
        "&sig=fixture_gateway_signature"
    ),
    "magic_link": "https://crm.example.invalid/login?magic_link_token=fixture_gateway_magic_link",
    "jwt_meta": "meta=eyJmaXh0dXJlIjoiZ2F0ZXdheSJ9.eyJzdWIiOiJmaXh0dXJlIn0.signatureFixture",
    "provider_redirect": (
        "https://provider.example.invalid/oauth?"
        "private_redirect_url=https%3A%2F%2Fcrm.internal.invalid%2Fadmin"
        "&provider_token=fixture_gateway_provider_token"
    ),
}


def _assert_provider_report_value_free(text: str) -> None:
    for label, fixture in PROVIDER_REPORT_FIXTURES.items():
        if fixture in text:
            raise AssertionError(f"unredacted provider report fixture leaked: {label}")
    for fragment in (
        "fixture_gateway_cookie",
        "fixture_gateway_session",
        "fixture_gateway_authorization",
        "fixture_gateway_signature",
        "fixture_gateway_magic_link",
        "fixture_gateway_provider_token",
        "crm.internal.invalid",
    ):
        if fragment in text:
            raise AssertionError("unredacted provider report fixture fragment leaked")


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

class TestHashHelpers:
    def test_hash_id_deterministic(self):
        assert _hash_id("12345") == _hash_id("12345")

    def test_hash_id_12_hex_chars(self):
        h = _hash_id("user-abc")
        assert len(h) == 12
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_sender_id_prefix(self):
        assert _hash_sender_id("12345").startswith("user_")
        assert len(_hash_sender_id("12345")) == 17  # "user_" + 12

    def test_hash_chat_id_preserves_prefix(self):
        result = _hash_chat_id("telegram:12345")
        assert result.startswith("telegram:")
        assert "12345" not in result

    def test_hash_chat_id_no_prefix(self):
        result = _hash_chat_id("12345")
        assert len(result) == 12
        assert "12345" not in result


# ---------------------------------------------------------------------------
# Integration: build_session_context_prompt
# ---------------------------------------------------------------------------

def _make_context(
    user_id="user-123",
    user_name=None,
    chat_id="telegram:99999",
    platform=Platform.TELEGRAM,
    home_channels=None,
):
    source = SessionSource(
        platform=platform,
        chat_id=chat_id,
        chat_type="dm",
        user_id=user_id,
        user_name=user_name,
    )
    return SessionContext(
        source=source,
        connected_platforms=[platform],
        home_channels=home_channels or {},
    )


class TestBuildSessionContextPromptRedaction:
    def test_no_redaction_by_default(self):
        ctx = _make_context(user_id="user-123")
        prompt = build_session_context_prompt(ctx)
        assert "user-123" in prompt

    def test_user_id_hashed_when_redact_pii(self):
        ctx = _make_context(user_id="user-123")
        prompt = build_session_context_prompt(ctx, redact_pii=True)
        assert "user-123" not in prompt
        assert "user_" in prompt  # hashed ID present

    def test_user_name_not_redacted(self):
        ctx = _make_context(user_id="user-123", user_name="Alice")
        prompt = build_session_context_prompt(ctx, redact_pii=True)
        assert "Alice" in prompt
        # user_id should not appear when user_name is present (name takes priority)
        assert "user-123" not in prompt

    def test_home_channel_id_hashed(self):
        hc = {
            Platform.TELEGRAM: HomeChannel(
                platform=Platform.TELEGRAM,
                chat_id="telegram:99999",
                name="Home Chat",
            )
        }
        ctx = _make_context(home_channels=hc)
        prompt = build_session_context_prompt(ctx, redact_pii=True)
        assert "99999" not in prompt
        assert "telegram:" in prompt  # prefix preserved
        assert "Home Chat" in prompt  # name not redacted

    def test_home_channel_id_preserved_without_redaction(self):
        hc = {
            Platform.TELEGRAM: HomeChannel(
                platform=Platform.TELEGRAM,
                chat_id="telegram:99999",
                name="Home Chat",
            )
        }
        ctx = _make_context(home_channels=hc)
        prompt = build_session_context_prompt(ctx, redact_pii=False)
        assert "99999" in prompt

    def test_redaction_is_deterministic(self):
        ctx = _make_context(user_id="+15551234567")
        prompt1 = build_session_context_prompt(ctx, redact_pii=True)
        prompt2 = build_session_context_prompt(ctx, redact_pii=True)
        assert prompt1 == prompt2

    def test_different_ids_produce_different_hashes(self):
        ctx1 = _make_context(user_id="user-A")
        ctx2 = _make_context(user_id="user-B")
        p1 = build_session_context_prompt(ctx1, redact_pii=True)
        p2 = build_session_context_prompt(ctx2, redact_pii=True)
        assert p1 != p2

    def test_discord_ids_not_redacted_even_with_flag(self):
        """Discord needs real IDs for <@user_id> mentions."""
        ctx = _make_context(user_id="123456789", platform=Platform.DISCORD)
        prompt = build_session_context_prompt(ctx, redact_pii=True)
        assert "123456789" in prompt

    def test_whatsapp_ids_redacted(self):
        ctx = _make_context(user_id="+15551234567", platform=Platform.WHATSAPP)
        prompt = build_session_context_prompt(ctx, redact_pii=True)
        assert "+15551234567" not in prompt
        assert "user_" in prompt

    def test_signal_ids_redacted(self):
        ctx = _make_context(user_id="+15551234567", platform=Platform.SIGNAL)
        prompt = build_session_context_prompt(ctx, redact_pii=True)
        assert "+15551234567" not in prompt
        assert "user_" in prompt

    def test_slack_ids_not_redacted(self):
        """Slack may need IDs for mentions too."""
        ctx = _make_context(user_id="U12345ABC", platform=Platform.SLACK)
        prompt = build_session_context_prompt(ctx, redact_pii=True)
        assert "U12345ABC" in prompt


def test_gateway_final_report_redacts_provider_private_material():
    from agent.redact import redact_sensitive_text

    raw_report = "\n".join(PROVIDER_REPORT_FIXTURES.values())
    redacted = redact_sensitive_text(raw_report, force=True)

    _assert_provider_report_value_free(redacted)
