"""Tests for the WhatsApp Cloud API setup wizard.

Covers:
- Field-shape validators (catch the #1 setup mistake — phone number in
  the Phone Number ID field — plus the OpenAI / Slack / GitHub token
  paste-by-mistake cases)
- Wizard end-to-end flow with mocked stdin/stdout — verifies each step
  writes the expected env var, validation errors block invalid input,
  optional fields can be skipped, and the SETUP COMPLETE block prints
  the post-setup tunnel + Meta-dashboard instructions the user needs
  (the wizard can't smoke-test reachability itself because the gateway
  isn't running yet during setup).
"""

from __future__ import annotations

import io
import os
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from hermes_cli.setup_whatsapp_cloud import (
    _validate_phone_number_id,
    _validate_waba_id,
    _validate_app_id,
    _validate_app_secret,
    _validate_access_token,
    run_whatsapp_cloud_setup,
)


# ---------------------------------------------------------------------------
# Validator tests — the cheap, exhaustive coverage layer
# ---------------------------------------------------------------------------


class TestPhoneNumberIdValidator:
    def test_accepts_real_meta_phone_number_id(self):
        ok, _ = _validate_phone_number_id("7794189252778687")
        assert ok

    def test_rejects_actual_phone_number_with_helpful_message(self):
        """The #1 setup trap — pasting the phone number instead of the ID."""
        ok, reason = _validate_phone_number_id("15556422442")
        assert not ok
        assert "phone number" in reason.lower()
        assert "Phone number ID" in reason  # tells them where to look

    def test_rejects_phone_number_with_plus(self):
        ok, reason = _validate_phone_number_id("+15556422442")
        assert not ok
        assert "numeric" in reason.lower() or "phone number" in reason.lower()

    def test_rejects_empty(self):
        ok, reason = _validate_phone_number_id("")
        assert not ok
        assert "required" in reason.lower()

    def test_rejects_too_short(self):
        ok, _ = _validate_phone_number_id("12345")
        assert not ok

    def test_rejects_too_long(self):
        ok, _ = _validate_phone_number_id("1" * 25)
        assert not ok

    def test_strips_surrounding_whitespace(self):
        ok, _ = _validate_phone_number_id("  7794189252778687  ")
        assert ok


class TestAccessTokenValidator:
    def test_accepts_eaa_token(self):
        ok, _ = _validate_access_token("EAA" + "a" * 100)
        assert ok

    def test_rejects_empty(self):
        ok, reason = _validate_access_token("")
        assert not ok
        assert "required" in reason.lower()

    def test_rejects_openai_key_with_helpful_message(self):
        ok, reason = _validate_access_token("sk-proj-" + "a" * 100)
        assert not ok
        assert "OpenAI" in reason

    def test_rejects_slack_token_with_helpful_message(self):
        ok, reason = _validate_access_token("xoxb-1234-5678-abcdef")
        assert not ok
        assert "Slack" in reason

    def test_rejects_github_token_with_helpful_message(self):
        ok, reason = _validate_access_token("ghp_abcdefghijklmnop")
        assert not ok
        assert "GitHub" in reason

    def test_rejects_garbage_with_helpful_message(self):
        ok, reason = _validate_access_token("random-string-here")
        assert not ok
        assert "EAA" in reason  # tells them what to look for

    def test_rejects_short_token(self):
        ok, reason = _validate_access_token("EAAabc")
        assert not ok
        assert "short" in reason.lower()


class TestAppSecretValidator:
    def test_accepts_32_hex_chars(self):
        ok, _ = _validate_app_secret("0123456789abcdef0123456789abcdef")
        assert ok

    def test_accepts_uppercase_hex(self):
        ok, _ = _validate_app_secret("0123456789ABCDEF0123456789ABCDEF")
        assert ok

    def test_rejects_wrong_length(self):
        ok, reason = _validate_app_secret("0123456789abcdef")  # 16 chars
        assert not ok
        assert "32" in reason

    def test_rejects_non_hex(self):
        ok, reason = _validate_app_secret("zzzz56789abcdef0123456789abcdezz")
        assert not ok
        assert "hex" in reason.lower()

    def test_rejects_empty(self):
        ok, _ = _validate_app_secret("")
        assert not ok


class TestAppIdValidator:
    def test_accepts_valid(self):
        ok, _ = _validate_app_id("1234567890123456")
        assert ok

    def test_rejects_non_numeric(self):
        ok, _ = _validate_app_id("abcdef")
        assert not ok

    def test_rejects_too_short(self):
        ok, _ = _validate_app_id("123")
        assert not ok


class TestWabaIdValidator:
    def test_accepts_valid(self):
        ok, _ = _validate_waba_id("215589313241560883")
        assert ok

    def test_rejects_non_numeric(self):
        ok, _ = _validate_waba_id("abc-def")
        assert not ok


# ---------------------------------------------------------------------------
# End-to-end wizard flow
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Redirect HERMES_HOME so save_env_value writes into a temp .env."""
    home = tmp_path / "home"
    hermes = home / ".hermes"
    hermes.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("HERMES_HOME", str(hermes))
    for key in list(os.environ):
        if key.startswith("WHATSAPP_CLOUD_"):
            monkeypatch.delenv(key, raising=False)
    return hermes


def _env_value(hermes_home: Path, key: str) -> str | None:
    env_file = hermes_home / ".env"
    if not env_file.exists():
        return None
    for line in env_file.read_text().splitlines():
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() == key:
            return v.strip().strip('"').strip("'")
    return None


class TestWizardFlow:
    def test_happy_path_minimal(self, isolated_home, monkeypatch):
        """Provide only the required fields; skip optional steps."""
        inputs = iter([
            "",                                       # press Enter to continue
            "7794189252778687",                       # Phone Number ID
            "EAA" + "x" * 200,                        # Access Token
            "0123456789abcdef0123456789abcdef",       # App Secret
            "",                                       # App ID — skip
            "",                                       # WABA ID — skip
            "15551234567",                            # Allowed users
        ])
        monkeypatch.setattr("builtins.input", lambda *a, **kw: next(inputs))
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = run_whatsapp_cloud_setup()
        assert rc == 0
        out = buf.getvalue()
        assert "SETUP COMPLETE" in out
        # Required fields written
        assert _env_value(isolated_home, "WHATSAPP_CLOUD_PHONE_NUMBER_ID") == "7794189252778687"
        assert _env_value(isolated_home, "WHATSAPP_CLOUD_ACCESS_TOKEN").startswith("EAA")
        assert len(_env_value(isolated_home, "WHATSAPP_CLOUD_APP_SECRET")) == 32
        assert _env_value(isolated_home, "WHATSAPP_CLOUD_ALLOWED_USERS") == "15551234567"
        # Verify token auto-generated
        assert _env_value(isolated_home, "WHATSAPP_CLOUD_VERIFY_TOKEN")
        # Optional fields stayed unset
        assert _env_value(isolated_home, "WHATSAPP_CLOUD_APP_ID") is None
        assert _env_value(isolated_home, "WHATSAPP_CLOUD_WABA_ID") is None

    def test_phone_number_id_validator_catches_phone_number(self, isolated_home, monkeypatch):
        """The trap test — user pastes their phone number into the
        Phone Number ID field. Wizard MUST reject with a helpful
        explanation, not pass through."""
        inputs = iter([
            "",                                       # press Enter to continue
            "15556422442",                            # phone number — rejected
            "",                                       # empty — gives up
        ])
        monkeypatch.setattr("builtins.input", lambda *a, **kw: next(inputs))
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = run_whatsapp_cloud_setup()
        assert rc == 1
        out = buf.getvalue()
        # Must surface the specific guidance about Phone Number ID
        assert "Phone number ID" in out
        assert "15-17 digits" in out
        # Should NOT have saved the bad value
        assert _env_value(isolated_home, "WHATSAPP_CLOUD_PHONE_NUMBER_ID") is None

    def test_access_token_validator_catches_openai_key(self, isolated_home, monkeypatch):
        """User pastes 'sk-proj-...' by mistake. Wizard rejects."""
        inputs = iter([
            "",                                       # continue
            "7794189252778687",                       # good Phone ID
            "sk-proj-" + "x" * 100,                   # OpenAI key — rejected
            "",                                       # give up
        ])
        monkeypatch.setattr("builtins.input", lambda *a, **kw: next(inputs))
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = run_whatsapp_cloud_setup()
        assert rc == 1
        out = buf.getvalue()
        assert "OpenAI" in out  # diagnostic in error message
        # Phone Number ID was saved (it was valid), but access token was not
        assert _env_value(isolated_home, "WHATSAPP_CLOUD_PHONE_NUMBER_ID") == "7794189252778687"
        assert _env_value(isolated_home, "WHATSAPP_CLOUD_ACCESS_TOKEN") is None

    def test_verify_token_is_auto_generated(self, isolated_home, monkeypatch):
        """The verify token is one of the few things the user shouldn't
        have to invent. Wizard generates a strong random one."""
        inputs = iter([
            "",                                       # continue
            "7794189252778687",                       # Phone ID
            "EAA" + "x" * 200,                        # Token
            "0123456789abcdef0123456789abcdef",       # App Secret
            "",                                       # App ID — skip
            "",                                       # WABA ID — skip
            "15551234567",                            # Allowed users
        ])
        monkeypatch.setattr("builtins.input", lambda *a, **kw: next(inputs))
        buf = io.StringIO()
        with redirect_stdout(buf):
            run_whatsapp_cloud_setup()
        verify_token = _env_value(isolated_home, "WHATSAPP_CLOUD_VERIFY_TOKEN")
        assert verify_token is not None
        # secrets.token_urlsafe(32) produces ~43 chars (base64-of-32-bytes)
        assert len(verify_token) >= 32
        # Should also be echoed to user output so they can paste into Meta
        assert verify_token in buf.getvalue()

    def test_setup_complete_block_includes_post_setup_instructions(self, isolated_home, monkeypatch):
        """The wizard can't smoke-test the webhook itself (the gateway
        isn't running yet), so it MUST print the exact curl/cloudflared
        steps the user needs after the wizard exits."""
        inputs = iter([
            "",                                       # continue
            "7794189252778687",                       # Phone ID
            "EAA" + "x" * 200,                        # Token
            "0123456789abcdef0123456789abcdef",       # App Secret
            "",                                       # App ID — skip
            "",                                       # WABA ID — skip
            "15551234567",                            # Allowed users
        ])
        monkeypatch.setattr("builtins.input", lambda *a, **kw: next(inputs))
        buf = io.StringIO()
        with redirect_stdout(buf):
            run_whatsapp_cloud_setup()
        out = buf.getvalue()
        # Required post-setup guidance
        assert "cloudflared tunnel --url http://localhost:8090" in out
        assert "hermes gateway" in out
        assert "Verify and save" in out
        assert "messages" in out
        # The verify token should be quotable on the curl line
        verify_token = _env_value(isolated_home, "WHATSAPP_CLOUD_VERIFY_TOKEN")
        assert verify_token in out

    def test_existing_token_preserved_on_rerun(self, isolated_home, monkeypatch):
        """Re-running the wizard with existing config should let the
        user keep current values by hitting Enter."""
        # Pre-populate .env as if a previous run succeeded
        env_file = isolated_home / ".env"
        env_file.write_text(
            "WHATSAPP_CLOUD_PHONE_NUMBER_ID=7794189252778687\n"
            "WHATSAPP_CLOUD_ACCESS_TOKEN=EAAprevious_token_here_" + "x" * 100 + "\n"
            "WHATSAPP_CLOUD_APP_SECRET=0123456789abcdef0123456789abcdef\n"
            "WHATSAPP_CLOUD_VERIFY_TOKEN=existing_verify_token_already_set\n"
        )
        inputs = iter([
            "",                                       # continue
            "",                                       # Phone ID — keep existing
            "",                                       # Token — keep existing
            "",                                       # App Secret — keep existing
            "",                                       # App ID — skip
            "",                                       # WABA ID — skip
            "",                                       # verify token: regenerate? [y/N] — no
            "",                                       # Allowed users — keep
        ])
        monkeypatch.setattr("builtins.input", lambda *a, **kw: next(inputs))
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = run_whatsapp_cloud_setup()
        assert rc == 0
        # Values preserved
        token = _env_value(isolated_home, "WHATSAPP_CLOUD_ACCESS_TOKEN")
        assert token is not None
        assert token.startswith("EAAprevious_token_here_")
        # Verify token preserved (user said no to regenerate)
        assert _env_value(isolated_home, "WHATSAPP_CLOUD_VERIFY_TOKEN") == "existing_verify_token_already_set"


# =========================================================================
# Profile polish block (SETUP COMPLETE → optional WhatsApp profile setup)
# =========================================================================


class TestProfilePolishGuidance:
    """The wizard can't set the bot's WhatsApp display name or profile
    picture via the API — those go through Meta's Business Manager UI.
    Verify that the SETUP COMPLETE block points the user at the right
    place rather than leaving them to figure it out on their own."""

    def test_polish_block_present_and_points_at_business_manager(
        self, isolated_home, monkeypatch
    ):
        inputs = iter([
            "",
            "7794189252778687",
            "EAA" + "x" * 200,
            "0123456789abcdef0123456789abcdef",
            "",                          # App ID — skip
            "",                          # WABA ID — skip
            "15551234567",
        ])
        monkeypatch.setattr("builtins.input", lambda *a, **kw: next(inputs))
        buf = io.StringIO()
        with redirect_stdout(buf):
            run_whatsapp_cloud_setup()
        out = buf.getvalue()
        # Polish block header
        assert "polish your bot's WhatsApp profile" in out
        # Direct user at Meta's Business Manager (not the developer dash)
        assert "business.facebook.com/wa/manage/phone-numbers" in out
        # Mention each of the three things the user can do there
        assert "Display name" in out
        assert "profile picture" in out
        assert "Edit profile" in out
        # Set expectations about display-name reviews
        assert "24-48h" in out or "24–48h" in out

    def test_polish_block_deeplinks_when_waba_id_known(
        self, isolated_home, monkeypatch
    ):
        """If the user gave us the WABA ID earlier in the wizard, the
        Business Manager URL should preselect their account."""
        waba = "987654321098765"
        inputs = iter([
            "",
            "7794189252778687",
            "EAA" + "x" * 200,
            "0123456789abcdef0123456789abcdef",
            "",                          # App ID — skip
            waba,                        # WABA ID — provided
            "15551234567",
        ])
        monkeypatch.setattr("builtins.input", lambda *a, **kw: next(inputs))
        buf = io.StringIO()
        with redirect_stdout(buf):
            run_whatsapp_cloud_setup()
        out = buf.getvalue()
        # Deep-linked URL with the user's WABA preselected
        assert f"waba_id={waba}" in out
        # Without WABA, we tell the user they'll need to pick their account
        assert "select your WhatsApp Business Account" not in out
