"""Tests for agent.prompt_privacy — prompt-level PII anonymization."""
import pytest
from agent.prompt_privacy import PrivacyMiddleware, get_middleware, clear_middleware


class TestPrivacyMiddleware:
    def test_email_is_scrubbed_and_restored(self):
        mw = PrivacyMiddleware()
        s = mw.scrub("Contact: alice@example.com")
        assert "alice@example.com" not in s
        assert s.startswith("Contact: [alice@_")
        r = mw.restore(s)
        assert "alice@example.com" in r

    def test_github_token_is_scrubbed(self):
        mw = PrivacyMiddleware()
        token = "ghp_" + "a" * 36
        s = mw.scrub(f"GH: {token}")
        assert token not in s

    def test_sk_token_is_scrubbed(self):
        mw = PrivacyMiddleware()
        token = "sk-" + "p" * 24
        s = mw.scrub(f"Key: {token}")
        assert token not in s

    def test_ip_is_scrubbed(self):
        mw = PrivacyMiddleware()
        s = mw.scrub("Server: 192.168.1.100")
        assert "192.168.1.100" not in s

    def test_phone_is_scrubbed_and_restored(self):
        mw = PrivacyMiddleware()
        phone = "+336" + "12345678"
        s = mw.scrub(f"Tel: {phone}")
        assert phone not in s
        r = mw.restore(s)
        assert phone in r

    def test_bearer_token_is_scrubbed(self):
        mw = PrivacyMiddleware()
        s = mw.scrub("Auth: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9")
        assert "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9" not in s

    def test_url_with_secret_is_scrubbed(self):
        mw = PrivacyMiddleware()
        s = mw.scrub("URL: https://api.io?token=secret_k3y")
        assert "secret_k3y" not in s

    def test_placeholder_deterministic(self):
        mw = PrivacyMiddleware()
        email = "user@test.org"
        assert mw.scrub(email) == mw.scrub(email)

    def test_safe_text_passes_through(self):
        mw = PrivacyMiddleware()
        safe = "The quick brown fox."
        assert mw.scrub(safe) == safe
        assert mw.restore(safe) == safe

    def test_restore_idempotent(self):
        mw = PrivacyMiddleware()
        email = "idem@potent.com"
        s = mw.scrub(email)
        r1 = mw.restore(s)
        r2 = mw.restore(r1)
        assert r1 == r2


class TestGetMiddleware:
    def test_returns_none_when_no_config(self):
        clear_middleware()
        assert get_middleware() is None

    def test_returns_none_when_disabled(self):
        clear_middleware()
        fake = {"privacy": {"anonymize_prompts": False}}
        assert get_middleware(config=fake) is None

    def test_returns_middleware_when_enabled(self):
        clear_middleware()
        fake = {"privacy": {"anonymize_prompts": True}}
        assert get_middleware(config=fake) is not None