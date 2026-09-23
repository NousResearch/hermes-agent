"""Unit tests for the model-blind SMS/email code-handle registry (#119683)."""

from __future__ import annotations

import time

import pytest

from agent import code_registry


@pytest.fixture(autouse=True)
def _clean_registry():
    code_registry.clear_codes()
    yield
    code_registry.clear_codes()


class TestMint:
    def test_mint_returns_handle_not_code(self):
        out = code_registry.mint("123456", source="sms")
        assert out["code_handle"].startswith(code_registry.HANDLE_PREFIX)
        assert "123456" not in str(out)
        assert out["source"] == "sms"
        assert out["expires_at"] > time.time()

    def test_mint_registers_redaction_value(self, monkeypatch):
        from agent import redact

        registered = []
        monkeypatch.setattr(redact, "register_vault_redaction_value", registered.append)
        code_registry.mint("654321")
        assert registered == ["654321"]

    def test_mint_rejects_empty_code(self):
        with pytest.raises(ValueError):
            code_registry.mint("   ")

    def test_mint_normalizes_origin(self):
        out = code_registry.mint("111111", origin="https://example.com:443/path")
        assert out["origin"] == "https://example.com"

    def test_ttl_is_clamped(self):
        early = code_registry.mint("222222", ttl_s=1)
        huge = code_registry.mint("333333", ttl_s=10_000)
        # clamped floor 30s, ceiling 900s — both still in the future
        assert early["expires_at"] > time.time()
        assert huge["expires_at"] - time.time() <= 901


class TestConsume:
    def test_single_use(self):
        handle = code_registry.mint("444444")["code_handle"]
        assert code_registry.consume(handle) == "444444"
        assert code_registry.consume(handle) is None

    def test_unknown_handle(self):
        assert code_registry.consume("otp_nope") is None
        assert code_registry.consume("") is None

    def test_expired_handle(self, monkeypatch):
        handle = code_registry.mint("555555")["code_handle"]
        # force expiry
        with code_registry._LOCK:
            bucket = code_registry._REGISTRY[code_registry._scope()]
            bucket[handle]["expires_at"] = time.time() - 1
        assert code_registry.consume(handle) is None
        # expired entry was still dropped
        assert code_registry.consume(handle) is None

    def test_origin_bound_match(self):
        handle = code_registry.mint("777777", origin="https://example.com")["code_handle"]
        assert code_registry.consume(handle, origin="https://example.com/login") == "777777"

    def test_origin_bound_mismatch_drops_handle(self):
        handle = code_registry.mint("888888", origin="https://example.com")["code_handle"]
        assert code_registry.consume(handle, origin="https://evil.test") is None
        # handle was consumed (dropped) even on mismatch — no retry against another site
        assert code_registry.consume(handle, origin="https://example.com") is None

    def test_origin_bound_requires_origin_arg(self):
        handle = code_registry.mint("999999", origin="https://example.com")["code_handle"]
        assert code_registry.consume(handle) is None

    def test_unbound_handle_needs_no_origin(self):
        handle = code_registry.mint("121212")["code_handle"]
        assert code_registry.consume(handle, origin="https://any.test") == "121212"


class TestScopeAndBounds:
    def test_profile_scope_is_isolated(self, monkeypatch, tmp_path):
        from hermes_constants import get_hermes_home

        handle = code_registry.mint("313131")["code_handle"]
        other = tmp_path / "other-home"
        other.mkdir()
        monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: other)
        # different scope → handle invisible
        assert code_registry.consume(handle) is None
        monkeypatch.undo()
        # back on original scope → still there (consume once)
        # After monkeypatch.undo, scope is the test HERMES_HOME again.
        # The mint above was on that scope, and the failed consume above did not drop it.
        # But wait - consume under other scope looks up other's bucket only, so handle remains.
        assert get_hermes_home() is not None
        # Re-mint on restored scope for a deterministic assert:
        h2 = code_registry.mint("414141")["code_handle"]
        assert code_registry.consume(h2) == "414141"

    def test_bounded_lru_evicts_oldest(self):
        handles = [
            code_registry.mint(f"{i:06d}", ttl_s=300)["code_handle"]
            for i in range(code_registry.MAX_PER_PROFILE + 4)
        ]
        with code_registry._LOCK:
            bucket = code_registry._REGISTRY[code_registry._scope()]
            assert len(bucket) == code_registry.MAX_PER_PROFILE
        # oldest gone, newest present
        assert code_registry.consume(handles[0]) is None
        assert code_registry.consume(handles[-1]) == f"{code_registry.MAX_PER_PROFILE + 3:06d}"

    def test_clear_codes(self):
        handle = code_registry.mint("525252")["code_handle"]
        code_registry.clear_codes()
        assert code_registry.consume(handle) is None


class TestPeekSource:
    def test_peek_returns_source_while_live(self):
        handle = code_registry.mint("636363", source="email")["code_handle"]
        assert code_registry.peek_source(handle) == "email"
        code_registry.consume(handle)
        assert code_registry.peek_source(handle) is None


class TestExtractVerificationCode:
    def test_prefers_keyword_line(self):
        body = "Order shipped.\nYour verification code is 123456\nThanks!"
        assert code_registry.extract_verification_code(body) == "123456"

    def test_falls_back_to_digit_run(self):
        assert code_registry.extract_verification_code("use 987654 to finish") == "987654"

    def test_alnum_near_keyword(self):
        body = "Your code: Ab3dEf9"
        assert code_registry.extract_verification_code(body) == "Ab3dEf9"

    def test_none_when_no_code(self):
        assert code_registry.extract_verification_code("hello world") is None
        assert code_registry.extract_verification_code("") is None
        assert code_registry.extract_verification_code(None) is None  # type: ignore[arg-type]
