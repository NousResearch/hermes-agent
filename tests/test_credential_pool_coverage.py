"""Extended tests for credential_pool utility functions and CredentialPool methods.

Covers functions and methods that have no existing test coverage:
- label_from_token()
- _parse_absolute_timestamp()
- _extract_retry_delay_seconds()
- _normalize_error_context()
- _exhausted_until()
- _exhausted_ttl()
- _is_manual_source()
- _next_priority()
- CredentialPool.resolve_target()
- CredentialPool.add_entry()
- CredentialPool.remove_index()
- CredentialPool.peek()
- CredentialPool.reset_statuses()
- CredentialPool._entry_needs_refresh()
- get_pool_strategy()
"""

from __future__ import annotations

import json
import time

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_auth_store(tmp_path, payload: dict) -> None:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps(payload, indent=2))


def _make_pool(tmp_path, monkeypatch, entries, provider="anthropic", strategy="fill_first"):
    """Create a CredentialPool backed by a real auth.json on disk."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))

    _write_auth_store(
        tmp_path,
        {"version": 1, "credential_pool": {provider: entries}},
    )

    from agent.credential_pool import load_pool

    monkeypatch.setattr(
        "agent.credential_pool._load_config_safe",
        lambda: {"credential_pool_strategies": {provider: strategy}},
    )
    return load_pool(provider)


# ---- sample credential dicts reused across tests ----

_CRED_A = {
    "id": "aaa",
    "label": "primary",
    "auth_type": "api_key",
    "priority": 0,
    "source": "manual",
    "access_token": "sk-a",
}

_CRED_B = {
    "id": "bbb",
    "label": "secondary",
    "auth_type": "api_key",
    "priority": 1,
    "source": "manual",
    "access_token": "sk-b",
}

_CRED_C = {
    "id": "ccc",
    "label": "tertiary",
    "auth_type": "api_key",
    "priority": 2,
    "source": "manual",
    "access_token": "sk-c",
}


# ===================================================================
# _parse_absolute_timestamp
# ===================================================================


class TestParseAbsoluteTimestamp:
    def test_none_returns_none(self):
        from agent.credential_pool import _parse_absolute_timestamp

        assert _parse_absolute_timestamp(None) is None

    def test_empty_string_returns_none(self):
        from agent.credential_pool import _parse_absolute_timestamp

        assert _parse_absolute_timestamp("") is None

    def test_whitespace_string_returns_none(self):
        from agent.credential_pool import _parse_absolute_timestamp

        assert _parse_absolute_timestamp("   ") is None

    def test_zero_returns_none(self):
        from agent.credential_pool import _parse_absolute_timestamp

        assert _parse_absolute_timestamp(0) is None
        assert _parse_absolute_timestamp(0.0) is None

    def test_negative_returns_none(self):
        from agent.credential_pool import _parse_absolute_timestamp

        assert _parse_absolute_timestamp(-100) is None

    def test_epoch_seconds_int(self):
        from agent.credential_pool import _parse_absolute_timestamp

        ts = 1700000000
        assert _parse_absolute_timestamp(ts) == float(ts)

    def test_epoch_seconds_float(self):
        from agent.credential_pool import _parse_absolute_timestamp

        ts = 1700000000.5
        assert _parse_absolute_timestamp(ts) == ts

    def test_epoch_milliseconds_converted(self):
        from agent.credential_pool import _parse_absolute_timestamp

        ts_ms = 1700000000000
        result = _parse_absolute_timestamp(ts_ms)
        assert result == pytest.approx(1700000000.0, abs=0.01)

    def test_numeric_string_epoch_seconds(self):
        from agent.credential_pool import _parse_absolute_timestamp

        result = _parse_absolute_timestamp("1700000000")
        assert result == 1700000000.0

    def test_numeric_string_epoch_milliseconds(self):
        from agent.credential_pool import _parse_absolute_timestamp

        result = _parse_absolute_timestamp("1700000000000")
        assert result == pytest.approx(1700000000.0, abs=0.01)

    def test_iso8601_utc(self):
        from agent.credential_pool import _parse_absolute_timestamp

        result = _parse_absolute_timestamp("2024-01-01T00:00:00+00:00")
        assert isinstance(result, float)
        assert result > 0

    def test_iso8601_with_z_suffix(self):
        from agent.credential_pool import _parse_absolute_timestamp

        result = _parse_absolute_timestamp("2024-01-01T00:00:00Z")
        assert isinstance(result, float)
        assert result > 0

    def test_invalid_string_returns_none(self):
        from agent.credential_pool import _parse_absolute_timestamp

        assert _parse_absolute_timestamp("not-a-timestamp") is None

    def test_non_numeric_non_string_returns_none(self):
        from agent.credential_pool import _parse_absolute_timestamp

        assert _parse_absolute_timestamp([1, 2, 3]) is None
        assert _parse_absolute_timestamp({"ts": 123}) is None


# ===================================================================
# _extract_retry_delay_seconds
# ===================================================================


class TestExtractRetryDelaySeconds:
    def test_none_returns_none(self):
        from agent.credential_pool import _extract_retry_delay_seconds

        assert _extract_retry_delay_seconds(None) is None

    def test_empty_string_returns_none(self):
        from agent.credential_pool import _extract_retry_delay_seconds

        assert _extract_retry_delay_seconds("") is None

    def test_no_delay_info_returns_none(self):
        from agent.credential_pool import _extract_retry_delay_seconds

        assert _extract_retry_delay_seconds("some random error message") is None

    def test_quota_reset_delay_milliseconds(self):
        from agent.credential_pool import _extract_retry_delay_seconds

        result = _extract_retry_delay_seconds("quotaResetDelay: 3600ms")
        assert result == pytest.approx(3.6, abs=0.01)

    def test_quota_reset_delay_seconds(self):
        from agent.credential_pool import _extract_retry_delay_seconds

        result = _extract_retry_delay_seconds("quotaResetDelay: 60s")
        assert result == 60.0

    def test_retry_after_seconds(self):
        from agent.credential_pool import _extract_retry_delay_seconds

        result = _extract_retry_delay_seconds("Please retry after 120 seconds")
        assert result == 120.0

    def test_retry_secs_shorthand(self):
        from agent.credential_pool import _extract_retry_delay_seconds

        result = _extract_retry_delay_seconds("retry 300 secs")
        assert result == 300.0

    def test_quota_delay_takes_precedence(self):
        from agent.credential_pool import _extract_retry_delay_seconds

        # quotaResetDelay match is checked first
        msg = "quotaResetDelay: 5000ms - retry after 120 seconds"
        result = _extract_retry_delay_seconds(msg)
        assert result == pytest.approx(5.0, abs=0.01)


# ===================================================================
# _normalize_error_context
# ===================================================================


class TestNormalizeErrorContext:
    def test_none_returns_empty_dict(self):
        from agent.credential_pool import _normalize_error_context

        assert _normalize_error_context(None) == {}

    def test_non_dict_returns_empty_dict(self):
        from agent.credential_pool import _normalize_error_context

        assert _normalize_error_context("string") == {}
        assert _normalize_error_context(42) == {}
        assert _normalize_error_context([1, 2]) == {}

    def test_empty_dict_returns_empty_dict(self):
        from agent.credential_pool import _normalize_error_context

        assert _normalize_error_context({}) == {}

    def test_reason_extracted_and_stripped(self):
        from agent.credential_pool import _normalize_error_context

        result = _normalize_error_context({"reason": "  rate_limited  "})
        assert result["reason"] == "rate_limited"

    def test_message_extracted_and_stripped(self):
        from agent.credential_pool import _normalize_error_context

        result = _normalize_error_context({"message": "  Too many requests  "})
        assert result["message"] == "Too many requests"

    def test_empty_reason_not_included(self):
        from agent.credential_pool import _normalize_error_context

        result = _normalize_error_context({"reason": "   "})
        assert "reason" not in result

    def test_reset_at_parsed_from_epoch(self):
        from agent.credential_pool import _normalize_error_context

        ts = 1700000000
        result = _normalize_error_context({"reset_at": ts})
        assert result["reset_at"] == float(ts)

    def test_resets_at_alias_works(self):
        from agent.credential_pool import _normalize_error_context

        ts = 1700000000
        result = _normalize_error_context({"resets_at": ts})
        assert result["reset_at"] == float(ts)

    def test_retry_until_alias_works(self):
        from agent.credential_pool import _normalize_error_context

        ts = 1700000000
        result = _normalize_error_context({"retry_until": ts})
        assert result["reset_at"] == float(ts)

    def test_fallback_to_message_parsing_for_reset_at(self):
        from agent.credential_pool import _normalize_error_context

        before = time.time()
        result = _normalize_error_context({
            "message": "quotaResetDelay: 60000ms",
        })
        after = time.time()
        assert "reset_at" in result
        # Should be ~60 seconds from now
        assert result["reset_at"] >= before + 59
        assert result["reset_at"] <= after + 61

    def test_all_fields_together(self):
        from agent.credential_pool import _normalize_error_context

        result = _normalize_error_context({
            "reason": "rate_limited",
            "message": "Too many requests",
            "reset_at": 1700000000,
        })
        assert result["reason"] == "rate_limited"
        assert result["message"] == "Too many requests"
        assert result["reset_at"] == 1700000000.0


# ===================================================================
# _exhausted_until
# ===================================================================


class TestExhaustedUntil:
    def test_non_exhausted_returns_none(self):
        from agent.credential_pool import PooledCredential, _exhausted_until

        entry = PooledCredential(
            provider="anthropic", id="x", label="x", auth_type="api_key",
            priority=0, source="manual", access_token="sk",
            last_status="ok",
        )
        assert _exhausted_until(entry) is None

    def test_exhausted_with_reset_at(self):
        from agent.credential_pool import PooledCredential, _exhausted_until

        reset_ts = time.time() + 3600
        entry = PooledCredential(
            provider="anthropic", id="x", label="x", auth_type="api_key",
            priority=0, source="manual", access_token="sk",
            last_status="exhausted",
            last_status_at=time.time(),
            last_error_code=429,
            last_error_reset_at=reset_ts,
        )
        assert _exhausted_until(entry) == reset_ts

    def test_exhausted_429_uses_1h_ttl(self):
        from agent.credential_pool import (
            EXHAUSTED_TTL_429_SECONDS,
            PooledCredential,
            _exhausted_until,
        )

        now = time.time()
        entry = PooledCredential(
            provider="anthropic", id="x", label="x", auth_type="api_key",
            priority=0, source="manual", access_token="sk",
            last_status="exhausted",
            last_status_at=now,
            last_error_code=429,
        )
        result = _exhausted_until(entry)
        assert result == pytest.approx(now + EXHAUSTED_TTL_429_SECONDS, abs=1)

    def test_exhausted_402_uses_24h_ttl(self):
        from agent.credential_pool import (
            EXHAUSTED_TTL_DEFAULT_SECONDS,
            PooledCredential,
            _exhausted_until,
        )

        now = time.time()
        entry = PooledCredential(
            provider="anthropic", id="x", label="x", auth_type="api_key",
            priority=0, source="manual", access_token="sk",
            last_status="exhausted",
            last_status_at=now,
            last_error_code=402,
        )
        result = _exhausted_until(entry)
        assert result == pytest.approx(now + EXHAUSTED_TTL_DEFAULT_SECONDS, abs=1)

    def test_exhausted_no_timing_returns_none(self):
        from agent.credential_pool import PooledCredential, _exhausted_until

        entry = PooledCredential(
            provider="anthropic", id="x", label="x", auth_type="api_key",
            priority=0, source="manual", access_token="sk",
            last_status="exhausted",
            last_status_at=None,
            last_error_code=None,
        )
        assert _exhausted_until(entry) is None


# ===================================================================
# _exhausted_ttl
# ===================================================================


class TestExhaustedTtl:
    def test_429_returns_1h(self):
        from agent.credential_pool import EXHAUSTED_TTL_429_SECONDS, _exhausted_ttl

        assert _exhausted_ttl(429) == EXHAUSTED_TTL_429_SECONDS

    def test_402_returns_24h(self):
        from agent.credential_pool import EXHAUSTED_TTL_DEFAULT_SECONDS, _exhausted_ttl

        assert _exhausted_ttl(402) == EXHAUSTED_TTL_DEFAULT_SECONDS

    def test_none_returns_24h(self):
        from agent.credential_pool import EXHAUSTED_TTL_DEFAULT_SECONDS, _exhausted_ttl

        assert _exhausted_ttl(None) == EXHAUSTED_TTL_DEFAULT_SECONDS

    def test_500_returns_24h(self):
        from agent.credential_pool import EXHAUSTED_TTL_DEFAULT_SECONDS, _exhausted_ttl

        assert _exhausted_ttl(500) == EXHAUSTED_TTL_DEFAULT_SECONDS


# ===================================================================
# _is_manual_source
# ===================================================================


class TestIsManualSource:
    def test_manual_exact(self):
        from agent.credential_pool import _is_manual_source

        assert _is_manual_source("manual") is True

    def test_manual_prefixed(self):
        from agent.credential_pool import _is_manual_source

        assert _is_manual_source("manual:cli") is True

    def test_manual_case_insensitive(self):
        from agent.credential_pool import _is_manual_source

        assert _is_manual_source("Manual") is True
        assert _is_manual_source("MANUAL") is True

    def test_non_manual(self):
        from agent.credential_pool import _is_manual_source

        assert _is_manual_source("env") is False
        assert _is_manual_source("claude_code") is False

    def test_none_returns_false(self):
        from agent.credential_pool import _is_manual_source

        assert _is_manual_source(None) is False

    def test_empty_returns_false(self):
        from agent.credential_pool import _is_manual_source

        assert _is_manual_source("") is False


# ===================================================================
# _next_priority
# ===================================================================


class TestNextPriority:
    def test_empty_list(self):
        from agent.credential_pool import _next_priority

        assert _next_priority([]) == 0

    def test_single_entry(self):
        from agent.credential_pool import PooledCredential, _next_priority

        entry = PooledCredential(
            provider="anthropic", id="x", label="x", auth_type="api_key",
            priority=0, source="manual", access_token="sk",
        )
        assert _next_priority([entry]) == 1

    def test_multiple_entries(self):
        from agent.credential_pool import PooledCredential, _next_priority

        entries = [
            PooledCredential(
                provider="anthropic", id=f"x{i}", label=f"x{i}",
                auth_type="api_key", priority=i, source="manual",
                access_token=f"sk-{i}",
            )
            for i in range(5)
        ]
        assert _next_priority(entries) == 5


# ===================================================================
# label_from_token
# ===================================================================


class TestLabelFromToken:
    def test_fallback_on_invalid_token(self, monkeypatch):
        from agent.credential_pool import label_from_token

        monkeypatch.setattr(
            "agent.credential_pool._decode_jwt_claims", lambda t: {},
        )
        assert label_from_token("invalid-token", "fallback-label") == "fallback-label"

    def test_email_claim_preferred(self, monkeypatch):
        from agent.credential_pool import label_from_token

        monkeypatch.setattr(
            "agent.credential_pool._decode_jwt_claims",
            lambda t: {"email": "user@example.com", "preferred_username": "user123"},
        )
        assert label_from_token("tok", "fb") == "user@example.com"

    def test_preferred_username_fallback(self, monkeypatch):
        from agent.credential_pool import label_from_token

        monkeypatch.setattr(
            "agent.credential_pool._decode_jwt_claims",
            lambda t: {"preferred_username": "user123"},
        )
        assert label_from_token("tok", "fb") == "user123"

    def test_upn_fallback(self, monkeypatch):
        from agent.credential_pool import label_from_token

        monkeypatch.setattr(
            "agent.credential_pool._decode_jwt_claims",
            lambda t: {"upn": "user@corp.com"},
        )
        assert label_from_token("tok", "fb") == "user@corp.com"

    def test_whitespace_only_claim_ignored(self, monkeypatch):
        from agent.credential_pool import label_from_token

        monkeypatch.setattr(
            "agent.credential_pool._decode_jwt_claims",
            lambda t: {"email": "   ", "preferred_username": "user"},
        )
        assert label_from_token("tok", "fb") == "user"

    def test_claim_values_stripped(self, monkeypatch):
        from agent.credential_pool import label_from_token

        monkeypatch.setattr(
            "agent.credential_pool._decode_jwt_claims",
            lambda t: {"email": "  user@example.com  "},
        )
        assert label_from_token("tok", "fb") == "user@example.com"


# ===================================================================
# CredentialPool.resolve_target
# ===================================================================


class TestResolveTarget:
    def test_resolve_by_id(self, tmp_path, monkeypatch):
        pool = _make_pool(tmp_path, monkeypatch, [_CRED_A, _CRED_B])
        idx, entry, err = pool.resolve_target("aaa")
        assert idx == 1
        assert entry.id == "aaa"
        assert err is None

    def test_resolve_by_label(self, tmp_path, monkeypatch):
        pool = _make_pool(tmp_path, monkeypatch, [_CRED_A, _CRED_B])
        idx, entry, err = pool.resolve_target("secondary")
        assert idx == 2
        assert entry.id == "bbb"
        assert err is None

    def test_resolve_by_label_case_insensitive(self, tmp_path, monkeypatch):
        pool = _make_pool(tmp_path, monkeypatch, [_CRED_A, _CRED_B])
        idx, entry, err = pool.resolve_target("PRIMARY")
        assert idx == 1
        assert entry.id == "aaa"
        assert err is None

    def test_resolve_by_numeric_index(self, tmp_path, monkeypatch):
        pool = _make_pool(tmp_path, monkeypatch, [_CRED_A, _CRED_B])
        idx, entry, err = pool.resolve_target("2")
        assert idx == 2
        assert entry.id == "bbb"
        assert err is None

    def test_resolve_empty_target(self, tmp_path, monkeypatch):
        pool = _make_pool(tmp_path, monkeypatch, [_CRED_A])
        idx, entry, err = pool.resolve_target("")
        assert idx is None
        assert entry is None
        assert "No credential target" in err

    def test_resolve_none_target(self, tmp_path, monkeypatch):
        pool = _make_pool(tmp_path, monkeypatch, [_CRED_A])
        idx, entry, err = pool.resolve_target(None)
        assert idx is None
        assert entry is None
        assert err is not None

    def test_resolve_ambiguous_label(self, tmp_path, monkeypatch):
        dup_a = {**_CRED_A, "label": "shared"}
        dup_b = {**_CRED_B, "label": "shared"}
        pool = _make_pool(tmp_path, monkeypatch, [dup_a, dup_b])
        idx, entry, err = pool.resolve_target("shared")
        assert idx is None
        assert entry is None
        assert "Ambiguous" in err

    def test_resolve_out_of_range_index(self, tmp_path, monkeypatch):
        pool = _make_pool(tmp_path, monkeypatch, [_CRED_A])
        idx, entry, err = pool.resolve_target("99")
        assert idx is None
        assert entry is None
        assert "No credential #99" in err

    def test_resolve_no_match(self, tmp_path, monkeypatch):
        pool = _make_pool(tmp_path, monkeypatch, [_CRED_A])
        idx, entry, err = pool.resolve_target("nonexistent")
        assert idx is None
        assert entry is None
        assert 'No credential matching' in err


# ===================================================================
# CredentialPool.add_entry
# ===================================================================


class TestAddEntry:
    def test_add_to_empty_pool(self, tmp_path, monkeypatch):
        pool = _make_pool(tmp_path, monkeypatch, [])

        from agent.credential_pool import PooledCredential

        new_entry = PooledCredential(
            provider="anthropic", id="new1", label="new",
            auth_type="api_key", priority=0, source="manual",
            access_token="sk-new",
        )
        added = pool.add_entry(new_entry)
        assert added.priority == 0
        assert len(pool.entries()) == 1

    def test_add_assigns_next_priority(self, tmp_path, monkeypatch):
        pool = _make_pool(tmp_path, monkeypatch, [_CRED_A, _CRED_B])

        from agent.credential_pool import PooledCredential

        new_entry = PooledCredential(
            provider="anthropic", id="new1", label="new",
            auth_type="api_key", priority=0, source="manual",
            access_token="sk-new",
        )
        added = pool.add_entry(new_entry)
        assert added.priority == 2
        assert len(pool.entries()) == 3

    def test_add_persists_to_disk(self, tmp_path, monkeypatch):
        pool = _make_pool(tmp_path, monkeypatch, [_CRED_A])

        from agent.credential_pool import PooledCredential

        new_entry = PooledCredential(
            provider="anthropic", id="new1", label="new",
            auth_type="api_key", priority=0, source="manual",
            access_token="sk-new",
        )
        pool.add_entry(new_entry)

        # Reload from disk
        from agent.credential_pool import load_pool

        reloaded = load_pool("anthropic")
        assert len(reloaded.entries()) == 2


# ===================================================================
# CredentialPool.remove_index
# ===================================================================


class TestRemoveIndex:
    def test_remove_first(self, tmp_path, monkeypatch):
        pool = _make_pool(tmp_path, monkeypatch, [_CRED_A, _CRED_B, _CRED_C])
        removed = pool.remove_index(1)
        assert removed.id == "aaa"
        assert len(pool.entries()) == 2
        # Priorities renumbered
        assert pool.entries()[0].priority == 0
        assert pool.entries()[1].priority == 1

    def test_remove_last(self, tmp_path, monkeypatch):
        pool = _make_pool(tmp_path, monkeypatch, [_CRED_A, _CRED_B])
        removed = pool.remove_index(2)
        assert removed.id == "bbb"
        assert len(pool.entries()) == 1

    def test_remove_middle(self, tmp_path, monkeypatch):
        pool = _make_pool(tmp_path, monkeypatch, [_CRED_A, _CRED_B, _CRED_C])
        removed = pool.remove_index(2)
        assert removed.id == "bbb"
        remaining = pool.entries()
        assert len(remaining) == 2
        assert remaining[0].id == "aaa"
        assert remaining[1].id == "ccc"

    def test_remove_invalid_index_zero(self, tmp_path, monkeypatch):
        pool = _make_pool(tmp_path, monkeypatch, [_CRED_A])
        assert pool.remove_index(0) is None
        assert len(pool.entries()) == 1

    def test_remove_invalid_index_negative(self, tmp_path, monkeypatch):
        pool = _make_pool(tmp_path, monkeypatch, [_CRED_A])
        assert pool.remove_index(-1) is None

    def test_remove_invalid_index_out_of_range(self, tmp_path, monkeypatch):
        pool = _make_pool(tmp_path, monkeypatch, [_CRED_A])
        assert pool.remove_index(5) is None

    def test_remove_current_clears_current_id(self, tmp_path, monkeypatch):
        pool = _make_pool(tmp_path, monkeypatch, [_CRED_A, _CRED_B])
        pool.select()  # selects first entry
        assert pool.current() is not None
        pool.remove_index(1)  # remove the current entry
        assert pool.current() is None


# ===================================================================
# CredentialPool.peek
# ===================================================================


class TestPeek:
    def test_peek_returns_current_when_set(self, tmp_path, monkeypatch):
        pool = _make_pool(tmp_path, monkeypatch, [_CRED_A, _CRED_B])
        pool.select()
        peeked = pool.peek()
        assert peeked is not None
        assert peeked.id == pool.current().id

    def test_peek_returns_first_available_when_no_current(self, tmp_path, monkeypatch):
        pool = _make_pool(tmp_path, monkeypatch, [_CRED_A, _CRED_B])
        # No select() called, so no current
        peeked = pool.peek()
        assert peeked is not None
        assert peeked.id == "aaa"

    def test_peek_empty_pool_returns_none(self, tmp_path, monkeypatch):
        pool = _make_pool(tmp_path, monkeypatch, [])
        assert pool.peek() is None

    def test_peek_all_exhausted_returns_none(self, tmp_path, monkeypatch):
        exhausted = {
            **_CRED_A,
            "last_status": "exhausted",
            "last_status_at": time.time(),
            "last_error_code": 402,
        }
        pool = _make_pool(tmp_path, monkeypatch, [exhausted])
        assert pool.peek() is None


# ===================================================================
# CredentialPool.reset_statuses
# ===================================================================


class TestResetStatuses:
    def test_reset_exhausted_entries(self, tmp_path, monkeypatch):
        exhausted_a = {
            **_CRED_A,
            "last_status": "exhausted",
            "last_status_at": time.time(),
            "last_error_code": 429,
            "last_error_reason": "rate_limited",
            "last_error_message": "Too many requests",
        }
        pool = _make_pool(tmp_path, monkeypatch, [exhausted_a, _CRED_B])
        count = pool.reset_statuses()
        assert count == 1
        # Entry should now be cleared
        entry = pool.entries()[0]
        assert entry.last_status is None
        assert entry.last_status_at is None
        assert entry.last_error_code is None
        assert entry.last_error_reason is None
        assert entry.last_error_message is None

    def test_reset_returns_zero_when_nothing_to_reset(self, tmp_path, monkeypatch):
        pool = _make_pool(tmp_path, monkeypatch, [_CRED_A, _CRED_B])
        count = pool.reset_statuses()
        assert count == 0

    def test_reset_multiple_exhausted(self, tmp_path, monkeypatch):
        exhausted_a = {**_CRED_A, "last_status": "exhausted", "last_status_at": time.time(), "last_error_code": 429}
        exhausted_b = {**_CRED_B, "last_status": "exhausted", "last_status_at": time.time(), "last_error_code": 402}
        pool = _make_pool(tmp_path, monkeypatch, [exhausted_a, exhausted_b])
        count = pool.reset_statuses()
        assert count == 2

    def test_reset_empty_pool(self, tmp_path, monkeypatch):
        pool = _make_pool(tmp_path, monkeypatch, [])
        assert pool.reset_statuses() == 0


# ===================================================================
# CredentialPool._entry_needs_refresh
# ===================================================================


class TestEntryNeedsRefresh:
    def test_api_key_never_needs_refresh(self, tmp_path, monkeypatch):
        pool = _make_pool(tmp_path, monkeypatch, [_CRED_A])
        entry = pool.entries()[0]
        assert pool._entry_needs_refresh(entry) is False

    def test_anthropic_oauth_no_expires_at(self, tmp_path, monkeypatch):
        oauth_entry = {
            **_CRED_A,
            "auth_type": "oauth",
            "refresh_token": "rt-xxx",
            "expires_at_ms": None,
        }
        pool = _make_pool(tmp_path, monkeypatch, [oauth_entry])
        entry = pool.entries()[0]
        assert pool._entry_needs_refresh(entry) is False

    def test_anthropic_oauth_expiring_soon(self, tmp_path, monkeypatch):
        # Expires in 60 seconds (within 120s skew window)
        oauth_entry = {
            **_CRED_A,
            "auth_type": "oauth",
            "refresh_token": "rt-xxx",
            "expires_at_ms": int(time.time() * 1000) + 60_000,
        }
        pool = _make_pool(tmp_path, monkeypatch, [oauth_entry])
        entry = pool.entries()[0]
        assert pool._entry_needs_refresh(entry) is True

    def test_anthropic_oauth_not_expiring(self, tmp_path, monkeypatch):
        # Expires in 1 hour (well outside 120s skew window)
        oauth_entry = {
            **_CRED_A,
            "auth_type": "oauth",
            "refresh_token": "rt-xxx",
            "expires_at_ms": int(time.time() * 1000) + 3_600_000,
        }
        pool = _make_pool(tmp_path, monkeypatch, [oauth_entry])
        entry = pool.entries()[0]
        assert pool._entry_needs_refresh(entry) is False

    def test_nous_never_needs_pool_refresh(self, tmp_path, monkeypatch):
        nous_entry = {
            **_CRED_A,
            "auth_type": "oauth",
            "refresh_token": "rt-xxx",
        }
        pool = _make_pool(tmp_path, monkeypatch, [nous_entry], provider="nous")
        entry = pool.entries()[0]
        assert pool._entry_needs_refresh(entry) is False


# ===================================================================
# get_pool_strategy
# ===================================================================


class TestGetPoolStrategy:
    def test_default_when_no_config(self, monkeypatch):
        from agent.credential_pool import STRATEGY_FILL_FIRST, get_pool_strategy

        monkeypatch.setattr("agent.credential_pool._load_config_safe", lambda: None)
        assert get_pool_strategy("anthropic") == STRATEGY_FILL_FIRST

    def test_default_when_no_strategies_key(self, monkeypatch):
        from agent.credential_pool import STRATEGY_FILL_FIRST, get_pool_strategy

        monkeypatch.setattr("agent.credential_pool._load_config_safe", lambda: {})
        assert get_pool_strategy("anthropic") == STRATEGY_FILL_FIRST

    def test_round_robin_configured(self, monkeypatch):
        from agent.credential_pool import STRATEGY_ROUND_ROBIN, get_pool_strategy

        monkeypatch.setattr(
            "agent.credential_pool._load_config_safe",
            lambda: {"credential_pool_strategies": {"anthropic": "round_robin"}},
        )
        assert get_pool_strategy("anthropic") == STRATEGY_ROUND_ROBIN

    def test_invalid_strategy_falls_back(self, monkeypatch):
        from agent.credential_pool import STRATEGY_FILL_FIRST, get_pool_strategy

        monkeypatch.setattr(
            "agent.credential_pool._load_config_safe",
            lambda: {"credential_pool_strategies": {"anthropic": "invalid_strategy"}},
        )
        assert get_pool_strategy("anthropic") == STRATEGY_FILL_FIRST

    def test_case_insensitive(self, monkeypatch):
        from agent.credential_pool import STRATEGY_RANDOM, get_pool_strategy

        monkeypatch.setattr(
            "agent.credential_pool._load_config_safe",
            lambda: {"credential_pool_strategies": {"anthropic": "RANDOM"}},
        )
        assert get_pool_strategy("anthropic") == STRATEGY_RANDOM


# ===================================================================
# PooledCredential.from_dict / to_dict round-trip
# ===================================================================


class TestPooledCredentialSerialization:
    def test_round_trip_preserves_fields(self):
        from agent.credential_pool import PooledCredential

        original = PooledCredential(
            provider="anthropic", id="test1", label="test",
            auth_type="api_key", priority=0, source="manual",
            access_token="sk-xxx",
            last_status="exhausted", last_status_at=1700000000.0,
            last_error_code=429, last_error_reason="rate_limited",
        )
        d = original.to_dict()
        restored = PooledCredential.from_dict("anthropic", d)
        assert restored.id == original.id
        assert restored.label == original.label
        assert restored.last_status == original.last_status
        assert restored.last_error_code == original.last_error_code

    def test_from_dict_defaults(self):
        from agent.credential_pool import PooledCredential

        entry = PooledCredential.from_dict("anthropic", {})
        assert entry.provider == "anthropic"
        assert entry.auth_type == "api_key"
        assert entry.priority == 0
        assert entry.source == "manual"
        assert entry.access_token == ""

    def test_extra_keys_round_trip(self):
        from agent.credential_pool import PooledCredential

        entry = PooledCredential.from_dict("nous", {
            "id": "x", "access_token": "tok",
            "token_type": "Bearer", "scope": "openid",
            "client_id": "hermes-cli",
        })
        assert entry.token_type == "Bearer"
        assert entry.scope == "openid"
        d = entry.to_dict()
        assert d["token_type"] == "Bearer"
        assert d["scope"] == "openid"

    def test_runtime_api_key_anthropic_uses_access_token(self):
        from agent.credential_pool import PooledCredential

        entry = PooledCredential(
            provider="anthropic", id="x", label="x", auth_type="api_key",
            priority=0, source="manual", access_token="sk-ant-xxx",
        )
        assert entry.runtime_api_key == "sk-ant-xxx"

    def test_runtime_base_url_nous_prefers_inference(self):
        from agent.credential_pool import PooledCredential

        entry = PooledCredential(
            provider="nous", id="x", label="x", auth_type="oauth",
            priority=0, source="env", access_token="tok",
            base_url="https://base.example.com",
            inference_base_url="https://inference.example.com",
        )
        assert entry.runtime_base_url == "https://inference.example.com"

    def test_runtime_base_url_anthropic_uses_base_url(self):
        from agent.credential_pool import PooledCredential

        entry = PooledCredential(
            provider="anthropic", id="x", label="x", auth_type="api_key",
            priority=0, source="manual", access_token="sk",
            base_url="https://custom.example.com",
        )
        assert entry.runtime_base_url == "https://custom.example.com"


# ===================================================================
# _normalize_custom_pool_name
# ===================================================================


class TestNormalizeCustomPoolName:
    def test_lowercase_and_dash(self):
        from agent.credential_pool import _normalize_custom_pool_name

        assert _normalize_custom_pool_name("Together AI") == "together-ai"

    def test_strips_whitespace(self):
        from agent.credential_pool import _normalize_custom_pool_name

        assert _normalize_custom_pool_name("  fireworks  ") == "fireworks"

    def test_already_normalized(self):
        from agent.credential_pool import _normalize_custom_pool_name

        assert _normalize_custom_pool_name("together.ai") == "together.ai"
