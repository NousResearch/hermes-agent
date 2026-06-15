"""Tests for agent.credits_tracker — CreditsState + parse_credits_headers.

Covers the 9-state matrix plus validation edge cases.  All header values
arrive as STRINGS (the producer calls String(...) on every field).
"""

from __future__ import annotations

import logging
import time
import pytest

from agent.credits_tracker import CreditsState, parse_credits_headers


# ── Helpers ─────────────────────────────────────────────────────────────────


def micros(dollars: float) -> str:
    """Convert a dollar amount to a micros string for header fixtures."""
    return str(round(dollars * 1_000_000))


# ── 9-State matrix fixtures ──────────────────────────────────────────────────


def _base_headers(**overrides) -> dict:
    """Base headers present in every valid response."""
    h = {
        "x-nous-credits-version": "1",
        "x-nous-credits-remaining-micros": micros(0),
        "x-nous-credits-remaining-usd": "0.00",
        "x-nous-credits-subscription-micros": micros(0),
        "x-nous-credits-subscription-usd": "0.00",
        "x-nous-credits-rollover-micros": micros(0),
        "x-nous-credits-purchased-micros": micros(0),
        "x-nous-credits-purchased-usd": "0.00",
        "x-nous-tool-pool-micros": micros(0),
        "x-nous-tool-pool-gated-off": "false",
        "x-nous-credits-denominator-kind": "none",
        "x-nous-credits-paid-access": "true",
        "x-nous-credits-as-of-ms": "1717000000000",
    }
    h.update(overrides)
    return h


# ── 9 STATES ────────────────────────────────────────────────────────────────


HEALTHY_HEADERS = _base_headers(
    **{
        "x-nous-credits-remaining-micros": micros(30.34),
        "x-nous-credits-remaining-usd": "30.34",
        "x-nous-credits-subscription-micros": micros(18.00),
        "x-nous-credits-subscription-usd": "18.00",
        "x-nous-credits-subscription-limit-micros": micros(20.00),
        "x-nous-credits-subscription-limit-usd": "20.00",
        "x-nous-credits-rollover-micros": micros(0),
        "x-nous-credits-purchased-micros": micros(12.34),
        "x-nous-credits-purchased-usd": "12.34",
        "x-nous-tool-pool-micros": micros(2.00),
        "x-nous-tool-pool-gated-off": "true",
        "x-nous-credits-denominator-kind": "subscription_cap",
        "x-nous-credits-paid-access": "true",
    }
)

SUB_90PCT_HEADERS = _base_headers(
    **{
        "x-nous-credits-remaining-micros": micros(2.00),
        "x-nous-credits-remaining-usd": "2.00",
        "x-nous-credits-subscription-micros": micros(2.00),
        "x-nous-credits-subscription-usd": "2.00",
        "x-nous-credits-subscription-limit-micros": micros(20.00),
        "x-nous-credits-subscription-limit-usd": "20.00",
        "x-nous-credits-purchased-micros": micros(0),
        "x-nous-credits-purchased-usd": "0.00",
        "x-nous-credits-denominator-kind": "subscription_cap",
        "x-nous-credits-paid-access": "true",
    }
)

GRANT_EXHAUSTED_HEADERS = _base_headers(
    **{
        "x-nous-credits-remaining-micros": micros(12.34),
        "x-nous-credits-remaining-usd": "12.34",
        "x-nous-credits-subscription-micros": micros(0),
        "x-nous-credits-subscription-usd": "0.00",
        "x-nous-credits-subscription-limit-micros": micros(20.00),
        "x-nous-credits-subscription-limit-usd": "20.00",
        "x-nous-credits-purchased-micros": micros(12.34),
        "x-nous-credits-purchased-usd": "12.34",
        "x-nous-credits-denominator-kind": "subscription_cap",
        "x-nous-credits-paid-access": "true",
    }
)

PURCHASED_ONLY_HEADERS = _base_headers(
    **{
        "x-nous-credits-remaining-micros": micros(30.00),
        "x-nous-credits-remaining-usd": "30.00",
        "x-nous-credits-subscription-micros": micros(0),
        "x-nous-credits-subscription-usd": "0.00",
        "x-nous-credits-purchased-micros": micros(30.00),
        "x-nous-credits-purchased-usd": "30.00",
        "x-nous-credits-denominator-kind": "none",
        "x-nous-credits-paid-access": "true",
        # No limit pair — denominator_kind=none
    }
)

TOOL_POOL_FREE_HEADERS = _base_headers(
    **{
        "x-nous-credits-remaining-micros": micros(0.05),
        "x-nous-credits-remaining-usd": "0.05",
        "x-nous-tool-pool-micros": micros(0.05),
        "x-nous-tool-pool-gated-off": "false",
        "x-nous-credits-paid-access": "true",
    }
)

DEPLETED_HEADERS = _base_headers(
    **{
        "x-nous-credits-remaining-micros": micros(0),
        "x-nous-credits-remaining-usd": "0.00",
        "x-nous-credits-subscription-micros": micros(0),
        "x-nous-credits-subscription-usd": "0.00",
        "x-nous-credits-purchased-micros": micros(0),
        "x-nous-credits-purchased-usd": "0.00",
        "x-nous-credits-paid-access": "false",
        "x-nous-credits-disabled-reason": "out_of_credits",
    }
)

DEBT_HEADERS = _base_headers(
    **{
        "x-nous-credits-remaining-micros": micros(0),
        "x-nous-credits-remaining-usd": "0.00",
        "x-nous-credits-subscription-micros": str(-5_000_000),
        "x-nous-credits-subscription-usd": "-5.00",
        "x-nous-credits-purchased-micros": micros(0),
        "x-nous-credits-purchased-usd": "0.00",
        "x-nous-credits-paid-access": "false",
    }
)


# ── State 1: healthy ─────────────────────────────────────────────────────────


class TestHealthyState:
    def test_parses_successfully(self):
        state = parse_credits_headers(HEALTHY_HEADERS)
        assert state is not None

    def test_from_header_set(self):
        state = parse_credits_headers(HEALTHY_HEADERS)
        assert state.from_header is True

    def test_captured_at_set(self):
        before = time.time()
        state = parse_credits_headers(HEALTHY_HEADERS)
        after = time.time()
        assert before <= state.captured_at <= after

    def test_remaining_fields(self):
        state = parse_credits_headers(HEALTHY_HEADERS)
        assert state.remaining_micros == round(30.34 * 1_000_000)
        assert state.remaining_usd == "30.34"

    def test_subscription_fields(self):
        state = parse_credits_headers(HEALTHY_HEADERS)
        assert state.subscription_micros == round(18.00 * 1_000_000)
        assert state.subscription_usd == "18.00"
        assert state.subscription_limit_micros == round(20.00 * 1_000_000)
        assert state.subscription_limit_usd == "20.00"

    def test_rollover_and_purchased(self):
        state = parse_credits_headers(HEALTHY_HEADERS)
        assert state.rollover_micros == 0
        assert state.purchased_micros == round(12.34 * 1_000_000)
        assert state.purchased_usd == "12.34"

    def test_tool_pool(self):
        state = parse_credits_headers(HEALTHY_HEADERS)
        assert state.tool_pool_micros == round(2.00 * 1_000_000)
        assert state.tool_pool_gated_off is True

    def test_denominator_and_access(self):
        state = parse_credits_headers(HEALTHY_HEADERS)
        assert state.denominator_kind == "subscription_cap"
        assert state.paid_access is True
        assert state.disabled_reason is None

    def test_used_fraction(self):
        state = parse_credits_headers(HEALTHY_HEADERS)
        # (20.00 - 18.00) / 20.00 = 0.10
        assert state.used_fraction == pytest.approx(0.10)

    def test_has_data(self):
        state = parse_credits_headers(HEALTHY_HEADERS)
        assert state.has_data is True

    def test_not_depleted(self):
        state = parse_credits_headers(HEALTHY_HEADERS)
        assert state.depleted is False

    def test_age_seconds_reasonable(self):
        state = parse_credits_headers(HEALTHY_HEADERS)
        # Should be very small — just parsed
        assert 0 <= state.age_seconds < 5


# ── State 2: sub_90pct ───────────────────────────────────────────────────────


class TestSub90Pct:
    def test_parses_successfully(self):
        state = parse_credits_headers(SUB_90PCT_HEADERS)
        assert state is not None

    def test_used_fraction_90pct(self):
        state = parse_credits_headers(SUB_90PCT_HEADERS)
        # (20.00 - 2.00) / 20.00 = 0.90
        assert state.used_fraction == pytest.approx(0.90)

    def test_paid_access(self):
        state = parse_credits_headers(SUB_90PCT_HEADERS)
        assert state.paid_access is True
        assert state.depleted is False


# ── State 3: grant_exhausted ─────────────────────────────────────────────────


class TestGrantExhausted:
    def test_used_fraction_100pct(self):
        state = parse_credits_headers(GRANT_EXHAUSTED_HEADERS)
        assert state is not None
        # subscription_micros=0, limit=20.00 → (20-0)/20 = 1.0
        assert state.used_fraction == pytest.approx(1.0)

    def test_paid_access_still_true(self):
        state = parse_credits_headers(GRANT_EXHAUSTED_HEADERS)
        assert state.paid_access is True
        assert state.depleted is False


# ── State 4: purchased_only ──────────────────────────────────────────────────


class TestPurchasedOnly:
    def test_parses_successfully(self):
        state = parse_credits_headers(PURCHASED_ONLY_HEADERS)
        assert state is not None

    def test_denominator_kind_none(self):
        state = parse_credits_headers(PURCHASED_ONLY_HEADERS)
        assert state.denominator_kind == "none"

    def test_used_fraction_is_none_no_limit(self):
        state = parse_credits_headers(PURCHASED_ONLY_HEADERS)
        # No subscription_limit_micros → used_fraction is None
        assert state.used_fraction is None

    def test_no_limit_pair(self):
        state = parse_credits_headers(PURCHASED_ONLY_HEADERS)
        assert state.subscription_limit_micros is None
        assert state.subscription_limit_usd is None


# ── State 5: tool_pool_free ──────────────────────────────────────────────────


class TestToolPoolFree:
    def test_parses_successfully(self):
        state = parse_credits_headers(TOOL_POOL_FREE_HEADERS)
        assert state is not None

    def test_tool_pool_gated_off_false(self):
        state = parse_credits_headers(TOOL_POOL_FREE_HEADERS)
        assert state.tool_pool_gated_off is False

    def test_tool_pool_micros(self):
        state = parse_credits_headers(TOOL_POOL_FREE_HEADERS)
        assert state.tool_pool_micros == round(0.05 * 1_000_000)

    def test_paid_access(self):
        state = parse_credits_headers(TOOL_POOL_FREE_HEADERS)
        assert state.paid_access is True


# ── State 6: depleted ────────────────────────────────────────────────────────


class TestDepleted:
    def test_parses_successfully(self):
        state = parse_credits_headers(DEPLETED_HEADERS)
        assert state is not None

    def test_paid_access_false(self):
        state = parse_credits_headers(DEPLETED_HEADERS)
        assert state.paid_access is False

    def test_depleted_true(self):
        state = parse_credits_headers(DEPLETED_HEADERS)
        assert state.depleted is True

    def test_disabled_reason(self):
        state = parse_credits_headers(DEPLETED_HEADERS)
        assert state.disabled_reason == "out_of_credits"

    def test_remaining_zero(self):
        state = parse_credits_headers(DEPLETED_HEADERS)
        assert state.remaining_micros == 0


# ── State 7: debt ────────────────────────────────────────────────────────────


class TestDebt:
    def test_parses_successfully(self):
        # Negative subscription_micros should NOT cause the parse to fail
        state = parse_credits_headers(DEBT_HEADERS)
        assert state is not None

    def test_negative_subscription_accepted(self):
        state = parse_credits_headers(DEBT_HEADERS)
        assert state.subscription_micros == -5_000_000

    def test_negative_subscription_usd_accepted(self):
        state = parse_credits_headers(DEBT_HEADERS)
        assert state.subscription_usd == "-5.00"

    def test_paid_access_false(self):
        state = parse_credits_headers(DEBT_HEADERS)
        assert state.paid_access is False
        assert state.depleted is True


# ── State 8: missing ─────────────────────────────────────────────────────────


class TestMissing:
    def test_no_credits_headers_returns_none(self):
        state = parse_credits_headers({})
        assert state is None

    def test_completely_empty_dict(self):
        assert parse_credits_headers({}) is None


# ── State 9: no_org ──────────────────────────────────────────────────────────


class TestNoOrg:
    def test_irrelevant_headers_return_none(self):
        headers = {
            "content-type": "application/json",
            "x-request-id": "abc123",
            "server": "nginx",
        }
        state = parse_credits_headers(headers)
        assert state is None

    def test_api_key_path_no_org_returns_none(self):
        # Headers that might appear on an api-key path with no org
        headers = {
            "content-type": "application/json",
            "authorization": "Bearer sk-test",
        }
        assert parse_credits_headers(headers) is None


# ── Version validation ───────────────────────────────────────────────────────


class TestVersionValidation:
    def test_version_string_1_parses(self):
        headers = _base_headers(**{"x-nous-credits-version": "1"})
        state = parse_credits_headers(headers)
        assert state is not None
        assert state.version == 1

    def test_version_2_returns_none(self):
        headers = _base_headers(**{"x-nous-credits-version": "2"})
        state = parse_credits_headers(headers)
        assert state is None

    def test_version_absent_returns_none(self):
        headers = {k: v for k, v in _base_headers().items() if k != "x-nous-credits-version"}
        state = parse_credits_headers(headers)
        assert state is None

    def test_version_greater_than_1_warns_once(self, caplog):
        """Version > 1 must log a warning, and ONLY ONCE across multiple calls."""
        import agent.credits_tracker as ct

        original = ct._version_warning_emitted
        try:
            # Reset the warn-once latch so this test starts clean regardless of order
            ct._version_warning_emitted = False

            headers = _base_headers(**{"x-nous-credits-version": "3"})
            with caplog.at_level(logging.WARNING, logger="agent.credits_tracker"):
                parse_credits_headers(headers)
                parse_credits_headers(headers)
                parse_credits_headers(headers)

            warning_records = [r for r in caplog.records if "unsupported" in r.message.lower() or "version" in r.message.lower()]
            assert len(warning_records) == 1, (
                f"Expected exactly 1 version warning, got {len(warning_records)}: {[r.message for r in warning_records]}"
            )
        finally:
            ct._version_warning_emitted = original

    def test_version_0_returns_none(self):
        headers = _base_headers(**{"x-nous-credits-version": "0"})
        assert parse_credits_headers(headers) is None

    def test_version_non_int_returns_none(self):
        headers = _base_headers(**{"x-nous-credits-version": "abc"})
        assert parse_credits_headers(headers) is None


# ── Bool-string trap ─────────────────────────────────────────────────────────


class TestBoolStringTrap:
    """Explicit tests for the bool("false") == True trap."""

    def test_paid_access_string_false_means_depleted(self):
        """paid_access='false' must yield paid_access=False — NOT True."""
        headers = _base_headers(**{"x-nous-credits-paid-access": "false"})
        state = parse_credits_headers(headers)
        assert state is not None
        assert state.paid_access is False
        assert state.depleted is True

    def test_paid_access_string_true_means_not_depleted(self):
        headers = _base_headers(**{"x-nous-credits-paid-access": "true"})
        state = parse_credits_headers(headers)
        assert state is not None
        assert state.paid_access is True
        assert state.depleted is False

    def test_paid_access_case_insensitive_FALSE(self):
        headers = _base_headers(**{"x-nous-credits-paid-access": "FALSE"})
        state = parse_credits_headers(headers)
        assert state is not None
        assert state.paid_access is False

    def test_paid_access_case_insensitive_True(self):
        headers = _base_headers(**{"x-nous-credits-paid-access": "True"})
        state = parse_credits_headers(headers)
        assert state is not None
        assert state.paid_access is True

    def test_tool_pool_gated_off_false(self):
        headers = _base_headers(**{"x-nous-tool-pool-gated-off": "false"})
        state = parse_credits_headers(headers)
        assert state is not None
        assert state.tool_pool_gated_off is False

    def test_tool_pool_gated_off_true(self):
        headers = _base_headers(**{"x-nous-tool-pool-gated-off": "true"})
        state = parse_credits_headers(headers)
        assert state is not None
        assert state.tool_pool_gated_off is True


# ── Tool-pool optional headers ────────────────────────────────────────────────


class TestToolPoolOptional:
    """x-nous-tool-pool-* headers are optional; absent → defaults; present-but-malformed → miss."""

    def _no_tool_pool_headers(self) -> dict:
        """Base headers with BOTH tool-pool headers removed."""
        h = _base_headers()
        h.pop("x-nous-tool-pool-micros", None)
        h.pop("x-nous-tool-pool-gated-off", None)
        return h

    def test_absent_tool_pool_headers_parse_succeeds(self):
        """Valid credits headers with no x-nous-tool-pool-* → parse succeeds."""
        state = parse_credits_headers(self._no_tool_pool_headers())
        assert state is not None

    def test_absent_tool_pool_micros_defaults_to_zero(self):
        state = parse_credits_headers(self._no_tool_pool_headers())
        assert state.tool_pool_micros == 0

    def test_absent_tool_pool_gated_off_defaults_to_false(self):
        state = parse_credits_headers(self._no_tool_pool_headers())
        assert state.tool_pool_gated_off is False

    def test_present_malformed_tool_pool_micros_returns_none(self):
        """x-nous-tool-pool-micros present but non-int → parse miss (returns None)."""
        headers = _base_headers(**{"x-nous-tool-pool-micros": "not-a-number"})
        assert parse_credits_headers(headers) is None

    def test_present_negative_tool_pool_micros_returns_none(self):
        """x-nous-tool-pool-micros present but negative → parse miss (returns None)."""
        headers = _base_headers(**{"x-nous-tool-pool-micros": "-1000"})
        assert parse_credits_headers(headers) is None

    def test_only_tool_pool_micros_absent_still_succeeds(self):
        """Only micros absent (gated-off still present) → tool_pool_micros = 0, parse succeeds."""
        h = _base_headers()
        h.pop("x-nous-tool-pool-micros", None)
        state = parse_credits_headers(h)
        assert state is not None
        assert state.tool_pool_micros == 0


# ── Half-pair subscription limit ─────────────────────────────────────────────


class TestHalfPairLimit:
    def test_only_limit_micros_present_both_absent(self):
        """Only -micros present → both None, parse SUCCEEDS."""
        headers = _base_headers(
            **{
                "x-nous-credits-subscription-limit-micros": micros(20.00),
                "x-nous-credits-denominator-kind": "subscription_cap",
            }
        )
        state = parse_credits_headers(headers)
        assert state is not None
        assert state.subscription_limit_micros is None
        assert state.subscription_limit_usd is None

    def test_only_limit_usd_present_both_absent(self):
        """Only -usd present → both None, parse SUCCEEDS."""
        headers = _base_headers(
            **{
                "x-nous-credits-subscription-limit-usd": "20.00",
                "x-nous-credits-denominator-kind": "subscription_cap",
            }
        )
        state = parse_credits_headers(headers)
        assert state is not None
        assert state.subscription_limit_micros is None
        assert state.subscription_limit_usd is None

    def test_half_pair_used_fraction_is_none(self):
        """With no limit pair, used_fraction is None regardless of denominator_kind."""
        headers = _base_headers(
            **{
                "x-nous-credits-subscription-limit-micros": micros(20.00),
                "x-nous-credits-denominator-kind": "subscription_cap",
            }
        )
        state = parse_credits_headers(headers)
        assert state is not None
        assert state.used_fraction is None

    def test_full_pair_present_parsed_correctly(self):
        """Both present → both populated, used_fraction computable."""
        headers = _base_headers(
            **{
                "x-nous-credits-subscription-micros": micros(10.00),
                "x-nous-credits-subscription-usd": "10.00",
                "x-nous-credits-subscription-limit-micros": micros(20.00),
                "x-nous-credits-subscription-limit-usd": "20.00",
                "x-nous-credits-denominator-kind": "subscription_cap",
            }
        )
        state = parse_credits_headers(headers)
        assert state is not None
        assert state.subscription_limit_micros == round(20.00 * 1_000_000)
        assert state.subscription_limit_usd == "20.00"
        assert state.used_fraction == pytest.approx(0.50)


# ── Negative value validation ─────────────────────────────────────────────────


class TestNegativeValues:
    def test_negative_remaining_micros_returns_none(self):
        headers = _base_headers(**{"x-nous-credits-remaining-micros": "-1000"})
        assert parse_credits_headers(headers) is None

    def test_negative_purchased_micros_returns_none(self):
        headers = _base_headers(**{"x-nous-credits-purchased-micros": "-500"})
        assert parse_credits_headers(headers) is None

    def test_negative_rollover_micros_returns_none(self):
        headers = _base_headers(**{"x-nous-credits-rollover-micros": "-100"})
        assert parse_credits_headers(headers) is None

    def test_negative_limit_micros_returns_none(self):
        headers = _base_headers(
            **{
                "x-nous-credits-subscription-limit-micros": "-1000",
                "x-nous-credits-subscription-limit-usd": "-0.00",
                "x-nous-credits-denominator-kind": "subscription_cap",
            }
        )
        assert parse_credits_headers(headers) is None

    def test_negative_subscription_accepted(self):
        """subscription_micros is the ONLY field allowed to be negative."""
        headers = _base_headers(**{"x-nous-credits-subscription-micros": "-5000000",
                                   "x-nous-credits-subscription-usd": "-5.00"})
        state = parse_credits_headers(headers)
        assert state is not None
        assert state.subscription_micros == -5_000_000


# ── USD format validation ─────────────────────────────────────────────────────


class TestUsdValidation:
    def test_valid_usd_format(self):
        headers = _base_headers(**{"x-nous-credits-remaining-usd": "18.00"})
        state = parse_credits_headers(headers)
        assert state is not None
        assert state.remaining_usd == "18.00"

    def test_usd_one_decimal_returns_none(self):
        """'18.0' does not match ^-?\d+\.\d{2}$"""
        headers = _base_headers(**{"x-nous-credits-remaining-usd": "18.0"})
        assert parse_credits_headers(headers) is None

    def test_usd_no_decimal_returns_none(self):
        headers = _base_headers(**{"x-nous-credits-remaining-usd": "18"})
        assert parse_credits_headers(headers) is None

    def test_usd_with_dollar_sign_returns_none(self):
        headers = _base_headers(**{"x-nous-credits-remaining-usd": "$18.00"})
        assert parse_credits_headers(headers) is None

    def test_usd_with_comma_returns_none(self):
        headers = _base_headers(**{"x-nous-credits-remaining-usd": "1,800.00"})
        assert parse_credits_headers(headers) is None

    def test_usd_negative_valid(self):
        """Negative USD string should parse (e.g. subscription debt)."""
        headers = _base_headers(
            **{
                "x-nous-credits-subscription-micros": "-5000000",
                "x-nous-credits-subscription-usd": "-5.00",
            }
        )
        state = parse_credits_headers(headers)
        assert state is not None
        assert state.subscription_usd == "-5.00"


# ── Non-int micros validation ─────────────────────────────────────────────────


class TestMicrosValidation:
    def test_non_int_micros_string_returns_none(self):
        headers = _base_headers(**{"x-nous-credits-remaining-micros": "abc"})
        assert parse_credits_headers(headers) is None

    def test_float_string_micros_returns_none(self):
        """'1.5' is not an integer string — should fail validation."""
        headers = _base_headers(**{"x-nous-credits-remaining-micros": "1.5"})
        assert parse_credits_headers(headers) is None

    def test_non_int_purchased_returns_none(self):
        headers = _base_headers(**{"x-nous-credits-purchased-micros": "abc"})
        assert parse_credits_headers(headers) is None


# ── as_of_ms validation ───────────────────────────────────────────────────────


class TestAsOfMs:
    def test_junk_as_of_ms_returns_none(self):
        headers = _base_headers(**{"x-nous-credits-as-of-ms": "not-a-timestamp"})
        assert parse_credits_headers(headers) is None

    def test_valid_as_of_ms(self):
        headers = _base_headers(**{"x-nous-credits-as-of-ms": "1717000000000"})
        state = parse_credits_headers(headers)
        assert state is not None
        assert state.as_of_ms == 1717000000000


# ── denominator_kind validation ────────────────────────────────────────────────


class TestDenominatorKind:
    def test_subscription_cap_valid(self):
        headers = _base_headers(
            **{
                "x-nous-credits-denominator-kind": "subscription_cap",
                "x-nous-credits-subscription-limit-micros": micros(20.00),
                "x-nous-credits-subscription-limit-usd": "20.00",
            }
        )
        state = parse_credits_headers(headers)
        assert state is not None
        assert state.denominator_kind == "subscription_cap"

    def test_none_valid(self):
        headers = _base_headers(**{"x-nous-credits-denominator-kind": "none"})
        state = parse_credits_headers(headers)
        assert state is not None
        assert state.denominator_kind == "none"

    def test_invalid_denominator_kind_returns_none(self):
        headers = _base_headers(**{"x-nous-credits-denominator-kind": "invalid_kind"})
        assert parse_credits_headers(headers) is None


# ── Zero-division guard ────────────────────────────────────────────────────────


class TestZeroDivisionGuard:
    def test_subscription_limit_zero_used_fraction_is_none(self):
        """subscription_limit_micros='0' + subscription_cap → used_fraction is None (no ZeroDivisionError)."""
        headers = _base_headers(
            **{
                "x-nous-credits-subscription-limit-micros": "0",
                "x-nous-credits-subscription-limit-usd": "0.00",
                "x-nous-credits-denominator-kind": "subscription_cap",
            }
        )
        state = parse_credits_headers(headers)
        assert state is not None
        # limit == 0, so used_fraction must be None (guard prevents division)
        assert state.used_fraction is None


# ── Unknown headers ignored ────────────────────────────────────────────────────


class TestUnknownHeaders:
    def test_unknown_extra_header_ignored(self):
        headers = {
            **_base_headers(),
            "x-nous-credits-future-field": "some-value",
            "x-request-id": "abc123",
        }
        state = parse_credits_headers(headers)
        assert state is not None

    def test_mixed_with_other_providers_headers(self):
        headers = {
            **_base_headers(),
            "x-ratelimit-limit-requests": "800",
            "content-type": "application/json",
        }
        state = parse_credits_headers(headers)
        assert state is not None


# ── Header normalization ──────────────────────────────────────────────────────


class TestHeaderNormalization:
    def test_uppercase_headers_parsed(self):
        headers = {k.upper(): v for k, v in _base_headers().items()}
        state = parse_credits_headers(headers)
        assert state is not None

    def test_mixed_case_headers_parsed(self):
        headers = {
            "X-Nous-Credits-Version": "1",
            "X-Nous-Credits-Remaining-Micros": micros(5.00),
            "X-Nous-Credits-Remaining-Usd": "5.00",
            "X-Nous-Credits-Subscription-Micros": micros(5.00),
            "X-Nous-Credits-Subscription-Usd": "5.00",
            "X-Nous-Credits-Rollover-Micros": "0",
            "X-Nous-Credits-Purchased-Micros": "0",
            "X-Nous-Credits-Purchased-Usd": "0.00",
            "X-Nous-Tool-Pool-Micros": "0",
            "X-Nous-Tool-Pool-Gated-Off": "false",
            "X-Nous-Credits-Denominator-Kind": "none",
            "X-Nous-Credits-Paid-Access": "true",
            "X-Nous-Credits-As-Of-Ms": "1717000000000",
        }
        state = parse_credits_headers(headers)
        assert state is not None
        assert state.remaining_micros == round(5.00 * 1_000_000)


# ── CreditsState dataclass defaults ──────────────────────────────────────────


class TestCreditsStateDefaults:
    def test_default_state(self):
        state = CreditsState()
        assert state.version == 0
        assert state.remaining_micros == 0
        assert state.remaining_usd == ""
        assert state.subscription_micros == 0
        assert state.subscription_usd == ""
        assert state.subscription_limit_micros is None
        assert state.subscription_limit_usd is None
        assert state.rollover_micros == 0
        assert state.purchased_micros == 0
        assert state.purchased_usd == ""
        assert state.tool_pool_micros == 0
        assert state.tool_pool_gated_off is False
        assert state.denominator_kind == "none"
        assert state.paid_access is True
        assert state.disabled_reason is None
        assert state.as_of_ms == 0
        assert state.captured_at == 0.0
        assert state.from_header is False

    def test_has_data_false_when_no_captured_at(self):
        state = CreditsState()
        assert state.has_data is False

    def test_age_seconds_inf_when_no_data(self):
        state = CreditsState()
        assert state.age_seconds == float("inf")

    def test_depleted_false_by_default(self):
        state = CreditsState()
        assert state.depleted is False

    def test_used_fraction_none_by_default(self):
        state = CreditsState()
        assert state.used_fraction is None


# ── depleted property ─────────────────────────────────────────────────────────


class TestDepletedProperty:
    def test_depleted_equals_not_paid_access(self):
        """depleted must be exactly `not paid_access`, never `remaining==0`."""
        state = CreditsState(paid_access=False, remaining_micros=0, captured_at=time.time())
        assert state.depleted is True

    def test_not_depleted_when_paid_access_true(self):
        state = CreditsState(paid_access=True, remaining_micros=0, captured_at=time.time())
        # remaining==0 but paid_access is True → NOT depleted
        assert state.depleted is False

    def test_depleted_independent_of_remaining(self):
        """Even with remaining > 0, if paid_access is False, depleted is True."""
        state = CreditsState(paid_access=False, remaining_micros=1_000_000, captured_at=time.time())
        assert state.depleted is True


# ── used_fraction edge cases ──────────────────────────────────────────────────


class TestUsedFraction:
    def test_none_without_limit(self):
        state = CreditsState(
            denominator_kind="subscription_cap",
            subscription_limit_micros=None,
            captured_at=time.time(),
        )
        assert state.used_fraction is None

    def test_none_when_limit_zero(self):
        state = CreditsState(
            denominator_kind="subscription_cap",
            subscription_limit_micros=0,
            subscription_micros=0,
            captured_at=time.time(),
        )
        assert state.used_fraction is None

    def test_clamped_at_zero(self):
        """If subscription_micros > limit (over-credited), fraction clamps to 0."""
        state = CreditsState(
            denominator_kind="subscription_cap",
            subscription_limit_micros=10_000_000,
            subscription_micros=15_000_000,  # more than limit
            captured_at=time.time(),
        )
        assert state.used_fraction == pytest.approx(0.0)

    def test_clamped_at_one(self):
        """If subscription_micros is very negative (debt), fraction clamps to 1.0."""
        state = CreditsState(
            denominator_kind="subscription_cap",
            subscription_limit_micros=10_000_000,
            subscription_micros=-5_000_000,  # deep debt
            captured_at=time.time(),
        )
        assert state.used_fraction == pytest.approx(1.0)

    def test_guarded_by_limit_field_not_denominator(self):
        """used_fraction depends on subscription_limit_micros being truthy, not denominator_kind."""
        # limit present but denominator_kind="none" — spec says guard on LIMIT FIELD
        state = CreditsState(
            denominator_kind="none",
            subscription_limit_micros=20_000_000,
            subscription_micros=10_000_000,
            captured_at=time.time(),
        )
        # With limit_micros set, fraction should be computable regardless of denominator_kind
        assert state.used_fraction == pytest.approx(0.50)

    def test_none_when_denominator_cap_but_no_limit(self):
        """denominator_kind=subscription_cap but no limit pair → None."""
        state = CreditsState(
            denominator_kind="subscription_cap",
            subscription_limit_micros=None,
            subscription_micros=5_000_000,
            captured_at=time.time(),
        )
        assert state.used_fraction is None
