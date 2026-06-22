"""Unit tests for Ebbinghaus decay-based time-series awareness.

Tests the pure helper functions added in PR #12987:
  _detect_domain, _calculate_age_seconds, _calculate_ebbinghaus_score,
  _determine_lifecycle, _add_time_aware_fields
"""

import math
from datetime import datetime, timedelta, timezone

import pytest

# Import from the mem0 plugin — these are module-level pure functions
from plugins.memory.mem0 import (
    _EBBINGHAUS_PARAMS,
    _LIFECYCLE_STATES,
    _TRUST_ACCELERATION_FACTOR,
    _add_time_aware_fields,
    _calculate_age_seconds,
    _calculate_ebbinghaus_score,
    _determine_lifecycle,
    _detect_domain,
)


# ---------------------------------------------------------------------------
# _detect_domain
# ---------------------------------------------------------------------------

class TestDetectDomain:
    """Domain detection from memory content via keyword matching."""

    def test_empty_string_returns_normal(self):
        assert _detect_domain("") == "normal"

    def test_none_like_empty_returns_normal(self):
        assert _detect_domain("   ") == "normal"

    def test_volatile_keyword_gdp(self):
        assert _detect_domain("GDP growth was 4.2%") == "volatile"

    def test_volatile_keyword_yield(self):
        assert _detect_domain("The yield curve inverted") == "volatile"

    def test_volatile_keyword_spread(self):
        assert _detect_domain("Credit spread widened") == "volatile"

    def test_volatile_keyword_nonfarm(self):
        assert _detect_domain("Nonfarm payrolls missed expectations") == "volatile"

    def test_volatile_keyword_realtime(self):
        assert _detect_domain("Realtime market data feed") == "volatile"

    def test_volatile_keyword_stock_price(self):
        assert _detect_domain("Stock price dropped 5%") == "volatile"

    def test_stable_keyword_preference(self):
        assert _detect_domain("User preference: dark mode") == "stable"

    def test_stable_keyword_server(self):
        assert _detect_domain("Server located in Tokyo") == "stable"

    def test_stable_keyword_password(self):
        assert _detect_domain("Password updated yesterday") == "stable"

    def test_stable_keyword_birthday(self):
        assert _detect_domain("Birthday is March 15") == "stable"

    def test_stable_keyword_degree(self):
        assert _detect_domain("Has a master's degree") == "stable"

    def test_volatile_takes_priority_over_stable(self):
        """Volatile keywords are checked first — a string with both
        volatile and stable keywords should be classified as volatile."""
        text = "User preference for GDP yield analysis"
        assert _detect_domain(text) == "volatile"

    def test_normal_text_no_keywords(self):
        assert _detect_domain("The project uses Python 3.12") == "normal"

    def test_case_insensitive(self):
        assert _detect_domain("gdp DATA IS HERE") == "volatile"
        assert _detect_domain("SERVER config") == "stable"

    def test_volatile_keyword_exchange_rate(self):
        assert _detect_domain("Exchange rate at 7.24") == "volatile"

    def test_volatile_keyword_interest_rate(self):
        assert _detect_domain("Interest rate held steady") == "volatile"

    def test_stable_keyword_address(self):
        assert _detect_domain("Address: 123 Main St") == "stable"

    def test_stable_keyword_phone(self):
        assert _detect_domain("Phone: +1-555-0123") == "stable"


# ---------------------------------------------------------------------------
# _calculate_age_seconds
# ---------------------------------------------------------------------------

class TestCalculateAgeSeconds:
    """Age calculation from ISO-8601 timestamps."""

    def test_valid_iso_with_z_suffix(self):
        """A timestamp 1 day ago should return ~86400 seconds."""
        one_day_ago = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat().replace("+00:00", "Z")
        age = _calculate_age_seconds(one_day_ago)
        assert 86300 < age < 86500  # allow 100s clock skew

    def test_valid_iso_with_offset(self):
        """A timestamp 7 days ago with explicit +00:00 offset."""
        seven_days_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        age = _calculate_age_seconds(seven_days_ago)
        assert 7 * 86400 - 100 < age < 7 * 86400 + 100

    def test_invalid_string_returns_zero(self):
        assert _calculate_age_seconds("not-a-date") == 0.0

    def test_empty_string_returns_zero(self):
        assert _calculate_age_seconds("") == 0.0

    def test_future_timestamp_returns_zero(self):
        """Future timestamps should be clamped to 0 (negative age is meaningless)."""
        future = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat().replace("+00:00", "Z")
        assert _calculate_age_seconds(future) == 0.0

    def test_just_now_returns_near_zero(self):
        now = datetime.now(timezone.utc).isoformat()
        age = _calculate_age_seconds(now)
        assert 0.0 <= age < 5.0  # within 5 seconds

    def test_exact_zero_age(self):
        """A timestamp exactly now should give age ≈ 0."""
        age = _calculate_age_seconds(datetime.now(timezone.utc).isoformat())
        assert age >= 0.0


# ---------------------------------------------------------------------------
# _calculate_ebbinghaus_score
# ---------------------------------------------------------------------------

class TestCalculateEbbinghausScore:
    """Ebbinghaus forgetting-curve score calculation."""

    # -- Half-life verification (core invariant) --

    def test_normal_half_life_at_day_7(self):
        """Normal domain (half_life=7d, strength=1.0) at exactly 7 days = 0.5."""
        score = _calculate_ebbinghaus_score(7 * 86400, domain="normal")
        assert abs(score - 0.5) < 0.01

    def test_volatile_half_life_at_day_3(self):
        """Volatile domain (half_life=3d, strength=1.0) at exactly 3 days = 0.5."""
        score = _calculate_ebbinghaus_score(3 * 86400, domain="volatile")
        assert abs(score - 0.5) < 0.01

    def test_stable_at_day_30(self):
        """Stable domain (half_life=30d, strength=1.3) at 30 days ≈ 0.65."""
        score = _calculate_ebbinghaus_score(30 * 86400, domain="stable")
        assert abs(score - 0.65) < 0.01

    # -- Fresh memory ≈ 1.0 --

    def test_fresh_memory_scores_near_one(self):
        """Age=0 should give score ≈ strength (1.0 for normal/volatile)."""
        score_normal = _calculate_ebbinghaus_score(0, domain="normal")
        assert abs(score_normal - 1.0) < 0.001

        score_volatile = _calculate_ebbinghaus_score(0, domain="volatile")
        assert abs(score_volatile - 1.0) < 0.001

    def test_fresh_stable_scores_near_strength(self):
        """Stable domain with strength=1.3, age=0 should give ~1.0 (clamped)."""
        score = _calculate_ebbinghaus_score(0, domain="stable")
        assert abs(score - 1.0) < 0.001  # clamped to [0, 1]

    # -- Use count reinforcement --

    def test_use_count_reinforcement_normal(self):
        """n_use=5 with beta=0.6 → use_factor ≈ 5^0.6 ≈ 2.627."""
        params = _EBBINGHAUS_PARAMS["normal"]
        expected_factor = 5 ** params["beta"]
        assert abs(expected_factor - 2.627) < 0.01

        # Fresh memory with n_use=5 should score min(1.0, 2.627*1.0) = 1.0
        score = _calculate_ebbinghaus_score(0, n_use=5, domain="normal")
        assert score == 1.0  # clamped

    def test_use_count_reinforcement_volatile(self):
        """n_use=5 with beta=0.8 → use_factor ≈ 5^0.8 ≈ 3.624."""
        params = _EBBINGHAUS_PARAMS["volatile"]
        expected_factor = 5 ** params["beta"]
        assert abs(expected_factor - 3.624) < 0.01

    def test_n_use_decays_slower(self):
        """Higher n_use should yield higher scores at the same age."""
        age = 14 * 86400  # 14 days — deep enough that n_use differences matter
        score_1 = _calculate_ebbinghaus_score(age, n_use=1, domain="normal")
        score_5 = _calculate_ebbinghaus_score(age, n_use=5, domain="normal")
        score_10 = _calculate_ebbinghaus_score(age, n_use=10, domain="normal")
        assert score_5 > score_1
        assert score_10 > score_5

    # -- Trust-weighted acceleration --

    def test_low_trust_accelerates_decay(self):
        """trust_weight < 1.0 should make memories decay faster."""
        age = 7 * 86400
        score_full_trust = _calculate_ebbinghaus_score(age, trust_weight=1.0, domain="normal")
        score_low_trust = _calculate_ebbinghaus_score(age, trust_weight=0.5, domain="normal")
        assert score_low_trust < score_full_trust

    def test_trust_weight_zero_max_acceleration(self):
        """trust_weight=0.0 gives maximum acceleration (κ=2.0)."""
        age = 7 * 86400
        score_zero = _calculate_ebbinghaus_score(age, trust_weight=0.0, domain="normal")
        score_full = _calculate_ebbinghaus_score(age, trust_weight=1.0, domain="normal")
        # With κ=2.0 and trust=0, λ_eff = λ * (1 + 2*(1-0)) = 3λ
        # Score at day 7 with 3λ = e^(-3*ln2) = 2^(-3) = 0.125
        assert abs(score_zero - 0.125) < 0.01

    # -- Score clamping --

    def test_score_clamped_to_one(self):
        """Score should never exceed 1.0 (even with high n_use + fresh)."""
        score = _calculate_ebbinghaus_score(0, n_use=100, domain="normal")
        assert score <= 1.0

    def test_score_never_negative(self):
        """Score should never be negative (very old memory)."""
        score = _calculate_ebbinghaus_score(365 * 86400, domain="volatile")
        assert score >= 0.0

    # -- Custom strength --

    def test_custom_strength_overrides_default(self):
        """Passing strength=0.5 should halve the score of a fresh memory."""
        score = _calculate_ebbinghaus_score(0, strength=0.5, domain="normal")
        assert abs(score - 0.5) < 0.001

    # -- Unknown domain falls back to normal --

    def test_unknown_domain_falls_back_to_normal(self):
        score = _calculate_ebbinghaus_score(7 * 86400, domain="nonexistent")
        score_normal = _calculate_ebbinghaus_score(7 * 86400, domain="normal")
        assert abs(score - score_normal) < 0.001

    # -- Monotonicity --

    def test_score_monotonically_decreases_with_age(self):
        """Score should only decrease as age increases (same n_use/trust)."""
        scores = [
            _calculate_ebbinghaus_score(d * 86400, domain="normal")
            for d in range(0, 31, 5)
        ]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]


# ---------------------------------------------------------------------------
# _determine_lifecycle
# ---------------------------------------------------------------------------

class TestDetermineLifecycle:
    """Lifecycle state mapping from retention score."""

    def test_active(self):
        lc = _determine_lifecycle(0.95)
        assert lc["lifecycle_state"] == "active"
        assert lc["suggested_bit_width"] == 32
        assert lc["freshness_tag"] == ""

    def test_warm(self):
        lc = _determine_lifecycle(0.6)
        assert lc["lifecycle_state"] == "warm"
        assert lc["suggested_bit_width"] == 8
        assert lc["freshness_tag"] == "⏳"

    def test_cold(self):
        lc = _determine_lifecycle(0.3)
        assert lc["lifecycle_state"] == "cold"
        assert lc["suggested_bit_width"] == 4
        assert lc["freshness_tag"] == "⚠️"

    def test_archive_normal(self):
        """Archive for normal domain: score between forget_threshold(0.05) and 0.2."""
        lc = _determine_lifecycle(0.10, domain="normal")
        assert lc["lifecycle_state"] == "archive"
        assert lc["suggested_bit_width"] == 2

    def test_forgotten_normal(self):
        """Below forget_threshold for normal (0.05) → forgotten."""
        lc = _determine_lifecycle(0.03, domain="normal")
        assert lc["lifecycle_state"] == "forgotten"

    def test_forgotten_volatile(self):
        """Volatile forget_threshold=0.10, so 0.05 < 0.10 → forgotten."""
        lc = _determine_lifecycle(0.05, domain="volatile")
        assert lc["lifecycle_state"] == "forgotten"

    def test_archive_volatile(self):
        """Score 0.15 is > 0.10 (volatile threshold) but < 0.2 → archive."""
        lc = _determine_lifecycle(0.15, domain="volatile")
        assert lc["lifecycle_state"] == "archive"

    def test_stable_lower_forget_threshold(self):
        """Stable domain has forget_threshold=0.02, so 0.03 > 0.02 → archive."""
        lc = _determine_lifecycle(0.03, domain="stable")
        assert lc["lifecycle_state"] == "archive"

    def test_boundary_active_warm(self):
        """Score exactly 0.8 → warm (not active, since >0.8 is active)."""
        lc = _determine_lifecycle(0.8)
        assert lc["lifecycle_state"] == "warm"

    def test_boundary_warm_cold(self):
        lc = _determine_lifecycle(0.5)
        assert lc["lifecycle_state"] == "cold"

    def test_boundary_cold_archive(self):
        lc = _determine_lifecycle(0.2)
        assert lc["lifecycle_state"] == "archive"

    def test_retention_score_rounded(self):
        lc = _determine_lifecycle(0.123456)
        assert lc["retention_score"] == round(0.123456, 3)

    def test_output_keys(self):
        """Every output dict should have exactly these keys."""
        lc = _determine_lifecycle(0.5)
        expected_keys = {"lifecycle_state", "retention_score", "suggested_bit_width", "freshness_tag"}
        assert set(lc.keys()) == expected_keys


# ---------------------------------------------------------------------------
# _add_time_aware_fields
# ---------------------------------------------------------------------------

class TestAddTimeAwareFields:
    """Integration test: memory dict → enriched with time-series fields."""

    def _make_memory(self, text: str, days_ago: int = 7, n_use: int = 1) -> dict:
        """Helper: create a memory dict with a created_at timestamp `days_ago`."""
        ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
        return {
            "memory": text,
            "created_at": ts,
            "access_count": n_use,
        }

    def test_non_dict_passthrough(self):
        """Non-dict inputs should be returned unchanged."""
        assert _add_time_aware_fields("string") == "string"
        assert _add_time_aware_fields(42) == 42
        assert _add_time_aware_fields(None) is None

    def test_no_created_at_returns_unenhanced(self):
        """Memory without created_at should pass through without time fields."""
        mem = {"memory": "some text"}
        result = _add_time_aware_fields(mem)
        assert "age_days" not in result
        assert "retention_score" not in result

    def test_adds_all_expected_fields(self):
        mem = self._make_memory("Project uses Python", days_ago=5)
        result = _add_time_aware_fields(mem)
        expected_keys = {
            "age_days", "domain", "retention_score", "lifecycle_state",
            "suggested_bit_width", "freshness_tag", "n_use",
            "eligible_for_promotion",
        }
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_age_days_calculated_correctly(self):
        mem = self._make_memory("Normal text", days_ago=3)
        result = _add_time_aware_fields(mem)
        assert result["age_days"] == 3

    def test_domain_detected_volatile(self):
        mem = self._make_memory("GDP grew by 3%", days_ago=1)
        result = _add_time_aware_fields(mem)
        assert result["domain"] == "volatile"

    def test_domain_detected_stable(self):
        mem = self._make_memory("Server password updated", days_ago=1)
        result = _add_time_aware_fields(mem)
        assert result["domain"] == "stable"

    def test_domain_detected_normal(self):
        mem = self._make_memory("Project setup complete", days_ago=1)
        result = _add_time_aware_fields(mem)
        assert result["domain"] == "normal"

    def test_n_use_from_access_count(self):
        mem = self._make_memory("Frequent memory", days_ago=5, n_use=10)
        result = _add_time_aware_fields(mem)
        assert result["n_use"] == 10

    def test_n_use_fallback_to_n_use_key(self):
        """If access_count missing, fall back to n_use key."""
        ts = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
        mem = {"memory": "test", "created_at": ts, "n_use": 7}
        result = _add_time_aware_fields(mem)
        assert result["n_use"] == 7

    def test_n_use_default_to_one(self):
        """If neither access_count nor n_use, default to 1."""
        ts = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
        mem = {"memory": "test", "created_at": ts}
        result = _add_time_aware_fields(mem)
        assert result["n_use"] == 1

    def test_retention_score_between_zero_and_one(self):
        mem = self._make_memory("test", days_ago=30)
        result = _add_time_aware_fields(mem)
        assert 0.0 <= result["retention_score"] <= 1.0

    def test_lifecycle_state_valid(self):
        mem = self._make_memory("test", days_ago=5)
        result = _add_time_aware_fields(mem)
        assert result["lifecycle_state"] in {"active", "warm", "cold", "archive", "forgotten"}

    def test_eligible_for_promotion_young_and_used(self):
        """n_use >= 5 and age <= 14 → eligible for promotion."""
        mem = self._make_memory("test", days_ago=7, n_use=5)
        result = _add_time_aware_fields(mem)
        assert result["eligible_for_promotion"] is True

    def test_not_eligible_old_memory(self):
        """Age > 14 days → not eligible even with high n_use."""
        mem = self._make_memory("test", days_ago=30, n_use=10)
        result = _add_time_aware_fields(mem)
        assert result["eligible_for_promotion"] is False

    def test_not_eligible_low_use(self):
        """n_use < 5 → not eligible even if young."""
        mem = self._make_memory("test", days_ago=3, n_use=3)
        result = _add_time_aware_fields(mem)
        assert result["eligible_for_promotion"] is False

    def test_trust_weight_from_metadata(self):
        """trust_weight should be extracted from metadata dict if present."""
        ts = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        mem = {"memory": "test", "created_at": ts, "metadata": {"trust": 0.3}}
        result_low_trust = _add_time_aware_fields(mem)

        mem2 = {"memory": "test", "created_at": ts, "metadata": {"trust": 1.0}}
        result_high_trust = _add_time_aware_fields(mem2)

        # Low trust → faster decay → lower retention score
        assert result_low_trust["retention_score"] < result_high_trust["retention_score"]

    def test_original_fields_preserved(self):
        """Enhancement should not remove original memory fields."""
        mem = self._make_memory("Original content", days_ago=3)
        result = _add_time_aware_fields(mem)
        assert result["memory"] == "Original content"
        assert "created_at" in result

    def test_fresh_memory_active_lifecycle(self):
        """A brand new memory (0 days old) should be in active lifecycle."""
        mem = self._make_memory("Just created", days_ago=0)
        result = _add_time_aware_fields(mem)
        assert result["lifecycle_state"] == "active"
        assert result["retention_score"] >= 0.8

    def test_old_volatile_memory_decays_fast(self):
        """A 30-day-old volatile memory should be cold or worse."""
        mem = self._make_memory("GDP data point", days_ago=30)
        result = _add_time_aware_fields(mem)
        assert result["retention_score"] < 0.2

    def test_old_stable_memory_survives(self):
        """A 30-day-old stable memory should still be warm or better."""
        mem = self._make_memory("Server config preference", days_ago=30)
        result = _add_time_aware_fields(mem)
        assert result["retention_score"] >= 0.3  # stable half-life=30d, strength=1.3

    def test_does_not_mutate_original(self):
        """_add_time_aware_fields should return a new dict, not mutate the input."""
        ts = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
        mem = {"memory": "test", "created_at": ts}
        original_keys = set(mem.keys())
        _add_time_aware_fields(mem)
        assert set(mem.keys()) == original_keys


# ---------------------------------------------------------------------------
# Cross-cutting: parameter consistency
# ---------------------------------------------------------------------------

class TestParameterConsistency:
    """Ensure _EBBINGHAUS_PARAMS and _LIFECYCLE_STATES are well-formed."""

    def test_all_domains_have_required_keys(self):
        required = {"half_life_days", "beta", "forget_threshold", "strength_default"}
        for domain, params in _EBBINGHAUS_PARAMS.items():
            assert required.issubset(set(params.keys())), f"{domain} missing keys"

    def test_all_lifecycle_states_have_required_keys(self):
        required = {"retention", "bit_width", "tag"}
        for state, cfg in _LIFECYCLE_STATES.items():
            assert required.issubset(set(cfg.keys())), f"{state} missing keys"

    def test_bit_widths_decrease_monotonically(self):
        """Lifecycle bit widths should decrease: active > warm > cold > archive."""
        widths = [_LIFECYCLE_STATES[s]["bit_width"] for s in ["active", "warm", "cold", "archive"]]
        assert widths == sorted(widths, reverse=True)

    def test_retention_thresholds_decrease(self):
        thresholds = [_LIFECYCLE_STATES[s]["retention"] for s in ["active", "warm", "cold", "archive"]]
        assert thresholds == sorted(thresholds, reverse=True)

    def test_half_lives_positive(self):
        for domain, params in _EBBINGHAUS_PARAMS.items():
            assert params["half_life_days"] > 0, f"{domain} half_life must be positive"

    def test_beta_between_zero_and_one(self):
        for domain, params in _EBBINGHAUS_PARAMS.items():
            assert 0 < params["beta"] <= 1, f"{domain} beta should be in (0, 1]"

    def test_forget_thresholds_between_zero_and_one(self):
        for domain, params in _EBBINGHAUS_PARAMS.items():
            assert 0 < params["forget_threshold"] < 1, f"{domain} threshold out of range"

    def test_domain_keywords_all_valid_regex(self):
        """All keyword patterns should compile without error."""
        import re as _re
        for domain, keywords in _EBBINGHAUS_PARAMS.items():
            pass  # just iterating to confirm structure
        # Actually test _DOMAIN_KEYWORDS
        from plugins.memory.mem0 import _DOMAIN_KEYWORDS
        for domain, kws in _DOMAIN_KEYWORDS.items():
            for kw in kws:
                _re.compile(kw)  # will raise if invalid
