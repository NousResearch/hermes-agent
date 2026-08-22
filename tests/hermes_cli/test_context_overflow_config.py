"""DEFAULT_CONFIG + schema for opt-in ``context_overflow.large_context`` (#86070).

Empty / omitted ``large_context`` must stay compress-only. Do not ship a
default provider or model — that would escalate every user on overflow.
"""

from hermes_cli.config import DEFAULT_CONFIG, _KNOWN_ROOT_KEYS


class TestContextOverflowDefaultConfig:
    def test_default_large_context_is_opt_in_empty(self):
        overflow = DEFAULT_CONFIG["context_overflow"]
        assert isinstance(overflow, dict)
        large = overflow.get("large_context")
        assert isinstance(large, dict)
        # No provider+model pair that would enable the feature.
        assert not (large.get("provider") or "").strip()
        assert not (large.get("model") or "").strip()

    def test_context_overflow_is_a_known_root_key(self):
        assert "context_overflow" in _KNOWN_ROOT_KEYS
