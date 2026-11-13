"""Per-model reasoning-effort vocabulary declared via ``model_overrides`` (#154352).

A custom OpenAI-compatible endpoint's effort vocabulary is undiscoverable, so the
entry clamp passes the widest wire set and a model that rejects ``max``/``ultra``
answers HTTP 400. An explicit
``model_overrides.<provider>.<model>.supported_reasoning_efforts`` declaration
names the model's real ceiling; the entry clamp (chat-completions and Responses),
the auxiliary-client clamp and the desktop "(sends X on this route)" label all
consult it, and — consistent with the rest of ``model_overrides`` — an explicit
declaration wins over catalog and built-in ladders.

Explicit-only by design (mirrors ``_override_context_window``): a ``_default``
fill-gap must not silently cap every unknown model's reasoning vocabulary.
"""

import logging
from unittest.mock import patch

import pytest

from agent.reasoning_effort import (
    OPENAI_COMPAT_WIRE_EFFORTS,
    chat_endpoint_efforts,
    clamp_effort,
    route_supported_efforts,
)

QWEN_CAP = ("low", "medium", "high", "xhigh")
_OVERRIDES = {"bailian": {"qwen3.6-plus": {"supported_reasoning_efforts": list(QWEN_CAP)}}}


@pytest.fixture
def declared_overrides():
    with patch("agent.models_dev._load_model_overrides", return_value=_OVERRIDES):
        yield _OVERRIDES


class TestDeclaredSupportedEfforts:
    def test_explicit_declaration_resolves(self, declared_overrides):
        from agent.models_dev import declared_supported_efforts

        assert declared_supported_efforts("bailian", "qwen3.6-plus") == QWEN_CAP

    def test_custom_prefixed_provider_matches_bare_config_key(self, declared_overrides):
        """Dashboard/model-switch surfaces pass ``custom:<name>``; users key the bare name."""
        from agent.models_dev import declared_supported_efforts

        assert declared_supported_efforts("custom:bailian", "qwen3.6-plus") == QWEN_CAP

    def test_unknown_model_or_provider_is_none(self, declared_overrides):
        from agent.models_dev import declared_supported_efforts

        assert declared_supported_efforts("bailian", "other-model") is None
        assert declared_supported_efforts("other-provider", "qwen3.6-plus") is None

    def test_default_fill_gap_does_not_apply(self):
        """``_default`` entries never declare an effort vocabulary (explicit-only key)."""
        from agent.models_dev import declared_supported_efforts

        overrides = {"_default": {"supported_reasoning_efforts": list(QWEN_CAP)}}
        with patch("agent.models_dev._load_model_overrides", return_value=overrides):
            assert declared_supported_efforts("bailian", "qwen3.6-plus") is None

    @pytest.mark.parametrize(
        "bad",
        [
            ["low", "turbo"],          # non-ladder level
            [],                        # empty list
            "xhigh",                   # not a list
        ],
    )
    def test_invalid_declaration_ignored_with_warning(self, bad, caplog):
        from agent.models_dev import declared_supported_efforts

        overrides = {"bailian": {"qwen3.6-plus": {"supported_reasoning_efforts": bad}}}
        with patch("agent.models_dev._load_model_overrides", return_value=overrides), \
                caplog.at_level(logging.WARNING, logger="agent.models_dev"):
            assert declared_supported_efforts("bailian", "qwen3.6-plus") is None
        assert any("supported_reasoning_efforts" in r.getMessage() for r in caplog.records)

    def test_valid_declaration_dedupes_preserving_order(self):
        from agent.models_dev import declared_supported_efforts

        overrides = {"bailian": {"m": {"supported_reasoning_efforts": ["high", "low", "high"]}}}
        with patch("agent.models_dev._load_model_overrides", return_value=overrides):
            assert declared_supported_efforts("bailian", "m") == ("high", "low")


class TestEntryClampHonorsDeclaration:
    def test_route_supported_efforts_returns_declared_set(self, declared_overrides):
        assert route_supported_efforts("bailian", "qwen3.6-plus") == QWEN_CAP
        assert route_supported_efforts("custom:bailian", "qwen3.6-plus") == QWEN_CAP

    def test_declared_set_beats_widest_wire_default(self, declared_overrides):
        """Without the declaration the route answers the widest OpenAI-compat set (regression guard)."""
        assert route_supported_efforts("bailian", "qwen3.6-plus") != OPENAI_COMPAT_WIRE_EFFORTS
        with patch("agent.models_dev._load_model_overrides", return_value={}):
            assert route_supported_efforts("bailian", "qwen3.6-plus") == OPENAI_COMPAT_WIRE_EFFORTS

    def test_explicit_declaration_wins_over_codex_ladder(self):
        """model_overrides is the explicit-wins layer, including over built-in per-model ladders."""
        overrides = {"openai-codex": {"gpt-5.6-sol": {"supported_reasoning_efforts": list(QWEN_CAP)}}}
        with patch("agent.models_dev._load_model_overrides", return_value=overrides):
            assert route_supported_efforts("openai-codex", "gpt-5.6-sol") == QWEN_CAP

    def test_clamp_lands_on_declared_ceiling(self, declared_overrides):
        assert clamp_effort("max", route_supported_efforts("bailian", "qwen3.6-plus")) == "xhigh"
        assert clamp_effort("ultra", route_supported_efforts("bailian", "qwen3.6-plus")) == "xhigh"
        assert clamp_effort("high", route_supported_efforts("bailian", "qwen3.6-plus")) == "high"

    def test_chat_endpoint_efforts_threads_provider(self, declared_overrides):
        assert chat_endpoint_efforts("qwen3.6-plus", "https://relay.example/v1", provider="bailian") == QWEN_CAP
        # provider omitted → today's behaviour (widest wire set), no declaration consulted
        assert chat_endpoint_efforts("qwen3.6-plus", "https://relay.example/v1") == OPENAI_COMPAT_WIRE_EFFORTS

    def test_responses_transport_profile_lookup_consults_declaration(self, declared_overrides):
        from agent.transports.codex import _profile_declared_efforts

        assert _profile_declared_efforts("custom:bailian", "qwen3.6-plus", None) == QWEN_CAP
