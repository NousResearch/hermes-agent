"""Provider attribution — the fleet's longest-standing silent failure.

2026-09-05. `session_model_usage` has had `provider_name`, `native_tokens_prompt`,
`native_tokens_cached`, `cache_discount` and `total_cost` since 2026-09-01;
`conversation_loop` has always passed all five through to the writer. Yet **0 of
5,387 rows** carried a real value, because nothing in the transport ever put
them on the response object: `getattr(response, "provider_name", "")` returned
"" every single time, and the docs recorded the feature as working.

That blindness is why DeepInfra overcharging 11x went uncaught — every usage row
said "openrouter", so `cache-hit-watch` averaged a 96.8%-discount host together
with a 0%-discount host and stayed silent.

Measured the same day: the SAME model on the SAME minute cost $0.000572 on Z.AI
and $0.002839 on Novita/GMICloud — 5x, invisible in the ledger.

Every test here has a negative control: each asserts the value is extracted AND
that a response without it stays at the default, so the suite cannot pass by
extracting nothing.
"""
import pytest

from agent.transports.chat_completions import (
    _aggregator_provider_name,
    _aggregator_usage_extras,
    _model_extra,
)
from agent.transports.types import NormalizedResponse, Usage


class _Extra:
    """Stand-in for an OpenAI SDK pydantic model carrying unknown fields."""

    def __init__(self, **fields):
        self.model_extra = dict(fields)
        for k, v in fields.items():
            setattr(self, k, v)


class _Bare:
    """A response object with no extras at all — a direct provider."""

    def __init__(self, **fields):
        for k, v in fields.items():
            setattr(self, k, v)


# ── provider_name ────────────────────────────────────────────────────

def test_provider_name_from_top_level_field():
    assert _aggregator_provider_name(_Extra(provider="Z.AI")) == "Z.AI"


def test_provider_name_from_model_extra_only():
    r = _Bare()
    r.model_extra = {"provider": "Novita"}
    assert _aggregator_provider_name(r) == "Novita"


def test_provider_name_is_stripped():
    assert _aggregator_provider_name(_Extra(provider="  DeepInfra  ")) == "DeepInfra"


def test_provider_name_negative_control_direct_provider():
    """NEGATIVE CONTROL: a response with no provider field must yield "".

    Without this the extraction could "succeed" by inventing a value, and every
    row would be attributed to a host that never served it — worse than $0.
    """
    assert _aggregator_provider_name(_Bare(id="chatcmpl-1")) == ""
    assert _aggregator_provider_name(None) == ""
    assert _aggregator_provider_name(_Extra(provider="")) == ""
    assert _aggregator_provider_name(_Extra(provider=None)) == ""


# ── usage extras ─────────────────────────────────────────────────────

def test_total_cost_from_usage_cost():
    u = _Extra(cost=0.000571975, prompt_tokens=37821)
    assert _aggregator_usage_extras(u)["total_cost"] == pytest.approx(0.000571975)


def test_total_cost_falls_back_to_upstream_inference_cost():
    u = _Extra(cost_details=_Extra(upstream_inference_cost=0.0042))
    assert _aggregator_usage_extras(u)["total_cost"] == pytest.approx(0.0042)


def test_cached_tokens_and_native_prompt():
    u = _Extra(prompt_tokens=37821,
               prompt_tokens_details=_Extra(cached_tokens=37760))
    out = _aggregator_usage_extras(u)
    assert out["native_tokens_cached"] == 37760
    assert out["native_tokens_prompt"] == 37821


def test_cache_discount_is_only_recorded_when_stated():
    """A cache HIT is not a cache SAVING.

    Providers exist that report ~100% cached tokens and bill every one at full
    input rate (Parasail on qwen3.8-27b: 99.8% cached, 0% discount). So the
    discount must never be inferred from the cached-token count — only recorded
    when the aggregator states it.
    """
    stated = _aggregator_usage_extras(_Extra(cache_discount=0.0045312))
    assert stated["cache_discount"] == pytest.approx(0.0045312)

    inferred = _aggregator_usage_extras(
        _Extra(prompt_tokens=1000, prompt_tokens_details=_Extra(cached_tokens=998)))
    assert "cache_discount" not in inferred


def test_usage_extras_negative_control_plain_provider():
    """NEGATIVE CONTROL: a plain OpenAI usage block adds nothing but the split."""
    out = _aggregator_usage_extras(_Bare(prompt_tokens=10, completion_tokens=5,
                                         total_tokens=15))
    assert "total_cost" not in out
    assert "cache_discount" not in out
    assert "native_tokens_cached" not in out
    assert _aggregator_usage_extras(None) == {}


def test_malformed_values_do_not_raise():
    """Accounting must never break a turn. Garbage is dropped, not raised."""
    out = _aggregator_usage_extras(
        _Extra(cost="not-a-number", cache_discount=[], prompt_tokens="x",
               prompt_tokens_details=_Extra(cached_tokens=None)))
    assert "total_cost" not in out
    assert "cache_discount" not in out


def test_model_extra_handles_every_shape():
    assert _model_extra(_Extra(a=1)) == {"a": 1}
    assert _model_extra({"b": 2}) == {"b": 2}
    assert _model_extra(_Bare()) == {}
    assert _model_extra(None) == {}


# ── the contract the writer depends on ───────────────────────────────

def test_defaults_match_what_the_writer_expects():
    """conversation_loop reads these five off the response with getattr
    defaults. If a field is renamed or dropped, attribution silently returns to
    zero with no error — exactly how this bug survived. Pin the names."""
    u = Usage()
    for f in ("native_tokens_prompt", "native_tokens_cached", "cache_discount",
              "total_cost"):
        assert hasattr(u, f), f
    assert NormalizedResponse(content="x", tool_calls=None,
                              finish_reason="stop").provider_name == ""


def test_openrouter_profile_requests_usage_accounting():
    """Extraction is useless if the request never asks for the data.

    OpenRouter returns no `usage.cost` unless `usage: {include: true}` is sent,
    so this assertion is half the fix, not a detail.
    """
    import importlib.util
    from pathlib import Path

    here = Path(__file__).resolve()
    root = next(p for p in here.parents if (p / "plugins").is_dir())
    mod_path = root / "plugins/model-providers/openrouter/__init__.py"
    src = mod_path.read_text()
    assert '"usage"' in src and '"include": True' in src, (
        "the OpenRouter profile must request usage accounting")
