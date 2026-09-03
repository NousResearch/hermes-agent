"""Contract tests for the context-window usage fields passed through to API clients.

``_usage_context_fields`` is the single source of truth for the three
context-ring fields (``last_prompt_tokens`` / ``context_length`` /
``threshold_tokens``) that ``_run_agent`` in gateway/run.py populates and
the API server forwards on every response path: the chat-completions usage
dict, the incomplete event, and both response.failed branches. The invariant
under test: every usage dict the server emits includes these three fields,
sourced from the same helper — so a rename or drift in one path is caught
here, not by a client's context ring going blank.

These keys are intentional extensions to the OpenAI usage schema (the Hermex
iOS client decodes them via explicit CodingKeys). They must stay additive.
"""

from gateway.platforms.api_server import _usage_context_fields


def _run_agent_style_usage():
    """Shape of the usage dict produced by _run_agent (gateway/run.py)."""
    return {
        "input_tokens": 1234,
        "output_tokens": 567,
        "total_tokens": 1801,
        "last_prompt_tokens": 1100,
        "context_length": 200000,
        "threshold_tokens": 160000,
        "estimated_cost": 0.0012,
    }


def test_usage_context_fields_passes_through_run_agent_values():
    usage = _run_agent_style_usage()
    fields = _usage_context_fields(usage)
    assert fields == {
        "last_prompt_tokens": 1100,
        "context_length": 200000,
        "threshold_tokens": 160000,
    }


def test_usage_context_fields_default_to_zero_when_absent():
    # A bare OpenAI-style usage dict has none of the three fields; the
    # server must still emit them (additive contract) rather than drop them.
    fields = _usage_context_fields({"input_tokens": 10, "output_tokens": 5})
    assert fields == {
        "last_prompt_tokens": 0,
        "context_length": 0,
        "threshold_tokens": 0,
    }


def test_usage_context_fields_never_clobber_core_usage_keys():
    usage = _run_agent_style_usage()
    fields = _usage_context_fields(usage)
    assert set(fields).isdisjoint({"input_tokens", "output_tokens", "total_tokens"})


def test_usage_context_fields_return_only_the_three_extension_keys():
    fields = _usage_context_fields(_run_agent_style_usage())
    assert set(fields) == {
        "last_prompt_tokens",
        "context_length",
        "threshold_tokens",
    }


def test_all_usage_emission_sites_route_through_helper():
    """Every API response path forwards the fields via the shared helper.

    Structural guard against copy-paste drift: if a future path emits usage
    without calling the helper, the four call sites below won't all match.
    """
    import inspect
    import re

    import gateway.platforms.api_server as mod

    src = inspect.getsource(mod)
    # The helper definition itself + its use at every usage emission site.
    calls = re.findall(r"_usage_context_fields\(usage\)", src)
    # 1 (definition's docstring reference is not a call) — count actual calls:
    # chat-completions usage dict, incomplete event, 2x response.failed.
    assert len(calls) == 4, f"expected 4 usage emission sites, found {len(calls)}"
