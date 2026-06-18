"""Regression tests for #46527: Copilot primary api_mode resolution must not
honor a stale/incompatible explicit ``api_mode``.

The gateway resolves and snapshots the primary provider/model at startup via
``resolve_runtime_provider`` -> ``_copilot_runtime_api_mode``.  When the
config carries a stale ``api_mode: codex_responses`` (e.g. left over from a
previous Nous/Codex primary) and the primary is later changed to Copilot +
``gpt-5-mini``, the old code honored the stale ``codex_responses`` verbatim.
That dispatches gpt-5-mini through the Responses API — which Copilot serves
on chat completions only — and the response comes back with no reasoning/
thinking content (the symptom in #46527).
"""
from hermes_cli.runtime_provider import _copilot_runtime_api_mode


def test_stale_codex_responses_vetoed_for_gpt5_mini():
    """gpt-5-mini + stale api_mode=codex_responses must resolve to chat_completions."""
    cfg = {"provider": "copilot", "default": "gpt-5-mini", "api_mode": "codex_responses"}
    assert _copilot_runtime_api_mode(cfg, api_key="x") == "chat_completions"


def test_explicit_codex_responses_kept_for_gpt5_variant():
    """gpt-5.4-mini legitimately uses the Responses API on Copilot — an explicit
    codex_responses must still be honored (no over-correction)."""
    cfg = {"provider": "copilot", "default": "gpt-5.4-mini", "api_mode": "codex_responses"}
    assert _copilot_runtime_api_mode(cfg, api_key="x") == "codex_responses"


def test_no_explicit_mode_derives_chat_completions_for_gpt5_mini():
    """Without an explicit api_mode, gpt-5-mini derives chat_completions (unchanged)."""
    cfg = {"provider": "copilot", "default": "gpt-5-mini"}
    assert _copilot_runtime_api_mode(cfg, api_key="x") == "chat_completions"


def test_explicit_chat_completions_unchanged():
    """An explicit chat_completions stays chat_completions for gpt-5-mini."""
    cfg = {"provider": "copilot", "default": "gpt-5-mini", "api_mode": "chat_completions"}
    assert _copilot_runtime_api_mode(cfg, api_key="x") == "chat_completions"


def test_stale_mode_from_different_provider_not_honored():
    """When the config's provider doesn't match copilot, the explicit mode is
    not honored at all (existing _provider_supports_explicit_api_mode guard),
    so gpt-5-mini still derives chat_completions."""
    cfg = {"provider": "nous", "default": "gpt-5-mini", "api_mode": "codex_responses"}
    assert _copilot_runtime_api_mode(cfg, api_key="x") == "chat_completions"
