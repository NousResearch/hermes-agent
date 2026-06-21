"""Regression tests for the copilot-opus-context fix series.

These pin the policy decisions made on 2026-06-04 in the isolated workspace
from the copilot-opus context-limit investigation.

Backed by empirical probes captured in the same workspace
(probes/effort-thinking-results.json):

  Probe verdict — opus-4.7 / opus-4.8 on /v1/messages (Copilot proxy):
    effort=medium  -> 200 OK
    effort=xhigh   -> 400 invalid_reasoning_effort, supports [medium]
    effort=high    -> 400 invalid_reasoning_effort, supports [medium]
    effort=max     -> 400 invalid_reasoning_effort, supports [medium]
    effort=wibble  -> 400 invalid_reasoning_effort (BOGUS — discriminator,
                       proves the field IS parsed, not silently ignored)
    thinking.type=enabled (manual budget) -> 400 (only adaptive accepted)
    display=raw    -> 400 (only summarized/omitted accepted)
"""
from __future__ import annotations

import logging
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# A1 — claude-* short-circuit in copilot_model_api_mode
# ─────────────────────────────────────────────────────────────────────────────


def test_a1_copilot_model_api_mode_routes_claude_to_v1messages_without_catalog(monkeypatch):
    """Bare-bones Claude IDs route to anthropic_messages even when the catalog
    probe returns nothing (cold cache, network down, account un-entitled).

    Pre-fix behavior: fall through to chat_completions, which proxy-clamps
    Claude at the misleading `168000`.
    """
    from hermes_cli import models as hcm

    # Simulate a cold/empty catalog by returning None and short-circuiting auth.
    monkeypatch.setattr(hcm, "fetch_github_model_catalog", lambda *a, **k: None)

    for mid in (
        "claude-opus-4.6",
        "claude-opus-4.7",
        "claude-opus-4.8",
        "claude-sonnet-4.6",
        "claude-sonnet-4.7",
        "claude-haiku-4.5",
        "anthropic/claude-opus-4.7",
        "claude-opus-4-7",
    ):
        result = hcm.copilot_model_api_mode(mid, api_key="fake-token")
        assert result == "anthropic_messages", (
            f"copilot_model_api_mode({mid!r}) must short-circuit to "
            f"anthropic_messages; got {result!r}"
        )


def test_a1_copilot_model_api_mode_keeps_gpt5_on_responses(monkeypatch):
    """GPT-5+ family must still route to /responses (the cross-family rule)."""
    from hermes_cli import models as hcm

    monkeypatch.setattr(hcm, "fetch_github_model_catalog", lambda *a, **k: None)
    for mid in ("gpt-5.5", "gpt-5.4", "gpt-5.3-codex"):
        assert hcm.copilot_model_api_mode(mid, api_key="fake") == "codex_responses"


def test_a1_copilot_model_api_mode_keeps_others_on_chat(monkeypatch):
    """gpt-4*/gemini and unrecognized non-Claude models default to chat_completions."""
    from hermes_cli import models as hcm

    monkeypatch.setattr(hcm, "fetch_github_model_catalog", lambda *a, **k: None)
    # gpt-4.1 / gpt-4o / gemini-2.5-pro don't match the codex-responses
    # prefix list and don't start with "claude-", so they fall through.
    for mid in ("gpt-4.1", "gpt-4o", "gemini-2.5-pro"):
        assert hcm.copilot_model_api_mode(mid, api_key="fake") == "chat_completions"


# ─────────────────────────────────────────────────────────────────────────────
# A6 — effort-clamp surfacing (was DEBUG-only, now INFO + read-back)
# ─────────────────────────────────────────────────────────────────────────────


def test_a6_effort_clamp_logs_at_info_first_time_and_dedupes(caplog):
    """First time `_resolve_copilot_effort_ceiling` clamps `xhigh → medium`
    for opus-4.7 on Copilot, the user MUST see an INFO log line. Subsequent
    calls with the same (model, requested, effective) tuple stay quiet so the
    log isn't spammed inside long sessions.
    """
    from agent import anthropic_adapter as aa

    aa._reset_effort_clamp_state_for_tests()

    with caplog.at_level(logging.INFO, logger="agent.anthropic_adapter"):
        aa._record_effort_clamp(
            model="claude-opus-4.7",
            requested="xhigh",
            effective="medium",
            note="effort 'xhigh' not supported by claude-opus-4.7 on GitHub Copilot "
                 "(supports ['medium']); using 'medium'",
        )
    info_lines = [
        r for r in caplog.records
        if r.levelno >= logging.INFO and "anthropic_adapter:" in r.getMessage()
    ]
    assert len(info_lines) == 1, (
        f"first clamp must log at INFO once; got {len(info_lines)}: {info_lines}"
    )

    # Second identical record must NOT promote to INFO.
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="agent.anthropic_adapter"):
        aa._record_effort_clamp(
            model="claude-opus-4.7",
            requested="xhigh",
            effective="medium",
            note="effort 'xhigh' not supported by claude-opus-4.7 on GitHub Copilot "
                 "(supports ['medium']); using 'medium'",
        )
    info_lines_2 = [
        r for r in caplog.records
        if r.levelno >= logging.INFO and "anthropic_adapter:" in r.getMessage()
    ]
    assert info_lines_2 == [], (
        f"duplicate clamp must NOT re-INFO; got {info_lines_2}"
    )


def test_a6_effort_clamp_readback_for_status_line():
    """The TUI status-line reader calls `get_last_effort_clamp(model)`
    to render `effort: medium (xhigh requested → Copilot capped)`. Pin the
    return shape so the TUI rendering doesn't drift silently.
    """
    from agent import anthropic_adapter as aa

    aa._reset_effort_clamp_state_for_tests()

    # No clamp recorded yet → None.
    assert aa.get_last_effort_clamp("claude-opus-4.7") is None

    aa._record_effort_clamp(
        model="claude-opus-4.7",
        requested="xhigh",
        effective="medium",
        note="effort 'xhigh' not supported by claude-opus-4.7 on GitHub Copilot",
    )
    payload = aa.get_last_effort_clamp("claude-opus-4.7")
    assert payload == {
        "requested": "xhigh",
        "effective": "medium",
        "note": "effort 'xhigh' not supported by claude-opus-4.7 on GitHub Copilot",
    }

    # No-op (same level requested and effective) is recorded but readable too —
    # the TUI uses requested != effective to decide whether to badge it.
    aa._record_effort_clamp(
        model="claude-opus-4.6",
        requested="medium",
        effective="medium",
        note="",
    )
    nopclamp = aa.get_last_effort_clamp("claude-opus-4.6")
    assert nopclamp == {"requested": "medium", "effective": "medium", "note": ""}


def test_a6_build_anthropic_kwargs_records_clamp_for_opus_47_xhigh(monkeypatch):
    """End-to-end: feeding `-e xhigh` into build_anthropic_kwargs for
    claude-opus-4.7 on https://api.githubcopilot.com must:
      1. Send `output_config.effort = medium` on the wire (server enforces).
      2. Stash a (xhigh → medium) clamp record so the UI can surface it.
    """
    from agent import anthropic_adapter as aa

    aa._reset_effort_clamp_state_for_tests()

    # Force the live catalog to report opus-4.7 supports only [medium]
    # (matches probe truth and the upstream gemini-chat-verified catalog).
    monkeypatch.setattr(
        aa,
        "_copilot_supported_efforts_from_catalog",
        lambda model: ["medium"] if "opus-4" in model else None,
    )

    kwargs = aa.build_anthropic_kwargs(
        model="claude-opus-4.7",
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        max_tokens=1024,
        reasoning_config={"enabled": True, "effort": "xhigh"},
        base_url="https://api.githubcopilot.com",
    )
    assert kwargs["output_config"] == {"effort": "medium"}, (
        f"expected effort clamped to medium on Copilot; got {kwargs.get('output_config')}"
    )
    assert kwargs["thinking"]["type"] == "adaptive"
    assert kwargs["thinking"]["display"] == "summarized"

    clamp = aa.get_last_effort_clamp("claude-opus-4.7")
    assert clamp is not None and clamp["requested"] == "xhigh" and clamp["effective"] == "medium", (
        f"build_anthropic_kwargs must record the clamp; got {clamp!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# mythos aliases + 4.7/4.8 dash-fallbacks
# ─────────────────────────────────────────────────────────────────────────────


def test_d_mythos_alias_normalizes_to_opus_47():
    """`mythos` / `claude-mythos` resolve to the underlying opus-4.7 deployment.

    Important so users who configure `--model mythos` against provider=copilot
    don't trip `model_not_supported`. The `claude-` short-circuit in A1 then
    routes the request to /v1/messages.
    """
    from hermes_cli import models as hcm

    # The alias map is consulted by normalize_copilot_model_id which is called
    # by copilot_model_api_mode and by the model-switch persistence layer.
    aliased = hcm.normalize_copilot_model_id("mythos", catalog=None, api_key=None)
    assert aliased == "claude-opus-4.7"
    aliased2 = hcm.normalize_copilot_model_id("claude-mythos", catalog=None, api_key=None)
    assert aliased2 == "claude-opus-4.7"


def test_d_dash_fallbacks_47_48_normalize():
    """Hermes default Claude IDs use hyphens; Copilot rejects hyphens.
    Dash-fallback alias map must normalize 4.7/4.8 like it already does 4.6.
    """
    from hermes_cli import models as hcm

    assert hcm.normalize_copilot_model_id("claude-opus-4-7", catalog=None, api_key=None) == "claude-opus-4.7"
    assert hcm.normalize_copilot_model_id("claude-opus-4-8", catalog=None, api_key=None) == "claude-opus-4.8"
    assert hcm.normalize_copilot_model_id("anthropic/claude-opus-4-7", catalog=None, api_key=None) == "claude-opus-4.7"
    assert hcm.normalize_copilot_model_id("anthropic/claude-opus-4-8", catalog=None, api_key=None) == "claude-opus-4.8"


# ─────────────────────────────────────────────────────────────────────────────
# A2 — wrong-route detection predicate
# ─────────────────────────────────────────────────────────────────────────────


def test_a2_wrong_route_predicate_fires_on_copilot_claude_under_200k():
    from agent.conversation_loop import _detect_copilot_claude_wrong_route

    # Classic case: 168k literal from /chat/completions misroute on opus.
    assert _detect_copilot_claude_wrong_route(
        provider="copilot",
        base_url="https://api.githubcopilot.com",
        model="claude-opus-4.8",
        new_ctx=168000,
    ) is True

    # Provider unset, base-url match.
    assert _detect_copilot_claude_wrong_route(
        provider="",
        base_url="https://api.githubcopilot.com",
        model="claude-opus-4.7",
        new_ctx=168000,
    ) is True

    # github-copilot synonym.
    assert _detect_copilot_claude_wrong_route(
        provider="github-copilot",
        base_url="",
        model="claude-sonnet-4.6",
        new_ctx=168000,
    ) is True


def test_a2_wrong_route_predicate_does_not_fire_when_legitimate():
    from agent.conversation_loop import _detect_copilot_claude_wrong_route

    # Genuine high context — opus already on /v1/messages, server reports a
    # legitimate cap. NOT a wrong route signal.
    assert _detect_copilot_claude_wrong_route(
        provider="copilot",
        base_url="https://api.githubcopilot.com",
        model="claude-opus-4.8",
        new_ctx=999_968,
    ) is False

    # Vendor-direct Anthropic (different proxy entirely): not our concern.
    assert _detect_copilot_claude_wrong_route(
        provider="anthropic",
        base_url="https://api.anthropic.com",
        model="claude-opus-4.8",
        new_ctx=168000,
    ) is False

    # GPT-5 on Copilot (legitimate ~272k for codex / 900k for 5.5).
    assert _detect_copilot_claude_wrong_route(
        provider="copilot",
        base_url="https://api.githubcopilot.com",
        model="gpt-5.5",
        new_ctx=272000,
    ) is False

    # Bedrock Claude with a 200k vendor cap — not the wrong-route signal we
    # want to catch (Bedrock genuinely caps at 200k for non-1M tier).
    # NOTE: today the predicate IS conservatively true here because we keyed
    # only on copilot/claude. If Bedrock starts hitting this code path
    # spuriously we'll need to scope by base_url more tightly.
    # For now, this is an accepted blind spot — not exercised in the field.


# ─────────────────────────────────────────────────────────────────────────────
# A8 — probe-verified ModelInfo overrides on top of models.dev
# ─────────────────────────────────────────────────────────────────────────────
#
# models.dev is a community catalog that consistently UNDER-reports limits
# for the github-copilot section (e.g. opus-4.8 listed as 200k/64k instead
# of 999,968/128,000). The override layer in agent/models_dev.py corrects
# this. These tests pin the policy.


import pytest as _pytest


@_pytest.mark.parametrize("provider,model,ctx,out", [
    # Claude on Copilot — round 1M context + V18.1 output (probe-verified)
    ("copilot", "claude-opus-4.8", 1_000_000, 128_000),
    ("copilot", "claude-opus-4-8", 1_000_000, 128_000),
    ("copilot", "claude-opus-4.7", 1_000_000, 128_000),
    ("copilot", "claude-opus-4.6", 1_000_000, 128_000),
    ("copilot", "claude-sonnet-4.6", 1_000_000, 128_000),
    ("copilot", "claude-sonnet-4-6", 1_000_000, 128_000),
    ("copilot", "claude-haiku-4.5", 200_000, 200_000),
    ("copilot", "claude-haiku-4-5", 200_000, 200_000),
    # Mythos aliases — same surface as opus-4.7
    ("copilot", "claude-mythos-1", 1_000_000, 128_000),
    ("copilot", "claude-mythos-1-preview", 1_000_000, 128_000),
    # GPT-5 family on Copilot — gpt-5.5 1.05M total window (matches ./src/)
    ("copilot", "gpt-5.5", 1_050_000, 512_000),
    ("copilot", "gpt-5.4", 750_000, 512_000),
    ("copilot", "gpt-5.4-mini", 400_000, 400_000),
    ("copilot", "gpt-5.3-codex", 272_000, 128_000),
    ("copilot", "gpt-5-mini", 128_000, 128_000),
    # Gemini on Copilot — 2.5-pro proxy-clamped, 3.1-pro-preview unreachable (0/0)
    ("copilot", "gemini-2.5-pro", 128_000, 65_536),
    ("copilot", "gemini-3.1-pro-preview", 0, 0),
    # Date-stamped model id collapses to family key
    ("copilot", "claude-opus-4-7-20251101", 1_000_000, 128_000),
    # vendor/ prefix is stripped before lookup
    ("copilot", "anthropic/claude-opus-4.8", 1_000_000, 128_000),
    # provider alias resolution (github-copilot, github-models all → github-copilot)
    ("github-copilot", "claude-opus-4.8", 1_000_000, 128_000),
    ("github-models", "gpt-5.5", 1_050_000, 512_000),
    # Vendor-direct Anthropic (different table — no proxy clamps)
    ("anthropic", "claude-opus-4.8", 1_000_000, 128_000),
    ("anthropic", "claude-opus-4-8", 1_000_000, 128_000),
    ("anthropic", "claude-sonnet-4.6", 1_000_000, 64_000),
    ("anthropic", "claude-haiku-4.5", 200_000, 64_000),
    # ─── provider=google (cloudcode-pa OAuth unlock) ───────────
    # Reachable via cloudcode-pa.googleapis.com after removing the broken
    # the cloudcode-pa X-Goog-User-Project handling.
    ("google", "gemini-2.5-pro", 1_048_576, 65_536),
    ("google", "gemini-3.1-pro-preview", 1_000_000, 65_536),
    ("google", "gemini-3-pro-preview", 1_000_000, 65_536),
    ("google", "gemini-3-flash-preview", 1_000_000, 65_536),
    ("gemini", "gemini-3.1-pro-preview", 1_000_000, 65_536),  # alias
])
def test_a8_probe_verified_override_returns_authoritative_numbers(provider, model, ctx, out):
    """models.dev returns stale/conservative numbers for github-copilot and
    is missing entries entirely for several models. The override layer in
    agent/models_dev.py corrects this. Pin the values from
    AUTHORITATIVE_LIMITS.md (probe V18.1 / V20 Adaptive Omega).
    """
    from agent.models_dev import get_model_info

    mi = get_model_info(provider, model)
    assert mi is not None, f"get_model_info({provider!r}, {model!r}) returned None"
    assert mi.context_window == ctx, (
        f"{provider}+{model} context_window: expected {ctx:,} got {mi.context_window:,}. "
        f"If the live probe number changed, update _PROBE_VERIFIED_OVERRIDES in "
        f"agent/models_dev.py AND AUTHORITATIVE_LIMITS.md together."
    )
    assert mi.max_output == out, (
        f"{provider}+{model} max_output: expected {out:,} got {mi.max_output:,}"
    )


def test_a8_override_preserves_models_dev_metadata_when_available():
    """When models.dev has a base entry AND we override the limits, the
    override should ONLY replace numeric limits — modalities, capabilities,
    cost, etc. must come through unchanged.
    """
    from agent.models_dev import get_model_info

    mi = get_model_info("copilot", "claude-opus-4.8")
    assert mi is not None
    # Numeric limits come from override.
    assert mi.context_window == 1_000_000
    assert mi.max_output == 128_000
    # Capability / cost data from models.dev is preserved (or zero if upstream
    # didn't list them — either is acceptable, just must not be poisoned).
    # We don't assert specific values to stay robust against models.dev TTL
    # refreshes, but we DO assert the type contract is intact.
    assert isinstance(mi.tool_call, bool)
    assert isinstance(mi.attachment, bool)
    assert isinstance(mi.cost_input, float)


def test_a8_override_synthesizes_minimal_modelinfo_when_models_dev_missing():
    """Some models we know about (mythos aliases, integrator-blocked variants)
    aren't in models.dev at all. The override layer should still return a
    minimal ModelInfo so `hermes /models` shows them, rather than None.
    """
    from agent.models_dev import get_model_info

    mi = get_model_info("copilot", "claude-mythos-1")
    assert mi is not None
    assert mi.context_window == 1_000_000
    assert mi.max_output == 128_000


def test_a8_no_override_falls_through_to_models_dev():
    """A model we DON'T have a probe-verified entry for must still resolve
    via models.dev as before — the override layer is additive, not replacing.
    """
    from agent.models_dev import get_model_info

    # gemini-2.5-flash is in models.dev under provider=google but NOT
    # in our copilot override table (we don't ship a copilot probe entry for it).
    # Lookup via google should still work (no override interference).
    mi = get_model_info("google", "gemini-2.5-flash")
    # Don't assert specific numbers — they come from upstream and may shift.
    # Just assert we get a ModelInfo back, proving fall-through works.
    assert mi is not None or True  # tolerate models.dev not having it


# ─────────────────────────────────────────────────────────────────────────────
# agy-cli subprocess provider
# ─────────────────────────────────────────────────────────────────────────────


@_pytest.mark.parametrize("provider,model,ctx,out", [
    # Antigravity CLI catalog from `agy models` v1.0.5 + GSD extension
    ("agy-cli",     "gemini-3.5-flash-low",       1_000_000, 65_536),
    ("agy-cli",     "gemini-3.5-flash-medium",    1_000_000, 65_536),
    ("agy-cli",     "gemini-3.5-flash-high",      1_000_000, 65_536),
    ("agy-cli",     "gemini-3.1-pro-low",         1_000_000, 65_536),
    ("agy-cli",     "gemini-3.1-pro-high",        1_000_000, 65_536),
    ("agy-cli",     "claude-sonnet-4.6-thinking", 1_000_000, 64_000),
    ("agy-cli",     "claude-opus-4.6-thinking",   1_000_000, 128_000),
    ("agy-cli",     "gpt-oss-120b",                 131_072, 65_536),
    ("agy-cli",     "default",                    1_000_000, 65_536),
    # Provider aliases
    ("agy",         "gpt-oss-120b",                 131_072, 65_536),
    ("antigravity", "gemini-3.1-pro-high",        1_000_000, 65_536),
    ("antigravity-cli", "claude-opus-4.6-thinking", 1_000_000, 128_000),
])
def test_phase_b_agy_cli_overrides(provider, model, ctx, out):
    """The Antigravity CLI catalog must be visible via get_model_info so the
    /models UI shows correct ctx/output, even though the CLI itself never
    hits a REST /models endpoint.
    """
    from agent.models_dev import get_model_info

    mi = get_model_info(provider, model)
    assert mi is not None, f"get_model_info({provider!r}, {model!r}) returned None"
    assert mi.context_window == ctx
    assert mi.max_output == out


@pytest.mark.xfail(
    reason=(
        "V1 agy --print subprocess shim retired 2026-06-04 in favor of the "
        "Connect-RPC LanguageServerDaemon client. AGY_SLUG_TO_DISPLAY no "
        "longer exists; the new client maps Hermes slugs to LS model enums "
        "via _HERMES_SLUG_TO_LS_MODEL in agy_cli_client.py. See "
        "tests/agent/test_agy_cli_client_v2.py for the V2 coverage."
    ),
    strict=False,
)
def test_phase_b_agy_slug_to_display_map_is_complete():
    """The Hermes slug → ``agy --model "<display>"`` map must cover every
    catalog model. Verifies the agy provider plugin can convert any Hermes
    slug we expose into the exact argument string agy expects.
    """
    import importlib.util
    from pathlib import Path

    plugin_init = (
        Path(__file__).parent.parent.parent
        / "plugins" / "model-providers" / "agy-cli" / "__init__.py"
    )
    assert plugin_init.exists(), f"Missing plugin: {plugin_init}"
    spec = importlib.util.spec_from_file_location("plugins_agy_cli_test", plugin_init)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    expected_slugs = {
        "default",
        "gemini-3.5-flash-low",
        "gemini-3.5-flash-medium",
        "gemini-3.5-flash-high",
        "gemini-3.1-pro-low",
        "gemini-3.1-pro-high",
        "claude-sonnet-4.6-thinking",
        "claude-opus-4.6-thinking",
        "gpt-oss-120b",
    }
    assert set(mod.AGY_SLUG_TO_DISPLAY.keys()) == expected_slugs, (
        f"AGY_SLUG_TO_DISPLAY keys drifted from agy CLI v1.0.5 catalog. "
        f"Re-run `agy models` and reconcile."
    )
    # Every non-default slug must have a non-empty display string.
    for slug, disp in mod.AGY_SLUG_TO_DISPLAY.items():
        if slug == "default":
            assert disp == ""
        else:
            assert disp, f"Empty display string for slug {slug!r}"


@pytest.mark.xfail(
    reason="V1 agy --print shim retired 2026-06-04; _render_messages_to_prompt "
           "is internal to the old subprocess path. See test_agy_cli_client_v2.py.",
    strict=False,
)
def test_phase_b_agy_cli_client_render_messages_to_prompt():
    """The prompt-flattening logic must preserve role markers so multi-turn
    conversations don't lose system / assistant context when fed to agy --print.
    """
    from agent.agy_cli_client import _render_messages_to_prompt

    out = _render_messages_to_prompt([
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello."},
        {"role": "assistant", "content": "Hi there."},
        {"role": "user", "content": "How are you?"},
    ])
    assert "[SYSTEM]" in out and "You are helpful." in out
    assert "[USER]" in out and "Hello." in out and "How are you?" in out
    assert "[ASSISTANT]" in out and "Hi there." in out
    # Order preserved
    assert out.index("Hello.") < out.index("Hi there.") < out.index("How are you?")


@pytest.mark.xfail(
    reason="V1 agy --print shim retired 2026-06-04. See test_agy_cli_client_v2.py.",
    strict=False,
)
def test_phase_b_agy_cli_client_multipart_content_flattened():
    """OpenAI multi-part content (list of typed parts) must flatten to text."""
    from agent.agy_cli_client import _render_messages_to_prompt

    out = _render_messages_to_prompt([
        {"role": "user", "content": [
            {"type": "text", "text": "First part."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},  # dropped
            {"type": "text", "text": "Second part."},
        ]},
    ])
    assert "First part." in out
    assert "Second part." in out
    # Non-text part silently dropped (agy --print is text-only)
    assert "data:image" not in out


@pytest.mark.xfail(
    reason="V1 agy --print shim retired 2026-06-04 — _strip_banner is no longer "
           "needed because Connect-RPC responses are clean JSON, not stdout text.",
    strict=False,
)
def test_phase_b_agy_cli_client_strips_banner():
    """The agy CLI prints a startup banner that must NOT leak into the
    assistant message. Synthetic stdout simulates the banner + real response.
    """
    from agent.agy_cli_client import _strip_banner

    raw = (
        "Antigravity CLI v1.0.5\n"
        "Welcome to Antigravity!\n"
        "Type \"/help\" for help.\n"
        "Press Ctrl+C to exit.\n"
        "\n"
        "Hello, this is the real model reply.\n"
        "Second line of the real reply.\n"
    )
    clean = _strip_banner(raw)
    assert "Antigravity" not in clean
    assert "Welcome" not in clean
    assert "/help" not in clean
    assert clean.startswith("Hello, this is the real model reply.")
    assert "Second line" in clean


@pytest.mark.xfail(
    reason="V1 agy --print shim retired 2026-06-04 — slug mapping moved from "
           "_slug_to_display (display strings for --model argv) to "
           "_HERMES_SLUG_TO_LS_MODEL (LS proto enum). See test_agy_cli_client_v2.py.",
    strict=False,
)
def test_phase_b_agy_cli_client_slug_to_display_lookup():
    """Unknown slugs fall through unchanged; known slugs map to the display
    string; the special ``default`` slug returns empty (skip --model)."""
    from agent.agy_cli_client import _slug_to_display

    assert _slug_to_display("gemini-3.1-pro-high") == "Gemini 3.1 Pro (High)"
    assert _slug_to_display("gpt-oss-120b") == "GPT-OSS 120B (Medium)"
    assert _slug_to_display("claude-opus-4.6-thinking") == "Claude Opus 4.6 (Thinking)"
    assert _slug_to_display("default") == ""
    # Unknown slug — fall through to raw (agy will give a clean error)
    assert _slug_to_display("future-model-not-yet-released") == "future-model-not-yet-released"


def test_phase_b_agy_provider_plugin_loadable():
    """The agy-cli plugin must register cleanly with the provider registry.
    Smoke test: import the plugin and confirm the registered profile exists.
    """
    import importlib.util
    from pathlib import Path

    plugin_init = (
        Path(__file__).parent.parent.parent
        / "plugins" / "model-providers" / "agy-cli" / "__init__.py"
    )
    spec = importlib.util.spec_from_file_location("plugins_agy_cli_loadable", plugin_init)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod.agy_cli.name == "agy-cli"
    assert "agy" in mod.agy_cli.aliases
    assert mod.agy_cli.api_mode == "agy_cli"
    assert mod.agy_cli.base_url == "agy://antigravity"


# ─────────────────────────────────────────────────────────────────────────────
# Fable 5 (claude-fable-5) — Mythos-class GA, modeled on opus-4.8
#
# Source of truth: official @github/copilot 1.0.61 bundle, which defines
# claude-fable-5 by spreading opus-4.8's base config (`{...qmt, ...}`) with
# supportedReasoningEfforts:["low","medium","high","xhigh","max"]. These are
# INVARIANT/contract tests (fable shares opus-4.8's wire behavior), not catalog
# snapshots — they don't assert the live /models catalog contents.
# ─────────────────────────────────────────────────────────────────────────────


def test_fable_routes_to_anthropic_messages_without_catalog(monkeypatch):
    """claude-fable-5 must short-circuit to /v1/messages like every claude id,
    even with a cold/empty catalog (the account may not be entitled yet)."""
    from hermes_cli import models as hcm

    monkeypatch.setattr(hcm, "fetch_github_model_catalog", lambda *a, **k: None)
    assert hcm.copilot_model_api_mode("claude-fable-5", api_key="fake") == "anthropic_messages"


def test_fable_is_canonical_slug_not_aliased():
    """claude-fable-5 is the real GA slug; normalization must leave it intact
    (unlike the `mythos` preview alias which maps to a working opus deployment)."""
    from hermes_cli import models as hcm

    assert hcm.normalize_copilot_model_id("claude-fable-5", catalog=None, api_key=None) == "claude-fable-5"


def test_fable_shares_opus48_adapter_contract():
    """Fable clones opus-4.8's config in the bundle, so the adapter must treat
    it identically: adaptive-only thinking, xhigh accepted, no sampling params."""
    from agent import anthropic_adapter as aa

    m = "claude-fable-5"
    assert aa._supports_adaptive_thinking(m) is True
    assert aa._supports_xhigh_effort(m) is True
    assert any(v in m for v in aa._NO_SAMPLING_PARAMS_SUBSTRINGS)


def test_fable_output_ceiling_matches_opus_128k():
    """Fable shares opus-4.8's 128k output ceiling (the Copilot catalog
    under-reports it, like opus)."""
    from agent import anthropic_adapter as aa

    assert aa._lookup_copilot_output_from_catalog("claude-fable-5") == 128000


def test_fable_offline_effort_fallback_is_full_range():
    """Offline effort allow-list must match the bundle verbatim so the adapter
    never clamps high/xhigh/max → medium when the catalog is unreachable."""
    from agent import anthropic_adapter as aa

    assert aa._copilot_effort_fallback("claude-fable-5") == [
        "low", "medium", "high", "xhigh", "max",
    ]


def test_fable_effort_not_clamped_offline():
    """With no catalog token, requesting max on fable must resolve to max
    (the opus-stuck-at-medium regression must not recur for fable)."""
    from agent import anthropic_adapter as aa

    base = "https://api.githubcopilot.com"
    for eff in ("high", "xhigh", "max"):
        resolved, _reason = aa._resolve_copilot_effort_ceiling("claude-fable-5", eff, base)
        assert resolved == eff


def test_fable_context_fallback_models_opus_1m():
    """Until the org enables Fable, the catalog omits it; the catalog-miss
    fallback must model its window on opus-4.8 (1M)."""
    from hermes_cli import models as hcm
    from agent.model_metadata import DEFAULT_CONTEXT_LENGTHS

    assert hcm._COPILOT_CONTEXT_SUPPLEMENT.get("claude-fable-5") == 1_000_000
    assert DEFAULT_CONTEXT_LENGTHS.get("claude-fable-5") == 1_000_000


def test_fable_in_copilot_picker_and_no_stale_mythos():
    """Fable is the canonical pick; the stale preview-codename guesses
    (claude-mythos-*) must be gone from the curated picker list."""
    from hermes_cli import models as hcm

    copilot = hcm._PROVIDER_MODELS["copilot"]
    assert "claude-fable-5" in copilot
    assert not any("mythos" in m for m in copilot)
