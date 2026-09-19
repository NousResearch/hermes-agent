"""Unit tests for the opt-in ``account`` / ``billing`` / ``effort`` runtime-footer fields.

Companion to tests/gateway/test_runtime_footer.py, which covers the base footer. These cover
``resolve_agent_runtime_fields()`` (live-agent → field values) and the render path for the three
fields, including the invariant that the DEFAULT field set is unchanged — a footer whose
``fields`` are unset must render byte-identically to before this feature existed.
"""

from __future__ import annotations

import types

import pytest

from gateway.runtime_footer import (
    _DEFAULT_FIELDS,
    build_footer_line,
    format_runtime_footer,
    resolve_agent_runtime_fields,
)


def _cfg(fields):
    return {"display": {"runtime_footer": {"enabled": True, "fields": fields}}}


def _fake_agent(**kw):
    agent = types.SimpleNamespace()
    agent.provider = kw.get("provider", "anthropic")
    agent._credential_pool_entry_id = kw.get("pool_id")
    agent._is_anthropic_oauth = kw.get("oauth", True)
    agent.reasoning_config = kw.get("reasoning_config", {"enabled": True, "effort": "high"})
    if "overage" in kw:
        agent._last_anthropic_overage_in_use = kw["overage"]
    return agent


# ---------------------------------------------------------------------------
# billing: sub / extra / API from the provider's own per-response header
# ---------------------------------------------------------------------------

def test_billing_extra_when_provider_stamps_overage():
    """anthropic-ratelimit-unified-overage-in-use: true → the turn was paid overage."""
    out = resolve_agent_runtime_fields(_fake_agent(overage=True))
    assert out["billing"] == "extra"


def test_billing_sub_when_overage_header_false():
    out = resolve_agent_runtime_fields(_fake_agent(overage=False))
    assert out["billing"] == "sub"


def test_billing_sub_when_overage_header_absent():
    # Attribute missing entirely (header never seen) → included allowance assumed.
    out = resolve_agent_runtime_fields(_fake_agent())
    assert out["billing"] == "sub"


def test_billing_api_for_key_auth_ignores_overage_flag():
    # Per-token API keys are "API" regardless of any stale overage attribute.
    out = resolve_agent_runtime_fields(_fake_agent(oauth=False, overage=True))
    assert out["billing"] == "API"


def test_billing_api_for_generic_provider_and_sub_for_codex():
    assert resolve_agent_runtime_fields(_fake_agent(provider="openrouter", oauth=False))["billing"] == "API"
    assert resolve_agent_runtime_fields(_fake_agent(provider="openai-codex"))["billing"] == "sub"


# ---------------------------------------------------------------------------
# account: pool label lookup with provider-name fallback
# ---------------------------------------------------------------------------

def test_account_resolves_pool_entry_label(monkeypatch):
    import hermes_cli.auth as auth_mod

    monkeypatch.setattr(
        auth_mod, "read_credential_pool",
        lambda provider_id=None: {"anthropic": [{"id": "abc123", "label": "work seat"}]},
    )
    out = resolve_agent_runtime_fields(_fake_agent(pool_id="abc123"))
    assert out["account"] == "work seat"


def test_account_falls_back_to_pool_id_then_provider(monkeypatch):
    import hermes_cli.auth as auth_mod

    monkeypatch.setattr(auth_mod, "read_credential_pool", lambda provider_id=None: {})
    assert resolve_agent_runtime_fields(_fake_agent(pool_id="abc123"))["account"] == "abc123"
    assert resolve_agent_runtime_fields(_fake_agent(pool_id=None))["account"] == "anthropic"


# ---------------------------------------------------------------------------
# effort
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "reasoning_config,expected",
    [
        ({"enabled": True, "effort": "high"}, "high"),
        ({"reasoning_effort": "medium"}, "medium"),
        ("low", "low"),
        ({}, None),
        (None, None),
    ],
)
def test_effort_resolved_from_reasoning_config(reasoning_config, expected):
    out = resolve_agent_runtime_fields(_fake_agent(reasoning_config=reasoning_config))
    assert out["effort"] == expected


# ---------------------------------------------------------------------------
# render path
# ---------------------------------------------------------------------------

def test_footer_renders_account_billing_effort():
    line = build_footer_line(
        user_config=_cfg(["model", "context_pct", "account", "billing", "effort"]),
        platform_key="telegram",
        model="anthropic/claude-sonnet-4.6",
        context_tokens=50_000,
        context_length=1_000_000,
        account_label="work seat",
        billing="sub",
        effort="high",
    )
    assert line == "claude-sonnet-4.6 · 5% · work seat · sub · effort high"


def test_unresolved_new_fields_are_skipped_not_blanked():
    line = build_footer_line(
        user_config=_cfg(["model", "context_pct", "account", "billing", "effort"]),
        platform_key="telegram",
        model="gpt-5.4",
        context_tokens=40_000,
        context_length=400_000,
        account_label=None,
        billing=None,
        effort=None,
    )
    assert line == "gpt-5.4 · 10%"


def test_default_field_set_unchanged():
    """Invariant: the new fields are opt-in only. The default field set stays exactly
    (model, context_pct, cwd), and a default-fields render never shows the new values."""
    assert _DEFAULT_FIELDS == ("model", "context_pct", "cwd")
    line = format_runtime_footer(
        model="m",
        context_tokens=1,
        context_length=100,
        cwd="/",
        account_label="work seat",
        billing="API",
        effort="high",
    )
    assert "work seat" not in line and "API" not in line and "effort" not in line


# ---------------------------------------------------------------------------
# fail-open
# ---------------------------------------------------------------------------

def test_resolver_never_raises_on_broken_agent():
    class Broken:
        def __getattr__(self, name):
            raise RuntimeError("boom")

    out = resolve_agent_runtime_fields(Broken())
    assert out == {"account": None, "billing": None, "effort": None}


def test_resolver_survives_broken_auth_store(monkeypatch):
    import hermes_cli.auth as auth_mod

    def _boom(provider_id=None):
        raise RuntimeError("auth store unreadable")

    monkeypatch.setattr(auth_mod, "read_credential_pool", _boom)
    out = resolve_agent_runtime_fields(_fake_agent(pool_id="abc123"))
    assert out["account"] == "abc123"  # falls back to the raw pool id
    assert out["billing"] == "sub"
