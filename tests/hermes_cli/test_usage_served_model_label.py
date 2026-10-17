"""The ``/usage`` model row must name the model that actually served the turn.

On a proxy combo (e.g. ``combo-free-9r-ctx-1000k``) the requested name is only an alias; the member
that answered is what determines real cost and latency. ``_usage_model_label`` reads the served
member through ``agent.served_model.result_model_fields`` — the same source the gateway footer and
the CLI status bar use — so all three surfaces agree.

Contract pinned here:
  - direct model (no served member, or served == requested): output is EXACTLY ``str(agent.model)``
    so existing users see no change at all
  - combo with a different member: ``<requested> -><member> (<provider>)``
  - combo member known but provider blank: ``<requested> -><member>`` (no empty parens)
  - a broken/absent served-model helper must not break ``/usage`` (fail-open to the requested name)
"""

import builtins

from hermes_cli.cli_info_mixin import _usage_model_label


class _Agent:
    def __init__(self, model, provider="", served=None, fallback=False, primary=None):
        self.model = model
        self.provider = provider
        self.last_served_model = served
        self._fallback_activated = fallback
        self._primary_runtime = {"model": primary} if primary else {}


def test_direct_model_is_unchanged():
    """No served member -> the row must be byte-identical to the old str(agent.model)."""
    assert _usage_model_label(_Agent("gpt-5.6-sol", provider="openai")) == "gpt-5.6-sol"


def test_direct_model_unchanged_when_served_equals_requested():
    assert _usage_model_label(_Agent("gpt-5.6-sol", "openai", served="gpt-5.6-sol")) == "gpt-5.6-sol"


def test_combo_shows_member_and_provider():
    agent = _Agent("combo-free-9r-ctx-1000k", provider="9router", served="poolside/laguna-s-2.1:free")
    assert _usage_model_label(agent) == "combo-free-9r-ctx-1000k ->poolside/laguna-s-2.1:free (9router)"


def test_combo_without_provider_has_no_empty_parens():
    agent = _Agent("combo-free-9r-ctx-1000k", provider="", served="qwen/qwen3.8-max")
    assert _usage_model_label(agent) == "combo-free-9r-ctx-1000k ->qwen/qwen3.8-max"


def test_fallback_route_is_surfaced():
    """Hermes' own primary -> active fallback reports the primary as requested and the active one
    as served, so the row shows the swap that actually happened."""
    agent = _Agent("gpt-5.6-sol", provider="9router", served=None, fallback=True, primary="qwen/qwen3.8-max")
    assert _usage_model_label(agent) == "qwen/qwen3.8-max ->gpt-5.6-sol (9router)"


def test_missing_helper_fails_open(monkeypatch):
    """A broken import must degrade to the requested name, never raise inside /usage."""
    real_import = builtins.__import__

    def boom(name, *a, **kw):
        if name == "agent.served_model":
            raise RuntimeError("simulated broken helper")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", boom)
    assert _usage_model_label(_Agent("combo-x", "9r", served="member/y")) == "combo-x"


def test_agent_without_model_attribute():
    class Bare:
        provider = "9r"

    assert _usage_model_label(Bare()) == ""


def test_served_value_is_trimmed():
    agent = _Agent("combo-x", "9r", served="  member/y  ")
    assert _usage_model_label(agent) == "combo-x ->member/y (9r)"
