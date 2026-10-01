"""Output cap on the Responses (``/v1/responses``) wire surface.

A provider profile may declare a per-model completion-token cap
(``OpencodeGoProfile._MODEL_MAX_TOKENS``; #68745 is why the cap exists — the relay
otherwise reserves its own maximum and rejects the request with
"Set max_output_tokens to avoid being charged the model maximum"). The
chat-completions transport sends that cap; the Responses transport and the auxiliary
Responses adapter never did, so a capped model on a Responses route went out uncapped.

Contract asserted here, on both Responses paths: caller budget > profile cap > omit.
"""

from types import SimpleNamespace

import pytest

from agent.auxiliary_client import _CodexCompletionsAdapter
from agent.transports import get_transport
from providers import get_provider_profile

MESSAGES = [{"role": "user", "content": "hi"}]
OPCODE_HOST = "https://opencode.ai/zen/go/v1"
# A model whose cap the opencode-go profile already declares, so the assertion is a
# relationship between the profile and the wire rather than a frozen catalog value.
CAPPED = "mimo-v2.5-pro"


class _Profile:
    """Profile stub exposing only the cap lookup the transports may use."""

    def __init__(self, cap):
        self.cap = cap

    def get_max_tokens(self, model):
        return self.cap


@pytest.fixture
def transport():
    import agent.transports.codex  # noqa: F401

    return get_transport("codex_responses")


class TestResponsesTransportOutputCap:
    def test_profile_cap_sent_when_caller_sets_no_budget(self, transport):
        kw = transport.build_kwargs(model="m", messages=MESSAGES, tools=[], provider_profile=_Profile(1234))
        assert kw["max_output_tokens"] == 1234

    def test_caller_budget_wins_over_profile_cap(self, transport):
        kw = transport.build_kwargs(
            model="m", messages=MESSAGES, tools=[], max_tokens=99, provider_profile=_Profile(1234)
        )
        assert kw["max_output_tokens"] == 99

    def test_no_cap_anywhere_omits_the_field(self, transport):
        kw = transport.build_kwargs(model="m", messages=MESSAGES, tools=[])
        assert "max_output_tokens" not in kw

    def test_profile_without_a_cap_omits_the_field(self, transport):
        kw = transport.build_kwargs(model="m", messages=MESSAGES, tools=[], provider_profile=_Profile(None))
        assert "max_output_tokens" not in kw

    def test_codex_backend_route_still_omits_the_field(self, transport):
        kw = transport.build_kwargs(
            model="gpt-6-sol", messages=MESSAGES, tools=[],
            is_codex_backend=True, provider_profile=_Profile(1234),
        )
        assert "max_output_tokens" not in kw

    def test_declared_cap_reaches_the_wire(self, transport):
        profile = get_provider_profile("opencode-go")
        assert profile is not None
        declared = profile.get_max_tokens(CAPPED)
        assert declared, "premise: the opencode-go profile declares a cap for this model"
        kw = transport.build_kwargs(model=CAPPED, messages=MESSAGES, tools=[], provider_profile=profile)
        assert kw["max_output_tokens"] == declared


class TestAuxiliaryResponsesOutputCap:
    """The auxiliary Responses adapter is a second builder for the same wire surface."""

    def _build(self, host, provider, model=CAPPED, *, max_tokens=None):
        client = SimpleNamespace(base_url=host)
        client._hermes_aux_effective_provider = provider
        kwargs, _model, _timeout = _CodexCompletionsAdapter(client, model)._build_responses_kwargs(
            {"model": model, "messages": MESSAGES, "max_tokens": max_tokens}
        )
        return kwargs

    def test_declared_cap_reaches_the_aux_wire(self):
        profile = get_provider_profile("opencode-go")
        assert profile is not None
        declared = profile.get_max_tokens(CAPPED)
        assert declared, "premise: the opencode-go profile declares a cap for this model"
        kw = self._build(OPCODE_HOST, "opencode-go")
        assert kw["max_output_tokens"] == declared

    def test_aux_caller_budget_wins_over_profile_cap(self):
        kw = self._build(OPCODE_HOST, "opencode-go", max_tokens=64)
        assert kw["max_output_tokens"] == 64

    def test_codex_endpoint_never_gets_the_field(self):
        kw = self._build("https://chatgpt.com/backend-api/codex", "openai-codex", model="gpt-6-sol")
        assert "max_output_tokens" not in kw

    def test_provider_without_a_declared_cap_is_untouched(self):
        kw = self._build("https://api.openai.com/v1", "openai", model="gpt-6-sol")
        assert "max_output_tokens" not in kw
