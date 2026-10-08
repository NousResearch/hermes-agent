"""Provider-native replay carriers stay private to their owning provider."""
from copy import deepcopy

from agent.transports.chat_completions import ChatCompletionsTransport
from providers.base import ProviderProfile

OPENROUTER = "https://openrouter.ai/api/v1"


def test_native_carriers_follow_only_their_owner_on_each_request():
    owner = ProviderProfile(name="native-owner", base_url="process://native-owner")
    owner.native_reasoning_details_type = "native-owner.native_assistant"
    carrier = {"type": owner.native_reasoning_details_type, "messages": [{"text": "private"}]}
    standard = {"type": "reasoning.encrypted", "data": "opaque-signature"}
    history = [{"role": "assistant", "content": "answer", "reasoning_details": [carrier, standard]}]
    original = deepcopy(history)
    transport = ChatCompletionsTransport()

    # The declaring profile gets its carrier on its own (non-HTTP) route, on every request,
    # including after another provider served a turn in between; the other provider on a
    # replaying route sees standard records only.
    for profile in (owner, ProviderProfile(name="other"), owner):
        base_url = owner.base_url if profile is owner else OPENROUTER
        wire = transport.build_kwargs("test", history, provider_profile=profile, base_url=base_url)["messages"]
        expected = [carrier, standard] if profile is owner else [standard]
        assert wire[0]["reasoning_details"] == expected
        assert history == original

    # No declaring profile: a replaying route keeps standard records, a strict route drops the
    # field wholesale (#70233) — never the stored history.
    assert transport.build_kwargs("test", history, base_url=OPENROUTER)["messages"][0]["reasoning_details"] == [standard]
    assert "reasoning_details" not in transport.build_kwargs("test", history, base_url="https://api.groq.com/openai/v1")["messages"][0]
    assert history == original

    only_native = [{"role": "assistant", "content": "answer", "reasoning_details": [carrier]}]
    assert "reasoning_details" not in transport.convert_messages(only_native, base_url=OPENROUTER)[0]


NOUS_PORTAL = "https://inference-api.nousresearch.com/v1"


def test_nous_portal_strips_replayed_reasoning_details_from_wire():
    """The Portal enforces a cumulative replayed-reasoning budget; replaying stored
    reasoning_details wedges long sessions with a non-retryable 400 (#118182). Only the
    wire copy is stripped — stored history keeps the field, so switching to a replaying
    route (OpenRouter) still replays it."""
    standard = {"type": "reasoning.encrypted", "data": "opaque-signature"}
    history = [{"role": "assistant", "content": "answer", "reasoning_details": [standard]}]
    original = deepcopy(history)
    transport = ChatCompletionsTransport()

    for base_url in (NOUS_PORTAL, "https://stg-inference-api.nousresearch.com/v1"):
        wire = transport.convert_messages(deepcopy(history), base_url=base_url)
        assert "reasoning_details" not in wire[0]
        assert history == original

    # Substring lookalikes never matched the allowlist; they still strip (strict routes).
    assert "reasoning_details" not in transport.convert_messages(
        deepcopy(history), base_url="https://nousresearch.com.evil.io/v1")[0]


def test_openrouter_replay_unchanged_by_portal_strip():
    """Sibling #129037: the OpenRouter branch of the allowlist still keeps standard
    reasoning_details on the wire."""
    standard = {"type": "reasoning.encrypted", "data": "opaque-signature"}
    history = [{"role": "assistant", "content": "answer", "reasoning_details": [standard]}]
    transport = ChatCompletionsTransport()

    assert transport.convert_messages(deepcopy(history), base_url=OPENROUTER)[0]["reasoning_details"] == [standard]


def test_profile_native_carrier_still_replayed_on_portal_route():
    """A profile declaring a native carrier type consumes replayed details by contract,
    independent of the route allowlist — the Portal strip must not break that."""
    owner = ProviderProfile(name="native-owner", base_url="process://native-owner")
    owner.native_reasoning_details_type = "native-owner.native_assistant"
    carrier = {"type": owner.native_reasoning_details_type, "messages": [{"text": "private"}]}
    standard = {"type": "reasoning.encrypted", "data": "opaque-signature"}
    history = [{"role": "assistant", "content": "answer", "reasoning_details": [carrier, standard]}]
    original = deepcopy(history)
    transport = ChatCompletionsTransport()

    wire = transport.build_kwargs("test", history, provider_profile=owner, base_url=NOUS_PORTAL)["messages"]
    assert wire[0]["reasoning_details"] == [carrier, standard]
    assert history == original
