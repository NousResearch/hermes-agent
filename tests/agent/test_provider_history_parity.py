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
