"""Provider-native replay carriers stay private to their owning provider."""
from copy import deepcopy


from agent.transports.chat_completions import ChatCompletionsTransport
from providers.base import ProviderProfile


def test_native_carriers_follow_only_their_owner_on_each_request():
    owner = ProviderProfile(name="native-owner")
    owner.native_reasoning_details_type = "native-owner.native_assistant"
    carrier = {"type": owner.native_reasoning_details_type, "messages": [{"text": "private"}]}
    standard = {"type": "reasoning.encrypted", "data": "opaque-signature"}
    history = [{"role": "assistant", "content": "answer", "reasoning_details": [carrier, standard]}]
    original = deepcopy(history)
    transport = ChatCompletionsTransport()
    for profile in (owner, ProviderProfile(name="other"), None, owner):
        wire = transport.build_kwargs("test", history, provider_profile=profile)["messages"]
        expected = [carrier, standard] if profile is owner else [standard]
        assert wire[0]["reasoning_details"] == expected
        assert history == original
    only_native = [{"role": "assistant", "content": "answer", "reasoning_details": [carrier]}]
    assert "reasoning_details" not in transport.convert_messages(only_native)[0]

