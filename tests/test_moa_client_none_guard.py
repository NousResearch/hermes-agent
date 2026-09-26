"""Verify the MoA-client-None guard in _dispatch_nonstreaming_api_request.

Reproduces the nixiang 'NoneType' object has no attribute 'chat' crash:
provider stays "moa" but agent.client is None after a fallback/restore
cycle. The fix rebuilds the facade via build_moa_facade instead of calling
None.chat.completions.create().
"""
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, "/home/ntx/.hermes/hermes-agent")

from agent import chat_completion_helpers  # noqa: E402


class _FakeFacadeCompletions:
    """Mimics MoAChatCompletions.create() consumed via the rebuilt facade."""
    def __init__(self):
        self.prepare = MagicMock(return_value=None)
        self.create = MagicMock(return_value="facade-response")


class _FakeFacade:
    """Mimics the MoA facade client (has .chat.completions)."""
    def __init__(self):
        self.chat = types.SimpleNamespace()
        self.chat.completions = _FakeFacadeCompletions()


def _make_agent(client, provider="moa", model="nolan"):
    agent = MagicMock()
    agent.provider = provider
    agent.client = client
    agent.model = model
    agent.api_mode = "chat_completions"
    agent._anthropic_client = None
    return agent


class TestMoaClientNoneGuard(unittest.TestCase):
    def _run(self, agent, _moa_key=True):
        api_kwargs = {"model": "nolan", "messages": [{"role": "user", "content": "hi"}]}
        if _moa_key:
            api_kwargs["_moa_prepared_request"] = {"messages": [], "model": "x"}
        def _make_client(label=None):
            return agent.client
        # call via _dispatch_nonstreaming_api_request
        return chat_completion_helpers._dispatch_nonstreaming_api_request(
            agent, api_kwargs, make_client=_make_client
        )

    def test_client_none_rebuilds_facade(self):
        """agent.client None + provider moa => rebuild via build_moa_facade."""
        agent = _make_agent(client=None, provider="moa", model="nolan")

        fake_facade = _FakeFacade()
        with patch("agent.moa_loop.build_moa_facade",
                   return_value=fake_facade) as m_build:
            response = self._run(agent)

        m_build.assert_called_once_with(agent, "nolan")
        self.assertIs(agent.client, fake_facade, "facade should be rebuilt onto agent.client")
        # the MoA-internal key must still be popped for a non-facade path
        # (facade DOES consume it, but prepare() callable check decides)
        self.assertEqual(response, "facade-response")
        # warning logged
        print("OK: client-None path rebuilt facade and returned", response)

    def test_client_none_no_moa_key(self):
        """Even without _moa_prepared_request, a None client must not crash."""
        agent = _make_agent(client=None, provider="moa", model="nolan")
        fake_facade = _FakeFacade()
        with patch("agent.moa_loop.build_moa_facade",
                   return_value=fake_facade):
            response = self._run(agent, _moa_key=False)
        self.assertEqual(response, "facade-response")

    def test_client_present_normal_path(self):
        """A live native client path still works (regression guard)."""
        # Native client: no prepare() => key popped, create called directly.
        class NativeCompletions:
            def __init__(self):
                self.create = MagicMock(return_value="native-ok")
            # no prepare() attribute
        class NativeChat:
            def __init__(self):
                self.completions = NativeCompletions()
        class NativeClient:
            def __init__(self):
                self.chat = NativeChat()
        agent = _make_agent(client=NativeClient(), provider="moa", model="nolan")
        response = self._run(agent)
        self.assertEqual(response, "native-ok")
        # _moa_prepared_request must have been popped for a native client
        call_kwargs = agent.client.chat.completions.create.call_args[1]
        self.assertNotIn("_moa_prepared_request", call_kwargs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
