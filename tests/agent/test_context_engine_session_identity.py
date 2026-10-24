"""The context engine's first ``on_session_start`` says whom the agent serves, as memory providers
learn it in ``initialize()``."""

from types import SimpleNamespace

import pytest

from agent.agent_init import _inject_context_engine_tools, _memory_provider_init_kwargs
from agent.agent_init_params import _GATEWAY_IDENTITY_PARAMS


class RecordingEngine:
    context_length = 100_000

    def __init__(self):
        self.starts = []

    def get_tool_schemas(self):
        return []

    def on_session_start(self, session_id, **kwargs):
        self.starts.append((session_id, kwargs))


class LegacyEngine(RecordingEngine):
    """An engine written before the identity kwargs existed, with an explicit signature."""

    def on_session_start(self, session_id, hermes_home=None, platform=None, model=None,
                         context_length=None, conversation_id=None, user_id=None):
        self.starts.append((session_id, {"platform": platform, "user_id": user_id}))


def _agent(engine, *, platform, parent_session_id=None, **identity):
    gateway = {f"_{name}": identity.get(name) for name in _GATEWAY_IDENTITY_PARAMS}
    return SimpleNamespace(
        context_compressor=engine, tools=None, enabled_toolsets=None, session_id="session-1",
        platform=platform, model="test/model", _parent_session_id=parent_session_id,
        _session_db=None, session_cwd=None, **gateway,
    )


@pytest.mark.parametrize(
    ("platform", "parent", "identity", "agent_context"),
    [
        ("telegram", None, {"user_id": "42", "user_name": "Ada", "chat_id": "-100", "chat_type": "group",
                            "gateway_session_key": "agent:main:telegram:group:-100"}, "primary"),
        ("subagent", "parent-session", {}, "subagent"),
        ("cron", None, {}, "cron"),
    ],
    ids=["gateway-user", "subagent", "cron"],
)
def test_engine_learns_whom_the_agent_serves(platform, parent, identity, agent_context):
    engine = RecordingEngine()
    agent = _agent(engine, platform=platform, parent_session_id=parent, **identity)

    _inject_context_engine_tools(agent)

    [(session_id, kwargs)] = engine.starts
    assert session_id == "session-1"
    assert kwargs["agent_context"] == agent_context
    assert kwargs.get("parent_session_id") == parent
    assert kwargs["agent_identity"]
    # Same identity the memory slot receives, so the two slots cannot drift apart.
    provider_kwargs = _memory_provider_init_kwargs(agent, platform)
    for key in ("agent_context", "agent_identity", *_GATEWAY_IDENTITY_PARAMS):
        assert kwargs.get(key) == provider_kwargs.get(key)
    for key, value in identity.items():
        assert kwargs[key] == value


def test_engine_with_an_explicit_signature_gets_only_what_it_declares():
    engine = LegacyEngine()

    _inject_context_engine_tools(_agent(engine, platform="telegram", user_id="42", chat_id="-100"))

    assert engine.starts == [("session-1", {"platform": "telegram", "user_id": "42"})]
