"""A cached agent reused across senders must carry the CURRENT sender's identity.

Shared Discord threads key the agent cache by session, so the second sender's
turn reuses an agent built with the first sender's ``_user_id``. Tools, hooks,
and cron origins read those attrs, so the reuse path has to refresh them from
the turn's ``SessionSource`` — not just reset the idle clock.
"""

import threading
import time
from types import SimpleNamespace

from gateway.config import Platform
from gateway.run_turn_runner import TurnRunner, _refresh_cached_agent_identity
from gateway.session import SessionSource
from gateway.turn_context import TurnContext

SESSION_KEY = "discord:channel:shared"
SIG = ("model", "sig")


def _cached_agent(user_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        _user_id=user_id, _user_id_alt=None, _user_name="owner", _chat_id="shared", _chat_name="general",
        _chat_type="channel", _thread_id=None,
        _last_activity_ts=time.time() - 10, _last_activity_desc="prev", _last_activity_provenance=None,
        _api_call_count=3, _last_flushed_db_idx=2, max_iterations=10,
    )


def _runner_for(source: SessionSource) -> TurnRunner:
    from gateway.run import GatewayRunner

    ctx = TurnContext(source=source, session_key=SESSION_KEY, session_id="sid-1")
    stub = SimpleNamespace(_init_cached_agent_for_turn=GatewayRunner._init_cached_agent_for_turn)
    return TurnRunner(stub, ctx)


def test_cache_hit_refreshes_identity_from_current_source():
    agent = _cached_agent("111")
    cache = {SESSION_KEY: (agent, SIG, None, "sid-1")}
    source = SessionSource(platform=Platform.DISCORD, chat_id="shared", chat_type="channel",
                           user_id="222", user_name="wife", thread_id="t-9")

    out = _runner_for(source)._lookup_cached_agent(
        SIG, threading.Lock(), cache, max_iterations=20, peek_sid=None, dead=False, msg_count=None)

    assert out.reused is True and out.agent is agent
    assert agent._user_id == "222"
    assert agent._user_name == "wife"
    assert agent._thread_id == "t-9"
    assert agent.max_iterations == 20  # existing per-turn refresh still applies


def test_refresh_only_touches_identity_attrs_the_agent_has():
    agent = SimpleNamespace(_user_id="111", _user_name="owner")
    source = SimpleNamespace(user_id="222", user_name="wife", chat_id="c", user_id_alt=None,
                             chat_name=None, chat_type="dm", thread_id=None)

    _refresh_cached_agent_identity(agent, source)

    assert (agent._user_id, agent._user_name) == ("222", "wife")
    assert not hasattr(agent, "_chat_id")
