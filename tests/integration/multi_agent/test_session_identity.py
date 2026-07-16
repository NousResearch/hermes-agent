"""D — session identity: a session BOUND to agent A must not be hijackable to
agent B via a request header.

The security-adjacent invariant of PR #62944's stateful session tier. A session
persists the agent it was routed to at creation time (first-writer-wins); every
later turn on that session runs under the PERSISTED agent, never a header the
caller re-sends. These drive the REAL /api/sessions/* routes through the actual
APIServerAdapter + a real SessionDB (state.db under HERMES_HOME) and observe, via
the spy, what the run actually executed as.
"""
import pytest

pytestmark = pytest.mark.asyncio


async def test_create_session_persists_routed_agent_id(integ):
    """(1) POST /api/sessions with X-Hermes-Chat-Id: coder persists
    agent_id=coder on the session row."""
    env = integ({"coder": "sk-coder-key", "research": "sk-research-key"},
                default_agent="main", multiplex=True)

    session, status = await env.create_session("coder", session_id="s-coder")
    assert status == 201
    assert session["id"] == "s-coder"
    # Ground truth: the persisted row carries the routed agent.
    assert env.persisted_agent_id("s-coder") == "coder"


async def test_session_chat_runs_as_persisted_agent_not_header(integ):
    """(2) THE CORE INVARIANT. A coder-bound session, chatted with a CONFLICTING
    X-Hermes-Chat-Id: research header, executes as CODER (persisted agent), and
    resolves CODER's credential — the header cannot hijack it."""
    env = integ({"coder": "sk-coder-key", "research": "sk-research-key"},
                default_agent="main", multiplex=True)

    _, status = await env.create_session("coder", session_id="s-hijack")
    assert status == 201
    assert env.persisted_agent_id("s-hijack") == "coder"

    # Attacker re-sends a research routing header on the coder-bound session.
    run = await env.session_chat("s-hijack", "hello", chat_id_header="research")

    assert run["agent_id"] == "coder", (
        f"session hijacked: ran as {run['agent_id']} via header, expected coder")
    assert run["resolved_key"] == "sk-coder-key", (
        f"leaked wrong credential: {run['resolved_key']}")
    assert run["home"].endswith("/profiles/coder")
    # And definitely not research's identity/credential.
    assert run["resolved_key"] != "sk-research-key"
    assert "sk-ROOT-env" not in run["resolved_key"]


async def test_session_chat_without_header_still_runs_as_persisted_agent(integ):
    """(2b) The mirror case: NO header on the chat must not fall back to
    default 'main'; it still runs as the session's persisted agent."""
    env = integ({"coder": "sk-coder-key", "research": "sk-research-key"},
                default_agent="main", multiplex=True)

    await env.create_session("research", session_id="s-research")
    run = await env.session_chat("s-research", "hi")  # no routing header at all

    assert run["agent_id"] == "research"
    assert run["resolved_key"] == "sk-research-key"


async def test_fork_inherits_parent_agent_id(integ):
    """(3) A fork inherits the parent's persisted agent_id (fork carries no
    routing headers; re-resolving would wrongly fall back to default)."""
    env = integ({"coder": "sk-coder-key", "research": "sk-research-key"},
                default_agent="main", multiplex=True)

    await env.create_session("coder", session_id="s-parent")
    fork, status = await env.fork_session("s-parent", new_id="s-fork")
    assert status == 201
    assert fork["id"] == "s-fork"
    assert fork["parent_session_id"] == "s-parent"
    # Inherited on the row...
    assert env.persisted_agent_id("s-fork") == "coder"
    # ...and honoured at chat time.
    run = await env.session_chat("s-fork", "continue")
    assert run["agent_id"] == "coder"
    assert run["resolved_key"] == "sk-coder-key"


async def test_legacy_no_agent_session_runs_as_default(integ):
    """(4) A session created with NO routing header on a gateway whose default
    'main' has no registered profile runs at the root home reading the root
    .env (None profile), unchanged from legacy single-agent behaviour."""
    # 'main' is the default_agent but is NOT in the agent registry, so
    # _profile_for_agent_id('main') -> None (the legacy no-profile path).
    env = integ({"coder": "sk-coder-key"}, default_agent="main", multiplex=True)

    session, status = await env.create_session(None, session_id="s-legacy")
    assert status == 201
    # Defaulted to 'main' at creation (first-writer-wins default).
    assert env.persisted_agent_id("s-legacy") == "main"

    run = await env.session_chat("s-legacy", "hello")
    assert run["agent_id"] == "main"
    assert "/profiles/" not in run["home"]  # root home, not a per-agent dir
    assert run["resolved_key"] == "sk-ROOT-env"  # legacy root .env read
