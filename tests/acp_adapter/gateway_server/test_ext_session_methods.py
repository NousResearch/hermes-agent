"""The panel's three ACP extension methods, answered by the profile's own store (#160752).

A connected editor asks ``_session/catalog`` for the tabs it may draw (hidden sessions flagged),
``_session/setHidden`` to close or reopen a tab, and ``_session/setTitle`` to rename one. Each is
answered from the store this profile owns, so the next turn sees what the panel wrote.
"""
import pytest
from acp.exceptions import RequestError

from acp_adapter.gateway_server import GatewayACPAgent
from persistence.conversation.sessions import TitleSource
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_seed import seed_messages, seed_session


@pytest.fixture
def agent():
    """A gateway agent over this test's isolated profile home, with one open and one hidden tab."""
    agent = GatewayACPAgent()
    with open_persistence(profile_scope(home=agent._home)) as store:
        seed_session(store, "open-tab", "acp", cwd=str(agent._home))
        seed_messages(store, "open-tab", {"role": "user", "content": "open input"})
        seed_session(store, "closed-tab", "acp", cwd=str(agent._home), hidden=True)
        seed_messages(store, "closed-tab", {"role": "user", "content": "closed input"})
    return agent


async def _rows(agent):
    answer = await agent.ext_method("session/catalog", {})
    return {row["session_id"]: row for row in answer["sessions"]}


@pytest.mark.asyncio
async def test_catalog_answers_every_session_with_its_hidden_flag(agent):
    rows = await _rows(agent)
    assert set(rows) == {"open-tab", "closed-tab"}
    assert rows["open-tab"]["hidden"] is False
    assert rows["closed-tab"]["hidden"] is True
    assert rows["closed-tab"]["message_count"] == 1


@pytest.mark.asyncio
async def test_set_hidden_hides_a_tab_and_unhides_it_again(agent):
    assert await agent.ext_method(
        "session/setHidden", {"sessionId": "open-tab", "hidden": True}) == {
        "ok": True, "sessionId": "open-tab", "hidden": True}
    assert (await _rows(agent))["open-tab"]["hidden"] is True

    assert await agent.ext_method(
        "session/setHidden", {"sessionId": "open-tab", "hidden": False}) == {
        "ok": True, "sessionId": "open-tab", "hidden": False}
    assert (await _rows(agent))["open-tab"]["hidden"] is False


@pytest.mark.asyncio
async def test_set_title_renames_a_tab_as_a_user_title(agent):
    assert await agent.ext_method(
        "session/setTitle", {"sessionId": "open-tab", "title": "Renamed"}) == {
        "ok": True, "sessionId": "open-tab", "title": "Renamed"}
    assert (await _rows(agent))["open-tab"]["title"] == "Renamed"

    # A user title is the highest authority, so the agent's own titler may not overwrite it.
    with open_persistence(profile_scope(home=agent._home)) as store:
        assert store.sessions.set_auto_title("open-tab", "Auto", source=TitleSource.LLM) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("method, params", [
    ("session/setHidden", {"sessionId": "gone", "hidden": True}),
    ("session/setTitle", {"sessionId": "gone", "title": "Renamed"}),
])
async def test_an_unknown_session_is_refused_with_its_id(agent, method, params):
    with pytest.raises(RequestError) as refused:
        await agent.ext_method(method, params)
    assert refused.value.code == -32602
    assert refused.value.data == {"reason": "unknown session gone"}


@pytest.mark.asyncio
async def test_malformed_params_are_refused(agent):
    for method in ("session/setHidden", "session/setTitle"):
        with pytest.raises(RequestError) as refused:
            await agent.ext_method(method, {})
        assert refused.value.code == -32602


@pytest.mark.asyncio
async def test_an_unknown_extension_method_is_not_found(agent):
    with pytest.raises(RequestError) as refused:
        await agent.ext_method("session/nope", {})
    assert refused.value.code == -32601
    assert refused.value.data == {"method": "_session/nope"}


@pytest.mark.asyncio
async def test_the_wire_reaches_the_handler(agent):
    """``HermesACPAgent()`` answers as a ``GatewayACPAgent``, so the handlers must live on it."""
    from acp.agent.router import build_agent_router
    router = build_agent_router(agent)
    answer = await router("_session/catalog", {}, False)
    assert {row["session_id"] for row in answer["sessions"]} == {"open-tab", "closed-tab"}
