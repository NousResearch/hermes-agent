import json
from types import SimpleNamespace

import pytest

from test_progress import setup, report
from test_supervision import supervised, review


@pytest.mark.parametrize("count", [17, 32])
def test_latest_checkpoint_reaches_context_behind_stale_rows(supervised, count):
    plugin, ctx, parent, child, sup, clock = supervised
    for number in range(count):
        saved = report(plugin, completed=f"Independent milestone {number}")
        assert saved["success"]
    event = SimpleNamespace(internal=True, text=ctx.wakes[-1][0], metadata={
        "hermes_plugin_injection": True, "hermes_plugin_id": "subagent-progress"})
    assert plugin.dispatch(event=event) is None
    delivered = plugin.context(session_id=parent.session_id, platform="telegram")
    assert delivered is not None, "admitted current wake yielded no report context"
    assert f"Independent milestone {count - 1}" in delivered["context"]
    assert plugin.context(session_id=parent.session_id, platform="telegram") is None


@pytest.mark.parametrize("decision", ["approve", "steer", "stop"])
def test_grandparent_is_not_the_owning_parent(supervised, decision):
    plugin, ctx, parent, child, sup, clock = supervised
    grandparent = SimpleNamespace(session_id="grandparent", _interrupt_requested=False)
    parent._delegate_parent_ref = lambda: grandparent
    checkpoint = report(plugin)["checkpoint_id"]
    clock.now = 650
    assert not review(plugin, sup, grandparent, checkpoint, decision)["success"]
    assert child._delegate_reviewed_deadline.snapshot()["renewals"] == 0
    assert child._delegate_reviewed_deadline.remaining() == 50
    with plugin.db() as db:
        assert db.execute("SELECT COUNT(*) FROM reviews").fetchone()[0] == 0
    assert review(plugin, sup, parent, checkpoint)["success"]


def context_payloads(plugin, parent):
    result = plugin.context(session_id=parent.session_id, platform="telegram")
    if result is None:
        return [], ""
    text = result["context"]
    return json.loads(text.split("\n")[-2]), text


@pytest.mark.parametrize("goal_size,count,first_size", [(0, 20, 16), (5000, 3, 2), (13000, 2, 1)])
def test_context_limits_preserve_unselected_valid_children(setup, monkeypatch, goal_size, count, first_size):
    from tools import delegate_tool_registry as registry
    plugin, ctx, parent, child = setup
    expected = []
    for index in range(count):
        sid = f"bounded-child-{index}"
        other = SimpleNamespace(**vars(child))
        other._subagent_id = sid
        other.session_id = sid
        monkeypatch.setitem(registry._active_subagents, sid, {
            "agent": other, "owner_agent_session_id": parent.session_id})
        plugin.start(parent_session_id=parent.session_id, child_session_id=sid,
                     child_subagent_id=sid, child_goal="G" * goal_size)
        plugin.current_child = lambda other=other: other
        expected.append(report(plugin)["checkpoint_id"])
    first, text = context_payloads(plugin, parent)
    assert len(first) == first_size
    assert [item["checkpoint_id"] for item in first] == expected[:first_size]
    assert all(item["goal"] == "G" * goal_size for item in first)
    assert len(text) <= 12000 or (len(first) == 1 and goal_size > 12000)
    with plugin.db() as db:
        pending = [row[0] for row in db.execute("SELECT id FROM reports WHERE consumed=0 ORDER BY id")]
    assert pending == expected[first_size:]
    seen = [item["checkpoint_id"] for item in first]
    while pending:
        batch, text = context_payloads(plugin, parent)
        assert batch
        assert len(batch) <= 16
        assert len(text) <= 12000 or (len(batch) == 1 and goal_size > 12000)
        seen.extend(item["checkpoint_id"] for item in batch)
        pending = pending[len(batch):]
    assert seen == expected
    assert context_payloads(plugin, parent) == ([], "")


@pytest.mark.parametrize("session,compressed,accepted", [
    ("parent-a", False, True), ("compressed-parent", True, True),
    ("unrelated", False, False), ("new-parent", True, False),
])
def test_rebuilt_parent_requires_same_verified_conversation(supervised, session, compressed, accepted):
    plugin, ctx, parent, child, sup, clock = supervised
    checkpoint = report(plugin)["checkpoint_id"]
    child._delegate_parent_ref = lambda: None
    rebuilt = SimpleNamespace(session_id=session, _interrupt_requested=False)
    if compressed:
        rebuilt._session_db = SimpleNamespace(resolve_resume_session_id=lambda sid:
            "compressed-parent" if sid in {"parent-a", "compressed-parent"} else sid)
    clock.now = 650
    result = review(plugin, sup, rebuilt, checkpoint)
    assert result["success"] is accepted
    assert child._delegate_reviewed_deadline.snapshot()["renewals"] == int(accepted)
    if not accepted:
        assert review(plugin, sup, parent, checkpoint)["success"]


def test_live_parent_identity_cannot_authorize_new_session(supervised):
    plugin, ctx, parent, child, sup, clock = supervised
    checkpoint = report(plugin)["checkpoint_id"]
    parent.session_id = "new-parent"
    assert not review(plugin, sup, parent, checkpoint)["success"]
    assert child._delegate_reviewed_deadline.snapshot()["renewals"] == 0
