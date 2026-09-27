import json
import weakref
from types import SimpleNamespace

import pytest

from plugins.subagent_progress import ProgressPlugin


class Context:
    def __init__(self):
        self.wakes = []
        self.wake_kwargs = []

    def inject_message(self, content, role="user", *, session_key=None, expected_session_id=None, on_delivery=None):
        self.wakes.append((content, role, session_key))
        self.wake_kwargs.append(dict(expected_session_id=expected_session_id, on_delivery=on_delivery))
        if on_delivery:
            on_delivery(True)
        return True


class Parent:
    session_id = "parent-a"
    _interrupt_requested = False

    def __init__(self):
        self.notices = []

    def _emit_warning(self, text):
        self.notices.append(text)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    ctx, parent = Context(), Parent()
    child = SimpleNamespace(session_id="child-a", _subagent_id="sub-a", _delegate_depth=1,
                            _parent_session_id="parent-a", _delegate_parent_ref=weakref.ref(parent),
                            _interrupt_requested=False)
    from tools import delegate_tool_registry as registry
    monkeypatch.setitem(registry._active_subagents, "sub-a", {
        "agent": child, "owner_agent_session_id": parent.session_id, "accepting_steer": True})
    plugin = ProgressPlugin(ctx, tmp_path)
    monkeypatch.setattr(plugin, "current_child", lambda: child)
    monkeypatch.setattr(plugin, "session_key", lambda: "agent:main:telegram:dm:test-a")
    plugin.start(parent_session_id="parent-a", child_session_id="child-a",
                 child_subagent_id="sub-a", child_goal="Check data")
    return plugin, ctx, parent, child


def report(plugin, **extra):
    return json.loads(plugin.report({"completed": "Input checked", "evidence": ["evidence.json"],
                                     "next_step": "Check calculations", **extra}))


def test_milestone_is_durable_notifies_without_waking(setup):
    plugin, ctx, parent, child = setup
    result = report(plugin)
    assert result["success"]
    assert "Input checked" in parent.notices[0]
    assert ctx.wakes == []
    reloaded = ProgressPlugin(ctx, plugin.home)
    assert "Input checked" in reloaded.context(session_id="parent-a", platform="telegram")["context"]
    assert reloaded.context(session_id="parent-a", platform="telegram") is None


def test_decision_request_only_wakes_bound_parent(setup):
    plugin, ctx, parent, child = setup
    assert report(plugin, needs_decision=True, blocker="Choose sample A or B")["wake_scheduled"]
    assert len(ctx.wakes) == 1
    assert ctx.wakes[0][2] == "agent:main:telegram:dm:test-a"
    assert "SUBAGENT CHECKPOINT ID:" in ctx.wakes[0][0]
    assert not report(plugin, session_key="victim")["success"]


def test_duplicate_does_not_spam(setup):
    plugin, ctx, parent, child = setup
    first = report(plugin)
    second = report(plugin)
    assert first["checkpoint_id"] == second["checkpoint_id"]
    assert second["duplicate"]
    assert len(parent.notices) == 1


def test_cross_session_isolation_and_stop_preserves_checkpoint(setup):
    plugin, ctx, parent, child = setup
    report(plugin)
    assert plugin.context(session_id="parent-b", platform="telegram") is None
    plugin.context(session_id="parent-a", platform="telegram")
    plugin.stop(child_session_id="child-a", child_subagent_id="sub-a", child_status="interrupted")
    saved = plugin.context(session_id="parent-a", platform="telegram")["context"]
    assert "interrupted" in saved and "evidence.json" in saved and "Input checked" in saved


def test_parent_cannot_impersonate_a_child(setup):
    plugin, ctx, parent, child = setup
    child._delegate_depth = 0
    assert not report(plugin)["success"]
    assert not parent.notices


def test_stopped_parent_rejects_late_reports(setup):
    plugin, ctx, parent, child = setup
    parent._interrupt_requested = True
    assert not report(plugin)["success"]
    assert not ctx.wakes


@pytest.mark.parametrize("bad", [{"completed": ""}, {"completed": "x" * 1001},
                                  {"evidence": "not-a-list"}, {"needs_decision": "yes"},
                                  {"needs_decision": True, "blocker": ""}])
def test_invalid_payload_never_notifies(setup, bad):
    plugin, ctx, parent, child = setup
    assert not report(plugin, **bad)["success"]
    assert parent.notices == [] and ctx.wakes == []
