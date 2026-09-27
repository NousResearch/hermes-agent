from types import SimpleNamespace

from test_progress import setup, report


def test_compression_keeps_checkpoint_but_other_sessions_do_not(setup):
    plugin, ctx, parent, child = setup
    report(plugin)
    child._session_db = SimpleNamespace(get_compression_lineage=lambda sid:
        ["parent-a", "compressed-a"] if sid == "compressed-a" else [sid])
    assert plugin.context(session_id="unrelated", platform="telegram") is None
    assert "Input checked" in plugin.context(session_id="compressed-a", platform="telegram")["context"]


def test_deferred_child_gets_milestone_guidance(setup):
    plugin, ctx, parent, child = setup
    child.valid_tool_names = {"tool_call", "tool_describe"}
    child.enabled_toolsets = ["subagent_progress"]
    guidance = plugin.context(session_id="child-a", platform="subagent", is_first_turn=True)["context"]
    assert "report_progress" in guidance and "tool_describe" in guidance
