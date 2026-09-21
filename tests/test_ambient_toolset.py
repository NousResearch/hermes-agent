"""Regression test for the ambient-toolset mechanism.

Pins the producer/consumer availability symmetry whose violation caused the
Toolaria rescue bug: a tool registered under an *ambient* toolset must be
reachable in a session whose ``enabled_toolsets`` does NOT name that toolset.

Without this guarantee a plugin whose result-transform hook fires ungated
(Toolaria rescues oversized results in every session) can replace a large
result with a handle the model can never fetch, because the retrieval tool is
filtered out of restricted (cron/profile) sessions. The fix mirrors the
kanban-lifecycle exemption: ambient toolsets are unioned into every session's
tool set and allow-set, and are never deferred behind tool_search.

A non-ambient probe tool is the negative control: it must stay scoped out, so
the test proves the union is targeted, not a blanket "include everything".
"""
from model_tools import get_tool_definitions, resolve_allowed_tool_names
from tools.registry import registry
from tools import tool_search

AMBIENT_TS = "_test_ambient_ts"
AMBIENT_TOOL = "_test_ambient_probe"
PLAIN_TS = "_test_plain_ts"
PLAIN_TOOL = "_test_plain_probe"


def _schema(name):
    return {
        "name": name,
        "description": "probe",
        "parameters": {"type": "object", "properties": {}},
    }


def _names(defs):
    return {(d.get("function") or {}).get("name") for d in defs}


def setup_module(module):
    # Handler signature matches the dispatcher's handler(args, **kwargs) call.
    registry.register(name=AMBIENT_TOOL, toolset=AMBIENT_TS,
                      schema=_schema(AMBIENT_TOOL), handler=lambda args=None, **k: "")
    registry.register(name=PLAIN_TOOL, toolset=PLAIN_TS,
                      schema=_schema(PLAIN_TOOL), handler=lambda args=None, **k: "")
    registry.mark_ambient(AMBIENT_TS)


def teardown_module(module):
    registry.deregister(AMBIENT_TOOL)
    registry.deregister(PLAIN_TOOL)
    # Drop the ambient marker so the process-wide singleton isn't polluted.
    registry.unmark_ambient(AMBIENT_TS)


def test_ambient_tool_present_in_restricted_session():
    # A session scoped to an unrelated toolset must STILL see the ambient tool,
    # but NOT a plain tool from another un-enabled toolset.
    names = _names(get_tool_definitions(enabled_toolsets=["file"], quiet_mode=True))
    assert AMBIENT_TOOL in names, "ambient tool must be visible under restricted scope"
    assert PLAIN_TOOL not in names, "non-ambient tool must stay scoped out"


def test_ambient_tool_allowed_by_runtime_fence():
    # The execution-time scope fence must agree with the schemas the model saw.
    allowed = resolve_allowed_tool_names(enabled_toolsets=["file"])
    assert allowed is not None
    assert AMBIENT_TOOL in allowed, "runtime fence must allow the ambient tool"
    assert PLAIN_TOOL not in allowed


def test_ambient_tool_never_deferred():
    assert tool_search.is_deferrable_tool_name(AMBIENT_TOOL) is False
    # Sanity: an ordinary plugin tool IS eligible for deferral.
    assert tool_search.is_deferrable_tool_name(PLAIN_TOOL) is True


def test_unrestricted_session_unaffected():
    # enabled_toolsets=None is unchanged by the ambient tier: everything is
    # included as before. Assert on the raw selection (skip_tool_search_assembly):
    # the Tool Search tier layer (added 2026-08-29) defers ordinary plugin tools
    # from the final collapsed list — orthogonal to the ambient tier, and the
    # negative control must not be conflated with it.
    names = _names(get_tool_definitions(enabled_toolsets=None, quiet_mode=True,
                                        skip_tool_search_assembly=True))
    assert AMBIENT_TOOL in names and PLAIN_TOOL in names
