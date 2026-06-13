from hermes_cli.tools_config import _get_platform_tools
from toolsets import CORP_DANGEROUS_TOOLSETS, resolve_toolset
from model_tools import get_tool_definitions

# Individual tool names that must never reach a Time session.
DANGEROUS = {"terminal", "process", "write_file", "patch",
             "execute_code", "computer_use", "browser_navigate",
             "ha_call_service", "delegate_task", "kanban_create"}


def _schema_names(defs):
    names = set()
    for d in defs or []:
        fn = d.get("function") if isinstance(d, dict) else None
        if isinstance(fn, dict) and "name" in fn:
            names.add(fn["name"])
        elif isinstance(d, dict) and "name" in d:
            names.add(d["name"])
    return names


def _time_tool_schema_names(platform_toolsets_list):
    config = {"platform_toolsets": {"time": platform_toolsets_list}}
    enabled = sorted(_get_platform_tools(config, "time"))
    return _schema_names(get_tool_definitions(enabled_toolsets=enabled))


def test_time_platform_strips_dangerous_even_if_misconfigured():
    # End-to-end: even if an admin wrongly grants the composite "coding"
    # toolset to Time, NO dangerous tool schema may reach the model.
    names = _time_tool_schema_names(["coding"])
    leaked = names & DANGEROUS
    assert not leaked, f"Time leaked dangerous tool SCHEMAS: {sorted(leaked)}"


def test_time_corp_safe_end_to_end_has_safe_tools_no_dangerous():
    names = _time_tool_schema_names(["corp_safe"])
    assert "web_search" in names and "memory" in names
    assert not (names & DANGEROUS)


def test_time_keeps_safe_tools_when_corp_safe_configured():
    config = {"platform_toolsets": {"time": ["corp_safe"]}}
    tools = _get_platform_tools(config, "time")
    # _get_platform_tools returns a mix of toolset NAMES and individual tool
    # names. For a composite like corp_safe, the safe toolset keys are
    # recovered by the reverse-mapping loop, so "web" (not "web_search") and
    # "memory" are the expected values in the returned set.
    assert "web" in tools  # web toolset key (web_search + web_extract)
    assert "memory" in tools
    assert not (tools & DANGEROUS)


def test_non_time_platform_unaffected():
    # The hard-deny is Time-only: a non-restricted platform with the same
    # config still gets terminal.
    config = {"platform_toolsets": {"slack": ["coding"]}}
    tools = _get_platform_tools(config, "slack")
    assert "terminal" in tools, "non-time platform must NOT be hard-denied"
