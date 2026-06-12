from toolsets import resolve_toolset, CORP_DANGEROUS_TOOLSETS, TOOLSETS

DANGEROUS_TOOL_NAMES = {
    "terminal", "process", "read_terminal",
    "write_file", "patch",
    "execute_code", "computer_use",
    "browser_navigate", "browser_click", "browser_type",
    "ha_call_service", "delegate_task",
}

def test_corp_safe_has_no_dangerous_tools():
    tools = set(resolve_toolset("corp_safe"))
    leaked = tools & DANGEROUS_TOOL_NAMES
    assert not leaked, f"corp_safe leaks dangerous tools: {leaked}"

def test_corp_safe_keeps_core_tools():
    tools = set(resolve_toolset("corp_safe"))
    for core in ("web_search", "vision_analyze", "memory",
                 "session_search", "skills_list", "clarify", "todo"):
        assert core in tools, f"corp_safe missing core tool {core}"

def test_dangerous_toolsets_listed():
    for ts in ("terminal", "file", "code_execution", "computer_use",
               "browser", "homeassistant", "delegation"):
        assert ts in CORP_DANGEROUS_TOOLSETS
        assert ts in TOOLSETS  # name is real
