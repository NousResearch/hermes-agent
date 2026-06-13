from toolsets import resolve_toolset, CORP_DANGEROUS_TOOLSETS, TOOLSETS

DANGEROUS_TOOL_NAMES = {
    "terminal", "process", "read_terminal",
    "write_file", "patch",
    "execute_code", "computer_use",
    "browser_navigate", "browser_click", "browser_type",
    "ha_call_service", "delegate_task",
    "kanban_create", "kanban_unblock",
    "discord_admin",
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
               "browser", "homeassistant", "delegation",
               "kanban", "discord_admin"):
        assert ts in CORP_DANGEROUS_TOOLSETS
        assert ts in TOOLSETS  # name is real


def _schema_names(defs):
    names = set()
    for d in defs or []:
        fn = d.get("function") if isinstance(d, dict) else None
        if isinstance(fn, dict) and "name" in fn:
            names.add(fn["name"])
        elif isinstance(d, dict) and "name" in d:
            names.add(d["name"])
    return names


def test_corp_safe_scope_excludes_dangerous_schemas():
    # Defense-in-depth: when a session is scoped to corp_safe, the model never
    # even sees the JSON schema of a dangerous tool — so it cannot call one
    # directly, and the tool_search bridge (which scopes off the same set)
    # cannot surface or invoke one either.
    from model_tools import get_tool_definitions

    names = _schema_names(get_tool_definitions(enabled_toolsets=["corp_safe"]))
    leaked = names & DANGEROUS_TOOL_NAMES
    assert not leaked, f"corp_safe scope leaked dangerous tool schemas: {sorted(leaked)}"
    # Sanity: core tools ARE present as callable schemas.
    for core in ("web_search", "memory", "session_search", "clarify"):
        assert core in names, f"corp_safe scope missing core schema {core}"
