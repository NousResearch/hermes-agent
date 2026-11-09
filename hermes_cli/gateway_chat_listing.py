"""``--list-tools`` / ``--list-toolsets`` on the gateway path: a catalog listing needs no session.

Main built a whole in-process CLI to print these and exit; the listing reads only the registry
and this profile's CLI toolset selection, so it runs in the client without an agent or a gateway.
"""


def _selection(args):
    from hermes_cli.config import load_config
    from hermes_cli.tools_config import _get_platform_tools
    requested = getattr(args, "toolsets", None)
    if isinstance(requested, (list, tuple)):
        requested = ",".join(map(str, requested))
    if isinstance(requested, str) and requested.strip():
        return sorted({name.strip() for name in requested.split(",") if name.strip()}), []
    config = load_config()
    disabled = (config.get("agent") or {}).get("disabled_toolsets") or []
    return sorted(_get_platform_tools(config, "cli")), list(disabled)


def print_tool_listing(args) -> int:
    from agent.i18n import t
    enabled, disabled = _selection(args)
    if getattr(args, "list_tools", False):
        from model_tools import get_tool_definitions, get_toolset_for_tool
        # Pre-assembly list, as main's /tools: deferred tool_search tools are shown too.
        tools = get_tool_definitions(enabled_toolsets=enabled, disabled_toolsets=disabled, quiet_mode=True,
                                     skip_tool_search_assembly=True)
        if not tools:
            print(t("cli.tools.none_available"))
            return 0
        grouped = {}
        for tool in sorted(tools, key=lambda item: item["function"]["name"]):
            name = tool["function"]["name"]
            desc = tool["function"].get("description", "").split("\n")[0]
            if ". " in desc:
                desc = desc[:desc.index(". ") + 1]
            grouped.setdefault(get_toolset_for_tool(name) or "unknown", []).append((name, desc))
        print(t("cli.tools.header"))
        for toolset in sorted(grouped):
            print(f"\n  [{toolset}]")
            for name, desc in grouped[toolset]:
                print(f"    * {name:<20} - {desc}")
        print(f"\n  {t('cli.tools.total', count=str(len(tools)))}")
        return 0
    from toolsets import get_all_toolsets, get_toolset_info
    print(t("cli.toolsets.header"))
    for name in sorted(get_all_toolsets()):
        info = get_toolset_info(name)
        if info:
            marker = "(*)" if name in enabled else "   "
            print(f"  {marker} {name:<18} [{info['tool_count']:>2} tools] - {info['description']}")
    print(f"\n  {t('cli.toolsets.currently_enabled_legend')}")
    return 0
