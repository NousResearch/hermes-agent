"""Startup notice lines for toolsets that are switched off because their requirements are not met.

Pure rendering: ``cli.py::_show_tool_availability_warnings`` feeds it the ``unavailable`` list from
``check_tool_availability()`` (entries carry ``name`` / ``env_vars`` / ``tools``) and prints the lines.
Each line says what is off and the exact command that fixes it, in plain words; env-var names only
appear as secondary detail for toolsets with a single obvious key.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional

# Toolsets whose ``env_vars`` list is a multi-provider dump that means nothing to a user; render one
# sentence per toolset instead. Provider names must exist under plugins/web/ (or be the Nous-managed row).
_MULTI_PROVIDER_NOTICES: dict[str, str] = {
    "web": ("[yellow]⚠ Web search is off[/] — no search provider is set up yet (any one of Nous subscription, Exa, "
            "Tavily, Firecrawl, Brave, or free DuckDuckGo works). Run [bold]hermes setup tools[/] and set one up under "
            "\"Web Search & Scraping\"."),
}

_GENERIC_FOOTER = "[dim]   Run 'hermes setup tools' to configure[/]"


def filter_to_enabled_toolsets(unavailable: list[dict], enabled: Iterable[str],
                               resolve: Callable[[str], Iterable[str]]) -> list[dict]:
    """Keep only the *unavailable* entries this session would actually load.

    ``enabled`` is the CLI's toolset selection as configured — on a default install that is a
    composite bundle such as ``["hermes-cli"]``, never the individual names ``check_tool_availability``
    reports — so each entry is expanded to tool names through ``resolve`` (``toolsets.resolve_toolset``)
    and an unavailable toolset counts as enabled when its name is listed directly or any of its tools
    is inside the expansion. An empty selection means "everything", so nothing is filtered."""
    names = [str(t) for t in (enabled or []) if str(t)]
    if not names:
        return list(unavailable)
    enabled_tools: set[str] = set()
    for name in names:
        try:
            enabled_tools.update(str(t) for t in (resolve(name) or ()))
        except Exception:
            continue
    name_set = set(names)

    def _kept(item: dict) -> bool:
        if str(item.get("name") or "") in name_set:
            return True
        return any(str(t) in enabled_tools for t in (item.get("tools") or ()))

    return [item for item in unavailable if _kept(item)]


def current_terminal_backend() -> str:
    """The terminal backend selected for this process (``terminal.backend`` / TERMINAL_ENV), e.g. 'docker'."""
    from tools.terminal_tool import _get_env_config
    return str(_get_env_config().get("env_type") or "local")


def _terminal_line(backend: str, reason: Optional[str]) -> str:
    detail = f" ({reason})" if reason else ""
    return (f"[yellow]⚠ Terminal tool disabled:[/] the '{backend}' backend is not usable{detail}. "
            "Run [bold]hermes doctor[/] for details, or [bold]hermes setup terminal[/] to pick another backend.")


def tool_availability_warning_lines(unavailable: list[dict], *, terminal_reason: Optional[str],
                                    terminal_backend: str = "local") -> list[str]:
    """Rich-markup lines to print at CLI startup for *unavailable* toolsets; ``[]`` when there is nothing
    worth saying. ``terminal_reason`` is ``terminal_backend_unavailable_reason()`` (None when unknown)."""
    lines: list[str] = []
    generic: list[dict] = []
    for item in unavailable:
        name = str(item.get("name") or "")
        if name == "terminal":
            lines.append(_terminal_line(terminal_backend, terminal_reason))
        elif name in _MULTI_PROVIDER_NOTICES:
            lines.append(_MULTI_PROVIDER_NOTICES[name])
        elif item.get("env_vars"):
            generic.append(item)
    if generic:
        lines.append("[yellow]⚠️  Some tools disabled (missing API keys):[/]")
        lines.extend(f"   [dim]• {item['name']}[/] [dim italic]({', '.join(item['env_vars'])})[/]" for item in generic)
        lines.append(_GENERIC_FOOTER)
    return lines
