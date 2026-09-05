#!/usr/bin/env python3
"""Skills Hub CLI — Unified interface for the Hermes Skills Hub."""

import json
import logging
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# tools.skills_hub / tools.skills_guard are imported inside functions (cycles + startup cost).
from hermes_constants import display_hermes_home

_console = Console()

_VALID_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]*$")
_VALID_CATEGORY_RE = re.compile(r"^[a-z][a-z0-9_/-]*$")
_TRUST_STYLE = {"builtin": "bright_cyan", "trusted": "green", "community": "yellow", "local": "dim"}
_TRUST_RANK = {"builtin": 3, "trusted": 2, "community": 1}
# Browse per-source limits. With the centralized index available, parallel_search_sources
# serves everything from "hermes-index", so that entry MUST exceed the whole catalog or browse
# silently caps the hub (50 and 5000 both truncated). The index is disk-cached and browse
# paginates client-side. The external limits only apply when the index is unavailable.
_BROWSE_LIMITS = {
    "hermes-index": 1000000, "official": 200, "skills-sh": 200, "well-known": 50,
    "github": 200, "clawhub": 500, "lobehub": 500, "browse-sh": 500}
# Programmatic (TUI gateway) browse keeps its own, lower caps.
_BROWSE_API_LIMITS = {
    "hermes-index": 5000, "official": 100, "skills-sh": 100, "well-known": 25,
    "github": 100, "clawhub": 50, "lobehub": 50, "browse-sh": 500}
_EXTRA_META_LABELS = (
    ("repo_url", "Repo"), ("detail_url", "Detail Page"), ("index_url", "Index"),
    ("endpoint", "Endpoint"), ("install_command", "Install Command"),
    ("installs", "Installs"), ("weekly_installs", "Weekly Installs"))

# --- Small shared helpers ---

def _display_source(r) -> str:
    """Source label for a result row: GitHub-tap rows surface their per-tap provider label."""
    return ((r.extra or {}).get("provider") if r.source == "github" else None) or r.source


def _trust_cell(trust_level: str, source: str, official_label: str = "official") -> str:
    """Rich-styled trust cell; official-source rows show `official_label` instead of the level."""
    label = official_label if source == "official" else trust_level
    return f"[{_TRUST_STYLE.get(trust_level, 'dim')}]{label}[/]"


def _row(r, *fields) -> dict:
    """`{field: getattr(r, field)}` projection of a result/meta object, in the given order."""
    return {f: getattr(r, f) for f in fields}


def _truncate(text: str, width: int) -> str:
    return text[:width] + ("..." if len(text) > width else "")


def _ident_col(style: str) -> tuple:
    # overflow="fold" keeps the full slug visible (wraps instead of ellipsis-truncating):
    # browse.sh slugs end in a `-XXXXXX` hash that is part of the identifier users must
    # copy into `hermes skills install`.
    return "Identifier", {"style": style, "overflow": "fold", "no_wrap": False}


def _table(*columns, **table_kw) -> Table:
    """Rich table from (header, add_column-kwargs) pairs; a bare header is a dim column."""
    table = Table(**table_kw)
    for col in columns:
        header, kw = col if isinstance(col, tuple) else (col, {"style": "dim"})
        table.add_column(header, **kw)
    return table


def _try(fn, *args):
    """`fn(*args)`, or None when it raises (adapter probes are best-effort)."""
    try:
        return fn(*args)
    except Exception:
        return None


def _sources():
    """Source router over all registries (authenticated GitHub when available)."""
    from tools.skills_hub_github import GitHubAuth
    from tools.skills_hub_search import create_source_router
    return create_source_router(GitHubAuth())


def _confirm() -> bool:
    """Prompt `Confirm [y/N]:`; EOF/Ctrl-C counts as no."""
    try:
        return input("Confirm [y/N]: ").strip().lower() in {"y", "yes"}
    except (EOFError, KeyboardInterrupt):
        return False


def _confirm_or_cancel(c: Console, *lines: str, cancel: str = "[dim]Cancelled.[/]\n") -> bool:
    """Print `lines`, ask `_confirm()`; on refusal print `cancel` and return False."""
    for line in lines:
        c.print(line)
    if _confirm():
        return True
    c.print(cancel)
    return False


def _clear_skills_cache() -> None:
    """Invalidate the skills prompt cache so the change appears immediately."""
    try:
        from agent.prompt_builder import clear_skills_system_prompt_cache
        clear_skills_system_prompt_cache(clear_snapshot=True)
    except Exception:
        pass


def _finish_change(c: Console, invalidate_cache: bool, what: str = "Change will take effect",
                   verb: str = "apply", notice: bool = True) -> None:
    """Apply-now (cache clear) or, when `notice`, tell the user the change lands next session."""
    if invalidate_cache:
        _clear_skills_cache()
        return
    if not notice:
        return
    c.print(f"[dim]{what} in your next session.[/]")
    c.print(f"[dim]Use /reset to start a new session now, or --now to {verb} immediately (invalidates prompt cache).[/]\n")


def _print_error(c: Console, message: str) -> None:
    c.print(f"[bold red]Error:[/] {message}\n")


def _print_listed(c: Console, label: str, items) -> None:
    """`[dim]Label: a, b[/]` when `items` is non-empty."""
    if items:
        c.print(f"[dim]{label}: {', '.join(items)}[/]")


def _report_pair(c: Console, success: bool, msg: str) -> bool:
    """`(success, message)` result: green message + blank line, or red error."""
    if success:
        c.print(f"[bold green]{msg}[/]\n")
    else:
        _print_error(c, msg)
    return success


def _report_ok(c: Console, result: dict, fallback: str = "") -> bool:
    """skills_sync-style `{ok, message}` result: green message on success, red error otherwise."""
    if result.get("ok"):
        c.print(f"[bold green]{result['message']}[/]")
        return True
    _print_error(c, result.get("message", fallback))
    return False


def _skill_md_preview(bundle) -> Optional[str]:
    """First 50 lines of the bundle's SKILL.md (None when absent)."""
    if not bundle or "SKILL.md" not in bundle.files:
        return None
    content = bundle.files["SKILL.md"]
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")
    lines = content.split("\n")
    more = f"\n\n... ({len(lines) - 50} more lines)" if len(lines) > 50 else ""
    return "\n".join(lines[:50]) + more


def _format_extra_metadata_lines(extra: Dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for key, label in _EXTRA_META_LABELS:
        value = extra.get(key)
        if value is not None and (value or key == "installs"):  # installs: 0 is still shown
            lines.append(f"[bold]{label}:[/] {value}")
    security = extra.get("security_audits")
    if isinstance(security, dict) and security:
        ordered = ", ".join(f"{name}={status}" for name, status in sorted(security.items()))
        lines.append(f"[bold]Security:[/] {ordered}")
    return lines


# --- Identifier / source resolution ---

def _resolve_short_name(name: str, sources, console: Console) -> str:
    """Short name -> full identifier via search; "" when ambiguous/missing (one exact match wins,
    several -> the single official one, else they are listed)."""
    from tools.skills_hub_search import unified_search
    c = console or _console
    c.print(f"[dim]Resolving '{name}'...[/]")
    results = unified_search(name, sources, source_filter="all", limit=20)
    exact = [r for r in results if r.name.lower() == name.lower()]

    if len(exact) == 1:
        c.print(f"[dim]Resolved to: {exact[0].identifier}[/]")
        return exact[0].identifier
    if len(exact) > 1:
        official = [r for r in exact if r.source == "official"]  # outranks community mirrors
        if len(official) == 1:
            c.print(f"[dim]Resolved to: {official[0].identifier} (official catalog)[/]")
            return official[0].identifier
        c.print(f"\n[yellow]Multiple skills named '{name}' found:[/]")
        table = _table("Source", "Trust", _ident_col("bold cyan"))
        for r in exact:
            table.add_row(r.source, _trust_cell(r.trust_level, r.source), r.identifier)
        c.print(table)
        c.print("[bold]Use the full identifier to install a specific one.[/]\n")
        return ""
    if results:
        c.print(f"[yellow]No exact match for '{name}'. Did you mean one of these?[/]")
        for r in results[:5]:
            c.print(f"  [cyan]{r.name}[/] — {r.identifier}")
        c.print()
        return ""
    _print_error(c, f"No skill named '{name}' found in any source.")
    return ""


def _resolve_source_meta_and_bundle(identifier: str, sources):
    """(meta, bundle, source) from ONE adapter — mixing skills.sh metadata with a ClawHub zip of
    a same-named skill once showed the wrong SKILL.md. Falls back to the first meta-only hit.
    """
    first_meta = None
    first_meta_source = None
    for src in sources:
        meta = _try(src.inspect, identifier)
        bundle = _try(src.fetch, identifier)
        if bundle:
            if meta is None:
                meta = _try(src.inspect, identifier)
            return meta, bundle, src
        if first_meta is None and meta:
            first_meta, first_meta_source = meta, src
    return first_meta, None, first_meta_source


def _full_identifier(identifier: str, sources, c) -> str:
    """Identifiers without a slash are short names; "" when they cannot be resolved."""
    return identifier if "/" in identifier else _resolve_short_name(identifier, sources, c)


def _resolve_identifier(identifier: str, sources, c) -> tuple:
    """Short-name resolution + (meta, bundle, source). identifier == "" means unresolved."""
    identifier = _full_identifier(identifier, sources, c)
    if not identifier:
        return "", None, None, None
    return (identifier, *_resolve_source_meta_and_bundle(identifier, sources))


def _is_valid_installed_skill_name(name: str) -> bool:
    """Accept identifier-shaped names, reject empty / sentinel-y values."""
    candidate = name.strip().lower() if isinstance(name, str) else ""
    return bool(candidate not in {"", "skill", "readme", "index", "unnamed-skill"}
                and _VALID_NAME_RE.match(candidate))


def _existing_categories() -> List[str]:
    """Sorted category buckets under ``~/.hermes/skills/`` (children without their own SKILL.md)."""
    from tools.skills_hub import SKILLS_DIR
    from tools.skills_hub_install import _category_skill_dirs
    try:
        return sorted(name for name in set(_category_skill_dirs(SKILLS_DIR))
                      if not (SKILLS_DIR / name / "SKILL.md").exists())
    except OSError:  # FileNotFoundError is an OSError
        return []


def _line_input(prompt: str) -> Optional[str]:
    """Stripped interactive line; None on EOF/Ctrl-C."""
    from hermes_cli.cli_output import line_input
    try:
        return line_input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return None


def _prompt_for_skill_name(c: Console, url: str, default: str = "") -> Optional[str]:
    """Prompt interactively for a skill name. Returns None on cancel/EOF."""
    c.print()
    c.print(f"[yellow]The SKILL.md at {url} doesn't declare a `name:` in its frontmatter,[/]\n"
            "[yellow]and the URL path doesn't produce a valid identifier either.[/]")
    c.print(f"[bold]Enter a skill name{f' [{default}]' if default else ''}:[/] "
            "[dim](lowercase letters, digits, hyphens, underscores; starts with a letter)[/]")
    answer = _line_input("Name: ")
    if answer is None:
        return None
    answer = answer or default
    if not _is_valid_installed_skill_name(answer):
        c.print(f"[bold red]Invalid name:[/] {answer!r}. Aborting install.\n")
        return None
    return answer


def _prompt_for_category(c: Console, existing: List[str]) -> str:
    """Prompt interactively for a category. Empty/None input means flat install."""
    c.print()
    if existing:
        c.print("[bold]Pick a category[/] "
                "[dim](reuse an existing bucket, type a new one, or press Enter to install flat)[/]")
        c.print(f"[dim]Existing: {', '.join(existing)}[/]")
    else:
        c.print("[bold]Category[/] "
                "[dim](optional — press Enter to install flat at ~/.hermes/skills/<name>/)[/]")
    answer = _line_input("Category: ")
    if answer and not _VALID_CATEGORY_RE.match(answer):
        c.print(f"[dim]Invalid category {answer!r} — installing flat.[/]")
        return ""
    return answer or ""


# --- search / browse / inspect ---

def do_search(query: str, source: str = "all", limit: int = 10, console: Optional[Console] = None,
              as_json: bool = False) -> None:
    """Search registries -> Rich table, or a clean JSON array (``as_json``) for scripting."""
    from tools.skills_hub_search import unified_search
    c = console or _console
    sources = _sources()
    if as_json:
        results = unified_search(query, sources, source_filter=source, limit=limit)
        print(json.dumps([_row(r, "name", "identifier", "source", "trust_level", "description")
                          for r in results], indent=2))
        return

    c.print(f"\n[bold]Searching for:[/] {query}")
    with c.status("[bold]Searching registries..."):
        results = unified_search(query, sources, source_filter=source, limit=limit)
    if not results:
        c.print("[dim]No skills found matching your query.[/]\n")
        return

    table = _table(("Name", {"style": "bold cyan"}), ("Description", {"max_width": 60}), "Source",
                   "Trust", _ident_col("dim"), title=f"Skills Hub — {len(results)} result(s)")
    for r in results:
        table.add_row(r.name, _truncate(r.description, 60), _display_source(r),
                      _trust_cell(r.trust_level, r.source), r.identifier)
    c.print(table)
    c.print("[dim]Use: hermes skills inspect <identifier> to preview, "
            "hermes skills install <identifier> to install "
            "(--json for scripting)[/]\n")


def _rank_and_page(all_results, page: int, page_size: int):
    """Dedupe by identifier (higher trust wins; names are NOT unique across browse-sh sites),
    sort official-first, slice one page -> (deduped, page_items, page, total_pages, start)."""
    rank = lambda r: _TRUST_RANK.get(r.trust_level, 0)  # noqa: E731
    seen: dict = {}
    for r in all_results:
        if r.identifier not in seen or rank(r) > rank(seen[r.identifier]):
            seen[r.identifier] = r
    deduped = sorted(seen.values(),
                     key=lambda r: (-rank(r), r.source != "official", r.name.lower()))
    total_pages = max(1, (len(deduped) + page_size - 1) // page_size)
    page = max(1, min(page, total_pages))
    start = (page - 1) * page_size
    return deduped, deduped[start:start + page_size], page, total_pages, start


def _fetch_browse_results(c: Console, source: str):
    """Parallel fetch from all (or filtered) sources with a live per-source progress spinner."""
    from tools.skills_hub_search import parallel_search_sources
    with c.status("[bold]Fetching skills from registries...") as status:
        # parallel_search_sources invokes the callback from the collecting thread as each
        # source completes; the page itself is rendered once over the final, fully sorted set.
        _done: List[str] = []

        def _on_source_done(sid: str, count: int) -> None:
            _done.append(f"{sid} ({count})")
            status.update(
                f"[bold]Fetching skills from registries...[/]  [dim]done: {', '.join(_done)}[/]")
        return parallel_search_sources(
            _sources(), query="", per_source_limits=_BROWSE_LIMITS, source_filter=source,
            overall_timeout=30, on_source_done=_on_source_done)


def _render_browse_page(c: Console, deduped, page_items, page: int, total_pages: int,
                        start: int, source: str, source_counts, timed_out) -> None:
    official_count = sum(1 for r in deduped if r.source == "official")
    loaded_label = f"{len(deduped)} skills loaded" + (
        f", {len(timed_out)} source(s) still loading" if timed_out else "")
    c.print(f"\n[bold]Skills Hub — Browse — {source if source != 'all' else 'all sources'}[/]"
            f"  [dim]({loaded_label}, page {page}/{total_pages})[/]")
    if official_count > 0 and page == 1:
        c.print(f"[bright_cyan]★ {official_count} official optional skill(s) from Nous Research[/]")
    c.print()

    table = _table(("#", {"style": "dim", "width": 4, "justify": "right"}),
                   ("Name", {"style": "bold cyan", "max_width": 22}),
                   ("Description", {"max_width": 44}), ("Source", {"style": "dim", "width": 12}),
                   ("Trust", {"width": 10}),
                   _ident_col("dim"), show_header=True, header_style="bold")
    for i, r in enumerate(page_items, start=start + 1):
        table.add_row(str(i), r.name, _truncate(r.description, 44), _display_source(r),
                      _trust_cell(r.trust_level, r.source, official_label="★ official"),
                      r.identifier)
    c.print(table)

    nav_parts = ([f"[cyan]--page {page - 1}[/] ← prev"] if page > 1 else []) + (
        [f"[cyan]--page {page + 1}[/] → next"] if page < total_pages else [])
    if nav_parts:
        c.print(f"  {' | '.join(nav_parts)}")
    if source == "all" and source_counts:
        c.print(f"  [dim]Sources: {', '.join(f'{sid}: {ct}' for sid, ct in sorted(source_counts.items()))}[/]")
    if timed_out:
        c.print(f"  [yellow]⚡ Slow sources skipped: {', '.join(timed_out)} "
                f"— run again for cached results[/]")
    c.print("[dim]Tip: 'hermes skills inspect <identifier>' to preview, "
            "'hermes skills install <identifier>' to install, "
            "'hermes skills search <query>' to search deeper[/]\n")


def do_browse(page: int = 1, page_size: int = 20, source: str = "all",
              console: Optional[Console] = None) -> None:
    """Browse all available skills across registries, paginated; official skills first."""
    page_size = max(1, min(page_size, 100))
    c = console or _console
    all_results, source_counts, timed_out = _fetch_browse_results(c, source)
    if not all_results:
        c.print("[dim]No skills found in the Skills Hub.[/]\n")
        return
    # Provider filter (nvidia/openai/...) narrows GitHub-tap skills by their per-tap
    # ``extra.provider`` label (the runtime index stores them all under source="github").
    from tools.skills_hub_github import _PROVIDER_FILTER_VALUES, _filter_results_by_provider
    if source.strip().lower() in _PROVIDER_FILTER_VALUES:
        all_results = _filter_results_by_provider(all_results, source)
        if not all_results:
            c.print(f"[dim]No skills found for provider '{source}'.[/]\n")
            return
    deduped, page_items, page, total_pages, start = _rank_and_page(all_results, page, page_size)
    _render_browse_page(c, deduped, page_items, page, total_pages, start, source,
                        source_counts, timed_out)


def browse_skills(page: int = 1, page_size: int = 20, source: str = "all") -> dict:
    """Paginated hub browse for programmatic callers (e.g. TUI gateway)."""
    from tools.skills_hub_search import parallel_search_sources
    page_size = max(1, min(page_size, 100))
    # The shared parallel walker carries the index-aware source-skip logic — querying
    # hermes-index AND the external APIs at once would double-count every skill.
    all_results, _counts, _timed_out = parallel_search_sources(
        _sources(), query="", per_source_limits=_BROWSE_API_LIMITS,
        source_filter=source, overall_timeout=30)
    if not all_results:
        return {"items": [], "page": 1, "total_pages": 1, "total": 0}
    deduped, page_items, page, total_pages, _start = _rank_and_page(all_results, page, page_size)
    return {
        "items": [{**_row(r, "name", "description", "source"), "trust": r.trust_level,
                   "identifier": r.identifier} for r in page_items],
        "page": page, "total_pages": total_pages, "total": len(deduped)}


def do_inspect(identifier: str, console: Optional[Console] = None) -> None:
    """Preview a skill's SKILL.md content without installing."""
    c = console or _console
    identifier, meta, bundle, _src = _resolve_identifier(identifier, _sources(), c)
    if not identifier:
        return
    if not meta:
        _print_error(c, f"Could not find '{identifier}' in any source.")
        return
    c.print()
    info_lines = [f"[bold]Name:[/] {meta.name}", f"[bold]Description:[/] {meta.description}",
                  f"[bold]Source:[/] {meta.source}",
                  f"[bold]Trust:[/] {_trust_cell(meta.trust_level, meta.source)}",
                  f"[bold]Identifier:[/] {meta.identifier}"]
    if meta.tags:
        info_lines.append(f"[bold]Tags:[/] {', '.join(meta.tags)}")
    info_lines.extend(_format_extra_metadata_lines(meta.extra))
    c.print(Panel("\n".join(info_lines), title=f"Skill: {meta.name}"))
    preview = _skill_md_preview(bundle)
    if preview is not None:
        c.print(Panel(preview, title="SKILL.md Preview", subtitle="hermes skills install <id> to install"))
    c.print()


def inspect_skill(identifier: str) -> Optional[dict]:
    """Skill metadata (+ SKILL.md preview) for programmatic callers."""
    ident, meta, bundle, _ = _resolve_identifier(identifier, _sources(), Console(quiet=True))
    if not ident or not meta:
        return None
    out = {**_row(meta, "name", "description", "source", "identifier"),
           "tags": list(meta.tags) if meta.tags else []}
    preview = _skill_md_preview(bundle)
    if preview is not None:
        out["skill_md_preview"] = preview
    return out


# --- install ---

def _install_blocked(c: Console, bundle, message: str, verdict: str, detail: str,
                     q_path: Optional[Path] = None, lead: str = "") -> None:
    """Print the blocked-install line, drop the quarantine copy, append the audit row."""
    c.print(f"{lead}[bold red]Installation blocked:[/] {message}")
    if q_path is not None:
        shutil.rmtree(q_path, ignore_errors=True)
    from tools.skills_hub import append_audit_log
    append_audit_log("BLOCKED", bundle.name, bundle.source, bundle.trust_level, verdict, detail)


def _invalid_path(c: Console, bundle, exc: ValueError, q_path: Optional[Path] = None) -> None:
    _install_blocked(c, bundle, f"{exc}\n", "invalid_path", str(exc), q_path=q_path)


def _resolve_url_bundle_name(c: Console, bundle, meta, identifier: str,
                             name_override: str, skip_confirm: bool) -> bool:
    """Name a URL-sourced bundle whose SKILL.md has none: --name override, else TTY prompt,
    else an actionable refusal on non-interactive surfaces. False => abort the install."""
    bundle_meta = bundle.metadata
    if bundle.source != "url" or (bundle.name and not bundle_meta.get("awaiting_name")):
        return True
    url = bundle_meta.get("url") or identifier
    if name_override and _is_valid_installed_skill_name(name_override):
        bundle.name = name_override.strip()
    elif name_override:
        c.print(f"[bold red]Invalid --name:[/] {name_override!r}. Must be a lowercase identifier "
                "(letters, digits, hyphens, underscores; starts with a letter).\n")
        return False
    elif skip_confirm:
        # Non-interactive surface (slash command / TUI / gateway): can't prompt.
        c.print(f"[bold red]Cannot install from URL:[/] {url}\n"
                "[yellow]The SKILL.md has no `name:` in its frontmatter, "
                "and the URL path doesn't produce a valid identifier.[/]\n\n"
                "Retry with an explicit name:\n"
                f"  [bold]/skills install {url} --name <your-name>[/]\n"
                f"  [bold]hermes skills install {url} --name <your-name>[/]\n\n"
                "[dim]Or ask the SKILL.md's author to add a `name:` field to "
                "its YAML frontmatter.[/]\n")
        return False
    else:
        chosen = _prompt_for_skill_name(c, url)
        if not chosen:
            c.print("[dim]Installation cancelled.[/]\n")
            return False
        bundle.name = chosen
    bundle_meta["awaiting_name"] = False
    # Keep SkillMeta in sync so "already installed" checks, audit logs and display agree.
    if meta is not None:
        meta.name = bundle.name
        meta.path = bundle.name
    return True


def _announce_blueprint(c: Console, skill_name: str) -> None:
    """Offer an installed skill's ``metadata.hermes.blueprint`` via /suggestions — never
    auto-scheduled (installing must not silently create a recurring job). Never raises."""
    try:
        from tools.blueprints import BlueprintError, blueprint_spec_for_installed, register_blueprint_suggestion
        try:
            spec = blueprint_spec_for_installed(skill_name)
        except BlueprintError as _rec_err:
            c.print(f"[yellow]Blueprint block present but invalid:[/] {_rec_err}\n")
            return
        if spec is None:
            return
        lead = (f"[bold cyan]Blueprint:[/] '{skill_name}' is an automation "
                f"(schedule [bold]{spec.schedule}[/])")
        if register_blueprint_suggestion(spec) is not None:
            c.print(f"{lead}.")
            c.print("[dim]Added to your suggestions — run[/] [bold]/suggestions[/] "
                    "[dim]to schedule or dismiss it.[/]\n")
        else:
            # Dropped: already offered/dismissed (latched) or the pending list is at its cap.
            c.print(f"{lead}, but it wasn't added to your suggestions (already offered/dismissed, "
                    "or the pending list is full — run [bold]/suggestions[/] to review).")
            c.print("[dim]You can still schedule it any time by asking the agent "
                    "or via[/] [bold]hermes cron add[/][dim].[/]\n")
    except Exception:  # pragma: no cover - blueprint detection is best-effort
        pass


def _pinned_sources(c: Console, sources, source_id: Optional[str], identifier: str):
    """Restrict `sources` to the adapter matching `source_id`; None when it is unknown."""
    from tools.skills_hub_install import _source_matches
    pinned = [src for src in sources if _source_matches(src, source_id)] if source_id else sources
    if pinned:
        return pinned
    _print_error(c, f"no source adapter for '{source_id}'. "
                    f"Refusing to resolve '{identifier}' against other registries "
                    f"(that would change the skill's provenance).")
    return None


def _print_fetch_failure(c: Console, sources, identifier: str) -> None:
    rate_limited = any(getattr(src, "is_rate_limited", False)
                       or getattr(getattr(src, "github", None), "is_rate_limited", False)
                       for src in sources)
    c.print(f"[bold red]Error:[/] Could not fetch '{identifier}' from any source.")
    if rate_limited:
        c.print("[yellow]Hint:[/] GitHub API rate limit exhausted "
                "(unauthenticated: 60 requests/hour).\n"
                "Set [bold]GITHUB_TOKEN[/] in your .env or install the [bold]gh[/] CLI and run "
                "[bold]gh auth login[/] to raise the limit to 5,000/hr.\n")
    else:
        c.print()


def _scan_quarantined(c: Console, q_path: Path, bundle, meta, identifier: str):
    """Run the cached security scan on the quarantined bundle and print the report."""
    from tools.skills_hub import HUB_DIR
    from tools.skills_hub_models import source_url_for_bundle
    from tools.skills_guard import scan_skill_cached, format_scan_report
    c.print("[bold]Running security scan...[/]")
    scan_source = ("official" if bundle.source == "official"
                   else bundle.identifier or getattr(meta, "identifier", "") or identifier)
    result, prov = scan_skill_cached(
        q_path, source=scan_source, source_url=source_url_for_bundle(bundle),
        cache_dir=HUB_DIR / "scan-cache")
    c.print(format_scan_report(result))
    c.print(f"[dim]Scan provenance: {'fresh' if prov['fresh'] else 'cached'}; scanner "
            f"{prov['scanner_version']}; hash {prov['bundle_hash']}[/]")
    c.print(f"[dim]Source: {prov['source_url']}; scanned {prov['scanned_at']}; "
            f"rules: {', '.join(prov['rules']) or 'none'}[/]")
    return result


_INSTALL_PANELS = {
    "official": (
        "[bold bright_cyan]This is an official optional skill maintained by Nous Research.[/]\n\n"
        "It ships with hermes-agent but is not activated by default.\n"
        "Installing will copy it to your skills directory where the agent can use it.\n\n",
        "Official Skill", "bright_cyan"),
    "external": (
        "[bold yellow]You are installing a third-party skill at your own risk.[/]\n\n"
        "External skills can contain instructions that influence agent behavior,\n"
        "shell commands, and scripts. Even after automated scanning, you should\n"
        "review the installed files before use.\n\n",
        "Disclaimer", "yellow")}


def _confirm_install(c: Console, bundle, category: str) -> bool:
    """Source-appropriate disclaimer panel + `Install '<name>'?` prompt."""
    files_at = f"Files will be at: [cyan]{display_hermes_home()}/skills/{category + '/' if category else ''}{bundle.name}/[/]"
    body, title, style = _INSTALL_PANELS["official" if bundle.source == "official" else "external"]
    c.print()
    c.print(Panel(body + files_at, title=title, border_style=style))
    return _confirm_or_cancel(c, f"[bold]Install '{bundle.name}'?[/]",
                              cancel="[dim]Installation cancelled.[/]\n")


def do_install(identifier: str, category: str = "", force: bool = False,
               console: Optional[Console] = None, skip_confirm: bool = False,
               invalidate_cache: bool = True, name_override: str = "",
               source_id: Optional[str] = None) -> None:
    """Fetch, quarantine, scan, confirm, and install a skill. ``source_id`` pins resolution to one
    adapter; callers that know the provenance (``do_update``) must pass it so a bare identifier
    cannot resolve to a same-named skill elsewhere."""
    from tools.skills_hub import HubLockFile, ensure_hub_dirs
    from tools.skills_hub_install import install_from_quarantine, quarantine_bundle
    from tools.skills_guard import should_allow_install
    c = console or _console
    ensure_hub_dirs()
    sources = _pinned_sources(c, _sources(), source_id, identifier)
    if sources is None:
        return
    identifier = _full_identifier(identifier, sources, c)
    if not identifier:
        return
    c.print(f"\n[bold]Fetching:[/] {identifier}")
    meta, bundle, _matched_source = _resolve_source_meta_and_bundle(identifier, sources)
    if not bundle:
        _print_fetch_failure(c, sources, identifier)
        return
    if not _resolve_url_bundle_name(c, bundle, meta, identifier, name_override, skip_confirm):
        return

    # URL-sourced skills: pick a category interactively when none was given (TTY only;
    # non-interactive installs fall through to flat install like every other source).
    if bundle.source == "url" and not category and not skip_confirm:
        category = _prompt_for_category(c, _existing_categories())
    # Official skills can be nested ("official/mlops/training/trl-fine-tuning"): keep every
    # identifier segment between "official" and the final slug as the category path.
    if bundle.source == "official" and not category:
        category = "/".join(bundle.identifier.split("/")[1:-1])

    existing = HubLockFile().get_installed(bundle.name)
    if existing:
        c.print(f"[yellow]Warning:[/] '{bundle.name}' is already installed at {existing['install_path']}")
        if not force:
            c.print("Use --force to reinstall.\n")
            return

    extra_metadata = {**(getattr(meta, "extra", {}) or {}), **bundle.metadata}

    try:
        q_path = quarantine_bundle(bundle)
    except ValueError as exc:
        _invalid_path(c, bundle, exc)
        return
    c.print(f"[dim]Quarantined to {q_path.relative_to(q_path.parent.parent.parent)}[/]")

    result = _scan_quarantined(c, q_path, bundle, meta, identifier)
    allowed, reason = should_allow_install(result, force=force)
    if not allowed:
        _install_blocked(c, bundle, reason, result.verdict, f"{len(result.findings)}_findings",
                         q_path=q_path, lead="\n")
        return
    # Advisory second opinion — warn-and-continue by design (PII-class findings are
    # informational); the install confirmation below is where the user decides.
    _print_tier1_advisory(q_path, c)
    metadata_lines = _format_extra_metadata_lines(extra_metadata)
    if metadata_lines:
        c.print(Panel("\n".join(metadata_lines), title="Upstream Metadata", border_style="blue"))

    # skip_confirm bypasses the prompt (TUI mode, where input() hangs).
    if not force and not skip_confirm and not _confirm_install(c, bundle, category):
        shutil.rmtree(q_path, ignore_errors=True)
        return

    try:
        install_dir = install_from_quarantine(q_path, bundle.name, category, bundle, result)
    except ValueError as exc:
        _invalid_path(c, bundle, exc, q_path)
        return
    from tools.skills_hub import SKILLS_DIR
    c.print(f"[bold green]Installed:[/] {install_dir.resolve().relative_to(Path(SKILLS_DIR).resolve()).as_posix()}")
    c.print(f"[dim]Files: {', '.join(bundle.files.keys())}[/]\n")
    _announce_blueprint(c, bundle.name)
    _finish_change(c, invalidate_cache, "Skill will be available", "activate")


def _print_tier1_advisory(skill_dir, console) -> None:
    """Advisory SkillEvaluator Tier 1 report. Never raises/blocks: scanner missing, disabled via
    ``skills.tier1_advisory: false``, or erroring all degrade to silence. Secrets render red."""
    try:
        from tools.skillevaluator_scan import (format_tier1_report, run_tier1_scan,
                                               tier1_advisory_enabled)
        if not tier1_advisory_enabled():
            return
        report = run_tier1_scan(Path(skill_dir))
        if not report.available:
            return
        text = format_tier1_report(report)
        if not report.findings:
            console.print(f"[dim]{text}[/]")
            return
        console.print(Panel(text, title="SkillEvaluator Tier 1 (advisory)",
                            border_style="red" if report.secrets_findings else "yellow"))
        if report.secrets_findings:
            console.print("[bold red]Possible credentials detected above.[/] "
                          "Review the flagged lines before using this skill.\n")
    except Exception as exc:  # advisory only — never break an install
        logging.getLogger(__name__).debug("Tier 1 advisory scan skipped: %s", exc)


# --- list / check / update / audit ---

def do_list(source_filter: str = "all", enabled_only: bool = False,
            console: Optional[Console] = None) -> None:
    """List installed skills (hub / builtin / local). Enabled state comes from the active
    profile's config — ``-p`` swaps HERMES_HOME at process start, so no profile flag here."""
    from tools.skills_hub import HubLockFile, ensure_hub_dirs
    from tools.skills_sync import _read_manifest
    from tools.skills_tool import _find_all_skills
    from agent.skill_utils import get_disabled_skill_names
    c = console or _console
    ensure_hub_dirs()
    hub_installed = {e["name"]: e for e in HubLockFile().list_installed()}
    builtin_names = set(_read_manifest())
    all_skills = _find_all_skills(skip_disabled=True)  # include disabled ones to annotate status
    disabled_names = get_disabled_skill_names()

    table = _table(("Name", {"style": "bold cyan"}), "Category", "Source", "Trust", "Status",
                   title="Installed Skills" + (" (enabled only)" if enabled_only else ""))

    counts = {"hub": 0, "builtin": 0, "local": 0}
    enabled_count = disabled_count = 0
    for skill in sorted(all_skills, key=lambda s: (s.get("category") or "", s["name"])):
        name = skill["name"]
        hub_entry = hub_installed.get(name)
        if hub_entry:
            source_type, source_display = "hub", hub_entry.get("source", "hub")
            trust = hub_entry.get("trust_level", "community")
        else:
            source_type = source_display = trust = "builtin" if name in builtin_names else "local"
        is_enabled = name not in disabled_names
        if source_filter not in ("all", source_type) or (enabled_only and not is_enabled):
            continue
        counts[source_type] += 1
        enabled_count += is_enabled
        disabled_count += not is_enabled
        table.add_row(name, skill.get("category", ""), source_display,
                      _trust_cell(trust, source_display),
                      "[bold green]enabled[/]" if is_enabled else "[dim red]disabled[/]")

    c.print(table)
    tail = (f"{enabled_count} enabled shown" if enabled_only
            else f"{enabled_count} enabled, {disabled_count} disabled")
    c.print(f"[dim]{counts['hub']} hub-installed, {counts['builtin']} builtin, "
            f"{counts['local']} local — {tail}[/]\n")


def do_check(name: Optional[str] = None, console: Optional[Console] = None) -> None:
    """Check hub-installed skills for upstream updates."""
    from tools.skills_hub_install import check_for_skill_updates
    c = console or _console
    results = check_for_skill_updates(name=name)
    if not results:
        c.print("[dim]No hub-installed skills to check.[/]\n")
        return
    table = _table(("Name", {"style": "bold cyan"}), "Source", "Status", title="Skill Updates")
    for entry in results:
        table.add_row(entry.get("name", ""), entry.get("source", ""), entry.get("status", ""))
    c.print(table)
    update_count = sum(1 for entry in results if entry.get("status") == "update_available")
    c.print(f"[dim]{update_count} update(s) available across {len(results)} checked skill(s)[/]\n")


def _has_local_edits(installed: dict) -> bool:
    """True when the on-disk content no longer matches the install-time hash."""
    from tools.skills_hub import SKILLS_DIR
    from tools.skills_guard import content_hash
    recorded_hash = installed.get("content_hash", "")
    skill_path = SKILLS_DIR / installed.get("install_path", "")
    try:
        return (bool(recorded_hash) and skill_path.is_dir()
                and content_hash(skill_path) != recorded_hash)
    except OSError:
        return False


def do_update(name: Optional[str] = None, console: Optional[Console] = None,
              force: bool = False) -> None:
    """Update hub-installed skills. Locally edited ones are skipped unless ``force`` — the
    update rmtree-replaces the user's work, so that must be an explicit choice.

    Skills whose on-disk content no longer matches the hash recorded at install time have been edited
    locally; updating them would silently destroy the user's work (``do_install(force=True)``
    rmtree-replaces the directory). Those are skipped by default and only overwritten when ``force=True``.
    Mirrors the user-modified protection bundled skills already get from ``hermes update`` (ported from
    paperclipai/paperclip#10978's explicit-merge-mode rule: destructive replacement must be an explicit
    caller choice, never a rerun default).
    """
    from tools.skills_hub import HubLockFile
    from tools.skills_hub_install import check_for_skill_updates
    c = console or _console
    lock = HubLockFile()
    updates = [entry for entry in check_for_skill_updates(name=name) if entry.get("status") == "update_available"]
    if not updates:
        c.print("[dim]No updates available.[/]\n")
        return

    skipped_local: list[str] = []
    for entry in updates:
        installed = lock.get_installed(entry["name"])
        category = ""
        if installed:
            parent = str(Path(installed.get("install_path", "")).parent)
            category = "" if parent == "." else parent
            if not force and _has_local_edits(installed):
                skipped_local.append(entry["name"])
                c.print(f"[yellow]Skipping:[/] {entry['name']} — you have local edits "
                        "(update would overwrite them).")
                continue
        c.print(f"[bold]Updating:[/] {entry['name']}")
        # Pin to the lockfile's source registry: a bare identifier such as "reddit" would
        # otherwise fuzzy-resolve inside do_install to a same-named skill in a DIFFERENT
        # registry, overwriting the user's files and rewriting the lock's `source`.
        do_install(entry["identifier"], category=category, force=True, console=c,
                   source_id=entry.get("source", "") or None)

    if len(updates) > len(skipped_local):
        c.print(f"[bold green]Updated {len(updates) - len(skipped_local)} skill(s).[/]\n")
    if skipped_local:
        c.print(f"[dim]{len(skipped_local)} skill(s) kept your local edits: "
                f"{', '.join(sorted(skipped_local))}.[/]")
        c.print("[dim]Overwrite with: hermes skills update <name> --force[/]\n")


def do_audit(name: Optional[str] = None, console: Optional[Console] = None,
             deep: bool = False) -> None:
    """Re-scan installed hub skills; ``deep`` adds an AST diagnostic (review aid, not a gate)."""
    from tools.skills_hub import HubLockFile, SKILLS_DIR
    from tools.skills_guard import scan_skill, format_scan_report
    c = console or _console
    installed = HubLockFile().list_installed()
    if not installed:
        c.print("[dim]No hub-installed skills to audit.[/]\n")
        return
    targets = [e for e in installed if e["name"] == name] if name else installed
    if not targets:
        _print_error(c, f"'{name}' is not a hub-installed skill.")
        return
    c.print(f"\n[bold]Auditing {len(targets)} skill(s)...[/]\n")
    if deep:
        from tools.skills_ast_audit import ast_scan_path, format_ast_report
    for entry in targets:
        skill_path = SKILLS_DIR / entry["install_path"]
        if not skill_path.exists():
            c.print(f"[yellow]Warning:[/] {entry['name']} — path missing: {entry['install_path']}")
            continue
        c.print(format_scan_report(scan_skill(skill_path, source=entry.get("identifier", entry["source"]))))
        if deep:
            c.print(format_ast_report(ast_scan_path(skill_path), skill_name=entry["name"]))
        c.print()


# --- uninstall / reset / bundled-skill management ---

def do_uninstall(name: str, console: Optional[Console] = None, skip_confirm: bool = False,
                 invalidate_cache: bool = True) -> None:
    """Remove a hub-installed skill with confirmation."""
    from tools.skills_hub_install import uninstall_skill
    c = console or _console
    # skip_confirm bypasses the prompt (TUI mode, where input() hangs)
    if not skip_confirm and not _confirm_or_cancel(c, f"\n[bold]Uninstall '{name}'?[/]"):
        return
    if _report_pair(c, *uninstall_skill(name)):
        _finish_change(c, invalidate_cache)


def do_reset(name: str, restore: bool = False, console: Optional[Console] = None,
             skip_confirm: bool = False, invalidate_cache: bool = True) -> None:
    """Reset a bundled skill's manifest tracking (+ optionally restore from bundled)."""
    from tools.skills_sync_bundled_ops import reset_bundled_skill
    c = console or _console
    if not skip_confirm and restore and not _confirm_or_cancel(
        c, f"\n[bold]Restore '{name}' from bundled source?[/]",
        "[dim]This will DELETE your current copy and re-copy the bundled version.[/]"):
        return
    result = reset_bundled_skill(name, restore=restore)
    if not _report_ok(c, result):
        return
    synced = result.get("synced") or {}
    _print_listed(c, "Copied", synced.get("copied"))
    _print_listed(c, "Updated", synced.get("updated"))
    c.print()
    _finish_change(c, invalidate_cache)


def do_list_modified(console: Optional[Console] = None, as_json: bool = False) -> None:
    """List bundled skills the user has edited (which `hermes update` keeps)."""
    from tools.skills_sync_bundled_ops import list_user_modified_bundled_skills
    c = console or _console
    modified = list_user_modified_bundled_skills()
    if as_json:
        c.print(json.dumps([m["name"] for m in modified]))
        return
    if not modified:
        c.print("[dim]No user-modified bundled skills — everything tracks upstream.[/]\n")
        return
    c.print(f"\n[bold]{len(modified)} user-modified bundled skill(s)[/] "
            "[dim](kept as-is by `hermes update`):[/]")
    for entry in modified:
        c.print(f"  [yellow]~[/] {entry['name']}")
    c.print()
    c.print("[dim]See changes:   hermes skills diff <name>[/]")
    c.print("[dim]Resume updates: hermes skills reset <name>          (keep your copy, re-baseline)[/]")
    c.print("[dim]Revert to stock: hermes skills reset <name> --restore[/]\n")


def _print_diff_line(c: Console, line: str) -> None:
    """Unified-diff line with light coloring (file headers +++/--- stay plain)."""
    for prefix, style in (("+", "green"), ("-", "red"), ("@@", "cyan")):
        if line.startswith(prefix) and not line.startswith(prefix * 3):
            c.print(f"[{style}]{line}[/]")
            return
    c.print(line, highlight=False)


_DIFF_STATUS_LINE = {
    "added": "[green]+ only in your copy:[/] {path}", "removed": "[red]- only in stock:[/] {path}",
    "binary": "[yellow]~ {path}:[/] binary file differs"}


def do_diff(name: str, console: Optional[Console] = None) -> None:
    """Show how the user's copy of a bundled skill differs from the stock version."""
    from tools.skills_sync_bundled_ops import diff_bundled_skill
    c = console or _console
    result = diff_bundled_skill(name)
    if not result["ok"]:
        _print_error(c, result["message"])
        return
    if not result["modified"]:
        c.print(f"[green]{result['message']}[/]\n")
        return
    c.print(f"\n[bold]{result['message']}[/]\n")
    for entry in result["diffs"]:
        if entry["status"] == "modified":
            for line in entry["diff"].splitlines():
                _print_diff_line(c, line)
        else:
            line = _DIFF_STATUS_LINE.get(entry["status"], _DIFF_STATUS_LINE["binary"])
            c.print(line.format(**entry))
    c.print()
    c.print(f"[dim]Revert with: hermes skills reset {name} --restore[/]\n")


def do_opt_out(remove: bool = False, console: Optional[Console] = None, skip_confirm: bool = False,
               invalidate_cache: bool = True) -> None:
    """Write the .no-bundled-skills marker; with ``remove`` also delete pristine (tracked AND
    unmodified) bundled skills. User-edited and non-bundled skills are never touched."""
    from tools.skills_sync_bundled_ops import set_bundled_skills_opt_out, remove_pristine_bundled_skills
    c = console or _console
    res = set_bundled_skills_opt_out(True)  # the marker first: always-safe
    if not _report_ok(c, res):
        return
    c.print(f"[dim]Marker: {res['marker']}[/]")
    if not remove:
        c.print("[dim]Existing skills on disk were left in place. "
                "Re-run with --remove to also delete unmodified bundled skills.[/]\n")
        return

    # Destructive step: preview, confirm, then delete.
    preview = remove_pristine_bundled_skills(dry_run=True)
    candidates = preview["removed"]
    if not candidates:
        c.print("[dim]No pristine bundled skills to remove "
                "(nothing tracked, or all are user-modified/local).[/]\n")
        return
    c.print(f"\n[bold]Will remove {len(candidates)} unmodified bundled skill(s):[/]")
    c.print(f"[dim]{', '.join(candidates)}[/]")
    if preview["skipped"]:
        c.print(f"[dim]Keeping {len(preview['skipped'])} (user-modified or non-bundled).[/]")
    if not skip_confirm and not _confirm_or_cancel(
        c, "[dim]This deletes the on-disk copies. User-edited and hub/local skills are NOT touched.[/]",
        cancel="[dim]Marker kept; no skills deleted.[/]\n"):
        return
    result = remove_pristine_bundled_skills(dry_run=False)
    c.print(f"[bold green]{result['message']}[/]")
    _print_listed(c, "Removed", result["removed"])
    c.print()
    _finish_change(c, invalidate_cache, notice=False)


def do_opt_in(sync: bool = False, console: Optional[Console] = None,
              invalidate_cache: bool = True) -> None:
    """Remove the opt-out marker so bundled-skill seeding resumes."""
    from tools.skills_sync import sync_skills
    from tools.skills_sync_bundled_ops import set_bundled_skills_opt_out
    c = console or _console
    if not _report_ok(c, set_bundled_skills_opt_out(False)):
        return
    if sync:
        copied = len(sync_skills(quiet=True).get("copied", []))
        c.print(f"[dim]Re-seeded {copied} bundled skill(s).[/]")
        _finish_change(c, invalidate_cache, notice=False)
    c.print()


def do_repair_official(name: str, restore: bool = False, console: Optional[Console] = None,
                       skip_confirm: bool = False, invalidate_cache: bool = True) -> None:
    """Backfill or restore official optional skills from repo source."""
    from tools.skills_sync_optional import restore_official_optional_skill
    c = console or _console
    if restore and not skip_confirm and not _confirm_or_cancel(
        c, f"\n[bold]Restore official optional skill '{name}' from repo source?[/]",
        "[dim]Existing matching active copies will be moved to a restore backup before copying the official source.[/]",
    ):
        return
    result = restore_official_optional_skill(name, restore=restore)
    if not _report_ok(c, result, "Repair failed"):
        return
    _print_listed(c, "Restored", result.get("restored"))
    _print_listed(c, "Backfilled provenance", result.get("backfilled"))
    if result.get("backed_up"):
        c.print(f"[dim]Backed up: {', '.join(result['backed_up'])}[/]")
        c.print(f"[dim]Backup dir: {result.get('backup_dir')}[/]")
    c.print()
    _finish_change(c, invalidate_cache, notice=False)


# --- taps / publish / snapshot ---

# action -> (TapsManager method, success line, failure line)
_TAP_OPS = {
    "add": ("add", "[bold green]Added tap:[/] {repo}\n", "[yellow]Tap already exists:[/] {repo}\n"),
    "remove": ("remove", "[bold green]Removed tap:[/] {repo}\n", "[bold red]Error:[/] Tap not found: {repo}\n"),
}


def do_tap(action: str, repo: str = "", console: Optional[Console] = None) -> None:
    """Manage taps (custom GitHub repo sources)."""
    from tools.skills_hub import TapsManager
    c = console or _console
    mgr = TapsManager()
    if action == "list":
        taps = mgr.list_taps()
        if not taps:
            c.print("[dim]No custom taps configured. Using default sources only.[/]\n")
            return
        table = _table(("Repo", {"style": "bold cyan"}), "Path", title="Configured Taps")
        for t in taps:
            table.add_row(t.get("repo") or t.get("name") or t.get("path", "unknown"),
                          t.get("path", "skills/"))
        c.print(table)
        c.print()
    elif action in _TAP_OPS:
        method, ok_line, fail_line = _TAP_OPS[action]
        if not repo:
            _print_error(c, f"Repo required. Usage: hermes skills tap {action} owner/repo")
            return
        c.print((ok_line if getattr(mgr, method)(repo) else fail_line).format(repo=repo))
    else:
        c.print(f"[bold red]Unknown tap action:[/] {action}. Use: list, add, remove\n")


def _read_frontmatter(skill_md: str) -> dict:
    """YAML frontmatter of a SKILL.md body ({} when absent/invalid)."""
    import yaml
    match = re.search(r'\n---\s*\n', skill_md[3:]) if skill_md.startswith("---") else None
    try:
        return (yaml.safe_load(skill_md[3:match.start() + 3]) or {}) if match else {}
    except yaml.YAMLError:
        return {}


def do_publish(skill_path: str, target: str = "github", repo: str = "",
               console: Optional[Console] = None) -> None:
    """Publish a local skill to a registry (GitHub PR or ClawHub submission)."""
    from tools.skills_hub import SKILLS_DIR
    from tools.skills_hub_github import GitHubAuth
    from tools.skills_guard import scan_skill, format_scan_report
    c = console or _console
    path = Path(skill_path)
    if not path.is_absolute():
        path = SKILLS_DIR / path
    if not (path / "SKILL.md").exists():
        _print_error(c, f"No SKILL.md found at {path}")
        return
    skill_md = (path / "SKILL.md").read_text(encoding="utf-8").lstrip("\ufeff")  # tolerate BOM
    fm = _read_frontmatter(skill_md)
    name = fm.get("name", path.name)
    if not fm.get("description", ""):
        _print_error(c, "SKILL.md must have a 'description' in frontmatter.")
        return

    c.print(f"[bold]Scanning '{name}' before publish...[/]")
    # The only local-author path: `path` is the author's own working copy, so their `.skillignore`
    # (docs/, node_modules/, ...) still applies. Every fetched-bundle path scans the whole tree.
    result = scan_skill(path, source="self", honor_ignore_file=True)
    c.print(format_scan_report(result))
    if result.verdict == "dangerous":
        c.print("[bold red]Cannot publish a skill with DANGEROUS verdict.[/]\n")
        return

    if target == "github":
        if not repo:
            _print_error(c, "--repo required for GitHub publish.\n"
                            "Usage: hermes skills publish <path> --to github --repo owner/repo")
            return
        auth = GitHubAuth()
        if not auth.is_authenticated():
            _print_error(c, "GitHub authentication required.\n"
                            f"Set GITHUB_TOKEN in {display_hermes_home()}/.env "
                            "or run 'gh auth login'.")
            return
        c.print(f"[bold]Publishing '{name}' to {repo}...[/]")
        _report_pair(c, *_github_publish(path, name, repo, auth))
    elif target == "clawhub":
        c.print("[yellow]ClawHub publishing is not yet supported. "
                "Submit manually at https://clawhub.ai/submit[/]\n")
    else:
        c.print(f"[bold red]Unknown target:[/] {target}. Use 'github' or 'clawhub'.\n")


def _github_publish(skill_path: Path, skill_name: str, target_repo: str, auth) -> tuple:
    """Fork, branch, upload, and open a PR with the skill. Returns (success, message)."""
    import base64
    import httpx
    headers = auth.get_headers()
    api = "https://api.github.com/repos"

    def call(method: str, path: str, timeout: int = 15, **kw):
        return getattr(httpx, method)(f"{api}/{path}", headers=headers, timeout=timeout, **kw)

    try:
        resp = call("post", f"{target_repo}/forks", timeout=30)
        if resp.status_code in {200, 202}:
            fork_repo = resp.json()["full_name"]
        elif resp.status_code == 403:
            return False, "GitHub token lacks permission to fork repos"
        else:
            return False, f"Failed to fork {target_repo}: {resp.status_code}"
    except httpx.HTTPError as e:
        return False, f"Network error forking repo: {e}"

    try:
        default_branch = call("get", target_repo).json().get("default_branch", "main")
    except Exception:
        default_branch = "main"
    try:
        ref = call("get", f"{fork_repo}/git/refs/heads/{default_branch}").json()
        base_sha = ref["object"]["sha"]
    except Exception as e:
        return False, f"Failed to get base branch: {e}"

    branch_name = f"add-skill-{skill_name}"
    try:
        call("post", f"{fork_repo}/git/refs",
             json={"ref": f"refs/heads/{branch_name}", "sha": base_sha})
    except Exception as e:
        return False, f"Failed to create branch: {e}"

    for f in skill_path.rglob("*"):
        if not f.is_file():
            continue
        rel = str(f.relative_to(skill_path))
        try:
            call("put", f"{fork_repo}/contents/skills/{skill_name}/{rel}",
                 json={"message": f"Add {skill_name} skill: {rel}",
                       "content": base64.b64encode(f.read_bytes()).decode(), "branch": branch_name})
        except Exception as e:
            return False, f"Failed to upload {rel}: {e}"

    try:
        resp = call("post", f"{target_repo}/pulls", json={
            "title": f"Add skill: {skill_name}",
            "body": f"Submitting the `{skill_name}` skill via Hermes Skills Hub.\n\n"
                    f"This skill was scanned by the Hermes Skills Guard before submission.",
            "head": f"{fork_repo.split('/')[0]}:{branch_name}", "base": default_branch})
        if resp.status_code == 201:
            return True, f"PR created: {resp.json().get('html_url', '')}"
        return False, f"Failed to create PR: {resp.status_code} {resp.text[:200]}"
    except httpx.HTTPError as e:
        return False, f"Network error creating PR: {e}"


def do_snapshot_export(output_path: str, console: Optional[Console] = None) -> None:
    """Export current hub skill configuration to a portable JSON file."""
    from tools.skills_hub import HubLockFile, TapsManager
    c = console or _console
    installed = HubLockFile().list_installed()
    tap_list = TapsManager().list_taps()
    snapshot = {
        "hermes_version": "0.1.0",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "skills": [
            {"name": entry["name"], "source": entry.get("source", ""),
             "identifier": entry.get("identifier", ""),
             "category": (str(Path(entry["install_path"]).parent)
                          if "/" in entry.get("install_path", "") else "")}
            for entry in installed],
        "taps": tap_list}
    payload = json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n"
    if output_path == "-":
        sys.stdout.write(payload)
        return
    Path(output_path).write_text(payload, encoding="utf-8")
    c.print(f"[bold green]Snapshot exported:[/] {Path(output_path)}")
    c.print(f"[dim]{len(installed)} skill(s), {len(tap_list)} tap(s)[/]\n")


def do_snapshot_import(input_path: str, force: bool = False,
                       console: Optional[Console] = None) -> None:
    """Re-install skills from a snapshot file."""
    from tools.skills_hub import TapsManager
    c = console or _console
    inp = Path(input_path)
    if not inp.exists():
        _print_error(c, f"File not found: {inp}")
        return
    try:
        snapshot = json.loads(inp.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _print_error(c, f"Invalid JSON in {inp}")
        return

    taps = snapshot.get("taps", [])
    if taps:
        mgr = TapsManager()
        for tap in taps:
            if tap.get("repo", ""):
                mgr.add(tap["repo"], tap.get("path", "skills/"))
        c.print(f"[dim]Restored {len(taps)} tap(s)[/]")

    skills = snapshot.get("skills", [])
    if not skills:
        c.print("[dim]No skills in snapshot to install.[/]\n")
        return
    c.print(f"[bold]Importing {len(skills)} skill(s) from snapshot...[/]\n")
    for entry in skills:
        identifier = entry.get("identifier", "")
        if not identifier:
            c.print(f"[yellow]Skipping entry with no identifier: {entry.get('name', '?')}[/]")
            continue
        c.print(f"[bold]--- {entry.get('name', identifier)} ---[/]")
        do_install(identifier, category=entry.get("category", ""), force=force, console=c)
    c.print("[bold green]Snapshot import complete.[/]\n")


# --- CLI argparse entry point ---

def _snapshot_cli(args) -> None:
    snap_action = getattr(args, "snapshot_action", None)
    if snap_action == "export":
        do_snapshot_export(args.output)
    elif snap_action == "import":
        do_snapshot_import(args.input, force=getattr(args, "force", False))
    else:
        _console.print("Usage: hermes skills snapshot [export|import]\n")


def _tap_cli(args) -> None:
    tap_action = getattr(args, "tap_action", None)
    if not tap_action:
        _console.print("Usage: hermes skills tap [list|add|remove]\n")
        return
    do_tap(tap_action, repo=getattr(args, "repo", "") or getattr(args, "name", ""))


# `hermes skills <action>` -> handler(args). Lambdas late-bind the do_* names so
# tests that patch("hermes_cli.skills_hub.do_install") still intercept.
_CLI_ACTIONS = {
    "browse": lambda a: do_browse(page=a.page, page_size=a.size, source=a.source),
    "search": lambda a: do_search(a.query, source=a.source, limit=a.limit,
                                  as_json=getattr(a, "json", False)),
    "install": lambda a: do_install(a.identifier, category=a.category, force=a.force,
                                    skip_confirm=getattr(a, "yes", False),
                                    name_override=getattr(a, "name", "") or ""),
    "inspect": lambda a: do_inspect(a.identifier),
    "list": lambda a: do_list(source_filter=a.source,
                              enabled_only=getattr(a, "enabled_only", False)),
    "check": lambda a: do_check(name=getattr(a, "name", None)),
    "update": lambda a: do_update(name=getattr(a, "name", None), force=getattr(a, "force", False)),
    "audit": lambda a: do_audit(name=getattr(a, "name", None), deep=getattr(a, "deep", False)),
    "uninstall": lambda a: do_uninstall(a.name, skip_confirm=getattr(a, "yes", False)),
    "reset": lambda a: do_reset(a.name, restore=getattr(a, "restore", False),
                                skip_confirm=getattr(a, "yes", False)),
    "list-modified": lambda a: do_list_modified(as_json=getattr(a, "json", False)),
    "diff": lambda a: do_diff(a.name),
    "opt-out": lambda a: do_opt_out(remove=getattr(a, "remove", False),
                                    skip_confirm=getattr(a, "yes", False)),
    "opt-in": lambda a: do_opt_in(sync=getattr(a, "sync", False)),
    "repair-official": lambda a: do_repair_official(a.name, restore=getattr(a, "restore", False),
                                                    skip_confirm=getattr(a, "yes", False)),
    "publish": lambda a: do_publish(a.skill_path, target=getattr(a, "to", "github"),
                                    repo=getattr(a, "repo", "")),
    "snapshot": _snapshot_cli, "tap": _tap_cli}


def skills_command(args) -> None:
    """Router for `hermes skills <subcommand>` — called from hermes_cli/main.py."""
    handler = _CLI_ACTIONS.get(getattr(args, "skills_action", None))
    if handler is None:
        _console.print("Usage: hermes skills [browse|search|install|inspect|list|list-modified|diff|check|update|audit|uninstall|reset|opt-out|opt-in|publish|snapshot|tap]\n")
        _console.print("Run 'hermes skills <command> --help' for details.\n")
        return
    handler(args)


# --- Slash command entry point (/skills in chat) ---

def _opt_value(args: List[str], flag: str, default: str, last: bool = False) -> str:
    """Value following `flag` (default if absent/trailing); `last` makes a repeated flag's final
    occurrence win (historical install/publish/browse behaviour), else the first."""
    hits = [args[i + 1] for i, a in enumerate(args) if a == flag and i + 1 < len(args)]
    return (hits[-1] if last else hits[0]) if hits else default


def _int_or(text: str, default: int) -> int:
    try:
        return int(text)
    except ValueError:
        return default


def _opt_int(args: List[str], flag: str, default: int) -> int:
    """Like _opt_value(last=True) but int-parsed; a non-integer keeps the default."""
    return _int_or(_opt_value(args, flag, str(default), last=True), default)


def _slash_search(args, c):
    source, limit, as_json, query_parts, i = "all", 25, False, [], 0
    while i < len(args):
        flag, value = args[i], args[i + 1] if i + 1 < len(args) else None
        takes_value = flag in ("--source", "--limit") and value is not None
        if flag == "--source" and takes_value:
            source = value
        elif flag == "--limit" and takes_value:
            limit = _int_or(value, limit)
        elif flag == "--json":
            as_json = True
        else:
            query_parts.append(flag)
        i += 2 if takes_value else 1
    do_search(" ".join(query_parts), source=source, limit=limit, console=c, as_json=as_json)


def _slash_snapshot(args, c):
    if len(args) > 1 and args[0] == "export":
        do_snapshot_export(args[1], console=c)
    elif len(args) > 1 and args[0] == "import":
        do_snapshot_import(args[1], force="--force" in args, console=c)
    else:
        c.print("[bold red]Usage:[/] /skills snapshot export <file> | /skills snapshot import <file>\n")


def _first_positional(args):
    """First argument unless it is a flag (audit's historical parse)."""
    return args[0] if args and not args[0].startswith("--") else None


# Slash commands run inside prompt_toolkit where input() hangs, so install/uninstall/reset
# always skip confirmation — typing the command is implicit consent. `--now` invalidates the
# prompt cache immediately (costs more money); the default defers to the next session.
_SLASH_ACTIONS = {
    "browse": lambda args, c: do_browse(
        page=_opt_int(args, "--page", 1), page_size=_opt_int(args, "--size", 20),
        source=_opt_value(args, "--source", "all", last=True), console=c),
    "search": _slash_search,
    "install": lambda args, c: do_install(
        args[0], category=_opt_value(args, "--category", "", last=True),
        force="--force" in args, skip_confirm=True, invalidate_cache="--now" in args,
        name_override=_opt_value(args, "--name", "", last=True), console=c),
    "inspect": lambda args, c: do_inspect(args[0], console=c),
    "list": lambda args, c: do_list(
        source_filter=_opt_value(args, "--source", "all"),
        enabled_only="--enabled-only" in args or "--enabled" in args, console=c),
    "check": lambda args, c: do_check(name=args[0] if args else None, console=c),
    "update": lambda args, c: do_update(
        name=next((a for a in args if not a.startswith("--")), None), console=c,
        force="--force" in args),
    "audit": lambda args, c: do_audit(name=_first_positional(args), console=c,
                                      deep="--deep" in args),
    "uninstall": lambda args, c: do_uninstall(
        args[0], console=c, skip_confirm=True, invalidate_cache="--now" in args),
    "reset": lambda args, c: do_reset(
        args[0], restore="--restore" in args, console=c, skip_confirm=True,
        invalidate_cache="--now" in args),
    **dict.fromkeys(("list-modified", "modified"),
                    lambda args, c: do_list_modified(console=c, as_json="--json" in args)),
    "diff": lambda args, c: do_diff(args[0], console=c),
    "publish": lambda args, c: do_publish(
        args[0], target=_opt_value(args, "--to", "github", last=True),
        repo=_opt_value(args, "--repo", "", last=True), console=c),
    "snapshot": _slash_snapshot,
    "tap": lambda args, c: (do_tap(args[0], repo=args[1] if len(args) > 1 else "", console=c)
                            if args else do_tap("list", console=c)),
    **dict.fromkeys(("help", "--help", "-h"), lambda args, c: _print_skills_help(c))}

# Actions that need at least one argument -> usage lines printed when called bare.
_SLASH_USAGE = {
    "search": ("[bold red]Usage:[/] /skills search <query> [--source skills-sh|github|official|nvidia|openai|anthropic|huggingface] [--limit N] [--json]\n",),
    "install": ("[bold red]Usage:[/] /skills install <identifier-or-url> [--name <name>] [--category <cat>] [--force] [--now]\n",),
    "inspect": ("[bold red]Usage:[/] /skills inspect <identifier>\n",),
    "uninstall": ("[bold red]Usage:[/] /skills uninstall <name> [--now]\n",),
    "reset": (
        "[bold red]Usage:[/] /skills reset <name> [--restore] [--now]\n",
        "[dim]Clears the bundled-skills manifest entry so future updates stop marking it as user-modified.[/]",
        "[dim]Pass --restore to also replace the current copy with the bundled version.[/]\n"),
    "diff": ("[bold red]Usage:[/] /skills diff <name>\n",),
    "publish": ("[bold red]Usage:[/] /skills publish <skill-path> [--to github] [--repo owner/repo]\n",),
}


def handle_skills_slash(cmd: str, console: Optional[Console] = None) -> None:
    """Parse and dispatch `/skills <subcommand> [args]` from the chat interface."""
    c = console or _console
    parts = cmd.strip().split()
    if parts and parts[0].lower() == "/skills":
        parts = parts[1:]
    if not parts:
        _print_skills_help(c)
        return
    action, args = parts[0].lower(), parts[1:]
    handler = _SLASH_ACTIONS.get(action)
    if handler is None:
        c.print(f"[bold red]Unknown action:[/] {action}")
        _print_skills_help(c)
        return
    if not args and action in _SLASH_USAGE:
        for line in _SLASH_USAGE[action]:
            c.print(line)
        return
    handler(args, c)


def _print_skills_help(console: Console) -> None:
    """Print help for the /skills slash command."""
    console.print(Panel(
        "[bold]Skills Hub Commands:[/]\n\n"
        "  [cyan]browse[/] [--source official]   Browse all available skills (paginated)\n"
        "  [cyan]search[/] <query>              Search registries for skills\n"
        "  [cyan]install[/] <identifier>        Install a skill (with security scan)\n"
        "  [cyan]inspect[/] <identifier>        Preview a skill without installing\n"
        "  [cyan]list[/] [--source hub|builtin|local] [--enabled-only]\n"
        "       List installed skills; --enabled-only filters to the active profile's live set\n"
        "  [cyan]check[/] [name]                Check hub skills for upstream updates\n"
        "  [cyan]update[/] [name]               Update hub skills with upstream changes\n"
        "  [cyan]audit[/] [name]                Re-scan hub skills for security\n"
        "  [cyan]uninstall[/] <name>            Remove a hub-installed skill\n"
        "  [cyan]list-modified[/]               List bundled skills you've edited (kept by update)\n"
        "  [cyan]diff[/] <name>                 Diff your copy of a bundled skill vs the stock version\n"
        "  [cyan]reset[/] <name> [--restore]    Reset bundled-skill tracking (fix 'user-modified' flag)\n"
        "  [cyan]publish[/] <path> --repo <r>   Publish a skill to GitHub via PR\n"
        "  [cyan]snapshot[/] export|import      Export/import skill configurations\n"
        "  [cyan]tap[/] list|add|remove         Manage skill sources\n",
        title="/skills"))
