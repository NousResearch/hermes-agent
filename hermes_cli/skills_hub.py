#!/usr/bin/env python3
"""
Skills Hub CLI — Unified interface for the Hermes Skills Hub.

Powers both:
  - `hermes skills <subcommand>` (CLI argparse entry point)
  - `/skills <subcommand>` (slash command in the interactive chat)

All logic lives in shared do_* functions. The CLI entry point and slash command
handler are thin wrappers that parse args and delegate.
"""

import json
import shutil
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Lazy imports to avoid circular dependencies and slow startup.
# tools.skills_hub and tools.skills_guard are imported inside functions.

_console = Console()


# ---------------------------------------------------------------------------
# Shared do_* functions
# ---------------------------------------------------------------------------

def _resolve_short_name(name: str, sources, console: Console) -> str:
    """
    Resolve a short skill name (e.g. 'pptx') to a full identifier by searching
    all sources. If exactly one match is found, returns its identifier. If multiple
    matches exist, shows them and asks the user to use the full identifier.
    Returns empty string if nothing found or ambiguous.
    """
    from tools.skills_hub import unified_search

    c = console or _console
    c.print(f"[dim]Resolving '{name}'...[/]")

    results = unified_search(name, sources, source_filter="all", limit=20)

    # Filter to exact name matches (case-insensitive)
    exact = [r for r in results if r.name.lower() == name.lower()]

    if len(exact) == 1:
        c.print(f"[dim]Resolved to: {exact[0].identifier}[/]")
        return exact[0].identifier

    if len(exact) > 1:
        c.print(f"\n[yellow]Multiple skills named '{name}' found:[/]")
        table = Table()
        table.add_column("Source", style="dim")
        table.add_column("Trust", style="dim")
        table.add_column("Identifier", style="bold cyan")
        for r in exact:
            trust_style = {"trusted": "green", "community": "yellow"}.get(r.trust_level, "dim")
            table.add_row(r.source, f"[{trust_style}]{r.trust_level}[/]", r.identifier)
        c.print(table)
        c.print("[bold]Use the full identifier to install a specific one.[/]\n")
        return ""

    # No exact match — check if there are partial matches to suggest
    if results:
        c.print(f"[yellow]No exact match for '{name}'. Did you mean one of these?[/]")
        for r in results[:5]:
            c.print(f"  [cyan]{r.name}[/] — {r.identifier}")
        c.print()
        return ""

    c.print(f"[bold red]Error:[/] No skill named '{name}' found in any source.\n")
    return ""


def do_search(query: str, source: str = "all", limit: int = 10,
              console: Optional[Console] = None) -> None:
    """Search registries and display results as a Rich table."""
    from tools.skills_hub import GitHubAuth, create_source_router, unified_search

    c = console or _console
    c.print(f"\n[bold]Searching for:[/] {query}")

    auth = GitHubAuth()
    sources = create_source_router(auth)
    results = unified_search(query, sources, source_filter=source, limit=limit)

    if not results:
        c.print("[dim]No skills found matching your query.[/]\n")
        return

    table = Table(title=f"Skills Hub — {len(results)} result(s)")
    table.add_column("Name", style="bold cyan")
    table.add_column("Description", max_width=60)
    table.add_column("Source", style="dim")
    table.add_column("Trust", style="dim")
    table.add_column("Identifier", style="dim")

    for r in results:
        trust_style = {"trusted": "green", "community": "yellow"}.get(r.trust_level, "dim")
        table.add_row(
            r.name,
            r.description[:60] + ("..." if len(r.description) > 60 else ""),
            r.source,
            f"[{trust_style}]{r.trust_level}[/]",
            r.identifier,
        )

    c.print(table)
    c.print("[dim]Use: hermes skills inspect <identifier> to preview, "
            "hermes skills install <identifier> to install[/]\n")


def do_install(identifier: str, category: str = "", force: bool = False,
               console: Optional[Console] = None) -> None:
    """Fetch, quarantine, scan, confirm, and install a skill."""
    from tools.skills_hub import (
        GitHubAuth, create_source_router, ensure_hub_dirs,
        quarantine_bundle, install_from_quarantine, HubLockFile,
    )
    from tools.skills_guard import scan_skill, should_allow_install, format_scan_report

    c = console or _console
    ensure_hub_dirs()

    # Resolve which source adapter handles this identifier
    auth = GitHubAuth()
    sources = create_source_router(auth)

    # If identifier looks like a short name (no slashes), resolve it via search
    if "/" not in identifier:
        identifier = _resolve_short_name(identifier, sources, c)
        if not identifier:
            return

    c.print(f"\n[bold]Fetching:[/] {identifier}")

    bundle = None
    for src in sources:
        bundle = src.fetch(identifier)
        if bundle:
            break

    if not bundle:
        c.print(f"[bold red]Error:[/] Could not fetch '{identifier}' from any source.\n")
        return

    # Check if already installed
    lock = HubLockFile()
    existing = lock.get_installed(bundle.name)
    if existing:
        c.print(f"[yellow]Warning:[/] '{bundle.name}' is already installed at {existing['install_path']}")
        if not force:
            c.print("Use --force to reinstall.\n")
            return

    # Quarantine the bundle
    q_path = quarantine_bundle(bundle)
    c.print(f"[dim]Quarantined to {q_path.relative_to(q_path.parent.parent.parent)}[/]")

    # Scan
    c.print("[bold]Running security scan...[/]")
    result = scan_skill(q_path, source=identifier)
    c.print(format_scan_report(result))

    # Check install policy
    allowed, reason = should_allow_install(result, force=force)
    if not allowed:
        c.print(f"\n[bold red]Installation blocked:[/] {reason}")
        # Clean up quarantine
        shutil.rmtree(q_path, ignore_errors=True)
        from tools.skills_hub import append_audit_log
        append_audit_log("BLOCKED", bundle.name, bundle.source,
                         bundle.trust_level, result.verdict,
                         f"{len(result.findings)}_findings")
        return

    # Confirm with user — always show risk warning regardless of source
    if not force:
        c.print()
        c.print(Panel(
            "[bold yellow]You are installing a third-party skill at your own risk.[/]\n\n"
            "External skills can contain instructions that influence agent behavior,\n"
            "shell commands, and scripts. Even after automated scanning, you should\n"
            "review the installed files before use.\n\n"
            f"Files will be at: [cyan]~/.hermes/skills/{category + '/' if category else ''}{bundle.name}/[/]",
            title="Disclaimer",
            border_style="yellow",
        ))
        c.print(f"[bold]Install '{bundle.name}'?[/]")
        try:
            answer = input("Confirm [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        if answer not in ("y", "yes"):
            c.print("[dim]Installation cancelled.[/]\n")
            shutil.rmtree(q_path, ignore_errors=True)
            return

    # Install
    install_dir = install_from_quarantine(q_path, bundle.name, category, bundle, result)
    from tools.skills_hub import SKILLS_DIR
    c.print(f"[bold green]Installed:[/] {install_dir.relative_to(SKILLS_DIR)}")
    c.print(f"[dim]Files: {', '.join(bundle.files.keys())}[/]\n")


def do_inspect(identifier: str, console: Optional[Console] = None) -> None:
    """Preview a skill's SKILL.md content without installing."""
    from tools.skills_hub import GitHubAuth, create_source_router

    c = console or _console
    auth = GitHubAuth()
    sources = create_source_router(auth)

    if "/" not in identifier:
        identifier = _resolve_short_name(identifier, sources, c)
        if not identifier:
            return

    meta = None
    for src in sources:
        meta = src.inspect(identifier)
        if meta:
            break

    if not meta:
        c.print(f"[bold red]Error:[/] Could not find '{identifier}' in any source.\n")
        return

    # Also fetch full content for preview
    bundle = None
    for src in sources:
        bundle = src.fetch(identifier)
        if bundle:
            break

    c.print()
    trust_style = {"trusted": "green", "community": "yellow"}.get(meta.trust_level, "dim")

    info_lines = [
        f"[bold]Name:[/] {meta.name}",
        f"[bold]Description:[/] {meta.description}",
        f"[bold]Source:[/] {meta.source}",
        f"[bold]Trust:[/] [{trust_style}]{meta.trust_level}[/]",
        f"[bold]Identifier:[/] {meta.identifier}",
    ]
    if meta.tags:
        info_lines.append(f"[bold]Tags:[/] {', '.join(meta.tags)}")

    c.print(Panel("\n".join(info_lines), title=f"Skill: {meta.name}"))

    if bundle and "SKILL.md" in bundle.files:
        content = bundle.files["SKILL.md"]
        # Show first 50 lines as preview
        lines = content.split("\n")
        preview = "\n".join(lines[:50])
        if len(lines) > 50:
            preview += f"\n\n... ({len(lines) - 50} more lines)"
        c.print(Panel(preview, title="SKILL.md Preview", subtitle="hermes skills install <id> to install"))

    c.print()


def do_list(source_filter: str = "all", console: Optional[Console] = None) -> None:
    """List installed skills, distinguishing builtins from hub-installed."""
    from tools.skills_hub import HubLockFile, SKILLS_DIR
    from tools.skills_tool import _find_all_skills

    c = console or _console
    lock = HubLockFile()
    hub_installed = {e["name"]: e for e in lock.list_installed()}

    all_skills = _find_all_skills()

    table = Table(title="Installed Skills")
    table.add_column("Name", style="bold cyan")
    table.add_column("Category", style="dim")
    table.add_column("Source", style="dim")
    table.add_column("Trust", style="dim")

    for skill in sorted(all_skills, key=lambda s: (s.get("category") or "", s["name"])):
        name = skill["name"]
        category = skill.get("category", "")
        hub_entry = hub_installed.get(name)

        if hub_entry:
            source_display = hub_entry.get("source", "hub")
            trust = hub_entry.get("trust_level", "community")
        else:
            source_display = "builtin"
            trust = "builtin"

        if source_filter == "hub" and not hub_entry:
            continue
        if source_filter == "builtin" and hub_entry:
            continue

        trust_style = {"builtin": "blue", "trusted": "green", "community": "yellow"}.get(trust, "dim")
        table.add_row(name, category, source_display, f"[{trust_style}]{trust}[/]")

    c.print(table)
    c.print(f"[dim]{len(hub_installed)} hub-installed, "
            f"{len(all_skills) - len(hub_installed)} builtin[/]\n")


def do_audit(name: Optional[str] = None, console: Optional[Console] = None) -> None:
    """Re-run security scan on installed hub skills."""
    from tools.skills_hub import HubLockFile, SKILLS_DIR
    from tools.skills_guard import scan_skill, format_scan_report

    c = console or _console
    lock = HubLockFile()
    installed = lock.list_installed()

    if not installed:
        c.print("[dim]No hub-installed skills to audit.[/]\n")
        return

    targets = installed
    if name:
        targets = [e for e in installed if e["name"] == name]
        if not targets:
            c.print(f"[bold red]Error:[/] '{name}' is not a hub-installed skill.\n")
            return

    c.print(f"\n[bold]Auditing {len(targets)} skill(s)...[/]\n")

    for entry in targets:
        skill_path = SKILLS_DIR / entry["install_path"]
        if not skill_path.exists():
            c.print(f"[yellow]Warning:[/] {entry['name']} — path missing: {entry['install_path']}")
            continue

        result = scan_skill(skill_path, source=entry.get("identifier", entry["source"]))
        c.print(format_scan_report(result))
        c.print()


def do_uninstall(name: str, console: Optional[Console] = None) -> None:
    """Remove a hub-installed skill with confirmation."""
    from tools.skills_hub import uninstall_skill

    c = console or _console

    c.print(f"\n[bold]Uninstall '{name}'?[/]")
    try:
        answer = input("Confirm [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = "n"
    if answer not in ("y", "yes"):
        c.print("[dim]Cancelled.[/]\n")
        return

    success, msg = uninstall_skill(name)
    if success:
        c.print(f"[bold green]{msg}[/]\n")
    else:
        c.print(f"[bold red]Error:[/] {msg}\n")


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
        table = Table(title="Configured Taps")
        table.add_column("Repo", style="bold cyan")
        table.add_column("Path", style="dim")
        for t in taps:
            table.add_row(t["repo"], t.get("path", "skills/"))
        c.print(table)
        c.print()

    elif action == "add":
        if not repo:
            c.print("[bold red]Error:[/] Repo required. Usage: hermes skills tap add owner/repo\n")
            return
        if mgr.add(repo):
            c.print(f"[bold green]Added tap:[/] {repo}\n")
        else:
            c.print(f"[yellow]Tap already exists:[/] {repo}\n")

    elif action == "remove":
        if not repo:
            c.print("[bold red]Error:[/] Repo required. Usage: hermes skills tap remove owner/repo\n")
            return
        if mgr.remove(repo):
            c.print(f"[bold green]Removed tap:[/] {repo}\n")
        else:
            c.print(f"[bold red]Error:[/] Tap not found: {repo}\n")

    else:
        c.print(f"[bold red]Unknown tap action:[/] {action}. Use: list, add, remove\n")


def do_publish(skill_path: str, target: str = "github", repo: str = "",
               console: Optional[Console] = None) -> None:
    """Publish a local skill to a registry (GitHub PR or ClawHub submission)."""
    from tools.skills_hub import GitHubAuth, SKILLS_DIR
    from tools.skills_guard import scan_skill, format_scan_report

    c = console or _console
    path = Path(skill_path)

    # Resolve relative to skills dir if not absolute
    if not path.is_absolute():
        path = SKILLS_DIR / path
    if not path.exists() or not (path / "SKILL.md").exists():
        c.print(f"[bold red]Error:[/] No SKILL.md found at {path}\n")
        return

    # Validate the skill
    import yaml
    skill_md = (path / "SKILL.md").read_text(encoding="utf-8")
    fm = {}
    if skill_md.startswith("---"):
        import re
        match = re.search(r'\n---\s*\n', skill_md[3:])
        if match:
            try:
                fm = yaml.safe_load(skill_md[3:match.start() + 3]) or {}
            except yaml.YAMLError:
                pass

    name = fm.get("name", path.name)
    description = fm.get("description", "")
    if not description:
        c.print("[bold red]Error:[/] SKILL.md must have a 'description' in frontmatter.\n")
        return

    # Self-scan before publishing
    c.print(f"[bold]Scanning '{name}' before publish...[/]")
    result = scan_skill(path, source="self")
    c.print(format_scan_report(result))
    if result.verdict == "dangerous":
        c.print("[bold red]Cannot publish a skill with DANGEROUS verdict.[/]\n")
        return

    if target == "github":
        if not repo:
            c.print("[bold red]Error:[/] --repo required for GitHub publish.\n"
                    "Usage: hermes skills publish <path> --to github --repo owner/repo\n")
            return

        auth = GitHubAuth()
        if not auth.is_authenticated():
            c.print("[bold red]Error:[/] GitHub authentication required.\n"
                    "Set GITHUB_TOKEN in ~/.hermes/.env or run 'gh auth login'.\n")
            return

        c.print(f"[bold]Publishing '{name}' to {repo}...[/]")
        success, msg = _github_publish(path, name, repo, auth)
        if success:
            c.print(f"[bold green]{msg}[/]\n")
        else:
            c.print(f"[bold red]Error:[/] {msg}\n")

    elif target == "clawhub":
        c.print("[yellow]ClawHub publishing is not yet supported. "
                "Submit manually at https://clawhub.ai/submit[/]\n")
    else:
        c.print(f"[bold red]Unknown target:[/] {target}. Use 'github' or 'clawhub'.\n")


def _github_publish(skill_path: Path, skill_name: str, target_repo: str,
                    auth) -> tuple:
    """Create a PR to a GitHub repo with the skill. Returns (success, message)."""
    import httpx

    headers = auth.get_headers()

    # 1. Fork the repo
    try:
        resp = httpx.post(
            f"https://api.github.com/repos/{target_repo}/forks",
            headers=headers, timeout=30,
        )
        if resp.status_code in (200, 202):
            fork = resp.json()
            fork_repo = fork["full_name"]
        elif resp.status_code == 403:
            return False, "GitHub token lacks permission to fork repos"
        else:
            return False, f"Failed to fork {target_repo}: {resp.status_code}"
    except httpx.HTTPError as e:
        return False, f"Network error forking repo: {e}"

    # 2. Get default branch
    try:
        resp = httpx.get(
            f"https://api.github.com/repos/{target_repo}",
            headers=headers, timeout=15,
        )
        default_branch = resp.json().get("default_branch", "main")
    except Exception:
        default_branch = "main"

    # 3. Get the base tree SHA
    try:
        resp = httpx.get(
            f"https://api.github.com/repos/{fork_repo}/git/refs/heads/{default_branch}",
            headers=headers, timeout=15,
        )
        base_sha = resp.json()["object"]["sha"]
    except Exception as e:
        return False, f"Failed to get base branch: {e}"

    # 4. Create a new branch
    branch_name = f"add-skill-{skill_name}"
    try:
        httpx.post(
            f"https://api.github.com/repos/{fork_repo}/git/refs",
            headers=headers, timeout=15,
            json={"ref": f"refs/heads/{branch_name}", "sha": base_sha},
        )
    except Exception as e:
        return False, f"Failed to create branch: {e}"

    # 5. Upload skill files
    for f in skill_path.rglob("*"):
        if not f.is_file():
            continue
        rel = str(f.relative_to(skill_path))
        upload_path = f"skills/{skill_name}/{rel}"
        try:
            import base64
            content_b64 = base64.b64encode(f.read_bytes()).decode()
            httpx.put(
                f"https://api.github.com/repos/{fork_repo}/contents/{upload_path}",
                headers=headers, timeout=15,
                json={
                    "message": f"Add {skill_name} skill: {rel}",
                    "content": content_b64,
                    "branch": branch_name,
                },
            )
        except Exception as e:
            return False, f"Failed to upload {rel}: {e}"

    # 6. Create PR
    try:
        resp = httpx.post(
            f"https://api.github.com/repos/{target_repo}/pulls",
            headers=headers, timeout=15,
            json={
                "title": f"Add skill: {skill_name}",
                "body": f"Submitting the `{skill_name}` skill via Hermes Skills Hub.\n\n"
                        f"This skill was scanned by the Hermes Skills Guard before submission.",
                "head": f"{fork_repo.split('/')[0]}:{branch_name}",
                "base": default_branch,
            },
        )
        if resp.status_code == 201:
            pr_url = resp.json().get("html_url", "")
            return True, f"PR created: {pr_url}"
        else:
            return False, f"Failed to create PR: {resp.status_code} {resp.text[:200]}"
    except httpx.HTTPError as e:
        return False, f"Network error creating PR: {e}"


def do_snapshot_export(output_path: str, console: Optional[Console] = None) -> None:
    """Export current hub skill configuration to a portable JSON file."""
    from tools.skills_hub import HubLockFile, TapsManager

    c = console or _console
    lock = HubLockFile()
    taps = TapsManager()

    installed = lock.list_installed()
    tap_list = taps.list_taps()

    snapshot = {
        "hermes_version": "0.1.0",
        "exported_at": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "skills": [
            {
                "name": entry["name"],
                "source": entry.get("source", ""),
                "identifier": entry.get("identifier", ""),
                "category": str(Path(entry.get("install_path", "")).parent)
                            if "/" in entry.get("install_path", "") else "",
            }
            for entry in installed
        ],
        "taps": tap_list,
    }

    out = Path(output_path)
    out.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n")
    c.print(f"[bold green]Snapshot exported:[/] {out}")
    c.print(f"[dim]{len(installed)} skill(s), {len(tap_list)} tap(s)[/]\n")


def do_snapshot_import(input_path: str, force: bool = False,
                       console: Optional[Console] = None) -> None:
    """Re-install skills from a snapshot file."""
    from tools.skills_hub import TapsManager

    c = console or _console
    inp = Path(input_path)
    if not inp.exists():
        c.print(f"[bold red]Error:[/] File not found: {inp}\n")
        return

    try:
        snapshot = json.loads(inp.read_text())
    except json.JSONDecodeError:
        c.print(f"[bold red]Error:[/] Invalid JSON in {inp}\n")
        return

    # Restore taps first
    taps = snapshot.get("taps", [])
    if taps:
        mgr = TapsManager()
        for tap in taps:
            repo = tap.get("repo", "")
            if repo:
                mgr.add(repo, tap.get("path", "skills/"))
        c.print(f"[dim]Restored {len(taps)} tap(s)[/]")

    # Install skills
    skills = snapshot.get("skills", [])
    if not skills:
        c.print("[dim]No skills in snapshot to install.[/]\n")
        return

    c.print(f"[bold]Importing {len(skills)} skill(s) from snapshot...[/]\n")
    for entry in skills:
        identifier = entry.get("identifier", "")
        category = entry.get("category", "")
        if not identifier:
            c.print(f"[yellow]Skipping entry with no identifier: {entry.get('name', '?')}[/]")
            continue

        c.print(f"[bold]--- {entry.get('name', identifier)} ---[/]")
        do_install(identifier, category=category, force=force, console=c)

    c.print("[bold green]Snapshot import complete.[/]\n")


# ---------------------------------------------------------------------------
# CLI argparse entry point
# ---------------------------------------------------------------------------

def skills_command(args) -> None:
    """Router for `hermes skills <subcommand>` — called from hermes_cli/main.py."""
    action = getattr(args, "skills_action", None)

    if action == "search":
        do_search(args.query, source=args.source, limit=args.limit)
    elif action == "install":
        do_install(args.identifier, category=args.category, force=args.force)
    elif action == "inspect":
        do_inspect(args.identifier)
    elif action == "list":
        do_list(source_filter=args.source)
    elif action == "audit":
        do_audit(name=getattr(args, "name", None))
    elif action == "uninstall":
        do_uninstall(args.name)
    elif action == "publish":
        do_publish(
            args.skill_path,
            target=getattr(args, "to", "github"),
            repo=getattr(args, "repo", ""),
        )
    elif action == "snapshot":
        snap_action = getattr(args, "snapshot_action", None)
        if snap_action == "export":
            do_snapshot_export(args.output)
        elif snap_action == "import":
            do_snapshot_import(args.input, force=getattr(args, "force", False))
        else:
            _console.print("Usage: hermes skills snapshot [export|import]\n")
    elif action == "tap":
        tap_action = getattr(args, "tap_action", None)
        repo = getattr(args, "repo", "") or getattr(args, "name", "")
        if not tap_action:
            _console.print("Usage: hermes skills tap [list|add|remove]\n")
            return
        do_tap(tap_action, repo=repo)
    else:
        _console.print("Usage: hermes skills [search|install|inspect|list|audit|uninstall|publish|snapshot|tap]\n")
        _console.print("Run 'hermes skills <command> --help' for details.\n")


# ---------------------------------------------------------------------------
# Slash command entry point (/skills in chat)
# ---------------------------------------------------------------------------

def handle_skills_slash(cmd: str, console: Optional[Console] = None) -> None:
    """
    Parse and dispatch `/skills <subcommand> [args]` from the chat interface.

    Examples:
        /skills search kubernetes
        /skills install openai/skills/skill-creator
        /skills install openai/skills/skill-creator --force
        /skills inspect openai/skills/skill-creator
        /skills list
        /skills list --source hub
        /skills audit
        /skills audit my-skill
        /skills uninstall my-skill
        /skills tap list
        /skills tap add owner/repo
        /skills tap remove owner/repo
    """
    c = console or _console
    parts = cmd.strip().split()

    # Strip the leading "/skills" if present
    if parts and parts[0].lower() == "/skills":
        parts = parts[1:]

    if not parts:
        _print_skills_help(c)
        return

    action = parts[0].lower()
    args = parts[1:]

    if action == "search":
        if not args:
            c.print("[bold red]Usage:[/] /skills search <query> [--source github] [--limit N]\n")
            return
        source = "all"
        limit = 10
        query_parts = []
        i = 0
        while i < len(args):
            if args[i] == "--source" and i + 1 < len(args):
                source = args[i + 1]
                i += 2
            elif args[i] == "--limit" and i + 1 < len(args):
                try:
                    limit = int(args[i + 1])
                except ValueError:
                    pass
                i += 2
            else:
                query_parts.append(args[i])
                i += 1
        do_search(" ".join(query_parts), source=source, limit=limit, console=c)

    elif action == "install":
        if not args:
            c.print("[bold red]Usage:[/] /skills install <identifier> [--category <cat>] [--force]\n")
            return
        identifier = args[0]
        category = ""
        force = "--force" in args
        for i, a in enumerate(args):
            if a == "--category" and i + 1 < len(args):
                category = args[i + 1]
        do_install(identifier, category=category, force=force, console=c)

    elif action == "inspect":
        if not args:
            c.print("[bold red]Usage:[/] /skills inspect <identifier>\n")
            return
        do_inspect(args[0], console=c)

    elif action == "list":
        source_filter = "all"
        if "--source" in args:
            idx = args.index("--source")
            if idx + 1 < len(args):
                source_filter = args[idx + 1]
        do_list(source_filter=source_filter, console=c)

    elif action == "audit":
        name = args[0] if args else None
        do_audit(name=name, console=c)

    elif action == "uninstall":
        if not args:
            c.print("[bold red]Usage:[/] /skills uninstall <name>\n")
            return
        do_uninstall(args[0], console=c)

    elif action == "publish":
        if not args:
            c.print("[bold red]Usage:[/] /skills publish <skill-path> [--to github] [--repo owner/repo]\n")
            return
        skill_path = args[0]
        target = "github"
        repo = ""
        for i, a in enumerate(args):
            if a == "--to" and i + 1 < len(args):
                target = args[i + 1]
            if a == "--repo" and i + 1 < len(args):
                repo = args[i + 1]
        do_publish(skill_path, target=target, repo=repo, console=c)

    elif action == "snapshot":
        if not args:
            c.print("[bold red]Usage:[/] /skills snapshot export <file> | /skills snapshot import <file>\n")
            return
        snap_action = args[0]
        if snap_action == "export" and len(args) > 1:
            do_snapshot_export(args[1], console=c)
        elif snap_action == "import" and len(args) > 1:
            force = "--force" in args
            do_snapshot_import(args[1], force=force, console=c)
        else:
            c.print("[bold red]Usage:[/] /skills snapshot export <file> | /skills snapshot import <file>\n")

    elif action == "tap":
        if not args:
            do_tap("list", console=c)
            return
        tap_action = args[0]
        repo = args[1] if len(args) > 1 else ""
        do_tap(tap_action, repo=repo, console=c)

    elif action in ("help", "--help", "-h"):
        _print_skills_help(c)

    else:
        c.print(f"[bold red]Unknown action:[/] {action}")
        _print_skills_help(c)


def _print_skills_help(console: Console) -> None:
    """Print help for the /skills slash command."""
    console.print(Panel(
        "[bold]Skills Hub Commands:[/]\n\n"
        "  [cyan]search[/] <query>              Search registries for skills\n"
        "  [cyan]install[/] <identifier>        Install a skill (with security scan)\n"
        "  [cyan]inspect[/] <identifier>        Preview a skill without installing\n"
        "  [cyan]list[/] [--source hub|builtin] List installed skills\n"
        "  [cyan]audit[/] [name]                Re-scan hub skills for security\n"
        "  [cyan]uninstall[/] <name>            Remove a hub-installed skill\n"
        "  [cyan]publish[/] <path> --repo <r>   Publish a skill to GitHub via PR\n"
        "  [cyan]snapshot[/] export|import      Export/import skill configurations\n"
        "  [cyan]tap[/] list|add|remove         Manage skill sources\n",
        title="/skills",
    ))
