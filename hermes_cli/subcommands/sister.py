#!/usr/bin/env python3
"""``hermes sister`` subcommand — manage the 12-Sister Personality System.

Usage::

    hermes sister list                    # list all sisters with status
    hermes sister show <id>               # show full sister details
    hermes sister match <task>            # match a task to the best sister
    hermes sister reload                  # reload sisters.yaml from disk
    hermes sister run <id> <prompt>       # one-shot chat as a specific sister
    hermes sister status                  # show active sister registry info

"""

from __future__ import annotations

import sys
import os
import json
from typing import Optional

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_registry():
    """Import sister_registry, adjusting sys.path if needed."""
    try:
        from agent import sister_registry
        return sister_registry
    except ImportError:
        pass

    # When run from the hermes CLI, the agent package may not be on sys.path.
    # Try common locations.
    candidates = [
        os.environ.get("HERMES_AGENT_DIR", ""),
        os.path.expanduser("~/.hermes/hermes-agent"),
        os.path.join(os.path.dirname(__file__), "..", ".."),
    ]
    for d in candidates:
        if d and os.path.isdir(os.path.join(d, "agent")) and d not in sys.path:
            sys.path.insert(0, d)

    from agent import sister_registry
    return sister_registry


def _load_prompt_loader():
    """Import sister_prompt_loader, adjusting sys.path if needed."""
    try:
        from agent import sister_prompt_loader
        return sister_prompt_loader
    except ImportError:
        pass

    candidates = [
        os.environ.get("HERMES_AGENT_DIR", ""),
        os.path.expanduser("~/.hermes/hermes-agent"),
        os.path.join(os.path.dirname(__file__), "..", ".."),
    ]
    for d in candidates:
        if d and os.path.isdir(os.path.join(d, "agent")) and d not in sys.path:
            sys.path.insert(0, d)

    from agent import sister_prompt_loader
    return sister_prompt_loader


def _banner():
    print("=" * 60)
    print("  🌟 Hermes 12-Sister Personality System — CLI")
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_list(args) -> None:
    """List all sisters in a formatted table."""
    registry_module = _load_registry()
    registry = registry_module.get_registry()
    sisters = registry.list_sisters()

    if not sisters:
        print("No sisters found. Check ~/.hermes/config/sisters.yaml")
        return

    print(f"{'ID':<12} {'Name':<10} {'Role':<28} {'Domain':<18} {'Risk':<8} {'Enabled'}")
    print("-" * 94)
    for s in sisters:
        sid = s.get("id", "")
        name = s.get("name", "")
        role = s.get("role", "")[:27]
        domain = s.get("domain", "")[:17]
        risk = s.get("risk_level", "")
        enabled = "yes" if s.get("enabled", True) else "no"
        print(f"{sid:<12} {name:<10} {role:<28} {domain:<18} {risk:<8} {enabled}")

    print(f"\n{len(sisters)} sisters loaded.")
    print("Legacy aliases: fofoqueiro→bia, vini→vitoria, larissinha→larissa, daiane→daine")


def cmd_show(args) -> None:
    """Show full details for one sister."""
    registry_module = _load_registry()
    registry = registry_module.get_registry()
    prompt_loader_module = _load_prompt_loader()
    prompt_loader = prompt_loader_module.get_prompt_loader()

    sister_id = getattr(args, "sister_id", None)
    if not sister_id:
        print("Usage: hermes sister show <id>")
        sys.exit(1)

    sister = registry.get(sister_id)
    if not sister:
        print(f"Sister '{sister_id}' not found.")
        sys.exit(1)

    prompt = prompt_loader.build_sister_prompt(sister)

    print(f"\n{prompt.metadata.get('role', '')}  {sister.name} ({sister.id})")
    print(f"  Archetype:   {sister.archetype}")
    print(f"  Domain:      {sister.domain}")
    print(f"  Risk Level:  {sister.risk_level}")
    print(f"  Model Hint:  {sister.model_preference or 'none'}")
    if sister.legacy_aliases:
        print(f"  Legacy IDs:  {', '.join(sister.legacy_aliases)}")
    print(f"  Delegation:  {', '.join(sister.delegation_scope)}")
    print(f"  Enabled:     {sister.enabled}")
    print()
    print("  System Prompt:")
    print("-" * 60)
    # Show first 20 lines
    lines = prompt.system_prompt.strip().split("\n")
    for line in lines[:20]:
        print(f"  {line}")
    if len(lines) > 20:
        print(f"  ... ({len(lines) - 20} more lines)")


def cmd_match(args) -> None:
    """Match a task description to the best sister."""
    registry_module = _load_registry()
    registry = registry_module.get_registry()

    task = getattr(args, "task_text", "")
    if isinstance(task, list):
        task = " ".join(task)
    if not task:
        print("Usage: hermes sister match <task description>")
        sys.exit(1)

    matches = registry.match_task(task, top_k=3)

    if matches:
        print(f"\n  Task:   \"{task[:80]}{'...' if len(task) > 80 else ''}\"")
        print(f"  Matches (top {len(matches)}):")
        for i, sister in enumerate(matches, 1):
            score = sister.matches_keywords(task)
            print(f"    {i}. {sister.name} ({sister.id}) — score: {score:.1f}")
            print(f"       Domain: {sister.domain} | Scope: {', '.join(sister.delegation_scope[:3])}")
    else:
        print(f'\n  Task:   "{task[:80]}{"..." if len(task) > 80 else ""}"')
        print(f"  Match:  none — will use fallback: astra")


def cmd_reload(args) -> None:
    """Reload sisters.yaml from disk."""
    registry_module = _load_registry()
    reload_registry = getattr(registry_module, 'reload_registry', None)
    if reload_registry:
        reload_registry()
        registry = registry_module.get_registry()
        sisters = registry.list_sisters()
        print(f"Reloaded sisters.yaml: {len(sisters)} sisters loaded.")
    else:
        print("Reload not available in this version.")


def cmd_status(args) -> None:
    """Show active sister registry info."""
    registry_module = _load_registry()
    registry = registry_module.get_registry()
    sisters = registry.get_all()

    config_path = os.path.expanduser("~/.hermes/config/sisters.yaml")

    print(f"  Config path:    {config_path}")
    print(f"  Sisters loaded: {len(sisters)}")
    print(f"  Fallback:       astra (orchestrator)")
    print(f"  HARP routing:   SEPARATE — model selection not in sister prompts")
    print()
    print("  Delegation map:")
    print("    Code → Novus | Vision/Web → Nova | Legal → Helena | Risk → Bia")
    print("    Sales → Clara | Support → Daine | Creative → Vitoria | Data → Daine")
    print("    Research → Luna | Implementation → Maya | Orchestration → Astra")
    print()
    print("  Sister roster:")
    for s in sisters:
        print(f"    {s.id:<10} {s.name:<10} [{s.domain}] {s.role}")


def cmd_run(args) -> None:
    """One-shot chat as a specific sister."""
    registry_module = _load_registry()
    registry = registry_module.get_registry()
    prompt_loader_module = _load_prompt_loader()
    prompt_loader = prompt_loader_module.get_prompt_loader()

    sister_id = getattr(args, "sister_id", None)
    prompt_text = getattr(args, "prompt", None)

    if not sister_id or not prompt_text:
        print("Usage: hermes sister run <id> <prompt>")
        sys.exit(1)

    sister = registry.get(sister_id)
    if not sister:
        print(f"Sister '{sister_id}' not found.")
        sys.exit(1)

    # Load the sister's system prompt
    sp = prompt_loader.build_sister_prompt(sister)
    system_prompt = sp.to_system_prompt()

    # Run a one-shot using the model_tools
    try:
        import model_tools
        # Get default tool definitions
        tool_defs = model_tools.get_tool_definitions(quiet_mode=True)

        # For now, just show what would happen
        print(f"\nWould run one-shot as {sister.name} ({sister.id})")
        print(f"System prompt: {system_prompt[:200]}...")
        print(f"Prompt: {prompt_text}")
        print("\nNote: Full one-shot execution requires running inside an agent context.")
        print("Use 'hermes chat' and mention the sister by name for interactive use.")
    except Exception as e:
        print(f"Error: {e}")


# ---------------------------------------------------------------------------
# Parser builder
# ---------------------------------------------------------------------------

def build_sister_parser(subparsers, *, cmd_sister=None) -> None:
    """Attach the ``sister`` subcommand to ``subparsers``."""
    sister_parser = subparsers.add_parser(
        "sister",
        help="Manage the 12-Sister Personality System",
    )
    sister_parser.set_defaults(func=cmd_sister)
    sister_subparsers = sister_parser.add_subparsers(dest="sister_action")

    # `hermes sister list`
    sister_subparsers.add_parser("list", help="List all sisters")

    # `hermes sister show <id>`
    show_parser = sister_subparsers.add_parser("show", help="Show sister details")
    show_parser.add_argument("sister_id", help="Sister ID (e.g., novus, astra)")

    # `hermes sister match <task>`
    match_parser = sister_subparsers.add_parser("match", help="Match task to best sister")
    match_parser.add_argument("task_text", nargs="+", help="Task description")

    # `hermes sister reload`
    sister_subparsers.add_parser("reload", help="Reload sisters.yaml from disk")

    # `hermes sister status`
    sister_subparsers.add_parser("status", help="Show registry status")

    # `hermes sister run <id> <prompt>`
    run_parser = sister_subparsers.add_parser("run", help="One-shot chat as a specific sister")
    run_parser.add_argument("sister_id", help="Sister ID (e.g., novus, astra)")
    run_parser.add_argument("prompt", help="Prompt to send to the sister")


def cmd_sister(args) -> None:
    """Dispatch sister subcommand."""
    action = getattr(args, "sister_action", None)

    if action is None:
        cmd_status(args)
        return

    dispatch = {
        "list": cmd_list,
        "show": cmd_show,
        "match": cmd_match,
        "reload": cmd_reload,
        "status": cmd_status,
        "run": cmd_run,
    }

    handler = dispatch.get(action)
    if handler:
        handler(args)
    else:
        print(f"Unknown sister action: {action}")
        sys.exit(1)