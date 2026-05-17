"""
hermes agent — Manage multi-agent profiles and routing.

  hermes agent list          Show all agents, models, and route counts
  hermes agent show <id>     Display agent details (paths, routes, SOUL)
  hermes agent add <id>      Add a new agent to config.yaml
  hermes agent remove <id>   Remove an agent (warns about orphaned routes)
"""

import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml

from hermes_cli.colors import color, Colors
from hermes_cli.config import load_config, save_config
from hermes_constants import get_hermes_home


def _load_config() -> Dict[str, Any]:
    cfg = load_config()
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        cfg = {}
    return cfg


def _ensure_agent_section(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if "agents" not in cfg:
        cfg["agents"] = {}
    if not isinstance(cfg["agents"], dict):
        cfg["agents"] = {}
    if "routes" not in cfg:
        cfg["routes"] = []
    if not isinstance(cfg["routes"], list):
        cfg["routes"] = []
    if "default_agent" not in cfg:
        cfg["default_agent"] = "main"
    return cfg


def _count_routes_for_agent(cfg: Dict[str, Any], agent_id: str) -> int:
    routes = cfg.get("routes", [])
    return sum(1 for r in routes if isinstance(r, dict) and r.get("agent") == agent_id)


def _routes_for_agent(cfg: Dict[str, Any], agent_id: str) -> List[Dict[str, Any]]:
    routes = cfg.get("routes", [])
    return [r for r in routes if isinstance(r, dict) and r.get("agent") == agent_id]


def _summarize_soul(path: Path, max_lines: int = 8) -> str:
    if not path.exists():
        return "(no SOUL.md)"
    lines = path.read_text(encoding="utf-8").splitlines()
    non_empty = [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]
    preview = " ".join(non_empty[:max_lines])
    if len(preview) > 200:
        preview = preview[:200] + "..."
    return preview or "(empty SOUL.md)"


def _green(text: str) -> str:
    return color(text, Colors.GREEN)


def _red(text: str) -> str:
    return color(text, Colors.RED)


def _yellow(text: str) -> str:
    return color(text, Colors.YELLOW)


def cmd_agent_list(args) -> int:
    """List all agents with model, home dir, and route count."""
    cfg = _ensure_agent_section(_load_config())
    agents = cfg.get("agents", {})
    default_agent = cfg.get("default_agent", "main")

    if not agents:
        print("No agents configured. Run 'hermes agent add <id>' to create one.")
        return 0

    # Determine column widths
    id_width = max(len(str(aid)) for aid in agents.keys())
    id_width = max(id_width, 6)

    header = f"{'ID':<{id_width}}  {'Model':<28} {'Routes':>6}  {'Home Dir'}"
    print(color(header, Colors.BOLD))
    print("-" * (id_width + 2 + 28 + 1 + 6 + 2 + 10))

    for aid, spec in agents.items():
        if not isinstance(spec, dict):
            spec = {}
        model = spec.get("model", "(default)")
        home = spec.get("home_dir", "(default)")
        route_count = _count_routes_for_agent(cfg, aid)
        marker = " *" if aid == default_agent else "  "
        print(f"{marker}{aid:<{id_width}}  {model:<28} {route_count:>6}  {home}")

    print(f"\n* = default agent ({default_agent})")
    return 0


def cmd_agent_show(args) -> int:
    """Show detailed info for a single agent."""
    cfg = _ensure_agent_section(_load_config())
    agent_id = args.agent_id
    agents = cfg.get("agents", {})

    if agent_id not in agents:
        print(_red(f"Agent '{agent_id}' not found."))
        print("Run 'hermes agent list' to see available agents.")
        return 1

    spec = agents[agent_id]
    if not isinstance(spec, dict):
        spec = {}

    home_dir = spec.get("home_dir")
    if home_dir:
        home_path = Path(home_dir).expanduser()
    else:
        home_path = get_hermes_home()

    print(color(f"Agent: {agent_id}", Colors.BOLD))
    print(f"  Model:     {spec.get('model', '(default)')}")
    print(f"  Provider:  {spec.get('provider', '(default)')}")
    print(f"  Home Dir:  {home_path}")
    print(f"  Memory:    {home_path / 'memories'}")
    print(f"  Skills:    {home_path / 'skills'}")
    print(f"  Sessions:  {home_path / 'sessions.json'}")

    routes = _routes_for_agent(cfg, agent_id)
    print(f"\n  Routes ({len(routes)}):")
    for r in routes:
        match = r.get("match", {})
        parts = []
        for k in ("platform", "chat_type", "chat_id", "thread_id", "topic_id",
                  "user_id", "user_id_alt", "guild_id", "parent_chat_id"):
            v = match.get(k)
            if v:
                parts.append(f"{k}={v}")
        print(f"    → {' '.join(parts) or '(any)'}")

    soul_path = home_path / "SOUL.md"
    print(f"\n  SOUL.md Preview:")
    print(f"    {_summarize_soul(soul_path)}")
    return 0


def cmd_agent_add(args) -> int:
    """Add a new agent to config.yaml."""
    cfg = _ensure_agent_section(_load_config())
    agent_id = args.agent_id

    if not agent_id or not agent_id.replace("-", "").replace("_", "").isalnum():
        print(_red(f"Invalid agent ID '{agent_id}'. Use alphanumeric, hyphens, underscores only."))
        return 1

    if agent_id in cfg.get("agents", {}):
        print(_red(f"Agent '{agent_id}' already exists."))
        return 1

    spec: Dict[str, Any] = {}

    if args.model:
        spec["model"] = args.model
    if args.provider:
        spec["provider"] = args.provider
    if args.home_dir:
        spec["home_dir"] = args.home_dir
    if args.enabled_toolsets:
        spec["enabled_toolsets"] = args.enabled_toolsets.split(",")

    # If cloning from an existing profile, copy directory
    if args.from_profile:
        src = get_hermes_home() / "profiles" / args.from_profile
        dst = get_hermes_home() / "profiles" / agent_id

        if not src.exists():
            print(_red(f"Source profile '{args.from_profile}' not found at {src}"))
            return 1

        if dst.exists():
            print(_red(f"Destination already exists: {dst}"))
            return 1

        try:
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*.pyc", "__pycache__"))
            spec["home_dir"] = str(dst)
            print(_green(f"Cloned profile from '{args.from_profile}' to {dst}"))
        except Exception as e:
            print(_red(f"Failed to clone profile: {e}"))
            return 1

    cfg["agents"][agent_id] = spec
    save_config(cfg)
    print(_green(f"Agent '{agent_id}' added."))
    print(f"  Run 'hermes agent show {agent_id}' for details.")
    return 0


def cmd_agent_remove(args) -> int:
    """Remove an agent from config.yaml."""
    cfg = _ensure_agent_section(_load_config())
    agent_id = args.agent_id

    if agent_id == "main":
        print(_red("Cannot remove the 'main' agent."))
        return 1

    if agent_id not in cfg.get("agents", {}):
        print(_red(f"Agent '{agent_id}' not found."))
        return 1

    routes = _routes_for_agent(cfg, agent_id)
    if routes and not args.yes:
        print(_yellow(f"Warning: {len(routes)} route(s) reference agent '{agent_id}':"))
        for r in routes:
            print(f"  - {r}")
        print("Use --yes to confirm removal.")
        return 1

    # Clean up routes
    cfg["routes"] = [r for r in cfg.get("routes", []) if not (isinstance(r, dict) and r.get("agent") == agent_id)]

    del cfg["agents"][agent_id]
    save_config(cfg)
    print(_green(f"Agent '{agent_id}' removed."))
    if routes:
        print(f"  {len(routes)} orphaned route(s) cleaned up.")
    return 0
