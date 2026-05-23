"""
Hermes Skills MCP Tools — expose agent skills, registry, knowledge layer,
and learnings as MCP tools for Cursor/Claude Code integration.

This module extends the existing Hermes MCP server (mcp_serve.py) with
read-only tools that let MCP clients discover and read:

  - Custom agent SOUL.md files and configurations
  - Agent registry (AGENT_REGISTRY.json)
  - Knowledge layer artifacts (latest_state, held_spec_ledger, etc.)
  - Learnings/memory files (.learnings/)
  - Heartbeat status per agent
  - Skill documents from the repo skills/ directory
  - Cron schedule configuration

Paths resolve via HERMES_AGENTS_DIR (runtime fleet), then HERMES_REPO,
then HERMES_HOME — same profile conventions as mcp_serve.py.

Tool surface (8 tools; read-only by design):
  fleet_context_snapshot — one-call bounded fleet bootstrap for IDEs
  skills_list          — list available agent SOUL.md files and repo skills
  skills_read          — read a specific skill/SOUL.md document
  agents_list          — list agents from registry with status summary
  agents_get           — get full agent config, heartbeat, and SOUL.md
  knowledge_read       — read knowledge layer artifacts
  learnings_read       — read .learnings/ memory files
  artifacts_list       — browse the artifacts/ directory tree
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("hermes.mcp_skills")

_SNAPSHOT_TEXT_CAP = 4_000
_SNAPSHOT_LIST_CAP = 25
_HEARTBEAT_STALE_SECONDS = 24 * 60 * 60
# Bump this version when the documented authority hierarchy changes.
# It is an API contract marker for MCP clients, not derived from docs.
_SOURCE_OF_TRUTH_HIERARCHY = {
    "version": "2026-05-22.cursor-hermes.v1",
    "reference": "website/docs/user-guide/features/cursor-hermes.md#source-of-truth-hierarchy",
    "layers": [
        {"layer": "runtime_wrappers_scripts", "authority": "execution_truth"},
        {"layer": "SOUL_IDENTITY_HEARTBEAT", "authority": "behavioral_truth"},
        {"layer": "AGENT_REGISTRY.json", "authority": "index_discovery_only"},
        {"layer": "knowledge_layer", "authority": "operational_state"},
        {"layer": ".learnings", "authority": "memory_reference"},
        {"layer": "CLAUDE.md_cursor_rules", "authority": "operator_workflow_constraints"},
    ],
}


# ---------------------------------------------------------------------------
# Path resolution (mirrors mcp_serve.py conventions)
# ---------------------------------------------------------------------------

def _get_hermes_home() -> Path:
    """Return HERMES_HOME, defaulting to ~/.hermes."""
    try:
        from hermes_constants import get_hermes_home
        return get_hermes_home()
    except ImportError:
        return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))


def _get_hermes_repo() -> Path:
    """Return HERMES_REPO, defaulting to the directory containing this file."""
    env = os.environ.get("HERMES_REPO")
    if env:
        return Path(env)
    # Fall back to the parent of this file (assumes in repo root)
    this_file = Path(__file__).resolve()
    candidate = this_file.parent
    if (candidate / "hermes").exists() or (candidate / "run_agent.py").exists():
        return candidate
    # Last resort: check HERMES_HOME/hermes-agent
    home = _get_hermes_home()
    if (home / "hermes-agent").exists():
        return home / "hermes-agent"
    return candidate


def _safe_read(path: Path, max_bytes: int = 100_000) -> str:
    """Read a file safely with size cap. Returns content or error string."""
    if not path.exists():
        return f"[File not found: {path}]"
    if not path.is_file():
        return f"[Not a file: {path}]"
    try:
        size = path.stat().st_size
        if size > max_bytes:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(max_bytes)
            return content + f"\n\n[TRUNCATED — file is {size:,} bytes, showing first {max_bytes:,}]"
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"[Error reading {path}: {e}]"


def _file_mtime_iso(path: Path) -> str:
    """Return file modification time as ISO string, or empty string."""
    try:
        mtime = path.stat().st_mtime
        return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    except Exception:
        return ""


def _dir_listing(path: Path, max_depth: int = 2, _depth: int = 0) -> List[dict]:
    """Recursively list a directory up to max_depth."""
    if not path.is_dir() or _depth > max_depth:
        return []
    entries = []
    try:
        for item in sorted(path.iterdir()):
            if item.name.startswith("."):
                continue
            entry: dict = {
                "name": item.name,
                "type": "directory" if item.is_dir() else "file",
                "path": str(item.relative_to(_get_hermes_repo())),
            }
            if item.is_file():
                try:
                    entry["size"] = item.stat().st_size
                    entry["modified"] = _file_mtime_iso(item)
                except OSError:
                    pass
            elif item.is_dir() and _depth < max_depth:
                entry["children"] = _dir_listing(item, max_depth, _depth + 1)
            entries.append(entry)
    except PermissionError:
        pass
    return entries


# ---------------------------------------------------------------------------
# Agent discovery helpers
# ---------------------------------------------------------------------------

def _find_agents_dir() -> Optional[Path]:
    """Find the custom agents/ directory (local clone, not upstream repo).

    Checks HERMES_AGENTS_DIR env var first (explicit override), then
    HERMES_REPO/agents/, then HERMES_HOME/hermes-agent/agents/.
    """
    explicit = os.environ.get("HERMES_AGENTS_DIR")
    if explicit:
        p = Path(explicit)
        if p.is_dir():
            return p

    repo = _get_hermes_repo()
    candidates = [
        repo / "agents",
        _get_hermes_home() / "hermes-agent" / "agents",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return None


def _find_agent_registry() -> Optional[Path]:
    """Find AGENT_REGISTRY.json."""
    agents_dir = _find_agents_dir()
    if agents_dir and (agents_dir / "AGENT_REGISTRY.json").exists():
        return agents_dir / "AGENT_REGISTRY.json"
    # Also check repo root
    repo = _get_hermes_repo()
    if (repo / "AGENT_REGISTRY.json").exists():
        return repo / "AGENT_REGISTRY.json"
    return None


def _load_agent_registry() -> dict:
    """Load and parse agent registry, returning dict or empty."""
    path = _find_agent_registry()
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.debug("Failed to load agent registry: %s", e)
        return {}


def _registry_agents(registry: dict) -> dict:
    """Return the agent mapping from supported registry shapes."""
    if isinstance(registry.get("agents"), dict):
        return registry["agents"]
    return registry


def _count_by_field(entries: dict, field: str) -> dict:
    counts: dict[str, int] = {}
    for meta in entries.values():
        if not isinstance(meta, dict):
            continue
        value = str(meta.get(field) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _bounded_read(path: Optional[Path], *, max_chars: int = _SNAPSHOT_TEXT_CAP) -> dict:
    """Read a file without exceeding the snapshot character budget."""
    if not path or not path.exists() or not path.is_file():
        return {"present": False, "path": str(path) if path else None, "content": ""}
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return {"present": False, "path": str(path), "error": str(exc), "content": ""}
    truncated = len(content) > max_chars
    return {
        "present": True,
        "path": str(path),
        "modified": _file_mtime_iso(path),
        "truncated": truncated,
        "content": content[:max_chars],
    }


def _find_latest_state_path() -> Optional[Path]:
    artifacts_dir = _get_artifacts_dir()
    if not artifacts_dir:
        return None
    for suffix in ("md", "json"):
        path = artifacts_dir / "ops" / "knowledge_layer" / f"latest_state.{suffix}"
        if path.exists():
            return path
    return None


def _find_held_spec_ledger_path() -> Optional[Path]:
    artifacts_dir = _get_artifacts_dir()
    if not artifacts_dir:
        return None
    for suffix in ("md", "json"):
        path = artifacts_dir / "ops" / "held_spec_ledger" / f"latest.{suffix}"
        if path.exists():
            return path
    return None


def _extract_held_spec_flags(content: str) -> list[str]:
    """Return a bounded list of held-spec lines that look operationally active."""
    flags: list[str] = []
    markers = ("held", "hold", "blocked", "frozen", "active", "must", "cannot")
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        if any(marker in lowered for marker in markers):
            flags.append(line[:240])
        if len(flags) >= _SNAPSHOT_LIST_CAP:
            break
    return flags


def _gateway_reachable() -> bool:
    """Best-effort read-only probe for live gateway availability."""
    try:
        from gateway.status import is_gateway_running

        return bool(is_gateway_running(cleanup_stale=False))
    except Exception as exc:
        logger.debug("Gateway reachability probe failed: %s", exc)
        return False


def _source_of_truth_label(agents_dir: Optional[Path]) -> Optional[str]:
    """Return the authority layer that supplied the agent documents."""
    if agents_dir is None:
        return None
    explicit = os.environ.get("HERMES_AGENTS_DIR")
    if explicit and agents_dir == Path(explicit):
        return "HERMES_AGENTS_DIR"
    repo_agents = _get_hermes_repo() / "agents"
    if agents_dir == repo_agents:
        return "HERMES_REPO/agents"
    home_agents = _get_hermes_home() / "hermes-agent" / "agents"
    if agents_dir == home_agents:
        return "HERMES_HOME/hermes-agent/agents"
    return str(agents_dir)


def build_fleet_context_snapshot() -> dict:
    """Build a bounded, read-only fleet bootstrap snapshot for MCP clients."""
    hermes_home = _get_hermes_home()
    hermes_repo = _get_hermes_repo()
    agents_dir = _find_agents_dir()
    registry_path = _find_agent_registry()
    registry = _load_agent_registry()
    registry_entries = _registry_agents(registry)
    warnings: list[str] = []
    missing_layers: list[str] = []

    explicit_agents_dir = os.environ.get("HERMES_AGENTS_DIR")
    if explicit_agents_dir and not Path(explicit_agents_dir).is_dir():
        warnings.append(f"HERMES_AGENTS_DIR does not exist: {explicit_agents_dir}")

    if not agents_dir:
        missing_layers.append("agents_dir")
        warnings.append("No agents directory found; checked HERMES_AGENTS_DIR, HERMES_REPO/agents, and HERMES_HOME/hermes-agent/agents.")
    if not registry_path:
        missing_layers.append("registry")
        warnings.append("AGENT_REGISTRY.json not found.")

    registry_summary = {
        "path": str(registry_path) if registry_path else None,
        "agents": sorted(registry_entries.keys())[:_SNAPSHOT_LIST_CAP],
        "agents_truncated": len(registry_entries) > _SNAPSHOT_LIST_CAP,
        "by_status": _count_by_field(registry_entries, "status"),
        "by_lane": _count_by_field(registry_entries, "lane"),
        "by_authority": _count_by_field(registry_entries, "authority"),
    }

    stale_heartbeats: list[dict] = []
    now = datetime.now(timezone.utc).timestamp()
    if agents_dir and registry_entries:
        for agent_name in sorted(registry_entries.keys()):
            heartbeat = agents_dir / agent_name / "HEARTBEAT.md"
            if not heartbeat.exists():
                stale_heartbeats.append({"agent": agent_name, "status": "missing"})
            else:
                try:
                    age_seconds = max(0.0, now - heartbeat.stat().st_mtime)
                    if age_seconds > _HEARTBEAT_STALE_SECONDS:
                        stale_heartbeats.append({
                            "agent": agent_name,
                            "status": "stale",
                            "age_hours": round(age_seconds / 3600, 1),
                            "path": str(heartbeat),
                            "modified": _file_mtime_iso(heartbeat),
                        })
                except OSError as exc:
                    stale_heartbeats.append({"agent": agent_name, "status": "unreadable", "error": str(exc)})

    stale_truncated = len(stale_heartbeats) > _SNAPSHOT_LIST_CAP
    if stale_truncated:
        warnings.append(f"stale_heartbeats truncated at {_SNAPSHOT_LIST_CAP} entries.")
    stale_heartbeats = stale_heartbeats[:_SNAPSHOT_LIST_CAP]

    learnings_dir = _get_learnings_dir()
    hot_path = learnings_dir / "memory.md" if learnings_dir else None
    hot = _bounded_read(hot_path)
    if not hot["present"]:
        missing_layers.append("hot_learnings")

    latest_state = _bounded_read(_find_latest_state_path())
    if not latest_state["present"]:
        missing_layers.append("latest_state")

    held_ledger = _bounded_read(_find_held_spec_ledger_path())
    held_spec_flags = _extract_held_spec_flags(str(held_ledger.get("content") or "")) if held_ledger["present"] else []
    if not held_ledger["present"]:
        missing_layers.append("held_spec_ledger")

    gateway = _gateway_reachable()
    source_of_truth = _source_of_truth_label(agents_dir)
    return {
        "mode": "live_ops" if gateway else "skills_only",
        "writes_allowed": False,
        "source_of_truth": source_of_truth,
        "authority_boundary": {
            "mode": "live_ops" if gateway else "skills_only",
            "writes_allowed": False,
            "source_of_truth": source_of_truth,
            "gateway_reachable": gateway,
        },
        "hermes_home": str(hermes_home),
        "hermes_repo": str(hermes_repo),
        "agents_dir": str(agents_dir) if agents_dir else None,
        "registry_present": bool(registry_path),
        "agent_count": len(registry_entries),
        "registry_summary": registry_summary,
        "stale_heartbeats": stale_heartbeats,
        "hot_learnings_excerpt": hot,
        "latest_state_digest": latest_state,
        "held_spec_flags": held_spec_flags,
        "gateway_reachable": gateway,
        "missing_layers": sorted(set(missing_layers)),
        "warnings": warnings,
        "source_of_truth_hierarchy": _SOURCE_OF_TRUTH_HIERARCHY,
    }


def _find_heartbeat(agent_name: str) -> Optional[dict]:
    """Find and parse an agent's HEARTBEAT.md for status info."""
    agents_dir = _find_agents_dir()
    if not agents_dir:
        return None
    heartbeat = agents_dir / agent_name / "HEARTBEAT.md"
    if not heartbeat.exists():
        return None
    try:
        content = heartbeat.read_text(encoding="utf-8", errors="replace")
        return {
            "path": str(heartbeat),
            "modified": _file_mtime_iso(heartbeat),
            "content": content[:5000],
        }
    except Exception:
        return None


def _find_soul_md(agent_name: str) -> Optional[Path]:
    """Find an agent's SOUL.md file."""
    agents_dir = _find_agents_dir()
    if not agents_dir:
        return None
    soul = agents_dir / agent_name / "SOUL.md"
    if soul.exists():
        return soul
    return None


# ---------------------------------------------------------------------------
# Knowledge layer & artifacts helpers
# ---------------------------------------------------------------------------

def _get_artifacts_dir() -> Optional[Path]:
    """Find the artifacts/ directory."""
    repo = _get_hermes_repo()
    candidates = [
        repo / "artifacts",
        _get_hermes_home() / "hermes-agent" / "artifacts",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return None


def _get_knowledge_layer_dir() -> Optional[Path]:
    """Find artifacts/ops/knowledge_layer/."""
    artifacts = _get_artifacts_dir()
    if artifacts:
        kl = artifacts / "ops" / "knowledge_layer"
        if kl.is_dir():
            return kl
    return None


def _get_learnings_dir() -> Optional[Path]:
    """Find .learnings/ directory."""
    repo = _get_hermes_repo()
    candidates = [
        repo / ".learnings",
        _get_hermes_home() / ".learnings",
        _get_hermes_home() / "hermes-agent" / ".learnings",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return None


# ---------------------------------------------------------------------------
# Repo skills helpers
# ---------------------------------------------------------------------------

def _get_repo_skills_dir() -> Optional[Path]:
    """Find the repo skills/ directory."""
    repo = _get_hermes_repo()
    skills = repo / "skills"
    if skills.is_dir():
        return skills
    return None


def _list_repo_skills() -> List[dict]:
    """List skill categories and their contents from the repo skills/ dir."""
    skills_dir = _get_repo_skills_dir()
    if not skills_dir:
        return []
    result = []
    try:
        for category in sorted(skills_dir.iterdir()):
            if not category.is_dir() or category.name.startswith("."):
                continue
            entry: dict = {"category": category.name, "skills": []}
            for skill in sorted(category.iterdir()):
                if skill.is_dir():
                    # Check for SKILL.md or similar entry file
                    skill_file = None
                    for candidate_name in ["SKILL.md", "skill.md", "README.md"]:
                        if (skill / candidate_name).exists():
                            skill_file = candidate_name
                            break
                    entry["skills"].append({
                        "name": skill.name,
                        "entry_file": skill_file,
                        "path": str(skill.relative_to(_get_hermes_repo())),
                    })
                elif skill.is_file() and skill.suffix in (".md", ".txt", ".yaml", ".yml"):
                    entry["skills"].append({
                        "name": skill.stem,
                        "type": "file",
                        "path": str(skill.relative_to(_get_hermes_repo())),
                    })
            if entry["skills"]:
                result.append(entry)
    except PermissionError:
        pass
    return result


# ---------------------------------------------------------------------------
# Cron helpers
# ---------------------------------------------------------------------------

def _get_cron_config() -> Optional[Path]:
    """Find cron configuration."""
    repo = _get_hermes_repo()
    candidates = [
        repo / "cron",
        repo / "cron.d",
    ]
    for c in candidates:
        if c.is_dir():
            return c
        if c.with_suffix(".yaml").exists():
            return c.with_suffix(".yaml")
        if c.with_suffix(".json").exists():
            return c.with_suffix(".json")
    return None


# ---------------------------------------------------------------------------
# Register MCP tools onto an existing FastMCP server
# ---------------------------------------------------------------------------

def register_skills_tools(mcp) -> None:
    """Register skills/knowledge layer tools onto an existing MCP server.

    Call this from mcp_serve.py after creating the FastMCP instance:

        from hermes_skills_mcp import register_skills_tools
        register_skills_tools(mcp)
    """

    # -- fleet_context_snapshot -------------------------------------------

    @mcp.tool()
    def fleet_context_snapshot() -> str:
        """Return a bounded, read-only fleet bootstrap snapshot for IDE clients.

        The snapshot works in skills/context mode even when the live gateway is
        down. It never writes files and reports missing layers explicitly so
        Cursor can continue with partial but trustworthy context.
        """
        return json.dumps(build_fleet_context_snapshot(), indent=2)

    # -- skills_list -------------------------------------------------------

    @mcp.tool()
    def skills_list(
        source: str = "all",
    ) -> str:
        """List available Hermes skills and agent SOUL.md documents.

        Returns custom agent skills (SOUL.md files from agents/) and
        repo skills (from skills/) with paths for reading.

        Args:
            source: Filter by source — "agents" for custom SOUL.md files,
                    "repo" for repo skills, "all" for both (default "all")
        """
        result: dict = {"agents_skills": [], "repo_skills": [], "paths": {}}

        # Custom agent SOUL.md files
        if source in ("all", "agents"):
            agents_dir = _find_agents_dir()
            if agents_dir:
                result["paths"]["agents_dir"] = str(agents_dir)
                for agent_dir in sorted(agents_dir.iterdir()):
                    if not agent_dir.is_dir() or agent_dir.name.startswith("."):
                        continue
                    soul = agent_dir / "SOUL.md"
                    entry: dict = {
                        "agent": agent_dir.name,
                        "has_soul_md": soul.exists(),
                        "path": str(agent_dir.relative_to(_get_hermes_repo())),
                    }
                    if soul.exists():
                        entry["soul_md_modified"] = _file_mtime_iso(soul)
                        entry["soul_md_size"] = soul.stat().st_size
                    # Check for other key files
                    for extra in ["HEARTBEAT.md", "config.yaml", "config.json"]:
                        if (agent_dir / extra).exists():
                            entry.setdefault("extra_files", []).append(extra)
                    result["agents_skills"].append(entry)

        # Repo skills
        if source in ("all", "repo"):
            skills_dir = _get_repo_skills_dir()
            if skills_dir:
                result["paths"]["skills_dir"] = str(skills_dir)
            result["repo_skills"] = _list_repo_skills()

        result["counts"] = {
            "agent_skills": len(result["agents_skills"]),
            "repo_skill_categories": len(result["repo_skills"]),
        }

        return json.dumps(result, indent=2)

    # -- skills_read -------------------------------------------------------

    @mcp.tool()
    def skills_read(
        name: str,
        file: str = "SOUL.md",
    ) -> str:
        """Read a specific skill or agent document.

        For custom agents, reads SOUL.md (default) or other files from
        agents/<name>/. For repo skills, reads from skills/<category>/<name>/.

        Args:
            name: Agent name (e.g. "herald", "bellringer") or repo skill
                  path (e.g. "research/web-search")
            file: Filename to read within the agent/skill directory
                  (default "SOUL.md")
        """
        # Try custom agent first
        agents_dir = _find_agents_dir()
        if agents_dir:
            agent_path = agents_dir / name / file
            if agent_path.exists():
                content = _safe_read(agent_path)
                return json.dumps({
                    "source": "agent",
                    "agent": name,
                    "file": file,
                    "path": str(agent_path),
                    "modified": _file_mtime_iso(agent_path),
                    "content": content,
                }, indent=2)

        # Try repo skills (name can be "category/skill" or just "skill")
        skills_dir = _get_repo_skills_dir()
        if skills_dir:
            # Direct path
            skill_path = skills_dir / name / file
            if skill_path.exists():
                content = _safe_read(skill_path)
                return json.dumps({
                    "source": "repo_skill",
                    "skill": name,
                    "file": file,
                    "path": str(skill_path),
                    "modified": _file_mtime_iso(skill_path),
                    "content": content,
                }, indent=2)
            # Search across categories
            for category in skills_dir.iterdir():
                if not category.is_dir():
                    continue
                candidate = category / name / file
                if candidate.exists():
                    content = _safe_read(candidate)
                    return json.dumps({
                        "source": "repo_skill",
                        "skill": f"{category.name}/{name}",
                        "file": file,
                        "path": str(candidate),
                        "modified": _file_mtime_iso(candidate),
                        "content": content,
                    }, indent=2)

        return json.dumps({
            "error": f"Skill not found: {name}/{file}",
            "searched": {
                "agents_dir": str(agents_dir) if agents_dir else None,
                "skills_dir": str(skills_dir) if skills_dir else None,
            },
        }, indent=2)

    # -- agents_list -------------------------------------------------------

    @mcp.tool()
    def agents_list(
        include_heartbeat: bool = False,
    ) -> str:
        """List all Hermes agents with status summary.

        Returns agents from AGENT_REGISTRY.json if available, otherwise
        scans the agents/ directory. Optionally includes heartbeat data.

        Args:
            include_heartbeat: Include HEARTBEAT.md content for each agent
                               (default false — set true for health check)
        """
        registry = _load_agent_registry()
        agents_dir = _find_agents_dir()

        agents = []

        if registry:
            # Use registry as authoritative source
            for agent_key, agent_data in registry.items():
                entry: dict = {
                    "name": agent_key,
                    **{k: v for k, v in agent_data.items()
                       if k in ("lane", "tier", "authority", "status",
                                "description", "cron", "dependencies",
                                "suppressed", "retired")},
                }
                if include_heartbeat:
                    hb = _find_heartbeat(agent_key)
                    if hb:
                        entry["heartbeat"] = hb
                agents.append(entry)
        elif agents_dir:
            # Fall back to directory scan
            for agent_dir in sorted(agents_dir.iterdir()):
                if not agent_dir.is_dir() or agent_dir.name.startswith("."):
                    continue
                entry = {
                    "name": agent_dir.name,
                    "has_soul_md": (agent_dir / "SOUL.md").exists(),
                    "has_heartbeat": (agent_dir / "HEARTBEAT.md").exists(),
                }
                if include_heartbeat:
                    hb = _find_heartbeat(agent_dir.name)
                    if hb:
                        entry["heartbeat"] = hb
                agents.append(entry)

        return json.dumps({
            "count": len(agents),
            "registry_path": str(_find_agent_registry()) if _find_agent_registry() else None,
            "agents_dir": str(agents_dir) if agents_dir else None,
            "agents": agents,
        }, indent=2)

    # -- agents_get --------------------------------------------------------

    @mcp.tool()
    def agents_get(
        name: str,
    ) -> str:
        """Get full details for a specific agent.

        Returns registry entry, SOUL.md content, heartbeat status,
        config, and directory listing for the agent.

        Args:
            name: Agent name (e.g. "herald", "ops_supervisor", "bellringer")
        """
        result: dict = {"name": name}

        # Registry entry
        registry = _load_agent_registry()
        if name in registry:
            result["registry"] = registry[name]

        # SOUL.md
        soul = _find_soul_md(name)
        if soul:
            result["soul_md"] = {
                "path": str(soul),
                "modified": _file_mtime_iso(soul),
                "content": _safe_read(soul),
            }

        # Heartbeat
        hb = _find_heartbeat(name)
        if hb:
            result["heartbeat"] = hb

        # Agent directory listing
        agents_dir = _find_agents_dir()
        if agents_dir:
            agent_dir = agents_dir / name
            if agent_dir.is_dir():
                result["files"] = []
                for item in sorted(agent_dir.iterdir()):
                    if item.name.startswith("."):
                        continue
                    result["files"].append({
                        "name": item.name,
                        "type": "directory" if item.is_dir() else "file",
                        "size": item.stat().st_size if item.is_file() else None,
                        "modified": _file_mtime_iso(item),
                    })

        if len(result) == 1:
            result["error"] = f"Agent not found: {name}"

        return json.dumps(result, indent=2)

    # -- knowledge_read ----------------------------------------------------

    @mcp.tool()
    def knowledge_read(
        artifact: str = "latest_state",
        format: str = "md",
    ) -> str:
        """Read a knowledge layer artifact.

        Available artifacts:
          - latest_state: Current knowledge layer state
          - held_spec_ledger: Held specification ledger
          - first_fire_ledger: First-fire validation ledger
          - contradiction_ledger: Contradiction/discrepancy ledger
          - operator_brief: Latest daily operator brief

        Args:
            artifact: Artifact name (default "latest_state")
            format: File format — "md" or "json" (default "md")
        """
        artifacts_dir = _get_artifacts_dir()
        if not artifacts_dir:
            return json.dumps({
                "error": "Artifacts directory not found",
                "searched": [
                    str(_get_hermes_repo() / "artifacts"),
                    str(_get_hermes_home() / "hermes-agent" / "artifacts"),
                ],
            }, indent=2)

        # Map artifact names to paths
        artifact_paths: dict = {
            "latest_state": artifacts_dir / "ops" / "knowledge_layer" / f"latest_state.{format}",
            "held_spec_ledger": artifacts_dir / "ops" / "held_spec_ledger" / f"latest.{format}",
            "first_fire_ledger": artifacts_dir / "ops" / "first_fire_ledger" / f"latest.{format}",
            "contradiction_ledger": artifacts_dir / "ops" / "contradiction_ledger" / "latest.md",
        }

        # Operator brief is date-keyed
        if artifact == "operator_brief":
            brief_dir = artifacts_dir / "ops" / "operator_brief" / "daily"
            if brief_dir.is_dir():
                # Find most recent brief
                briefs = sorted(brief_dir.glob("*.md"), reverse=True)
                if briefs:
                    artifact_paths["operator_brief"] = briefs[0]

        path = artifact_paths.get(artifact)
        if not path:
            return json.dumps({
                "error": f"Unknown artifact: {artifact}",
                "available": list(artifact_paths.keys()),
            }, indent=2)

        if not path.exists():
            # Try alternate format
            alt_format = "json" if format == "md" else "md"
            alt_path = path.with_suffix(f".{alt_format}")
            if alt_path.exists():
                path = alt_path
            else:
                return json.dumps({
                    "error": f"Artifact not found: {path}",
                    "tried": [str(path), str(alt_path)],
                }, indent=2)

        content = _safe_read(path)
        return json.dumps({
            "artifact": artifact,
            "path": str(path),
            "modified": _file_mtime_iso(path),
            "format": path.suffix.lstrip("."),
            "content": content,
        }, indent=2)

    # -- learnings_read ----------------------------------------------------

    @mcp.tool()
    def learnings_read(
        file: str = "memory.md",
    ) -> str:
        """Read Hermes learnings/memory files.

        The .learnings/ directory contains the agent's persistent memory:
          - memory.md: HOT tier memory (loaded every session, 100-line cap)
          - projects/: Per-project/namespace memory files
          - domains/: Per-domain knowledge files

        Args:
            file: File path within .learnings/ (default "memory.md")
        """
        learnings_dir = _get_learnings_dir()
        if not learnings_dir:
            return json.dumps({
                "error": ".learnings/ directory not found",
                "searched": [
                    str(_get_hermes_repo() / ".learnings"),
                    str(_get_hermes_home() / ".learnings"),
                    str(_get_hermes_home() / "hermes-agent" / ".learnings"),
                ],
            }, indent=2)

        target = learnings_dir / file

        # If requesting a directory, list it
        if target.is_dir():
            entries = []
            for item in sorted(target.iterdir()):
                entries.append({
                    "name": item.name,
                    "type": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None,
                    "modified": _file_mtime_iso(item),
                })
            return json.dumps({
                "path": str(target),
                "type": "directory",
                "entries": entries,
            }, indent=2)

        if not target.exists():
            # List what IS available
            available = []
            for item in learnings_dir.rglob("*"):
                if item.is_file():
                    available.append(str(item.relative_to(learnings_dir)))
            return json.dumps({
                "error": f"File not found: {file}",
                "available": available[:50],
                "learnings_dir": str(learnings_dir),
            }, indent=2)

        content = _safe_read(target)
        return json.dumps({
            "file": file,
            "path": str(target),
            "modified": _file_mtime_iso(target),
            "content": content,
        }, indent=2)

    # -- artifacts_list ----------------------------------------------------

    @mcp.tool()
    def artifacts_list(
        path: str = "",
        depth: int = 2,
    ) -> str:
        """Browse the artifacts/ directory tree.

        Lists files and subdirectories under artifacts/, useful for
        discovering what knowledge layer outputs and operational data exist.

        Args:
            path: Subdirectory within artifacts/ to list (default "" = root)
            depth: Directory traversal depth (default 2, max 4)
        """
        artifacts_dir = _get_artifacts_dir()
        if not artifacts_dir:
            return json.dumps({
                "error": "Artifacts directory not found",
            }, indent=2)

        target = artifacts_dir / path if path else artifacts_dir
        if not target.is_dir():
            return json.dumps({
                "error": f"Directory not found: {target}",
            }, indent=2)

        depth = max(1, min(depth, 4))
        entries = _dir_listing(target, max_depth=depth)

        return json.dumps({
            "path": str(target.relative_to(_get_hermes_repo())),
            "depth": depth,
            "entries": entries,
        }, indent=2)

    logger.debug("Registered 7 skills/knowledge MCP tools")
