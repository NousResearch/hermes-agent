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

Tool surface (13 tools; read-only by design):
  fleet_context_snapshot — one-call bounded fleet bootstrap for IDEs
  agent_health_summary — compact actionable fleet health anomalies
  self_improvement_snapshot — HOT/WARM memory health + knowledge reconciliation
  town_brief — human-facing Cursor/Town integration brief
  town_handoff_bundle — bounded agent/spec handoff context bundle
  knowledge_query — bounded keyword query over knowledge graph artifacts
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


def _display_path(path: Path) -> str:
    """Return a repo-relative path when possible, otherwise an absolute path."""
    try:
        return str(path.relative_to(_get_hermes_repo()))
    except ValueError:
        return str(path)


def _safe_child_path(root: Path, *parts: str) -> Optional[Path]:
    """Resolve a user-supplied child path without escaping ``root``."""
    try:
        for part in parts:
            if Path(str(part)).is_absolute():
                return None
        base = root.resolve()
        candidate = base.joinpath(*(str(part) for part in parts)).resolve()
        if not candidate.is_relative_to(base):
            return None
        return candidate
    except (OSError, RuntimeError, ValueError):
        return None


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
                "path": _display_path(item),
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


def _find_contradiction_ledger_path() -> Optional[Path]:
    artifacts_dir = _get_artifacts_dir()
    if not artifacts_dir:
        return None
    path = artifacts_dir / "ops" / "contradiction_ledger" / "latest.md"
    if path.exists():
        return path
    return None


def _matching_lines(content: str, terms: list[str], *, limit: int = 12) -> list[str]:
    """Return bounded lines matching any non-empty search term."""
    needles = [term.lower() for term in terms if term]
    if not needles:
        return []
    matches: list[str] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        if any(needle in lowered for needle in needles):
            matches.append(line[:240])
        if len(matches) >= limit:
            break
    return matches


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


def build_fleet_context_snapshot(*, summary: bool = False) -> dict:
    """Build a bounded, read-only fleet bootstrap snapshot for MCP clients.

    Args:
        summary: If True, return a compressed payload that omits text blobs
                 (hot_learnings content, latest_state content, held_spec raw content)
                 and returns only structured metadata, counts, and flags.
                 Reduces token consumption by ~60-80% for IDE sessions that only
                 need state awareness without full text context.
    """
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
    if summary:
        # Summary mode: check presence only, skip content reads
        hot_present = bool(hot_path and hot_path.exists())
        if not hot_present:
            missing_layers.append("hot_learnings")
        latest_state_path = _find_latest_state_path()
        latest_state_present = bool(latest_state_path and latest_state_path.exists())
        if not latest_state_present:
            missing_layers.append("latest_state")
        held_ledger_path = _find_held_spec_ledger_path()
        held_ledger_present = bool(held_ledger_path and held_ledger_path.exists())
        if not held_ledger_present:
            missing_layers.append("held_spec_ledger")
        # Extract flag count from held ledger without full content read
        held_spec_flags: list[str] = []
        if held_ledger_present and held_ledger_path:
            try:
                content = held_ledger_path.read_text(encoding="utf-8", errors="replace")
                held_spec_flags = _extract_held_spec_flags(content)
            except Exception:
                pass
    else:
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

    if summary:
        # Compressed output: structured metadata only, no text blobs
        return {
            "format": "summary",
            "mode": "live_ops" if gateway else "skills_only",
            "writes_allowed": False,
            "source_of_truth": source_of_truth,
            "gateway_reachable": gateway,
            "hermes_home": str(hermes_home),
            "hermes_repo": str(hermes_repo),
            "agents_dir": str(agents_dir) if agents_dir else None,
            "registry_present": bool(registry_path),
            "agent_count": len(registry_entries),
            "registry_summary": registry_summary,
            "stale_heartbeats": stale_heartbeats,
            "hot_learnings_present": hot_present,
            "hot_learnings_path": str(hot_path) if hot_path else None,
            "latest_state_present": latest_state_present,
            "latest_state_path": str(latest_state_path) if latest_state_path else None,
            "held_spec_flags_count": len(held_spec_flags),
            "held_spec_flags": held_spec_flags[:10],
            "missing_layers": sorted(set(missing_layers)),
            "warnings": warnings,
        }

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


def build_agent_health_summary() -> dict:
    """Build a compact, actionable, read-only fleet health summary."""
    snapshot = build_fleet_context_snapshot()
    stale_heartbeats = list(snapshot.get("stale_heartbeats") or [])
    missing_layers = list(snapshot.get("missing_layers") or [])
    warnings = list(snapshot.get("warnings") or [])
    held_flags = list(snapshot.get("held_spec_flags") or [])

    anomaly_count = (
        len(stale_heartbeats)
        + len(missing_layers)
        + len(warnings)
        + len(held_flags)
        + (0 if snapshot.get("registry_present") else 1)
    )

    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "status": "attention" if anomaly_count else "ok",
        "mode": snapshot.get("mode"),
        "gateway_reachable": bool(snapshot.get("gateway_reachable")),
        "writes_allowed": False,
        "source_of_truth": snapshot.get("source_of_truth"),
        "agents_dir": snapshot.get("agents_dir"),
        "registry_present": bool(snapshot.get("registry_present")),
        "agent_count": int(snapshot.get("agent_count") or 0),
        "stale_heartbeats": stale_heartbeats[:_SNAPSHOT_LIST_CAP],
        "stale_heartbeats_truncated": len(stale_heartbeats) > _SNAPSHOT_LIST_CAP,
        "missing_layers": missing_layers,
        "held_spec_flags_count": len(held_flags),
        "held_spec_flags_sample": held_flags[:10],
        "warnings": warnings[:_SNAPSHOT_LIST_CAP],
        "warning_count": len(warnings),
        "next_action": (
            "Review stale/missing layers before changing governed code."
            if anomaly_count else
            "No actionable fleet health anomalies in local MCP snapshot."
        ),
    }


def build_self_improvement_snapshot(*, summary: bool = False) -> dict:
    """Read-only audit for recursive self-improvement across knowledge + memory.

    Surfaces HOT/WARM tier health, knowledge↔HOT reconciliation hints, and a
    session loop checklist. Does not write .learnings/ or artifacts.
    """
    from tools.self_improvement_audit import build_self_improvement_audit

    learnings_dir = _get_learnings_dir()
    if not learnings_dir:
        learnings_dir = _get_hermes_repo() / ".learnings"

    audit = build_self_improvement_audit(
        learnings_dir=learnings_dir,
        latest_state_path=_find_latest_state_path(),
        held_spec_path=_find_held_spec_ledger_path(),
        contradiction_path=_find_contradiction_ledger_path(),
    )
    proposals = list(audit.get("proposals") or [])
    status = "attention" if proposals else "ok"

    if summary:
        return {
            "format": "summary",
            "as_of": audit.get("as_of"),
            "status": status,
            "writes_allowed": False,
            "hot_tier": audit.get("hot_tier"),
            "warm_file_count": len(audit.get("warm_tiers") or []),
            "corrections_log_present": bool(
                (audit.get("corrections_log") or {}).get("present")
            ),
            "knowledge_present": audit.get("knowledge_present"),
            "proposal_count": len(proposals),
            "proposals": proposals[:_SNAPSHOT_LIST_CAP],
            "recursive_loop": audit.get("recursive_loop"),
            "governance_note": audit.get("governance_note"),
            "recommended_mcp_calls": audit.get("recommended_mcp_calls"),
        }

    audit["status"] = status
    return audit


def build_town_brief() -> dict:
    """Build a concise, read-only Town/Cursor operational brief."""
    snapshot = build_fleet_context_snapshot(summary=True)
    health = build_agent_health_summary()

    active_issues: list[dict] = []
    if not snapshot.get("registry_present"):
        active_issues.append({
            "kind": "missing_registry",
            "severity": "warning",
            "detail": "AGENT_REGISTRY.json was not found.",
        })
    for layer in snapshot.get("missing_layers") or []:
        active_issues.append({
            "kind": "missing_layer",
            "severity": "warning",
            "detail": layer,
        })
    for heartbeat in snapshot.get("stale_heartbeats") or []:
        active_issues.append({
            "kind": "heartbeat",
            "severity": "attention",
            "detail": heartbeat,
        })
    for warning in snapshot.get("warnings") or []:
        active_issues.append({
            "kind": "warning",
            "severity": "warning",
            "detail": warning,
        })

    held_flags = snapshot.get("held_spec_flags") or []
    if held_flags:
        active_issues.append({
            "kind": "held_specs",
            "severity": "governance",
            "detail": {
                "count": snapshot.get("held_spec_flags_count", len(held_flags)),
                "sample": held_flags[:10],
            },
        })

    status = "ok"
    if active_issues:
        status = "attention"
    if any(issue["kind"] in {"missing_registry", "missing_layer"} for issue in active_issues):
        status = "degraded"

    next_actions = [
        "Use town_brief or fleet_context_snapshot(summary=True) at Cursor session start.",
        "Use self_improvement_snapshot(summary=True) before editing memory.md or promoting learnings.",
        "Use learnings_read(file='memory.md') for read-only HOT memory/reference context.",
        "Use agents_get(name) and skills_read(name) before modifying a named agent.",
        "Use knowledge_read('held_spec_ledger') before changing governed specs or pipelines.",
    ]
    if not snapshot.get("gateway_reachable"):
        next_actions.append(
            "Gateway is unavailable; skills/context MCP tools can still be used, "
            "but live messaging tools require a running gateway."
        )
    if active_issues:
        next_actions.insert(0, health.get("next_action") or "Review active Town issues.")

    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "mode": snapshot.get("mode"),
        "writes_allowed": False,
        "source_of_truth": snapshot.get("source_of_truth"),
        "paths": {
            "hermes_home": snapshot.get("hermes_home"),
            "hermes_repo": snapshot.get("hermes_repo"),
            "agents_dir": snapshot.get("agents_dir"),
            "latest_state": snapshot.get("latest_state_path"),
            "hot_learnings": snapshot.get("hot_learnings_path"),
        },
        "counts": {
            "agents": snapshot.get("agent_count", 0),
            "stale_heartbeats": len(snapshot.get("stale_heartbeats") or []),
            "held_spec_flags": snapshot.get("held_spec_flags_count", 0),
            "missing_layers": len(snapshot.get("missing_layers") or []),
            "memory_files": 1 if snapshot.get("hot_learnings_present") else 0,
        },
        "registry_summary": snapshot.get("registry_summary", {}),
        "memory": {
            "read_only": True,
            "hot_memory_present": bool(snapshot.get("hot_learnings_present")),
            "hot_memory_path": snapshot.get("hot_learnings_path"),
            "tool": "learnings_read(file='memory.md')",
            "note": "Skills/context mode includes read-only .learnings memory/reference reads.",
        },
        "gateway": {
            "reachable": bool(snapshot.get("gateway_reachable")),
            "skills_context_available": True,
            "live_ops_requires_gateway": True,
        },
        "active_issues": active_issues[:_SNAPSHOT_LIST_CAP],
        "active_issues_truncated": len(active_issues) > _SNAPSHOT_LIST_CAP,
        "recommended_cursor_calls": [
            "town_brief()",
            "agent_health_summary()",
            "fleet_context_snapshot(summary=True)",
            "learnings_read(file='memory.md')",
            "knowledge_read(artifact='held_spec_ledger')",
        ],
        "next_actions": next_actions,
    }


def build_town_handoff_bundle(
    *,
    agent_name: str = "",
    spec_id: str = "",
    include_learnings: bool = True,
) -> dict:
    """Build a bounded read-only context bundle for Cursor agent handoffs."""
    agent_name = str(agent_name or "").strip()
    spec_id = str(spec_id or "").strip()
    search_terms = [term for term in (agent_name, spec_id) if term]
    registry = _registry_agents(_load_agent_registry())
    agents_dir = _find_agents_dir()
    latest_state = _bounded_read(_find_latest_state_path())
    held_ledger = _bounded_read(_find_held_spec_ledger_path())
    contradiction_ledger = _bounded_read(_find_contradiction_ledger_path())
    learnings_dir = _get_learnings_dir()
    hot_learnings = (
        _bounded_read(learnings_dir / "memory.md", max_chars=2_000)
        if include_learnings and learnings_dir else
        {"present": False, "path": str(learnings_dir / "memory.md") if learnings_dir else None, "content": ""}
    )

    bundle: dict = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "writes_allowed": False,
        "inputs": {
            "agent_name": agent_name or None,
            "spec_id": spec_id or None,
            "include_learnings": include_learnings,
        },
        "source_of_truth": _source_of_truth_label(agents_dir),
        "paths": {
            "agents_dir": str(agents_dir) if agents_dir else None,
            "registry": str(_find_agent_registry()) if _find_agent_registry() else None,
            "latest_state": latest_state.get("path"),
            "held_spec_ledger": held_ledger.get("path"),
            "contradiction_ledger": contradiction_ledger.get("path"),
            "hot_learnings": hot_learnings.get("path"),
        },
        "town": build_town_brief(),
        "knowledge": {
            "latest_state": latest_state,
            "held_spec_ledger": held_ledger,
            "contradiction_ledger": contradiction_ledger,
            "matches": {
                "latest_state": _matching_lines(str(latest_state.get("content") or ""), search_terms),
                "held_spec_ledger": _matching_lines(str(held_ledger.get("content") or ""), search_terms),
                "contradiction_ledger": _matching_lines(
                    str(contradiction_ledger.get("content") or ""),
                    search_terms,
                ),
            },
        },
        "learnings": {
            "hot": hot_learnings,
            "matches": _matching_lines(str(hot_learnings.get("content") or ""), search_terms),
        },
        "memory": {
            "read_only": True,
            "source": ".learnings",
            "hot": hot_learnings,
            "matches": _matching_lines(str(hot_learnings.get("content") or ""), search_terms),
            "tool": "learnings_read(file='memory.md')",
        },
        "next_actions": [
            "Treat SOUL.md/IDENTITY/HEARTBEAT as behavioral truth for named-agent work.",
            "Treat AGENT_REGISTRY.json as index/discovery only.",
            "Use learnings_read(file='memory.md') for read-only HOT memory/reference context.",
            "Check held_spec_ledger and contradiction_ledger before governed edits.",
            "Do not write to .learnings, artifacts, registry, or SOUL files through MCP.",
        ],
    }

    if agent_name:
        agent: dict = {
            "name": agent_name,
            "found": False,
            "registry": registry.get(agent_name),
        }
        soul = _find_soul_md(agent_name)
        if soul:
            agent["found"] = True
            agent["soul_md"] = _bounded_read(soul, max_chars=8_000)
        heartbeat = _find_heartbeat(agent_name)
        if heartbeat:
            agent["found"] = True
            agent["heartbeat"] = heartbeat
        if agents_dir:
            agent_dir = _safe_child_path(agents_dir, agent_name)
            if agent_dir and agent_dir.is_dir():
                agent["found"] = True
                files = []
                for item in sorted(agent_dir.iterdir()):
                    if item.name.startswith("."):
                        continue
                    files.append({
                        "name": item.name,
                        "type": "directory" if item.is_dir() else "file",
                        "size": item.stat().st_size if item.is_file() else None,
                        "modified": _file_mtime_iso(item),
                    })
                    if len(files) >= _SNAPSHOT_LIST_CAP:
                        break
                agent["files"] = files
        if agent["registry"] is not None:
            agent["found"] = True
        if not agent["found"]:
            agent["error"] = f"Agent not found: {agent_name}"
        bundle["agent"] = agent

    if spec_id:
        bundle["spec"] = {
            "id": spec_id,
            "held_matches": bundle["knowledge"]["matches"]["held_spec_ledger"],
            "contradiction_matches": bundle["knowledge"]["matches"]["contradiction_ledger"],
            "latest_state_matches": bundle["knowledge"]["matches"]["latest_state"],
            "guardrails": [
                "If held_matches are present, do not implement constrained changes without operator approval.",
                "If contradiction_matches are present, resolve or escalate conflicts before editing.",
                "If no matches are present, absence in local artifacts is not proof of approval.",
            ],
        }

    if not agent_name and not spec_id:
        bundle["warning"] = "No agent_name or spec_id supplied; returning general Town handoff context."

    return bundle


def _find_heartbeat(agent_name: str) -> Optional[dict]:
    """Find and parse an agent's HEARTBEAT.md for status info."""
    agents_dir = _find_agents_dir()
    if not agents_dir:
        return None
    heartbeat = _safe_child_path(agents_dir, agent_name, "HEARTBEAT.md")
    if not heartbeat:
        return None
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
    soul = _safe_child_path(agents_dir, agent_name, "SOUL.md")
    if not soul:
        return None
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


def _get_knowledge_graph_dir() -> Optional[Path]:
    """Find artifacts/ops/knowledge_graph/."""
    artifacts = _get_artifacts_dir()
    if artifacts:
        kg = artifacts / "ops" / "knowledge_graph"
        if kg.is_dir():
            return kg
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
    def fleet_context_snapshot(
        summary: bool = False,
    ) -> str:
        """Return a bounded, read-only fleet bootstrap snapshot for IDE clients.

        The snapshot works in skills/context mode even when the live gateway is
        down. It never writes files and reports missing layers explicitly so
        Cursor can continue with partial but trustworthy context.

        Args:
            summary: If True, return compressed output that omits text blobs
                     (learnings content, latest_state content, held_spec raw text)
                     and returns only structured metadata, counts, and flags.
                     Reduces token consumption by ~60-80% for sessions that only
                     need state awareness without full text context.
        """
        return json.dumps(build_fleet_context_snapshot(summary=summary), indent=2)

    # -- agent_health_summary ---------------------------------------------

    @mcp.tool()
    def agent_health_summary() -> str:
        """Return compact actionable fleet health anomalies.

        Use this when a Cursor session needs a fast "is anything broken?"
        answer instead of the full fleet context snapshot. Read-only.
        """
        return json.dumps(build_agent_health_summary(), indent=2)

    # -- self_improvement_snapshot ------------------------------------------

    @mcp.tool()
    def self_improvement_snapshot(
        summary: bool = False,
    ) -> str:
        """Return read-only memory/knowledge health for recursive self-improvement.

        Audits HOT (.learnings/memory.md) and WARM tiers, compares operational
        anchors against knowledge-layer artifacts, and returns operator proposals.
        Never writes files; Town-Hermes automated memory sync remains frozen.

        Args:
            summary: If True, omit full reconciliation lists and return counts,
                     proposal samples, and the recursive loop checklist only.
        """
        return json.dumps(build_self_improvement_snapshot(summary=summary), indent=2)

    # -- town_brief ---------------------------------------------------------

    @mcp.tool()
    def town_brief() -> str:
        """Return a concise Cursor/Town operational brief.

        This is a human-facing summary of the read-only fleet context:
        source-of-truth paths, health counts, held-spec flags, gateway mode,
        and recommended next MCP calls for Cursor agents.
        """
        return json.dumps(build_town_brief(), indent=2)

    # -- town_handoff_bundle -----------------------------------------------

    @mcp.tool()
    def town_handoff_bundle(
        agent_name: str = "",
        spec_id: str = "",
        include_learnings: bool = True,
    ) -> str:
        """Return a bounded read-only context bundle for Cursor handoffs.

        Use this before handing a named agent or governed spec to another
        Cursor agent. The bundle packages source-of-truth paths, agent
        SOUL/registry/heartbeat context, latest-state snippets, held-spec
        and contradiction matches, and HOT learnings excerpts. It never
        writes fleet state.
        """
        return json.dumps(
            build_town_handoff_bundle(
                agent_name=agent_name,
                spec_id=spec_id,
                include_learnings=include_learnings,
            ),
            indent=2,
        )

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
                        "path": _display_path(agent_dir),
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
            agent_path = _safe_child_path(agents_dir, name, file)
            if agent_path is None:
                return json.dumps({"error": "Invalid agent path"}, indent=2)
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
            skill_path = _safe_child_path(skills_dir, name, file)
            if skill_path is None:
                return json.dumps({"error": "Invalid skill path"}, indent=2)
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
                candidate = _safe_child_path(category, name, file)
                if candidate is None:
                    continue
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
        registry = _registry_agents(_load_agent_registry())
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
        registry = _registry_agents(_load_agent_registry())
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
            agent_dir = _safe_child_path(agents_dir, name)
            if agent_dir and agent_dir.is_dir():
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
        """Read Hermes learnings/memory reference files.

        The .learnings/ directory contains read-only memory/reference files:
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

        target = _safe_child_path(learnings_dir, file)
        if target is None:
            return json.dumps({"error": "Invalid learnings path"}, indent=2)

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

        target = _safe_child_path(artifacts_dir, path) if path else artifacts_dir.resolve()
        if target is None:
            return json.dumps({"error": "Invalid artifacts path"}, indent=2)
        if not target.is_dir():
            return json.dumps({
                "error": f"Directory not found: {target}",
            }, indent=2)

        depth = max(1, min(depth, 4))
        entries = _dir_listing(target, max_depth=depth)

        return json.dumps({
            "path": _display_path(target),
            "depth": depth,
            "entries": entries,
        }, indent=2)

    # -- knowledge_query ---------------------------------------------------

    @mcp.tool()
    def knowledge_query(question: str) -> str:
        """Query knowledge graph artifacts with bounded keyword matching.

        Reads `artifacts/ops/knowledge_graph/nodes.jsonl` and `edges.jsonl`
        when present. This is deterministic keyword matching, not semantic
        search, and never writes files.
        """
        kg_dir = _get_knowledge_graph_dir()
        if not kg_dir:
            return json.dumps({
                "question": question,
                "error": "knowledge_graph directory not found",
                "matches": [],
                "related_edges": [],
            }, indent=2)

        stop_words = {
            "what", "which", "who", "how", "does", "do", "is", "are", "the",
            "a", "an", "in", "on", "of", "to", "for", "and", "or", "that",
            "this", "from", "with", "by", "at", "it", "its", "my",
        }
        keywords = [
            word.lower().strip("?.,!;:")
            for word in str(question or "").split()
        ]
        keywords = [
            word for word in keywords
            if word and word not in stop_words and len(word) > 2
        ][:20]

        nodes: list[dict] = []
        edges: list[dict] = []
        for filename, target in (("nodes.jsonl", nodes), ("edges.jsonl", edges)):
            path = kg_dir / filename
            if not path.exists():
                continue
            try:
                for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                    line = line.strip()
                    if line:
                        target.append(json.loads(line))
            except (OSError, json.JSONDecodeError) as exc:
                return json.dumps({
                    "question": question,
                    "error": f"Failed to parse {filename}: {exc}",
                    "matches": [],
                    "related_edges": [],
                }, indent=2)

        matches: list[dict] = []
        matched_ids: set[str] = set()
        for node in nodes:
            node_text = json.dumps(node, ensure_ascii=False).lower()
            score = sum(1 for kw in keywords if kw in node_text)
            if score > 0:
                item = dict(node)
                item["_relevance"] = score
                matches.append(item)
                node_id = item.get("id") or item.get("name")
                if node_id:
                    matched_ids.add(str(node_id))

        matches.sort(key=lambda item: item.get("_relevance", 0), reverse=True)
        matches = matches[:20]
        for item in matches:
            item.pop("_relevance", None)

        related_edges: list[dict] = []
        for edge in edges:
            source = str(edge.get("source", edge.get("from", "")))
            target = str(edge.get("target", edge.get("to", "")))
            edge_text = json.dumps(edge, ensure_ascii=False).lower()
            if (
                source in matched_ids
                or target in matched_ids
                or any(kw in edge_text for kw in keywords)
            ):
                related_edges.append(edge)
            if len(related_edges) >= 30:
                break

        return json.dumps({
            "question": question,
            "as_of": datetime.now(timezone.utc).isoformat(),
            "keywords_used": keywords,
            "matches": matches,
            "related_edges": related_edges,
            "stats": {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "matched_nodes": len(matches),
                "related_edges": len(related_edges),
            },
        }, indent=2)

    logger.debug("Registered 12 skills/knowledge MCP tools")
