#!/usr/bin/env python3
"""openclaw_agent_self_check.py

Self-check utility for OpenClaw agent configuration + policy surfaces.

What it prints (best-effort):
- agentId (resolved)
- sessionKey (resolved from `openclaw status --json` recent sessions)
- auth profiles (from OpenClaw config + model status)
- effective model (resolved default for the agent)
- allowed tools (expanded from agent tool policy in config)

Usage:
  python3 scripts/openclaw_agent_self_check.py
  python3 scripts/openclaw_agent_self_check.py --agent architect
  python3 scripts/openclaw_agent_self_check.py --agent walletdb-research --json

Notes:
- No sudo required.
- Requires the `openclaw` CLI on PATH and a readable OpenClaw config.
- Tool policy expansion is heuristic for group:* entries; unknown groups are
  preserved verbatim.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


GROUP_EXPANSION: Dict[str, List[str]] = {
    # Common OpenClaw group shorthands.
    "group:fs": ["read", "write", "edit"],
    "group:runtime": ["exec", "process"],
    "group:web": ["web_search", "web_fetch", "browser"],
    "group:ui": ["browser", "canvas"],
    "group:devices": ["nodes", "canvas"],
}

# A conservative baseline of tool names that frequently exist in OpenClaw.
# We only use this for "default profile" reporting.
KNOWN_TOOLS: List[str] = sorted(
    {
        "read",
        "write",
        "edit",
        "exec",
        "process",
        "web_search",
        "web_fetch",
        "browser",
        "canvas",
        "nodes",
        "message",
        "tts",
    }
)


@dataclass
class SelfCheck:
    agent_id: str
    session_key: Optional[str]
    session_id: Optional[str]
    effective_model: Optional[str]
    auth_profiles_config: List[str]
    auth_profiles_effective: List[Dict[str, Any]]
    tools_policy: Dict[str, Any]
    allowed_tools_effective: List[str]
    denied_tools_effective: List[str]
    unknown_groups: List[str]
    sources: Dict[str, Any]


def _run_json(cmd: List[str], timeout_s: int = 30) -> Dict[str, Any]:
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_s,
    )
    if p.returncode != 0:
        raise RuntimeError(
            f"command failed ({p.returncode}): {' '.join(cmd)}\n"
            f"stderr:\n{p.stderr.strip()}"
        )
    out = p.stdout.strip()
    if not out:
        return {}
    try:
        return json.loads(out)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"command did not return JSON: {' '.join(cmd)}\n"
            f"error: {e}\nstdout (first 2k):\n{out[:2000]}"
        )


def _read_json(path: pathlib.Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _default_config_path() -> pathlib.Path:
    # Respect explicit override first.
    p = os.environ.get("OPENCLAW_CONFIG_PATH")
    if p:
        return pathlib.Path(os.path.expanduser(p))

    # If a profile is in effect, OpenClaw uses ~/.openclaw-<name>/openclaw.json.
    profile = os.environ.get("OPENCLAW_PROFILE") or os.environ.get("PI_OPENCLAW_PROFILE")
    if profile:
        return pathlib.Path(os.path.expanduser(f"~/.openclaw-{profile}/openclaw.json"))

    return pathlib.Path(os.path.expanduser("~/.openclaw/openclaw.json"))


def _resolve_agent_id(explicit: Optional[str], config: Dict[str, Any]) -> str:
    if explicit:
        return explicit

    env_id = os.environ.get("OPENCLAW_AGENT_ID")
    if env_id:
        return env_id

    # Try to match by workspace == cwd (or a parent).
    cwd = pathlib.Path.cwd().resolve()
    agents = (config.get("agents") or {}).get("list") or []
    best: Optional[Tuple[int, str]] = None
    for a in agents:
        ws = a.get("workspace")
        aid = a.get("id")
        if not ws or not aid:
            continue
        try:
            ws_path = pathlib.Path(ws).resolve()
        except Exception:
            continue
        if ws_path == cwd or ws_path in cwd.parents:
            # Prefer the deepest workspace match.
            depth = len(ws_path.parts)
            if best is None or depth > best[0]:
                best = (depth, aid)

    if best:
        return best[1]

    # Fallback to configured defaultAgentId if present.
    try:
        status = _run_json(["openclaw", "status", "--json"], timeout_s=15)
        hb = status.get("heartbeat") or {}
        default_aid = hb.get("defaultAgentId")
        if default_aid:
            return str(default_aid)
    except Exception:
        pass

    return "main"


def _agent_tools_policy(config: Dict[str, Any], agent_id: str) -> Dict[str, Any]:
    agents = (config.get("agents") or {}).get("list") or []
    for a in agents:
        if a.get("id") == agent_id:
            return a.get("tools") or {}
    return {}


def _expand_allow_deny(tools_policy: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
    allow_raw = tools_policy.get("allow") or []
    deny_raw = tools_policy.get("deny") or []

    unknown_groups: List[str] = []

    def expand(items: List[str]) -> List[str]:
        out: List[str] = []
        for it in items:
            if it in GROUP_EXPANSION:
                out.extend(GROUP_EXPANSION[it])
            elif isinstance(it, str) and it.startswith("group:"):
                unknown_groups.append(it)
                out.append(it)
            else:
                out.append(it)
        # stable unique
        seen = set()
        uniq = []
        for x in out:
            if x not in seen:
                uniq.append(x)
                seen.add(x)
        return uniq

    return expand(list(allow_raw)), expand(list(deny_raw)), sorted(set(unknown_groups))


def _effective_allowed_tools(config: Dict[str, Any], agent_id: str) -> Tuple[List[str], List[str], List[str], Dict[str, Any]]:
    tools_policy = _agent_tools_policy(config, agent_id)
    profile = tools_policy.get("profile")

    allow, deny, unknown_groups = _expand_allow_deny(tools_policy)

    if profile == "minimal":
        # minimal starts from nothing.
        allowed = [t for t in allow if t not in deny]
        denied = sorted(set(deny))
        return sorted(allowed), denied, unknown_groups, tools_policy

    if tools_policy:
        # If a tools policy exists but profile isn't minimal, treat it as
        # additive allow/deny on top of "default".
        base = set(KNOWN_TOOLS)
        base.update([t for t in allow if not (isinstance(t, str) and t.startswith("group:"))])
        for d in deny:
            base.discard(d)
        return sorted(base), sorted(set(deny)), unknown_groups, tools_policy

    # No policy => default.
    return KNOWN_TOOLS, [], [], tools_policy


def _recent_session_for_agent(status: Dict[str, Any], agent_id: str) -> Tuple[Optional[str], Optional[str]]:
    sessions = ((status.get("sessions") or {}).get("recent")) or []
    for s in sessions:
        if s.get("agentId") == agent_id:
            return s.get("key"), s.get("sessionId")
    return None, None


def build_self_check(agent_id: Optional[str]) -> SelfCheck:
    config_path = _default_config_path()
    if not config_path.exists():
        raise FileNotFoundError(f"OpenClaw config not found: {config_path}")

    config = _read_json(config_path)
    resolved_agent_id = _resolve_agent_id(agent_id, config)

    status = {}
    try:
        status = _run_json(["openclaw", "status", "--json"], timeout_s=20)
    except Exception:
        status = {}

    session_key, session_id = _recent_session_for_agent(status, resolved_agent_id)

    # Model status is per-agent aware.
    models_status = {}
    effective_model = None
    auth_effective: List[Dict[str, Any]] = []
    try:
        models_status = _run_json(
            ["openclaw", "models", "status", "--json", "--agent", resolved_agent_id],
            timeout_s=30,
        )
        effective_model = models_status.get("resolvedDefault") or models_status.get("defaultModel")
        # Prefer oauth profile list if present.
        auth_effective = (((models_status.get("auth") or {}).get("oauth") or {}).get("profiles")) or []
    except Exception:
        models_status = {}

    auth_profiles_config = sorted(((config.get("auth") or {}).get("profiles") or {}).keys())

    allowed_tools, denied_tools, unknown_groups, tools_policy = _effective_allowed_tools(config, resolved_agent_id)

    sources = {
        "configPath": str(config_path),
        "status": {
            "available": bool(status),
            "command": "openclaw status --json",
        },
        "models": {
            "available": bool(models_status),
            "command": f"openclaw models status --json --agent {resolved_agent_id}",
        },
    }

    return SelfCheck(
        agent_id=resolved_agent_id,
        session_key=session_key,
        session_id=session_id,
        effective_model=effective_model,
        auth_profiles_config=auth_profiles_config,
        auth_profiles_effective=auth_effective,
        tools_policy=tools_policy,
        allowed_tools_effective=allowed_tools,
        denied_tools_effective=denied_tools,
        unknown_groups=unknown_groups,
        sources=sources,
    )


def _to_plain(sc: SelfCheck) -> str:
    lines = []
    lines.append("OpenClaw Agent Self-Check")
    lines.append("======================")
    lines.append(f"agentId:     {sc.agent_id}")
    lines.append(f"sessionKey:  {sc.session_key or '(not found)'}")
    lines.append(f"sessionId:   {sc.session_id or '(not found)'}")
    lines.append(f"model:       {sc.effective_model or '(unknown)'}")
    lines.append("")

    lines.append("Auth profiles (config):")
    if sc.auth_profiles_config:
        for p in sc.auth_profiles_config:
            lines.append(f"  - {p}")
    else:
        lines.append("  (none found)")

    lines.append("")
    lines.append("Auth profiles (effective / model status):")
    if sc.auth_profiles_effective:
        for p in sc.auth_profiles_effective:
            label = p.get("label") or p.get("profileId")
            status = p.get("status")
            typ = p.get("type")
            lines.append(f"  - {label} [{typ}] status={status}")
    else:
        lines.append("  (unavailable)")

    lines.append("")
    lines.append("Tools policy (config):")
    lines.append(json.dumps(sc.tools_policy or {}, indent=2, sort_keys=True))

    lines.append("")
    lines.append("Allowed tools (effective):")
    lines.append("  " + ", ".join(sc.allowed_tools_effective) if sc.allowed_tools_effective else "  (none)")
    if sc.denied_tools_effective:
        lines.append("Denied tools (effective):")
        lines.append("  " + ", ".join(sc.denied_tools_effective))
    if sc.unknown_groups:
        lines.append("Unknown group:* entries (not expanded):")
        for g in sc.unknown_groups:
            lines.append(f"  - {g}")

    lines.append("")
    lines.append("Sources:")
    lines.append(json.dumps(sc.sources, indent=2, sort_keys=True))
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--agent", help="Agent id to inspect (default: auto-resolve)")
    ap.add_argument("--json", action="store_true", help="Output JSON")
    args = ap.parse_args()

    sc = build_self_check(args.agent)
    if args.json:
        print(json.dumps(sc.__dict__, indent=2, sort_keys=True))
    else:
        sys.stdout.write(_to_plain(sc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
