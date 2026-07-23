"""Prompt-size diagnostic: ``hermes prompt-size``.

Reports a byte/char breakdown of the system prompt the agent would build for
a fresh session — system prompt total, the ``<available_skills>`` index,
memory + user profile, and tool-schema JSON. Lets users see where their fixed
prompt budget goes (issue #34667) without parsing a saved session JSON by hand.

The diagnostic builds a real inspection agent (so the numbers match what
actually ships on the wire) but never makes a network call: it passes dummy
credentials so ``AIAgent.__init__`` takes the direct-construction path, then
calls ``build_system_prompt_parts`` / inspects ``agent.tools`` offline.
"""

from __future__ import annotations

import io
import json
import re
import subprocess
import sys
from contextlib import redirect_stdout
from typing import Any, Dict, List, Tuple

# The skills index is wrapped in this tag pair inside the stable tier.
_SKILLS_BLOCK_RE = re.compile(r"<available_skills>.*?</available_skills>", re.DOTALL)


def _bytes(s: str) -> int:
    return len(s.encode("utf-8"))


def _build_inspection_agent(platform: str) -> Any:
    """Construct an offline AIAgent for prompt inspection.

    Dummy ``api_key`` + ``base_url`` force the direct-construction path in
    ``run_agent.py`` (no provider auto-detection, no network). Toolsets and
    platform come from the caller so the breakdown matches a real session.
    """
    from run_agent import AIAgent
    from hermes_cli.config import load_config
    from hermes_cli.tools_config import _get_platform_tools

    cfg = load_config()
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
    model = model_cfg.get("default") or model_cfg.get("model") or ""

    # Resolve platform-specific toolsets the same way the gateway does.
    enabled_toolsets = sorted(_get_platform_tools(cfg, platform))
    agent_cfg = cfg.get("agent") or {}
    disabled_toolsets = agent_cfg.get("disabled_toolsets") or None

    return AIAgent(
        model=model,
        api_key="inspect-only",
        base_url="https://openrouter.ai/api/v1",
        quiet_mode=True,
        save_trajectories=False,
        platform=platform,
        enabled_toolsets=enabled_toolsets,
        disabled_toolsets=disabled_toolsets,
    )


_CHILD_DIAGNOSTIC_LIMIT = 1000


def _bounded_child_output(value: str) -> str:
    """Return a compact single-line child diagnostic suitable for CLI errors."""
    text = " ".join((value or "").strip().split())
    if len(text) <= _CHILD_DIAGNOSTIC_LIMIT:
        return text
    return text[:_CHILD_DIAGNOSTIC_LIMIT] + "…"


def _require_measurement(
    data: Dict[str, Any], section: str, key: str, profile: str
) -> int:
    """Read one required non-negative integer from a child measurement."""
    section_data = data.get(section)
    value = section_data.get(key) if isinstance(section_data, dict) else None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(
            f"Invalid prompt-size output for profile '{profile}': "
            f"{section}.{key} must be a non-negative integer"
        )
    return value


def compute_all_profile_breakdowns(platform: str = "cli") -> List[Dict[str, Any]]:
    """Measure every profile in a fresh isolated CLI process.

    Profile selection must happen before Hermes modules import because many
    paths cache ``HERMES_HOME`` at module scope. Spawning the normal CLI with an
    explicit ``--profile`` preserves that contract and makes each result match
    what the selected profile would actually send on a fresh session.
    """
    from hermes_cli.profiles import list_profiles

    results: List[Dict[str, Any]] = []
    for profile in list_profiles():
        name = profile.name
        command = [
            sys.executable,
            "-m",
            "hermes_cli.main",
            "--profile",
            name,
            "prompt-size",
            "--platform",
            platform,
            "--json",
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Could not measure profile '{name}': child timed out after "
                f"{exc.timeout} seconds"
            ) from exc
        except OSError as exc:
            raise RuntimeError(
                f"Could not start prompt-size child for profile '{name}': {exc}"
            ) from exc
        if completed.returncode != 0:
            detail = _bounded_child_output(
                completed.stderr or completed.stdout or "unknown error"
            )
            raise RuntimeError(f"Could not measure profile '{name}': {detail}")
        try:
            data = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            detail = _bounded_child_output(completed.stdout) or "empty output"
            raise RuntimeError(
                f"Could not parse prompt-size output for profile '{name}': "
                f"{exc}; child output: {detail}"
            ) from exc
        if not isinstance(data, dict):
            raise RuntimeError(
                f"Invalid prompt-size output for profile '{name}': expected a JSON object"
            )
        prompt_bytes = _require_measurement(data, "system_prompt", "bytes", name)
        schema_bytes = _require_measurement(data, "tools", "json_bytes", name)
        _require_measurement(data, "tools", "count", name)
        data["profile"] = name
        data["fixed_bytes"] = prompt_bytes + schema_bytes
        results.append(data)
    results.sort(key=lambda row: int(row["fixed_bytes"]), reverse=True)
    return results


def compute_prompt_breakdown(platform: str = "cli") -> Dict[str, Any]:
    """Return a dict of prompt-size measurements for a fresh session.

    Keys: ``system_prompt`` (chars/bytes), ``skills_index``, ``memory``,
    ``user_profile``, ``tools`` (count + json bytes), and ``sections`` (a list
    of (label, chars, bytes) for the three prompt tiers).
    """
    from agent.system_prompt import build_system_prompt, build_system_prompt_parts

    agent = _build_inspection_agent(platform)

    parts = build_system_prompt_parts(agent)
    full = build_system_prompt(agent)

    stable = parts.get("stable", "")
    context = parts.get("context", "")
    volatile = parts.get("volatile", "")

    # Skills index — the <available_skills> block (the largest single block
    # when many skills are installed). Measured inside the stable tier.
    skills_match = _SKILLS_BLOCK_RE.search(stable)
    skills_index = skills_match.group(0) if skills_match else ""

    # Memory + user profile live in the volatile tier. We re-derive their
    # blocks directly from the memory store so the numbers are attributable
    # even though they're joined into ``volatile``.
    memory_block = ""
    user_block = ""
    store = getattr(agent, "_memory_store", None)
    if store is not None:
        try:
            if getattr(agent, "_memory_enabled", True):
                memory_block = store.format_for_system_prompt("memory") or ""
            if getattr(agent, "_user_profile_enabled", True):
                user_block = store.format_for_system_prompt("user") or ""
        except Exception:
            pass

    # Tool-schema JSON — the other half of the fixed per-call payload.
    tools = getattr(agent, "tools", None) or []
    tools_json = json.dumps(tools, ensure_ascii=False)

    sections: List[Tuple[str, int, int]] = [
        ("stable (identity/guidance/skills)", len(stable), _bytes(stable)),
        ("context (AGENTS.md/cwd files)", len(context), _bytes(context)),
        ("volatile (memory/profile/timestamp)", len(volatile), _bytes(volatile)),
    ]

    return {
        "platform": platform,
        "model": getattr(agent, "model", "") or "",
        "system_prompt": {"chars": len(full), "bytes": _bytes(full)},
        "skills_index": {"chars": len(skills_index), "bytes": _bytes(skills_index)},
        "memory": {"chars": len(memory_block), "bytes": _bytes(memory_block)},
        "user_profile": {"chars": len(user_block), "bytes": _bytes(user_block)},
        "tools": {"count": len(tools), "json_bytes": _bytes(tools_json)},
        "sections": sections,
    }


def _fmt_kb(n: int) -> str:
    return f"{n / 1024:.1f} KB"


def render_profile_comparison(
    rows: List[Dict[str, Any]], *, platform: str = "cli"
) -> str:
    """Render a largest-first comparison of fixed prompt footprints."""
    ordered = sorted(rows, key=lambda row: int(row.get("fixed_bytes", 0)), reverse=True)
    profile_width = max(
        18,
        max((len(str(row.get("profile", ""))) for row in ordered), default=0),
    )
    lines = [
        f"Profile prompt-size comparison (platform={platform})",
        "",
        f"  {'Profile':<{profile_width}} {'Model':<24} {'Prompt':>10} {'Schemas':>10} {'Tools':>7} {'Fixed':>10}",
        f"  {'-' * profile_width} {'-' * 24} {'-' * 10} {'-' * 10} {'-' * 7} {'-' * 10}",
    ]
    for row in ordered:
        prompt_bytes = int(row.get("system_prompt", {}).get("bytes", 0))
        tools = row.get("tools", {})
        schema_bytes = int(tools.get("json_bytes", 0))
        tool_count = int(tools.get("count", 0))
        fixed_bytes = int(row.get("fixed_bytes", prompt_bytes + schema_bytes))
        model = str(row.get("model", "") or "unset")[:24]
        lines.append(
            f"  {str(row.get('profile', '')):<{profile_width}} {model:<24} "
            f"{_fmt_kb(prompt_bytes):>10} {_fmt_kb(schema_bytes):>10} "
            f"{tool_count:>7} {_fmt_kb(fixed_bytes):>10}"
        )
    lines.extend(
        [
            "",
            "  Fixed payload = system prompt + tool-schema JSON (conversation history excluded).",
        ]
    )
    return "\n".join(lines)


def render_breakdown(data: Dict[str, Any]) -> str:
    """Render the breakdown as plain text suitable for a terminal."""
    lines: List[str] = []
    sp = data["system_prompt"]
    lines.append(f"Prompt-size breakdown (platform={data['platform']}, model={data['model'] or 'unset'})")
    lines.append("")
    lines.append(f"  System prompt total : {sp['bytes']:>8,} B  ({_fmt_kb(sp['bytes'])}, {sp['chars']:,} chars)")
    lines.append("")
    lines.append("  Major blocks:")
    si = data["skills_index"]
    mem = data["memory"]
    up = data["user_profile"]
    lines.append(f"    skills index       : {si['bytes']:>8,} B  ({_fmt_kb(si['bytes'])})")
    lines.append(f"    memory             : {mem['bytes']:>8,} B  ({_fmt_kb(mem['bytes'])})")
    lines.append(f"    user profile       : {up['bytes']:>8,} B  ({_fmt_kb(up['bytes'])})")
    lines.append("")
    lines.append("  Prompt tiers:")
    for label, chars, byts in data["sections"]:
        lines.append(f"    {label:<36}: {byts:>8,} B  ({_fmt_kb(byts)})")
    lines.append("")
    tools = data["tools"]
    lines.append(f"  Tool schemas         : {tools['json_bytes']:>8,} B  ({_fmt_kb(tools['json_bytes'])}, {tools['count']} tools)")
    return "\n".join(lines)


def _compute_for_output(compute: Any, platform: str, *, as_json: bool) -> Any:
    """Keep machine-readable stdout pure while preserving incidental diagnostics."""
    if not as_json:
        return compute(platform)
    captured = io.StringIO()
    try:
        with redirect_stdout(captured):
            return compute(platform)
    finally:
        diagnostic = captured.getvalue()
        if diagnostic:
            print(diagnostic, file=sys.stderr, end="" if diagnostic.endswith("\n") else "\n")


def cmd_prompt_size(args: Any) -> None:
    """Entry point for ``hermes prompt-size``."""
    platform = getattr(args, "platform", "cli") or "cli"
    as_json = getattr(args, "json", False)
    all_profiles = getattr(args, "all_profiles", False)
    if all_profiles:
        try:
            rows = _compute_for_output(
                compute_all_profile_breakdowns,
                platform,
                as_json=as_json,
            )
        except Exception as e:
            print(
                f"Could not compute all-profile prompt-size breakdown: {e}",
                file=sys.stderr,
            )
            raise SystemExit(1) from e
        if as_json:
            print(
                json.dumps(
                    {"platform": platform, "profiles": rows},
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(render_profile_comparison(rows, platform=platform))
        return
    try:
        data = _compute_for_output(
            compute_prompt_breakdown,
            platform,
            as_json=as_json,
        )
    except Exception as e:
        print(f"Could not compute prompt-size breakdown: {e}", file=sys.stderr)
        raise SystemExit(1) from e
    if as_json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(render_breakdown(data))
