"""Context-budget audit for ``hermes context audit``.

This diagnostic is read-only and offline.  It measures the fixed prompt/context
surfaces that tend to surprise users: project context files from the current
working directory, the fresh-session prompt breakdown, memory/profile blocks,
and tool-schema payload size.  It never calls an LLM.
"""

from __future__ import annotations

import io
import json
import logging
import os
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Any, Dict, List

def _bytes(text: str) -> int:
    return len(text.encode("utf-8"))


def _fmt_kb(n: int) -> str:
    return f"{n / 1024:.1f} KB"


def _safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


@contextmanager
def _suppress_context_truncation_warnings():
    """Keep audit output compact while measuring known-large context files.

    The prompt builder logs truncation warnings for real agent startup. This
    audit reports the same over-cap condition in its own table, so repeating
    those warnings on stderr makes the diagnostic noisy without adding signal.
    """
    logger = logging.getLogger("agent.prompt_builder")
    previous_disabled = logger.disabled
    previous_level = logger.level
    try:
        logger.disabled = True
        logger.setLevel(logging.CRITICAL + 1)
        yield
    finally:
        logger.disabled = previous_disabled
        logger.setLevel(previous_level)


def _context_file_limit(prompt_builder: Any) -> int:
    """Return the same effective cap used by the prompt builder."""
    getter = getattr(prompt_builder, "_get_context_file_max_chars", None)
    if getter:
        return int(getter())
    return int(getattr(prompt_builder, "CONTEXT_FILE_MAX_CHARS", 20_000))


def _find_project_context_files(cwd: Path, prompt_builder: Any) -> List[Path]:
    """Mirror ``build_context_files_prompt`` discovery, returning only loaded files.

    The builder uses first-match-wins priority: nearest .hermes.md/HERMES.md
    up to the git root, else AGENTS.md, else CLAUDE.md, else cursor rules.
    The audit must not list every candidate file because that over-reports
    files that the real prompt never injects.
    """
    finder = getattr(prompt_builder, "_find_hermes_md", None)
    hermes_md = finder(cwd) if finder else None
    if hermes_md and hermes_md.is_file():
        return [hermes_md]

    for name in ("AGENTS.md", "agents.md"):
        candidate = cwd / name
        if candidate.is_file():
            return [candidate]

    for name in ("CLAUDE.md", "claude.md"):
        candidate = cwd / name
        if candidate.is_file():
            return [candidate]

    cursor_files: List[Path] = []
    cursorrules = cwd / ".cursorrules"
    if cursorrules.is_file():
        cursor_files.append(cursorrules)
    cursor_dir = cwd / ".cursor" / "rules"
    if cursor_dir.is_dir():
        cursor_files.extend(p for p in sorted(cursor_dir.glob("*.mdc")) if p.is_file())
    return cursor_files


def measure_context_budget(cwd: str | Path | None = None, platform: str = "cli") -> Dict[str, Any]:
    """Return a read-only context budget report.

    ``cwd`` defaults to the process cwd.  The returned dict is JSON-serialisable
    and intentionally contains sizes/paths only, not file contents or secrets.
    """
    from agent import prompt_builder as pb
    from hermes_cli.config import get_hermes_home
    from hermes_cli.prompt_size import compute_prompt_breakdown

    root = Path(cwd or Path.cwd()).expanduser().resolve()
    context_limit = _context_file_limit(pb)

    files: List[Dict[str, Any]] = []
    for path in _find_project_context_files(root, pb):
        text = _safe_read(path)
        files.append(
            {
                "path": str(path),
                "chars": len(text),
                "bytes": _bytes(text),
                "over_default_limit": len(text) > int(context_limit),
            }
        )

    try:
        with _suppress_context_truncation_warnings(), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            rendered_project_context = pb.build_context_files_prompt(cwd=str(root), skip_soul=True) or ""
            if hasattr(pb, "drain_truncation_warnings"):
                pb.drain_truncation_warnings()
        project_context_error = ""
    except Exception as exc:  # pragma: no cover - defensive
        rendered_project_context = ""
        project_context_error = str(exc)

    try:
        old_cwd = os.getcwd()
        os.chdir(root)
        try:
            with _suppress_context_truncation_warnings(), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                prompt = compute_prompt_breakdown(platform=platform)
                if hasattr(pb, "drain_truncation_warnings"):
                    pb.drain_truncation_warnings()
        finally:
            os.chdir(old_cwd)
        prompt_error = ""
    except Exception as exc:  # pragma: no cover - defensive
        prompt = {}
        prompt_error = str(exc)

    home = Path(get_hermes_home())
    home_files = []
    for rel in ("SOUL.md", "memories/MEMORY.md", "memories/USER.md"):
        p = home / rel
        text = _safe_read(p) if p.exists() else ""
        home_files.append({"path": str(p), "chars": len(text), "bytes": _bytes(text), "exists": p.exists()})

    recommendations: List[str] = []
    if rendered_project_context:
        recommendations.append(
            "Project context is non-empty for this cwd; for routine Ops, launch from a neutral cwd and use git -C/path commands."
        )
    large_files = [f for f in files if f["over_default_limit"]]
    if large_files:
        recommendations.append(
            "At least one context file exceeds the configured/default char cap; split or narrow it if it loads in non-coding sessions."
        )
    if prompt and (prompt.get("skills_index") or {}).get("bytes", 0) > 5 * 1024:
        recommendations.append("Skills index is a major fixed block; disable/archive unused overlapping skills before trimming safety guidance.")
    if prompt and (prompt.get("tools") or {}).get("json_bytes", 0) > 20 * 1024:
        recommendations.append("Tool schemas are large; prefer minimal toolsets for one-shot/profile-specific routine work.")

    return {
        "cwd": str(root),
        "platform": platform,
        "context_file_char_limit": int(context_limit),
        "context_files": files,
        "rendered_project_context": {
            "chars": len(rendered_project_context),
            "bytes": _bytes(rendered_project_context),
            "error": project_context_error,
        },
        "home_files": home_files,
        "prompt_breakdown": prompt,
        "prompt_error": prompt_error,
        "recommendations": recommendations,
    }


def render_context_audit(data: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"Context audit (platform={data['platform']})")
    lines.append(f"  cwd: {data['cwd']}")
    lines.append(f"  context file cap: {data['context_file_char_limit']:,} chars")
    lines.append("")

    rpc = data["rendered_project_context"]
    lines.append(
        f"  Rendered project context: {rpc['bytes']:>8,} B  ({_fmt_kb(rpc['bytes'])}, {rpc['chars']:,} chars)"
    )
    if rpc.get("error"):
        lines.append(f"    error: {rpc['error']}")

    lines.append("")
    lines.append("  Context files in cwd:")
    if data["context_files"]:
        for item in sorted(data["context_files"], key=lambda x: x["bytes"], reverse=True):
            flag = "  OVER-CAP" if item.get("over_default_limit") else ""
            lines.append(f"    {item['bytes']:>8,} B  {_fmt_kb(item['bytes']):>8}  {item['path']}{flag}")
    else:
        lines.append("    (none)")

    lines.append("")
    lines.append("  Hermes home prompt files:")
    for item in data["home_files"]:
        exists = "" if item.get("exists") else "  missing"
        lines.append(f"    {item['bytes']:>8,} B  {_fmt_kb(item['bytes']):>8}  {item['path']}{exists}")

    prompt = data.get("prompt_breakdown") or {}
    if prompt:
        sp = prompt.get("system_prompt", {})
        lines.append("")
        lines.append(
            f"  Fresh-session system prompt: {sp.get('bytes', 0):>8,} B  ({_fmt_kb(sp.get('bytes', 0))})"
        )
        si = prompt.get("skills_index", {})
        mem = prompt.get("memory", {})
        up = prompt.get("user_profile", {})
        tools = prompt.get("tools", {})
        lines.append(f"    skills index : {si.get('bytes', 0):>8,} B  ({_fmt_kb(si.get('bytes', 0))})")
        lines.append(f"    memory       : {mem.get('bytes', 0):>8,} B  ({_fmt_kb(mem.get('bytes', 0))})")
        lines.append(f"    user profile : {up.get('bytes', 0):>8,} B  ({_fmt_kb(up.get('bytes', 0))})")
        lines.append(
            f"    tool schemas : {tools.get('json_bytes', 0):>8,} B  ({_fmt_kb(tools.get('json_bytes', 0))}, {tools.get('count', 0)} tools)"
        )
    elif data.get("prompt_error"):
        lines.append("")
        lines.append(f"  Prompt breakdown unavailable: {data['prompt_error']}")

    if data["recommendations"]:
        lines.append("")
        lines.append("  Recommendations:")
        for rec in data["recommendations"]:
            lines.append(f"    - {rec}")
    return "\n".join(lines)


def cmd_context(args: Any) -> None:
    action = getattr(args, "context_command", None) or "audit"
    if action != "audit":
        raise SystemExit(f"unknown context subcommand: {action}")
    data = measure_context_budget(cwd=getattr(args, "cwd", None), platform=getattr(args, "platform", "cli") or "cli")
    if getattr(args, "json", False):
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(render_context_audit(data))
