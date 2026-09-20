"""Per-file manifest of the context/instruction files behind the ``/context`` "Rules" figure.

Read-only: enumerates the same candidates ``build_context_files_prompt`` loads (through
``agent.prompt_builder.discover_context_files`` — one discovery walk, so the listing cannot drift from the
prompt) and reports, per file, its size and whether it was loaded, truncated over the context-file cap,
shadowed by a higher-priority context type, blocked by the injection scan (or, for the user's own SOUL.md,
flagged but loaded), empty/unreadable, or suppressed by the install-tree guard. Nothing here builds a prompt or touches the truncation-warning ContextVar, so it
is free of cache impact.

Approximations (the manifest re-derives, it does not re-render): the truncation check sizes the raw
``## label`` section, so a .hermes.md whose YAML frontmatter the builder strips can read a few chars larger
here, and the AGENTS.md directory-chain cap (applied to the merged chain after per-file caps) is not modelled.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent import prompt_builder as _pb
from agent.model_metadata import estimate_tokens_rough

# status -> (glyph, note shown after the token count; "" = none)
_STATUS_DISPLAY = {
    "loaded": ("✓", ""),
    "truncated": ("◐", "truncated — over context_file_max_chars"),
    "shadowed": ("○", "not loaded — higher-priority context type wins"),
    "blocked": ("✗", "not loaded — blocked by the prompt-injection scan"),
    "flagged": ("⚠", "loaded — matched prompt-injection pattern(s); review the file"),
    "empty": ("○", "not loaded — empty file"),
    "unreadable": ("✗", "not loaded — could not be read"),
    "suppressed": ("○", "not loaded — cwd fell back to the Hermes install tree"),
}


def _entry(label: str, path: Path, content: str, status: str) -> Dict[str, Any]:
    return {
        "label": label, "path": str(path), "chars": len(content), "est_tokens": estimate_tokens_rough(content),
        "loaded": status in ("loaded", "truncated", "flagged"), "status": status,
    }


def _empty_status(path: Path) -> str:
    """A file the builder read as "" is either genuinely empty or unreadable (permissions, timeout)."""
    try:
        return "unreadable" if path.stat().st_size > 0 else "empty"
    except OSError:
        return "unreadable"


def _loaded_status(content: str, rendered_len: int, max_chars: int) -> str:
    """Same scan the builder runs (``_scan_context_content``): a hit replaces a project file with a BLOCKED
    marker, including SOUL.md."""
    if _pb._scan_for_threats(content.lstrip("\ufeff"), scope="context"):
        return "blocked"
    return "truncated" if rendered_len > max_chars else "loaded"


def list_context_file_sources(
    cwd: Optional[str] = None, context_length: Optional[int] = None, allow_install_tree_fallback: bool = False,
    home_override: "Path | None" = None, skip_soul: bool = False,
) -> List[Dict[str, Any]]:
    """One dict per context file Hermes considered, in the builder's priority order.

    Same signature semantics as ``build_context_files_prompt`` (``cwd=None`` → launch dir, install-tree guard
    unless *allow_install_tree_fallback*). Keys: ``label``, ``path``, ``chars``, ``est_tokens``, ``loaded``
    and ``status`` ∈ loaded / truncated / flagged / shadowed / blocked / empty / unreadable / suppressed.
    """
    cwd_path = Path(cwd if cwd is not None else os.getcwd()).resolve()
    max_chars = _pb._get_context_file_max_chars(context_length)
    suppressed = _pb._project_context_suppressed(cwd, cwd_path, allow_install_tree_fallback)
    sources: List[Dict[str, Any]] = []
    winner: Optional[str] = None
    for kind, label, path, content in _pb.discover_context_files(cwd_path):
        if not content:
            status = _empty_status(path)
        elif suppressed:
            status = "suppressed"
        elif winner in (None, kind):
            winner = kind
            # The builder caps the rendered ``## label`` section, not the raw file.
            status = _loaded_status(content, len(f"## {label}\n\n{content}"), max_chars)
        else:
            status = "shadowed"
        sources.append(_entry(label, path, content, status))

    if not skip_soul:
        home = Path(home_override) if home_override is not None else _pb.get_hermes_home()
        soul_path = home / "SOUL.md"
        if _pb._exists_or_denied(soul_path):
            content = _pb._read_context_file(soul_path)
            status = (_loaded_status(content, len(content), max_chars) if content
                      else _empty_status(soul_path))
            sources.append(_entry("SOUL.md", soul_path, content, status))
    return sources


def context_file_sources_for_agent(agent: Any) -> List[Dict[str, Any]]:
    """The manifest for a live agent, resolved exactly like ``agent.system_prompt._context_files_part``
    (session cwd, install-tree policy per platform, the agent's own profile home)."""
    if getattr(agent, "skip_context_files", False):
        return []
    from agent.runtime_cwd import resolve_context_cwd
    from agent.system_prompt import _agent_home
    launch_artifact = getattr(agent, "_context_cwd_is_launch_artifact", False)
    cwd = None if launch_artifact else resolve_context_cwd()
    ctx_len = getattr(getattr(agent, "context_compressor", None), "context_length", None)
    return list_context_file_sources(
        cwd=str(cwd) if cwd is not None else None, context_length=ctx_len if isinstance(ctx_len, int) else None,
        allow_install_tree_fallback=getattr(agent, "platform", None) in ("cli", "tui"), home_override=_agent_home(agent),
    )


def render_context_file_lines(sources: List[Dict[str, Any]]) -> List[str]:
    """Plain-text ``Context files`` block for ``/context``; [] when nothing was found."""
    if not sources:
        return []
    width = max(len(str(src.get("label") or "")) for src in sources)
    lines = ["Context files"]
    for src in sources:
        glyph, note = _STATUS_DISPLAY.get(str(src.get("status") or ""), ("•", ""))
        suffix = f"  ({note})" if note else ""
        lines.append(f"{glyph} {str(src.get('label') or ''):<{width}} ~{int(src.get('est_tokens') or 0):>9,} tokens{suffix}")
    return lines
