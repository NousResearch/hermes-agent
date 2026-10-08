"""L1 Project Context Cache connecting Obsidian SSOT and local .planning/STATE.md."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional


def normalize_project_name(name: str) -> str:
    """Normalize project name for path lookup."""
    if not name:
        return "general"
    n = name.lower().strip()
    return re.sub(r"[_\s]+", "-", n)


def get_project_summary(
    project_name: str,
    session_cwd: Optional[str] = None,
    vault_dir: Optional[str] = None,
) -> Optional[str]:
    """Retrieve concise project summary from Obsidian SSOT or .planning/STATE.md."""
    norm = normalize_project_name(project_name)
    if norm in ("general", "unknown"):
        return None

    # 1. Check Obsidian Vault
    vdir = (
        Path(vault_dir)
        if vault_dir
        else Path(
            os.environ.get("HERMES_VAULT_DIR", Path.home() / "Documents" / "Wojciech")
        )
    )
    candidates = [
        vdir / "projects" / f"{norm}.md",
        vdir / "context" / "projects" / f"{norm}.md",
        vdir / f"{norm}.md",
    ]
    for cand in candidates:
        if cand.exists() and cand.is_file():
            try:
                text = cand.read_text(encoding="utf-8")
                # Extract first meaningful section or first 300 chars
                lines = [
                    line.strip()
                    for line in text.splitlines()
                    if line.strip() and not line.startswith("#")
                ]
                summary = " ".join(lines[:4])
                if len(summary) > 400:
                    summary = summary[:397] + "..."
                return f"[Obsidian SSOT: {cand.name}] {summary}"
            except Exception:
                pass

    # 2. Check local workspace .planning/STATE.md
    if session_cwd:
        cwd_p = Path(session_cwd)
        state_candidates = [
            cwd_p / ".planning" / "STATE.md",
            cwd_p / "STATE.md",
        ]
        for scand in state_candidates:
            if scand.exists() and scand.is_file():
                try:
                    text = scand.read_text(encoding="utf-8")
                    lines = [
                        line.strip()
                        for line in text.splitlines()
                        if line.strip() and not line.startswith("#")
                    ]
                    summary = " ".join(lines[:3])
                    if len(summary) > 300:
                        summary = summary[:297] + "..."
                    return f"[Workspace STATE.md] {summary}"
                except Exception:
                    pass

    return None
