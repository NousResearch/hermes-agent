"""Slim prompt summary for Hermes self-knowledge."""

from __future__ import annotations

import os
import re
from pathlib import Path

from hermes_cli.self_knowledge.drift import DOC_PATH
from hermes_cli.self_knowledge.parser import parse_auto_blocks


SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def _section(text: str, heading: str) -> str:
    matches = list(SECTION_RE.finditer(text))
    for idx, match in enumerate(matches):
        if match.group(1).strip().lower() == heading.lower():
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            return text[start:end].strip()
    return ""


def _capability_names(text: str, *, max_capabilities: int) -> list[str]:
    blocks = parse_auto_blocks(text)
    block = blocks.get("capabilities")
    if not block:
        return []
    names: list[str] = []
    for line in block.body.splitlines():
        line = line.strip()
        if not line.startswith("|") or "---" in line or "Tool |" in line:
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if parts and parts[0] and parts[0] != "-":
            names.append(parts[0].replace("\\|", "|"))
        if len(names) >= max_capabilities:
            break
    return names


def build_slim_summary(doc_path: Path = DOC_PATH, *, max_capabilities: int = 60) -> str:
    """Build a compact self-knowledge block for stable prompt injection.

    Controlled by HERMES_SELF_KNOWLEDGE_PROMPT:
    - off/false/0/no: disabled
    - slim/true/1/yes or unset: slim summary
    """
    mode = os.getenv("HERMES_SELF_KNOWLEDGE_PROMPT", "slim").strip().lower()
    if mode in {"off", "false", "0", "no", "disabled"}:
        return ""
    path = Path(doc_path)
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""

    identity = _section(text, "Identity")
    principles = _section(text, "Core Principles")
    capability_names = _capability_names(text, max_capabilities=max_capabilities)

    parts = ["Hermes self-knowledge (repo-grounded slim summary):"]
    if identity:
        parts.append("Identity:\n" + identity)
    if principles:
        parts.append("Core principles:\n" + principles)
    if capability_names:
        parts.append("Capabilities: " + ", ".join(capability_names))
    return "\n\n".join(parts).strip()
