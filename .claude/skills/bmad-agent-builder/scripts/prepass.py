#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["tiktoken"]
# ///
"""prepass — the Analyze pre-pass for the agent builder.

Reads an agent skill directory and emits one compact JSON object that every
lens and the analyze orchestrator consume. The pre-pass does the one thing the
lenses should not each redo: it classifies the agent along the three-point
gradient (stateless, memory, autonomous), counts tokens for SKILL.md and every
in-tree file, and sets the gate that turns the conditional sanctum lens on.

Detection rests on the sanctum, the built agent's runtime memory at
`{project-root}/_bmad/memory/{skillName}/`. An agent that reloads a sanctum on
waking is a memory agent; one that also carries live wake behavior (a PULSE
file or a pulse/autonomous wake reference with named-task routing) is
autonomous; one with no sanctum at all is stateless. This is the BUILT agent's
memory, never the builder's process log (.memlog.md), and the two are kept
apart here.

Lengths come from tokens, never line counts. The count uses count_tokens.py
(imported as a sibling, then shelled out, then a chars // 4 fallback) so the
metric matches the rest of the builder and runs under a bare python3.

Output contract (one line of JSON on stdout, the pinned prepass shape):
  {
    "agent_type": "stateless" | "memory" | "autonomous",
    "is_memory_agent": bool,           # true for memory and autonomous
    "skill_md_tokens": int,
    "files": [{"path": str, "tokens": int}, ...]
  }

Read-only over the target agent directory. It opens files to count and classify
and writes nothing inside the agent tree.

Usage:
  prepass.py <agent-dir>     classify and count the agent at this directory
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# Directories we never descend into while counting agent files.
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "node_modules", ".venv", "venv"}

# Extensions we treat as countable text. Binary or opaque assets are skipped.
TEXT_SUFFIXES = {
    ".md", ".py", ".toml", ".yaml", ".yml", ".json", ".txt",
    ".csv", ".html", ".sh", ".cfg", ".ini",
}


# --- token counting ---------------------------------------------------------

def _count_via_import(text: str):
    """Count tokens by importing the sibling count_tokens module."""
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    try:
        import count_tokens  # type: ignore
    except Exception:
        return None
    try:
        tokens, _method = count_tokens.count_tokens(text)
        return int(tokens)
    except Exception:
        return None


def _count_via_shell(text: str):
    """Count tokens by shelling out to count_tokens.py with text on stdin."""
    script = SCRIPT_DIR / "count_tokens.py"
    if not script.exists():
        return None
    try:
        proc = subprocess.run(
            [sys.executable, str(script), "--stdin"],
            input=text,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    try:
        return int(json.loads(proc.stdout)["tokens"])
    except Exception:
        return None


def count_tokens(text: str) -> int:
    """Token length of text via count_tokens.py, falling back to chars // 4.

    Prefers importing the vendored count_tokens module, then shelling out to it,
    then a bare character estimate so the pre-pass always produces a number.
    """
    for counter in (_count_via_import, _count_via_shell):
        result = counter(text)
        if result is not None:
            return result
    return len(text) // 4


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


# --- agent classification ---------------------------------------------------

def iter_files(root: Path):
    """Yield countable text files under root, skipping noise directories."""
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if path.suffix.lower() in TEXT_SUFFIXES:
            yield path


def has_sanctum(root: Path, skill_text: str) -> bool:
    """True when the agent reloads a runtime sanctum on waking (a memory agent).

    The sanctum is the built agent's memory at `_bmad/memory/{skillName}/`. We
    treat any of these as a sanctum signal: the SKILL referencing that memory
    path, the Sacred-Truth / waking bootloader language, a wake or init-sanctum
    scaffolder, or the sanctum template assets (PERSONA / CREED / BOND / MEMORY
    / INDEX / CAPABILITIES). This is the built agent's memory, distinct from the
    builder's .memlog.md, which is never a sanctum signal.
    """
    if re.search(r"_bmad/memory/", skill_text):
        return True
    if re.search(r"\bsanctum\b", skill_text, re.IGNORECASE):
        return True
    if "Sacred Truth" in skill_text and re.search(r"\b(waking|wake)\b", skill_text, re.IGNORECASE):
        return True

    for pattern in ("scripts/wake*", "scripts/init-sanctum*"):
        for script in root.glob(pattern):
            if script.is_file():
                return True

    sanctum_seed = re.compile(
        r"^(PERSONA|CREED|BOND|MEMORY|INDEX|CAPABILITIES)-template\.md$"
    )
    assets = root / "assets"
    if assets.is_dir():
        for asset in assets.iterdir():
            if asset.is_file() and sanctum_seed.match(asset.name):
                return True
    return False


def has_autonomous_wake(root: Path, skill_text: str) -> bool:
    """True when a memory agent also carries live autonomous wake behavior.

    Autonomous is memory plus a PULSE-driven wake: a deployed PULSE.md, a
    pulse/autonomous-wake reference, or SKILL wake routing (named-task pulse
    routing, a default wake behavior, quiet hours, or a wake frequency).

    The standard memory bootloader already names a Pulse Mode (`--pulse`) path
    that loads PULSE.md, and ships a PULSE template asset, in every memory
    agent. Those are seeds, not live wake behavior, so neither the bootloader's
    Pulse-Mode line nor a PULSE template asset counts here. The wake behavior
    must be deployed: a real PULSE.md, a wake reference file, or SKILL routing
    that names tasks or schedules a recurring wake.
    """
    if (root / "PULSE.md").is_file():
        return True

    refs = root / "references"
    if refs.is_dir():
        for ref in refs.iterdir():
            name = ref.name.lower()
            if ref.is_file() and ("pulse-wake" in name or "autonomous-wake" in name):
                return True

    wake_signals = [
        r"--pulse:\{",                    # named-task pulse routing
        r"-p:\{",                          # short-flag named-task routing
        r"default pulse wake",
        r"default wake behavior",
        r"\bquiet hours\b",
        r"wake frequency",
        r"autonomous wake",
    ]
    for pattern in wake_signals:
        if re.search(pattern, skill_text, re.IGNORECASE):
            return True
    return False


def classify(root: Path, skill_text: str) -> str:
    """Return the agent_type along the gradient."""
    if not has_sanctum(root, skill_text):
        return "stateless"
    if has_autonomous_wake(root, skill_text):
        return "autonomous"
    return "memory"


# --- main -------------------------------------------------------------------

def build_payload(root: Path) -> dict:
    skill_path = root / "SKILL.md"
    skill_text = read_text(skill_path) if skill_path.is_file() else ""

    agent_type = classify(root, skill_text)
    is_memory_agent = agent_type in ("memory", "autonomous")

    files = []
    skill_md_tokens = 0
    for path in iter_files(root):
        tokens = count_tokens(read_text(path))
        rel = path.relative_to(root).as_posix()
        files.append({"path": rel, "tokens": tokens})
        if path == skill_path:
            skill_md_tokens = tokens

    return {
        "agent_type": agent_type,
        "is_memory_agent": is_memory_agent,
        "skill_md_tokens": skill_md_tokens,
        "files": files,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("agent_dir", help="path to the agent skill directory to analyze")
    args = p.parse_args(argv)

    root = Path(args.agent_dir).expanduser().resolve()
    if not root.is_dir():
        p.error(f"not a directory: {root}")

    print(json.dumps(build_payload(root)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
