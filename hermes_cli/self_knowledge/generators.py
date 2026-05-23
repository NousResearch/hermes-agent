"""AUTO-block generators for the Hermes self-knowledge document."""

from __future__ import annotations

import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _md(value: Any) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split())
    return text.replace("|", "\\|") or "-"


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_none discovered_"
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_md(cell) for cell in row) + " |")
    return "\n".join(lines)


def _load_tool_entries():
    from tools.registry import discover_builtin_tools, registry

    discover_builtin_tools()
    return registry._snapshot_entries()


def generate_capabilities() -> str:
    """Generate a compact inventory from the runtime tool registry."""
    try:
        entries = sorted(_load_tool_entries(), key=lambda e: (e.toolset, e.name))
    except Exception as exc:
        return f"_unavailable: could not load tool registry ({type(exc).__name__})_"

    rows = [[entry.name, entry.toolset, entry.description] for entry in entries]
    return _table(["Tool", "Toolset", "Description"], rows)


def _load_toolsets():
    from toolsets import TOOLSETS, _HERMES_CORE_TOOLS

    return TOOLSETS, _HERMES_CORE_TOOLS


def generate_toolsets() -> str:
    """Generate toolset inventory from `toolsets.py`."""
    try:
        toolsets, core_tools = _load_toolsets()
    except Exception as exc:
        return f"_unavailable: could not load toolsets ({type(exc).__name__})_"

    rows = []
    for name, data in sorted(toolsets.items()):
        tools = data.get("tools", []) or []
        includes = data.get("includes", []) or []
        rows.append([name, data.get("description", ""), len(tools), ", ".join(includes) or "-"])
    rendered = _table(["Toolset", "Description", "Tools", "Includes"], rows)
    return f"Core default tools: {len(core_tools)}\n\n{rendered}"


def _load_commands():
    from hermes_cli.commands import COMMAND_REGISTRY

    return COMMAND_REGISTRY


def _command_scope(command: Any) -> str:
    if getattr(command, "cli_only", False):
        return "cli"
    if getattr(command, "gateway_only", False):
        return "gateway"
    return "cli+gateway"


def generate_slash_commands() -> str:
    """Generate slash command inventory from the shared command registry."""
    try:
        commands = sorted(_load_commands(), key=lambda c: (c.category, c.name))
    except Exception as exc:
        return f"_unavailable: could not load slash commands ({type(exc).__name__})_"

    rows = [[f"/{cmd.name}", cmd.category, _command_scope(cmd), cmd.description] for cmd in commands]
    return _table(["Command", "Category", "Scope", "Description"], rows)


def generate_gateway_platforms() -> str:
    """Generate platform adapter inventory by static file scan."""
    platforms_dir = PROJECT_ROOT / "gateway" / "platforms"
    if not platforms_dir.exists():
        return "_unavailable: gateway/platforms directory not found_"
    rows = []
    for path in sorted(platforms_dir.glob("*.py")):
        if path.name == "__init__.py" or path.stem.startswith("_"):
            continue
        rows.append([path.stem, str(path.relative_to(PROJECT_ROOT))])
    return _table(["Platform", "Adapter"], rows)


def generate_voice_loop() -> str:
    """Generate a compact voice/STT/TTS surface summary without secrets."""
    known_files = [
        "tools/transcription_tools.py",
        "tools/tts_tool.py",
        "tools/voice_mode.py",
        "gateway/platforms/discord.py",
        "gateway/run.py",
    ]
    rows = []
    for rel in known_files:
        path = PROJECT_ROOT / rel
        rows.append([rel, "present" if path.exists() else "missing"])
    return "Voice input routes platform audio through gateway callbacks, STT, agent response generation, optional TTS, and voice playback.\n\n" + _table(["Surface", "Status"], rows)


def _skill_frontmatter_name(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return path.parent.name
    if not text.startswith("---"):
        return path.parent.name
    end = text.find("\n---", 3)
    if end == -1:
        return path.parent.name
    for line in text[3:end].splitlines():
        if line.startswith("name:"):
            return line.split(":", 1)[1].strip().strip('"\'') or path.parent.name
    return path.parent.name


def generate_skills_profiles() -> str:
    """Generate skill counts and profile-skill inventory by static scan."""
    roots = [PROJECT_ROOT / "skills", Path.home() / ".hermes" / "skills"]
    skill_paths: list[Path] = []
    for root in roots:
        if root.exists():
            skill_paths.extend(root.glob("**/SKILL.md"))

    categories = Counter()
    profiles = []
    for path in skill_paths:
        name = _skill_frontmatter_name(path)
        parts = path.parent.parts
        category = "uncategorized"
        if "skills" in parts:
            idx = parts.index("skills")
            if idx + 1 < len(parts):
                category = parts[idx + 1]
        categories[category] += 1
        if name.startswith("profile-") or category == "hermes-agents":
            profiles.append(name)

    rows = [[category, count] for category, count in sorted(categories.items())]
    profile_line = ", ".join(sorted(set(profiles))) or "none discovered"
    return f"Total skills discovered: {len(skill_paths)}\n\nProfiles: {profile_line}\n\n" + _table(["Category", "Skills"], rows)


def generate_plugins_integrations() -> str:
    """Generate plugin/integration inventory from local plugin directories and env key names."""
    plugins_dir = PROJECT_ROOT / "plugins"
    rows = []
    if plugins_dir.exists():
        for path in sorted(p for p in plugins_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
            rows.append([path.name, str(path.relative_to(PROJECT_ROOT))])
    env_names = sorted(name for name in os.environ if name.endswith(("API_KEY", "TOKEN", "SECRET")))
    env_summary = f"Credential-like environment keys present: {len(env_names)} names redacted by design."
    return env_summary + "\n\n" + _table(["Plugin", "Path"], rows)


def generate_recent_activity() -> str:
    """Generate recent commit activity from local git history."""
    try:
        result = subprocess.run(
            ["git", "log", "--since=14 days ago", "--pretty=format:%h %ad %s", "--date=short"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        return f"_unavailable: git log failed ({type(exc).__name__})_"
    if result.returncode != 0:
        return f"_unavailable: git log failed (exit {result.returncode})_"
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return "_no commits in the last 14 days_"
    return "\n".join(f"- {line}" for line in lines[:40])
