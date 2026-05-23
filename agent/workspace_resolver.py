"""Workspace resolver for Hermes channels/topics.

Provides stat-cached discovery of per-topic and per-channel prompts + skills
from a folder hierarchy under ~/.hermes/workspaces/ and ~/.hermes/platforms/.

This module is import-safe: no heavy dependencies, no tool registry.

Folder layout::

    ~/.hermes/
    ├── workspaces/
    │   ├── news-feed/
    │   │   ├── SYSTEM.md          # YAML frontmatter + body
    │   │   └── skills/            # optional workspace-local skills
    │   │       └── some-skill/
    │   │           └── SKILL.md
    │   └── code-review/
    │       ├── SYSTEM.md
    │       └── skills/
    └── platforms/
        └── telegram/
            └── -1003682109119/     # chat_id as folder name
                └── topics.yaml      # thread_id → workspace_name

SYSTEM.md frontmatter format::

    ---
    skills:
      - telegram-summary-bot
      - conventional-commits
    ---
    Respond in Hebrew. Focus on regional news...

Resolution (per platform message)::

  1. Discover platform mapping  →  workspace name
  2. Resolve workspace SYSTEM.md →  prompt + skill list
  3. Inject into MessageEvent   →  channel_prompt + auto_skill

Fallback chain (most-specific wins)::

  topic workspace prompt  >  channel workspace prompt  >  channel_prompts dict
  topic workspace skills  >  channel workspace skills  >  existing auto_skill

All disk reads are stat-cached (1-second TTL) so changes take effect
automatically --- no gateway restart required.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple, Any, Iterable

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# ── File cache (path → (mtime_ns, content)) ──────────────────────────────────
_STAT_CACHE: Dict[Tuple[str, int], Tuple[int, str]] = {}
_CACHE_TTL_SECS = 1.0

def _stat_cached(path: Path) -> Optional[Tuple[int, str]]:
    """Read *path* returning (mtime_ns, content).  Returns None if missing.

    Uses a per-second TTL keyed on absolute path.  This means edits on disk
    are picked up within one second --- cheap enough to call on every
    incoming message without restart.
    """
    abs_str = str(path.resolve())
    now = time.monotonic()
    key = (abs_str, int(now // _CACHE_TTL_SECS))
    cached = _STAT_CACHE.get(key)
    if cached is not None:
        return cached
    if not path.exists():
        _STAT_CACHE[key] = None  # type: ignore[assignment]
        return None
    try:
        stat = path.stat()
        content = path.read_text(encoding="utf-8")
        result = (stat.st_mtime_ns, content)
        _STAT_CACHE[key] = result  # type: ignore[assignment]
        return result
    except (OSError, UnicodeDecodeError):
        _STAT_CACHE[key] = None  # type: ignore[assignment]
        return None


def _clear_stat_cache() -> None:
    """Test hook."""
    _STAT_CACHE.clear()


# ── Frontmatter parsing (minimal, dependency-light) ──────────────────────────

def _parse_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
    """Extract YAML frontmatter and body from SYSTEM.md-style content.

    Supports the ---\n...\n--- pattern.  If no frontmatter, returns ({}, text).
    """
    lines = text.splitlines(keepends=False)
    if not lines or lines[0].strip() != "---":
        return {}, text
    try:
        end = lines.index("---", 1)
    except ValueError:
        return {}, text
    fm_text = "\n".join(lines[1:end])
    body = "\n".join(lines[end + 1 :]).strip()
    try:
        import yaml
        fm = yaml.safe_load(fm_text) or {}
        if not isinstance(fm, dict):
            fm = {}
    except Exception:
        fm = {}
    return fm, body


# ── Named result type ────────────────────────────────────────────────────────

class WorkspaceResult(NamedTuple):
    """Resolved workspace metadata for a single message."""

    prompt: str | None
    skills: List[str] | None
    model: str | None


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════


def resolve_workspace(
    platform: str,
    chat_id: str,
    thread_id: str | None = None,
) -> WorkspaceResult:
    """Resolve the workspace prompt + skills for a given (platform, chat, topic).

    Args:
        platform: Platform slug  (telegram, discord, slack, …)
        chat_id:  Numeric channel / group / chat id
        thread_id: Optional topic / thread id (None for channel-level)

    Returns:
        WorkspaceResult(prompt=str|None, skills=list|None)
    """
    home = get_hermes_home()
    workspace_name = _resolve_workspace_name(home, platform, chat_id, thread_id)
    if not workspace_name:
        return WorkspaceResult(None, None, None)
    return _resolve_workspace_content(home, workspace_name)


def get_workspace_skill_dirs(
    platform: str,
    chat_id: str,
    thread_id: str | None = None,
) -> List[Path]:
    """Return workspace-local skill directories for the resolved topic.

    These paths are meant to be appended to ``get_all_skills_dirs()`` so
    workspace-specific skills are discoverable without global installation.
    """
    home = get_hermes_home()
    workspace_name = _resolve_workspace_name(home, platform, chat_id, thread_id)
    if not workspace_name:
        return []
    ws_dir = home / "workspaces" / workspace_name
    skills_dir = ws_dir / "skills"
    if skills_dir.exists() and skills_dir.is_dir():
        return [skills_dir]
    return []


# ═══════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════════════════


def _resolve_workspace_name(
    home: Path,
    platform: str,
    chat_id: str,
    thread_id: str | None = None,
) -> str | None:
    """Read platforms/<platform>/<chat_id>/topics.yaml, resolve workspace name."""
    mapping_file = home / "platforms" / platform / _safe_dir_name(chat_id) / "topics.yaml"
    stat_result = _stat_cached(mapping_file)
    if stat_result is None:
        return None
    _, text = stat_result
    try:
        import yaml
        data = yaml.safe_load(text) or {}
    except Exception:
        logger.debug("[workspace] Failed to parse %s", mapping_file)
        return None

    topics = data.get("topics", {})
    if isinstance(topics, dict):
        # Direct dict form: {"7695": "news-feed"}
        if thread_id is not None and thread_id in topics:
            return topics[thread_id]
    elif isinstance(topics, list):
        # List-of-dicts form (allows YAML comments / ordering)
        for entry in topics:
            if isinstance(entry, dict) and str(entry.get("thread_id", "")) == str(thread_id or ""):
                return entry.get("workspace")

    # Fall back to channel-level workspace
    channel_ws = data.get("workspace") if isinstance(data, dict) else None
    return channel_ws


def _resolve_workspace_content(
    home: Path,
    workspace_name: str,
) -> WorkspaceResult:
    """Read workspaces/<name>/SYSTEM.md, extract prompt + skills + model."""
    system_file = home / "workspaces" / _safe_dir_name(workspace_name) / "SYSTEM.md"
    stat_result = _stat_cached(system_file)
    if stat_result is None:
        return WorkspaceResult(None, None, None)
    _, text = stat_result
    fm, body = _parse_frontmatter(text)
    prompt = body.strip() or None
    skills = fm.get("skills")
    if isinstance(skills, str):
        skills = [skills.strip()] if skills.strip() else None
    elif isinstance(skills, list):
        parsed: List[str] = []
        for s in skills:
            if isinstance(s, str) and s.strip():
                parsed.append(s.strip())
        skills = parsed if parsed else None
    else:
        skills = None
    model = fm.get("model")
    if not isinstance(model, str) or not model.strip():
        model = None
    return WorkspaceResult(prompt, skills, model)


def _safe_dir_name(value: str) -> str:
    """Escape a chat id or workspace name for use as a directory name.

    Leading minus becomes escaped (e.g. "-1003" → "_-1003") so ``Path``
    doesn't interpret it as a relative-segment trick.
    """
    if value.startswith("-"):
        return "_-" + value[1:]
    return value


def _write_system_md(path: Path, frontmatter: Dict[str, Any], body: str) -> None:
    """Write a SYSTEM.md file with YAML frontmatter + body.

    Preserves frontmatter fields and writes them in a deterministic order.
    If frontmatter is empty, writes body only (no frontmatter fence).
    """
    import yaml
    if frontmatter:
        fm_text = yaml.safe_dump(frontmatter, default_flow_style=False).strip()
        content = f"---\n{fm_text}\n---\n{body}"
    else:
        content = body
    path.write_text(content, encoding="utf-8")
