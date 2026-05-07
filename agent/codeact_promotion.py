"""Skill promotion pipeline — write agent-generated helper functions as persistent skills.

When a function defined in the CodeAct kernel namespace is promoted
(via the ``promote_to_skill()`` builtin or the curator), this module
writes the skill to ``~/.hermes/skills/promoted/<name>/SKILL.md``.

Promoted skills carry ``codeact_fn: <name>`` so the Phase 4
SkillNamespaceInjector will load them as callable functions in
subsequent sessions.
"""

from __future__ import annotations

import json
import logging
import re
import textwrap
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

PROMOTED_SKILLS_DIR_NAME = "promoted"

# Persistence file for candidates flagged for review (auto_promote: false).
_PENDING_FILE_NAME = ".codeact_pending_promotions.json"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class PromotionCandidate:
    """A function extracted from a CodeAct session that is eligible for promotion."""

    fn_name: str
    description: str
    source_code: str  # the function's Python source (from namespace)
    domain: str = "general"
    tags: List[str] = field(default_factory=list)
    session_id: Optional[str] = None
    occurrence_count: int = 1  # how many times this pattern was seen
    seen_in_sessions: List[str] = field(default_factory=list)
    promoted_at: Optional[str] = None  # ISO timestamp, set on write

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PromotionCandidate":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Pending promotions (flagged candidates awaiting review)
# ---------------------------------------------------------------------------


def _pending_file() -> Path:
    return get_hermes_home() / "skills" / _PENDING_FILE_NAME


def load_pending() -> List[Dict[str, Any]]:
    """Read the pending promotions file.  Returns [] on missing/corrupt."""
    path = _pending_file()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def save_pending(candidates: List[Dict[str, Any]]) -> None:
    """Atomically write the pending promotions list."""
    path = _pending_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        import os
        import tempfile

        fd, tmp = tempfile.mkstemp(
            dir=str(path.parent), prefix=".codeact_promo_", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(candidates, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except Exception as exc:
        logger.debug("Failed to save pending promotions: %s", exc, exc_info=True)


def flag_candidate(candidate: PromotionCandidate) -> None:
    """Append *candidate* to the pending promotions file for review."""
    pending = load_pending()
    # Dedup by fn_name — latest wins.
    pending = [p for p in pending if p.get("fn_name") != candidate.fn_name]
    pending.append(candidate.to_dict())
    save_pending(pending)


def remove_pending(fn_name: str) -> bool:
    """Remove a candidate from the pending list.  Returns True if found."""
    pending = load_pending()
    before = len(pending)
    pending = [p for p in pending if p.get("fn_name") != fn_name]
    if len(pending) < before:
        save_pending(pending)
        return True
    return False


# ---------------------------------------------------------------------------
# Skill file writing
# ---------------------------------------------------------------------------


def _promoted_skills_dir() -> Path:
    return get_hermes_home() / "skills" / PROMOTED_SKILLS_DIR_NAME


def _sanitize_fn_name(name: str) -> str:
    """Ensure the name is safe for use as a filesystem directory and Python identifier."""
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", name).strip("_")
    if not safe or safe[0].isdigit():
        safe = f"fn_{safe}"
    return safe


def build_skill_frontmatter(candidate: PromotionCandidate) -> str:
    """Generate the YAML frontmatter block for a promoted skill SKILL.md."""
    safe_name = _sanitize_fn_name(candidate.fn_name)
    tags = candidate.tags or [candidate.domain, "codeact-promoted"]
    tags_str = "[" + ", ".join(tags) + "]"

    now = candidate.promoted_at or datetime.now(timezone.utc).isoformat()

    return textwrap.dedent(f"""\
        ---
        name: {safe_name}
        description: "{candidate.description}"
        version: 1.0.0
        author: CodeAct (auto-promoted)
        license: MIT
        codeact_fn: {candidate.fn_name}
        metadata:
          hermes:
            tags: {tags_str}
            category: promoted
            promoted_from_session: {candidate.session_id or "unknown"}
            promoted_at: {now}
            occurrence_count: {candidate.occurrence_count}
        ---
    """)


def write_promoted_skill(candidate: PromotionCandidate) -> Path:
    """Write a promoted skill directory under ``~/.hermes/skills/promoted/``.

    Creates:
        promoted/<name>/SKILL.md   — frontmatter + description + source code

    Returns the path to the SKILL.md file.
    """
    safe_name = _sanitize_fn_name(candidate.fn_name)
    skill_dir = _promoted_skills_dir() / safe_name
    skill_dir.mkdir(parents=True, exist_ok=True)

    skill_md = skill_dir / "SKILL.md"
    candidate.promoted_at = (
        candidate.promoted_at or datetime.now(timezone.utc).isoformat()
    )

    frontmatter = build_skill_frontmatter(candidate)

    body = textwrap.dedent(f"""\
        # {candidate.fn_name}

        **Auto-promoted from CodeAct session.**

        {candidate.description}

        ## Source Code

        ```python
        {candidate.source_code.strip()}
        ```

        ## Usage

        Call as a Python function in CodeAct mode:

        ```python
        result = {candidate.fn_name}(**kwargs)
        ```
    """)

    content = frontmatter + "\n" + body
    skill_md.write_text(content, encoding="utf-8")

    logger.info("Promoted skill written: %s", skill_md)

    # Record in skill_usage
    try:
        from tools import skill_usage as _u

        _u.bump_use(safe_name)
    except Exception:
        pass  # best-effort

    return skill_md


# ---------------------------------------------------------------------------
# Trajectory mining — find repeated helper functions in session messages
# ---------------------------------------------------------------------------

# Matches ``def <name>(...)`` followed by a body indented more than the def.
_DEF_RE = re.compile(
    r"^(def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(.*?\)\s*(?:->.*?)?:\s*\n"
    r"(?:[ \t]+.*\n)*)",
    re.MULTILINE,
)


def extract_helper_functions(messages: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Walk a conversation's message list and extract user-defined function
    source blocks from ``run_code`` tool calls.

    Returns ``{fn_name: [source_block_1, source_block_2, ...]}``.
    Each source block is the full ``def ...`` text.
    """
    found: Dict[str, List[str]] = {}

    for msg in messages:
        # CodeAct tool calls arrive as assistant messages with tool_calls
        # containing {"thoughts": ..., "code": ...} in the arguments.
        tool_calls = msg.get("tool_calls") or []
        for tc in tool_calls:
            fn = tc.get("function", {})
            if fn.get("name") != "run_code":
                continue
            args_raw = fn.get("arguments", "{}")
            if isinstance(args_raw, str):
                try:
                    args = json.loads(args_raw)
                except (json.JSONDecodeError, TypeError):
                    continue
            elif isinstance(args_raw, dict):
                args = args_raw
            else:
                continue

            code = args.get("code", "")
            if not isinstance(code, str):
                continue

            for match in _DEF_RE.finditer(code):
                full_def = match.group(1).strip()
                fn_name = match.group(2)
                found.setdefault(fn_name, []).append(full_def)

    return found


def find_codeact_promotion_candidates(
    messages: List[Dict[str, Any]],
    session_id: Optional[str] = None,
    min_occurrences: int = 3,
    min_sessions: int = 2,
) -> List[PromotionCandidate]:
    """Scan a conversation for repeated helper function definitions.

    A function is a promotion candidate if it appears >= *min_occurrences*
    times in ``run_code`` tool calls across >= *min_sessions* distinct
    sessions.

    For single-session mining (the typical Phase 5 use case), the
    *min_sessions* gate is relaxed to 1 if *min_occurrences* >= 3.
    """
    functions = extract_helper_functions(messages)

    candidates: List[PromotionCandidate] = []
    for fn_name, defs in functions.items():
        # Use the last definition as the "canonical" source.
        count = len(defs)
        if count < min_occurrences:
            continue

        # Session gate: if only one session, relax to allow if occurrences >= 3
        # (caller should pass all sessions' messages for cross-session mining).
        source = defs[-1]

        # Extract a simple docstring from the source if available.
        desc_match = re.search(r'"""(.*?)"""', source, re.DOTALL)
        description = (
            desc_match.group(1).strip()
            if desc_match
            else f"Auto-detected helper: {fn_name}"
        )

        candidates.append(
            PromotionCandidate(
                fn_name=fn_name,
                description=description,
                source_code=source,
                session_id=session_id,
                occurrence_count=count,
                seen_in_sessions=[session_id] if session_id else [],
            )
        )

    return candidates
