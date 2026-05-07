"""SkillNamespaceInjector — maps CodeAct callable skills into the kernel.

Discovers skills that declare ``codeact_fn`` in their YAML frontmatter,
selects the most relevant ones for the current session, and generates
Python stub wrappers that are injected into the kernel namespace.

Selection priority:
  1. Explicitly loaded skills (``/skill load``) — always included
  2. Recently-used skills (from skill_usage tracking)
  3. All remaining skills with ``codeact_fn``, up to ``max_injected``

Collision handling:
  - Tool stubs take priority.  If a skill function name collides with a
    registered tool name, the skill is renamed ``skill_{original}`` and
    a warning is logged.
  - If two skill functions share a name, the more recently used one wins.
"""

from __future__ import annotations

import logging
import textwrap
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from agent.skill_utils import (
    get_all_skills_dirs,
    iter_skill_index_files,
    parse_frontmatter,
)

if TYPE_CHECKING:
    from tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Default limits (overridable via codeact.skills config).
DEFAULT_MAX_INJECTED = 20
DEFAULT_RECENTLY_USED_COUNT = 5


# ---------------------------------------------------------------------------
# Frontmatter scanning
# ---------------------------------------------------------------------------


def get_all_skill_frontmatters() -> Dict[str, Dict[str, Any]]:
    """Scan every SKILL.md across all skills directories and return
    ``{skill_name: frontmatter_dict}``."""
    results: Dict[str, Dict[str, Any]] = {}
    for base_dir in get_all_skills_dirs():
        if not base_dir.exists():
            continue
        for skill_md_path in iter_skill_index_files(base_dir, "SKILL.md"):
            try:
                content = skill_md_path.read_text(encoding="utf-8")
                frontmatter, _ = parse_frontmatter(content)
                name = frontmatter.get("name")
                if not name:
                    name = skill_md_path.parent.name
                results[name] = frontmatter
            except Exception as exc:
                logger.debug(
                    "Failed to parse frontmatter from %s: %s",
                    skill_md_path,
                    exc,
                )
    return results


def _get_codeact_fn(frontmatter: Dict[str, Any]) -> Optional[str]:
    """Extract the ``codeact_fn`` value from a skill's frontmatter.

    Returns the function name string if present and truthy, else None.
    Accepts both string and boolean-true (falls back to ``name``).
    """
    raw = frontmatter.get("codeact_fn")
    if raw is True:
        # ``codeact_fn: true`` — use the skill's own name as fn name.
        return frontmatter.get("name")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


# ---------------------------------------------------------------------------
# Recently-used skill lookup
# ---------------------------------------------------------------------------


def _load_recently_used_skills(count: int) -> List[str]:
    """Return up to *count* most recently used skill names.

    Best-effort: if ``skill_usage`` is unavailable or the sidecar is
    missing/corrupt, returns an empty list.
    """
    try:
        from tools.skill_usage import load_usage, latest_activity_at

        usage = load_usage()
        scored: list[Tuple[str, str]] = []  # (name, latest_iso)
        for name, rec in usage.items():
            ts = latest_activity_at(rec)
            if ts:
                scored.append((name, ts))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in scored[:count]]
    except Exception as exc:
        logger.debug("Could not load skill usage data: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Collision resolution
# ---------------------------------------------------------------------------


def _resolve_collisions(
    candidates: List[Tuple[str, str, Dict[str, Any]]],
    tool_names: Set[str],
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Deduplicate and rename to avoid collisions with tool stubs.

    Parameters
    ----------
    candidates:
        List of ``(skill_name, codeact_fn_name, frontmatter)`` tuples.
    tool_names:
        Set of all tool function names currently in the kernel namespace.

    Returns
    -------
    Deduplicated list with renamed functions where needed.  Each returned
    tuple is ``(skill_name, resolved_fn_name, frontmatter)``.
    """
    used_fn_names: Set[str] = set()
    result: list[Tuple[str, str, Dict[str, Any]]] = []

    for skill_name, fn_name, meta in candidates:
        resolved = fn_name

        # Collision with a tool stub → rename.
        if resolved in tool_names:
            resolved = f"skill_{resolved}"
            logger.info(
                "Skill '%s' function '%s' collides with tool — renamed to '%s'",
                skill_name,
                fn_name,
                resolved,
            )

        # Collision with an already-accepted skill → drop the duplicate
        # (the first one to arrive wins, which is the higher-priority one).
        if resolved in used_fn_names:
            logger.info(
                "Skill '%s' function '%s' duplicates an earlier skill — skipped",
                skill_name,
                resolved,
            )
            continue

        used_fn_names.add(resolved)
        result.append((skill_name, resolved, meta))

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class SkillNamespaceInjector:
    """Discovers, selects, and injects callable skills into the CodeAct namespace.

    Parameters
    ----------
    registry:
        The live Hermes tool registry (used for collision checks).
    max_skills:
        Maximum number of skill functions to inject.  Read from
        ``codeact.skills.max_injected`` config; defaults to 20.
    recently_used_count:
        How many recently-used skills to auto-include.  Read from
        ``codeact.skills.recently_used_count``; defaults to 5.
    """

    def __init__(
        self,
        registry: "ToolRegistry",
        max_skills: int = DEFAULT_MAX_INJECTED,
        recently_used_count: int = DEFAULT_RECENTLY_USED_COUNT,
    ) -> None:
        self._registry = registry
        self._max_skills = max_skills
        self._recently_used_count = recently_used_count

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select_skills(
        self,
        explicitly_loaded: Optional[Set[str]] = None,
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Return the ordered list of skills to inject.

        Each entry is ``(skill_name, codeact_fn_name, frontmatter)``.

        Selection priority:
          1. Skills in *explicitly_loaded* that have ``codeact_fn``
          2. Recently-used skills with ``codeact_fn``
          3. Remaining skills with ``codeact_fn``, up to ``max_skills``

        Duplicates are removed (first occurrence wins).
        """
        explicitly_loaded = explicitly_loaded or set()

        all_fms = get_all_skill_frontmatters()
        recent = _load_recently_used_skills(self._recently_used_count)

        # Build an ordered candidate list with priority tiers.
        seen: Set[str] = set()
        tier1: list[Tuple[str, str, Dict[str, Any]]] = []
        tier2: list[Tuple[str, str, Dict[str, Any]]] = []
        tier3: list[Tuple[str, str, Dict[str, Any]]] = []

        for name, fm in all_fms.items():
            fn_name = _get_codeact_fn(fm)
            if fn_name is None:
                continue
            if name in seen:
                continue
            seen.add(name)

            entry = (name, fn_name, fm)
            if name in explicitly_loaded:
                tier1.append(entry)
            elif name in recent:
                tier2.append(entry)
            else:
                tier3.append(entry)

        # Tier 2: maintain the recency order.
        tier2.sort(key=lambda e: recent.index(e[0]) if e[0] in recent else 999)

        # Combine tiers, truncate to max_skills.
        combined = tier1 + tier2 + tier3
        combined = combined[: self._max_skills]

        # Resolve collisions with tool names.
        tool_names = self._collect_tool_names()
        return _resolve_collisions(combined, tool_names)

    # ------------------------------------------------------------------
    # Source generation
    # ------------------------------------------------------------------

    def get_skill_stubs(
        self,
        explicitly_loaded: Optional[Set[str]] = None,
    ) -> str:
        """Return Python source defining stub functions for callable skills.

        Each stub calls ``_call_tool('__skill__', ...)`` to route through the
        existing IPC bridge back to the parent process's ``handle_function_call``.
        """
        selected = self.select_skills(explicitly_loaded=explicitly_loaded)

        lines: list[str] = []
        for _skill_name, fn_name, meta in selected:
            desc = meta.get("description", f"Executes the {_skill_name} skill.")
            # Ensure the description is safe for triple-quoted string embedding.
            clean_desc = desc.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
            clean_name = _skill_name.replace("\\", "\\\\").replace('"', '\\"')

            stub = textwrap.dedent(f'''\
                def {fn_name}(**kwargs):
                    """{clean_desc}"""
                    return _call_tool('__skill__', {{"skill_name": "{clean_name}", "args": kwargs}})
            ''')
            lines.append(stub)

        return "\n".join(lines)

    def get_skill_names(
        self,
        explicitly_loaded: Optional[Set[str]] = None,
    ) -> List[str]:
        """Return the resolved function names of all injected skills.

        Used by ``codeact_namespace.build_tool_namespace_source`` to add
        skill names to ``__protected__`` so they survive ``soft_reset``.
        """
        selected = self.select_skills(explicitly_loaded=explicitly_loaded)
        return [fn_name for _, fn_name, _ in selected]

    def get_skill_help_registry(
        self,
        explicitly_loaded: Optional[Set[str]] = None,
    ) -> Dict[str, Tuple[str, str]]:
        """Return a help-registry dict for callable skills.

        Maps ``{fn_name: (compact_line, full_doc)}`` — same shape as the
        tool help registry, so ``help()`` and ``help('skill_name')`` work
        uniformly for both tools and skills.
        """
        selected = self.select_skills(explicitly_loaded=explicitly_loaded)

        registry: Dict[str, Tuple[str, str]] = {}
        for skill_name, fn_name, meta in selected:
            desc = meta.get("description", f"Callable skill: {skill_name}")
            compact = f"  {fn_name}(**kwargs) — {desc[:80]}"
            full_doc = f"Skill: {skill_name}\nType: callable skill (CodeAct)\n\n{desc}"
            registry[fn_name] = (compact, full_doc)

        return registry

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _collect_tool_names(self) -> Set[str]:
        """Return the set of all tool function names in the kernel namespace."""
        names: Set[str] = set()
        try:
            for entry in self._registry._snapshot_entries():
                if entry.schema is not None:
                    names.add(entry.name)
        except Exception:
            # Registry may not have _snapshot_entries in all configurations.
            pass
        # Always include the built-in kernel functions.
        names |= {"help", "promote_to_skill", "_call_tool", "__protected__"}
        return names
