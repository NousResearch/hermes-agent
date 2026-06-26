"""
agent.prompt_builder
===================
Builds the identity, skills, context, memory, and platform layers of the
system prompt. Runs at AIAgent construction time and is cached on
``agent._cached_system_prompt`` for the lifetime of the session (Hermes
never rebuilds or reinjects parts of it mid-conversation — that is what
keeps upstream prompt caches warm across turns).
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Cap defaults — see ``_get_context_file_max_chars`` for the dynamic
# version that scales with the model's context window.
CONTEXT_FILE_MAX_CHARS = 20_000

# Hard ceiling so a "dynamic" misconfiguration can never blow the prompt.
# Set this well above the largest legitimate context file you'd inject
# (most project rule files are <2k chars; a 200k ceiling is generous
# while still bounding prompt size).
_CONTEXT_FILE_DYNAMIC_CEILING = 200_000


def _scan_context_content(content: str, label: str) -> str:
    """Trim obviously-harmful content from a project context file.

    Hermes injects the raw contents of AGENTS.md / .cursorrules / SOUL.md
    into the system prompt. The user owns those files, but a careless
    file can still degrade the agent: trailing newlines, control bytes,
    and excessively long single-line "log dumps" all waste tokens without
    changing behavior. This helper applies a small set of well-defined
    normalizations so the file's *intent* is preserved but noisy
    artifacts are stripped. Used by ``build_context_files_prompt`` and
    ``load_soul_md`` before injection.
    """
    if not content:
        return content
    # Strip trailing whitespace on every line (preserves indentation).
    content = "\n".join(line.rstrip() for line in content.splitlines())
    # Collapse runs of 3+ blank lines into a single blank line.
    content = re.sub(r"\n{3,}", "\n\n", content)
    # Drop ASCII control bytes except \t \n \r — these almost never appear
    # in legitimate rule files and are usually paste-from-PDF artifacts.
    content = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", content)
    return content


def _truncate_content(
    content: str,
    label: str,
    *,
    context_length: Optional[int] = None,
    read_path: str = "",
) -> str:
    """Cap a single context-file's contribution to the system prompt.

    Returns ``content`` unchanged if it already fits under the active cap.
    When truncation kicks in, prepends a single-line marker so the agent
    can tell something was cut. The cap is computed from
    ``_get_context_file_max_chars`` (see that function for the override
    hierarchy: explicit config → dynamic-from-context → 20k default).
    """
    if not content:
        return content
    cap = _get_context_file_max_chars(context_length)
    if len(content) <= cap:
        return content
    head = content[:cap]
    marker = (
        f"\n\n[…truncated at {cap} chars; original was {len(content)} chars. "
        f"Read the full file at {read_path or label} if you need the rest.]"
    )
    # Drop the cut fragment before prepending the marker so we don't
    # exceed the cap on the returned string.
    return head + marker


def load_soul_md(
    context_length: Optional[int] = None,
    profile: Optional[str] = None,
) -> Optional[str]:
    """Load SOUL.md for the given profile (or the default profile).

    Used as the agent identity (slot #1 in the system prompt).  When this
    returns content, ``build_context_files_prompt`` should be called with
    ``skip_soul=True`` so SOUL.md isn't injected twice.

    Resolution order:
      1. ``profile.yaml.soul_path`` (relative to the profile dir) — lets
         a profile rename the file away from the conventional ``SOUL.md``
         (e.g. ``soul_path: CEO.soul.md`` for the default profile).
      2. ``<HERMES_HOME>/profiles/<profile>/SOUL.md`` — the conventional
         per-profile location used by every existing profile.
      3. ``<HERMES_HOME>/SOUL.md`` — the legacy default-profile location
         when no profile was given or the per-profile file is missing.

    Returns None if no SOUL file is found.
    """
    try:
        from hermes_cli.config import ensure_hermes_home
        ensure_hermes_home()
    except Exception as e:
        logger.debug("Could not ensure HERMES_HOME before loading SOUL.md: %s", e)

    from hermes_constants import get_hermes_home
    soul_path: Optional[Path] = None
    label = "SOUL.md"
    if profile:
        # Per-profile lookup. HERMES_HOME may be the root or already a
        # profile dir (gateway sets it per profile when spawning agents),
        # so we anchor from the root to keep behavior consistent.
        hermes_root = get_hermes_home()
        # If HERMES_HOME itself is a profile dir, walk up to the root so
        # `hermes_root / "profiles" / profile` still resolves correctly.
        if hermes_root.name and (hermes_root / "SOUL.md").exists() is False and (
            hermes_root.parent / "pairing"
        ).is_dir() is False and (hermes_root / "profiles").is_dir() is False:
            # Heuristic: profile-shaped homes end in `profiles/<name>`. If
            # we look one level up and the per-profile ``pairing`` exists
            # there, treat the parent as the root. Keeps load_soul_md
            # correct whether the gateway sets HERMES_HOME=root or root/profile.
            anchor = hermes_root
            if (anchor / "pairing").is_dir() and (
                anchor / "profiles"
            ).is_dir() is False:
                anchor = anchor.parent
            hermes_root = anchor
        profile_dir = hermes_root / "profiles" / profile
        # 1) honor a custom soul_path from profile.yaml
        profile_yaml = profile_dir / "profile.yaml"
        if profile_yaml.exists():
            try:
                import yaml
                with profile_yaml.open(encoding="utf-8") as f:
                    meta = yaml.safe_load(f) or {}
                custom = meta.get("soul_path")
                if custom and isinstance(custom, str) and custom.strip():
                    candidate = profile_dir / custom
                    if candidate.exists():
                        soul_path = candidate
                        label = custom
            except Exception as e:
                logger.debug("Could not read profile.yaml soul_path: %s", e)
        # 2) conventional per-profile SOUL.md
        if soul_path is None:
            candidate = profile_dir / "SOUL.md"
            if candidate.exists():
                soul_path = candidate

    if soul_path is None:
        # 3) legacy global default
        soul_path = get_hermes_home() / "SOUL.md"

    if not soul_path.exists():
        return None
    try:
        content = soul_path.read_text(encoding="utf-8").strip()
        if not content:
            return None
        content = _scan_context_content(content, label)
        content = _truncate_content(
            content, label, context_length=context_length,
            read_path=str(soul_path),
        )
        return content
    except Exception as e:
        logger.debug("Could not read SOUL.md from %s: %s", soul_path, e)
        return None


def _load_hermes_md(cwd_path: Path, context_length: Optional[int] = None) -> str:
