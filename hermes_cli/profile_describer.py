"""Profile describer — auto-generate ``description`` for a profile.

Mirrors ``hermes_cli/kanban_specify.py``: lazy aux client import, lenient response parse,
never raises on expected failure modes. Reads at most ``MAX_SKILLS_FOR_PROMPT`` skill
names to keep the prompt bounded.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from hermes_cli import profiles as profiles_mod
from agent.skill_utils import is_excluded_skill_path

logger = logging.getLogger(__name__)

# Cap on skill names fed to the LLM (200+ skill profiles would blow context).
MAX_SKILLS_FOR_PROMPT = 60


_SYSTEM_PROMPT = """You are a profile-describer for the Hermes Agent kanban board.

A user runs multiple "profiles" — distinct agent identities, each with their
own skills, model, and configuration. The kanban board's orchestrator routes
work to whichever profile best fits each task. To do that well, every
profile needs a short, concrete description of what it's good at.

You are given a profile's:
  - Name
  - Model / provider
  - List of installed skill names (a strong signal of role / domain)

Produce a single JSON object with exactly one key:

  {
    "description": "<1-2 sentence description, plain prose, no preamble>"
  }

Rules:
  - The description is what an orchestrator will read to decide whether to
    route a task here. Lead with the profile's strongest capability.
  - Stay concrete. Bad: "an AI agent that helps users."
                  Good: "Reads and modifies Python codebases — runs tests,
                         refactors functions, opens GitHub PRs."
  - 1-2 sentences, <= 280 characters total.
  - Never invent capabilities the skills don't suggest.
  - Never write "Hermes Agent profile" or other meta-narration.
  - No code fences, no preamble, no closing remarks. Output only JSON.
"""


_USER_TEMPLATE = """Profile name: {name}
Default model: {model}
Provider: {provider}
Installed skill count: {skill_count}
Notable skills (up to {skill_cap}):
{skill_list}
"""


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


@dataclass
class DescribeOutcome:
    """Result of describing a single profile."""

    profile_name: str
    ok: bool
    reason: str = ""
    description: Optional[str] = None


def _collect_skills(profile_dir: Path) -> list[str]:
    """Sorted non-excluded skill names: ``category/skill_name`` (category = immediate subdir
    under ``skills/``), or bare ``skill_name`` for skills directly under ``skills/``."""
    skills_dir = profile_dir / "skills"
    if not skills_dir.is_dir():
        return []
    names: list[str] = []
    for md in skills_dir.rglob("SKILL.md"):
        if is_excluded_skill_path(md):
            continue
        try:
            parts = md.relative_to(skills_dir).parts[:-1]  # drop SKILL.md
        except ValueError:
            continue
        if parts:
            names.append(parts[0] if len(parts) == 1 else f"{parts[0]}/{parts[-1]}")
    names.sort()
    return names


def _sample_skills(names: list[str]) -> list[str]:
    """Cap *names* to the prompt budget with evenly-spaced picks: alphabetical position isn't
    importance, so a profile with skills A..Z must not read as "starts with A"."""
    if len(names) <= MAX_SKILLS_FOR_PROMPT:
        return names
    step = len(names) / MAX_SKILLS_FOR_PROMPT
    return [names[int(i * step)] for i in range(MAX_SKILLS_FOR_PROMPT)]


def _extract_json_blob(raw: str) -> Optional[dict]:
    from hermes_cli.kanban_specify import _extract_json_blob as _extract
    return _extract(raw, _FENCE_RE)


def describe_profile(profile_name: str, *, overwrite: bool = False, timeout: Optional[int] = None) -> DescribeOutcome:
    """Auto-generate a description for one profile. Expected failures (profile missing, no aux
    client, API error, malformed response) return ``ok=False`` so a sweep continues.

    ``overwrite`` allows replacing a user-authored (``description_auto: false``) description;
    auto-generated ones are always replaceable."""
    canon = profiles_mod.normalize_profile_name(profile_name)
    if not profiles_mod.profile_exists(canon):  # handles the virtual "default" name
        return DescribeOutcome(canon, False, "profile not found")
    try:
        if canon == "default":
            from hermes_constants import get_hermes_home  # type: ignore
            profile_dir = Path(get_hermes_home())
        else:
            profile_dir = profiles_mod.get_profile_dir(canon)
    except Exception as exc:
        return DescribeOutcome(canon, False, f"cannot resolve profile dir: {exc}")
    existing = profiles_mod.read_profile_meta(profile_dir)
    if existing.get("description") and not existing.get("description_auto") and not overwrite:
        return DescribeOutcome(
            canon, False, "profile already has a user-authored description (use --overwrite to replace)"
        )
    all_skills = _collect_skills(profile_dir)
    skill_list = "\n".join(f"  - {n}" for n in _sample_skills(all_skills)) or "  (no skills installed)"
    try:
        model, provider = profiles_mod._read_config_model(profile_dir)
    except Exception:
        model, provider = None, None
    try:
        from agent.auxiliary_client import call_llm  # type: ignore
    except Exception as exc:
        logger.debug("describe: auxiliary client import failed: %s", exc)
        return DescribeOutcome(canon, False, "auxiliary client unavailable")
    user_msg = _USER_TEMPLATE.format(
        name=canon, model=(model or "(unset)"), provider=(provider or "(unset)"), skill_count=len(all_skills),
        skill_cap=MAX_SKILLS_FOR_PROMPT, skill_list=skill_list,
    )
    try:
        # call_llm applies auxiliary.profile_describer.* config (provider/model/base_url,
        # extra_body, reasoning_effort, retries); the direct-create path dropped extra_body.
        # See #35566.
        resp = call_llm(
            task="profile_describer",
            messages=[{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": user_msg}],
            temperature=0.3,
            max_tokens=400,
            timeout=timeout or 60,
        )
    except Exception as exc:
        logger.info("describe: API call failed for %s (%s)", canon, exc)
        return DescribeOutcome(canon, False, f"LLM error: {type(exc).__name__}")
    try:
        raw = resp.choices[0].message.content or ""
    except Exception:
        raw = ""
    parsed = _extract_json_blob(raw)
    if parsed is None:
        # Fall back: raw text trimmed to one paragraph.
        text = raw.strip().split("\n\n", 1)[0]
        if not text:
            return DescribeOutcome(canon, False, "LLM returned an empty response")
        description = text[:280]
    else:
        val = parsed.get("description")
        if not isinstance(val, str) or not val.strip():
            return DescribeOutcome(canon, False, "LLM response missing 'description' field")
        description = val.strip()[:280]
    try:
        profiles_mod.write_profile_meta(profile_dir, description=description, description_auto=True)
    except Exception as exc:
        return DescribeOutcome(canon, False, f"failed to write profile.yaml: {exc}")
    return DescribeOutcome(canon, True, "described", description=description)


def list_describable_profiles(*, missing_only: bool = True) -> list[str]:
    """Profile names that can be described; ``missing_only`` keeps only those without a
    user-authored description."""
    return [
        p.name for p in profiles_mod.list_profiles()
        if not (missing_only and (p.description or "").strip() and not p.description_auto)
    ]


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import json  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
