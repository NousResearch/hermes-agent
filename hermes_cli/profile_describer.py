"""Profile describer — auto-generate ``description`` for a profile.

Used by ``hermes profile describe <name> --auto`` and the dashboard's
"auto-generate description" button. Reads the profile's installed
skills, model+provider, name, and optionally a small slice of memory,
then asks the auxiliary LLM to produce a 1-2 sentence description of
what the profile is good at.

Result is written to ``<profile_dir>/profile.yaml`` with
``description_auto: true`` so the dashboard can surface a "review"
badge. User can edit afterward to confirm.

Design notes
------------
- Mirrors the shape of ``hermes_cli/kanban_specify.py``: lazy aux
  client import inside the function, lenient response parse, never
  raises on expected failure modes.
- Reads at most ``MAX_SKILLS_FOR_PROMPT`` skill names to keep the
  prompt bounded. No skill body — names + categories are enough
  signal and avoid blowing context on profiles with 100+ skills.
- Memory is intentionally NOT read here. Memories are personal and
  the orchestrator routes work to a *role* not a *biography*. If we
  find later that memory adds signal we can wire it; for now,
  skills + name + model is plenty.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from hermes_cli import profiles as profiles_mod

logger = logging.getLogger(__name__)

# Cap on how many skill names we feed the LLM. Profiles with 200+
# skills (uncommon but possible) would blow context otherwise. The cap
# is per-category — see _collect_skills.
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
    """Return a stable, capped list of skill names for the prompt.

    Format: ``category/skill_name`` where category is the immediate
    subdir under ``skills/`` (e.g. ``devops``, ``research``). Skills
    that live directly under ``skills/`` show as bare ``skill_name``.
    """
    skills_dir = profile_dir / "skills"
    if not skills_dir.is_dir():
        return []
    names: list[str] = []
    for md in skills_dir.rglob("SKILL.md"):
        path_str = str(md)
        if "/.hub/" in path_str or "/.git/" in path_str:
            continue
        try:
            rel = md.relative_to(skills_dir)
        except ValueError:
            continue
        parts = rel.parts[:-1]  # drop SKILL.md filename
        if not parts:
            continue
        # parts[-1] is the skill dir name; parts[:-1] is the category path
        if len(parts) == 1:
            names.append(parts[0])
        else:
            names.append(f"{parts[0]}/{parts[-1]}")
    names.sort()
    # Keep within prompt budget. Skills earlier in alphabet aren't more
    # important — we'll let the LLM see a sample. Pick evenly-spaced
    # entries instead of just the head so a profile with skills A..Z
    # doesn't get described as "starts with A".
    if len(names) <= MAX_SKILLS_FOR_PROMPT:
        return names
    step = len(names) / MAX_SKILLS_FOR_PROMPT
    sampled = [names[int(i * step)] for i in range(MAX_SKILLS_FOR_PROMPT)]
    return sampled


def _extract_json_blob(raw: str) -> Optional[dict]:
    if not raw:
        return None
    stripped = _FENCE_RE.sub("", raw.strip())
    first = stripped.find("{")
    last = stripped.rfind("}")
    if first == -1 or last == -1 or last <= first:
        return None
    candidate = stripped[first : last + 1]
    try:
        val = json.loads(candidate)
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(val, dict):
        return None
    return val


def describe_profile(
    profile_name: str,
    *,
    overwrite: bool = False,
    timeout: Optional[int] = None,
) -> DescribeOutcome:
    """Auto-generate a description for one profile.

    Returns an outcome describing what happened. Never raises for
    expected failure modes (profile missing, no aux client configured,
    API error, malformed response) — those surface via ``ok=False`` so
    a sweep can continue past individual failures.

    ``overwrite`` controls whether an existing user-authored description
    is replaced. By default we refuse to overwrite a description with
    ``description_auto: false`` to protect curated text. Auto-generated
    descriptions (``description_auto: true``) are always replaceable.
    """
    canon = profiles_mod.normalize_profile_name(profile_name)
    if not profiles_mod.profile_exists(canon):
        # Special case: "default" exists as a virtual profile name
        # mapped to the default home dir. profile_exists() handles it.
        return DescribeOutcome(canon, False, "profile not found")

    try:
        if canon == "default":
            from hermes_constants import get_hermes_home  # type: ignore
            profile_dir = Path(get_hermes_home())
        else:
            profile_dir = profiles_mod.get_profile_dir(canon)
    except Exception as exc:
        return DescribeOutcome(canon, False, f"cannot resolve profile dir: {exc}")

    # Honor curated descriptions unless --overwrite.
    existing = profiles_mod.read_profile_meta(profile_dir)
    if existing.get("description") and not existing.get("description_auto") and not overwrite:
        return DescribeOutcome(
            canon,
            False,
            "profile already has a user-authored description "
            "(use --overwrite to replace)",
        )

    skill_names = _collect_skills(profile_dir)
    skill_list = "\n".join(f"  - {n}" for n in skill_names) or "  (no skills installed)"
    skill_count = sum(
        1 for _ in (profile_dir / "skills").rglob("SKILL.md")
        if "/.hub/" not in str(_) and "/.git/" not in str(_)
    ) if (profile_dir / "skills").is_dir() else 0

    # Read model + provider from the profile's config.
    try:
        model, provider = profiles_mod._read_config_model(profile_dir)
    except Exception:
        model, provider = None, None

    try:
        from agent.auxiliary_client import (  # type: ignore
            get_auxiliary_extra_body,
            get_text_auxiliary_client,
        )
    except Exception as exc:
        logger.debug("describe: auxiliary client import failed: %s", exc)
        return DescribeOutcome(canon, False, "auxiliary client unavailable")

    try:
        client, aux_model = get_text_auxiliary_client("profile_describer")
    except Exception as exc:
        logger.debug("describe: get_text_auxiliary_client failed: %s", exc)
        return DescribeOutcome(canon, False, "auxiliary client unavailable")

    if client is None or not aux_model:
        return DescribeOutcome(canon, False, "no auxiliary client configured")

    user_msg = _USER_TEMPLATE.format(
        name=canon,
        model=(model or "(unset)"),
        provider=(provider or "(unset)"),
        skill_count=skill_count,
        skill_cap=MAX_SKILLS_FOR_PROMPT,
        skill_list=skill_list,
    )

    try:
        resp = client.chat.completions.create(
            model=aux_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=400,
            timeout=timeout or 60,
            extra_body=get_auxiliary_extra_body() or None,
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
        # Fall back: take the raw text trimmed to one paragraph.
        text = raw.strip().split("\n\n", 1)[0]
        if not text:
            return DescribeOutcome(canon, False, "LLM returned an empty response")
        description = text[:280]
    else:
        val = parsed.get("description")
        if not isinstance(val, str) or not val.strip():
            return DescribeOutcome(
                canon, False, "LLM response missing 'description' field"
            )
        description = val.strip()[:280]

    try:
        profiles_mod.write_profile_meta(
            profile_dir,
            description=description,
            description_auto=True,
        )
    except Exception as exc:
        return DescribeOutcome(canon, False, f"failed to write profile.yaml: {exc}")

    return DescribeOutcome(canon, True, "described", description=description)


def list_describable_profiles(*, missing_only: bool = True) -> list[str]:
    """Return profile names that can be described.

    ``missing_only=True`` (default) returns only profiles without a
    description. ``missing_only=False`` returns every profile.
    """
    out: list[str] = []
    for p in profiles_mod.list_profiles():
        if missing_only and (p.description or "").strip() and not p.description_auto:
            continue
        out.append(p.name)
    return out
