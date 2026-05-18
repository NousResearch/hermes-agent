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
- Each skill name is tagged ``[built-in]`` or ``[user]`` based on
  whether it ships with Hermes (in ``skills/`` or ``optional-skills/``
  under the install root) or was added by the user. The system prompt
  instructs the LLM to weight ``[user]`` skills as the strongest signal
  of role/domain — built-ins are present in nearly every profile by
  default and carry minimal lane information.
- Memory is intentionally NOT read here. Memories are personal and
  the orchestrator routes work to a *role* not a *biography*. If we
  find later that memory adds signal we can wire it; for now,
  skills + name + model is plenty.
- ``include_soul=True`` (opt-in via CLI flag) pulls the profile's
  SOUL.md content into the prompt when the user keeps lane/role info
  in SOUL.md. Per the canonical Hermes docs SOUL.md is voice/tone
  only, so this is opt-in and never default.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from hermes_cli import profiles as profiles_mod

logger = logging.getLogger(__name__)

# Cap on how many skill names we feed the LLM. Profiles with 200+
# skills (uncommon but possible) would blow context otherwise. The cap
# is per-category — see _collect_skills.
MAX_SKILLS_FOR_PROMPT = 60

# Cap on SOUL.md content fed when --include-soul is passed. SOUL.md is
# meant to be small per Hermes docs, but we cap to be safe.
MAX_SOUL_CHARS = 4000

# Cap on AGENTS.md content fed when --include-agents is passed. AGENTS.md
# is the canonical place for lane/role information so we allow a slightly
# larger budget than SOUL.md, but still bounded.
MAX_AGENTS_CHARS = 6000


_SYSTEM_PROMPT_BASE = """You are a profile-describer for the Hermes Agent kanban board.

A user runs multiple "profiles" — distinct agent identities, each with their
own skills, model, and configuration. The kanban board's orchestrator routes
work to whichever profile best fits each task. To do that well, every
profile needs a short, concrete description of what it's good at.

You are given a profile's:
  - Name
  - Model / provider
  - List of installed skill names{tag_blurb}{soul_blurb}{agents_blurb}

Produce a single JSON object with exactly one key:

  {{
    "description": "<1-2 sentence description, plain prose, no preamble>"
  }}

Rules:
  - The description is what an orchestrator will read to decide whether to
    route a task here. Lead with the profile's strongest capability.
  - Stay concrete. Bad: "an AI agent that helps users."
                  Good: "Reads and modifies Python codebases — runs tests,
                         refactors functions, opens GitHub PRs."
  - 1-2 sentences, <= 280 characters total.{tag_rule}{soul_rule}{agents_rule}
  - Never invent capabilities the skills don't suggest.
  - Never write "Hermes Agent profile" or other meta-narration.
  - No code fences, no preamble, no closing remarks. Output only JSON.
"""

_TAG_BLURB = (
    ", each tagged either [built-in] (ships with Hermes by default — "
    "present in nearly every profile) or [user] (deliberately added by "
    "this user — strong role/domain signal)"
)
_SOUL_BLURB = (
    "\n  - SOUL.md content describing role/identity (only when the user "
    "explicitly opted in)"
)
_AGENTS_BLURB = (
    "\n  - AGENTS.md content describing the profile's lane, role, and "
    "workflow (the canonical Hermes location for project-specific role "
    "information)"
)
_TAG_RULE = (
    "\n  - Weight [user] skills as the dominant signal of role and domain. "
    "[built-in] skills are present in most profiles and carry weak lane "
    "information — a profile with 80 [built-in] skills and 5 [user] skills "
    "is defined by those 5, not the 80."
)
_SOUL_RULE = (
    "\n  - When SOUL.md content is provided, treat the role/identity "
    "portion as authoritative for the agent's purpose; let skills inform "
    "capability detail but not override stated role."
)
_AGENTS_RULE = (
    "\n  - When AGENTS.md content is provided, treat its role/lane "
    "statements as the highest-priority signal — AGENTS.md is the "
    "canonical Hermes location for declaring what an agent does. Skills "
    "and other signals fill in capability detail; they do not override "
    "stated role."
)


def _build_system_prompt(
    *, tag_builtins: bool, include_soul: bool, include_agents: bool
) -> str:
    """Build the system prompt with sections gated to the active flags."""
    return _SYSTEM_PROMPT_BASE.format(
        tag_blurb=_TAG_BLURB if tag_builtins else "",
        soul_blurb=_SOUL_BLURB if include_soul else "",
        agents_blurb=_AGENTS_BLURB if include_agents else "",
        tag_rule=_TAG_RULE if tag_builtins else "",
        soul_rule=_SOUL_RULE if include_soul else "",
        agents_rule=_AGENTS_RULE if include_agents else "",
    )


_USER_TEMPLATE = """Profile name: {name}
Default model: {model}
Provider: {provider}
Installed skill count: {skill_count_text}
Notable skills (up to {skill_cap}):
{skill_list}
{soul_block}{agents_block}"""


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


@dataclass
class DescribeOutcome:
    """Result of describing a single profile."""

    profile_name: str
    ok: bool
    reason: str = ""
    description: Optional[str] = None


def _hermes_install_root() -> Optional[Path]:
    """Locate the Hermes install root so we can find the canonical
    ``skills/`` and ``optional-skills/`` directories.

    Resolves from ``hermes_cli`` package location: walk up from
    ``hermes_cli/__file__`` until we find a parent that has both
    ``skills/`` and ``hermes_cli/`` siblings. Returns ``None`` if
    we can't locate it (e.g. site-packages install where the source
    tree isn't present); caller falls back to treating every skill
    as ``[user]`` which is safe — labels just become less informative.
    """
    try:
        from hermes_cli import __file__ as _cli_file
    except Exception:
        return None
    p = Path(_cli_file).resolve().parent  # .../hermes_cli/
    for candidate in [p.parent, p.parent.parent]:
        if (candidate / "skills").is_dir() and (candidate / "hermes_cli").is_dir():
            return candidate
    return None


def _builtin_skill_ids() -> set[str]:
    """Return the set of skill IDs that ship with Hermes.

    A skill ID is the same shape ``_collect_skills`` produces:
    ``category/name`` for nested skills, bare ``name`` for top-level
    ones. We scan both ``skills/`` (auto-installed on profile create)
    and ``optional-skills/`` (opt-in but still upstream-shipped).

    Result is cached on the module to avoid re-walking on every call.
    """
    cached = getattr(_builtin_skill_ids, "_cache", None)
    if cached is not None:
        return cached
    out: set[str] = set()
    root = _hermes_install_root()
    if root is None:
        _builtin_skill_ids._cache = out  # type: ignore[attr-defined]
        return out
    for sub in ("skills", "optional-skills"):
        d = root / sub
        if not d.is_dir():
            continue
        for md in d.rglob("SKILL.md"):
            try:
                rel = md.relative_to(d)
            except ValueError:
                continue
            parts = rel.parts[:-1]
            if not parts:
                continue
            if len(parts) == 1:
                out.add(parts[0])
            else:
                out.add(f"{parts[0]}/{parts[-1]}")
    _builtin_skill_ids._cache = out  # type: ignore[attr-defined]
    return out


def _collect_skills(
    profile_dir: Path,
    *,
    classify_builtins: bool = False,
) -> list[Tuple[str, bool]]:
    """Return a stable, capped list of ``(skill_id, is_user_authored)`` tuples.

    Format: ``category/skill_name`` where category is the immediate
    subdir under ``skills/`` (e.g. ``devops``, ``research``). Skills
    that live directly under ``skills/`` show as bare ``skill_name``.

    The boolean is ``True`` when the skill is NOT a built-in Hermes skill
    (i.e. the user added it deliberately — strong lane signal). When
    ``classify_builtins=False`` the boolean is always ``False`` and the
    cap is enforced via even sampling across the full sorted list rather
    than by biasing toward user-authored skills.
    """
    skills_dir = profile_dir / "skills"
    if not skills_dir.is_dir():
        return []
    builtin = _builtin_skill_ids() if classify_builtins else set()
    pairs: list[Tuple[str, bool]] = []
    seen: set[str] = set()
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
            sid = parts[0]
        else:
            sid = f"{parts[0]}/{parts[-1]}"
        if sid in seen:
            continue
        seen.add(sid)
        # When classify_builtins is False we report every skill as
        # is_user=False so callers don't accidentally emit [user]/[built-in]
        # tags. The flag is purely about how the prompt is shaped.
        is_user = (sid not in builtin) if classify_builtins else False
        pairs.append((sid, is_user))
    pairs.sort(key=lambda p: p[0])

    if len(pairs) <= MAX_SKILLS_FOR_PROMPT:
        return pairs

    if not classify_builtins:
        # Sample evenly across the sorted list — caller asked us not
        # to differentiate built-ins, so don't bias the sample either.
        step = len(pairs) / MAX_SKILLS_FOR_PROMPT
        return [pairs[int(i * step)] for i in range(MAX_SKILLS_FOR_PROMPT)]

    # Classified path: prioritize user-authored skills, then sample built-ins.
    user_pairs = [p for p in pairs if p[1]]
    builtin_pairs = [p for p in pairs if not p[1]]
    if len(user_pairs) >= MAX_SKILLS_FOR_PROMPT:
        step = len(user_pairs) / MAX_SKILLS_FOR_PROMPT
        return [user_pairs[int(i * step)] for i in range(MAX_SKILLS_FOR_PROMPT)]
    remaining = MAX_SKILLS_FOR_PROMPT - len(user_pairs)
    if not builtin_pairs or remaining <= 0:
        return user_pairs[:MAX_SKILLS_FOR_PROMPT]
    step = max(1.0, len(builtin_pairs) / remaining)
    sampled_builtin = [
        builtin_pairs[int(i * step)] for i in range(remaining)
        if int(i * step) < len(builtin_pairs)
    ]
    out = user_pairs + sampled_builtin
    out.sort(key=lambda p: p[0])
    return out


def _load_soul_md(profile_dir: Path) -> str:
    """Return SOUL.md content for the profile, capped to ``MAX_SOUL_CHARS``.

    Returns empty string if SOUL.md is missing, empty, or unreadable.
    """
    soul_path = profile_dir / "SOUL.md"
    if not soul_path.is_file():
        return ""
    try:
        text = soul_path.read_text(errors="replace").strip()
    except Exception as exc:
        logger.debug("describe: could not read SOUL.md at %s: %s", soul_path, exc)
        return ""
    if not text:
        return ""
    if len(text) > MAX_SOUL_CHARS:
        text = text[:MAX_SOUL_CHARS] + "\n... (truncated)"
    return text


# Candidate locations for AGENTS.md, in priority order. Hermes itself supports
# AGENTS.md at the project cwd (canonical) or anywhere in the profile tree;
# operators commonly stash a fleet-style AGENTS.md under a workspace/
# subdirectory of the profile when the profile maps to a fixed working
# directory. We try the canonical profile-root location first, then fall
# back to workspace/ which is the convention used in the multi-agent fleet
# layout this feature was originally designed to support.
_AGENTS_MD_CANDIDATES = (
    Path("AGENTS.md"),
    Path("workspace") / "AGENTS.md",
)


def _load_agents_md(profile_dir: Path) -> str:
    """Return AGENTS.md content for the profile, capped to ``MAX_AGENTS_CHARS``.

    AGENTS.md is the canonical place for project-specific lane / role /
    workflow information per the Hermes docs. We check the profile root
    first, then ``workspace/AGENTS.md`` as a fallback for fleet layouts
    that keep agent context in a workspace subdirectory.

    Returns empty string if no AGENTS.md is found, the file is empty, or
    it cannot be read.
    """
    for rel in _AGENTS_MD_CANDIDATES:
        candidate = profile_dir / rel
        if not candidate.is_file():
            continue
        try:
            text = candidate.read_text(errors="replace").strip()
        except Exception as exc:
            logger.debug("describe: could not read AGENTS.md at %s: %s", candidate, exc)
            continue
        if not text:
            continue
        if len(text) > MAX_AGENTS_CHARS:
            text = text[:MAX_AGENTS_CHARS] + "\n... (truncated)"
        return text
    return ""


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
    tag_builtins: bool = False,
    include_soul: bool = False,
    include_agents: bool = False,
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

    ``tag_builtins`` (default False) labels each skill in the prompt as
    [user] or [built-in] based on whether it ships with Hermes, and
    instructs the LLM to weight [user] skills as the dominant role/domain
    signal. Off by default so existing behavior is unchanged. Enable when
    profiles are heavy with built-in skills that drown out the lane.

    ``include_soul`` (default False) pulls the profile's SOUL.md content
    into the prompt as a role-identity signal. Per the canonical Hermes
    docs SOUL.md is voice/tone only, so this is off by default — only
    set True when the profile keeps lane/role information in SOUL.md.

    ``include_agents`` (default False) pulls the profile's AGENTS.md
    content into the prompt as the canonical role/lane signal. AGENTS.md
    is the docs-blessed location for project-specific role information,
    but it's off by default to keep existing prompt shape unchanged for
    callers who don't pass the flag.
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

    skill_pairs = _collect_skills(profile_dir, classify_builtins=tag_builtins)
    if skill_pairs:
        skill_lines = []
        for sid, is_user in skill_pairs:
            if tag_builtins:
                tag = "[user] " if is_user else "[built-in] "
            else:
                tag = ""
            skill_lines.append(f"  - {tag}{sid}")
        skill_list = "\n".join(skill_lines)
    else:
        skill_list = "  (no skills installed)"
    skill_count = sum(
        1 for _ in (profile_dir / "skills").rglob("SKILL.md")
        if "/.hub/" not in str(_) and "/.git/" not in str(_)
    ) if (profile_dir / "skills").is_dir() else 0
    user_count = sum(1 for _, u in skill_pairs if u) if tag_builtins else 0
    builtin_count = (
        sum(1 for _, u in skill_pairs if not u) if tag_builtins else 0
    )
    # Format the count line conditionally so unclassified runs don't
    # emit a misleading "(0 user, 0 built-in)" suffix.
    if tag_builtins:
        skill_count_text = (
            f"{skill_count} ({user_count} user, {builtin_count} built-in)"
        )
    else:
        skill_count_text = str(skill_count)

    # Optional SOUL.md identity block (opt-in).
    soul_block = ""
    if include_soul:
        soul_text = _load_soul_md(profile_dir)
        if soul_text:
            soul_block = (
                "\nProfile SOUL.md (role/identity per user opt-in):\n"
                f"---\n{soul_text}\n---"
            )

    # Optional AGENTS.md lane/role block (opt-in). This is the canonical
    # docs-blessed home for role information; SOUL.md is the escape hatch
    # for fleets that pollute SOUL with lane content.
    agents_block = ""
    if include_agents:
        agents_text = _load_agents_md(profile_dir)
        if agents_text:
            agents_block = (
                "\nProfile AGENTS.md (canonical lane/role/workflow):\n"
                f"---\n{agents_text}\n---"
            )

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

    system_prompt = _build_system_prompt(
        tag_builtins=tag_builtins,
        include_soul=include_soul,
        include_agents=include_agents,
    )
    user_msg = _USER_TEMPLATE.format(
        name=canon,
        model=(model or "(unset)"),
        provider=(provider or "(unset)"),
        skill_count_text=skill_count_text,
        skill_cap=MAX_SKILLS_FOR_PROMPT,
        skill_list=skill_list,
        soul_block=soul_block,
        agents_block=agents_block,
    )

    try:
        resp = client.chat.completions.create(
            model=aux_model,
            messages=[
                {"role": "system", "content": system_prompt},
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
