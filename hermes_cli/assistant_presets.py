"""Assistant presets — one-command role profiles.

Inspired by Energy (getenergy.com), whose desktop app ships one-click
specialized assistants ("Inbox Zero", "Research Scout", …) instead of making
users hand-assemble memory, skills, and automations per role. Hermes already
has every underlying piece — profiles (isolated HERMES_HOME), SOUL.md
personas, profile descriptions (used by the kanban decomposer for routing),
and the Automation Blueprints catalog — but until now composing them into a
role took four manual steps. A preset is a curated bundle of those existing
pieces:

  * a persona (written to the new profile's ``SOUL.md``)
  * a profile description (routable by the kanban orchestrator)
  * suggested automations, expressed as (blueprint_key, slot_values) pairs
    against the existing ``cron.blueprint_catalog`` — no second job engine.

Per the dev guide's "Extend, Don't Duplicate" rule there is NO new object
type, storage, or scheduler here: applying a preset just writes files the
profile system already owns and (optionally) creates ordinary cron jobs via
``fill_blueprint`` -> ``create_job`` inside the new profile's HERMES_HOME.

Usage surface:

  hermes profile presets                       list the catalog
  hermes profile create mail --preset inbox-zero
  hermes profile create mail --preset inbox-zero --with-automations
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "AssistantPreset",
    "PRESETS",
    "get_preset",
    "preset_keys",
    "apply_preset_files",
    "seed_preset_automations",
    "format_preset_catalog",
    "suggested_automation_commands",
]


@dataclass(frozen=True)
class AssistantPreset:
    """A curated role bundle applied at ``hermes profile create`` time."""

    key: str
    title: str
    tagline: str            # one line for the catalog listing
    description: str        # profile description (kanban-routable)
    soul: str               # SOUL.md persona text
    # (blueprint_key, slot_values) pairs against cron.blueprint_catalog.CATALOG.
    # Empty values dict = use the blueprint's defaults.
    automations: Tuple[Tuple[str, Dict[str, Any]], ...] = field(default_factory=tuple)


def _soul(role: str, mission: str, principles: List[str]) -> str:
    """Render a consistent SOUL.md persona for a preset role."""
    lines = [
        f"You are {role}, a specialist Hermes assistant. {mission}",
        "",
        "Operating principles:",
    ]
    lines += [f"- {p}" for p in principles]
    lines += [
        "",
        "You are one of several role assistants the user may run side by side; "
        "stay in your lane, and when a request is clearly another role's job, "
        "say so briefly and still help if asked.",
    ]
    return "\n".join(lines)


PRESETS: List[AssistantPreset] = [
    AssistantPreset(
        key="research-scout",
        title="Research Scout",
        tagline="Finds what matters and distills it",
        description=(
            "Research specialist: web research, source-grounded digests, "
            "competitive scans, and summarization. Route research, "
            "fact-finding, and monitoring tasks here."
        ),
        soul=_soul(
            "Research Scout",
            "Your job is finding what matters and bringing it back distilled.",
            [
                "Search broadly, then cut ruthlessly — deliver the three "
                "things worth knowing, not everything you found.",
                "Always keep links to primary sources; never launder a claim "
                "without its origin.",
                "Dedupe against what you already reported; only genuinely "
                "new developments count.",
                "State confidence honestly — a clearly-flagged rumor beats a "
                "false certainty.",
            ],
        ),
        automations=(("news-digest", {}),),
    ),
    AssistantPreset(
        key="inbox-zero",
        title="Inbox Zero",
        tagline="Clears the queue",
        description=(
            "Email specialist: inbox triage, urgent-mail surfacing, drafting "
            "replies, and unsubscribe hygiene. Route email and "
            "communications tasks here."
        ),
        soul=_soul(
            "Inbox Zero",
            "Your job is keeping the user's inbox from owning their day.",
            [
                "Surface only mail that actually needs the user; everything "
                "else gets summarized in one line or not at all.",
                "Draft replies in the user's voice, ready to send — but never "
                "send without explicit approval.",
                "Be aggressive about noise: recurring newsletters and "
                "notification spam are candidates for unsubscribe suggestions.",
                "When triaging, lead with the single most urgent item.",
            ],
        ),
        automations=(("important-mail", {}),),
    ),
    AssistantPreset(
        key="project-captain",
        title="Project Captain",
        tagline="Keeps work on track",
        description=(
            "Project coordination specialist: status tracking, priorities, "
            "weekly reviews, and follow-ups. Route planning, coordination, "
            "and progress-tracking tasks here."
        ),
        soul=_soul(
            "Project Captain",
            "Your job is keeping work moving and nothing falling through the cracks.",
            [
                "Every check-in ends with owners and next actions, not vibes.",
                "Distinguish blocked from stalled from done; chase the "
                "blocked ones first.",
                "Keep status updates short enough to read standing up.",
                "When priorities conflict, present the trade-off in two "
                "lines and ask for a call.",
            ],
        ),
        automations=(("workday-start", {}), ("weekly-review", {})),
    ),
    AssistantPreset(
        key="finance-keeper",
        title="Finance Keeper",
        tagline="Watches every number",
        description=(
            "Finance specialist: bills, renewals, budgets, spreadsheets, and "
            "spending summaries. Route financial tracking and "
            "number-crunching tasks here."
        ),
        soul=_soul(
            "Finance Keeper",
            "Your job is making sure no number surprises the user.",
            [
                "Flag renewals and charges BEFORE they hit, framed as an "
                "action (review / cancel / let it ride), not a notification.",
                "Show your arithmetic; a total without its breakdown is a "
                "claim, not an answer.",
                "Round for readability, keep precision in the working.",
                "Never move money or commit to a purchase — prepare the "
                "action and hand it to the user.",
            ],
        ),
        automations=(("bill-renewal-watch", {}),),
    ),
    AssistantPreset(
        key="sales-pilot",
        title="Sales Pilot",
        tagline="Moves deals forward",
        description=(
            "Sales and outreach specialist: prospect research, follow-up "
            "drafting, pipeline nudges, and meeting prep. Route outreach and "
            "deal-related tasks here."
        ),
        soul=_soul(
            "Sales Pilot",
            "Your job is moving conversations toward closed, without being pushy.",
            [
                "Every touch has a purpose the recipient can see; no "
                "'just checking in' filler.",
                "Research before outreach — reference something true and "
                "recent about the prospect.",
                "Track the next step for every open thread; a deal without "
                "a next step is a dead deal.",
                "Drafts are the deliverable: crisp, short, in the user's "
                "voice, ready to send after approval.",
            ],
        ),
        automations=(("weekly-review", {"day": "friday"}),),
    ),
]

_PRESETS_BY_KEY = {p.key: p for p in PRESETS}


def get_preset(key: str) -> Optional[AssistantPreset]:
    return _PRESETS_BY_KEY.get((key or "").strip().lower())


def preset_keys() -> List[str]:
    return [p.key for p in PRESETS]


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

def apply_preset_files(profile_dir: Path, preset: AssistantPreset) -> None:
    """Write the preset's SOUL.md + profile description into ``profile_dir``.

    Called from ``create_profile`` after directory bootstrap. Overwrites the
    default seeded SOUL.md (the preset IS the requested persona) but never
    raises — profile creation must not fail over persona cosmetics.
    """
    try:
        (profile_dir / "SOUL.md").write_text(preset.soul + "\n", encoding="utf-8")
    except OSError as e:
        logger.warning("preset %s: could not write SOUL.md: %s", preset.key, e)
    try:
        from hermes_cli.profiles import write_profile_meta

        write_profile_meta(
            profile_dir,
            description=preset.description,
            description_auto=False,
        )
    except Exception as e:
        logger.warning("preset %s: could not write description: %s", preset.key, e)


def seed_preset_automations(
    profile_dir: Path, preset: AssistantPreset, quiet: bool = False
) -> List[str]:
    """Create the preset's suggested automations inside the new profile.

    Runs in a subprocess with ``HERMES_HOME`` pointed at the profile (the same
    isolation pattern as ``seed_profile_skills``) so the jobs land in the
    profile's own cron store, not the invoking profile's. Returns the names of
    the jobs created.
    """
    if not preset.automations:
        return []
    project_root = Path(__file__).parent.parent.resolve()
    payload = json.dumps([[key, values] for key, values in preset.automations])
    script = (
        "import json, sys\n"
        "from cron.blueprint_catalog import get_blueprint, fill_blueprint\n"
        "from cron.jobs import create_job\n"
        "created = []\n"
        "for key, values in json.loads(sys.argv[1]):\n"
        "    bp = get_blueprint(key)\n"
        "    if bp is None:\n"
        "        continue\n"
        "    spec = fill_blueprint(bp, values)\n"
        "    # Preset-seeded jobs have no chat origin; let create_job fall back\n"
        "    # to local delivery instead of a dangling 'origin' target.\n"
        "    if spec.get('deliver') == 'origin' and not spec.get('origin'):\n"
        "        spec.pop('deliver')\n"
        "    job = create_job(**spec)\n"
        "    created.append(job.get('name') or key)\n"
        "print(json.dumps(created))\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, payload],
            env={**os.environ, "HERMES_HOME": str(profile_dir)},
            cwd=str(project_root),
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=60,
        )
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout.strip().splitlines()[-1])
        if not quiet:
            print(f"⚠ Preset automations returned exit code {result.returncode}")
            if result.stderr.strip():
                print(f"  {result.stderr.strip()[:200]}")
    except subprocess.TimeoutExpired:
        if not quiet:
            print("⚠ Preset automation seeding timed out (60s)")
    except Exception as e:
        if not quiet:
            print(f"⚠ Preset automation seeding failed: {e}")
    return []


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def suggested_automation_commands(preset: AssistantPreset, profile_name: str) -> List[str]:
    """Ready-to-paste commands for the preset's automations (when not auto-seeded)."""
    from cron.blueprint_catalog import blueprint_slash_command, get_blueprint

    cmds: List[str] = []
    for key, values in preset.automations:
        bp = get_blueprint(key)
        if bp is None:
            continue
        cmds.append(f"hermes -p {profile_name} chat  →  {blueprint_slash_command(bp, values)}")
    return cmds


def format_preset_catalog() -> str:
    """Human-readable catalog for ``hermes profile presets``."""
    lines = ["Assistant presets — one-command role profiles:", ""]
    for p in PRESETS:
        lines.append(f"  {p.key:<16} {p.title} — {p.tagline}")
        if p.automations:
            names = ", ".join(key for key, _ in p.automations)
            lines.append(f"  {'':<16} automations: {names}")
    lines += [
        "",
        "Create one:  hermes profile create <name> --preset <key>",
        "Add its automations too:  --with-automations",
    ]
    return "\n".join(lines)
