"""Kanban Swarm slash-command handler — /swarm <goal>.

Auto-decomposes a natural-language goal into a Kanban Swarm v1 topology
by calling the auxiliary LLM with the user's profile roster, then creates
the swarm via ``kanban_swarm.create_swarm()``.

Usage (in-session)::

    /swarm Research the Portia spider's depth perception and write it up

This module intentionally delegates to every existing abstraction:
- ``kanban_decompose._build_roster`` / ``_format_roster`` — profile roster
- ``kanban_swarm.create_swarm`` — durable swarm graph
- ``agent.auxiliary_client`` — LLM decomposition call
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_swarm as ks
from hermes_cli import profiles as profiles_mod

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class SwarmPlan:
    """LLM-generated plan for a swarm decomposition."""

    topology: str
    rationale: str
    workers: list[ks.SwarmWorkerSpec]
    verifier_profile: Optional[str] = None
    synthesizer_profile: Optional[str] = None


# ---------------------------------------------------------------------------
# Decomposition — calls the auxiliary LLM
# ---------------------------------------------------------------------------

_DECOMPOSER_TASK = "kanban_decomposer"  # reuses existing aux config

_SYSTEM_PROMPT = """\
You are a Swarm Decomposer for the Hermes Agent Kanban system.
Given a user goal and a roster of available profiles (each with a description),
you decompose the goal into a Kanban Swarm topology.

Available topologies:
- full: parallel workers → verifier gate → synthesizer.
  Use when the output needs verification before delivery (high-stakes research,
  content to be published, decisions to be acted on).
- research-only: parallel workers only, no verifier or synthesizer.
  Use for independent research findings, data gathering, or exploratory work
  where each worker's output stands alone.

Output a single JSON object with this exact shape:
{
  "topology": "full" or "research-only",
  "rationale": "One sentence explaining why this topology fits the goal.",
  "workers": [
    {
      "profile": "profile_name",
      "title": "Imperative task title, <= 80 chars",
      "body": "2-4 sentence detailed instruction for the worker"
    }
  ],
  "verifier_profile": "profile_name or null if research-only",
  "synthesizer_profile": "profile_name or null if research-only"
}

Rules:
- Generate 2-4 workers. More than 5 means the goal is too broad.
- Assign workers to profiles based on description match, not name alone.
- Worker titles must be concrete and actionable.
- Worker bodies must include: what to investigate, what sources to consult,
  what output format to produce.
- For "full" topology: verifier is always "reviewer", synthesizer is "writer"
  for content goals or "researcher" for research briefs.
"""


def _decompose_goal(
    goal: str,
    roster: list[dict],
    *,
    timeout: Optional[int] = None,
) -> SwarmPlan:
    """Call the auxiliary LLM to decompose *goal* into a SwarmPlan.

    Raises ``ValueError`` with a user-facing message on any failure
    (empty response, malformed JSON, missing fields).
    """
    from hermes_cli.kanban_decompose import _extract_json_blob, _format_roster
    from agent.auxiliary_client import (
        get_auxiliary_extra_body,
        get_text_auxiliary_client,
    )

    client, aux_model = get_text_auxiliary_client(_DECOMPOSER_TASK)
    if client is None or not aux_model:
        raise ValueError(
            "Swarm decomposer is not available — no auxiliary LLM configured "
            f"(auxiliary.{_DECOMPOSER_TASK})"
        )

    roster_text = _format_roster(roster)
    user_msg = f"Goal:\n{goal}\n\nAvailable profiles:\n{roster_text}"

    try:
        resp = client.chat.completions.create(
            model=aux_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=1500,
            timeout=timeout or 120,
            extra_body=get_auxiliary_extra_body(task=_DECOMPOSER_TASK) or None,
        )
    except Exception as exc:
        raise ValueError(f"Swarm decomposer LLM call failed: {type(exc).__name__}") from exc

    raw = (resp.choices[0].message.content or "").strip()
    if not raw:
        raise ValueError(
            "Swarm decomposer returned an empty response. "
            "Check auxiliary.{_DECOMPOSER_TASK} model and provider."
        )

    parsed = _extract_json_blob(raw)
    if parsed is None:
        raise ValueError(
            "Swarm decomposer returned unparseable JSON. "
            "The model may not be following instructions."
        )

    return _parse_swarm_plan(parsed)


def _parse_swarm_plan(parsed: dict) -> SwarmPlan:
    """Validate and convert the LLM JSON response into a SwarmPlan."""
    topology = parsed.get("topology", "full")
    if topology not in ("full", "research-only"):
        raise ValueError(f"Unknown topology: {topology!r}. Must be 'full' or 'research-only'.")

    rationale = (parsed.get("rationale") or "").strip() or "No rationale provided."

    raw_workers = parsed.get("workers")
    if not isinstance(raw_workers, list) or not raw_workers:
        raise ValueError("Swarm decomposer returned no workers. Cannot create a swarm without workers.")

    if len(raw_workers) > 5:
        logger.warning("swarm: %d workers is a lot — goal may be too broad", len(raw_workers))

    workers = []
    for w in raw_workers:
        profile = (w.get("profile") or "").strip()
        title = (w.get("title") or "").strip()
        body = (w.get("body") or "").strip()
        if not profile or not title:
            raise ValueError(f"Worker missing required field 'profile' or 'title': {w}")
        workers.append(ks.SwarmWorkerSpec(profile=profile, title=title, body=body))

    verifier = (parsed.get("verifier_profile") or "").strip() or None
    synthesizer = (parsed.get("synthesizer_profile") or "").strip() or None

    return SwarmPlan(
        topology=topology,
        rationale=rationale,
        workers=workers,
        verifier_profile=verifier,
        synthesizer_profile=synthesizer,
    )


def _resolve_assignee(
    profile_name: Optional[str],
    *,
    valid_names: set[str],
    default_assignee: str,
) -> Optional[str]:
    """Return a valid profile name or None.

    Falls back to *default_assignee* when *profile_name* is empty or unknown.
    """
    if not profile_name:
        return None
    if profile_name not in valid_names:
        logger.warning(
            "swarm: profile %r not found — using fallback %r",
            profile_name,
            default_assignee,
        )
        return default_assignee
    return profile_name


def _profile_author() -> str:
    """Return the active profile name or 'swarm'."""
    try:
        return profiles_mod.get_active_profile_name() or "swarm"
    except Exception:
        return "swarm"


# ---------------------------------------------------------------------------
# CLI handler
# ---------------------------------------------------------------------------


def _cmd_swarm(args: argparse.Namespace) -> int:
    """``/swarm`` handler — parses goal, decomposes, creates kanban tasks."""
    from hermes_cli.kanban_decompose import _build_roster

    goal = (getattr(args, "goal", None) or getattr(args, "goal_text", None) or "").strip()
    if not goal:
        print("/swarm: a goal is required.", file=sys.stderr)
        print("Usage: /swarm <goal>  — e.g., /swarm Research X and write a brief", file=sys.stderr)
        return 2

    # 1. Build profile roster
    roster, valid_names = _build_roster()

    # 2. Decompose via LLM
    try:
        plan = _decompose_goal(goal, roster)
    except ValueError as exc:
        print(f"/swarm: {exc}", file=sys.stderr)
        return 1

    # 3. Resolve assignees
    default = _profile_author()
    resolved_workers = []
    for w in plan.workers:
        resolved = _resolve_assignee(w.profile, valid_names=valid_names, default_assignee=default)
        resolved_workers.append(
            ks.SwarmWorkerSpec(
                profile=resolved or default,
                title=w.title,
                body=w.body,
            )
        )

    verifier = _resolve_assignee(
        plan.verifier_profile, valid_names=valid_names, default_assignee=default
    )
    synthesizer = _resolve_assignee(
        plan.synthesizer_profile, valid_names=valid_names, default_assignee=default
    )

    # 4. Create the swarm
    try:
        with kb.connect_closing() as conn:
            created = ks.create_swarm(
                conn,
                goal=goal,
                workers=resolved_workers,
                verifier_assignee=verifier or default,
                synthesizer_assignee=synthesizer or default,
                created_by=_profile_author(),
            )
    except Exception as exc:
        print(f"/swarm: failed to create swarm — {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    # 5. Report — this is a handoff, not a plan to execute
    if getattr(args, "json", False):
        print(json.dumps(created.as_dict(), indent=2, ensure_ascii=False))
    else:
        print(f"✅ Swarm launched. The goal has been delegated to specialist profiles.")
        print(f"   {plan.rationale}")
        if verifier:
            suff = " (gate before synthesis)" if synthesizer else ""
            print(f"   Verifier: {verifier} — will validate all outputs{suff}")
        if synthesizer:
            print(f"   Synthesizer: {synthesizer} — will assemble final output")
        print(f"   Workers ({len(created.worker_ids)}): {', '.join(created.worker_ids)}")
        print(f"   Track progress: hermes kanban list")

    return 0
