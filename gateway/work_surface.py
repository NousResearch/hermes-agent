"""Plan 067 Phase 1 — the ``/work`` deterministic dispatch surface (pure logic).

The ``/work`` Slack command lets the operator launch autonomous work WITHOUT
relying on the agent's LLM tool-selection (which — see Plan 067 reason #2 —
mis-routes a ``plan_lifecycle`` goal to ``code_execution`` because a detailed
goal reads like "code this now"). Instead, ``/work`` shows a preset menu + a
guided modal whose ``view_submission`` calls ``plan_lifecycle``'s
``_post_to_sensor`` DIRECTLY.

This module holds the **pure** pieces — preset loading, Block Kit builders, and
view-submission parsing — so they are unit-testable without a Slack client. The
async handlers that do I/O (open the modal, POST to the sensor, post results)
live on the SlackAdapter in ``gateway/platforms/slack.py``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

# Block Kit action / callback identifiers (shared with the slack.py handlers).
PRESET_ACTION_PREFIX = "work_preset_"  # + <preset id>; matched by a regex listener
CUSTOM_ACTION_ID = "work_custom"
SUBMIT_CALLBACK_ID = "work_submit"

# Modal input block_ids; the input element action_id is "<block_id>_input".
_FIELD_BLOCKS = ("goal", "operator", "repos", "budget", "success", "constraints")

# Co-located with this module (the repo's top-level data/ dir is gitignored).
_PRESETS_PATH = Path(__file__).resolve().parent / "plan_lifecycle_presets.json"


class WorkValidationError(ValueError):
    """A modal field failed validation — carries the offending block_id.

    The view-submission handler turns this into a Slack ``response_action:
    errors`` so the operator sees the message inline on the field.
    """

    def __init__(self, block_id: str, message: str) -> None:
        super().__init__(message)
        self.block_id = block_id
        self.message = message


# ---------------------------------------------------------------------------
# Preset loading
# ---------------------------------------------------------------------------


def load_presets(path: Optional[Path] = None) -> list[dict[str, Any]]:
    """Return the vendored preset list (empty list if the file is absent/bad)."""
    p = path or _PRESETS_PATH
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    presets = data.get("presets", [])
    return presets if isinstance(presets, list) else []


def find_preset(preset_id: str, presets: Optional[list[dict[str, Any]]] = None) -> Optional[dict[str, Any]]:
    """Look up a preset by id."""
    for preset in presets if presets is not None else load_presets():
        if preset.get("id") == preset_id:
            return preset
    return None


# ---------------------------------------------------------------------------
# Block Kit: the /work menu
# ---------------------------------------------------------------------------


def build_menu_blocks(presets: Optional[list[dict[str, Any]]] = None) -> list[dict[str, Any]]:
    """Build the ``/work`` menu: one button per preset + a Custom-goal button.

    Slack caps an ``actions`` block at 5 elements, so presets are chunked across
    multiple actions blocks; the Custom-goal button gets its own trailing block.
    """
    presets = presets if presets is not None else load_presets()
    blocks: list[dict[str, Any]] = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    "*Launch autonomous work* — pick a preset to review + dispatch, "
                    "or start from a custom goal. Each opens a form you confirm before "
                    "anything runs."
                ),
            },
        }
    ]

    preset_buttons = [
        {
            "type": "button",
            "text": {"type": "plain_text", "text": (p.get("title") or p.get("id") or "preset")[:75]},
            "action_id": f"{PRESET_ACTION_PREFIX}{p.get('id')}",
            "value": str(p.get("id", "")),
        }
        for p in presets
    ]
    # Chunk into actions blocks of <=5 (Slack's per-block element limit).
    for i in range(0, len(preset_buttons), 5):
        blocks.append({"type": "actions", "elements": preset_buttons[i : i + 5]})

    blocks.append(
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "✏️  Custom goal…"},
                    "style": "primary",
                    "action_id": CUSTOM_ACTION_ID,
                    "value": "custom",
                }
            ],
        }
    )
    return blocks


# ---------------------------------------------------------------------------
# Block Kit: the guided dispatch modal
# ---------------------------------------------------------------------------


def _input_block(
    block_id: str,
    label: str,
    *,
    initial: str = "",
    multiline: bool = False,
    hint: str = "",
    optional: bool = False,
) -> dict[str, Any]:
    element: dict[str, Any] = {
        "type": "plain_text_input",
        "action_id": f"{block_id}_input",
        "multiline": multiline,
    }
    if initial:
        element["initial_value"] = initial
    block: dict[str, Any] = {
        "type": "input",
        "block_id": block_id,
        "label": {"type": "plain_text", "text": label},
        "element": element,
        "optional": optional,
    }
    if hint:
        block["hint"] = {"type": "plain_text", "text": hint[:150]}
    return block


def build_modal_view(preset: Optional[dict[str, Any]], *, channel_id: str = "") -> dict[str, Any]:
    """Build the guided dispatch modal, pre-filled from *preset* (or empty).

    ``private_metadata`` carries the originating channel + preset id so the
    ``view_submission`` handler can post the result back to the right place.
    """
    preset = preset or {}
    title = preset.get("title") or "Custom goal"
    repos_initial = ", ".join(preset.get("repos", []) or [])
    budget = preset.get("budget_usd")
    budget_initial = "" if budget is None else (str(int(budget)) if float(budget).is_integer() else str(budget))
    success_initial = "\n".join(preset.get("success_criteria", []) or [])
    constraints_initial = "\n".join(preset.get("constraints", []) or [])

    blocks = [
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": (
                        f"Dispatching *{title}* — review and confirm. This goes straight to the "
                        "orchestrator (no LLM tool-pick)."
                    ),
                }
            ],
        },
        _input_block(
            "goal", "Goal", initial=preset.get("goal", ""), multiline=True,
            hint="What to build, concretely. The orchestrator decomposes this into phases.",
        ),
        _input_block(
            "operator", "Operator", initial=preset.get("operator", "blake"),
            hint="Who is accountable for this run (e.g. blake).",
        ),
        _input_block(
            "repos", "Repos", initial=repos_initial,
            hint="Comma- or newline-separated. The first is the primary repo.",
        ),
        _input_block(
            "budget", "Budget (USD)", initial=budget_initial,
            hint="Approved spend ceiling. The drain stalls without one.",
        ),
        _input_block(
            "success", "Success criteria", initial=success_initial, multiline=True,
            hint="One per line. What 'done' means.",
        ),
        _input_block(
            "constraints", "Constraints", initial=constraints_initial, multiline=True,
            hint="One per line. Guardrails (e.g. Python only, no terraform).",
        ),
    ]

    return {
        "type": "modal",
        "callback_id": SUBMIT_CALLBACK_ID,
        "title": {"type": "plain_text", "text": "Launch work"},
        "submit": {"type": "plain_text", "text": "Dispatch"},
        "close": {"type": "plain_text", "text": "Cancel"},
        "private_metadata": json.dumps(
            {"channel_id": channel_id, "preset_id": preset.get("id", "")}
        ),
        "blocks": blocks,
    }


# ---------------------------------------------------------------------------
# view_submission → sensor payload
# ---------------------------------------------------------------------------


def _read_field(view_values: dict[str, Any], block_id: str) -> str:
    block = view_values.get(block_id, {})
    element = block.get(f"{block_id}_input", {})
    return (element.get("value") or "").strip()


def _split_lines(raw: str) -> list[str]:
    """Split a textarea / comma list into a clean list of items."""
    parts: list[str] = []
    for chunk in raw.replace(",", "\n").splitlines():
        item = chunk.strip().lstrip("-•").strip()
        if item:
            parts.append(item)
    return parts


def parse_view_submission(view: dict[str, Any], *, plan_id: str) -> dict[str, Any]:
    """Turn a modal ``view`` into a ``/plan/decompose`` payload.

    Mirrors the exact payload shape ``tools.plan_lifecycle_tool.plan_lifecycle``
    POSTs (goal, plan_id, repo, ref, budget_usd, repos, success_criteria,
    constraints, operator). Raises :class:`WorkValidationError` on bad input so
    the handler can surface a per-field error.
    """
    values = (view.get("state", {}) or {}).get("values", {}) or {}

    goal = _read_field(values, "goal")
    if not goal:
        raise WorkValidationError("goal", "A goal is required.")

    operator = _read_field(values, "operator") or "unknown"

    repos = _split_lines(_read_field(values, "repos"))
    if not repos:
        raise WorkValidationError("repos", "At least one repo is required.")

    budget_raw = _read_field(values, "budget")
    try:
        budget_usd = float(budget_raw)
    except ValueError:
        raise WorkValidationError("budget", "Budget must be a number (USD).") from None
    if budget_usd < 0:
        raise WorkValidationError("budget", "Budget cannot be negative.")

    success_criteria = _split_lines(_read_field(values, "success")) or ["none"]
    constraints = _split_lines(_read_field(values, "constraints")) or ["none"]

    return {
        "goal": goal,
        "plan_id": plan_id,
        "repo": repos[0],
        "ref": "main",
        "budget_usd": budget_usd,
        "repos": repos,
        "success_criteria": success_criteria,
        "constraints": constraints,
        "operator": operator,
    }


def read_private_metadata(view: dict[str, Any]) -> dict[str, str]:
    try:
        meta = json.loads(view.get("private_metadata") or "{}")
    except json.JSONDecodeError:
        return {}
    return meta if isinstance(meta, dict) else {}


# ---------------------------------------------------------------------------
# Result + help formatting
# ---------------------------------------------------------------------------


def format_dispatch_result(plan_id: str, response_body: dict[str, Any]) -> str:
    """Slack-friendly summary of a dispatch.

    067 Phase 2: the sensor now decomposes asynchronously on the drain worker
    and returns immediately ({accepted, message, …}) — so the phase issues
    aren't created yet at response time. Prefer the sensor's own message when
    the response is the async ``accepted`` shape; fall back to the legacy
    inline summary (issues created synchronously) otherwise.
    """
    if response_body.get("accepted"):
        msg = response_body.get("message") or (
            f"Plan *{plan_id}* accepted — decomposing on the worker; "
            "the phase issues will appear in Linear shortly."
        )
        return f":rocket: {msg}"

    created = response_body.get("created_issue_ids", []) or []
    phases = response_body.get("phase_ids", []) or []
    signaled = response_body.get("signaled", False)
    issues = ", ".join(created) if created else "(none returned)"
    phases_s = ", ".join(str(p) for p in phases) if phases else "(none)"
    return (
        f":rocket: Plan *{plan_id}* dispatched to the orchestrator.\n"
        f"• Issues created: {issues}\n"
        f"• Phases: {phases_s}\n"
        f"• Drain signaled: {signaled}"
    )


def build_help_text(presets: Optional[list[dict[str, Any]]] = None) -> str:
    """The ``/work help`` text: command set + preset catalogue."""
    presets = presets if presets is not None else load_presets()
    lines = [
        "*/work* — launch autonomous work from Slack (deterministic; bypasses the agent).",
        "",
        "*Commands*",
        "• `/work` — show the preset menu + a Custom-goal button.",
        "• `/work help` — this message.",
        "",
        "*How it works:* pick a preset (or Custom goal) → a form opens pre-filled → "
        "confirm/edit → *Dispatch* posts it straight to the orchestrator sensor, which "
        "decomposes it into Linear phases and signals the drain (PR-not-merge).",
        "",
        "*Presets*",
    ]
    if presets:
        for p in presets:
            sub = f" — {p['subtitle']}" if p.get("subtitle") else ""
            lines.append(f"• *{p.get('title', p.get('id'))}*{sub}")
    else:
        lines.append("• (none loaded)")
    lines.append("")
    lines.append("To steer a running drain, use `/resume` and `/skip`.")
    return "\n".join(lines)
