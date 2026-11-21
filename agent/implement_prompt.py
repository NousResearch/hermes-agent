"""``/implement`` — execute a saved ``/plan`` file as the next normal turn.

Inspired by the approved-plan handoff in ChatGPT Work / Codex ("Yes, clear context and
implement", openai/codex#60763): the plan the user approved, not the chat that produced
it, is what the implementer works from. Hermes plans already live on disk under
``.hermes/plans/``, so the handoff is the file: ``/implement`` freezes the approved plan's
text into the turn (what you reviewed is what runs, even if the file changes later), and
``/new`` followed by ``/implement`` is the clean-context variant.

Same pattern as ``/plan``/``/learn``/``/init``: no engine, no model-tool footprint, the
prompt is a normal user turn (system prompt and history untouched, prompt-cache safe).
A plan the host cannot read (remote terminal backends keep it on the backend) is not an
error: the prompt tells the agent to open it with its own file tools.
"""

from __future__ import annotations

import os
from pathlib import Path

PLANS_DIR = os.path.join(".hermes", "plans")
# Larger plans are referenced, not inlined: the agent reads them with read_file in pages
# instead of the turn carrying a huge user message (nothing is dropped either way).
_INLINE_MAX_CHARS = 60_000

_IMPLEMENT_RULES = """\
For this turn, you are in IMPLEMENT MODE: carry out the approved plan.

- The plan is the user's approved spec. Follow its steps in order; do not
  re-plan, widen scope, or skip its verification steps.
- Work step by step. After each step, run the step's own verification
  command and check the result before moving on.
- If a step is impossible as written (a path moved, a command fails for a
  reason the plan did not anticipate), make the smallest change that keeps
  the plan's intent, and say what you changed and why.
- Stop and ask the user before anything the plan does not cover that is
  destructive or leaves the workspace (deleting data, pushing, deploying,
  sending messages, spending money).
- Finish with a short report: each step done or not, the verification
  output that proves it, and anything left open.
"""


def _plans_dir(cwd: str) -> Path:
    return Path(cwd) / PLANS_DIR


def find_plan(cwd: str, ref: str = "") -> Path | None:
    """The plan ``ref`` names, or the newest saved plan when ``ref`` is empty.

    ``ref`` may be a path (absolute, or relative to ``cwd``) or a fragment of a plan's file
    name (``/implement oauth`` picks the newest plan whose name contains "oauth").
    """
    ref = (ref or "").strip().strip("\"'")
    if ref:
        direct = Path(os.path.expanduser(ref))
        direct = direct if direct.is_absolute() else Path(cwd) / direct
        if direct.is_file():
            return direct
    try:
        plans = [p for p in _plans_dir(cwd).glob("*.md") if p.is_file()]
    except OSError:
        return None
    if ref:
        needle = ref.lower()
        plans = [p for p in plans if needle in p.name.lower()]
    # Plan names start with a sortable timestamp; mtime breaks ties and covers hand-named files.
    return max(plans, key=lambda p: (p.stat().st_mtime, p.name), default=None)


def _display_path(path: Path, cwd: str) -> str:
    try:
        return str(path.resolve().relative_to(Path(cwd).resolve()))
    except ValueError:
        return str(path)


def build_implement_prompt_for_session(ref: str = "", cwd: str | None = None,
                                       session_key: str | None = None) -> tuple[str, str]:
    """Every surface's entry point: plans resolve against the SESSION's active directory (the
    ``/init`` ladder in hermes_cli/init_command.py), never the backend's launch directory."""
    if not (cwd and os.path.isdir(cwd)):
        from hermes_cli.init_command import _resolve_session_cwd
        cwd = _resolve_session_cwd(session_key)
    return build_implement_prompt(ref, cwd)


def build_implement_prompt(ref: str = "", cwd: str | None = None) -> tuple[str, str]:
    """``(prompt, plan_label)`` for ``/implement [plan]``; ``plan_label`` is "" when the host
    found no plan (the prompt then asks the agent to locate it with its own tools)."""
    cwd = cwd or os.getcwd()
    plan = find_plan(cwd, ref)
    text = ""
    if plan is not None:
        try:
            text = plan.read_text(encoding="utf-8-sig", errors="replace")
        except OSError:
            plan = None
    head = "[/implement — implement mode]\n\n" + _IMPLEMENT_RULES + "\n"
    if plan is None:
        target = f"the plan matching {ref!r}" if ref.strip() else "the newest plan"
        return head + (
            f"Find {target} under `{PLANS_DIR}/` in the active workspace with your file "
            "tools (plan names start with a timestamp; newest is last), read all of it, "
            "then implement it. If there is no saved plan, say so and suggest `/plan <task>` "
            "instead of improvising one.\n"
        ), ""
    label = _display_path(plan, cwd)
    if len(text) > _INLINE_MAX_CHARS:
        return head + (
            f"Approved plan: `{plan}` ({len(text):,} characters). Read the whole file with "
            "read_file (page through it) before starting, then implement it.\n"
        ), label
    return head + (
        f"Approved plan (`{plan}`), as the user approved it:\n\n"
        f"<<<APPROVED_PLAN\n{text.rstrip()}\nAPPROVED_PLAN\n\n"
        "Work from this text. You may tick off completed steps in the plan file as you go.\n"
    ), label
