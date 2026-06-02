"""Caduceus — Hermes-native deep-planning mode.

Caduceus is Hermes's native answer to Claude Code's "UltraCode". When it's on,
the agent works like the Devin CLI: it raises reasoning effort and drives a
*visible, todo-driven plan* — decompose the task with the ``todo`` tool, execute
one step at a time, mark progress live, delegate parallelizable grunt work to
subagents, and verify before claiming done. The heavy "Loom" dynamic-workflow
engine (see :mod:`agent.workflow`) is the *explicit* escalation for parallel
fan-out — reached only when the user says "workflow"/asks to orchestrate, never
automatically.

A single toggle (``/caduceus on|off`` or the desktop switch) flips the mode;
model tiers, effort, and budget have sensible defaults and are power-user-only
overrides in the ``caduceus:`` config section (not surfaced as command knobs).

This module is the single source of truth for:

* the per-session :class:`CaduceusState` (mode flag, tiers, budget, reminder
  bookkeeping);
* the model-visible **prompt stack** — the standing reminder, the
  enter/sparse/exit lifecycle reminders, the workflow-keyword reminder, and the
  full ``Workflow`` tool description (ported verbatim where model-visible,
  adapted only for Python/Hermes tool names);
* the abstract ``effort`` → provider ``reasoning_config`` mapping; and
* orchestrator/worker tier resolution used by ``delegate_task`` and the Loom.

It is intentionally self-contained so Caduceus stays an *additive* fork: hot
files (run_agent, conversation_loop, system_prompt, delegate_tool) touch it only
through small, well-named hooks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Model-visible prompt strings
#
# These are ported from the UltraCode reverse-engineering (claude-ultracode-re)
# and rebranded to "Caduceus". The wording that *drives behavior* is preserved
# verbatim; only the proper noun and JS→Python tool details change.
# ---------------------------------------------------------------------------

# Standing reminder, injected into the system prompt's stable tier while the
# mode is active (mirrors UltraCode's placement right after the skills
# inventory). This is the line that sets Caduceus's default operating style:
# a Devin-CLI-grade, todo-driven planning loop — NOT "always run a workflow".
STANDING_REMINDER = (
    "Caduceus is on. Work like a senior engineer driving a visible plan, "
    "optimizing for the most correct, complete result over the fastest or "
    "cheapest answer.\n"
    "RIGHT-SIZE THE PLAN: Match the plan to the task. A simple 1-2 step ask — just "
    "do it, fast, no to-do list. Genuinely multi-step or hard work (3+ steps, "
    "multiple asks, multi-file, ambiguous, or error-prone) — your FIRST action is "
    "a `todo` plan, decomposed as finely as the difficulty actually warrants: a "
    "light task gets a short list, a hard one gets a deep, granular breakdown. "
    "Capture new instructions as todos immediately.\n"
    "ONE STEP AT A TIME: Keep exactly one item in_progress — mark it in_progress "
    "in its own `todo` write BEFORE you work it, do the work, then in the next "
    "write mark it completed and set the next item in_progress. Never jump an item "
    "from pending straight to completed, and never collapse several completions "
    "into one write. (The to-do tracks logical steps, not individual tool calls.)\n"
    "PARALLELIZE FOR SPEED: Don't serialize work that can run side by side. Issue "
    "independent tool calls together (they execute concurrently), and batch "
    "independent subtasks into ONE delegate_task call so the workers run in "
    "parallel — not one after another.\n"
    "COMPLETION HONESTY: Mark a step completed only when it is fully, verifiably "
    "done. If it is blocked, failing, or partial, keep it in_progress and add a "
    "new todo for what must be resolved — never mark done on a failing check or a "
    "partial result.\n"
    "VERIFY: Before claiming anything done, verify it (lint / typecheck / build / "
    "tests, or re-read the output, as the change warrants). When the project has "
    "test infrastructure, prefer writing a failing test first, then making it pass.\n"
    "ESCALATE: For LARGE parallel fan-out (many files / sources / checks at once), "
    "or when the user says \"workflow\" or asks to orchestrate, use the Workflow "
    "tool — the Loom runs many workers concurrently. Be proactive on the follow-ups "
    "the task implies, but don't take large unrequested actions out of nowhere."
)

# Lifecycle reminders, injected as per-turn meta notes (cache-friendly: they go
# into the current user message, never the cached system-prompt prefix).
ENTER_REMINDER_FULL = STANDING_REMINDER
ENTER_REMINDER_SPARSE = (
    "Caduceus is still on — keep driving the `todo` plan one step at a time: "
    "exactly one item in_progress, mark it in_progress before you work on it and "
    "completed right after (don't batch transitions), and verify before claiming "
    "done. Use the Workflow tool only if asked to fan out."
)
SPARSE_REMINDER = ENTER_REMINDER_SPARSE
EXIT_REMINDER = (
    "Caduceus is off — the deep plan/verify discipline relaxes to normal and the "
    "Workflow tool's standard opt-in rule applies again."
)

# Separate opt-in path: the user literally typed "workflow"/"workflows".
WORKFLOW_KEYWORD_REMINDER = (
    'The user included the keyword "workflow" or "workflows", which means you '
    "should use the Workflow tool to fulfill their request."
)


def wrap_reminder(text: str) -> str:
    """Wrap a reminder as a model-visible meta block."""
    return f"<system-reminder>\n{text}\n</system-reminder>"


# The full ``Workflow`` tool description. Ported from
# claude-ultracode-re/evidence/WORKFLOW_TOOL_PROMPT.md (the captured
# ``Workflow.description``) with these edits: the script body is restricted
# Python (async ``main()``) rather than JavaScript; "subagent" is a Hermes
# ``delegate_task`` child running on the worker tier; tool reachability is via
# Hermes toolsets; and the standing-opt-in clause is rebranded to **Caduceus**.
WORKFLOW_TOOL_DESCRIPTION = """\
Execute a workflow script that orchestrates many subagents deterministically. Returns when the workflow finishes; progress streams live to the Orchestration Theater. Each subagent is a Hermes delegate child running on the worker tier.

OPT-IN: only call this when the user opted into multi-agent orchestration — they said "workflow"/"workflows", they asked to "fan out / orchestrate with subagents", or a skill told you to. (Caduceus being on is NOT such an opt-in — see below.) Otherwise answer directly, plan with the `todo` tool, or use delegate_task for a single subagent.

**Caduceus.** Caduceus being on does NOT mean "always run a workflow." Under Caduceus you plan with the `todo` tool and dispatch individual subagents with delegate_task; that handles most work. Use THIS tool only when the user explicitly opts into multi-agent orchestration — they said "workflow"/"workflows", asked to "fan out / orchestrate with subagents", or the task clearly needs large parallel orchestration (e.g. "audit every module", a big map-reduce). When you do author one, aim for the most exhaustive, correct answer and scale the fan-out to the task (a simple ask = a small workflow; "audit thoroughly" = larger fan-out + adversarial verify). Multi-phase work = several workflows in sequence, reading each result before the next. Don't over-deliberate before authoring: scout briefly only if you must, then write the script and call this tool.

SCRIPTS ARE PYTHON, NOT JAVASCRIPT. Define `meta` (a pure literal) and an async `main()`. Use the injected globals directly — do NOT import them, do NOT wrap the script in markdown fences, do NOT use const/let/var or `=>` arrows.

    meta = {
        "name": "review-changes",
        "description": "Review changed files, verify each finding",
        "phases": [{"title": "Review"}, {"title": "Verify"}],
    }
    FINDINGS = {"type": "object", "properties": {"findings": {"type": "array"}}, "required": ["findings"]}
    VERDICT = {"type": "object", "properties": {"isReal": {"type": "boolean"}}, "required": ["isReal"]}
    DIMENSIONS = [{"key": "bugs", "prompt": "Find bugs"}, {"key": "perf", "prompt": "Find perf issues"}]

    async def main():
        async def review(d):
            return await agent(d["prompt"], label="review:" + d["key"], phase="Review", schema=FINDINGS)
        async def verify(rev, d, i):
            async def vone(f):
                v = await agent("Adversarially verify: " + f["title"], phase="Verify", schema=VERDICT)
                return {**f, "verdict": v}
            return await parallel([(lambda f=f: vone(f)) for f in rev["findings"]])
        results = await pipeline(DIMENSIONS, review, verify)
        confirmed = [f for r in results if r for f in r if f and f["verdict"]["isReal"]]
        return {"confirmed": confirmed}

Injected globals (no import):
- agent(prompt, *, label=None, phase=None, schema=None, model=None, isolation=None, agent_type=None) -> await it. Without schema returns the subagent's final text (str). With schema (a JSON Schema dict) it returns a validated object (the subagent is forced to return JSON; it retries on mismatch). Returns None on failure — filter with `[x for x in results if x]`. Omit `model` (inherits the worker tier); set it only to escalate one hard leaf.
- pipeline(items, *stages) -> await -> list. Each item flows through all stages independently, NO barrier (item A in stage 3 while B in stage 1). THE DEFAULT for multi-stage work. Each stage is called (prev_result, original_item, index) — take only the args you need. A raising stage drops that item to None.
- parallel(thunks) -> await -> list. Runs zero-arg callables concurrently; BARRIER (awaits all). A failing thunk resolves to None (never raises) — filter None before use. Use ONLY when stage N genuinely needs ALL of stage N-1 (dedup/merge, early-exit on zero, cross-item comparison).
- phase(title), log(message): narrate the Theater.
- budget: .total (int|None), .spent(), .remaining(). HARD ceiling — agent() raises once spent>=total. Loop with `while budget.total and budget.remaining() > 50_000: ...`.
- args: the verbatim `args` input (None if absent).
- workflow(name_or_ref, args=None) -> await: run a saved workflow / {"scriptPath": ...} inline (one level deep).
- Quality-pattern helpers (all async): adversarial_verify(claim, n=3), perspective_verify(claim, lenses), judge_panel(attempts, criteria), loop_until_dry(finder, k=2, key=None), multimodal_sweep(searches), completeness_critic(state).

DEFAULT TO pipeline(); reach for parallel() only for a real barrier. Concurrency is capped per workflow; total agents are capped as a runaway backstop. Subagents reach the session's tools via Hermes toolsets and are told their final text IS the return value, so they return raw data.

Standard pure builtins + the `json` and `math` modules are available. `time`, `random`, and wall-clock are NOT (they break resume) — pass timestamps via `args`, stamp after return, vary randomness by index.

To iterate or resume: the result includes a runId and the persisted scriptPath. Edit that file and re-invoke with {"scriptPath": ..., "resumeFromRunId": ...} — unchanged agent() calls return cached results instantly; the first edited/new call runs live. Same script + same args = 100% cache hit."""


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

_UNSET = object()


@dataclass
class CaduceusState:
    """Per-session Caduceus state, carried on the AIAgent.

    Defaults are the "off" state. :func:`state_from_config` seeds it from the
    ``caduceus:`` config section; the /caduceus command and the desktop toggle
    mutate it at runtime.
    """

    enabled: bool = False
    effort: str = "high"
    apply_effort_to_worker: bool = False
    orchestrator: Dict[str, str] = field(default_factory=lambda: {"provider": "", "model": ""})
    worker: Dict[str, str] = field(default_factory=lambda: {"provider": "", "model": ""})
    budget_tokens: Optional[int] = None

    # Workflow/Loom knobs (copied from config for convenience).
    workflow: Dict[str, Any] = field(default_factory=dict)

    # Reminder lifecycle config + bookkeeping.
    enter_style: str = "full"
    turns_between_maintenance: int = 8
    # Set True when the mode flips on; the next turn emits the enter reminder.
    _enter_pending: bool = False
    # Set True when the mode flips off; the next turn emits the exit reminder.
    _exit_pending: bool = False
    # User-turn index of the last maintenance reminder.
    _last_maintenance_turn: int = 0

    # Saved reasoning_config so exiting the mode restores the prior effort.
    _saved_reasoning_config: Any = _UNSET

    # ---- mutation -----------------------------------------------------
    def activate(self) -> None:
        if not self.enabled:
            self.enabled = True
            self._enter_pending = True
            self._exit_pending = False

    def deactivate(self) -> None:
        if self.enabled:
            self.enabled = False
            self._exit_pending = True
            self._enter_pending = False

    # ---- introspection ------------------------------------------------
    def is_split(self) -> bool:
        """True when a distinct worker tier is configured (orchestrator != worker)."""
        w = self.worker or {}
        return bool((w.get("provider") or "").strip() or (w.get("model") or "").strip())

    def summary(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "effort": self.effort,
            "orchestrator": dict(self.orchestrator),
            "worker": dict(self.worker),
            "budget": self.budget_tokens,
            "split": self.is_split(),
        }


def state_from_config(cfg: Optional[Dict[str, Any]]) -> CaduceusState:
    """Build a :class:`CaduceusState` from the ``caduceus:`` config section."""
    c = dict((cfg or {}).get("caduceus") or {}) if cfg else {}
    orch = dict(c.get("orchestrator") or {})
    work = dict(c.get("worker") or {})
    wf = dict(c.get("workflow") or {})
    reminders = dict(c.get("reminders") or {})
    budget = wf.get("default_budget_tokens")
    try:
        budget = int(budget) if budget not in (None, "", "null") else None
    except (TypeError, ValueError):
        budget = None
    return CaduceusState(
        enabled=bool(c.get("enabled", False)),
        effort=str(c.get("effort") or "high"),
        apply_effort_to_worker=bool(c.get("apply_effort_to_worker", False)),
        orchestrator={"provider": str(orch.get("provider") or ""), "model": str(orch.get("model") or "")},
        worker={"provider": str(work.get("provider") or ""), "model": str(work.get("model") or "")},
        budget_tokens=budget,
        workflow=wf,
        enter_style=str(reminders.get("enter") or "full"),
        turns_between_maintenance=int(reminders.get("turns_between_maintenance") or 8),
    )


def get_state(agent: Any) -> Optional[CaduceusState]:
    """Return the agent's CaduceusState, or None if not initialised."""
    return getattr(agent, "caduceus", None)


def is_active(agent: Any) -> bool:
    st = get_state(agent)
    return bool(st and st.enabled)


# ---------------------------------------------------------------------------
# Effort mapping
# ---------------------------------------------------------------------------

def resolve_effort_config(effort: str) -> Optional[Dict[str, Any]]:
    """Map an abstract Caduceus ``effort`` to a Hermes ``reasoning_config``.

    Hermes already threads ``reasoning_config={"enabled": True, "effort": ...}``
    into every provider transport and maps each effort level to the provider's
    real knob (Anthropic adaptive-thinking + effort, OpenAI/Codex effort,
    others best-effort). ``xhigh`` is a first-class Hermes effort, so this is a
    thin wrapper around :func:`hermes_constants.parse_reasoning_effort` that
    degrades gracefully on unknown values.
    """
    try:
        from hermes_constants import parse_reasoning_effort
    except Exception:
        return None
    parsed = parse_reasoning_effort(effort or "")
    # Unknown effort → don't force anything (keep the session's current config).
    return parsed


# ---------------------------------------------------------------------------
# Reminder lifecycle scheduler
# ---------------------------------------------------------------------------

def compute_turn_reminder(state: Optional[CaduceusState], user_turn_count: int) -> Optional[str]:
    """Return the meta-reminder text to inject for this turn, or None.

    Lifecycle (UltraCode parity):
      * **exit**  — emitted once on the first turn after deactivation;
      * **enter** — emitted once on the first turn after activation
        (full or sparse per ``reminders.enter``);
      * **sparse maintenance** — every ``turns_between_maintenance`` user turns
        while the mode stays on.
    """
    if state is None:
        return None
    # Exit takes priority — it can fire even while disabled.
    if state._exit_pending:
        state._exit_pending = False
        return EXIT_REMINDER
    if not state.enabled:
        return None
    if state._enter_pending:
        state._enter_pending = False
        state._last_maintenance_turn = user_turn_count
        return ENTER_REMINDER_FULL if state.enter_style != "sparse" else ENTER_REMINDER_SPARSE
    cadence = max(1, int(state.turns_between_maintenance or 8))
    if user_turn_count - state._last_maintenance_turn >= cadence:
        state._last_maintenance_turn = user_turn_count
        return SPARSE_REMINDER
    return None


def message_has_workflow_keyword(text: Any) -> bool:
    """True if the user's message contains a standalone 'workflow(s)' keyword."""
    if not isinstance(text, str) or not text:
        return False
    low = text.lower()
    return "workflow" in low  # substring is fine; "workflows" matches too


# ---------------------------------------------------------------------------
# Prompt-stack injection helpers (called from system_prompt / conversation_loop)
# ---------------------------------------------------------------------------

def standing_reminder_for_prompt(agent: Any) -> Optional[str]:
    """Return the standing reminder for the system prompt, or None.

    Injected when the mode is active. The standing behavior is now todo-driven
    planning, so it only needs the (core, near-always-present) ``todo`` tool;
    the Workflow tool is referenced merely as an explicit-opt-in escalation, so
    its absence does not void the reminder. We still skip the reminder if we
    have a tool list and ``todo`` was explicitly disabled.
    """
    if not is_active(agent):
        return None
    valid = getattr(agent, "valid_tool_names", None) or ()
    if valid and "todo" not in valid:
        return None
    return STANDING_REMINDER


# ---------------------------------------------------------------------------
# Tier resolution (used by delegate_task for role-aware tiering)
# ---------------------------------------------------------------------------

def tier_for_role(state: Optional[CaduceusState], role: str) -> Optional[Dict[str, str]]:
    """Return the {provider, model} tier for a delegate role under Caduceus.

    Role-aware tiering (the "smartest integration"):
      * ``orchestrator`` (a nested role='orchestrator' delegate, which itself
        plans + delegates) → the **orchestrator** (heavy) tier;
      * everything else (leaves + plain delegate_task) → the **worker** (fast)
        tier.

    Returns None when the mode is off or the relevant tier is unset (so the
    child inherits the parent's model unchanged).
    """
    if state is None or not state.enabled:
        return None
    if role == "orchestrator":
        tier = state.orchestrator
    else:
        tier = state.worker
        # Solo: no distinct worker configured → fall back to orchestrator tier
        # if one is set, else inherit parent.
        if not ((tier.get("provider") or "").strip() or (tier.get("model") or "").strip()):
            tier = state.orchestrator
    if not tier:
        return None
    provider = (tier.get("provider") or "").strip()
    model = (tier.get("model") or "").strip()
    if not provider and not model:
        return None
    return {"provider": provider, "model": model}


def resolve_concurrency(state: Optional[CaduceusState]) -> int:
    """Resolve the workflow concurrency cap (handles the 'auto' sentinel)."""
    default = min(16, max(2, (os.cpu_count() or 4) - 2))
    if state is None:
        return default
    raw = (state.workflow or {}).get("max_concurrency", "auto")
    if isinstance(raw, str):
        if raw.strip().lower() == "auto":
            return default
        try:
            raw = int(raw)
        except ValueError:
            return default
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return default
