"""The policy enforcement plugin, as installed into the runtime.

This file is **copied verbatim** into each agent's profile as the plugin entry point. It
runs inside a worker process, not inside NOVA, so it imports nothing from NOVA and
nothing from the runtime — only the standard library and its sibling ``_decide`` module,
which is a verbatim copy of :mod:`nova.policy.decide`. (One exception, and it is not on the
enforcement path: :func:`on_session_start` records a refused task on the board through the
runtime's own API, best-effort.)

It implements ``pre_tool_call``, the runtime's documented policy hook: returning
``{"action": "block", "message": ...}`` vetoes a call, and ``{"action": "approve", ...}``
escalates it to the same human gate that guards dangerous shell commands — which fails
closed when no human is present.

**Every failure path here blocks.** A policy that cannot be read, a decision that raises,
a document with an unknown schema: all produce a refusal. A governance control that fails
open is not a control, and a worker that is too restricted fails loudly and visibly,
while one that is silently unrestricted does not.
"""

from __future__ import annotations

import calendar
import json
import os
import re
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

try:  # installed layout: the copied decision module sits beside this file
    from ._decide import (
        ALLOW, ASSIGNEE_ARG, ASSIGNING_TOOL, DENY, FILE_TOOLS, REFUSED_TASK_TOOLS,
        REQUIRE_APPROVAL, Decision, decide, decide_acceptance, decide_assignment, decide_budget,
        decide_paths,
    )
except ImportError:  # in-tree layout, for tests that import this module directly
    from nova.policy.decide import (
        ALLOW, ASSIGNEE_ARG, ASSIGNING_TOOL, DENY, FILE_TOOLS, REFUSED_TASK_TOOLS,
        REQUIRE_APPROVAL, Decision, decide, decide_acceptance, decide_assignment, decide_budget,
        decide_paths,
    )

#: Written beside the profile's configuration by the runtime adapter.
POLICY_FILENAME = "nova-policy.json"

_CACHE: Dict[str, Any] = {}

#: Budget-consuming tool calls made by this process. A worker handles one task per
#: process, so this is a per-run counter; in a long-lived interactive session it is
#: per-session. Named accordingly in the spec (``max_tool_calls_per_run``) so it is never
#: mistaken for a per-day or per-tenant budget, which this runtime cannot enforce.
_CALLS_USED = 0


def _policy_path() -> Path:
    """The compiled policy for the agent this worker is running as.

    Resolved from this file's location — ``<profile>/plugins/nova-policy/__init__.py`` —
    rather than from the environment, so it cannot be pointed elsewhere by a variable and
    is correct even when several agents run concurrently on one host.
    """
    return Path(__file__).resolve().parents[2] / POLICY_FILENAME


def _load_policy() -> Optional[Dict[str, Any]]:
    """Read and cache the compiled policy. None when it is absent or unreadable.

    Cached per process: a worker handles one task and the policy cannot change underneath
    it, so re-reading on every tool call would buy nothing and cost a syscall per call.
    """
    if "policy" in _CACHE:
        return _CACHE["policy"]
    policy: Optional[Dict[str, Any]] = None
    try:
        path = _policy_path()
        if path.is_file():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                policy = loaded
    except (OSError, ValueError):
        policy = None
    _CACHE["policy"] = policy
    return policy


#: A profile name as the runtime allows it. Checked before a name read from the board is
#: used to build a path, so a task row can never point this plugin outside the profiles.
_PROFILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def _sibling_policy(agent: str) -> Optional[Dict[str, Any]]:
    """Another agent's compiled policy, if ``agent`` is a NOVA agent on this host.

    Read from the sibling profile rather than copied into this one, so a change to the
    delegating agent's declaration is in force the moment that agent is re-applied.
    """
    if not agent or ".." in agent or not _PROFILE_NAME.match(agent):
        return None
    path = Path(__file__).resolve().parents[3] / agent / POLICY_FILENAME
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _task_origin(task_id: str) -> tuple[str, bool]:
    """``(authorizer, via_decomposer)`` for a task, read from the board without writing.

    The creator is answerable for an ordinary task. A child the runtime's decomposer made
    carries the decomposer as its creator, which answers for nothing; the card it was split
    from does — its original assignee (the agent that owned the work), else its creator.
    """
    db = os.environ.get("HERMES_KANBAN_DB", "")
    connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=10)
    try:
        row = connection.execute("SELECT created_by FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            raise LookupError(f"task {task_id} is not on the board")
        created = connection.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'created' ORDER BY id LIMIT 1",
            (task_id,),
        ).fetchone()
        root = (json.loads(created[0] or "{}") if created else {}).get("from_decompose_of")
        if not root:
            return row[0] or "", False
        root_row = connection.execute("SELECT created_by FROM tasks WHERE id = ?", (root,)).fetchone()
        root_created = connection.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'created' ORDER BY id LIMIT 1",
            (root,),
        ).fetchone()
        owner = (json.loads(root_created[0] or "{}") if root_created else {}).get("assignee")
        return owner or (root_row[0] if root_row else "") or "", True
    finally:
        connection.close()


def _acceptance(policy: Dict[str, Any]) -> Optional[Decision]:
    """Whether this worker may do the task it was spawned for. None outside a task.

    Decided once per process — a worker is spawned for exactly one task — and it fails
    closed: a task whose origin cannot be read is refused, not assumed to be fine.
    """
    if "acceptance" in _CACHE:
        return _CACHE["acceptance"]
    task_id = os.environ.get("HERMES_KANBAN_TASK", "").strip()
    decision: Optional[Decision] = None
    if task_id:
        try:
            authorizer, via_decomposer = _task_origin(task_id)
            decision = decide_acceptance(
                policy,
                authorizer=authorizer,
                authorizer_policy=_sibling_policy(authorizer),
                via_decomposer=via_decomposer,
            )
        except Exception as exc:  # noqa: BLE001 — an unreadable origin is not a permitted one
            decision = Decision(
                DENY,
                f"could not establish who assigned task {task_id} ({type(exc).__name__}); "
                "refusing rather than working a task no delegation may cover",
                rule="origin-unreadable",
            )
    _CACHE["acceptance"] = decision
    return decision


#: V4A patch headers name the files a ``patch`` in patch mode touches.
_PATCH_HEADER = re.compile(r"^\*\*\* (?:Update|Add|Delete|Move) File: (.+)$", re.MULTILINE)


def _absolute_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    return os.path.realpath(value) if value and os.path.isabs(value) else ""


def _file_scope() -> tuple[list[str], list[str], str]:
    """``(writable roots, readable roots, base for relative paths)``, all resolved.

    Writable: the task's workspace (``HERMES_KANBAN_WORKSPACE``, which the dispatcher also
    makes the terminal cwd). Readable in addition: this agent's own profile. Outside a task
    — a chat session — there is no workspace, so nothing is writable. The profiles
    directory is never writable, workspace or not: the compiled policy and this plugin
    live there, and an agent that could write them could rewrite its own rules.
    """
    workspace = _absolute_env("HERMES_KANBAN_WORKSPACE")
    profile = str(Path(__file__).resolve().parents[2])
    profiles = str(Path(__file__).resolve().parents[3])
    writable = [workspace] if workspace and not (workspace + "/").startswith(profiles + "/") else []
    readable = [profile]
    # The task's own attachments, which a worker reads by absolute path.
    task_id = os.environ.get("HERMES_KANBAN_TASK", "").strip()
    if task_id and _PROFILE_NAME.match(task_id):
        roots = [_absolute_env("HERMES_KANBAN_ATTACHMENTS_ROOT")]
        board_dir = os.path.dirname(_absolute_env("HERMES_KANBAN_DB"))
        if board_dir:
            roots += [os.path.join(board_dir, "kanban", "attachments"), os.path.join(board_dir, "attachments")]
        readable += [os.path.join(root, task_id) for root in roots if root]
    base = workspace or _absolute_env("TERMINAL_CWD") or os.getcwd()
    return writable, readable, base


def _paths_of(tool_name: str, args: Dict[str, Any], base: str) -> list[str]:
    raw = [args.get("path")]
    if tool_name == "search_files" and not args.get("path"):
        raw = ["."]  # the tool's own default: the working directory
    if tool_name == "patch" and isinstance(args.get("patch"), str):
        for named in _PATCH_HEADER.findall(args["patch"]):
            # "*** Move File: a -> b" names two files; both must be in scope.
            raw.extend(part.strip() for part in named.split(" -> "))
    resolved = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            continue
        path = os.path.expanduser(item.strip())
        if not os.path.isabs(path):
            path = os.path.join(base, path)
        resolved.append(os.path.realpath(path))
    return resolved


def _file_decision(tool_name: str, args: Dict[str, Any]) -> Optional[Decision]:
    if tool_name not in FILE_TOOLS:
        return None
    writable, readable, base = _file_scope()
    return decide_paths(
        tool_name, _paths_of(tool_name, args or {}, base),
        writable_roots=writable, readable_roots=readable,
    )


#: How long a month-to-date reading is reused. Spend moves by one model reply at a time,
#: and re-reading every agent's store on every tool call would cost more than it buys.
_SPEND_TTL_SECONDS = 20


def _month_start(now: float) -> float:
    """00:00 UTC on the first of the current month, as a Unix time."""
    t = time.gmtime(now)
    return float(calendar.timegm((t.tm_year, t.tm_mon, 1, 0, 0, 0, 0, 0, 0)))


def _profile_spend(profile: Path, since: float) -> float:
    """Month-to-date spend in one profile's session store, read-only. 0 when there is none.

    The runtime's actual cost where it has one, its estimate otherwise. A session is
    counted in the month it was last active.
    """
    db = profile / "state.db"
    if not db.is_file():
        return 0.0
    connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
    try:
        row = connection.execute(
            "SELECT COALESCE(SUM(CASE WHEN actual_cost_usd > 0 THEN actual_cost_usd "
            "ELSE estimated_cost_usd END), 0) FROM session_model_usage "
            "WHERE COALESCE(last_seen, first_seen, 0) >= ?",
            (since,),
        ).fetchone()
        return float(row[0] or 0.0)
    finally:
        connection.close()


def _budget(policy: Dict[str, Any]) -> Optional[Decision]:
    """The budget decision for this agent now, or None when it has no budget to keep.

    Fails closed: spend that cannot be read while a budget is set is treated as spent, the
    same rule every other check in this file follows.
    """
    agent_limit = policy.get("monthly_budget_usd") or 0
    tenant_limit = policy.get("tenant_monthly_budget_usd") or 0
    if not agent_limit and not tenant_limit:
        return None
    cached = _CACHE.get("budget")
    if cached and time.time() - cached[0] < _SPEND_TTL_SECONDS:
        return cached[1]
    try:
        since = _month_start(time.time())
        profile = Path(__file__).resolve().parents[2]
        agent_spend = _profile_spend(profile, since)
        tenant_spend = 0.0
        if tenant_limit:
            # Every NOVA agent on the host: the profiles that carry a compiled policy.
            for sibling in profile.parent.iterdir():
                if (sibling / POLICY_FILENAME).is_file():
                    tenant_spend += _profile_spend(sibling, since)
        decision = decide_budget(policy, agent_spend=agent_spend, tenant_spend=tenant_spend)
    except Exception as exc:  # noqa: BLE001 — see docstring
        decision = Decision(
            DENY,
            f"could not read this month's spend ({type(exc).__name__}) while a budget is set; "
            "refusing rather than spending blind",
            rule="budget-unreadable",
        )
    _CACHE["budget"] = (time.time(), decision)
    return decision


def _record(policy: Dict[str, Any], decision, tool_name: str) -> None:
    """Append a governance record for a refusal or an escalation.

    Permitted calls are not recorded: they are the overwhelming majority and recording
    them would bury the events a reviewer is actually looking for. What was refused, and
    what needed a human, is the governance question.
    """
    target = (policy or {}).get("audit_log")
    if not target:
        return
    event = {
        "event_id": uuid.uuid4().hex,
        "correlation_id": os.environ.get("HERMES_KANBAN_TASK", "") or "runtime",
        "ts": datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z"),
        "kind": "policy.decision",
        "phase": "record",
        "actor": "nova-policy-plugin",
        "tenant_id": (policy or {}).get("tenant_id", ""),
        "model_visible": False,
        "subject": (policy or {}).get("agent_id", ""),
        "digest": "",
        "detail": {
            "tool": tool_name,
            "effect": decision.effect,
            "reason": decision.reason,
            "rule": decision.rule,
            "action": decision.action,
            "calls_used": _CALLS_USED,
        },
        "error": "",
    }
    try:
        line = (json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        handle = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(handle, line)
        finally:
            os.close(handle)
    except OSError:
        # A governance record we cannot write must not stop the decision being enforced.
        pass


def pre_tool_call(tool_name: str = "", args: Optional[Dict[str, Any]] = None, **_: Any):
    """The runtime's policy hook. Returns a directive, or None to proceed.

    Unrecognised returns are ignored by the runtime, so returning ``None`` for a permitted
    call is the correct way to stay out of the way.
    """
    global _CALLS_USED
    try:
        policy = _load_policy()
        # A task no declared delegation put on this agent's queue is not worked at all:
        # every tool but the ones that read the card and refuse it is closed.
        acceptance = _acceptance(policy) if policy is not None else None
        if acceptance is not None and acceptance.effect == DENY and tool_name not in REFUSED_TASK_TOOLS:
            _record(policy, Decision(DENY, acceptance.reason, tool=tool_name, rule=acceptance.rule), tool_name)
            return {
                "action": "block",
                "message": (
                    f"BLOCKED by NOVA delegation policy: {acceptance.reason}. Do not do this "
                    "task. Call kanban_block with this reason so a person can route it."
                ),
            }
        # Out of budget: everything but the tools that close the task is refused, the same
        # shape as the per-run tool ceiling — an agent must still be able to say it stopped.
        budget = _budget(policy) if policy is not None else None
        if budget is not None and budget.effect == DENY and tool_name not in set(policy.get("baseline") or ()):
            _record(policy, Decision(DENY, budget.reason, tool=tool_name, rule=budget.rule), tool_name)
            return {"action": "block", "message": f"BLOCKED by NOVA budget: {budget.reason}."}
        decision = decide(policy, tool_name, calls_used=_CALLS_USED)
        if tool_name == ASSIGNING_TOOL and decision.effect in (ALLOW, REQUIRE_APPROVAL):
            assignment = decide_assignment(policy, (args or {}).get(ASSIGNEE_ARG))
            if assignment.effect == DENY:
                decision = assignment
        if decision.effect in (ALLOW, REQUIRE_APPROVAL):
            scoped = _file_decision(tool_name, args or {})
            if scoped is not None and scoped.effect == DENY:
                decision = scoped
    except Exception as exc:  # noqa: BLE001 — a policy bug must never permit a call
        return {
            "action": "block",
            "message": (
                f"BLOCKED: the NOVA policy check failed for {tool_name!r} "
                f"({type(exc).__name__}). Refusing rather than proceeding unchecked."
            ),
        }

    # Count only calls that will actually run, and only those the ceiling applies to.
    # Baseline calls are exempt (an agent out of budget must still close its task), and a
    # refused call costs nothing, so neither consumes the budget.
    if decision.effect in (ALLOW, REQUIRE_APPROVAL) and decision.rule != "baseline":
        _CALLS_USED += 1

    if decision.effect == ALLOW:
        return None

    _record(policy or {}, decision, tool_name)

    if decision.effect == REQUIRE_APPROVAL:
        return {
            "action": "approve",
            "message": decision.reason,
            # One allowlist grain per action, so approving "refund" once does not also
            # approve every other escalated action on the same agent.
            "rule_key": f"nova:{decision.action or tool_name}",
        }

    if decision.effect == DENY:
        return {"action": "block", "message": f"BLOCKED by NOVA policy: {decision.reason}"}

    return None


# -- registration -----------------------------------------------------------


def register(ctx: Any) -> None:
    """Wire the policy hook. Called once by the runtime's plugin loader.

    Declaring ``hooks: [pre_tool_call]`` in the manifest is documentation, not wiring: the
    loader calls ``register(ctx)`` and the plugin registers its own callbacks. Without this
    function the loader logs "no register() function", the plugin loads, registers nothing,
    and every tool call proceeds unchecked — a governance control that is installed,
    correct, and never consulted.

    That was the state of this plugin until a live worker run proved it. Unit tests called
    :func:`pre_tool_call` directly, which tested the decision and never tested that anything
    would ever call it; ``tests/platform/test_policy_enforcement.py`` now asserts the
    registration itself.
    """
    ctx.register_hook("pre_tool_call", pre_tool_call)
    ctx.register_hook("on_session_start", on_session_start)


def on_session_start(**_: Any) -> None:
    """Refuse an undelegated task on the board before the agent spends a turn on it.

    The tool hook above is the enforcement and needs nothing from the runtime. This only
    makes the refusal *durable*: the task is blocked with the reason, through the runtime's
    own API — the same call its goal loop makes — so the board shows why, and the
    dispatcher does not re-run a task that will be refused every time. Best-effort by
    design: if it cannot write, the tool hook still stops the work.
    """
    try:
        policy = _load_policy()
        if policy is None or _CACHE.get("refusal_recorded") or not os.environ.get("HERMES_KANBAN_TASK"):
            return
        acceptance = _acceptance(policy)
        refusal = None
        if acceptance is not None and acceptance.effect == DENY:
            refusal = ("NOVA delegation policy", acceptance, "capability")
        else:
            budget = _budget(policy)
            if budget is not None and budget.effect == DENY:
                refusal = ("NOVA budget", budget, "needs_input")
        if refusal is None:
            return
        label, acceptance, kind = refusal
        _CACHE["refusal_recorded"] = True
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc

        task_id = os.environ["HERMES_KANBAN_TASK"]
        raw_run = os.environ.get("HERMES_KANBAN_RUN_ID", "").strip()
        with kbc.connect_closing() as connection:
            kb.block_task(
                connection, task_id,
                reason=f"{label}: {acceptance.reason}",
                kind=kind,
                expected_run_id=int(raw_run) if raw_run.isdigit() else None,
            )
        _record(policy, Decision(DENY, acceptance.reason, tool="(task)", rule=acceptance.rule), "(task)")
    except Exception:  # noqa: BLE001 — recording the refusal must never break the worker
        pass
