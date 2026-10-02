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

That gate only reaches a person in a chat session. A task worker runs unattended, and
there the runtime refuses every escalation on the spot (``approvals.single_query_mode``),
so in task work this plugin asks on the board instead: it files the exact call as an
approval request, holds the task, and lets the call through once — that call, with those
arguments — after a person approves it in the Control Centre. See "Approval in task work"
below.

**Every failure path here blocks.** A policy that cannot be read, a decision that raises,
a document with an unknown schema: all produce a refusal. A governance control that fails
open is not a control, and a worker that is too restricted fails loudly and visibly,
while one that is silently unrestricted does not.
"""

from __future__ import annotations

import calendar
import hashlib
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

# Risk triage for approval-gated calls (nova/autonomy): the rule, and the provider's wire
# client, each copied beside this file like ``_decide``. Optional in a way the decision is
# not: a profile installed without them (the gateway's default profile, or an install from
# before triage) never triages, which is the behaviour every escalation had before — it
# goes to a person. Their absence can only ever mean "ask", never "allow".
try:
    from . import _triage as _triage_rule
except ImportError:
    try:
        from nova.autonomy import triage as _triage_rule
    except ImportError:  # pragma: no cover — neither layout present
        _triage_rule = None
try:
    from . import _triage_typesafe as _triage_wire
except ImportError:
    try:
        from nova.autonomy.providers import typesafe as _triage_wire
    except ImportError:  # pragma: no cover
        _triage_wire = None

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


# -- approval in task work ---------------------------------------------------
#
# The store is one JSON file per request, ``<home>/nova-approvals/<task>/<request>.json``,
# shared by this plugin (which files a request and spends a grant) and the control plane
# (which records a person's answer — see ``nova.runtime.hermes.approvals``, which imports
# these helpers rather than re-implementing the format). It lives outside the profiles and
# outside every workspace, so an agent's file tools can neither read nor forge it.
#
# A request is keyed by the call itself — task, tool and canonical arguments — so what a
# person approves is exactly what runs. A re-run that changes one character of the
# arguments is a different call and is asked about again. A grant is spent by renaming
# its file, which only one process can do, so one approval lets one call through.

APPROVALS_DIRNAME = "nova-approvals"
PENDING, APPROVED, REFUSED = "pending", "approved", "refused"

#: The start of the block reason a held task carries. The Control Centre reads it as "a
#: person must decide", not as a failure to retry.
APPROVAL_BLOCK_PREFIX = "NOVA approval needed"

#: How much of a call's arguments a block reason quotes. The full arguments stay in the
#: request file; the board's reason is a summary a person can read at a glance.
_SUMMARY_CHARS = 400


def approvals_root(home: Path) -> Path:
    return Path(home) / APPROVALS_DIRNAME


def canonical_args(args: Any) -> str:
    return json.dumps(args if args is not None else {}, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)


def request_id(task_id: str, tool_name: str, args: Any) -> str:
    material = json.dumps([task_id, tool_name, canonical_args(args)], ensure_ascii=False)
    return "ap_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]


def request_path(root: Path, task_id: str, rid: str) -> Optional[Path]:
    """Where a request lives, or None for an id that could name anything else."""
    if not _PROFILE_NAME.match(task_id or "") or not re.match(r"^ap_[0-9a-f]{20}$", rid or ""):
        return None
    return Path(root) / task_id / f"{rid}.json"


def read_request(root: Path, task_id: str, rid: str) -> Optional[Dict[str, Any]]:
    path = request_path(root, task_id, rid)
    if path is None:
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    return loaded if isinstance(loaded, dict) else None


def write_request(root: Path, record: Dict[str, Any]) -> None:
    """Write a request whole or not at all: a reader never sees half a file."""
    path = request_path(root, record["task_id"], record["request_id"])
    if path is None:
        raise ValueError("not a request this store can hold")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}")
    handle = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(handle, (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        os.fsync(handle)
    finally:
        os.close(handle)
    os.replace(temporary, path)


def spend_grant(root: Path, task_id: str, rid: str) -> bool:
    """Use an approval once. True for exactly one caller, however many race for it."""
    path = request_path(root, task_id, rid)
    if path is None:
        return False
    try:
        os.rename(path, path.with_name(f"{rid}.used-{time.time_ns()}.json"))
    except FileNotFoundError:
        return False
    return True


def summarize_call(tool_name: str, args: Any) -> str:
    text = canonical_args(args)
    if len(text) > _SUMMARY_CHARS:
        text = text[:_SUMMARY_CHARS] + f"… ({len(text)} characters)"
    return f"{tool_name} {text}"


def _plugin_home() -> Path:
    """The runtime home: ``<home>/profiles/<agent>/plugins/nova-policy/__init__.py``."""
    return Path(__file__).resolve().parents[4]


def _hold_task(task_id: str, reason: str) -> bool:
    """Block the task on the board with ``reason``, through the runtime's own API."""
    try:
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc

        raw_run = os.environ.get("HERMES_KANBAN_RUN_ID", "").strip()
        with kbc.connect_closing() as connection:
            return bool(kb.block_task(
                connection, task_id, reason=reason, kind="needs_input",
                expected_run_id=int(raw_run) if raw_run.isdigit() else None,
            ))
    except Exception:  # noqa: BLE001 — the caller tells the agent to hold it instead
        return False


def _answered(task_id: str, tool_name: str, args: Dict[str, Any]) -> bool:
    """Whether a person has already been asked about exactly this call in this task."""
    try:
        record = read_request(approvals_root(_plugin_home()), task_id, request_id(task_id, tool_name, args))
    except Exception:  # noqa: BLE001 — unreadable: triage and the hold below decide
        return False
    return (record or {}).get("status") in (PENDING, APPROVED, REFUSED)


def _approval_in_task(policy: Dict[str, Any], decision: Decision, tool_name: str,
                      args: Dict[str, Any], task_id: str, *, triage_note: str = "") -> Optional[Dict[str, Any]]:
    """The directive for an escalated call in task work. None lets it run.

    Fails closed like everything else here: a store that cannot be read or written means
    the call is refused, never that it runs unasked.
    """
    root = approvals_root(_plugin_home())
    rid = request_id(task_id, tool_name, args)
    record = read_request(root, task_id, rid)
    status = (record or {}).get("status")
    if status == APPROVED and spend_grant(root, task_id, rid):
        _record(policy, Decision(ALLOW, f"approved by {record.get('decided_by') or 'a person'} "
                                 f"(request {rid}); used once", tool=tool_name,
                                 action=decision.action, rule="approval-granted"), tool_name)
        return None
    if status == REFUSED:
        why = record.get("decision_reason") or "no reason given"
        return {"action": "block", "message": (
            f"BLOCKED by NOVA policy: a person refused this exact call (request {rid}): {why}. "
            "Do not make it again. Finish what you can without it, or call kanban_block to "
            "explain what you need."
        )}
    if status != PENDING:
        write_request(root, {
            "request_id": rid, "task_id": task_id, "status": PENDING,
            "agent_id": policy.get("agent_id", ""), "tenant_id": policy.get("tenant_id", ""),
            "tool": tool_name, "action": decision.action, "reason": decision.reason,
            "args": json.loads(canonical_args(args)),
            "requested_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "run_id": os.environ.get("HERMES_KANBAN_RUN_ID", ""),
            **({"triage": triage_note} if triage_note else {}),
        })
    held = _hold_task(task_id, f"{APPROVAL_BLOCK_PREFIX} [{rid}]: {decision.reason} — "
                               f"{summarize_call(tool_name, args)}")
    _CACHE["on_hold"] = (rid, held)
    return {"action": "block", "message": _hold_message(rid, held)}


def _hold_message(rid: str, held: bool) -> str:
    then = ("When a person approves it you will be run again: make exactly this call again, "
            "with exactly the same arguments, and it will go through once.")
    if held:
        return (f"HELD by NOVA policy: this call needs a person's approval. NOVA has asked for "
                f"it (request {rid}) and put this task on hold. Stop now: make no other tool "
                f"calls and end your turn. {then}")
    return (f"HELD by NOVA policy: this call needs a person's approval (request {rid}). Call "
            f"kanban_block with the reason \"{APPROVAL_BLOCK_PREFIX} [{rid}]\" and stop. {then}")


# -- risk triage ----------------------------------------------------------------
#
# Runs only for a call ``decide()`` already sent to a person, and only for an action the
# compiled policy configures. The most it can do is let that one call run, and only when
# every condition holds: the tenant chose ``enforce``, the action is ``graduated``, the
# provider is a real one and the model answering is the one the action graduated on, and
# every question passed. Anything else — including any failure here — is the existing
# escalation, with what triage found added to the message the person reads.


def _ask_typesafe(state: Dict[str, Any], questions: list, timeout: float):
    if _triage_wire is None:
        raise RuntimeError("the provider client is not installed")
    normalized, version, _usage, latency = _triage_wire.request_answers(
        os.environ.get(_triage_wire.API_KEY_ENV, ""), state, questions, timeout=timeout)
    return normalized, version, latency


def _ask_fake(state: Dict[str, Any], questions: list, timeout: float):
    return _triage_rule.safe_answers(questions), "fake-1", 0


#: Provider name -> ``ask(state, questions, timeout) -> (answers, model_version, latency_ms)``.
#: Tests replace an entry to script answers; nothing else does.
_PROVIDERS: Dict[str, Any] = {"typesafe": _ask_typesafe, "fake": _ask_fake}

#: Calls this process let through without a person, awaiting their outcome from
#: ``post_tool_call``: tool_call_id (or tool name) -> the intent's correlation id.
_AUTONOMOUS: Dict[str, str] = {}


def _triage(policy: Dict[str, Any], decision: Decision, tool_name: str, args: Dict[str, Any],
            call: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """``{"proceed", "note"}`` for an escalated call, or None when triage does not apply."""
    config = policy.get("autonomy")
    if not isinstance(config, dict) or _triage_rule is None:
        return None
    action = (config.get("actions") or {}).get(decision.action)
    provider = config.get("provider")
    if not isinstance(action, dict) or provider in (None, "", "none"):
        return None
    mode = config.get("mode") if config.get("mode") in ("shadow", "enforce") else "shadow"
    started = time.monotonic()
    detail: Dict[str, Any] = {"tool": tool_name, "action": decision.action, "mode": mode,
                              "action_state": action.get("state", "supervised"), "provider": provider,
                              "data": config.get("data", "metadata_only")}
    try:
        questions = list(action.get("questions") or [])
        timeout = min(max(float(config.get("timeout_seconds") or 2.0), 0.2), 10.0)
        state = _triage_rule.build_state(tool_name, decision.action, args, detail["data"],
                                         _triage_rule.secret_values(os.environ))
        ask = _PROVIDERS.get(provider)
        failure: Dict[str, str] = {}

        def run():
            try:
                return ask(state, questions, timeout)
            except Exception as exc:  # noqa: BLE001 — recorded, and means "no answer"
                failure["reason"] = str(getattr(exc, "reason", "") or type(exc).__name__)
                return None

        answered = _triage_rule.within_deadline(run, timeout) if ask else None
        normalized, version, latency = answered if answered else (None, "", int((time.monotonic() - started) * 1000))
        result = _triage_rule.combine(questions, normalized)
        if answered is None:
            result["failed"] = [{"id": "", "why": "provider_unavailable" + (
                f" ({failure['reason']})" if failure.get("reason") else
                "" if ask else f" (unknown provider {provider!r})")}]
        graduated = action.get("state") == "graduated"
        expected = str(action.get("model_version") or "")
        if graduated and result["verdict"] == _triage_rule.AUTO_OK and version != expected:
            result = {"verdict": _triage_rule.ESCALATE, "failed": [{"id": "model_version", "why": (
                f"answered by model {version or '(none)'!r}, but this action graduated on {expected!r}")}]}
        proceed = (mode == "enforce" and graduated and provider != "fake"
                   and result["verdict"] == _triage_rule.AUTO_OK and not result["failed"])
        if proceed and not _still_graduated(decision.action, expected):
            # A demotion lands in the compiled file; a long-lived process (the gateway)
            # holds the policy it read at start. Re-read before acting without a person.
            proceed = False
            result = {"verdict": _triage_rule.ESCALATE, "failed": [
                {"id": "state", "why": "this action is no longer graduated"}]}
        note = _triage_rule.explain(result, mode=mode, proceed=proceed)
        detail.update({
            "verdict": result["verdict"], "failed": result["failed"], "proceed": proceed,
            "answers": normalized or {}, "model_version": version, "latency_ms": latency,
            # Fingerprints, never content: what was sent and what was asked about.
            "state_digest": _triage_rule.digest(state), "args_digest": _triage_rule.digest(args),
        })
    except Exception as exc:  # noqa: BLE001 — triage that cannot run is "ask a person"
        proceed, note = False, f"Triage could not run ({type(exc).__name__}); a person decides."
        detail.update({"verdict": "escalate", "proceed": False,
                       "failed": [{"id": "", "why": f"triage error ({type(exc).__name__})"}]})
    _write_event(policy, "policy.triage", {**detail, **{k: v for k, v in call.items() if v}})
    return {"proceed": proceed, "note": note}


def _still_graduated(action: str, model_version: str) -> bool:
    """The compiled policy on disk, read now, still lets ``action`` run without a person."""
    try:
        current = json.loads(_policy_path().read_text(encoding="utf-8"))
        config = current.get("autonomy") or {}
        entry = (config.get("actions") or {}).get(action) or {}
        return (config.get("mode") == "enforce" and entry.get("state") == "graduated"
                and str(entry.get("model_version") or "") == model_version)
    except Exception:  # noqa: BLE001 — unreadable means not shown to be still graduated
        return False


#: The runtime's answers to an approval prompt, as the ledger counts them. ``deny`` also
#: covers a turn interrupted while waiting (the runtime reports both the same way), so the
#: ledger's "rejected" is an upper bound; ``notify_failed`` means nobody was asked.
_CHOICE_OUTCOME = {"once": "approved", "session": "approved", "always": "approved",
                   "deny": "rejected", "timeout": "timed_out", "notify_failed": "not_asked"}


def post_approval_response(pattern_key: str = "", choice: str = "", tool_call_id: str = "",
                           session_id: str = "", surface: str = "", **_: Any) -> None:
    """Record a person's answer to one of NOVA's escalations, joined by the call's id.

    Only NOVA's own prompts (``plugin_rule:nova:<action>:<call>``) are recorded; the
    runtime's dangerous-command prompts are not NOVA decisions. Who answered is not in the
    payload, so it is not recorded — the ledger says so rather than guessing.
    """
    try:
        key = str(pattern_key or "")
        if not key.startswith("plugin_rule:nova:"):
            return
        parts = key[len("plugin_rule:nova:"):].split(":")
        _write_event(_load_policy() or {}, "policy.approval_outcome", {
            "action": parts[0] if parts else "",
            "choice": str(choice or ""),
            "outcome": _CHOICE_OUTCOME.get(str(choice or ""), "unknown"),
            "surface": str(surface or ""),
            "tool_call_id": str(tool_call_id or ""),
            "session_id": str(session_id or ""),
        })
    except Exception:  # noqa: BLE001 — an observer must never break the approval flow
        pass


def _autonomous_intent(policy: Dict[str, Any], decision: Decision, tool_name: str,
                       args: Dict[str, Any], call: Dict[str, str]) -> None:
    """The write-ahead record of a call that runs with no person: intent now, outcome later.

    ``post_tool_call`` writes ``committed`` or ``failed``. A worker that dies in between
    leaves the intent open, which is how a reviewer finds an action whose outcome is unknown.
    """
    correlation = "autonomous-" + (call.get("tool_call_id") or uuid.uuid4().hex)
    _AUTONOMOUS[call.get("tool_call_id") or tool_name] = correlation
    _write_event(policy, "policy.autonomous_action", {
        "tool": tool_name, "action": decision.action, "args_digest": _triage_rule.digest(args),
        **{k: v for k, v in call.items() if v},
    }, phase="intent", correlation_id=correlation)


def post_tool_call(tool_name: str = "", tool_call_id: str = "", status: str = "",
                   error_type: str = "", **_: Any) -> None:
    """Close the write-ahead record of a call triage let through. Nothing else."""
    try:
        correlation = _AUTONOMOUS.pop(str(tool_call_id or ""), None) or _AUTONOMOUS.pop(tool_name, None)
        if correlation is None:
            return
        policy = _load_policy() or {}
        ok = str(status or "ok") == "ok"
        _write_event(policy, "policy.autonomous_action",
                     {"tool": tool_name, "status": status or "ok", "tool_call_id": tool_call_id,
                      **({"error_type": error_type} if error_type else {})},
                     phase="committed" if ok else "failed", correlation_id=correlation)
    except Exception:  # noqa: BLE001 — an observer must never break the worker
        pass


def _write_event(policy: Dict[str, Any], kind: str, detail: Dict[str, Any], *,
                 phase: str = "record", correlation_id: str = "") -> None:
    """Append one event to the tenant's audit log, in the control plane's format."""
    target = (policy or {}).get("audit_log")
    if not target:
        return
    event = {
        "event_id": uuid.uuid4().hex,
        "correlation_id": correlation_id or os.environ.get("HERMES_KANBAN_TASK", "") or "runtime",
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "kind": kind,
        "phase": phase,
        "actor": "nova-policy-plugin",
        "tenant_id": (policy or {}).get("tenant_id", ""),
        "model_visible": False,
        "subject": (policy or {}).get("agent_id", ""),
        "digest": "",
        "detail": detail,
        "error": "",
    }
    try:
        line = (json.dumps(event, sort_keys=True, separators=(",", ":"), default=str) + "\n").encode("utf-8")
        handle = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(handle, line)
        finally:
            os.close(handle)
    except OSError:
        # A governance record we cannot write must not stop the decision being enforced.
        pass


def _record(policy: Dict[str, Any], decision, tool_name: str, call: Optional[Dict[str, str]] = None) -> None:
    """Append a governance record for a refusal or an escalation.

    Permitted calls are not recorded: they are the overwhelming majority and recording
    them would bury the events a reviewer is actually looking for. What was refused, and
    what needed a human, is the governance question.
    """
    _write_event(policy, "policy.decision", {
            "tool": tool_name,
            "effect": decision.effect,
            "reason": decision.reason,
            "rule": decision.rule,
            "action": decision.action,
            "calls_used": _CALLS_USED,
            # The runtime's ids for this call, so a person's answer to an escalation
            # (the approval hooks carry the same ids) can be joined back to it.
            **{key: value for key, value in (call or {}).items() if value},
    })


def pre_tool_call(tool_name: str = "", args: Optional[Dict[str, Any]] = None,
                  tool_call_id: str = "", session_id: str = "", **_: Any):
    """The runtime's policy hook. Returns a directive, or None to proceed.

    Unrecognised returns are ignored by the runtime, so returning ``None`` for a permitted
    call is the correct way to stay out of the way.
    """
    global _CALLS_USED
    call = {"tool_call_id": str(tool_call_id or ""), "session_id": str(session_id or "")}
    held = _CACHE.get("on_hold")
    if held:
        # This task is waiting for a person. Nothing more runs in this process: the board
        # already says why, and a worker that carried on would act past the question.
        return {"action": "block", "message": _hold_message(*held)}
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

    if decision.effect == REQUIRE_APPROVAL:
        task_id = os.environ.get("HERMES_KANBAN_TASK", "").strip()
        if task_id:
            # The board request this call becomes, so triage and the person's answer join.
            call["request_id"] = request_id(task_id, tool_name, args or {})
        # A call a person already answered on the board is not triaged again: their answer
        # stands, and asking the provider would only spend a call to second-guess it.
        triage = None
        if not (task_id and _answered(task_id, tool_name, args or {})):
            triage = _triage(policy, decision, tool_name, args or {}, call)
        if triage and triage["proceed"]:
            _autonomous_intent(policy, decision, tool_name, args or {}, call)
            return None
        note = (triage or {}).get("note", "")
        _record(policy or {}, decision, tool_name, call)
        if task_id:
            try:
                return _approval_in_task(policy, decision, tool_name, args or {}, task_id, triage_note=note)
            except Exception as exc:  # noqa: BLE001 — an approval we cannot track is refused
                return {"action": "block", "message": (
                    f"BLOCKED by NOVA policy: {decision.reason}, and the approval request could "
                    f"not be filed ({type(exc).__name__}). Call kanban_block with this reason."
                )}
        return {
            "action": "approve",
            "message": f"{decision.reason}\n{note}" if note else decision.reason,
            # One key per call. The runtime's prompt offers "always", which saves the key to
            # the profile's allowlist and from then on lets every call carrying it through
            # before any prompt or hook — so a per-action key turned one click into a NOVA
            # approval switched off for good, silently. A key no later call can carry means
            # "once", "session" and "always" all approve this call and nothing else.
            "rule_key": f"nova:{decision.action or tool_name}:{tool_call_id or uuid.uuid4().hex}",
        }

    _record(policy or {}, decision, tool_name, call)

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
    ctx.register_hook("post_tool_call", post_tool_call)
    ctx.register_hook("post_approval_response", post_approval_response)
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
