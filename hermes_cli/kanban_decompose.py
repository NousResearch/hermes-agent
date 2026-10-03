"""Kanban decomposer — fan a triage task out into a graph of child tasks.

Invoked by ``hermes kanban decompose [task_id | --all]`` and the gateway
dispatcher's auto-decompose path. Reads the profile roster (with
descriptions), asks the auxiliary LLM for a task graph in JSON, then
atomically creates the children, links them under the root, and flips the
root ``triage -> todo``. The root stays alive as parent of every leaf child so
it wakes back up when the graph completes and its assignee (the orchestrator
profile) can judge completion and add more work.

Mirrors ``kanban_specify`` (lazy aux import, lenient parse, never raises on
expected failures). ``fanout=false`` collapses to the ``specify`` behaviour
(tighten + promote, no children), making ``decompose`` a strict superset.
Unknown assignees are rewritten to ``default_assignee`` — a child NEVER ends
up with ``assignee=None``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_graph import decompose_triage_task
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import profiles as profiles_mod
from hermes_cli.kanban_specify import (
    _call_aux, _extract_json_blob, _load_triage_task, _task_prompt_fields, _title_body,
)
from hermes_cli.kanban_specify import _profile_author as _specify_author

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """You are the Kanban decomposer for the Hermes Agent board.

A user dropped a rough idea into the Triage column. Your job is to break it
into a small graph of concrete child tasks and route each one to the best-
matching profile from the available roster.

You will be given:
  - The original task title and body
  - The list of available profiles (each with name + description)
  - The fallback "default_assignee" used when no profile fits

Output a single JSON object with this exact shape:

  {
    "fanout": true,
    "rationale": "<one sentence on why this decomposition>",
    "tasks": [
      {
        "title": "<concrete task title, imperative voice, <= 80 chars>",
        "body":  "<detailed spec for the worker on this child task>",
        "assignee": "<profile name from the roster, or null for default>",
        "parents": [<int>, ...]
      },
      ...
    ]
  }

Rules:
  - "parents" is a list of INDICES (0-based) into this same "tasks" list,
    expressing actual data dependencies. Tasks with no parents run in
    PARALLEL. Tasks with parents wait until every parent completes.
  - Prefer parallelism. If two tasks can be done independently, give
    them no parents so the dispatcher fans them out at once.
  - Use 2-6 tasks for normal work. Don't create 20 tiny tasks. Don't
    cram everything into 1 task.
  - Pick assignees from the roster by matching the task to the profile's
    DESCRIPTION (not just the name). When nothing matches well, use null
    and the system will route to the default_assignee.
  - Each child task body is what a fresh worker will read with no other
    context — be specific about goal, approach, and acceptance criteria.

When the task is genuinely a single unit of work (no useful decomposition),
return:

  {
    "fanout": false,
    "rationale": "<one sentence>",
    "title": "<tightened title>",
    "body":  "<concrete spec for a single worker>",
    "assignee": "<profile name from the roster, or null for default>"
  }

In that case the task stays as one work item, just with a tightened spec and
a concrete assignee. If no profile fits, use null and the system will route to
the default_assignee.

No preamble, no closing remarks, no code fences. Output only the JSON object.
"""


_USER_TEMPLATE = """Task id: {task_id}
Title: {title}
Body:
{body}

Available profiles (assignees you may pick from):
{roster}

Default assignee (used when no profile fits a task): {default_assignee}
"""


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


@dataclass
class DecomposeOutcome:
    """Result of decomposing a single triage task."""

    task_id: str
    ok: bool
    reason: str = ""
    fanout: bool = False
    child_ids: list[str] | None = None
    new_title: Optional[str] = None
    # Set ONLY on the auto_decompose-disabled routing path, and ONLY when the
    # wake was CONFIRMED delivered to that single eligible profile. ``ok`` stays
    # False there — nothing was decomposed — but callers can tell "routed"
    # (non-None) from "delivery failed" (None) instead of trusting a claim.
    routed_to: Optional[str] = None


def _profile_author() -> str:
    """Mirror of ``hermes_cli.kanban._profile_author``."""
    return _specify_author("decomposer")


def _resolve_profile_from_cfg(cfg: dict, key: str, *, fallback: Optional[str] = None) -> str:
    """``kanban.<key>`` if it names an existing profile, else ``fallback``
    (the root task's own assignee) if that does, else the active default
    profile — so a task is never stranded for lack of an owner.
    ``orchestrator_profile`` owns the root after fan-out; ``default_assignee``
    catches children the decomposer can't route.

    The root's assignee sits before the active profile because the decomposer
    runs inside whatever profile hosts the dispatcher — an operator's
    credential-less incognito profile, say — and that profile must never
    silently become the owner of work the card was assigned away from (#114294).
    """
    kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    explicit = (kanban_cfg.get(key) or "").strip()
    for candidate in (explicit, (fallback or "").strip()):
        if candidate:
            try:
                if profiles_mod.profile_exists(candidate):
                    return candidate
            except Exception:
                pass
    try:
        return profiles_mod.get_active_profile_name() or "default"
    except Exception:
        return "default"


def _build_roster() -> tuple[list[dict], set[str]]:
    """``(roster_for_prompt, valid_assignee_names)``; entries are
    ``{name, description, has_description}``."""
    try:
        all_profiles = profiles_mod.list_profiles()
    except Exception as exc:
        logger.warning("decompose: failed to list profiles: %s", exc)
        return [], set()
    roster = []
    for p in all_profiles:
        desc = (p.description or "").strip()
        roster.append({
            "name": p.name,
            "description": desc or f"(no description; profile named {p.name!r})",
            "has_description": bool(desc),
        })
    return roster, {p.name for p in all_profiles}


def _format_roster(roster: list[dict]) -> str:
    if not roster:
        return "  (no profiles installed — decomposer cannot route work)"
    return "\n".join(
        f"  - {entry['name']}{'' if entry['has_description'] else ' ⚠ undescribed'}: {entry['description']}"
        for entry in roster
    )


def _normalize_assignee_choice(assignee: object, *, default_assignee: str, valid_names: set[str]) -> str:
    """A valid assignee, else ``default_assignee`` — promoted work is never
    left unassigned."""
    if not isinstance(assignee, str) or not assignee.strip():
        return default_assignee
    chosen = assignee.strip()
    return chosen if chosen in valid_names else default_assignee


@dataclass
class _Routing:
    """Config-derived routing context for one decomposition."""

    orchestrator: str
    default_assignee: str
    auto_promote: bool
    roster: list[dict]
    valid_names: set[str]


def _load_routing(*, root_assignee: Optional[str] = None) -> _Routing:
    from hermes_cli.config import load_config_readonly
    try:
        cfg = load_config_readonly()
    except Exception:  # decompose_task promises ok=False, never a raise, on config trouble
        cfg = {}
    kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    roster, valid_names = _build_roster()
    return _Routing(
        orchestrator=_resolve_profile_from_cfg(cfg, "orchestrator_profile", fallback=root_assignee),
        default_assignee=_resolve_profile_from_cfg(cfg, "default_assignee", fallback=root_assignee),
        auto_promote=bool(kanban_cfg.get("auto_promote_children", True)),
        roster=roster,
        valid_names=valid_names,
    )


def _apply_single(task: kb.Task, parsed: dict, routing: _Routing, author: str) -> DecomposeOutcome:
    """``fanout=false``: single-task spec promotion (same effect as specify)."""
    title_val, body_val = _title_body(parsed)
    assignee_val = None
    if not task.assignee:
        assignee_val = _normalize_assignee_choice(
            parsed.get("assignee"), default_assignee=routing.default_assignee, valid_names=routing.valid_names,
        )
    if title_val is None and body_val is None:
        return DecomposeOutcome(task.id, False, "decomposer returned fanout=false with no title/body")
    with kbc.connect_closing() as conn:
        ok = kb.specify_triage_task(
            conn, task.id, title=title_val, body=body_val, assignee=assignee_val, author=author,
        )
    if not ok:
        return DecomposeOutcome(task.id, False, "task moved out of triage before promotion")
    return DecomposeOutcome(task.id, True, "single task (no fanout)", fanout=False, new_title=title_val)


def _clean_children(task_id: str, raw_tasks: list, routing: _Routing) -> tuple[list[dict], str]:
    """Validate/normalise the LLM's ``tasks`` list; ``(children, "")`` or ``([], reason)``.
    Unknown assignees route to the default; never assignee=None."""
    children: list[dict] = []
    for idx, entry in enumerate(raw_tasks):
        if not isinstance(entry, dict):
            return [], f"tasks[{idx}] is not an object"
        title = entry.get("title")
        if not isinstance(title, str) or not title.strip():
            return [], f"tasks[{idx}].title is missing or empty"
        body = entry.get("body")
        assignee = entry.get("assignee")
        chosen = _normalize_assignee_choice(
            assignee, default_assignee=routing.default_assignee, valid_names=routing.valid_names,
        )
        if isinstance(assignee, str) and assignee.strip() and assignee.strip() not in routing.valid_names:
            logger.info(
                "decompose: task %s child %d picked unknown assignee %r — "
                "routing to default_assignee %r",
                task_id, idx, assignee, routing.default_assignee,
            )
        parents = entry.get("parents") or []
        if not isinstance(parents, list):
            parents = []
        children.append({
            "title": title.strip()[:200],
            "body": body.strip() if isinstance(body, str) else "",
            "assignee": chosen,
            # Drop non-int, out-of-range and self parent indices.
            "parents": [p for p in parents if isinstance(p, int) and 0 <= p < len(raw_tasks) and p != idx],
        })
    return children, ""


def _apply_fanout(task_id: str, parsed: dict, routing: _Routing, author: str) -> DecomposeOutcome:
    raw_tasks = parsed.get("tasks") or []
    if not isinstance(raw_tasks, list) or not raw_tasks:
        return DecomposeOutcome(task_id, False, "decomposer returned fanout=true with empty tasks list")
    children, reason = _clean_children(task_id, raw_tasks, routing)
    if reason:
        return DecomposeOutcome(task_id, False, reason)
    try:
        with kbc.connect_closing() as conn:
            child_ids = decompose_triage_task(
                conn,
                task_id,
                root_assignee=routing.orchestrator,
                children=children,
                author=author,
                auto_promote=routing.auto_promote,
            )
    except ValueError as exc:
        return DecomposeOutcome(task_id, False, f"DB rejected graph: {exc}")
    except Exception as exc:
        logger.exception("decompose: DB error on task %s", task_id)
        return DecomposeOutcome(task_id, False, f"DB error: {type(exc).__name__}")
    if child_ids is None:
        return DecomposeOutcome(task_id, False, "task already decomposed or moved out of triage")
    return DecomposeOutcome(
        task_id, True, f"decomposed into {len(child_ids)} children", fanout=True, child_ids=child_ids,
    )


# ---------------------------------------------------------------------------
# Routing gate (kanban.auto_decompose == false)
# ---------------------------------------------------------------------------
# With auto-decompose OFF the auxiliary model must never be called. Instead the
# existing prompt is DELIVERED to the task's single eligible subscriber as a
# wake/instruction; the graph is left alone until that profile (or a human)
# makes an explicit kanban_decompose call. Delivery has two legs, in order:
# the in-process gateway transport when this process hosts it, else the local
# gateway control socket (``deliver-decompose-instruction``) — a CLI or an
# out-of-process dashboard holds no adapters, but the running gateway that
# does answers there. ``routed_to`` is reported ONLY for a confirmed wake.

# In-process gateway transport, registered at gateway start
# (``gateway.kanban_watchers.install_kanban_instruction_transport``). ``None``
# in every process that is not the gateway — those take the control-socket leg.
_INSTRUCTION_TRANSPORT: Optional[Callable[..., tuple[bool, str]]] = None

# Client-side bound for the control-socket leg; slightly above the gateway's
# own INSTRUCTION_DELIVERY_TIMEOUT so the gateway's honest answer (delivered
# or not) arrives before the client gives up on the socket.
_CONTROL_DELIVERY_TIMEOUT = 70.0


def set_instruction_transport(fn: Optional[Callable[..., tuple[bool, str]]]) -> None:
    """Register (or clear, with ``None``) the in-process gateway wake transport.

    The gateway installs this at startup so a decompose instruction can be
    pushed to the subscriber's destination without leaving the process; every
    other process (CLI, dashboard threadpool, tests) leaves it unset and takes
    the control-socket leg to a running gateway instead — and gets an honest
    "could not deliver" from ``deliver_decompose_instruction`` when no gateway
    is reachable either.
    """
    global _INSTRUCTION_TRANSPORT
    _INSTRUCTION_TRANSPORT = fn


def _auto_decompose_enabled() -> bool:
    """Current ``kanban.auto_decompose`` (default True, read fresh per call)."""
    from hermes_cli.config import cfg_get, load_config_readonly
    try:
        cfg = load_config_readonly()
    except Exception:  # unreadable config keeps the shipped default
        cfg = {}
    return bool(cfg_get(cfg, "kanban", "auto_decompose", default=True))


def profile_has_kanban_toolset(profile: str) -> bool:
    """True when ``profile``'s OWN ``config.yaml`` enables the kanban toolset.

    Mirrors ``tools.kanban_tools._profile_has_kanban_toolset`` but judges a
    named profile rather than the active one: routing must score each subscriber
    against the config that profile actually runs with, never against the config
    of whichever profile happens to host this call. Fail-closed — an unreadable
    or absent config means "not capable".
    """
    if not profile or not str(profile).strip():
        return False
    name = str(profile).strip()
    try:
        from hermes_cli.profiles import _load_yaml_dict, get_profile_dir, profile_exists
        from hermes_cli.tools_config import _get_platform_tools

        if not profile_exists(name):
            return False
        path = get_profile_dir(name) / "config.yaml"
        raw = _load_yaml_dict(path) if path.exists() else None
        if not isinstance(raw, dict):
            return False
        if "kanban" in [str(t) for t in (raw.get("toolsets") or [])]:
            return True
        platforms = raw.get("platform_toolsets")
        if not isinstance(platforms, dict):
            return False
        return any(
            "kanban" in _get_platform_tools(raw, platform, include_default_mcp_servers=False)
            for platform, names in platforms.items() if isinstance(names, list)
        )
    except Exception:
        logger.debug("decompose: could not read toolsets for profile %r", name, exc_info=True)
        return False


def eligible_decompose_subscribers(
    conn, task_id: str,
) -> tuple[list[str], dict[str, list[dict]], list[dict]]:
    """``(distinct eligible profiles, every sub grouped by profile, every sub)``.

    Multiple destinations for the same profile (several chats or threads)
    collapse to that one profile — routing must never fan out. Subscriptions
    without a ``notifier_profile`` stamp cannot name a profile and are never
    eligible. Read-only. The raw list comes back too so a caller can explain a
    refusal without querying the same rows a second time.
    """
    from hermes_cli import kanban_db_notify as _kbn

    all_subs = _kbn.list_notify_subs(conn, task_id)
    grouped: dict[str, list[dict]] = {}
    for sub in all_subs:
        owner = str(sub.get("notifier_profile") or "").strip()
        if owner:
            grouped.setdefault(owner, []).append(sub)
    return sorted(p for p in grouped if profile_has_kanban_toolset(p)), grouped, all_subs


def build_decompose_instruction(task_id: str, *, system: str, user: str) -> str:
    """The existing decompose prompt, wrapped as a wake/instruction.

    The system + user pair is byte-for-byte the one the auxiliary path sends, so
    a routed profile reasons over exactly the context the decomposer would have.
    """
    return (
        "Kanban decomposition requested — you are the single eligible subscribed "
        f"Kanban-toolset-capable profile for task {task_id}.\n\n"
        "This message changes NOTHING on the board. Review the prompt below and, "
        "if you accept the graph, apply it with an explicit "
        "kanban_decompose(task_id, children) tool call. No child row, edge, event "
        "or status is created until you make that call.\n\n"
        "=== SYSTEM ===\n" + system + "\n\n=== USER ===\n" + user
    )


def _confirmed_delivery(delivered, detail) -> tuple[bool, str]:
    """A delivery flag counts only as an EXACT boolean ``True``.

    Both delivery boundaries — the gateway control-socket answer and the
    in-process transport — once ran their flag through ``bool()``, which turns
    a malformed truthy answer (the string ``"false"``, the integer ``1``) into
    a confirmed wake and lets ``routed_to`` name a handoff that never
    happened. Truthy-but-not-boolean is therefore refused, with the malformed
    value named; every genuinely falsy answer keeps its own detail verbatim.
    """
    text = str(detail)
    if delivered is True:
        return True, text
    if not delivered:
        return False, text
    return False, (
        f"{text} — refused: delivered={delivered!r} is not boolean True, "
        "so the wake is NOT confirmed"
    )


def _deliver_via_control_socket(
    profile: str, *, task_id: str, text: str, subs: list[dict],
) -> tuple[bool, str]:
    """Leg 2 of delivery: ask the local running GATEWAY to push the wake.

    This process (a normal CLI invocation, or an out-of-process dashboard
    thread) holds no adapters, so it cannot confirm a wake itself — but the
    gateway that does answers the ``deliver-decompose-instruction`` control
    verb. Candidate homes are the process's own ``HERMES_HOME`` (where the
    gateway that spawned this process binds its socket) and, under profile
    mode, the default root the multiplexer actually serves.

    Safety rules: only the FIRST home with a live control socket is asked (a
    gateway that answers — even with ``delivered: false`` — is authoritative,
    so a possibly-in-flight wake is never retried against a second gateway and
    cannot double-wake); ``delivered`` is true only for the gateway's confirmed
    answer — an exact boolean ``True``, never inferred from a connected socket
    and never coerced from a truthy non-boolean.
    """
    try:
        from gateway.control_socket import query_gateway_control, resolve_client_socket_path
        from hermes_constants import get_default_hermes_root, get_process_hermes_home
    except Exception as exc:  # pragma: no cover - gateway package always present in-tree
        return False, f"gateway control-socket client unavailable: {type(exc).__name__}: {exc}"

    params = json.loads(json.dumps(
        {"profile": profile, "task_id": task_id, "text": text, "subs": list(subs)},
        default=str))
    homes: list[Path] = []
    for candidate in (get_process_hermes_home(), get_default_hermes_root()):
        try:
            resolved = Path(candidate).expanduser()
        except Exception:
            continue
        if resolved not in homes:
            homes.append(resolved)

    socket_homes = [home for home in homes if resolve_client_socket_path(home) is not None]
    if not socket_homes:
        return False, (
            "no gateway control socket is reachable from this process, so no wake "
            "could be pushed (the prompt is ready for explicit handoff)"
        )
    home = socket_homes[0]
    try:
        result = query_gateway_control(
            home, "deliver-decompose-instruction",
            params=params, timeout=_CONTROL_DELIVERY_TIMEOUT,
        )
    except Exception as exc:
        return False, f"gateway control query failed: {type(exc).__name__}: {exc}"
    if not isinstance(result, dict):
        return False, (
            f"a gateway control socket exists at {home} but gave no usable "
            f"deliver-decompose-instruction answer — the wake is NOT confirmed"
        )
    delivered = result.get("delivered")
    detail = result.get("detail") or (
        "instruction delivered via the gateway control socket"
        if delivered is True
        else "the gateway could not confirm the instruction wake"
    )
    return _confirmed_delivery(delivered, detail)


def deliver_decompose_instruction(
    profile: str, *, task_id: str, system: str, user: str,
    subs: Optional[list[dict]] = None,
) -> tuple[bool, str]:
    """Push the decompose prompt to ``profile`` as a wake/instruction.

    Two legs, tried in order: the in-process gateway transport when this
    process hosts the gateway, else the ``deliver-decompose-instruction``
    control-socket verb asked of the local running gateway. Board-read only:
    it never writes a task, link, event or comment row, because a
    notification on its own must not change the task graph. Returns
    ``(delivered, detail)``; when neither leg can confirm a wake the detail
    says so plainly rather than pretending the wake went out.
    """
    if not subs:
        return False, f"profile {profile!r} has no live subscription on task {task_id}"
    text = build_decompose_instruction(task_id, system=system, user=user)
    transport = _INSTRUCTION_TRANSPORT
    if transport is None:
        return _deliver_via_control_socket(profile, task_id=task_id, text=text, subs=subs)
    try:
        delivered, detail = transport(profile=profile, task_id=task_id, text=text, subs=subs)
    except Exception as exc:
        logger.debug("decompose: instruction delivery to %s failed: %s", profile, exc, exc_info=True)
        return False, f"delivery failed: {type(exc).__name__}: {exc}"
    # Same strictness as the control-socket leg: the in-process transport's
    # flag is an exact boolean True or it is not a confirmed wake.
    return _confirmed_delivery(delivered, detail)


def _route_decompose_instruction(task: kb.Task) -> DecomposeOutcome:
    """auto_decompose-off path: route to exactly one subscriber, never the LLM.

    Zero eligible profiles -> clear no-subscriber result, task left in triage.
    Several distinct profiles -> clear ambiguity, no broadcast, graph untouched.
    Exactly one -> the existing prompt is DELIVERED to that profile only (and
    ``routed_to`` is reported only once that wake is confirmed); the graph
    stays unchanged until an explicit ``kanban_decompose`` call either way.
    """
    routing = _load_routing(root_assignee=task.assignee)
    # One sentence, stated identically in every refusal below so the four
    # outcomes cannot drift apart.
    off = "kanban.auto_decompose is disabled and the auxiliary model was NOT called"
    user = _USER_TEMPLATE.format(
        **_task_prompt_fields(task),
        roster=_format_roster(routing.roster),
        default_assignee=routing.default_assignee,
    )
    with kbc.connect_closing() as conn:
        eligible, grouped, all_subs = eligible_decompose_subscribers(conn, task.id)
    # Count subscriptions the eligibility pass dropped, so the refusal can say
    # WHY there is no eligible profile instead of claiming the task is unsubscribed.
    unowned = sum(1 for s in all_subs
                  if not str(s.get("notifier_profile") or "").strip())

    if not eligible:
        if not all_subs:
            detail = "this task has no subscriptions"
        elif unowned:
            detail = (
                f"{len(all_subs)} subscription(s) but none is eligible "
                f"({unowned} carry no notifier profile, "
                f"{len(all_subs) - unowned} name a profile without the kanban "
                f"toolset)"
            )
        else:
            detail = (
                f"{len(all_subs)} subscription(s) name a profile without the "
                f"kanban toolset"
            )
        return DecomposeOutcome(
            task.id, False,
            f"{off}: no eligible Kanban-toolset-capable subscriber ({detail}). The task stays in "
            f"triage and the graph is unchanged. Subscribe exactly one Kanban profile "
            f"(`hermes kanban notify-subscribe {task.id} ... --notifier-profile <name>`) "
            f"or enable kanban.auto_decompose.",
        )

    if len(eligible) > 1:
        return DecomposeOutcome(
            task.id, False,
            f"{off}: {len(eligible)} distinct eligible Kanban profiles are subscribed "
            f"({', '.join(eligible)}). Refusing to broadcast or to mutate the graph — pick "
            f"one owner and call kanban_decompose explicitly. The task is unchanged.",
        )

    profile = eligible[0]
    delivered, detail = deliver_decompose_instruction(
        profile, task_id=task.id, system=_SYSTEM_PROMPT, user=user, subs=grouped[profile],
    )
    if not delivered:
        # routed_to stays None unless the wake was actually confirmed: a prompt
        # that never reached the profile was NOT handed to it.
        return DecomposeOutcome(
            task.id, False,
            f"{off}. The decomposition prompt for {task.id} could NOT be delivered to the single "
            f"eligible subscribed Kanban profile {profile!r}: {detail}. The task stays in "
            f"triage and the graph is unchanged — no wake reached that profile, so retry "
            f"when a gateway is reachable, or hand the prompt off explicitly.",
        )
    return DecomposeOutcome(
        task.id, False,
        f"{off}. Decomposition prompt for {task.id} handed to the single eligible subscribed "
        f"Kanban profile {profile!r}: {detail}. The task graph is unchanged — only an "
        f"explicit kanban_decompose tool call from that profile creates children or "
        f"promotes the root.",
        routed_to=profile,
    )


def apply_explicit_decomposition(
    conn, task_id: str, *, children: list, author: Optional[str] = None,
) -> list[str]:
    """Apply an agent-authored child graph with NO auxiliary/LLM call.

    Every spec is validated up front (shape, title, parents, installed
    assignees) and persistence is handed to :func:`decompose_triage_task`, which
    performs the whole fan-out inside ONE transaction. An invalid or cyclic
    graph therefore writes no child row, edge or event and leaves the root
    exactly as it was; returns the created child ids in input order.

    Raises ``ValueError`` with a specific reason on any rejection — the caller
    (the tool handler) renders it as a structured tool error.
    """
    if not isinstance(children, list) or not children:
        raise ValueError("children must be a non-empty array of child specs")
    task = kb.get_task(conn, task_id)
    if task is None:
        raise ValueError(f"unknown task {task_id}")
    if task.status != "triage":
        raise ValueError(
            f"task {task_id} is {task.status!r}; only a 'triage' task can be decomposed"
        )
    routing = _load_routing(root_assignee=task.assignee)
    normalized: list[dict] = []
    for idx, spec in enumerate(children):
        if not isinstance(spec, dict):
            raise ValueError(f"children[{idx}] is not an object")
        title = spec.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"children[{idx}].title is required")
        parents = spec.get("parents")
        if parents is not None and not isinstance(parents, list):
            raise ValueError(f"children[{idx}].parents must be a list of 0-based indexes")
        assignee = spec.get("assignee")
        if isinstance(assignee, str) and assignee.strip() and assignee.strip() not in routing.valid_names:
            raise ValueError(
                f"children[{idx}].assignee {assignee.strip()!r} is not an installed profile; "
                f"installed: {', '.join(sorted(routing.valid_names)) or '(none)'}"
            )
        body = spec.get("body")
        entry: dict = {
            "title": title.strip()[:200],
            "body": body.strip() if isinstance(body, str) else "",
            # Omitted assignees fall back to the established default so a child
            # is never created unassigned (and therefore never dispatchable).
            "assignee": _normalize_assignee_choice(
                assignee, default_assignee=routing.default_assignee,
                valid_names=routing.valid_names,
            ),
            "parents": list(parents) if parents else [],
        }
        for key in ("workspace_kind", "workspace_path"):
            if spec.get(key):
                entry[key] = spec[key]
        normalized.append(entry)

    child_ids = decompose_triage_task(
        conn, task_id,
        # Established decomposition contract: the root is handed to the
        # configured orchestrator profile, which falls back to the root's own
        # assignee — so an unconfigured board keeps its original owner.
        root_assignee=routing.orchestrator,
        children=normalized, author=author, auto_promote=routing.auto_promote,
    )
    if child_ids is None:
        raise ValueError("task already decomposed or moved out of triage")
    return child_ids


def decompose_task(
    task_id: str,
    *,
    author: Optional[str] = None,
    timeout: Optional[int] = None,
) -> DecomposeOutcome:
    """Decompose a triage task into a graph of child tasks. Expected failures
    (not in triage, no aux client, API error, malformed/empty reply) surface
    as ``ok=False``."""
    task, reason = _load_triage_task(task_id)
    if task is None:
        return DecomposeOutcome(task_id, False, reason)

    # Routing gate: with kanban.auto_decompose off the auxiliary model must never
    # be reached. This sits BEFORE every aux call so the CLI (`hermes kanban
    # decompose`), the dashboard POST /tasks/{id}/decompose and any other caller
    # of this function share one implementation of the contract.
    if not _auto_decompose_enabled():
        return _route_decompose_instruction(task)

    routing = _load_routing(root_assignee=task.assignee)
    raw, reason = _call_aux(
        "decompose", task_id, aux_task="kanban_decomposer", system=_SYSTEM_PROMPT,
        user=_USER_TEMPLATE.format(
            **_task_prompt_fields(task),
            roster=_format_roster(routing.roster),
            default_assignee=routing.default_assignee,
        ),
        max_tokens=4000, timeout=timeout or 180, log=logger,
    )
    if raw is None:
        return DecomposeOutcome(task_id, False, reason)

    parsed = _extract_json_blob(raw, _FENCE_RE)
    if parsed is None:
        return DecomposeOutcome(task_id, False, "LLM returned malformed JSON")

    audit_author = author or _profile_author()
    if not parsed.get("fanout"):
        return _apply_single(task, parsed, routing, audit_author)
    return _apply_fanout(task_id, parsed, routing, audit_author)


def list_triage_ids(*, tenant: Optional[str] = None) -> list[str]:
    """Return task ids currently in the triage column."""
    with kbc.connect_closing() as conn:
        rows = kb.list_tasks(conn, status="triage", tenant=tenant, limit=1000)
    return [row.id for row in rows]
