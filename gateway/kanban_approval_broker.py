"""Explicit Kanban approval transport selection and canonical response binding.

A terminal-event subscription is only a destination until its operator separately marks it for
approval forwarding. Even then, exactly one currently connected, route-valid Telegram subscriber
must exist, and its stored identity must pass the adapter's normal allowlist/pairing policy.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import uuid
from typing import Any


logger = logging.getLogger(__name__)
APPROVAL_FORWARDING_METADATA_KEY = "kanban_approval_forwarding"
APPROVAL_FORWARDING_TASK_KEY = "kanban_approval_task_id"
APPROVAL_FORWARDING_V1 = "v1"
FORWARDING_METADATA_KEY = "_kanban_approval_binding_v1"


def _subscription_home(runner, sub):
    """Home whose allowlist/scope owns the notifier, never the worker's ambient profile."""
    if not getattr(getattr(runner, "config", None), "multiplex_profiles", False):
        return None
    from gateway.config import Platform
    from gateway.session import SessionSource

    source = SessionSource(platform=Platform.TELEGRAM, chat_id=str(sub["chat_id"]),
                           profile=sub["notifier_profile"])
    return runner._resolve_profile_home_for_source(source)


def _subscription_thread_id(sub):
    """Canonicalize the DB's absent-thread sentinel to Telegram's callback representation."""
    thread_id = sub.get("thread_id")
    return None if thread_id is None or thread_id == "" else str(thread_id)


def _authorized_subscriber(adapter, sub):
    return adapter._is_callback_user_authorized(
        str(sub["user_id"]), chat_id=str(sub["chat_id"]), chat_type=str(sub.get("chat_type") or "dm"),
        thread_id=_subscription_thread_id(sub), user_name=None)


def _eligible_subscription(runner, sub):
    if (sub.get("platform") or "").lower() != "telegram":
        return None
    metadata = sub.get("delivery_metadata") or {}
    if metadata.get(APPROVAL_FORWARDING_METADATA_KEY) != APPROVAL_FORWARDING_V1:
        return None
    if not all(sub.get(key) for key in ("chat_id", "user_id", "notifier_profile")):
        return None
    from gateway.config import Platform
    from gateway.kanban_watchers_notifier import _adapter_for_subscription

    try:
        adapter = _adapter_for_subscription(runner, Platform.TELEGRAM, sub, sub["notifier_profile"])
    except Exception:
        logger.debug("Kanban approval subscription route resolution failed", exc_info=True)
        return None
    if adapter is None or getattr(adapter, "_bot", None) is None or not adapter.is_connected:
        return None
    try:
        buttons = adapter.supports_exec_approval_buttons()
        home = _subscription_home(runner, sub)
        if home is None:
            authorized = _authorized_subscriber(adapter, sub)
        else:
            from gateway.run import _profile_runtime_scope
            with _profile_runtime_scope(home):
                authorized = _authorized_subscriber(adapter, sub)
    except Exception:
        logger.debug("Kanban approval subscriber authorization failed", exc_info=True)
        return None
    return (adapter, sub) if buttons and authorized else None


def select_approval_subscription(runner, subscriptions) -> tuple[Any, dict] | None:
    """Return the one authorized approval route, or None for zero/ambiguous/invalid routes."""
    selected = []
    for sub in subscriptions:
        eligible = _eligible_subscription(runner, sub)
        if eligible is not None:
            selected.append(eligible)
    return selected[0] if len(selected) == 1 else None


def approval_subscription_for_task(runner, db_path, task_id):
    """Read one task's current routes without turning a DB/routing failure into permission."""
    from pathlib import Path
    from hermes_cli import kanban_db_notify as kbn
    from hermes_cli.kanban_db_connect import connect_closing

    try:
        with connect_closing(Path(db_path)) as conn:
            subscriptions = kbn.list_notify_subs(conn, task_id)
    except Exception:
        logger.debug("Kanban approval subscription lookup failed for %s", task_id, exc_info=True)
        return None
    # Notification subscriptions are inherited by child cards. Approval authority is not:
    # only the explicit subscription for this exact task may mint a broker capability.
    subscriptions = [
        sub for sub in subscriptions
        if (sub.get("delivery_metadata") or {}).get(APPROVAL_FORWARDING_TASK_KEY) == task_id
    ]
    return select_approval_subscription(runner, subscriptions)


def _subscription_authorization_home(runner, sub) -> str:
    """Freeze the selected notifier's home; never resolve it from the worker's callback scope."""
    home = _subscription_home(runner, sub)
    if home is None:
        from hermes_constants import get_process_hermes_home
        home = get_process_hermes_home()
    return str(home)


@dataclass(frozen=True)
class ForwardedApproval:
    """Process-local, exact authority/execution/request/notifier binding carried by one card."""

    authority: Any
    session_id: str
    generation: int
    route: str
    request_id: str
    user_id: str
    chat_id: str
    thread_id: str | None
    task_id: str
    run_id: int
    notifier_profile: str
    notifier_home: str


def authorize_forwarded_callback(adapter, binding: ForwardedApproval, **context) -> bool:
    """Re-read the current allowlist under the immutable notifier scope, failing closed."""
    from pathlib import Path
    try:
        from gateway.run import _profile_runtime_scope
        with _profile_runtime_scope(Path(binding.notifier_home), hydrate_secrets=False):
            return bool(adapter._is_callback_user_authorized(**context))
    except Exception:
        logger.debug("Kanban approval callback authorization scope failed for task %s request %s",
                     binding.task_id, binding.request_id, exc_info=True)
        return False


def log_forwarded_callback_refusal(binding: ForwardedApproval, reason: str, *, user_id, chat_id, thread_id) -> None:
    """Record only hashed authorization coordinates — never paths, prompt text, tool arguments, or credentials."""
    from gateway.platforms.helpers import hash_for_log
    logger.warning(
        "Kanban forwarded Telegram approval callback refused reason=%s task_id=%s request_id=%s "
        "notifier_profile=%s expected_user=%s expected_chat=%s expected_thread=%s "
        "actual_user=%s actual_chat=%s actual_thread=%s",
        reason, binding.task_id, binding.request_id, binding.notifier_profile,
        hash_for_log(binding.user_id), hash_for_log(binding.chat_id), hash_for_log(binding.thread_id),
        hash_for_log(user_id), hash_for_log(chat_id), hash_for_log(thread_id))


def _withdraw(authority, session_id, generation, request_id, cause):
    live = authority.sessions[session_id]
    return live.controls.withdraw(session_id, generation, request_id, cause)


async def forward_approval(authority, row, data, timeout_s=None):
    """Forward one registered prompt, or withdraw it when no authorized transport remains reachable."""
    session_id, generation = row["target_session_id"], row["generation"]
    live = authority.sessions[session_id]
    request_id = data["request_id"]
    card = row["payload"]["kanban_card_v1"]
    selected = await asyncio.to_thread(
        approval_subscription_for_task, authority.runner, card.get("db"), card["task_id"])
    if selected is None:
        _withdraw(authority, session_id, generation, request_id, "no reachable authorized Telegram subscriber")
        return None
    adapter, sub = selected
    entry = live.controls.pending.get(request_id)
    if entry is None:
        return None  # answered/timed out while route preflight was running
    _, prompt = entry
    binding = ForwardedApproval(
        authority=authority, session_id=session_id, generation=generation, route=live.route,
        request_id=request_id, user_id=str(sub["user_id"]), chat_id=str(sub["chat_id"]),
        thread_id=_subscription_thread_id(sub),
        task_id=card["task_id"], run_id=card["run_id"],
        notifier_profile=str(sub["notifier_profile"]),
        notifier_home=_subscription_authorization_home(authority.runner, sub))
    metadata = dict(sub.get("delivery_metadata") or {})
    metadata["hermes_profile"] = sub["notifier_profile"]
    if binding.thread_id is not None:
        metadata["thread_id"] = binding.thread_id
    metadata[FORWARDING_METADATA_KEY] = binding

    async def deliver():
        return await adapter.send_exec_approval(
            chat_id=binding.chat_id, command=prompt["command"], description=prompt["description"],
            session_key=live.route, allow_session=False, allow_permanent=False,
            smart_denied=bool(data.get("smart_denied", False)), request_id=request_id,
            metadata=metadata)

    try:
        authority.check_approval_generation(session_id, generation)
        home = _subscription_home(authority.runner, sub)
        if home is None:
            result = await deliver()
        else:
            from gateway.run import _async_profile_runtime_scope
            async with _async_profile_runtime_scope(home):
                result = await deliver()
    except Exception:
        logger.warning("Kanban approval delivery raised for task %s request %s",
                       binding.task_id, request_id, exc_info=True)
        _withdraw(authority, session_id, generation, request_id, "authorized Telegram approval delivery failed")
        return None
    if not getattr(result, "success", False):
        _withdraw(authority, session_id, generation, request_id, "authorized Telegram approval delivery failed")
        return None
    return binding


def schedule_approval_forward(authority, row, data, timeout_s=None):
    """Track delivery tasks so tests and shutdown can observe settlement; never leave a prompt parked."""
    live = authority.sessions[row["target_session_id"]]
    try:
        # In-process agent threads have no running loop; the session drain owns the canonical one.
        loop = live.task.get_loop() if live.task is not None else asyncio.get_running_loop()
    except RuntimeError:
        _withdraw(authority, row["target_session_id"], row["generation"], data["request_id"],
                  "authorized Telegram approval delivery unavailable")
        return
    future = asyncio.run_coroutine_threadsafe(forward_approval(authority, row, data, timeout_s), loop)
    task = asyncio.wrap_future(future, loop=loop)
    tasks = getattr(authority, "_kanban_approval_tasks", None)
    if tasks is None:
        tasks = authority._kanban_approval_tasks = set()
    tasks.add(task)
    task.add_done_callback(tasks.discard)


async def respond_forwarded(binding, choice, *, user_id, chat_id, thread_id):
    """Resolve through the canonical authority path after exact audience and one-shot choice checks."""
    if (choice not in {"once", "deny"}
            or str(user_id) != binding.user_id or str(chat_id) != binding.chat_id
            or (str(thread_id) if thread_id is not None else None) != binding.thread_id):
        return 0
    from gateway.session_contract import SessionActor, SessionRef

    authority = binding.authority
    ref = SessionRef(authority.profile_id, binding.session_id)
    actor = SessionActor(
        f"kanban-telegram:{binding.user_id}", authority.profile_id,
        frozenset({"session:read", "session:approve", "session:operator"}),
        "kanban-telegram:" + uuid.uuid4().hex)
    snapshot = await authority.attach(actor, ref, history=False)
    try:
        if not any(prompt["prompt_id"] == binding.request_id and prompt["kind"] == "approval"
                   and prompt["execution_generation"] == binding.generation
                   for prompt in snapshot.prompts):
            return 0
        result = await authority.respond(
            actor, ref, binding.generation, binding.request_id, {"choice": choice}, kind="approval")
        return int(result.get("status") == "resolved")
    finally:
        await authority.detach(actor, snapshot.subscription_id)
