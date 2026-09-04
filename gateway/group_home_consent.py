"""Short-lived, requester-bound audience consent on the existing home config."""

from __future__ import annotations

import asyncio
import contextvars
import re
import secrets
import time
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from functools import wraps
from threading import RLock

from gateway.group_home_identity import (
    acknowledgement,
    audience_accepted,
    home_identity,
    home_thread_from_source,
    is_private_source,
    trusted_person,
)

PROCEED = object()
PROMPT_SECONDS = 120
MAX_PENDING = 128
_active_confirmation = contextvars.ContextVar("group_home_confirmation", default=None)


class ControlResult(str):
    """Keep an internal, non-sensitive outcome through nested command handlers."""

    def __new__(cls, value, safe_key):
        result = super().__new__(cls, value)
        result.safe_key = safe_key
        return result


def control_result(value, safe_key):
    return ControlResult(value, safe_key)


class DisclosureChanged(RuntimeError):
    """A private continuation no longer belongs to its original authorization."""


async def disclosed_call(runner, event, stamp, function, *args, _work_action=None, **kwargs):
    def require_current():
        if stamp is None or stamp != _disclosure_stamp(runner, event):
            raise DisclosureChanged(
                text("expired", command_prefix=runner._typed_command_prefix_for(event.source))
            )

    def invoke():
        require_current()
        return function(*args, **kwargs)

    if _work_action is None:
        result = await asyncio.to_thread(invoke)
    else:
        from gateway.group_chat_work import run_group_command_work

        result = await run_group_command_work(runner, _work_action, invoke)
    require_current()
    return result


def text(key, *, command_prefix="/"):
    from agent.i18n import t

    return t("gateway.group_home." + key, command_prefix=command_prefix)


def emergency(event):
    words = str(event.text or "").split()
    return (
        len(words) >= 3
        and words[1].isdecimal()
        and words[2].casefold() in {"stop", "deny"}
    )


def denial(runner, event):
    from gateway.hosted_room_messaging import is_machine_authored, is_message_edit

    if is_machine_authored(event):
        return text("people")
    if is_message_edit(event):
        return text("edited")
    if not trusted_person(event):
        return text("identity")
    from gateway.group_chat_policy import group_policy_for_source

    prefix = runner._typed_command_prefix_for(event.source)
    policy = group_policy_for_source(runner, event.source)
    if (
        policy.enabled
        and policy.is_admin(event.source.user_id)
        and not is_private_source(event.source)
    ):
        return text("binding", command_prefix=prefix)
    return text("setup", command_prefix=prefix)


def slash_denial(command, *, command_prefix="/"):
    return text("setup", command_prefix=command_prefix) if command == "group" else None


def _key(runner, event):
    from gateway.hosted_room_messaging import messaging_transport_profile
    from hermes_cli.config import get_config_path

    source = event.source
    return (
        str(get_config_path()),
        runner._group_chat_profile(event),
        messaging_transport_profile(event),
        source.platform.value,
        str(source.chat_id),
        str(home_thread_from_source(source) or ""),
        str(source.user_id or ""),
        str(source.scope_id or ""),
    )


def _disclosure_stamp(runner, event, *, require_audience=True):
    from gateway.group_chat_policy import receiving_group_transport

    if not _confirmation_allows_output(runner):
        return None
    if not runner._can_control_group_chats(event, require_audience=require_audience):
        return None
    home = runner.config.get_home_channel(event.source.platform)
    receiver = receiving_group_transport(runner, event.source)
    return (
        _key(runner, event),
        home_identity(home) if home else None,
        id(receiver[0]) if receiver is not None else None,
    )


def protect_group_result(function):
    @wraps(function)
    async def guarded(runner, event, *args, **kwargs):
        if runner._group_chat_command_args(event).strip().casefold() in {
            "help",
            "usage",
            "?",
            "cancel",
        }:
            return await function(runner, event, *args, **kwargs)
        stamp = _disclosure_stamp(runner, event)
        result = await function(runner, event, *args, **kwargs)
        if emergency(event) and (
            stamp is None or stamp != _disclosure_stamp(runner, event)
        ):
            safe_key = (
                result.safe_key
                if isinstance(result, ControlResult)
                else "control_unavailable"
            )
            return control_result(text(safe_key), safe_key)
        if stamp is not None and stamp != _disclosure_stamp(runner, event):
            return _changed_access(runner, event)
        return result

    return guarded


def _changed_access(runner, event):
    if not runner._can_control_group_chats(event, require_audience=False):
        return denial(runner, event)
    return text("expired", command_prefix=runner._typed_command_prefix_for(event.source))


def protect_group_callback(runner, event):
    stamp, context = _disclosure_stamp(runner, event), contextvars.copy_context()
    prefix = runner._typed_command_prefix_for(event.source)
    # A successfully displayed chooser outlives the one-time confirmation.
    context.run(_active_confirmation.set, None)

    def decorate(function):
        @wraps(function)
        async def guarded(*args, **kwargs):
            async def apply():
                if stamp is None or stamp != _disclosure_stamp(runner, event):
                    return text("expired", command_prefix=prefix)
                result = await function(*args, **kwargs)
                return (
                    result
                    if stamp == _disclosure_stamp(runner, event)
                    else text("expired", command_prefix=prefix)
                )

            return await asyncio.create_task(apply(), context=context.copy())

        return guarded

    return decorate


def _pending(runner):
    pending = getattr(runner, "_group_home_confirmations", None)
    if not isinstance(pending, OrderedDict):
        pending = runner._group_home_confirmations = OrderedDict()
    for key, value in list(pending.items()):
        if value.deadline <= time.monotonic():
            _retire(runner, value)
    return pending


@dataclass
class Confirmation:
    key: tuple
    home: tuple
    token: str
    deadline: float
    adapter: object
    context: contextvars.Context
    state: str = "pending"
    disclose: bool = True
    commit_started: bool = False
    lock: object = field(default_factory=RLock, repr=False)


def _current(runner, pending):
    return getattr(runner, "_group_home_confirmations", {}).get(pending.key) is pending


def _confirmation_allows_output(runner):
    active = _active_confirmation.get()
    if active is None or active[0] is not runner:
        return True
    pending = active[1]
    with pending.lock:
        return (
            _current(runner, pending)
            and pending.disclose
            and pending.deadline > time.monotonic()
        )


def _discard(runner, pending):
    if _current(runner, pending):
        runner._group_home_confirmations.pop(pending.key)


def _retire(runner, pending):
    # Never hold this short state lock across config reads/writes or an await.
    with pending.lock:
        pending.disclose = False
        if pending.state == "committing":
            return True
        if pending.state in {"pending", "claimed"}:
            pending.state = "cancelled"
        _discard(runner, pending)
        return pending.commit_started


def _cancel(runner, event, pending=None):
    prefix = runner._typed_command_prefix_for(event.source)
    if pending is None:
        pending = _pending(runner).get(_key(runner, event))
    if pending is not None:
        return text(
            "cancel_late" if _retire(runner, pending) else "cancel", command_prefix=prefix
        )
    accepted = not is_private_source(event.source) and audience_accepted(
        runner.config, event.source
    )
    return text("cancel_late" if accepted else "cancel", command_prefix=prefix)


def _check(runner, event, pending):
    from gateway.group_chat_policy import receiving_group_transport

    with pending.lock:
        if not _current(runner, pending) or pending.state in {"cancelled", "failed"}:
            raise PermissionError
    if (
        pending.deadline <= time.monotonic()
        or _key(runner, event) != pending.key
        or not runner._can_control_group_chats(event, require_audience=False)
    ):
        raise PermissionError
    receiver = receiving_group_transport(runner, event.source)
    home = runner.config.get_home_channel(event.source.platform)
    if (
        receiver is None
        or receiver[0] is not pending.adapter
        or home is None
        or home_identity(home) != pending.home
    ):
        raise PermissionError
    return home


def _persist(runner, event, pending):
    try:
        return _persist_locked(runner, event, pending)
    except BaseException:
        with pending.lock:
            if pending.state in {"claimed", "committing"}:
                pending.state = "failed"
        raise


def _persist_locked(runner, event, pending):
    from gateway.config import HomeChannel
    from hermes_cli.config import _CONFIG_LOCK, load_config, save_config

    with _CONFIG_LOCK:
        live = _check(runner, event, pending)
        config = load_config()
        platform = config.get("platforms", {}).get(event.source.platform.value, {})
        raw = platform.get("home_channel")
        if (
            not isinstance(raw, dict)
            or home_identity(HomeChannel.from_dict(raw)) != pending.home
        ):
            raise PermissionError
        ack = acknowledgement(live)
        _check(runner, event, pending)
        with pending.lock:
            if not _current(runner, pending) or pending.state != "claimed":
                raise PermissionError
            # Cancellation cannot undo I/O once it starts. Keep the lock free
            # during disk work so cancellation and emergency controls can run.
            pending.state = "committing"
            pending.commit_started = True
        platform["home_channel"] = {**raw, "group_audience_ack": ack}
        save_config(config)
        saved = (
            load_config()
            .get("platforms", {})
            .get(event.source.platform.value, {})
            .get("home_channel", {})
        )
        if (
            saved.get("group_audience_ack") != ack
            or home_identity(HomeChannel.from_dict(saved)) != pending.home
        ):
            raise RuntimeError("save not confirmed")
        current = _check(runner, event, pending)
        current.group_audience_ack = ack
        with pending.lock:
            pending.state = "committed"


async def _confirm(runner, event, pending, *, native=False):
    prefix = runner._typed_command_prefix_for(event.source)
    with pending.lock:
        if not _current(runner, pending) or pending.state != "pending":
            return text("expired", command_prefix=prefix)
        pending.state = "claimed"
    try:
        await asyncio.to_thread(_persist, runner, event, pending)
        with pending.lock:
            if not _current(runner, pending) or not pending.disclose:
                return text(
                    "cancel_late" if pending.commit_started else "expired",
                    command_prefix=prefix,
                )
        _check(runner, event, pending)
        if not runner._can_control_group_chats(event):
            return text("expired", command_prefix=prefix)
        # Older one-shot pickers remove their state after the callback. Return
        # the existing text chooser there, avoiding destruction of a new picker.
        verb = (
            " list"
            if native
            and not getattr(type(pending.adapter), "supports_choice_pages", False)
            else ""
        )
        followup = replace(
            event, text=f"{runner._typed_command_prefix_for(event.source)}group{verb}"
        )
        token = _active_confirmation.set((runner, pending))
        try:
            result = await runner._handle_rooms_command(followup)
        finally:
            _active_confirmation.reset(token)
        with pending.lock:
            if not _current(runner, pending) or not pending.disclose:
                return text("cancel_late", command_prefix=prefix)
        return result or text("chooser")
    except asyncio.CancelledError:
        _retire(runner, pending)
        raise
    except PermissionError:
        return text("expired", command_prefix=prefix)
    except Exception:
        return text("failed", command_prefix=prefix)
    finally:
        with pending.lock:
            if pending.state != "committing":
                _discard(runner, pending)


def _verified_picker(adapter):
    method = getattr(type(adapter), "send_choice_picker", None)
    module = getattr(method, "__module__", "")
    if not isinstance(module, str):
        return False
    return module in {
        "plugins.platforms.telegram.adapter",
        "plugins.platforms.discord.adapter",
        "plugins.platforms.telegram.adapter_prompts",
        "plugins.platforms.discord.adapter_prompts",
    } or re.fullmatch(
        r"hermes_plugins\.platforms__(telegram|discord)(?:__home_[0-9a-f]{12})?\.adapter(?:_prompts)?",
        module,
    ) is not None


async def prepare_group_access(runner, event):
    prefix = runner._typed_command_prefix_for(event.source)
    if not _confirmation_allows_output(runner):
        return text("expired", command_prefix=prefix)
    query = runner._group_chat_command_args(event).strip().casefold()
    if query in {"help", "usage", "?"}:
        help_text = getattr(runner, "_group_chat_help", None)
        if callable(help_text):
            return help_text(prefix + "group")
        return text("help", command_prefix=prefix)
    if query == "cancel":
        return _cancel(runner, event)
    if not runner._can_control_group_chats(event, require_audience=False):
        return denial(runner, event)
    if query == "confirm":
        pending = _pending(runner).get(_key(runner, event))
        return (
            await _confirm(runner, event, pending)
            if pending is not None
            else text("expired", command_prefix=prefix)
        )
    if not emergency(event):
        previous = _pending(runner).get(_key(runner, event))
        active = _active_confirmation.get()
        same_confirmation = (
            active is not None and active[0] is runner and active[1] is previous
        )
        if previous is not None and not same_confirmation:
            with previous.lock:
                committing = previous.state == "committing"
                _retire(runner, previous)
                if committing:
                    return text("saving", command_prefix=prefix)
    if emergency(event) or audience_accepted(runner.config, event.source):
        return PROCEED
    from gateway.group_chat_policy import receiving_group_transport

    receiver = receiving_group_transport(runner, event.source)
    key = _key(runner, event)
    pending = Confirmation(
        key,
        home_identity(runner.config.get_home_channel(event.source.platform)),
        secrets.token_hex(16),
        time.monotonic() + PROMPT_SECONDS,
        receiver[0],
        contextvars.copy_context(),
    )
    prompts = _pending(runner)
    prompts[key] = pending
    while len(prompts) > MAX_PENDING:
        oldest = next(iter(prompts.values()))
        _retire(runner, oldest)
        if _current(runner, oldest):
            # A committing entry stays current until the writer settles.
            _retire(runner, pending)
            return text("saving", command_prefix=prefix)
    command = runner._typed_command_prefix_for(event.source) + "group"
    fallback = (
        text("warning")
        + f"\n{text('continue')}: {command} confirm\n{text('private')}: {command} cancel"
    )
    if not _verified_picker(pending.adapter):
        return fallback

    async def selected(chat_id, value):
        async def apply():
            destination = event.source.chat_id
            if event.source.platform.value == "discord" and event.source.thread_id:
                destination = event.source.thread_id
            if (
                str(chat_id) != str(destination)
                or _pending(runner).get(key) is not pending
                or value not in {pending.token + ":yes", pending.token + ":no"}
            ):
                return text("expired", command_prefix=prefix)
            if value.endswith(":no"):
                return _cancel(runner, event, pending)
            try:
                _check(runner, event, pending)
            except PermissionError:
                return text("expired", command_prefix=prefix)
            return await _confirm(runner, event, pending, native=True)

        return await asyncio.create_task(apply(), context=pending.context.copy())

    source = await asyncio.to_thread(
        runner._normalize_source_for_session_key, event.source
    )
    sent = await runner._try_send_choice_picker(
        event,
        runner._session_key_for_source(source),
        title=text("warning"),
        choices=[
            {"label": text("continue"), "value": pending.token + ":yes"},
            {"label": text("private"), "value": pending.token + ":no"},
        ],
        on_choice_selected=selected,
    )
    return None if sent else fallback
