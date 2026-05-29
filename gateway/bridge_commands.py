"""Safe command helpers for the CLI ↔ Telegram bridge MVP.

The bridge uses local opt-in binding plus emergency status/disable controls.
After binding, plain Telegram DM text can continue the linked CLI-originated
Hermes session through the normal gateway agent path. It never injects raw text
into a live CLI/PTY.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Callable, Optional

from hermes_constants import get_hermes_home

from .bridge import BridgeDecision, BridgeStateStore, BridgeVerdict
from .config import Platform
from .platforms.base import MessageEvent


def default_bridge_db_path() -> Path:
    return get_hermes_home() / "bridge" / "cli_telegram_bridge.sqlite"


def default_bridge_kill_switch_path() -> Path:
    return get_hermes_home() / "telegram_bridge.disabled"


def default_bridge_store() -> BridgeStateStore:
    return BridgeStateStore(
        default_bridge_db_path(),
        kill_switch_path=default_bridge_kill_switch_path(),
    )


def _parse_args(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def _option_value(tokens: list[str], *names: str) -> Optional[str]:
    for idx, token in enumerate(tokens):
        for name in names:
            if token == name and idx + 1 < len(tokens):
                return tokens[idx + 1]
            prefix = f"{name}="
            if token.startswith(prefix):
                return token[len(prefix):]
    return None


def _bridge_id_for_session(session_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(session_id))
    return f"bridge-{safe}"


def handle_local_bridge_command(
    command: str,
    *,
    session_id: str,
    store: BridgeStateStore | None = None,
) -> str:
    """Handle local CLI `/bridge ...` commands.

    Safety rule: only the local side may mint binding tokens. Telegram must
    consume a token; it cannot create a binding from scratch.
    """
    store = store or default_bridge_store()
    tokens = _parse_args(command)
    if len(tokens) < 2 or tokens[1] in {"help", "-h", "--help"}:
        return (
            "Usage:\n"
            "  /bridge bind telegram [--chat CHAT_ID] [--user USER_ID] [--ttl SECONDS]\n"
            "  /bridge status\n"
            "  /bridge off\n"
            "\nSafe A-mode: local CLI mints a short-lived token; Telegram consumes it with /bridge_bind <token>."
        )

    sub = tokens[1].lower()
    if sub == "bind":
        target = tokens[2].lower() if len(tokens) > 2 else ""
        if target != "telegram":
            return "Usage: /bridge bind telegram [--chat CHAT_ID] [--user USER_ID] [--ttl SECONDS]"
        ttl_raw = _option_value(tokens, "--ttl") or "600"
        try:
            ttl = int(ttl_raw)
        except ValueError:
            return "Invalid --ttl value. Use seconds, e.g. --ttl 600."
        ttl = max(30, min(ttl, 3600))
        chat_id = _option_value(tokens, "--chat", "--chat-id")
        user_id = _option_value(tokens, "--user", "--user-id")
        thread_id = _option_value(tokens, "--thread", "--thread-id")
        bridge_id = _bridge_id_for_session(session_id)
        token = store.create_binding_token(
            bridge_id=bridge_id,
            hermes_session_id=session_id,
            ttl_seconds=ttl,
            telegram_chat_id=chat_id,
            telegram_user_id=user_id,
            telegram_thread_id=thread_id,
        )
        scope = []
        if chat_id:
            scope.append(f"chat={chat_id}")
        if user_id:
            scope.append(f"user={user_id}")
        if thread_id:
            scope.append(f"thread={thread_id}")
        scope_text = f" Scope: {', '.join(scope)}." if scope else " Scope: first authorized Telegram DM to consume it."
        return (
            f"Bridge binding token created for Hermes session `{session_id}`.\n"
            f"Send this in the Telegram DM within {ttl}s:\n"
            f"/bridge_bind {token.token}\n"
            f"{scope_text}\n"
            "After Telegram consumes this token, plain DM text continues the linked Hermes session. "
            "Slash commands keep normal gateway semantics."
        )

    if sub == "status":
        return "Local bridge state store is available. Use /bridge bind telegram to create an opt-in token."

    if sub in {"off", "pause", "disable"}:
        store.disable_bridge()
        return f"Bridge disabled. Kill switch: {store.kill_switch_path}"

    if sub in {"on", "resume", "enable"}:
        store.enable_bridge()
        return "Bridge kill switch removed. Existing per-session bindings still enforce their own state."

    return "Unknown /bridge subcommand. Use /bridge help."


def _telegram_identity(event: MessageEvent) -> tuple[str, str, Optional[str]] | None:
    source = event.source
    if not source or source.platform != Platform.TELEGRAM or source.chat_type != "dm":
        return None
    if not source.chat_id or not source.user_id:
        return None
    return str(source.chat_id), str(source.user_id), str(source.thread_id) if source.thread_id else None


def _decision_message(decision: BridgeDecision, success: str) -> str:
    if decision.verdict is BridgeVerdict.ACCEPT:
        return success
    return f"Bridge request rejected: {decision.reason}"


def maybe_apply_gateway_bridge_binding(
    event: MessageEvent,
    *,
    session_key: str,
    session_store,
    store: BridgeStateStore | None = None,
    evict_cached_agent: Callable[[str], None] | None = None,
) -> BridgeDecision | None:
    """Route bound Telegram DM plain text into the linked CLI session.

    Returns:
      - None when the event is outside bridge scope (non-Telegram, non-DM,
        unbound identity, or a slash command that should keep normal gateway
        semantics).
      - ACCEPT when the session key was bound to the CLI session id.
      - REJECT when a known binding exists but pause/kill-switch blocks input.

    This never injects raw text into a PTY. It only repoints the normal gateway
    session key to the CLI-originated Hermes session so the standard agent path,
    approval handling, queueing, and transcript machinery remain in force.
    """
    store = store or default_bridge_store()
    if event.get_command():
        return None
    identity = _telegram_identity(event)
    if identity is None:
        return None
    chat_id, user_id, thread_id = identity
    decision = store.validate_telegram_direct_input(
        chat_id=chat_id,
        user_id=user_id,
        thread_id=thread_id,
    )
    if decision.verdict is BridgeVerdict.REJECT:
        if decision.reason == "no bridge binding for telegram identity":
            return None
        return decision

    session_store.get_or_create_session(event.source)
    switched = session_store.switch_session(session_key, decision.hermes_session_id or "")
    if switched is None:
        return BridgeDecision(BridgeVerdict.REJECT, "could not switch gateway session to bridge target")
    if evict_cached_agent is not None:
        evict_cached_agent(session_key)
    return decision


def handle_gateway_bridge_command(
    event: MessageEvent,
    *,
    store: BridgeStateStore | None = None,
) -> str:
    """Handle safe Telegram-side bridge commands.

    This is intentionally limited to DM-only bind/status/off/pause/resume. It
    does not forward arbitrary text into Hermes.
    """
    store = store or default_bridge_store()
    identity = _telegram_identity(event)
    if identity is None:
        return "CLI-Telegram bridge commands are Telegram DM only."
    chat_id, user_id, thread_id = identity
    command = event.get_command() or ""
    args = event.get_command_args().strip()

    if command in {"bridge_bind", "bridge-bind"}:
        token = args.split()[0] if args else ""
        if not token:
            return "Usage: /bridge_bind <token>"
        decision = store.consume_binding_token(
            token=token,
            telegram_chat_id=chat_id,
            telegram_user_id=user_id,
            telegram_thread_id=thread_id,
        )
        return _decision_message(
            decision,
            f"Bridge linked to Hermes session `{decision.hermes_session_id}`. Plain DM text now continues that session.",
        )

    if command in {"bridge_status", "bridge-status", "bridge"}:
        return store.describe_telegram_status(chat_id=chat_id, user_id=user_id, thread_id=thread_id)

    if command in {"bridge_off", "bridge-off"}:
        store.disable_bridge()
        return "Bridge disabled by Telegram kill switch. Remote input is now blocked."

    if command in {"bridge_pause", "bridge-pause"}:
        row = store.binding_for_telegram(chat_id=chat_id, user_id=user_id, thread_id=thread_id)
        if row is None:
            return "No bridge binding is active for this Telegram chat/user."
        store.pause_binding(row["bridge_id"], reason="paused from Telegram")
        return f"Bridge paused for Hermes session `{row['hermes_session_id']}`."

    if command in {"bridge_resume", "bridge-resume"}:
        row = store.binding_for_telegram(chat_id=chat_id, user_id=user_id, thread_id=thread_id)
        if row is None:
            return "No bridge binding is active for this Telegram chat/user."
        store.resume_binding(row["bridge_id"])
        return f"Bridge resumed for Hermes session `{row['hermes_session_id']}`."

    return "Unknown bridge command. Use /bridge_status or /bridge_bind <token>."
