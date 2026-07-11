"""WhatsApp auth-relay for gateway authentication prompts.

Problem this solves
-------------------
Several Hermes authentication paths block the agent thread waiting for a
human response, but only work when the *originating* session is itself a
rich messaging platform:

* ``clarify`` / ``approval`` prompts deliver to WhatsApp **only** when the
  session that triggered them is a WhatsApp chat (they use the originating
  session's ``_status_adapter``).  For non-WhatsApp sessions (CLI
  background tasks, cron jobs, Telegram, ...) they time out silently.
* ``sudo`` password prompts (``tools/terminal_tool._prompt_for_sudo_password``)
  have a hard 45s timeout and **no gateway callback** — in messaging/cron
  mode they fail closed with no notification to anyone.
* ``secret`` capture (``tools/skills_tool``) **deliberately short-circuits**
  on messaging platforms, so a skill that needs an API key set up while the
  agent runs unattended never gets it.

The auth-relay routes **all** of these prompts to the operator's WhatsApp
number (configured explicitly via ``gateway.auth_relay.operator_chat``),
regardless of which session triggered them, and relays the operator's reply
back to the waiting agent thread.

Security model
--------------
* Opt-in. Nothing changes unless ``gateway.auth_relay.enabled`` is true AND a
  WhatsApp adapter is connected.
* The operator is identified by an explicit WhatsApp ID (``operator_chat``),
  cross-checked against the adapter's existing DM allowlist.  Relayed
  secrets / sudo passwords are ONLY accepted from that sender — an arbitrary
  inbound WhatsApp message can never resolve a relayed prompt.
* Secrets are persisted through the **same** trusted writer the CLI uses
  (``hermes_cli.config.save_env_value_secure`` → ``~/.hermes/.env`` with
  atomic replace + mode preservation).  They are never returned to the model
  as text and never logged.
* Approval command text is redacted before it is shown to the operator
  (reusing ``_redact_approval_command`` semantics — see
  ``tools/approval._redact_approval_command``).
* A timeout (configurable, default 120s) returns a safe sentinel so the
  agent thread unblocks and adapts instead of hanging forever.

Pluggability / footprint
------------------------
This is a *gateway* feature (notification/relay), not a new core tool, so it
adds zero permanent schema surface.  It is enabled entirely through
``config.yaml`` and the already-connected WhatsApp adapter.

Session routing
---------------
For clarify/approval the relay reuses the adapter's existing interactive-reply
maps (``_clarify_state`` / ``_exec_approval_state`` map
``clarify_id``/``approval_id`` → *originating* session key).  When a prompt is
sent to WhatsApp on behalf of a non-WhatsApp session, the operator's tap on
the WhatsApp button resolves the originating session's pending entry via the
existing ``resolve_gateway_clarify`` / ``resolve_gateway_approval`` callers —
no new resolution plumbing is needed for those two kinds.

For sudo/secret there is no pre-existing WhatsApp resolution path, so this
module owns a small per-token pending map and the WhatsApp adapter's
``_dispatch_auth_relay_reply`` handler routes the operator's reply back.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


# =========================================================================
# Config
# =========================================================================

# Default timeout (seconds) for a relayed sudo/secret prompt.  Long enough
# for the operator to notice + type on a phone, short enough that an
# abandoned prompt unblocks the agent thread reasonably soon.
DEFAULT_RELAY_TIMEOUT = 120


@dataclass
class AuthRelayConfig:
    """Parsed ``gateway.auth_relay`` configuration."""

    enabled: bool = False
    operator_chat: str = ""          # WhatsApp ID (wa_id) of the operator
    # Per-kind toggles (all default on once the feature is enabled).
    clarify: bool = True
    approval: bool = True
    secret: bool = True
    sudo: bool = True
    # Require the operator to reply "yes"/"skip" to a confirm prompt BEFORE a
    # sensitive value (secret/sudo) is actually relayed, as an anti-misdelivery
    # guard.  clarify/approval are informational so they don't need it.
    require_confirm: bool = True
    timeout: int = DEFAULT_RELAY_TIMEOUT

    @property
    def active(self) -> bool:
        """Feature is live only with an operator target (explicit or derived)."""
        return self.enabled and bool(self.operator_chat)


def _default_operator_chat() -> str:
    """Best-effort operator WhatsApp ID from the locally-stored agent config.

    Used when ``gateway.auth_relay.operator_chat`` is not set.  Prefers the
    Baileys self-chat owner (``WHATSAPP_ALLOWED_USERS``), then the Cloud API
    business phone (``WHATSAPP_CLOUD_PHONE`` / ``WHATSAPP_CLOUD_BUSINESS_PHONE``).
    Returns the normalized numeric identifier (no ``+``), or \"\".
    """
    try:
        from gateway.whatsapp_identity import normalize_whatsapp_identifier
    except Exception:
        normalize_whatsapp_identifier = lambda v: str(v or "").strip().lstrip("+")
    for env_var in (
        "WHATSAPP_ALLOWED_USERS",
        "WHATSAPP_CLOUD_PHONE",
        "WHATSAPP_CLOUD_BUSINESS_PHONE",
        "WHATSAPP_CLOUD_OWNER_WA_ID",
    ):
        raw = os.getenv(env_var, "").strip()
        if not raw:
            continue
        # WHATSAPP_ALLOWED_USERS is a comma-separated list.
        first = raw.split(",")[0].strip()
        norm = normalize_whatsapp_identifier(first)
        if norm:
            return norm
    return ""


# Module-level live config, set by the gateway at startup.  Kept simple and
# process-global because the gateway is a single long-lived process.
_LIVE_CFG: AuthRelayConfig = AuthRelayConfig()


def configure(cfg: AuthRelayConfig) -> None:
    """Install the live relay config.  Called once at gateway startup.

    When ``operator_chat`` is empty but the feature is enabled, fall back to
    the locally-stored WhatsApp owner number (Baileys ``WHATSAPP_ALLOWED_USERS``
    or the Cloud business phone) so the operator doesn't have to repeat a number
    already known to the agent.  If nothing can be derived, the feature stays
    inert (active == False) rather than failing closed on every auth request.
    """
    global _LIVE_CFG
    cfg = cfg or AuthRelayConfig()
    if cfg.enabled and not cfg.operator_chat:
        derived = _default_operator_chat()
        if derived:
            cfg = AuthRelayConfig(
                enabled=cfg.enabled,
                operator_chat=derived,
                clarify=cfg.clarify,
                approval=cfg.approval,
                secret=cfg.secret,
                sudo=cfg.sudo,
                require_confirm=cfg.require_confirm,
                timeout=cfg.timeout,
            )
            logger.info(
                "auth_relay: operator_chat not set; derived from local WhatsApp "
                "config -> %s",
                derived,
            )
        else:
            logger.warning(
                "auth_relay: enabled but no operator_chat and none derivable from "
                "local WhatsApp config; relay inactive."
            )
    _LIVE_CFG = cfg


def get_config() -> AuthRelayConfig:
    return _LIVE_CFG


def is_enabled() -> bool:
    return _LIVE_CFG.active


def kind_enabled(kind: str) -> bool:
    if not _LIVE_CFG.active:
        return False
    return bool(getattr(_LIVE_CFG, kind, False))


# =========================================================================
# Pending sudo/secret entries (operator-relayed credentials)
# =========================================================================

@dataclass
class _RelayEntry:
    """One pending sudo/secret prompt awaiting the operator's WhatsApp reply."""

    relay_id: str
    kind: str                   # "sudo" | "secret"
    event: threading.Event = field(default_factory=threading.Event)
    result: Optional[str] = None
    # For secret: the env var name we will persist the value under.
    var_name: Optional[str] = None
    prompt: Optional[str] = None
    metadata: Optional[dict] = None


_lock = threading.RLock()
# relay_id → _RelayEntry
_entries: Dict[str, _RelayEntry] = {}


def _register(kind: str, var_name: Optional[str], prompt: Optional[str],
              metadata: Optional[dict]) -> _RelayEntry:
    entry = _RelayEntry(
        relay_id=uuid.uuid4().hex[:16],
        kind=kind,
        var_name=var_name,
        prompt=prompt,
        metadata=metadata,
    )
    with _lock:
        _entries[entry.relay_id] = entry
    return entry


def _resolve(relay_id: str, value: str) -> bool:
    """Resolve a pending relay entry.  Returns True if found + resolved.

    Does NOT pop the entry — ``wait_for_response`` consumes it after the
    waiter wakes.  This avoids a race where an inbound reply arrives (and
    resolves) before the blocked waiter calls ``wait_for_response``, which
    would otherwise see the entry already gone and return None.
    """
    with _lock:
        entry = _entries.get(relay_id)
        if entry is None:
            return False
        entry.result = value
        entry.event.set()
        return True


def _cancel(relay_id: str) -> None:
    """Cancel a pending relay entry (skip/timeout).  Sets result=None + event."""
    with _lock:
        entry = _entries.get(relay_id)
        if entry is None:
            return
        entry.result = None
        entry.event.set()


def clear_all() -> None:
    """Cancel every pending relay entry (gateway shutdown / interrupt)."""
    with _lock:
        pending = list(_entries.values())
        _entries.clear()
    for entry in pending:
        entry.result = None
        entry.event.set()


def wait_for_response(relay_id: str, timeout: float) -> Optional[str]:
    """Block on the entry's event until resolved or timeout fires.

    Mirrors ``tools.clarify_gateway.wait_for_response`` — polls in 1s slices
    and touches the gateway's inactivity watchdog so a long wait doesn't get
    the agent killed while the operator is typing.
    """
    with _lock:
        entry = _entries.get(relay_id)
    if entry is None:
        return None
    try:
        from tools.environments.base import touch_activity_if_due
    except Exception:  # pragma: no cover
        touch_activity_if_due = None
    deadline = time.monotonic() + max(timeout, 0.0)
    activity_state = {"last_touch": time.monotonic(), "start": time.monotonic()}
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            # Timed out — drop the entry so a late reply can't resolve a dead
            # waiter (which would leak into a future unrelated prompt).
            _cancel(relay_id)
            return None
        if entry.event.wait(timeout=min(1.0, remaining)):
            with _lock:
                _entries.pop(relay_id, None)
            return entry.result
        if touch_activity_if_due is not None:
            touch_activity_if_due(activity_state, "waiting for operator auth reply")


# =========================================================================
# Gateway-wide callbacks (registered at startup when relay is active)
# =========================================================================

# These replace the no-op default callbacks on messaging/cron surfaces.  They
# send a WhatsApp message to the operator, then block on the relay entry's
# Event.  The WhatsApp adapter's inbound handler resolves the entry.

def _whatsapp_operator_adapter():
    """Best-effort lookup of a connected WhatsApp adapter.

    Imported lazily to avoid a hard dependency on the gateway runner at
    module import time.  Returns the adapter instance or None.
    """
    try:
        from gateway.run import _get_whatsapp_operator_adapter
        return _get_whatsapp_operator_adapter()
    except Exception:
        return None


def _send_to_operator(text: str) -> bool:
    """Deliver a relay prompt to the operator's WhatsApp chat.

    Returns True if the send was accepted by the adapter.
    """
    adapter = _whatsapp_operator_adapter()
    if adapter is None:
        return False
    op = _LIVE_CFG.operator_chat
    if not op:
        return False
    try:
        from gateway.run import safe_schedule_threadsafe
        loop = _get_gateway_loop()
        if loop is None:
            return False

        async def _do_send():
            return await adapter.send(op, text)

        fut = safe_schedule_threadsafe(_do_send(), loop, logger=logger,
                                        log_message="auth_relay send failed")
        if fut is None:
            return False
        result = fut.result(timeout=15)
        return bool(getattr(result, "success", False))
    except Exception as exc:
        logger.warning("auth_relay: failed to send prompt to operator: %s", exc)
        return False


def _get_gateway_loop():
    try:
        from gateway.run import _get_gateway_event_loop
        return _get_gateway_event_loop()
    except Exception:
        return None


def _secret_capture_callback(var_name: str, prompt: str, metadata=None) -> dict:
    """Gateway-wide secret capture callback → operator WhatsApp relay.

    Sends the operator a prompt, waits for the relayed value, persists it via
    the trusted CLI writer, and returns the same shape the CLI callback does.
    """
    if not kind_enabled("secret"):
        return {
            "success": False,
            "stored_as": var_name,
            "validated": False,
            "skipped": True,
            "message": "Auth relay secret capture is disabled.",
        }

    # Confirmation gate (anti-misdelivery): the operator must first ack the
    # prompt before we surface a field asking them to type the secret.
    # Skipped when require_confirm is disabled in config.
    require_confirm = bool(getattr(_LIVE_CFG, "require_confirm", True))
    if require_confirm:
        entry = _register("secret", var_name, prompt, metadata)
        rid = entry.relay_id
        confirm_msg = (
            f"🔐 *Secret required* for skill setup\n\n"
            f"Variable: `{var_name}`\n"
            f"{prompt or ''}\n\n"
            f"Reply *yes* to receive a secure input prompt, or *skip* to skip."
        )
        if not _send_to_operator(confirm_msg):
            _cancel(rid)
            return {
                "success": True,
                "reason": "relay_failed",
                "stored_as": var_name,
                "validated": False,
                "skipped": True,
                "message": "Could not reach operator over WhatsApp; secret setup skipped.",
            }

        # Wait for the operator's confirm reply.
        confirm = wait_for_response(rid, timeout=_LIVE_CFG.timeout)
        if confirm is None:
            return {
                "success": True,
                "reason": "timeout",
                "stored_as": var_name,
                "validated": False,
                "skipped": True,
                "message": "Secret setup timed out (operator did not confirm).",
            }
        if confirm.strip().lower() not in {"yes", "y"}:
            return {
                "success": True,
                "reason": "cancelled",
                "stored_as": var_name,
                "validated": False,
                "skipped": True,
                "message": "Secret setup skipped by operator.",
            }

    # Register for the actual value (new relay id so a stale reply can't
    # cross wires).  When confirmation was skipped this is the first prompt.
    entry2 = _register("secret", var_name, prompt, metadata)
    rid2 = entry2.relay_id
    value_msg = (
        f"🔐 *Enter secret for* `{var_name}`\n\n"
        f"Reply with the value (it will be stored securely in "
        f"`~/.hermes/.env` and never shown to the agent). "
        f"Reply *skip* to cancel."
    )
    if not _send_to_operator(value_msg):
        _cancel(rid2)
        return {
            "success": True,
            "reason": "relay_failed",
            "stored_as": var_name,
            "validated": False,
            "skipped": True,
            "message": "Could not reach operator over WhatsApp; secret setup skipped.",
        }

    value = wait_for_response(rid2, timeout=_LIVE_CFG.timeout)
    if value is None:
        return {
            "success": True,
            "reason": "timeout",
            "stored_as": var_name,
            "validated": False,
            "skipped": True,
            "message": "Secret entry timed out.",
        }
    if not value:
        return {
            "success": True,
            "reason": "cancelled",
            "stored_as": var_name,
            "validated": False,
            "skipped": True,
            "message": "Secret setup skipped.",
        }

    # Persist through the identical trusted writer the CLI uses.  Never
    # returns the value to the model; the skill reads it from the env/secret
    # scope later.
    try:
        from hermes_cli.config import save_env_value_secure
        stored = save_env_value_secure(var_name, value)
    except Exception as exc:  # pragma: no cover - storage failure
        logger.error("auth_relay: failed to persist secret %s: %s", var_name, exc)
        return {
            "success": False,
            "stored_as": var_name,
            "validated": False,
            "skipped": False,
            "message": f"Failed to store secret: {exc}",
        }
    return {
        **stored,
        "skipped": False,
        "message": "Secret stored securely via operator relay. The value was not exposed to the agent.",
    }


def _sudo_password_callback() -> str:
    """Gateway-wide sudo password callback → operator WhatsApp relay.

    Returns the password string (or "" on timeout/skip).  The terminal tool
    caches it for the session like the CLI path does.
    """
    if not kind_enabled("sudo"):
        return ""

    entry = _register("sudo", None, "sudo password required", None)
    rid = entry.relay_id

    msg = (
        "🔑 *Sudo password required*\n\n"
        "An agent task needs sudo privileges. Reply with the sudo password, "
        "or *skip* to continue without sudo."
    )
    if not _send_to_operator(msg):
        _cancel(rid)
        return ""

    value = wait_for_response(rid, timeout=_LIVE_CFG.timeout)
    if value is None or not value:
        # Timeout / skip → continue without sudo (matches CLI fail-closed).
        return ""
    return value


def install_gateway_callbacks() -> bool:
    """Register the relay callbacks when the feature is active.

    Returns True if callbacks were installed.  Idempotent per process.
    """
    if not is_enabled():
        return False
    # Sanity: a WhatsApp adapter must be reachable.
    if _whatsapp_operator_adapter() is None:
        logger.warning(
            "auth_relay: enabled but no WhatsApp adapter connected; "
            "not installing sudo/secret relay callbacks."
        )
        return False
    try:
        from tools.skills_tool import set_secret_capture_callback
        from tools.terminal_tool import set_sudo_password_callback
        set_secret_capture_callback(_secret_capture_callback)
        set_sudo_password_callback(_sudo_password_callback)
    except Exception as exc:
        logger.error("auth_relay: failed to install callbacks: %s", exc)
        return False
    logger.info(
        "auth_relay: installed sudo/secret relay callbacks → operator %s",
        _LIVE_CFG.operator_chat,
    )
    return True


def uninstall_gateway_callbacks() -> None:
    """Clear the relay callbacks (gateway shutdown / feature disabled)."""
    try:
        from tools.skills_tool import set_secret_capture_callback
        from tools.terminal_tool import set_sudo_password_callback
        set_secret_capture_callback(None)
        set_sudo_password_callback(None)
    except Exception:
        pass
    clear_all()


# ---------------------------------------------------------------------------
# Inbound (operator → relay) resolution API.  The WhatsApp adapter calls these
# from its inbound message handler to deliver a relayed secret/sudo value back
# to the blocked agent thread.  Operator-only upstream; the adapter already
# enforces that the sender is the configured operator_chat.
# ---------------------------------------------------------------------------
def resolve_secret_relay(relay_id: str, value: Optional[str]) -> bool:
    """Resolve a pending secret relay entry with the operator's value.

    ``value`` None means skip/cancel.  Returns True if an entry was resolved.
    """
    with _lock:
        exists = relay_id in _entries
    if value is None:
        if not exists:
            return False
        _cancel(relay_id)
        return True
    return _resolve(relay_id, value)


def resolve_sudo_relay(relay_id: str, value: Optional[str]) -> bool:
    """Resolve a pending sudo relay entry with the operator's password.

    ``value`` None means skip/cancel (continue without sudo).  Returns True if
    an entry was resolved.
    """
    with _lock:
        exists = relay_id in _entries
    if value is None:
        if not exists:
            return False
        _cancel(relay_id)
        return True
    return _resolve(relay_id, value)


def pending_secret_for(operator_id: str) -> Optional[str]:
    """Return the active secret relay id awaiting ``operator_id``'s reply.

    Returns None when there is no pending secret prompt.  The operator_id
    match is enforced when an explicit ``operator_chat`` is configured; in
    auto/self-chat mode (operator_chat derived or empty) the adapter itself
    gates the caller (e.g. Baileys only invokes this for ``fromOwner``
    messages), so the match check is relaxed.
    """
    explicit = _LIVE_CFG.operator_chat
    if explicit and operator_id != explicit:
        return None
    with _lock:
        for rid, entry in _entries.items():
            if entry.kind == "secret":
                return rid
    return None


def pending_sudo_for(operator_id: str) -> Optional[str]:
    """Return the active sudo relay id awaiting ``operator_id``'s reply."""
    explicit = _LIVE_CFG.operator_chat
    if explicit and operator_id != explicit:
        return None
    with _lock:
        for rid, entry in _entries.items():
            if entry.kind == "sudo":
                return rid
    return None
