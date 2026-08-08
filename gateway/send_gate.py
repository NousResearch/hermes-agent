"""Structural send gate: a per-platform config switch that makes outbound
sending impossible rather than merely discouraged.

Some deployments need an assurance stronger than "we audited the code and
nothing calls send()". Setting::

    platforms:
      slack:
        extra:
          send_gate: disabled

makes every outbound call on that platform raise :class:`SendGateDisabledError`
before any network I/O happens.

Where the gate is installed
---------------------------
The gate lives at the two chokepoints every outbound path funnels through, so
it does not have to be re-applied per adapter (and cannot rot when a new
adapter or a new ``send_*`` helper is added):

1. **Adapter layer** — ``gateway/platforms/base.py`` wraps every content-bearing
   ``send*`` coroutine on ``BasePlatformAdapter`` and, via ``__init_subclass__``,
   on every subclass at class-creation time. That covers the live in-process
   adapters (``plugins/platforms/*/adapter.py``), the built-in adapters under
   ``gateway/platforms/``, ``APIServerAdapter``, and the native media helpers
   (``send_image``/``send_voice``/``send_video``/``send_document``/…) that
   ``cron/scheduler.py`` calls directly.

2. **Standalone sender layer** — ``tools/send_message_tool.py::_send_to_platform``
   gates the out-of-process path, where there is no live adapter instance and
   delivery goes through a plugin's ``standalone_sender_fn`` (or a native
   helper, as Weixin does). That path never touches an adapter object, so the
   adapter wrapper alone would miss it.

Design rules
------------
* **Fail open on config errors.** A malformed or unreadable config must not
  silently wedge delivery for every platform; only an explicit
  ``send_gate: disabled`` blocks. A gate that fails closed on a typo is an
  outage, and operators would learn to stop using it.
* **Presence signals are exempt.** ``send_typing`` and ``send_read_receipt``
  carry no content and are called from paths that do not expect exceptions
  (the ``_keep_typing`` refresh loop). They stay in
  :data:`SEND_GATE_EXEMPT_METHODS`.
* **No runtime spoofing.** The value is read from the platform config the
  adapter was constructed with, not from the environment or from call
  arguments.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: ``extra`` key operators set to control the gate.
SEND_GATE_KEY = "send_gate"

#: The only value that blocks. Anything else (including absent) allows sends.
SEND_GATE_DISABLED = "disabled"

#: Outbound coroutines exempt from the gate. These transmit no content: they
#: are presence/acknowledgement signals invoked from loops that treat an
#: exception as a bug rather than a policy decision.
SEND_GATE_EXEMPT_METHODS = frozenset({
    "send_typing",
    "send_read_receipt",
})


class SendGateDisabledError(RuntimeError):
    """Raised by an outbound call on a platform whose send gate is disabled."""


def _extra_of(platform_config: Any) -> dict:
    """Best-effort read of a platform config's ``extra`` mapping.

    Tolerates ``None``, objects built via ``object.__new__()`` without
    ``__init__`` (several gateway tests do this), and plain dicts.
    """
    if platform_config is None:
        return {}
    if isinstance(platform_config, dict):
        extra = platform_config.get("extra")
    else:
        extra = getattr(platform_config, "extra", None)
    return extra if isinstance(extra, dict) else {}


def is_send_blocked(platform_config: Any) -> bool:
    """Return True only when *platform_config* explicitly disables sending."""
    try:
        raw = _extra_of(platform_config).get(SEND_GATE_KEY)
        if raw is None:
            return False
        return str(raw).strip().lower() == SEND_GATE_DISABLED
    except Exception:
        # Fail open: a config we cannot read must not block delivery.
        logger.debug(
            "send_gate: unreadable platform config; allowing send", exc_info=True
        )
        return False


def send_gate_message(
    platform_name: Optional[str], operation: Optional[str] = None
) -> str:
    """Human-facing explanation of a block, including how to undo it."""
    name = platform_name or "this platform"
    what = f"{operation}() is" if operation else "Sending is"
    return (
        f"{what} blocked on '{name}': platforms.{name}.extra.{SEND_GATE_KEY} is "
        f"'{SEND_GATE_DISABLED}'. To re-enable, remove that setting (or set it to "
        f"'enabled') and restart the gateway."
    )


def assert_send_allowed(
    platform_name: Optional[str],
    platform_config: Any,
    operation: Optional[str] = None,
) -> None:
    """Raise :class:`SendGateDisabledError` if the gate blocks this platform."""
    if is_send_blocked(platform_config):
        raise SendGateDisabledError(send_gate_message(platform_name, operation))


def platform_name_of(obj: Any) -> Optional[str]:
    """Extract a printable platform name from an adapter or Platform enum."""
    platform = getattr(obj, "platform", obj)
    value = getattr(platform, "value", None)
    if isinstance(value, str):
        return value
    if platform is None:
        return None
    return str(platform)
