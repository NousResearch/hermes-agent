"""Turn-level keepalive for ACP.

Fires a well-formed empty ``agent_message_chunk`` on a session at a configured
interval so ACP clients (e.g. the VS Code panel) don't hit their idle-timeout
and kill the process during long-running turns.

Design notes:
- One dedicated daemon thread per turn, waiting on a ``threading.Event`` with
  a computed timeout. ``mark_activity()`` just bumps an absolute deadline and
  signals the event — no per-delta ``threading.Timer`` churn during streaming.
- Payload is a valid ACP ``AgentMessageChunk`` with an empty ``TextContentBlock``
  (matches ``ACPAgent._history_message_update`` shape). Empty text so nothing
  visible surfaces in the client.
- Interval resolution order: explicit constructor arg → ``config.yaml``
  (``acp.keepalive_interval_s``) → default 45s. Behavioral config lives in
  ``config.yaml`` per AGENTS.md env-var-for-config policy — no ``HERMES_*``
  env var. Interval <= 0 disables the feature (``make_turn_keepalive``
  returns ``None``).
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any, Callable, Optional

from acp.schema import AgentMessageChunk, TextContentBlock

from acp_adapter.events import _send_update

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL_S = 45.0
_CONFIG_PATH = ("acp", "keepalive_interval_s")


def _coerce_float(value: Any) -> Optional[float]:
    # Reject bools explicitly (bool is a subclass of int, so float(True) == 1.0
    # would silently give the ACP keepalive a 1-second interval — never what
    # a config-file `true`/`false` was intending).
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    # Reject NaN and ±Inf. YAML accepts `.nan` / `.inf` as valid scalars, and
    # NaN in particular is toxic: `remaining = deadline - now` becomes NaN,
    # every comparison against 0 is False, and the loop either spins or fires
    # continuously instead of sleeping.
    if not math.isfinite(result):
        return None
    return result


def _read_config_interval() -> Optional[float]:
    """Read ``acp.keepalive_interval_s`` from config.yaml if present."""
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly()
    except Exception:
        return None
    node: Any = cfg
    for key in _CONFIG_PATH:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return _coerce_float(node)


def get_keepalive_interval(default: float = _DEFAULT_INTERVAL_S) -> float:
    """Resolve the keepalive interval in seconds.

    Precedence: config.yaml → provided default. No env-var override —
    behavioral config lives in ``config.yaml`` per AGENTS.md policy.
    """
    cfg_val = _read_config_interval()
    if cfg_val is not None:
        return cfg_val
    return default


class TurnKeepalive:
    """Emit a well-formed empty agent_message_chunk every ``interval_s`` seconds.

    A single daemon thread parks on ``threading.Event.wait(timeout)``. Real
    activity calls ``mark_activity()``, which just pushes the next-fire
    deadline forward and pokes the event so the thread recomputes its wait.
    """

    def __init__(
        self,
        conn: Any,
        session_id: str,
        loop: Any,
        interval_s: Optional[float] = None,
        payload_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.conn = conn
        self.session_id = session_id
        self.loop = loop
        self.interval_s = (
            interval_s if interval_s is not None else get_keepalive_interval()
        )
        self.payload_factory = payload_factory or self._default_payload_factory
        self._stopped = False
        self._started = False
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._next_fire = 0.0  # monotonic deadline
        self._thread: Optional[threading.Thread] = None
        logger.debug(
            "[TurnKeepalive] init session=%s interval=%.3fs",
            self.session_id,
            self.interval_s,
        )

    # -- payload ---------------------------------------------------------
    @staticmethod
    def _default_payload_factory() -> AgentMessageChunk:
        """Empty but well-formed ACP agent_message_chunk (no visible text)."""
        return AgentMessageChunk(
            session_update="agent_message_chunk",
            content=TextContentBlock(type="text", text=""),
        )

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            if self._stopped or self._started:
                return
            if self.interval_s <= 0:
                # Defensive: factory normally screens this out. If someone
                # constructs TurnKeepalive directly with a non-positive
                # interval, refuse to spawn a hot-loop worker.
                logger.warning(
                    "[TurnKeepalive] interval_s=%.3f <= 0; not starting (session=%s)",
                    self.interval_s,
                    self.session_id,
                )
                self._stopped = True
                return
            self._started = True
            self._next_fire = time.monotonic() + self.interval_s
            self._thread = threading.Thread(
                target=self._run,
                name=f"acp-keepalive-{self.session_id}",
                daemon=True,
            )
            self._thread.start()
            logger.debug("[TurnKeepalive] start session=%s", self.session_id)

    def stop(self) -> None:
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            self._wake.set()
            thread = self._thread
        # Join outside the lock so the worker can acquire it and exit.
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        logger.debug("[TurnKeepalive] stop session=%s", self.session_id)

    def mark_activity(self) -> None:
        """Push the next-fire deadline out by one full interval. Cheap: no thread ops."""
        with self._lock:
            if self._stopped:
                return
            self._next_fire = time.monotonic() + self.interval_s
            self._wake.set()

    # -- worker ----------------------------------------------------------
    def _run(self) -> None:
        while True:
            with self._lock:
                if self._stopped:
                    return
                remaining = self._next_fire - time.monotonic()
            if remaining > 0:
                # Wait up to `remaining`; wake early on stop/activity.
                self._wake.wait(timeout=remaining)
                with self._lock:
                    self._wake.clear()
                    if self._stopped:
                        return
                    # Either activity extended the deadline, or we're due.
                    if time.monotonic() < self._next_fire:
                        continue
            # Deadline hit — fire, then reset the deadline for the next tick.
            try:
                _send_update(
                    self.conn, self.session_id, self.loop, self.payload_factory()
                )
            except Exception:
                logger.debug("Keepalive send failed", exc_info=True)
            with self._lock:
                if self._stopped:
                    return
                self._next_fire = time.monotonic() + self.interval_s


def make_turn_keepalive(
    conn: Any,
    session_id: str,
    loop: Any,
    interval_s: Optional[float] = None,
    payload_factory: Optional[Callable[[], Any]] = None,
) -> Optional[TurnKeepalive]:
    """Factory. Returns ``None`` when interval <= 0 (kill switch)."""
    interval = (
        interval_s if interval_s is not None else get_keepalive_interval()
    )
    if interval <= 0:
        return None
    return TurnKeepalive(conn, session_id, loop, interval, payload_factory)
