"""Persisted home-channel shutdown-notification dedup (#71180).

WSL suspend/resume (and other host suspend/resume cycles) restart the gateway
process. Each new process re-broadcasts the "gateway shutting down" notice to
every home channel. The in-memory ``notified`` set in
``_notify_active_sessions_of_shutdown`` is per-process, so it cannot stop the
flood. A state file under HERMES_HOME records the last-sent timestamp per
destination. Fail open: an unreadable or corrupt file never silences a
legitimate broadcast. A future-dated timestamp is clamped to now and rewritten
so a skewed clock cannot pin suppression forever.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger("gateway.run")

# Default cooldown for home-channel shutdown notifications, in seconds.
# Override via config.yaml ``gateway.shutdown_notify_cooldown``. 0 disables dedup.
DEFAULT_SHUTDOWN_NOTIFY_COOLDOWN_SECONDS: int = 60

_STATE_NAME = ".shutdown_notify_sent.json"


def _shutdown_notify_home() -> Path:
    """Home the state file lives in.

    ``gateway.run._hermes_home`` is the process home (and the binding tests
    monkeypatch). Fall back to ``get_hermes_home()`` only if that binding is
    missing, so a HERMES_HOME redirect still works.
    """
    try:
        import gateway.run as gr

        bound = getattr(gr, "_hermes_home", None)
        if bound is not None:
            return Path(bound)
    except Exception as exc:
        logger.debug("shutdown-notify home binding unavailable: %s", exc)
    from hermes_constants import get_hermes_home

    return get_hermes_home()


def _shutdown_notify_cooldown_seconds() -> int:
    """Configured cooldown in seconds. 0 disables dedup.

    ``HERMES_GATEWAY_SHUTDOWN_NOTIFY_COOLDOWN`` wins when set (tests and
    emergency). Otherwise ``gateway.shutdown_notify_cooldown`` in config.yaml.
    Unparseable values fall back to the default.
    """
    raw = os.environ.get("HERMES_GATEWAY_SHUTDOWN_NOTIFY_COOLDOWN")
    if raw is None:
        try:
            from hermes_cli.config import read_user_config_raw

            cfg = read_user_config_raw(_shutdown_notify_home() / "config.yaml") or {}
            gw = cfg.get("gateway") if isinstance(cfg, dict) else None
            if isinstance(gw, dict) and "shutdown_notify_cooldown" in gw:
                raw = gw.get("shutdown_notify_cooldown")
        except Exception as exc:
            logger.debug("shutdown_notify_cooldown config read failed: %s", exc)
            raw = None
    if raw is not None:
        try:
            val = int(raw)
            if val < 0:
                raise ValueError
            return val
        except (TypeError, ValueError):
            logger.debug(
                "Unparseable gateway.shutdown_notify_cooldown=%r, falling back to default %ds",
                raw, DEFAULT_SHUTDOWN_NOTIFY_COOLDOWN_SECONDS,
            )
    return DEFAULT_SHUTDOWN_NOTIFY_COOLDOWN_SECONDS


def _shutdown_notify_sent_path() -> Path:
    """JSON map of destination key -> last-sent epoch, under HERMES_HOME."""
    return _shutdown_notify_home() / _STATE_NAME


def _shutdown_notify_dest_key(
    platform_value: str, chat_id: str, thread_id: str | None, profile: str | None = None,
) -> str:
    """Stable key for one home-channel destination. Profile is included when set."""
    key = f"{platform_value}:{chat_id}:{thread_id}" if thread_id else f"{platform_value}:{chat_id}"
    if profile:
        return f"{key}:profile={profile}"
    return key


def _load_state(path: Path) -> dict | None:
    """Return the state dict, or None when missing/corrupt (caller fails open)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        logger.debug("shutdown-notify dedup state unreadable (%s); sending", exc)
        return None
    if not isinstance(data, dict):
        logger.debug("shutdown-notify dedup state not a dict (%r); sending", type(data))
        return None
    return data


def _should_suppress_shutdown_notify(dest_key: str, cooldown_seconds: int) -> bool:
    """True when a notification was sent to *dest_key* within the cooldown.

    Fail-open on any read error. A cooldown of 0 always returns False.
    A future-dated stamp is clamped to now and rewritten so it cannot suppress forever.
    """
    if cooldown_seconds <= 0:
        return False
    path = _shutdown_notify_sent_path()
    data = _load_state(path)
    if data is None:
        return False
    last_sent_raw = data.get(dest_key)
    if not last_sent_raw:
        return False
    try:
        last_sent = float(last_sent_raw)
    except (TypeError, ValueError):
        logger.debug("shutdown-notify dedup timestamp unparseable (%r); sending", last_sent_raw)
        return False
    now = time.time()
    if last_sent > now:
        last_sent = now
        try:
            data[dest_key] = now
            from utils import atomic_json_write

            atomic_json_write(path, data)
        except Exception as exc:
            logger.debug("Failed to rewrite future shutdown-notify timestamp: %s", exc)
    return (now - last_sent) < cooldown_seconds


def _record_shutdown_notify_sent(dest_key: str) -> None:
    """Best-effort persist of a just-sent notification. A write failure may re-broadcast; that is safer."""
    try:
        path = _shutdown_notify_sent_path()
        data = _load_state(path)
        if data is None:
            data = {}
        data[dest_key] = time.time()
        from utils import atomic_json_write

        atomic_json_write(path, data)
    except Exception as exc:
        logger.debug("Failed to persist shutdown-notify timestamp: %s", exc)
