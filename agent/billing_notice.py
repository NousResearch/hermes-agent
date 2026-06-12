"""Persistent billing notice state manager.

Records billing exhaustion events and determines when to surface
a gentle daily PS reminder to the user — never a standalone message.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class BillingNoticeManager:
    """Manage billing notice state persisted to ``~/.hermes/.billing_notice``.

    The state file is a simple JSON document recording the date a billing
    failure was detected and when the user was last reminded.  The manager
    ensures reminders are at-most-once-per-day, never stale, and always
    appended as a PS to a substantive response (never a standalone message).
    """

    STATE_FILE = os.path.expanduser("~/.hermes/.billing_notice")

    def record(self, provider: str, model: str) -> None:
        """Record a billing failure with today's date.

        Only overwrites if the detected date is *newer* than the existing
        record — this prevents an older re-detection from clearing a more
        recent reminder state.
        """
        try:
            state = self._read_state()
            existing_detected = state.get("detected_date") if state else None
            today = date.today().isoformat()

            # Only update if no existing record or this is a newer detection
            if existing_detected and existing_detected >= today:
                return

            new_state = {
                "detected_date": today,
                "last_reminded_date": None,
                "provider": provider,
                "model": model,
            }
            self._write_state(new_state)
            logger.info(
                "Billing notice recorded: provider=%s, model=%s, date=%s",
                provider, model, today,
            )
        except Exception as exc:
            logger.warning("Failed to record billing notice: %s", exc)

    def should_remind(self) -> bool:
        """Return ``True`` if a billing reminder should be shown.

        Conditions (all must hold):
        1. State file exists with a ``detected_date``
        2. Detected date is today or yesterday (not stale)
        3. ``last_reminded_date`` is NOT today (at most once per day)
        """
        try:
            state = self._read_state()
            if not state:
                return False

            detected = state.get("detected_date")
            last_reminded = state.get("last_reminded_date")
            if not detected:
                return False

            today = date.today().isoformat()

            # Only remind if detected today or yesterday (fresh enough)
            if detected != today:
                from datetime import timedelta
                yesterday = (date.today() - timedelta(days=1)).isoformat()
                if detected != yesterday:
                    return False

            # Don't remind if already reminded today
            if last_reminded == today:
                return False

            return True
        except Exception as exc:
            logger.warning("Failed to check billing reminder: %s", exc)
            return False

    def mark_reminded(self) -> None:
        """Update ``last_reminded_date`` to today."""
        try:
            state = self._read_state()
            if not state:
                return
            state["last_reminded_date"] = date.today().isoformat()
            self._write_state(state)
        except Exception as exc:
            logger.warning("Failed to mark billing reminder: %s", exc)

    def clear(self) -> None:
        """Remove the state file (billing has been resolved)."""
        try:
            if os.path.exists(self.STATE_FILE):
                os.remove(self.STATE_FILE)
                logger.info("Billing notice cleared (billing resolved)")
        except Exception as exc:
            logger.warning("Failed to clear billing notice: %s", exc)

    def get_state(self) -> Optional[Dict[str, Any]]:
        """Return the current state dict, or ``None`` if no state exists."""
        return self._read_state()

    def _read_state(self) -> Optional[Dict[str, Any]]:
        """Read and parse the state file, returning ``None`` on any error."""
        try:
            if not os.path.exists(self.STATE_FILE):
                return None
            with open(self.STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read billing notice state: %s", exc)
            return None

    def _write_state(self, state: Dict[str, Any]) -> None:
        """Atomically write the state file via temporary file + replace."""
        tmp = self.STATE_FILE + ".tmp"
        try:
            os.makedirs(os.path.dirname(self.STATE_FILE), exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.STATE_FILE)
        except OSError as exc:
            logger.warning("Failed to write billing notice state: %s", exc)
            try:
                if os.path.exists(tmp):
                    os.unlink(tmp)
            except OSError:
                pass
