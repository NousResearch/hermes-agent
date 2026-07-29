"""Durable per-route automatic reset policies for supported threads."""

from __future__ import annotations

import dataclasses
import json
import logging
import threading
from pathlib import Path
from typing import Literal

from gateway.config import SessionResetPolicy
from utils import atomic_json_write

logger = logging.getLogger(__name__)

ThreadResetResolution = Literal["override", "inherited", "invalid"]

_STATE_VERSION = 1


class ThreadResetPolicyStateError(RuntimeError):
    """Raised when a command cannot safely mutate thread reset state."""


def _policy_to_state(policy: SessionResetPolicy) -> dict[str, object]:
    """Return the strict persisted representation of a route policy."""
    if policy.mode == "none":
        return {"mode": "none"}
    if policy.mode != "daily":
        raise ValueError("thread reset policy mode must be 'daily' or 'none'")
    if (
        isinstance(policy.at_hour, bool)
        or not isinstance(policy.at_hour, int)
        or not 0 <= policy.at_hour <= 23
    ):
        raise ValueError("thread reset policy at_hour must be an integer from 0 to 23")
    if (
        isinstance(policy.at_minute, bool)
        or not isinstance(policy.at_minute, int)
        or not 0 <= policy.at_minute <= 59
    ):
        raise ValueError(
            "thread reset policy at_minute must be an integer from 0 to 59"
        )
    return {
        "mode": "daily",
        "at_hour": policy.at_hour,
        "at_minute": policy.at_minute,
    }


def _policy_from_state(value: object) -> SessionResetPolicy:
    """Strictly validate and decode one persisted route policy."""
    if not isinstance(value, dict):
        raise ValueError("thread policies must be objects")
    mode = value.get("mode")
    if mode == "none":
        if set(value) != {"mode"}:
            raise ValueError("'none' thread policies may only contain 'mode'")
        return SessionResetPolicy(mode="none")
    if mode == "daily":
        if set(value) != {"mode", "at_hour", "at_minute"}:
            raise ValueError(
                "'daily' thread policies require mode, at_hour, and at_minute"
            )
        hour = value["at_hour"]
        minute = value["at_minute"]
        if isinstance(hour, bool) or not isinstance(hour, int) or not 0 <= hour <= 23:
            raise ValueError("daily at_hour must be an integer from 0 to 23")
        if (
            isinstance(minute, bool)
            or not isinstance(minute, int)
            or not 0 <= minute <= 59
        ):
            raise ValueError("daily at_minute must be an integer from 0 to 59")
        return SessionResetPolicy(mode="daily", at_hour=hour, at_minute=minute)
    raise ValueError("thread policy mode must be 'daily' or 'none'")


class ThreadResetPolicyStore:
    """Atomic JSON persistence for explicit thread reset policies.

    Route keys are constructed by :class:`gateway.session.SessionStore`; this
    helper deliberately knows nothing about users or live session IDs.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._policies: dict[str, SessionResetPolicy] = {}
        self._malformed = False
        self._load()

    @property
    def malformed(self) -> bool:
        return self._malformed

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if (
                not isinstance(data, dict)
                or set(data) != {"version", "threads"}
                or isinstance(data.get("version"), bool)
                or data.get("version") != _STATE_VERSION
            ):
                raise ValueError("expected a version 1 object")
            threads = data["threads"]
            if not isinstance(threads, dict):
                raise ValueError("'threads' must be an object")
            loaded: dict[str, SessionResetPolicy] = {}
            for route_key, value in threads.items():
                if not isinstance(route_key, str) or not route_key.strip():
                    raise ValueError("thread route keys must be non-empty strings")
                loaded[route_key] = _policy_from_state(value)
            self._policies = loaded
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            # A corrupt explicit "off" must never degrade into an inherited
            # enabled policy. Keep the store marked invalid so resolution
            # fails closed and commands cannot overwrite evidence/state.
            self._malformed = True
            self._policies = {}
            logger.warning(
                "Malformed thread auto-reset state at %s; "
                "automatic resets for supported threads are disabled: %s",
                self.path,
                exc,
            )

    def resolve(
        self,
        route_key: str,
        inherited: SessionResetPolicy,
    ) -> tuple[SessionResetPolicy, ThreadResetResolution]:
        """Resolve a route override ahead of an inherited reset policy."""
        with self._lock:
            if self._malformed:
                return dataclasses.replace(inherited, mode="none"), "invalid"
            override = self._policies.get(route_key)

        if override is not None:
            return (
                dataclasses.replace(
                    inherited,
                    mode=override.mode,
                    at_hour=override.at_hour,
                    at_minute=override.at_minute,
                ),
                "override",
            )
        return inherited, "inherited"

    def set(self, route_key: str, policy: SessionResetPolicy) -> None:
        # Validate before taking the lock or touching last-known-good state.
        if not isinstance(route_key, str) or not route_key.strip():
            raise ValueError("thread route key must be a non-empty string")
        _policy_to_state(policy)
        with self._lock:
            self._ensure_writable()
            updated = dict(self._policies)
            updated[route_key] = dataclasses.replace(policy)
            self._write(updated)
            self._policies = updated

    def delete(self, route_key: str) -> None:
        if not isinstance(route_key, str) or not route_key.strip():
            raise ValueError("thread route key must be a non-empty string")
        with self._lock:
            self._ensure_writable()
            if route_key not in self._policies:
                return
            updated = dict(self._policies)
            del updated[route_key]
            self._write(updated)
            self._policies = updated

    def _ensure_writable(self) -> None:
        if self._malformed:
            raise ThreadResetPolicyStateError(
                "thread auto-reset state is malformed; no changes were written"
            )

    def _write(self, policies: dict[str, SessionResetPolicy]) -> None:
        atomic_json_write(
            self.path,
            {
                "version": _STATE_VERSION,
                "threads": {
                    route_key: _policy_to_state(policy)
                    for route_key, policy in policies.items()
                },
            },
        )
