"""Cross-profile authorization bypass repro for GHSA-p47r-wg2f-2mw4.

``_validate_matrix_prompt_reactor`` and ``_on_invite`` read
``GATEWAY_ALLOW_ALL_USERS`` via raw ``os.getenv`` instead of the profile-scoped
secret accessor. Under ``gateway.multiplex_profiles`` the default profile may
set ``GATEWAY_ALLOW_ALL_USERS=true`` while a secondary profile explicitly
disables it and restricts ``MATRIX_ALLOWED_USERS``. The raw read leaks the
default profile's open gate into the secondary profile, so an unauthorized
Matrix user can react to approval / model-picker / choice prompts and auto-join
rooms via invites.

This test mirrors the style of ``tests/gateway/test_platform_authz_scope.py``
and can be run with pytest or directly as a script (it exits non-zero on bypass).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import types
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# Stub mautrix so plugins.platforms.matrix.adapter imports without the SDK.
# This is a minimal static stub aligned with the stubs used in
# tests/gateway/test_matrix.py and test_matrix_approval_reaction_fail_closed.py.
# If the real mautrix SDK adds imports required by the adapter at import time,
# this stub will need to be extended.
# ---------------------------------------------------------------------------

def _stub_mautrix():
    stub = types.ModuleType("mautrix")
    for sub in (
        "mautrix.types",
        "mautrix.client",
        "mautrix.client.api",
        "mautrix.errors",
        "mautrix.crypto",
        "mautrix.util",
        "mautrix.util.config",
        "mautrix.api",
    ):
        sys.modules.setdefault(sub, types.ModuleType(sub))
    sys.modules.setdefault("mautrix", stub)

    m = sys.modules["mautrix.types"]

    class EventType:
        ROOM_MESSAGE = "m.room.message"
        REACTION = "m.reaction"
        ROOM_ENCRYPTED = "m.room.encrypted"
        ROOM_NAME = "m.room.name"

    class PaginationDirection:
        BACKWARD = "b"
        FORWARD = "f"

    class PresenceState:
        ONLINE = "online"
        OFFLINE = "offline"
        UNAVAILABLE = "unavailable"

    class RoomCreatePreset:
        PRIVATE = "private_chat"
        PUBLIC = "public_chat"
        TRUSTED_PRIVATE = "trusted_private_chat"

    class TrustState:
        UNVERIFIED = 0
        VERIFIED = 1

    for attr in ("ContentURI", "EventID", "RoomID", "SyncToken", "UserID"):
        setattr(m, attr, str)
    m.EventType = EventType
    m.PaginationDirection = PaginationDirection
    m.PresenceState = PresenceState
    m.RoomCreatePreset = RoomCreatePreset
    m.TrustState = TrustState


_stub_mautrix()

import agent.secret_scope  # noqa: E402
from agent.secret_scope import (  # noqa: E402
    reset_secret_scope,
    set_multiplex_active,
    set_secret_scope,
)
from plugins.platforms.matrix.adapter import MatrixAdapter  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _profile_scope(secrets):
    """Install a secondary-profile secret scope; os.environ stays 'default'."""
    token = set_secret_scope(secrets)
    try:
        yield
    finally:
        reset_secret_scope(token)


@contextlib.contextmanager
def _multiplex(active):
    """Temporarily set the multiplex-active flag and restore it on exit."""
    original = agent.secret_scope._MULTIPLEX_ACTIVE
    set_multiplex_active(active)
    try:
        yield
    finally:
        set_multiplex_active(original)


def _make_secondary_adapter(allowed_user_ids):
    """Build a MatrixAdapter as if constructed under a secondary profile scope."""
    adapter = object.__new__(MatrixAdapter)
    adapter._allowed_user_ids = set(allowed_user_ids)
    adapter._approval_require_sender = False  # requester check is orthogonal here
    return adapter


async def _run_validation(adapter, sender):
    """Call _validate_matrix_prompt_reactor and return its boolean result."""
    prompt = SimpleNamespace(requester_user_id=None)

    # Stub send so an unauthorized result does not try to talk to a real client.
    async def _noop_send(*args, **kwargs):
        return SimpleNamespace(success=True)

    adapter.send = _noop_send

    return await adapter._validate_matrix_prompt_reactor(
        room_id="!testroom:matrix.org",
        target_event_id="$prompt-event-1",
        sender=sender,
        prompt=prompt,
        prompt_label="approval",
    )


def _make_invite_event(inviter, room_id="!testroom:matrix.org", is_direct=False):
    """Minimal Matrix invite event for _on_invite."""
    return SimpleNamespace(
        sender=inviter,
        room_id=room_id,
        content=SimpleNamespace(is_direct=is_direct),
    )


async def _run_invite(adapter, inviter):
    """Call _on_invite and return whether the join was scheduled."""
    calls = []

    def _record_join(room_id, *, is_direct=False, inviter=""):
        calls.append((room_id, is_direct, inviter))

    adapter._schedule_invite_join = _record_join
    await adapter._on_invite(_make_invite_event(inviter))
    return len(calls) > 0


# ---------------------------------------------------------------------------
# Repro tests
# ---------------------------------------------------------------------------

def test_matrix_prompt_reactor_ignores_scoped_gateway_allow_all(monkeypatch):
    """Default profile has GATEWAY_ALLOW_ALL_USERS=true.

    Secondary profile has it false and an allowlist that excludes the attacker.
    The validator must return False (deny). If it returns True, the default
    profile's open gate has leaked into the secondary profile.
    """
    # Default profile / process environment
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")

    authorized_user = "@authorized:matrix.org"
    attacker = "@attacker:matrix.org"

    with _multiplex(True), _profile_scope(
        {
            "GATEWAY_ALLOW_ALL_USERS": "",  # explicitly disabled
            "MATRIX_ALLOWED_USERS": authorized_user,
        }
    ):
        adapter = _make_secondary_adapter([authorized_user])
        result = asyncio.run(_run_validation(adapter, attacker))

    # A True result means the bypass reproduces.
    assert result is False, (
        f"Cross-profile authorization bypass reproduced: "
        f"_validate_matrix_prompt_reactor returned True for {attacker} "
        f"under a secondary profile that disabled allow-all and excluded them."
    )


def test_matrix_prompt_reactor_honors_scoped_allowlist_when_default_is_closed(
    monkeypatch,
):
    """Control: when the default profile does NOT open the gate, the secondary
    profile's allowlist is enforced and the attacker is denied. This proves the
    previous failure is specifically the os.environ leak, not a broken allowlist."""
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)

    authorized_user = "@authorized:matrix.org"
    attacker = "@attacker:matrix.org"

    with _multiplex(True), _profile_scope(
        {
            "GATEWAY_ALLOW_ALL_USERS": "",
            "MATRIX_ALLOWED_USERS": authorized_user,
        }
    ):
        adapter = _make_secondary_adapter([authorized_user])
        result = asyncio.run(_run_validation(adapter, attacker))

    assert result is False


def test_matrix_invite_auto_join_ignores_scoped_gateway_allow_all(monkeypatch):
    """Default profile has GATEWAY_ALLOW_ALL_USERS=true.

    Secondary profile has it false and an allowlist that excludes the attacker.
    _on_invite must NOT schedule a join for the attacker's invite. If it does,
    the default profile's open gate has leaked into the secondary profile.
    """
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")

    authorized_user = "@authorized:matrix.org"
    attacker = "@attacker:matrix.org"

    with _multiplex(True), _profile_scope(
        {
            "GATEWAY_ALLOW_ALL_USERS": "",
            "MATRIX_ALLOWED_USERS": authorized_user,
        }
    ):
        adapter = _make_secondary_adapter([authorized_user])
        joined = asyncio.run(_run_invite(adapter, attacker))

    assert joined is False, (
        f"Cross-profile authorization bypass reproduced: "
        f"_on_invite scheduled a join for {attacker} under a secondary profile "
        f"that disabled allow-all and excluded them."
    )


def test_matrix_invite_auto_join_honors_scoped_allowlist_when_default_is_closed(
    monkeypatch,
):
    """Control: when the default profile does NOT open the gate, the secondary
    profile's allowlist is enforced and the attacker's invite is rejected."""
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)

    authorized_user = "@authorized:matrix.org"
    attacker = "@attacker:matrix.org"

    with _multiplex(True), _profile_scope(
        {
            "GATEWAY_ALLOW_ALL_USERS": "",
            "MATRIX_ALLOWED_USERS": authorized_user,
        }
    ):
        adapter = _make_secondary_adapter([authorized_user])
        joined = asyncio.run(_run_invite(adapter, attacker))

    assert joined is False


if __name__ == "__main__":
    # Minimal pytest-free runner: returns 0 on safe, 1 on bypass.
    class _Monkeypatch:
        def __init__(self):
            self._backup: dict[str, str | None] = {}

        def setenv(self, name, value):
            self._backup.setdefault(name, os.environ.get(name))
            os.environ[name] = value

        def delenv(self, name, raising=True):
            self._backup.setdefault(name, os.environ.get(name))
            os.environ.pop(name, None)

        def undo(self):
            for name, value in self._backup.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    mp = _Monkeypatch()
    failures = []
    try:
        for test in (
            test_matrix_prompt_reactor_ignores_scoped_gateway_allow_all,
            test_matrix_invite_auto_join_ignores_scoped_gateway_allow_all,
        ):
            try:
                test(mp)
            except AssertionError as exc:
                failures.append(str(exc))
        if failures:
            print("BYPASS REPRODUCED:")
            for msg in failures:
                print(f"  - {msg}")
            sys.exit(1)
        print("RESULT: no bypass detected (both paths denied).")
        sys.exit(0)
    finally:
        mp.undo()
