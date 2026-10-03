"""Regression test: the obsolete-command delete tolerates Discord error 10063.

``_safe_sync_slash_commands`` reconciles the global command tree by deleting
commands Discord still has but the adapter no longer wants, then upserting the
desired set. The delete can 404 with error code 10063 ("Unknown Application
Command") when the command disappeared between ``fetch_commands()`` and the
delete — a parallel sync, or a second gateway instance reconciling the same
application. That 404 is convergence, not a failure, and must not abort the
remaining registrations.

The guard has to stay narrow: any *other* ``discord.errors.NotFound`` code (e.g.
10013 "Unknown Application", which means the whole app is wrong) must still
propagate rather than being silently swallowed.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock
import sys

import pytest

from gateway.config import PlatformConfig


def _ensure_discord_mock():
    """Install (or augment) a mock ``discord`` module.

    Mirrors the sibling gateway Discord test files: conftest.py already stubs
    ``sys.modules["discord"]`` before collection, so this is normally a no-op
    and only matters when the file is run outside the tests/gateway package.
    """
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return

    if sys.modules.get("discord") is None:
        from unittest.mock import MagicMock

        discord_mod = MagicMock()
        discord_mod.Intents.default.return_value = MagicMock()
        sys.modules["discord"] = discord_mod
        sys.modules["discord.ext"] = MagicMock()
        sys.modules["discord.ext.commands"] = MagicMock()


_ensure_discord_mock()

import plugins.platforms.discord.adapter as discord_platform  # noqa: E402
from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


class _FakeNotFound(Exception):
    """Stand-in for ``discord.errors.NotFound`` used when discord.py is absent.

    Mirrors ``discord.errors.HTTPException.__init__``: ``status`` is read off
    the response object, ``code``/``text`` out of the JSON message dict. The
    adapter's guard keys on ``getattr(exc, "code", None)``, so reproducing that
    attribute faithfully is what the test needs.
    """

    def __init__(self, response, message):
        self.response = response
        self.status = response.status
        if isinstance(message, dict):
            self.code = message.get("code", 0)
            self.text = message.get("message", "")
        else:
            self.code = 0
            self.text = message or ""
        super().__init__(
            f"{self.status} {getattr(response, 'reason', '')}"
            f" (error code: {self.code}): {self.text}"
        )


def _load_real_not_found():
    """Return the real ``discord.errors.NotFound`` class, or None.

    tests/gateway/conftest.py unconditionally replaces ``sys.modules["discord"]``
    with a MagicMock (it short-circuits only when the real library was *already*
    imported, which is never the case in a fresh per-file subprocess). The
    adapter resolves ``except discord.errors.NotFound`` against that mock, so
    the real class has to be pulled in explicitly. Purge the mock entries, import
    the genuine library, then restore ``sys.modules`` byte-for-byte so the rest
    of the suite still sees the conftest mock.

    Returns None when discord.py is not installed — the ``messaging`` extra is
    absent from the main CI test lane, so this is a normal, expected outcome
    there and the caller falls back to :class:`_FakeNotFound`.
    """
    saved = {k: v for k, v in sys.modules.items() if k == "discord" or k.startswith("discord.")}
    for key in saved:
        del sys.modules[key]
    try:
        import discord as real_discord

        not_found = real_discord.errors.NotFound
        if isinstance(not_found, type) and issubclass(not_found, BaseException):
            return not_found
        return None
    except Exception:
        return None
    finally:
        for key in [k for k in list(sys.modules) if k == "discord" or k.startswith("discord.")]:
            del sys.modules[key]
        sys.modules.update(saved)


@pytest.fixture(autouse=True)
def _speed_up_command_sync_mutation_pacing(monkeypatch):
    monkeypatch.setattr(
        DiscordAdapter,
        "_command_sync_mutation_interval_seconds",
        lambda self: 0.0,
    )


@pytest.fixture
def not_found_class(monkeypatch):
    """Bind the exception class the adapter's ``except`` clause will resolve.

    Prefers the genuine ``discord.errors.NotFound``; falls back to
    :class:`_FakeNotFound` when discord.py is not importable. Either way the
    adapter is handed a real (subclassable) exception class, which is what makes
    the ``except`` clause legal Python.
    """
    assert discord_platform.discord is not None, (
        "adapter imported with DISCORD_AVAILABLE=False; this test needs a "
        "discord module object to patch the exception class onto"
    )
    exc_cls = _load_real_not_found() or _FakeNotFound
    monkeypatch.setattr(discord_platform.discord.errors, "NotFound", exc_cls)
    return exc_cls


def _not_found(exc_cls, code: int):
    """Build a NotFound carrying ``code``, using the real library's signature."""
    return exc_cls(
        SimpleNamespace(status=404, reason="Not Found"),
        {"code": code, "message": f"Unknown Application (error code: {code})"},
    )


def _desired_payload(name: str, description: str) -> dict:
    return {
        "name": name,
        "description": description,
        "type": 1,
        "options": [],
        "nsfw": False,
        "dm_permission": True,
        "default_member_permissions": None,
    }


class _DesiredCommand:
    def __init__(self, payload):
        self._payload = payload

    def to_dict(self, tree):
        assert tree is not None
        return dict(self._payload)


class _ExistingCommand:
    def __init__(self, command_id, payload):
        self.id = command_id
        self.name = payload["name"]
        self.type = SimpleNamespace(value=payload["type"])
        self._payload = payload

    def to_dict(self):
        return {
            "id": self.id,
            "application_id": 999,
            **self._payload,
            "name_localizations": {},
            "description_localizations": {},
        }


def _build_adapter(*, existing, desired):
    """Adapter wired to a fake tree/http over the given existing/desired sets."""
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
    fake_tree = SimpleNamespace(
        get_commands=lambda: [_DesiredCommand(p) for p in desired],
        fetch_commands=AsyncMock(return_value=list(existing)),
    )
    fake_http = SimpleNamespace(
        upsert_global_command=AsyncMock(),
        edit_global_command=AsyncMock(),
        delete_global_command=AsyncMock(),
    )
    adapter._client = SimpleNamespace(
        tree=fake_tree,
        http=fake_http,
        application_id=999,
        user=SimpleNamespace(id=999),
    )
    return adapter, fake_http


@pytest.mark.asyncio
async def test_obsolete_delete_10063_is_swallowed_and_sync_continues(not_found_class):
    """A 10063 on the obsolete delete is convergence, not an error.

    Discord holds ``alpha`` and ``bravo``; the adapter only wants ``charlie``.
    ``alpha`` deletes cleanly, ``bravo`` 404s with 10063 (already removed by a
    concurrent sync). The 404 must be logged and stepped over so ``charlie``
    still gets registered — before the fix the whole reconciliation aborted with
    the 404 and every desired command stayed unregistered.
    """
    adapter, fake_http = _build_adapter(
        existing=[
            _ExistingCommand(11, _desired_payload("alpha", "Old command A")),
            _ExistingCommand(12, _desired_payload("bravo", "Old command B")),
        ],
        desired=[_desired_payload("charlie", "New command C")],
    )

    # The real adapter payload helpers run here (unlike the older sync-limit
    # test), so the diff logic itself is exercised rather than stubbed out.
    gone_id = 12

    async def delete_side_effect(_app_id, command_id):
        if command_id == gone_id:
            raise _not_found(not_found_class, 10063)

    adapter._client.http.delete_global_command = AsyncMock(side_effect=delete_side_effect)

    summary = await adapter._safe_sync_slash_commands()

    # Both obsolete commands were attempted; the 10063 one did not stop the loop.
    deleted_ids = [call.args[1] for call in fake_http.delete_global_command.await_args_list]
    assert sorted(deleted_ids) == [11, 12], (
        "the 404 on command 12 must be swallowed, not re-raised mid-loop"
    )

    # The desired command was still registered after the swallowed 404.
    assert summary == {
        "total": 1,
        "unchanged": 0,
        "updated": 0,
        "recreated": 0,
        "created": 1,
        # Only command 11 was actually deleted by us; 12 was already gone, so it
        # is not counted. `total` is assigned on the final line of the method,
        # so reaching the end is itself proof the sync ran to completion.
        "deleted": 1,
    }
    fake_http.upsert_global_command.assert_awaited_once_with(
        999, _desired_payload("charlie", "New command C")
    )


@pytest.mark.asyncio
async def test_obsolete_delete_other_not_found_code_still_propagates(not_found_class):
    """A NotFound carrying any code other than 10063 must still escape.

    This is the half that keeps the tolerance narrow: 10013 ("Unknown
    Application") means the application itself is wrong, not that the command
    converged, and swallowing it would hide a real misconfiguration. The delete
    also happens before any create, so propagating must leave the desired
    command unregistered rather than letting the sync limp on.
    """
    adapter, fake_http = _build_adapter(
        existing=[_ExistingCommand(13, _desired_payload("old-command", "To be deleted"))],
        desired=[_desired_payload("metricas", "Show Colmeio metrics dashboard")],
    )
    adapter._client.http.delete_global_command = AsyncMock(
        side_effect=_not_found(not_found_class, 10013)
    )

    with pytest.raises(not_found_class) as excinfo:
        await adapter._safe_sync_slash_commands()

    assert excinfo.value.code == 10013
    assert fake_http.upsert_global_command.await_count == 0, (
        "a non-10063 NotFound must abort the reconciliation, not continue to the upserts"
    )
    fake_http.edit_global_command.assert_not_awaited()
