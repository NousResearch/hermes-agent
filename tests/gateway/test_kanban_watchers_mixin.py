"""Tests for the extracted GatewayKanbanWatchersMixin (god-file Phase 3).

The kanban watcher loops were lifted out of gateway/run.py into a mixin that
GatewayRunner inherits. These tests confirm the mixin exposes the methods and
that GatewayRunner picks them up via the MRO (behavior-neutral relocation).
"""

from __future__ import annotations

import asyncio
import inspect

from gateway.config import GatewayConfig, Platform
from gateway.delivery import DeliveryRouter, DeliveryTarget
from gateway.kanban_watchers import (
    GatewayKanbanWatchersMixin,
    _project_finalizer_delivery_receipt,
    _normalize_project_finalizer_config,
)

KANBAN_METHODS = [
    "_kanban_notifier_watcher",
    "_kanban_dispatcher_watcher",
    "_kanban_advance",
    "_kanban_unsub",
    "_kanban_rewind",
    "_deliver_kanban_artifacts",
]


def test_mixin_defines_kanban_methods():
    for m in KANBAN_METHODS:
        assert hasattr(GatewayKanbanWatchersMixin, m), f"mixin missing {m}"


def test_gateway_runner_inherits_mixin(tmp_path, monkeypatch):
    # Import here so a heavy gateway import only happens if the first test passed.
    # The repository runner deliberately clears the normal Windows home vars.
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    from gateway.run import GatewayRunner

    assert issubclass(GatewayRunner, GatewayKanbanWatchersMixin)
    # Each kanban method resolves to the mixin's implementation via the MRO.
    for m in KANBAN_METHODS:
        owner = next(c for c in GatewayRunner.__mro__ if m in c.__dict__)
        assert owner is GatewayKanbanWatchersMixin, (
            f"{m} resolved to {owner.__name__}, expected the mixin"
        )


def test_watcher_loops_are_coroutines():
    # The two long-running watchers are async loops.
    assert inspect.iscoroutinefunction(GatewayKanbanWatchersMixin._kanban_notifier_watcher)
    assert inspect.iscoroutinefunction(GatewayKanbanWatchersMixin._kanban_dispatcher_watcher)


def test_project_finalizer_normalizes_delivery_router_send_result(tmp_path, monkeypatch):
    # The repository's clean-env runner removes Windows' home variables.  Set
    # the provider cache root before importing the real production result type.
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    from gateway.platforms.base import SendResult

    class TelegramAdapter:
        async def send(self, chat_id, content, metadata=None):
            return SendResult(success=True, message_id="telegram-message-42")

    target = DeliveryTarget(
        platform=Platform.TELEGRAM,
        chat_id="canary-chat",
        is_explicit=True,
    )
    router = DeliveryRouter(
        GatewayConfig(),
        adapters={Platform.TELEGRAM: TelegramAdapter()},
    )

    results = asyncio.run(
        router.deliver(
            "Result: COMPLETE",
            [target],
            metadata={"source": "project_finalizer"},
        )
    )
    receipt = results[target.to_string()]

    assert isinstance(receipt["result"], SendResult)
    assert _project_finalizer_delivery_receipt(receipt) == {
        "provider_message_id": "telegram-message-42"
    }


def test_project_finalizer_normalizes_mapping_fallback_and_rejection():
    assert _project_finalizer_delivery_receipt(
        {
            "success": True,
            "result": {"message_id": "", "id": "legacy-42"},
        }
    ) == {"provider_message_id": "legacy-42"}
    assert _project_finalizer_delivery_receipt(
        {
            "success": True,
            "result": {"message_id": "   ", "id": "legacy-43"},
        }
    ) == {"provider_message_id": "legacy-43"}
    assert _project_finalizer_delivery_receipt(
        {
            "success": True,
            "result": {"message_id": "\t", "id": "  "},
        }
    ) == {}
    assert _project_finalizer_delivery_receipt(
        {"success": True, "result": {"message_id": 0, "id": False}}
    ) == {}
    assert _project_finalizer_delivery_receipt(
        {"success": False, "error": "provider rejected send"}
    ) == {"rejected": True, "error": "provider rejected send"}


def test_project_finalizer_config_accepts_only_exact_preview_canary_scopes():
    assert _normalize_project_finalizer_config(
        {
            "project_finalizer": {
                "enabled": True,
                "canary_scope": ["release-board/t_deadbeef"],
                "interval_seconds": 15,
                "cleanup": {"mode": "preview"},
            }
        }
    ) == (True, ("release-board/t_deadbeef",), 15.0)


def test_project_finalizer_config_fails_closed_for_unsafe_or_malformed_values():
    valid = {
        "enabled": True,
        "canary_scope": ["release-board/t_deadbeef"],
        "interval_seconds": 15,
        "cleanup": {"mode": "preview"},
    }
    invalid_overrides = (
        {"canary_scope": ["release-board/*"]},
        {"canary_scope": ["t_deadbeef"]},
        {"canary_scope": ["release-board/t_deadbeef", "release-board/t_deadbeef"]},
        {"interval_seconds": 0},
        {"interval_seconds": float("inf")},
        {"cleanup": {"mode": "apply"}},
        {"cleanup_enabled": True},
        {"cleanup": []},
        {"canary_scope": "release-board/t_deadbeef"},
    )
    for overrides in invalid_overrides:
        candidate = valid | overrides
        assert _normalize_project_finalizer_config({"project_finalizer": candidate}) == (False, (), 60.0)


def test_project_finalizer_watcher_uses_normalized_scope_and_never_enables_cleanup(monkeypatch):
    """The gateway passes only the safe, normalized finalizer contract."""
    from gateway import kanban_watchers
    from gateway import project_finalization
    from hermes_cli import config as hermes_config
    from hermes_cli import kanban_db

    captured = []

    class Finalizer:
        def __init__(self, *_args, **kwargs):
            captured.append(kwargs)

        async def tick(self, *, board_id):
            runner._running = False

    class Runner(GatewayKanbanWatchersMixin):
        _running = True
        delivery_router = None

    runner = Runner()
    monkeypatch.setattr(
        hermes_config,
        "load_config",
        lambda: {
            "project_finalizer": {
                "enabled": True,
                "canary_scope": ["release-board/t_deadbeef"],
                "interval_seconds": 7,
                "cleanup": {"mode": "preview"},
            }
        },
    )
    monkeypatch.setattr(kanban_db, "list_boards", lambda **_kwargs: [{"slug": "release-board"}])
    monkeypatch.setattr(kanban_db, "connect", lambda **_kwargs: object())
    monkeypatch.setattr(project_finalization, "ProjectFinalizationService", Finalizer)

    async def no_sleep(_interval):
        return None

    monkeypatch.setattr(kanban_watchers.asyncio, "sleep", no_sleep)
    asyncio.run(runner._kanban_project_finalizer_watcher())

    assert len(captured) == 1
    assert captured[0]["owner"] == f"gateway:{id(runner)}"
    assert callable(captured[0]["now"])
    assert callable(captured[0]["deliver"])
    assert captured[0]["enabled"] is True
    assert captured[0]["canary_scope"] == ("release-board/t_deadbeef",)
    assert captured[0]["cleanup_enabled"] is False


def test_singleton_dispatcher_lock_is_exclusive(tmp_path):
    """Only one holder of the dispatcher lock at a time — the backstop that
    stops concurrent dispatchers double reclaiming and corrupting shared
    kanban SQLite index pages under wal_autocheckpoint=0."""
    import os

    from gateway.kanban_watchers import _acquire_singleton_lock, _release_singleton_lock

    lock = tmp_path / "kanban" / ".dispatcher.lock"

    h1, st1 = _acquire_singleton_lock(lock)
    assert st1 == "held" and h1 is not None

    # A second acquire while the first is held must be refused, not granted.
    h2, st2 = _acquire_singleton_lock(lock)
    assert st2 == "contended" and h2 is None

    # Releasing the first lets a fresh acquire succeed (lock is reusable).
    _release_singleton_lock(h1)
    h3, st3 = _acquire_singleton_lock(lock)
    assert st3 == "held" and h3 is not None
    _release_singleton_lock(h3)
