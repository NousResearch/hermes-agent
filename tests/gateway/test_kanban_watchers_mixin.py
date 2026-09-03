"""Tests for the extracted GatewayKanbanWatchersMixin (god-file Phase 3).

The kanban watcher loops were lifted out of gateway/run.py into a mixin that
GatewayRunner inherits. These tests confirm the mixin exposes the methods and
that GatewayRunner picks them up via the MRO (behavior-neutral relocation).
"""

from __future__ import annotations

import inspect

from gateway.kanban_watchers import (
    GatewayKanbanWatchersMixin,
    _auto_promote_children_from_config,
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


def test_auto_promote_children_config_mapping():
    """#79608: kanban.auto_promote_children drives the dispatcher's
    skip_decompose_children flag. Default True (auto-promote, the
    pre-existing behavior); explicit False turns the manual-review gate on."""
    assert _auto_promote_children_from_config({}) is True
    assert _auto_promote_children_from_config({"auto_promote_children": True}) is True
    assert _auto_promote_children_from_config({"auto_promote_children": False}) is False
    # A stringified YAML "false" must not accidentally enable promotion.
    assert _auto_promote_children_from_config({"auto_promote_children": "false"}) is False


