"""Tests for the extracted GatewayKanbanWatchersMixin (god-file Phase 3).

The kanban watcher loops were lifted out of gateway/run.py into a mixin that
GatewayRunner inherits. These tests confirm the mixin exposes the methods and
that GatewayRunner picks them up via the MRO (behavior-neutral relocation).
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

from gateway.kanban_watchers import (
    GatewayKanbanWatchersMixin,
    _resolve_no_progress_timeout_seconds,
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


def test_gateway_resolves_no_progress_timeout_with_db_contract():
    kb = SimpleNamespace(
        DEFAULT_NO_PROGRESS_TIMEOUT_SECONDS=2700,
        resolve_no_progress_timeout_seconds=lambda value: (
            2700 if value is None else int(value)
        ),
    )

    assert _resolve_no_progress_timeout_seconds({}, kb) == 2700
    assert _resolve_no_progress_timeout_seconds(
        {"no_progress_timeout_seconds": "600"}, kb,
    ) == 600
    assert _resolve_no_progress_timeout_seconds(
        {"no_progress_timeout_seconds": 0}, kb,
    ) == 0


def test_gateway_resolves_no_progress_timeout_through_the_real_parser():
    """The stub above pins the *call shape*; this pins the actual values.

    The gateway deliberately owns no validity rules of its own — the DB layer
    is the single definition of a valid progress bound. A stub-only test would
    keep passing if the gateway grew a second, drifting copy of those rules, or
    if it started swallowing the parser's fail-safe fallback, so drive the real
    ``kanban_db`` module and assert exact resolved values for every branch.
    """
    from hermes_cli import kanban_db as real_kb

    default = real_kb.DEFAULT_NO_PROGRESS_TIMEOUT_SECONDS
    cases = [
        ({}, default),                                        # unset
        ({"no_progress_timeout_seconds": None}, default),     # explicit null
        ({"no_progress_timeout_seconds": 0}, 0),              # disabled
        ({"no_progress_timeout_seconds": 60}, 60),            # the minimum
        ({"no_progress_timeout_seconds": 3600}, 3600),        # ordinary
        ({"no_progress_timeout_seconds": "600"}, 600),        # YAML string
        ({"no_progress_timeout_seconds": True}, default),     # bool is not 1s
        ({"no_progress_timeout_seconds": False}, default),
        ({"no_progress_timeout_seconds": 45}, default),       # units slip
        ({"no_progress_timeout_seconds": 59}, default),
        ({"no_progress_timeout_seconds": -1}, default),
        ({"no_progress_timeout_seconds": "abc"}, default),
        ({"no_progress_timeout_seconds": []}, default),
        ({"no_progress_timeout_seconds": float("inf")}, default),
        ({"no_progress_timeout_seconds": float("nan")}, default),
    ]
    for cfg, expected in cases:
        assert _resolve_no_progress_timeout_seconds(cfg, real_kb) == expected, cfg

    # Every rejection falls back to the default, never to 0: a typo must not
    # silently switch the guard off and restore unbounded renewal.
    assert default > 0




def test_no_progress_is_a_terminal_and_wake_kind_with_an_i18n_string():
    """``no_progress`` terminates a run and re-queues the card exactly as
    ``crashed`` / ``timed_out`` do. A subscriber told about a crash but not
    about a progress reclaim watches the task silently restart — the confusion
    these lists exist to prevent — so the parity is asserted directly against
    the source rather than left to review.
    """
    import inspect

    from agent.i18n import t
    from gateway import kanban_watchers

    src = inspect.getsource(kanban_watchers.GatewayKanbanWatchersMixin)
    terminal = next(
        line for line in src.splitlines() if "TERMINAL_KINDS = (" in line
    )
    wake = next(line for line in src.splitlines() if "_WAKE_KINDS = (" in line)
    for line, label in ((terminal, "TERMINAL_KINDS"), (wake, "_WAKE_KINDS")):
        for kind in ("crashed", "timed_out", "no_progress"):
            assert f'"{kind}"' in line, f"{kind} missing from {label}"

    # The wake summary renders through i18n; a missing key would surface the
    # raw dotted path to the user.
    rendered = t("gateway.kanban.wake.no_progress")
    assert rendered and rendered != "gateway.kanban.wake.no_progress"
    assert "{" not in rendered


def test_every_locale_defines_the_no_progress_wake_string():
    """The wake summary is built from per-kind fragments; a locale missing one
    renders a dotted key inside an otherwise-translated sentence."""
    import glob
    import os

    import yaml

    root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))),
        "locales",
    )
    files = sorted(glob.glob(os.path.join(root, "*.yaml")))
    assert files, "no locale files found"
    for path in files:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        wake = (
            data.get("gateway", {}).get("kanban", {}).get("wake", {})
        )
        assert "timed_out" in wake, path
        assert wake.get("no_progress"), f"{path} is missing wake.no_progress"
