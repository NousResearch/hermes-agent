"""Regression tests: multiplex cron must deliver from the OWNING profile's bot.

Incident shape (Aug 2026, local install, two Telegram bots on one multiplexed
gateway): a scheduled job created in the `medicina` profile, with a correct
`deliver: origin` and a correct origin block, was delivered into the `default`
profile's Telegram chat — the user saw a medicine job's report arrive from the
BrawlBala bot.

Root cause is a scoping asymmetry in the multiplex cron ticker. The loop scopes
the STORE per profile (`set_hermes_home_override` + `use_cron_store`), but hands
every profile the SAME adapter registry — the primary's:

    for entry in profile_homes:
        with use_cron_store(home):
            cron_tick(adapters=adapters, ...)   # <- primary's registry

That is invisible on most platforms, but a cron delivery target carries only
`platform` + `chat_id` (cron/scheduler.py::_resolve_single_delivery_target) and
delivery resolves an adapter out of whatever registry it is handed
(gateway/delivery.py::resolve_delivery_transport). On Telegram a private chat
reports the user's OWN id as `chat.id`, identical across every bot serving that
human — so the wrong-registry lookup silently succeeds and the wrong bot sends.

Fix under test: `profile_adapters` maps profile name -> that profile's own
registry, and `_adapters_for(name)` selects it, falling back to the primary's
registry when a profile has none (wrong-bot is recoverable; losing the output is
not).
"""

import threading
from pathlib import Path

import pytest

from cron.scheduler_provider import InProcessCronScheduler


class _Adapter:
    """Stand-in for a platform adapter, tagged with its owning profile."""

    def __init__(self, profile):
        self.profile = profile

    def __repr__(self):
        return f"<adapter {self.profile}>"


@pytest.fixture()
def registries():
    """Primary + secondary registries, as GatewayRunner would build them."""
    return {
        "default": {"telegram": _Adapter("default")},
        "medicina": {"telegram": _Adapter("medicina")},
    }


def _run_one_cycle(monkeypatch, tmp_path, profile_homes, **start_kwargs):
    """Drive _start_multiplex through exactly one tick, capturing each call.

    Returns [(profile_home_name, adapters_passed), ...] in tick order.
    """
    seen = []

    def _fake_tick(*, verbose=False, adapters=None, loop=None, sync=False, can_dispatch=None):
        seen.append(adapters)

    import cron.scheduler as _sched
    import cron.jobs as _jobs

    monkeypatch.setattr(_sched, "tick", _fake_tick)
    # Neutralise per-profile store bookkeeping — this test is about routing.
    monkeypatch.setattr(_jobs, "record_ticker_heartbeat", lambda **k: None)
    monkeypatch.setattr(_jobs, "clear_ticker_error", lambda: None)
    monkeypatch.setattr(_jobs, "record_ticker_error", lambda *a, **k: None)

    class _NullCtx:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(_jobs, "use_cron_store", lambda home: _NullCtx())

    sched = InProcessCronScheduler()
    monkeypatch.setattr(sched, "recover_interrupted", lambda: 0)

    stop = threading.Event()

    # Stop the loop as soon as the first full cycle has been ticked.
    real_wait = stop.wait

    def _wait_then_stop(timeout=None):
        stop.set()
        return real_wait(0)

    monkeypatch.setattr(stop, "wait", _wait_then_stop)

    sched._start_multiplex(
        stop,
        profile_homes=profile_homes,
        interval=1,
        **start_kwargs,
    )
    return seen


class TestCronDeliveryRegistryPerProfile:
    def test_each_profile_ticks_with_its_own_registry(
        self, monkeypatch, tmp_path, registries
    ):
        """The core bug: medicina's job must not be handed default's adapters."""
        homes = [("default", tmp_path / "default"), ("medicina", tmp_path / "medicina")]
        seen = _run_one_cycle(
            monkeypatch, tmp_path, homes,
            adapters=registries["default"],
            profile_adapters=registries,
        )
        assert len(seen) == 2, seen
        assert seen[0]["telegram"].profile == "default"
        assert seen[1]["telegram"].profile == "medicina", (
            "medicina ticked with another profile's adapter registry — its cron "
            "output would be delivered by that profile's bot"
        )

    def test_secondary_without_registry_falls_back_to_primary(
        self, monkeypatch, tmp_path, registries
    ):
        """A profile whose adapters never connected still delivers.

        Wrong-bot is recoverable; silently dropping the job's output is not.
        """
        homes = [("default", tmp_path / "default"), ("ytmed", tmp_path / "ytmed")]
        seen = _run_one_cycle(
            monkeypatch, tmp_path, homes,
            adapters=registries["default"],
            profile_adapters={"default": registries["default"]},  # no ytmed entry
        )
        assert len(seen) == 2
        assert seen[1] is registries["default"]

    def test_no_profile_adapters_keeps_legacy_behaviour(
        self, monkeypatch, tmp_path, registries
    ):
        """An older gateway passes no mapping — every tick uses `adapters`."""
        homes = [("default", tmp_path / "default"), ("medicina", tmp_path / "medicina")]
        seen = _run_one_cycle(
            monkeypatch, tmp_path, homes,
            adapters=registries["default"],
        )
        assert len(seen) == 2
        assert all(s is registries["default"] for s in seen)

    def test_bare_path_entries_still_supported(
        self, monkeypatch, tmp_path, registries
    ):
        """profile_homes may be bare paths (no name) — must not crash."""
        homes = [tmp_path / "default", tmp_path / "medicina"]
        seen = _run_one_cycle(
            monkeypatch, tmp_path, homes,
            adapters=registries["default"],
            profile_adapters=registries,
        )
        assert len(seen) == 2
        assert all(s is registries["default"] for s in seen), (
            "a nameless entry cannot be attributed to a profile; it must fall "
            "back to the primary registry rather than guess"
        )

    def test_empty_registry_is_not_selected(self, monkeypatch, tmp_path, registries):
        """An empty dict means 'no adapters connected' — fall back, don't send
        into a registry that can resolve nothing."""
        homes = [("default", tmp_path / "default"), ("medicina", tmp_path / "medicina")]
        seen = _run_one_cycle(
            monkeypatch, tmp_path, homes,
            adapters=registries["default"],
            profile_adapters={"default": registries["default"], "medicina": {}},
        )
        assert seen[1] is registries["default"]
