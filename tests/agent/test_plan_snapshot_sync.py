"""Reply headers that contradict the cached plan refresh it once, in the background."""

from types import SimpleNamespace as NS

import pytest

from agent import plan_snapshot_sync as sync
from agent.usage_gauges import BP_FULL, SCOPE_ALL, WindowGauge


def _state(window, full=()):
    return NS(is_usage_window=window, gauges=[NS(id=g, full=True) for g in full] + [NS(id="week", full=False)])


def _snap(gauges=None, source="network"):
    # The portal snapshot carries parsed gauges (agent.usage_gauges.WindowGauge), as nous_account stores them.
    window = None if gauges is None else NS(gauges=tuple(WindowGauge(g, SCOPE_ALL, g, BP_FULL if f else 0)
                                                         for g, f in gauges))
    return NS(usage_window=window, source=source)


ON_PLAN = _snap([("day", False), ("week", False)])


@pytest.mark.parametrize("state, snapshot, stale", [
    (_state(True), ON_PLAN, False),                  # in step
    (_state(True), _snap(None), True),               # upgraded in the browser
    (_state(False), ON_PLAN, True),                  # downgraded / cancelled
    (_state(True, ["day"]), ON_PLAN, True),          # a limit filled
    (_state(True), _snap([("day", True)]), True),    # a limit reset
    (_state(False), _snap(None), False),             # never a window plan: nothing to do
    (_state(True), None, False),                     # no snapshot yet: the session seed fetches it
    (_state(True), _snap(None, "error"), False),     # failed fetch: not evidence
    (None, ON_PLAN, False),
])
def test_staleness(state, snapshot, stale):
    assert sync.plan_snapshot_is_stale(state, snapshot) is stale


@pytest.fixture
def wired(monkeypatch):
    import hermes_cli.models_pricing as mp
    import hermes_cli.nous_account as na
    calls = []
    box = {"snap": _snap(None)}
    monkeypatch.setattr(na, "cached_nous_portal_account_info", lambda any_age=False: box["snap"])
    monkeypatch.setattr(na, "get_nous_portal_account_info", lambda force_fresh=False: calls.append("account"))
    monkeypatch.setattr(mp, "get_pricing_for_provider",
                        lambda provider, force_refresh=False: calls.append(("catalog", provider, force_refresh)))
    monkeypatch.setattr(sync, "_last_attempt", {})
    monkeypatch.setattr(sync, "_inflight", set())

    class _Thread:  # run inline: deterministic, no sleeps
        def __init__(self, target, args=(), **kw):
            self.target, self.args = target, args

        def start(self):
            self.target(*self.args)

    monkeypatch.setattr(sync.threading, "Thread", _Thread)
    return calls, box


def test_a_plan_change_refreshes_account_and_catalog_once(wired):
    calls, _box = wired
    assert sync.sync_plan_snapshot(_state(True)) is True
    assert calls == ["account", ("catalog", "nous", True)]
    # The portal can trail the gateway: the same mismatch on the next reply is not refetched.
    assert sync.sync_plan_snapshot(_state(True)) is False
    assert len(calls) == 2


def test_a_limit_change_refreshes_only_the_account(wired):
    calls, box = wired
    box["snap"] = ON_PLAN
    assert sync.sync_plan_snapshot(_state(True, ["day"])) is True
    assert calls == ["account"]


def test_a_mismatch_the_refresh_did_not_fix_retries(wired, monkeypatch):
    """A refresh that read a lagging portal (same stale gauges back) or a failed fetch reported as
    ``source="error"`` must not end re-syncing for that mismatch: an error rearms at once, a lag after the
    retry interval."""
    import hermes_cli.nous_account as na
    calls, box = wired
    box["snap"] = ON_PLAN
    monkeypatch.setattr(na, "get_nous_portal_account_info",
                        lambda force_fresh=False: calls.append("account") or _snap(None, "error"))
    assert sync.sync_plan_snapshot(_state(True, ["day"])) is True
    assert sync.sync_plan_snapshot(_state(True, ["day"])) is True  # the fetch failed: try again
    clock = {"now": 1_000.0}
    monkeypatch.setattr(sync.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(na, "get_nous_portal_account_info", lambda force_fresh=False: calls.append("account") or ON_PLAN)
    assert sync.sync_plan_snapshot(_state(True, ["day"])) is True
    assert sync.sync_plan_snapshot(_state(True, ["day"])) is False  # the portal lags: not on every reply
    clock["now"] += sync._RETRY_AFTER_S
    assert sync.sync_plan_snapshot(_state(True, ["day"])) is True
    assert calls == ["account"] * 4


def test_back_in_step_rearms(wired):
    _calls, box = wired
    assert sync.sync_plan_snapshot(_state(True)) is True
    box["snap"] = ON_PLAN
    assert sync.sync_plan_snapshot(_state(True)) is False  # in step: clears the guard
    box["snap"] = _snap(None)
    assert sync.sync_plan_snapshot(_state(True)) is True


def test_each_profile_refreshes_its_own_account_under_its_own_home(wired, tmp_path, monkeypatch):
    """One process, two profiles: the refresh runs with the asking profile's Hermes home, and one
    profile's pending mismatch never blocks another's."""
    from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override
    import hermes_cli.nous_account as na
    homes = []
    monkeypatch.setattr(na, "get_nous_portal_account_info", lambda force_fresh=False: homes.append(str(get_hermes_home())))
    for name in ("a", "b"):
        token = set_hermes_home_override(tmp_path / name)
        try:
            assert sync.sync_plan_snapshot(_state(True)) is True
        finally:
            reset_hermes_home_override(token)
    assert homes == [str(tmp_path / "a"), str(tmp_path / "b")]


def test_an_ordinary_reply_costs_nothing(wired):
    calls, box = wired
    box["snap"] = ON_PLAN
    for _ in range(5):
        assert sync.sync_plan_snapshot(_state(True)) is False
    assert calls == []


def test_never_raises(monkeypatch):
    import hermes_cli.nous_account as na
    monkeypatch.setattr(na, "cached_nous_portal_account_info", lambda any_age=False: 1 / 0)
    assert sync.sync_plan_snapshot(_state(True)) is False


def test_a_non_window_reply_without_a_window_snapshot_reads_nothing(monkeypatch):
    """Every other account: no snapshot read (no auth-state lookup) on any reply."""
    import hermes_cli.nous_account as na
    monkeypatch.setattr(na, "_account_info_cache", None)
    monkeypatch.setattr(na, "cached_nous_portal_account_info", lambda any_age=False: 1 / 0)
    assert sync.sync_plan_snapshot(_state(False)) is False
    monkeypatch.setattr(na, "_account_info_cache", ("k", 0.0, _snap(None)))
    assert sync.sync_plan_snapshot(_state(False)) is False
