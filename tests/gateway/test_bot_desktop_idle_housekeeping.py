"""The messaging gateway stops idle Bot Desktop screens for every served profile.

Before this chore, only the ``hermes serve`` lease watcher called ``stop_if_idle``: it runs once a
client connects and visits only profiles a client addressed, so a screen a cron or Slack turn
auto-started for an untouched profile never stopped.
"""

from contextlib import contextmanager
from types import SimpleNamespace

import gateway.run as gateway_run


class _OneTickStopEvent:
    """Run one housekeeping tick without a sleep or background thread."""

    def __init__(self):
        self.waited = False

    def is_set(self):
        return self.waited

    def wait(self, timeout=None):
        self.waited = True
        return True


def _record_stop_if_idle(monkeypatch):
    from hermes_constants import get_hermes_home
    from tools.bot_desktop import runtime

    calls = []
    monkeypatch.setattr(
        runtime, "stop_if_idle", lambda: calls.append(str(get_hermes_home())) or False
    )
    return calls


def test_single_profile_gateway_runs_the_idle_stop_each_tick(monkeypatch):
    calls = _record_stop_if_idle(monkeypatch)

    gateway_run._start_gateway_housekeeping(_OneTickStopEvent(), interval=0)

    assert len(calls) == 1


def test_multiplexed_gateway_runs_the_idle_stop_once_per_served_profile(
    monkeypatch, tmp_path
):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    homes = [
        ("default", tmp_path / "launch"),
        ("assistant", tmp_path / "profiles" / "assistant"),
    ]
    for _name, home in homes:
        home.mkdir(parents=True)
    calls = _record_stop_if_idle(monkeypatch)

    @contextmanager
    def scope(home):
        token = set_hermes_home_override(home)
        try:
            yield
        finally:
            reset_hermes_home_override(token)

    monkeypatch.setattr(gateway_run, "_multiplex_profile_homes", lambda config: homes)
    monkeypatch.setattr(gateway_run, "_profile_runtime_scope", scope)
    runner = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=True))

    gateway_run._start_gateway_housekeeping(
        _OneTickStopEvent(), interval=0, runner=runner
    )

    assert sorted(calls) == sorted(str(home) for _name, home in homes)
