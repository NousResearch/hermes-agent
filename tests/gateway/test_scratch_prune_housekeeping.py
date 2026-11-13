"""Scratch pruning rides the gateway housekeeping tick, because the per-process gate in
``get_scratch_dir`` fires once per process.

A gateway that stays up for days spends that single chance on whichever call comes first. When
that call lands inside the hour the previous pass stamped -- a restart minutes after a prune, or
any boot short of the hour -- it prunes nothing and the process never asks again, so a long-lived
deployment accumulates every artifact it wrote since. One such deployment reached 5 days and
18 MB that way. The chore re-asks the stamped hour on every tick; the hour, not the process age,
is the floor.
"""

import gateway.run as gateway_run


class _OneTickStopEvent:
    def __init__(self):
        self.waited = False

    def is_set(self):
        return self.waited

    def wait(self, timeout=None):
        self.waited = True
        return True


def test_gateway_housekeeping_reasks_the_scratch_hour(monkeypatch):
    import hermes_constants as constants

    calls = []
    monkeypatch.setattr(constants, "prune_scratch_dir_if_due", lambda: calls.append(True))

    gateway_run._start_gateway_housekeeping(_OneTickStopEvent(), interval=0)

    assert calls == [True]


def test_multiplexed_scratch_chore_reaches_every_served_profile(tmp_path, monkeypatch):
    """On a multiplexed gateway the chore must re-ask the hour for each profile's OWN scratch.

    ``prune_scratch_dir_if_due()`` resolves its root through ``get_hermes_home()``, and the
    housekeeping thread carries no turn, so nothing binds a profile for it: a bare registration
    pruned the launch profile's scratch on every tick while each named profile's cache grew with
    no hourly pass at all. The boot-time gate cannot cover a gateway that stays up for days.
    """
    from pathlib import Path
    from types import SimpleNamespace

    import hermes_constants as constants

    fake_home = tmp_path / "home"
    launch = fake_home / ".hermes"
    served = launch / "profiles" / "b"
    for home in (launch, served):
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("model:\n  provider: nous\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.setenv("HERMES_HOME", str(launch))

    # The hermetic conftest pins the state store; restore the sentinel so an argless acquire()
    # resolves through the profile scope instead of one sandbox path for every profile.
    import hermes_state
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)
    # Its ``git gc`` is not what this test is about, and it can block for tens of seconds.
    from tools import checkpoint_maintenance
    monkeypatch.setattr(checkpoint_maintenance, "auto_prune_from_config", lambda: None)

    visited: list[str] = []
    monkeypatch.setattr(
        constants, "prune_scratch_dir_if_due",
        lambda *args, **kwargs: visited.append(str(constants.get_hermes_home())))

    runner = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=True))
    gateway_run._start_gateway_housekeeping(_OneTickStopEvent(), interval=0, runner=runner)

    assert visited == [str(launch), str(served)]
