import threading
from types import SimpleNamespace

from agent import credits_tracker
from hermes_constants import (
    get_hermes_home,
    reset_hermes_home_override,
    set_hermes_home_override,
)


def test_background_seed_keeps_routed_profile(monkeypatch, tmp_path):
    """The portal lookup must inherit the profile that owns the agent being seeded."""
    launch_home = tmp_path / "launch"
    served_home = launch_home / "profiles" / "served"
    launch_home.mkdir()
    served_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    monkeypatch.delenv("HERMES_DEV_CREDITS", raising=False)
    monkeypatch.delenv("HERMES_DEV_CREDITS_FIXTURE", raising=False)

    observed_homes = []
    finished = threading.Event()

    def fake_account_info(*, force_fresh=False):
        assert force_fresh is True
        observed_homes.append(get_hermes_home())
        return SimpleNamespace(
            paid_service_access=True,
            paid_service_access_info=SimpleNamespace(
                total_usable_credits=10.0,
                subscription_credits_remaining=10.0,
                purchased_credits_remaining=0.0,
            ),
            subscription=SimpleNamespace(monthly_credits=20.0, rollover_credits=0.0),
        )

    monkeypatch.setattr("hermes_cli.nous_account.get_nous_portal_account_info", fake_account_info)

    class Agent:
        provider = "nous"
        _credits_state = None
        _credits_session_start_micros = None
        _credits_latch = credits_tracker.new_credits_latch()

        @staticmethod
        def _emit_credits_notices():
            finished.set()

    token = set_hermes_home_override(served_home)
    try:
        assert credits_tracker.seed_credits_at_session_start(Agent()) is True
        assert finished.wait(2.0), "background credits seed did not finish"
    finally:
        reset_hermes_home_override(token)

    assert observed_homes == [served_home]
