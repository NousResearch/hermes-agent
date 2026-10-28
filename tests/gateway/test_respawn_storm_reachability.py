"""The respawn-storm breaker must observe the exit-75 REFUSAL path.

The card's core finding: the two brakes that exist were not on the path that
actually respawned the gateway.

* ``record_start_and_check_storm`` was only called from the CLI start path
  (``hermes_cli.gateway._respawn_storm_backoff``), not from the boot path the
  VBS / Task Scheduler actually takes.
* The exit-75 refusals are raised ABOVE that recorder, so a storm of
  refusals never recorded a start at all -- the breaker could not see it.

These tests pin the ordering: the storm record happens BEFORE the claim that
can refuse with exit 75, so a refusal storm is countable.
"""

import pytest

# Imported at module scope ON PURPOSE: ``gateway.run`` must be in sys.modules
# before the live-home I/O guard installs, otherwise importing it here would
# trip the guard on the install's own manifest.json.
import gateway.run as gateway_run
from gateway import run_bootstrap
from gateway import status as gateway_status


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


def _ledger(home):
    path = home / "gateway-starts.log"
    if not path.exists():
        return []
    return [float(line) for line in path.read_text(encoding="utf-8").split() if line.strip()]


class TestStormBreakerCoversTheRefusalPath:
    def test_the_storm_check_runs_before_the_exit_75_claim(self):
        """Ordering is the whole fix.

        The claim at ``_start_gateway_claim_pid_file`` can raise
        ``SystemExit(75)`` -- the "please retry me" code the VBS loop retries
        on. If the storm record happened after the claim, a refusal storm
        would never be recorded and the breaker would stay blind to it.
        """
        import inspect

        source = inspect.getsource(run_bootstrap)
        storm_at = source.index("respawn_storm_backoff")
        claim_at = source.index("_start_gateway_claim_pid_file(force=force)")
        assert storm_at < claim_at, (
            "the storm breaker must record the start BEFORE the exit-75 claim, "
            "otherwise a refusal storm is never counted"
        )

    def test_a_refusal_still_records_a_start(self, tmp_path, monkeypatch):
        """A boot that refuses with exit 75 must still leave a start record.

        This drives the two real steps in the order ``start_gateway`` runs
        them: the storm recorder, then the host-lock claim that can refuse
        with exit 75. The refusal must not erase or skip the record, so the
        next boot's chain grows -- which is what makes the VBS retry loop
        converge instead of spinning.
        """
        from gateway.restart import GATEWAY_SERVICE_RESTART_EXIT_CODE

        order = []
        real_record = gateway_status.record_start_and_check_storm

        def _record(**kwargs):
            order.append("record")
            return real_record(**kwargs)

        def _refuse(*args, **kwargs):
            order.append("refuse")
            raise SystemExit(GATEWAY_SERVICE_RESTART_EXIT_CODE)

        monkeypatch.setattr(gateway_status, "record_start_and_check_storm", _record)

        # Step 1: the storm record, exactly as start_gateway takes it.
        gateway_status.respawn_storm_backoff(force=True)

        # Step 2: the claim that refuses with exit 75 -- the exact refusal the
        # VBS retry loop acts on.
        monkeypatch.setattr(gateway_run, "_claim_host_gateway_role", _refuse)

        with pytest.raises(SystemExit) as exc:
            run_bootstrap._start_gateway_claim_pid_file()
        assert exc.value.code == GATEWAY_SERVICE_RESTART_EXIT_CODE

        # The refusal's boot was recorded: the chain can grow.
        assert order == ["record", "refuse"]
        assert _ledger(tmp_path), "the exit-75 refusal recorded no start"


class TestStormBreakerIsIsolatedFromAutoResume:
    def test_a_refusal_storm_does_not_touch_the_auto_resume_store(self, tmp_path):
        """The auto-resume breaker counts only restart-interrupted boots.

        The storm breaker must not write into its store: a storm of healthy
        boots there made a legitimate restart sequence trip auto-resume
        suppression, and the pending session was silently never resumed.
        """
        from gateway.status import record_start_and_check_storm

        for i in range(4):
            record_start_and_check_storm(max_starts=2, window_s=120.0, now=1000.0 + i)
        assert not (tmp_path / "gateway" / "restart_loop.json").exists()

    def test_the_storm_ledger_is_the_breakers_own_store(self, tmp_path):
        from gateway.status import record_start_and_check_storm

        record_start_and_check_storm(max_starts=5, window_s=120.0, now=1000.0)
        ledger = tmp_path / "gateway-starts.log"
        assert ledger.exists()
        assert _ledger(tmp_path) == [1000.0]
