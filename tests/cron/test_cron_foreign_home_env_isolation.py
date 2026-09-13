"""Regression: ticking ANOTHER profile's cron store must not permanently rewrite this process's env.

The Desktop backend ticks every local profile's store (`CronScheduler._start_multiplex`), and a job
owned by another profile reloads THAT profile's `.env` into the shared `os.environ` with
`override=True` (`cron.scheduler._reload_dotenv_and_publish_delivery_target`,
`_run_no_agent_job`). The foreign values used to stick for the life of the process, so the
backend's OWN profile resolved the other profile's credential afterwards — observed in the field
as a profile whose model key was replaced by the root `.env`'s stale one, answering HTTP 401 on
every model call until the backend restarted (the CLI was fine, so it read as "only the Desktop is
broken").

The invariant: a foreign tick keeps its own values INSIDE the block (jobs must still see their
owning profile's secrets) and leaves nothing behind OUTSIDE it.
"""

import os

import pytest


@pytest.fixture()
def two_homes(tmp_path, monkeypatch):
    own = tmp_path / "own"
    own.mkdir()
    (own / ".env").write_text("AGENT_API_KEY=own-key\n", encoding="utf-8")
    other = tmp_path / "profiles" / "other"
    other.mkdir(parents=True)
    (other / ".env").write_text(
        "AGENT_API_KEY=other-key\nOTHER_ONLY=other-only\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(own))
    monkeypatch.setenv("AGENT_API_KEY", "own-key")
    monkeypatch.delenv("OTHER_ONLY", raising=False)
    return own, other


def _cron_dotenv_load(home):
    """The load every cron run performs for its owning profile."""
    from hermes_cli.env_loader import load_hermes_dotenv

    load_hermes_dotenv(hermes_home=home)


class TestForeignProfileTickIsIsolated:
    def test_foreign_tick_env_restored_after_block(self, two_homes):
        from cron.scheduler_provider import _profile_cron_scope

        _own, other = two_homes
        with _profile_cron_scope(other):
            _cron_dotenv_load(other)
            # The job itself must see its OWNING profile's credentials.
            assert os.environ["AGENT_API_KEY"] == "other-key"
            assert os.environ["OTHER_ONLY"] == "other-only"

        # ... and this process's own credential must survive the tick.
        assert os.environ["AGENT_API_KEY"] == "own-key"
        assert "OTHER_ONLY" not in os.environ

    def test_foreign_tick_env_restored_when_job_raises(self, two_homes):
        from cron.scheduler_provider import _profile_cron_scope

        _own, other = two_homes
        with pytest.raises(RuntimeError):
            with _profile_cron_scope(other):
                _cron_dotenv_load(other)
                raise RuntimeError("job blew up mid-run")

        assert os.environ["AGENT_API_KEY"] == "own-key"
        assert "OTHER_ONLY" not in os.environ

    def test_own_home_tick_is_not_rolled_back(self, two_homes, monkeypatch):
        """A tick for this process's own home IS this process's environment — untouched."""
        from cron.scheduler_provider import _profile_cron_scope

        own, _other = two_homes
        monkeypatch.setenv("AGENT_API_KEY", "stale-shell-export")
        with _profile_cron_scope(own):
            _cron_dotenv_load(own)
            assert os.environ["AGENT_API_KEY"] == "own-key"

        assert os.environ["AGENT_API_KEY"] == "own-key"
