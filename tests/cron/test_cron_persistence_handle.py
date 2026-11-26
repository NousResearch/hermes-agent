"""Teardown closes the run's storage interface handle."""
from cron.scheduler import _finalize_cron_session


class _Handle:
    closed = 0

    def close(self):
        self.closed += 1


def test_finalize_releases_the_handle_once():
    handle = _Handle()
    _finalize_cron_session(handle, None, "job", "name", "cron_job_1")
    assert handle.closed == 1


def test_finalize_returns_when_the_handle_release_fails():
    class _RefusingHandle(_Handle):
        def close(self):
            super().close()
            raise RuntimeError("release refused")

    handle = _RefusingHandle()
    _finalize_cron_session(handle, None, "job", "name", "cron_job_1")
    assert handle.closed == 1
