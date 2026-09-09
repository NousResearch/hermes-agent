"""Tests for issue #87033 — the cronjob tool must surface gateway liveness.

The builtin cron ticker only runs inside the gateway process. Before the
fix, ``cronjob(action="create")`` returned a clean success even with no
gateway running, so the agent confidently told the user a recurring task
was scheduled while the job could never fire. The CLI already warned
(``hermes cron list`` / ``hermes cron status``); the agent path did not.

Contract pinned here:

* create with a live gateway → ``gateway_running: true``, no warning;
* create with no gateway → ``gateway_running: false`` + explicit warning
  telling the model the job is saved but will not fire yet;
* non-builtin scheduler providers are exempt (they fire without the gateway);
* a failed liveness probe stays neutral (``gateway_running: null``) instead
  of claiming either way.
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture
def hermes_env(tmp_path, monkeypatch):
    """Isolate HERMES_HOME for each test so jobs don't leak."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "cron").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    import importlib

    import hermes_constants
    importlib.reload(hermes_constants)
    import cron.jobs
    importlib.reload(cron.jobs)
    import cron.scheduler
    importlib.reload(cron.scheduler)

    return home


def _create_job() -> dict:
    from tools.cronjob_tools import cronjob

    return json.loads(
        cronjob(
            action="create",
            schedule="every 10m",
            prompt="say hi",
            name="liveness-probe-job",
            deliver="local",
        )
    )


class TestCreateSurfacesGatewayLiveness:
    def test_create_with_gateway_running_has_no_warning(self, hermes_env):
        with patch_liveness(provider="builtin", pids=[12345]) as patches:
            result = _create_job()

        assert result["success"] is True
        assert result["gateway_running"] is True
        assert "warning" not in result

    def test_create_without_gateway_warns_not_scheduled(self, hermes_env):
        with (
            patch_liveness(provider="builtin", pids=[]),
        ):
            result = _create_job()

        assert result["success"] is True, (
            "the job itself is still created successfully"
        )
        assert result["gateway_running"] is False
        warning = result.get("warning", "")
        assert "not running" in warning.lower()
        assert "will NOT fire" in warning, (
            "the model must be told the job won't fire (#87033)"
        )
        assert "gateway" in warning.lower()

    def test_non_builtin_provider_is_exempt(self, hermes_env):
        """External schedulers (e.g. Chronos) fire without the gateway —
        no false alarm may be raised for them."""
        with patch_liveness(provider="chronos", pids=[]):
            result = _create_job()

        assert result["success"] is True
        assert result["gateway_running"] is True
        assert "warning" not in result

    def test_failed_probe_stays_neutral(self, hermes_env):
        """If liveness cannot be determined, say nothing either way."""
        with patch_liveness(provider=None, pids=[]):  # probe raises → None
            result = _create_job()

        assert result["success"] is True
        assert result["gateway_running"] is None
        assert "warning" not in result


class TestListSurfacesGatewayLiveness:
    """The `list` action has the same silent-inert-job failure mode as
    create (#87033): an agent inspecting jobs with no gateway running must
    learn they are not firing, not just see a clean list."""

    def _list_jobs(self) -> dict:
        from tools.cronjob_tools import cronjob

        return json.loads(cronjob(action="list"))

    def test_list_with_gateway_running_has_no_warning(self, hermes_env):
        _create_job()  # ensure at least one job exists
        with patch_liveness(provider="builtin", pids=[12345]):
            result = self._list_jobs()

        assert result["success"] is True
        assert result["count"] >= 1
        assert result["gateway_running"] is True
        assert "warning" not in result

    def test_list_without_gateway_warns_jobs_inert(self, hermes_env):
        _create_job()
        with patch_liveness(provider="builtin", pids=[]):
            result = self._list_jobs()

        assert result["success"] is True
        assert result["gateway_running"] is False
        warning = result.get("warning", "")
        assert "will NOT fire" in warning, (
            "the model must be told the listed jobs won't fire (#87033)"
        )
        assert "these jobs" in warning

    def test_list_empty_without_gateway_stays_quiet(self, hermes_env):
        """Nothing scheduled + no gateway → no alarm; there is nothing inert."""
        with patch_liveness(provider="builtin", pids=[]):
            result = self._list_jobs()

        assert result["success"] is True
        assert result["count"] == 0
        assert "warning" not in result

    def test_list_non_builtin_provider_is_exempt(self, hermes_env):
        _create_job()
        with patch_liveness(provider="chronos", pids=[]):
            result = self._list_jobs()

        assert result["success"] is True
        assert result["gateway_running"] is True
        assert "warning" not in result


# ---------------------------------------------------------------------------


from contextlib import ExitStack


class _LivenessPatches:
    """Context manager patching the provider/gateway-pid probes.

    Also pins the gateway runtime lock probe to *inactive* by default so
    these tests are deterministic even when a real gateway (holding the
    real lock) runs on the developer's machine — the lock-first check in
    ``_builtin_gateway_liveness`` would otherwise short-circuit to True
    and mask the pid-scan behavior under test. Pass ``lock_active=True``
    to exercise the lock-first path itself.
    """

    def __init__(self, *, provider, pids, lock_active=False):
        self._provider = provider
        self._pids = pids
        self._lock_active = lock_active

    def __enter__(self):
        from unittest.mock import patch

        self._stack = ExitStack()

        def _fake_provider_name():
            if self._provider is None:
                raise RuntimeError("probe failure")
            return self._provider

        self._stack.enter_context(
            patch(
                "hermes_cli.cron._active_cron_provider_name",
                side_effect=_fake_provider_name,
            )
        )
        self._stack.enter_context(
            patch(
                "hermes_cli.gateway.find_gateway_pids",
                return_value=list(self._pids),
            )
        )
        self._stack.enter_context(
            patch(
                "hermes_cli.gateway.named_profile_served_by_running_multiplexer",
                return_value=False,
            )
        )
        self._stack.enter_context(
            patch(
                "gateway.status.is_gateway_runtime_lock_active",
                return_value=self._lock_active,
            )
        )
        return self

    def __exit__(self, *exc):
        return self._stack.__exit__(*exc)


def patch_liveness(*, provider, pids, lock_active=False):
    return _LivenessPatches(provider=provider, pids=pids, lock_active=lock_active)


class TestRuntimeLockFirstLiveness:
    """The gateway runtime lock is the primary liveness signal (#95947).

    ``find_gateway_pids`` can transiently return empty while the gateway is
    up (right after a restart) and excludes the current PID by design
    (#13242), so a single-process gateway probed as dead while its own
    ticker was firing (#94143 class). The lock is held for exactly the
    gateway's lifetime and short-circuits to True before the pid scan.
    """

    def test_lock_active_reports_alive_despite_empty_pid_scan(self, hermes_env):
        """The reported false alarm: lock held, pid scan empty → alive."""
        _create_job()
        with patch_liveness(provider="builtin", pids=[], lock_active=True):
            from tools.cronjob_tools import cronjob

            result = json.loads(cronjob(action="list"))

        assert result["success"] is True
        assert result["gateway_running"] is True
        assert "warning" not in result

    def test_lock_inactive_falls_back_to_pid_scan(self):
        from unittest.mock import patch

        import hermes_cli.cron as cron_cli

        with (
            patch("hermes_cli.cron._active_cron_provider_name", return_value="builtin"),
            patch("gateway.status.is_gateway_runtime_lock_active", return_value=False),
            patch("hermes_cli.gateway.find_gateway_pids", return_value=[424242]),
        ):
            assert cron_cli._builtin_gateway_liveness() is True

    def test_no_lock_no_pids_is_false(self):
        from unittest.mock import patch

        import hermes_cli.cron as cron_cli

        with (
            patch("hermes_cli.cron._active_cron_provider_name", return_value="builtin"),
            patch("gateway.status.is_gateway_runtime_lock_active", return_value=False),
            patch("hermes_cli.gateway.find_gateway_pids", return_value=[]),
            patch(
                "hermes_cli.gateway.named_profile_served_by_running_multiplexer",
                return_value=False,
            ),
        ):
            assert cron_cli._builtin_gateway_liveness() is False

    def test_lock_probe_failure_still_falls_back(self):
        """A crashing lock probe must not poison the tri-state helper —
        the pid scan still decides (the outer except returns None only
        when both probes fail)."""
        from unittest.mock import patch

        import hermes_cli.cron as cron_cli

        with (
            patch("hermes_cli.cron._active_cron_provider_name", return_value="builtin"),
            patch(
                "gateway.status.is_gateway_runtime_lock_active",
                side_effect=OSError("lock probe failed"),
            ),
            patch("hermes_cli.gateway.find_gateway_pids", return_value=[424242]),
        ):
            assert cron_cli._builtin_gateway_liveness() is True

    def test_running_multiplexer_counts_as_alive_for_named_profile(self):
        """A satellite profile has no own PID; the default multiplexer ticks it."""
        from unittest.mock import patch

        import hermes_cli.cron as cron_cli

        with (
            patch("hermes_cli.cron._active_cron_provider_name", return_value="builtin"),
            patch("gateway.status.is_gateway_runtime_lock_active", return_value=False),
            patch("hermes_cli.gateway.find_gateway_pids", return_value=[]),
            patch(
                "hermes_cli.gateway.named_profile_served_by_running_multiplexer",
                return_value=True,
            ),
        ):
            assert cron_cli._builtin_gateway_liveness() is True

    def test_no_multiplexer_and_no_pids_is_still_false(self):
        from unittest.mock import patch

        import hermes_cli.cron as cron_cli

        with (
            patch("hermes_cli.cron._active_cron_provider_name", return_value="builtin"),
            patch("gateway.status.is_gateway_runtime_lock_active", return_value=False),
            patch("hermes_cli.gateway.find_gateway_pids", return_value=[]),
            patch(
                "hermes_cli.gateway.named_profile_served_by_running_multiplexer",
                return_value=False,
            ),
        ):
            assert cron_cli._builtin_gateway_liveness() is False


class TestCronStatusLockFirst:
    """`hermes cron status` shares the lock-first false-alarm fix (#95947).

    Sibling site of `_builtin_gateway_liveness`: it previously declared
    "Gateway is not running — cron jobs will NOT fire" from a bare
    `find_gateway_pids()` miss even while the runtime lock proved the
    gateway (and its ticker) alive.
    """

    def _run_status(self, *, pids, lock_active, lock_pid=None):
        from unittest.mock import patch
        import io
        from contextlib import redirect_stdout

        import hermes_cli.cron as cron_cli

        out = io.StringIO()
        with (
            patch("hermes_cli.cron._active_cron_provider_name", return_value="builtin"),
            patch("hermes_cli.gateway.find_gateway_pids", return_value=list(pids)),
            patch(
                "gateway.status.is_gateway_runtime_lock_active",
                return_value=lock_active,
            ),
            patch("gateway.status.get_running_pid", return_value=lock_pid),
            redirect_stdout(out),
        ):
            cron_cli.cron_status()
        return out.getvalue()

    def test_lock_active_suppresses_not_running_false_alarm(self, hermes_env):
        text = self._run_status(pids=[], lock_active=True, lock_pid=4242)
        # The lock-first contract (#87033): an active runtime lock means the
        # gateway process is alive, so the RED "Gateway is not running" alarm
        # must never fire. Since #98790 a never-written heartbeat is no longer
        # silently green — the YELLOW first-heartbeat notice (which also says
        # "NOT fire") is expected here, so assert on the red alarm itself
        # rather than the "NOT fire" substring both messages share.
        assert "Gateway is not running" not in text
        assert "has not reported a heartbeat" in text
        assert "Gateway is running" in text or "running" in text

    def test_no_lock_no_pids_still_warns(self, hermes_env):
        text = self._run_status(pids=[], lock_active=False)
        assert "NOT fire" in text
