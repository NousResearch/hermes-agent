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


# ---------------------------------------------------------------------------


from contextlib import ExitStack


class _LivenessPatches:
    """Context manager patching the provider/gateway-pid probes."""

    def __init__(self, *, provider, pids):
        self._provider = provider
        self._pids = pids

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
        return self

    def __exit__(self, *exc):
        return self._stack.__exit__(*exc)


def patch_liveness(*, provider, pids):
    return _LivenessPatches(provider=provider, pids=pids)
