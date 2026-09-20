"""Focused tests for optional external-worker health routing."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli import kanban_health


class _Response:
    def __init__(self, payload, *, status=200):
        self.status = status
        self._body = json.dumps(payload).encode("utf-8")
        self.closed = False

    def read(self):
        return self._body

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def _clear_health_cache():
    kanban_health.clear_health_cache()
    yield
    kanban_health.clear_health_cache()


def _policy(**overrides):
    values = {
        "profile": "clientgpu",
        "enabled": True,
        "base_url": "http://gpu.example/v1",
        "required_model": "worker-model",
        "timeout_seconds": 1,
        "cache_ttl_seconds": 30,
        "fail_closed": True,
    }
    values.update(overrides)
    return kanban_health.ExternalWorkerHealthPolicy(**values)


def _roster():
    return [
        {"name": "coder", "description": "writes code", "has_description": True},
        {"name": "clientgpu", "description": "remote GPU", "has_description": True},
    ], {"coder", "clientgpu"}


def test_healthy_endpoint_keeps_optional_profile():
    roster, valid_names = _roster()
    response = _Response({"data": [{"id": "worker-model"}]})

    with patch("hermes_cli.kanban_health.urllib.request.urlopen", return_value=response) as opener:
        filtered_roster, filtered_names = kanban_health.filter_roster(
            roster, valid_names, [_policy()]
        )

    assert [entry["name"] for entry in filtered_roster] == ["coder", "clientgpu"]
    assert filtered_names == {"coder", "clientgpu"}
    opener.assert_called_once()
    request = opener.call_args.args[0]
    timeout = opener.call_args.kwargs["timeout"]
    assert request.full_url == "http://gpu.example/v1/models"
    assert timeout == 1
    assert response.closed is True


@pytest.mark.parametrize("failure", ["connection", "timeout", "non_200", "missing_model"])
def test_probe_failures_exclude_optional_profile(failure):
    roster, valid_names = _roster()
    if failure == "connection":
        result = OSError("connection refused")
    elif failure == "timeout":
        result = TimeoutError("timed out")
    elif failure == "non_200":
        result = _Response({"error": "unavailable"}, status=503)
    else:
        result = _Response({"data": [{"id": "other-model"}]})

    with patch("hermes_cli.kanban_health.urllib.request.urlopen", side_effect=result):
        filtered_roster, filtered_names = kanban_health.filter_roster(
            roster, valid_names, [_policy()]
        )

    assert [entry["name"] for entry in filtered_roster] == ["coder"]
    assert filtered_names == {"coder"}


def test_cached_health_result_is_reused_until_ttl_expires():
    calls = []
    clock = [10.0]

    def opener(request, timeout):
        calls.append((request.full_url, timeout))
        return _Response({"data": [{"id": "worker-model"}]})

    policy = _policy(cache_ttl_seconds=5)
    assert kanban_health.check_health(policy, now=lambda: clock[0], opener=opener)
    assert kanban_health.check_health(policy, now=lambda: clock[0], opener=opener)
    assert len(calls) == 1

    clock[0] = 16.0
    assert kanban_health.check_health(policy, now=lambda: clock[0], opener=opener)
    assert len(calls) == 2


def test_cache_ttl_starts_after_slow_probe_completes():
    calls = []
    clock = [0.0]

    def opener(request, timeout):
        calls.append((request.full_url, timeout))
        clock[0] += 10.0
        return _Response({"data": [{"id": "worker-model"}]})

    policy = _policy(cache_ttl_seconds=5)
    assert kanban_health.check_health(policy, now=lambda: clock[0], opener=opener)
    assert kanban_health.check_health(policy, now=lambda: clock[0], opener=opener)
    assert len(calls) == 1

    clock[0] += 5.0
    assert kanban_health.check_health(policy, now=lambda: clock[0], opener=opener)
    assert len(calls) == 2


def test_disabled_or_absent_policy_preserves_roster_objects():
    roster, valid_names = _roster()
    assert kanban_health.load_health_policies({}) == ()
    assert kanban_health.load_health_policies(
        {"kanban": {"external_worker_health": {"enabled": False}}}
    ) == ()

    filtered_roster, filtered_names = kanban_health.filter_roster(
        roster, valid_names, []
    )
    assert filtered_roster is roster
    assert filtered_names is valid_names


def test_fail_open_policy_keeps_profile_after_probe_failure():
    roster, valid_names = _roster()
    with patch(
        "hermes_cli.kanban_health.urllib.request.urlopen",
        side_effect=OSError("offline"),
    ):
        filtered_roster, filtered_names = kanban_health.filter_roster(
            roster, valid_names, [_policy(fail_closed=False)]
        )

    assert filtered_roster is roster
    assert filtered_names is valid_names


def test_default_orchestrator_safety_prevents_filtering_protected_profile():
    roster, valid_names = _roster()
    with patch("hermes_cli.kanban_health.urllib.request.urlopen") as opener:
        filtered_roster, filtered_names = kanban_health.filter_roster(
            roster,
            valid_names,
            [_policy(profile="coder")],
            protected_names={"coder"},
        )

    assert filtered_roster is roster
    assert filtered_names is valid_names
    opener.assert_not_called()


def test_probe_failure_does_not_log_endpoint_or_exception_secret(caplog):
    secret = "do-not-log-this-token"
    policy = _policy(base_url=f"https://user:{secret}@gpu.example/v1")
    with patch(
        "hermes_cli.kanban_health.urllib.request.urlopen",
        side_effect=RuntimeError(secret),
    ), caplog.at_level("DEBUG", logger="hermes_cli.kanban_health"):
        assert kanban_health.check_health(policy) is False

    assert secret not in caplog.text


def test_default_config_disables_external_worker_health_gate():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    health_config = DEFAULT_CONFIG["kanban"]["external_worker_health"]
    assert health_config["enabled"] is False
    assert health_config["fail_closed"] is True
    assert health_config["timeout_seconds"] > 0
    assert health_config["cache_ttl_seconds"] > 0


def test_healthy_capacity_multiplier_doubles_configured_limits():
    config = {"kanban": {"external_worker_health": {
        "enabled": True, "profile": "clientgpu",
        "base_url": "http://g57:1234/v1",
        "required_model": "google/gemma-4-12b-qat",
        "capacity_multiplier": 2,
    }}}
    with patch("hermes_cli.kanban_health.check_health", return_value=True):
        assert kanban_health.resolve_capacity_limits(
            config, max_in_progress=6, max_in_progress_per_profile=2
        ) == (12, 4)


def test_unhealthy_capacity_multiplier_restores_baseline():
    config = {"kanban": {"external_worker_health": {
        "enabled": True, "profile": "clientgpu",
        "base_url": "http://g57:1234/v1",
        "required_model": "google/gemma-4-12b-qat",
        "capacity_multiplier": 2,
    }}}
    with patch("hermes_cli.kanban_health.check_health", return_value=False):
        assert kanban_health.resolve_capacity_limits(
            config, max_in_progress=6, max_in_progress_per_profile=2
        ) == (6, 2)


@pytest.mark.parametrize(
    "value",
    [0, -1, "nan", float("inf"), "bad", 2.5, "2.5", 101, 10**100],
)
def test_invalid_capacity_multiplier_does_not_raise_capacity(value):
    config = {"kanban": {"external_worker_health": {
        "enabled": True, "profile": "clientgpu",
        "base_url": "http://g57:1234/v1",
        "required_model": "google/gemma-4-12b-qat",
        "capacity_multiplier": value,
    }}}
    with patch("hermes_cli.kanban_health.check_health", return_value=True):
        assert kanban_health.resolve_capacity_limits(
            config, max_in_progress=6, max_in_progress_per_profile=2
        ) == (6, 2)


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_health_settings_fall_back_to_bounded_cacheable_defaults(non_finite):
    config = {
        "kanban": {
            "external_worker_health": {
                "enabled": True,
                "profile": "clientgpu",
                "base_url": "http://gpu.example/v1",
                "required_model": "worker-model",
                "timeout_seconds": non_finite,
                "cache_ttl_seconds": non_finite,
            }
        }
    }
    policy = kanban_health.load_health_policies(config)[0]
    calls = []

    def opener(request, timeout):
        calls.append(timeout)
        return _Response({"data": [{"id": "worker-model"}]})

    assert math.isfinite(policy.timeout_seconds)
    assert math.isfinite(policy.cache_ttl_seconds)
    assert policy.timeout_seconds == 2.0
    assert policy.cache_ttl_seconds == 30.0
    assert kanban_health.check_health(policy, now=lambda: 10.0, opener=opener)
    assert kanban_health.check_health(policy, now=lambda: 10.0, opener=opener)
    assert calls == [2.0]
