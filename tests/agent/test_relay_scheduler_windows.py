"""Pure contracts for trusted, loss-free scheduler summaries; no kernel tracing."""

from dataclasses import replace
import json

import pytest

from agent.relay_scheduler_windows import SchedulerToken, SchedulerWindow, project_window

IDENTITY = ("boot", "container-start", 4, 5, 6, 7)
COUNTS = dict(switch_in=19, switch_out=15, runnable_switch_out=5, nonrunnable_switch_out=10,
              within_container=3, direct_to_other_owned=4, direct_from_other_owned=7,
              to_idle=2, from_idle=1, to_unattributed=6, from_unattributed=8)


def _token():
    return SchedulerToken("observation", "measurement", "opaque-resource", 1_000_000_010, IDENTITY)


def _window(token=None, **changes):
    token = token or _token()
    start = token.source_start_ns + 1000
    window = SchedulerWindow(
        summary_id="00000000-0000-4000-8000-000000000001", start_monotonic_ns=start,
        end_monotonic_ns=start + 1_000_000, counts=dict(COUNTS), expected_cpus=(0, 1),
        trace_loss={str(cpu): {key: 0 for key in ("overrun", "commit overrun", "dropped events")}
                    for cpu in (0, 1)},
        source_end_unix_ns=start + 101_000_000, source_end_monotonic_ns=start + 1_000_000,
        clock_mapping_uncertainty_ns=100, source_identity_before=token.source_identity,
        source_identity_after=token.source_identity, attribution_verified=True,
    )
    return replace(window, **changes)


def test_scheduler_counts_clock_and_identity_projection():
    token, window = _token(), _window()
    data = project_window(window, token)
    assert data["counts"] == COUNTS and data["events_involving_container"] == 31
    assert data["window_end_unix_ns"] - data["window_start_unix_ns"] == 1_000_000
    assert data["window_start_unix_ns"] == window.start_monotonic_ns + 100_000_000
    assert (data["measurement_id"], data["observation_id"], data["resource_id"]) == (
        token.measurement_id, token.observation_id, token.resource_id,
    )
    assert data["coverage"] == "verified_kernel_trace_window" and data["clock_alignment"] == "source_host_only"
    assert "source_identity" not in json.dumps(data) and "container-start" not in json.dumps(data)


@pytest.mark.parametrize("changes", [
    {"counts": {**COUNTS, "switch_in": True}}, {"counts": {**COUNTS, "to_idle": -1}},
    {"counts": {key: value for key, value in COUNTS.items() if key != "switch_out"}},
    {"counts": {**COUNTS, "extra": 0}}, {"counts": {**COUNTS, "runnable_switch_out": 6}},
    {"counts": {**COUNTS, "direct_to_other_owned": 5}}, {"counts": {**COUNTS, "direct_from_other_owned": 8}},
    {"trace_loss": {}}, {"expected_cpus": (0,), "trace_loss": {"0": {"overrun": 0}}},
    {"expected_cpus": (0,), "trace_loss": {"0": {"overrun": True, "commit overrun": 0, "dropped events": 0}}},
    {"expected_cpus": (0,), "trace_loss": {"0": {"overrun": 1, "commit overrun": 0, "dropped events": 0}}},
    {"expected_cpus": ()}, {"expected_cpus": (0, 0)}, {"expected_cpus": (False, 1)},
    {"source_identity_before": ("old",)}, {"source_identity_after": ("restarted",)},
    {"source_clock": "global"}, {"attribution_verified": False}, {"clock_alignment": "guessed"},
    {"start_monotonic_ns": True}, {"start_monotonic_ns": 0}, {"end_monotonic_ns": 1},
    {"source_end_monotonic_ns": 1}, {"source_end_unix_ns": 0}, {"clock_mapping_uncertainty_ns": -1},
    {"summary_id": "raw-container-or-cgroup-name"},
])
def test_scheduler_rejects_incomplete_or_inconsistent_evidence(changes):
    with pytest.raises(ValueError):
        project_window(_window(**changes), _token())


def test_scheduler_windows_must_be_disjoint_but_may_be_adjacent():
    window = _window()
    with pytest.raises(ValueError):
        project_window(window, _token(), previous_end_ns=window.start_monotonic_ns + 1)
    assert project_window(window, _token(), previous_end_ns=window.start_monotonic_ns)["counts"] == COUNTS
