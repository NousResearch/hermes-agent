"""Verified scheduler windows supplied by a trusted, independently run collector.

Hermes does not enable kernel tracing or request elevated privileges. Windows
describe task switches, not container restarts, runqueue delay, or switch overhead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping
import uuid

SCHEDULER_EVENT = "hermes.execution_environment.scheduler_window"
COUNT_KEYS = (
    "switch_in", "switch_out", "runnable_switch_out", "nonrunnable_switch_out",
    "within_container", "direct_to_other_owned", "direct_from_other_owned",
    "to_idle", "from_idle", "to_unattributed", "from_unattributed",
)
LOSS_KEYS = {"overrun", "commit overrun", "dropped events"}


@dataclass(frozen=True)
class SchedulerToken:
    observation_id: str
    measurement_id: str
    resource_id: str
    source_start_ns: int
    source_identity: tuple = field(repr=False)


@dataclass(frozen=True)
class SchedulerWindow:
    summary_id: str
    start_monotonic_ns: int
    end_monotonic_ns: int
    counts: Mapping[str, int]
    trace_loss: Mapping[str, Mapping[str, int]]
    expected_cpus: tuple[int, ...]
    source_end_unix_ns: int
    source_end_monotonic_ns: int
    clock_mapping_uncertainty_ns: int
    source_identity_before: tuple = field(repr=False)
    source_identity_after: tuple = field(repr=False)
    source_clock: str = "mono"
    attribution_verified: bool = False
    clock_alignment: str = "source_host_only"


def _integer(value: int) -> int:
    if type(value) is not int or not 0 <= value < 2**63:
        raise ValueError("Invalid nonnegative scheduler counter or timestamp")
    return value


def _counts(values: Mapping[str, int]) -> dict[str, int]:
    if set(values) != set(COUNT_KEYS):
        raise ValueError("Unknown or missing scheduler count category")
    counts = {key: _integer(values[key]) for key in COUNT_KEYS}
    outgoing = counts["switch_out"]
    incoming = counts["switch_in"]
    if outgoing != counts["runnable_switch_out"] + counts["nonrunnable_switch_out"]:
        raise ValueError("Inconsistent switch-out state counts")
    if outgoing != sum(counts[key] for key in (
        "within_container", "direct_to_other_owned", "to_idle", "to_unattributed",
    )):
        raise ValueError("Inconsistent outgoing switch categories")
    if incoming != sum(counts[key] for key in (
        "within_container", "direct_from_other_owned", "from_idle", "from_unattributed",
    )):
        raise ValueError("Inconsistent incoming switch categories")
    return counts


def _loss(window: SchedulerWindow) -> dict[str, dict[str, int]]:
    cpus = window.expected_cpus
    if not 1 <= len(cpus) <= 4096 or len(set(cpus)) != len(cpus):
        raise ValueError("Missing or oversized CPU coverage")
    for cpu in cpus:
        _integer(cpu)
    if set(window.trace_loss) != {str(cpu) for cpu in cpus}:
        raise ValueError("Missing CPU trace-loss evidence")
    loss = {}
    for cpu, values in window.trace_loss.items():
        if set(values) != LOSS_KEYS:
            raise ValueError("Missing trace-loss counter")
        loss[cpu] = {key: _integer(values[key]) for key in sorted(LOSS_KEYS)}
        if any(loss[cpu].values()):
            raise ValueError("Lost events make container attribution incomplete")
    return loss


def _project_timing(window, token: SchedulerToken, *, previous_end_ns: int = 0) -> dict:
    """Bind an external source window to its observation and clock domain."""
    if (window.source_identity_before != token.source_identity
            or window.source_identity_after != token.source_identity):
        raise ValueError("Measurement source lifetime changed")
    if window.clock_alignment not in ("same_host", "source_host_only"):
        raise ValueError("Unsupported clock alignment claim")
    start, end = _integer(window.start_monotonic_ns), _integer(window.end_monotonic_ns)
    if start < max(token.source_start_ns, previous_end_ns) or end <= start:
        raise ValueError("Reversed, overlapping, or out-of-observation scheduler window")
    anchor_mono = _integer(window.source_end_monotonic_ns)
    anchor_unix = _integer(window.source_end_unix_ns)
    uncertainty = _integer(window.clock_mapping_uncertainty_ns)
    if anchor_mono < end:
        raise ValueError("Source clock calibration precedes the completed window")
    offset = anchor_unix - anchor_mono
    if offset + start < 0:
        raise ValueError("Invalid scheduler Unix clock mapping")
    return {
        "schema_version": "1", "status": "available",
        "observation_id": token.observation_id, "measurement_id": token.measurement_id,
        "resource_id": token.resource_id, "resource_id_scheme": "oci_container_id_sha256_128_v1",
        "clock_alignment": window.clock_alignment,
        "window_start_monotonic_ns": start, "window_end_monotonic_ns": end,
        "window_start_unix_ns": offset + start, "window_end_unix_ns": offset + end,
        "window_duration_ns": end - start, "clock_mapping_uncertainty_ns": uncertainty,
    }


def project_window(window: SchedulerWindow, token: SchedulerToken, *, previous_end_ns: int = 0) -> dict:
    """Validate a disjoint, loss-free window without exporting PID/cgroup identities."""
    if not isinstance(window.summary_id, str) or not 1 <= len(window.summary_id) <= 64:
        raise ValueError("Invalid scheduler summary ID")
    uuid.UUID(window.summary_id)
    if window.source_clock != "mono" or window.attribution_verified is not True:
        raise ValueError("Unverified scheduler clock or attribution")
    counts, loss = _counts(window.counts), _loss(window)
    return {
        **_project_timing(window, token, previous_end_ns=previous_end_ns),
        "source": "linux_tracefs_sched_switch", "summary_id": window.summary_id,
        "coverage": "verified_kernel_trace_window", "timestamp_resolution_ns": 1000,
        "counts": counts, "trace_loss": loss,
        "events_involving_container": counts["switch_in"] + counts["switch_out"] - counts["within_container"],
    }
