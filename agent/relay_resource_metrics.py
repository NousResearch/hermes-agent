"""Content-free CPU windows supplied by an external cgroup collector.

The collector owns counter reads, delta calculation and source-lifetime checks.
Hermes binds the supplied window; it does not run a hardware sampling service.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent.relay_resource_scheduler import SchedulerToken, _integer, project_timing

RESOURCE_EVENT = "hermes.execution_environment.resource_sample"


@dataclass(frozen=True)
class ResourceWindow:
    start_monotonic_ns: int
    end_monotonic_ns: int
    cpu_usage_us: int
    source_end_unix_ns: int
    source_end_monotonic_ns: int
    clock_mapping_uncertainty_ns: int
    source_identity_before: tuple = field(repr=False)
    source_identity_after: tuple = field(repr=False)
    memory_current_bytes: int | None = None
    cpu_pressure_us: int | None = None
    clock_alignment: str = "source_host_only"


def project_resource_window(window: ResourceWindow, token: SchedulerToken, *, previous_end_ns: int = 0) -> dict:
    data = project_timing(window, token, previous_end_ns=previous_end_ns)
    cpu_usage = _integer(window.cpu_usage_us)
    data.update(
        source="linux_cgroup_v2", coverage="observed_window",
        cpu_usage_delta_us=cpu_usage,
        average_cpu_cores=cpu_usage * 1000 / data["window_duration_ns"],
    )
    for name in ("memory_current_bytes", "cpu_pressure_us"):
        value = getattr(window, name)
        if value is not None:
            data[name] = _integer(value)
    return data
