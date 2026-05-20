"""GPU monitoring for the Hermes CLI.

A small, pluggable framework for reading live GPU state from a vendor-
specific endpoint and rendering it in the status bar. The public API is
vendor-neutral; each backend is a ``GpuProvider`` under ``providers/``.
Today the only registered provider is NVIDIA DCGM.
"""

from hermes_cli.gpu.base import (
    GpuInfo,
    GpuProvider,
    PrometheusProvider,
    get_provider,
    list_providers,
    register_provider,
)
from hermes_cli.gpu.monitor import (
    GpuMetricsCache,
    display_gpu_metrics,
    fetch_gpu_metrics,
    get_gpu_cache,
    get_gpu_endpoint,
    get_gpu_provider,
    is_gpu_monitoring_enabled,
    stop_gpu_cache,
    validate_gpu_endpoint,
    _invalidate_config_cache,  # test helper
)

# Import providers so they self-register at package import time.
from hermes_cli.gpu import providers as _providers  # noqa: F401

__all__ = [
    "GpuInfo",
    "GpuMetricsCache",
    "GpuProvider",
    "PrometheusProvider",
    "display_gpu_metrics",
    "fetch_gpu_metrics",
    "get_gpu_cache",
    "get_gpu_endpoint",
    "get_gpu_provider",
    "get_provider",
    "is_gpu_monitoring_enabled",
    "list_providers",
    "register_provider",
    "stop_gpu_cache",
    "validate_gpu_endpoint",
]
