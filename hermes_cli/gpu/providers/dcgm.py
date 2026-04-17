"""NVIDIA DCGM (Data Center GPU Manager) Prometheus exporter.

Declarative provider: only the metric and label maps differ from the
generic ``PrometheusProvider``. To add another Prometheus-based
exporter (AMD ROCm, Intel XPU, etc.), write a sibling module with the
same shape.
"""

from hermes_cli.gpu.base import PrometheusProvider, register_provider


class DcgmProvider(PrometheusProvider):
    name = "dcgm"
    label = "NVIDIA DCGM exporter"
    default_endpoint = "http://localhost:9400/metrics"

    GPU_ID_LABELS = ["gpu", "UUID"]

    METRIC_MAP = {
        "DCGM_FI_DEV_GPU_UTIL": "gpu_util",
        "DCGM_FI_DEV_GPU_TEMP": "gpu_temp",
        "DCGM_FI_DEV_FB_USED":  "fb_used_mib",
        "DCGM_FI_DEV_FB_FREE":  "fb_free_mib",
    }

    LABEL_MAP = {
        "gpu":                    "gpu_id",
        "UUID":                   "uuid",
        "pci_bus_id":             "pci_bus_id",
        "device":                 "device_name",
        "modelName":              "model_name",
        "Hostname":               "hostname",
        "DCGM_FI_DRIVER_VERSION": "driver_version",
    }


register_provider(DcgmProvider())
