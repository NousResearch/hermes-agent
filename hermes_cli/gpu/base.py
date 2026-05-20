"""Core GPU monitoring types, provider protocol, and registry.

A ``GpuProvider`` turns an endpoint string into a list of ``GpuInfo``.
The cache and CLI layers stay vendor-neutral; each backend lives in
``hermes_cli/gpu/providers/``. Most exporters (NVIDIA DCGM, AMD ROCm,
etc.) emit Prometheus text and can subclass ``PrometheusProvider`` by
declaring just two dicts. A provider that uses a different transport
(e.g. ``nvidia-smi`` XML, a vendor library, cloud metrics JSON) can
implement ``GpuProvider`` directly.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

_MAX_BODY_SIZE = 1024 * 1024  # 1 MB cap on response bodies


@dataclass
class GpuInfo:
    """Vendor-neutral GPU state snapshot."""
    gpu_id: str = ""
    uuid: str = ""
    pci_bus_id: str = ""
    device_name: str = ""
    model_name: str = ""
    hostname: str = ""
    driver_version: str = ""

    gpu_util: Optional[float] = None
    gpu_temp: Optional[float] = None
    fb_used_mib: Optional[float] = None
    fb_free_mib: Optional[float] = None

    @property
    def fb_total_mib(self) -> Optional[float]:
        if self.fb_used_mib is not None and self.fb_free_mib is not None:
            return self.fb_used_mib + self.fb_free_mib
        return None

    @property
    def fb_used_pct(self) -> Optional[float]:
        if self.fb_total_mib and self.fb_total_mib > 0:
            return round(self.fb_used_mib / self.fb_total_mib * 100, 1)
        return None

    @property
    def fb_free_pct(self) -> Optional[float]:
        if self.fb_total_mib and self.fb_total_mib > 0:
            return round(self.fb_free_mib / self.fb_total_mib * 100, 1)
        return None

    def short_model(self) -> str:
        """Return a short model name for display."""
        if not self.model_name:
            return "Unknown GPU"
        name = self.model_name
        # Strip longest matching prefix first so "NVIDIA GeForce RTX 5090"
        # becomes "RTX 5090", not "GeForce RTX 5090".
        for prefix in ("NVIDIA GeForce ", "NVIDIA RTX ", "NVIDIA ", "AMD "):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        return name


@runtime_checkable
class GpuProvider(Protocol):
    """Vendor-specific GPU metrics source."""
    name: str
    label: str
    default_endpoint: str

    def fetch(self, endpoint: str, timeout: float = 5.0) -> list[GpuInfo]: ...
    def validate(self, endpoint: str, timeout: float = 5.0) -> dict: ...


_PROVIDERS: dict[str, GpuProvider] = {}


def register_provider(provider: GpuProvider) -> None:
    """Register a GPU provider.  Call at module import time in providers/."""
    _PROVIDERS[provider.name] = provider


def get_provider(name: str) -> Optional[GpuProvider]:
    """Return the provider registered under ``name``, or None."""
    return _PROVIDERS.get(name)


def list_providers() -> list[GpuProvider]:
    """Return all registered providers."""
    return list(_PROVIDERS.values())


# ─── Prometheus Text-Format Helper Base ───────────────────────────────────────


def _parse_labels(label_str: str) -> dict:
    """Parse a Prometheus-style label group into a dict.

    Accepts the full ``{key="val",...}`` form or the inner portion.
    """
    if not label_str:
        return {}
    inner = label_str.strip("{}")
    if not inner:
        return {}
    return {m.group(1): m.group(2) for m in re.finditer(r'(\w+)="([^"]*)"', inner)}


_METRIC_LINE_RE = re.compile(r'^([^\{]+)\{([^}]*)\}\s+([\d.eE+\-]+)$')


def _parse_prometheus_line(
    line: str, metric_names: set
) -> Optional[tuple[dict, float, str]]:
    """Parse one Prometheus metric line if its name is in ``metric_names``.

    Returns ``(labels, value, metric_name)`` or ``None``.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    for name in metric_names:
        if line.startswith(name + "{"):
            match = _METRIC_LINE_RE.match(line)
            if match:
                return _parse_labels(match.group(2)), float(match.group(3)), name
    return None


class PrometheusProvider:
    """Reusable base for providers that scrape Prometheus-text metrics over HTTP.

    Subclasses declare:

    - ``name``, ``label``, ``default_endpoint``
    - ``GPU_ID_LABELS``: ordered label keys that identify a unique GPU row.
      The first non-empty match is used as the grouping key.
    - ``METRIC_MAP``: ``{metric_name: GpuInfo_float_field}`` — which exporter
      metrics populate which numeric ``GpuInfo`` fields.
    - ``LABEL_MAP``: ``{label_key: GpuInfo_string_field}`` — which Prometheus
      labels populate the identity fields (gpu_id, uuid, model_name, ...).
    """

    name: str = ""
    label: str = ""
    default_endpoint: str = ""

    GPU_ID_LABELS: list = []
    METRIC_MAP: dict = {}
    LABEL_MAP: dict = {}

    def fetch(self, endpoint: str, timeout: float = 5.0) -> list[GpuInfo]:
        text = self._fetch_text(endpoint, timeout)
        if text is None:
            return []
        return list(self._parse(text).values())

    def validate(self, endpoint: str, timeout: float = 5.0) -> dict:
        result = {"success": False, "gpu_count": 0, "gpus": [], "error": None}
        try:
            import httpx
        except ImportError:
            result["error"] = "httpx is not installed"
            return result
        try:
            response = httpx.get(endpoint, timeout=timeout)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            result["error"] = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            return result
        except httpx.ConnectError:
            result["error"] = (
                f"Connection refused — is the {self.label} running at {endpoint}?"
            )
            return result
        except httpx.RequestError as e:
            result["error"] = f"Request failed: {e}"
            return result
        except Exception as e:
            result["error"] = f"Unexpected error: {e}"
            return result

        gpus = self._parse(response.text)
        if not gpus:
            result["error"] = (
                f"Endpoint responded but contained no {self.label} metrics. "
                f"Is this a {self.label}?"
            )
            return result

        for gpu in gpus.values():
            info = {
                "gpu_id": gpu.gpu_id,
                "uuid": gpu.uuid,
                "model": gpu.model_name or gpu.device_name or "Unknown",
            }
            if gpu.gpu_util is not None:
                info["util"] = f"{gpu.gpu_util:.0f}%"
            if gpu.gpu_temp is not None:
                info["temp"] = f"{gpu.gpu_temp:.0f}C"
            if gpu.fb_total_mib:
                info["memory"] = (
                    f"{gpu.fb_used_mib:.0f}/{gpu.fb_total_mib:.0f} MiB "
                    f"({gpu.fb_used_pct:.0f}%)"
                )
            result["gpus"].append(info)

        result["success"] = True
        result["gpu_count"] = len(result["gpus"])
        return result

    def _fetch_text(self, endpoint: str, timeout: float) -> Optional[str]:
        """Fetch metrics text with a body-size guard (10 MB cap)."""
        try:
            import httpx
        except ImportError:
            logger.warning("httpx not installed — GPU monitor unavailable")
            return None
        body = b""
        try:
            with httpx.stream("GET", endpoint, timeout=timeout) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes(chunk_size=64 * 1024):
                    body += chunk
                    if len(body) > _MAX_BODY_SIZE:
                        logger.warning(
                            "GPU metrics response from %s exceeded %d bytes — truncated",
                            endpoint, _MAX_BODY_SIZE,
                        )
                        break
        except Exception as e:
            logger.warning(
                "Failed to fetch %s metrics from %s: %s", self.label, endpoint, e
            )
            return None
        return body.decode("utf-8", errors="replace")

    def _gpu_key(self, labels: dict) -> str:
        for key in self.GPU_ID_LABELS:
            value = labels.get(key)
            if value:
                return value
        return ""

    def _parse(self, text: str) -> dict:
        """Group metric lines by GPU and return dict of ``gpu_key → GpuInfo``."""
        metric_names = set(self.METRIC_MAP.keys())
        gpus: dict = {}
        for line in text.split("\n"):
            parsed = _parse_prometheus_line(line, metric_names)
            if parsed is None:
                continue
            labels, value, metric_name = parsed
            key = self._gpu_key(labels)
            if key not in gpus:
                gpu = GpuInfo()
                for label_key, field in self.LABEL_MAP.items():
                    if label_key in labels:
                        setattr(gpu, field, labels[label_key])
                gpus[key] = gpu
            setattr(gpus[key], self.METRIC_MAP[metric_name], value)
        return gpus
