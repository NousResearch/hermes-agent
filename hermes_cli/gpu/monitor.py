"""GPU metrics cache, config lookup, and terminal display.

Vendor-neutral: delegates fetching/parsing to the provider named in
``config.yaml`` under ``gpu.provider`` (see ``providers/``).
"""

import logging
import threading
import time
from typing import Optional

from hermes_cli.gpu.base import GpuInfo, GpuProvider, get_provider, list_providers

logger = logging.getLogger(__name__)

_DEFAULT_PROVIDER = "dcgm"
_DEFAULT_REFRESH_INTERVAL = 5.0
_DEFAULT_ENABLED = True

# Timed config cache — _get_gpu_config() is called on the hot path
# (every status-bar repaint, ~250 ms).  Skip load_config() for 30 s.
_CONFIG_TTL = 30.0
_config_cache: dict = {}            # populated config dict
_config_cache_stale: bool = True   # True until first successful load
_config_lock = threading.Lock()


def _get_gpu_config() -> dict:
    """Read GPU config from config.yaml with provider-aware fallbacks.

    Caches the result for _CONFIG_TTL seconds so the status-bar render
    path doesn't call load_config() on every repaint.
    """
    global _config_cache, _config_cache_stale
    now = time.monotonic()
    with _config_lock:
        if not _config_cache_stale:
            elapsed = now - _config_cache.get("_ts", now)
            if elapsed < _CONFIG_TTL:
                return {k: v for k, v in _config_cache.items() if k != "_ts"}

    try:
        from hermes_cli.config import load_config
        gpu_cfg = load_config().get("gpu", {})
        if not isinstance(gpu_cfg, dict):
            gpu_cfg = {}
    except Exception:
        gpu_cfg = {}

    provider_name = gpu_cfg.get("provider", _DEFAULT_PROVIDER)
    provider = get_provider(provider_name)
    default_endpoint = provider.default_endpoint if provider else ""

    cfg = {
        "provider": provider_name,
        "endpoint": gpu_cfg.get("endpoint", default_endpoint),
        "refresh_interval": gpu_cfg.get("refresh_interval", _DEFAULT_REFRESH_INTERVAL),
        "enabled": gpu_cfg.get("enabled", _DEFAULT_ENABLED),
        "_ts": now,
    }

    with _config_lock:
        _config_cache.clear()
        _config_cache.update(cfg)
        _config_cache_stale = False

    return {k: v for k, v in cfg.items() if k != "_ts"}


def _invalidate_config_cache() -> None:
    """Force a re-read of config.yaml on the next _get_gpu_config() call.

    Used by tests and after config mutations (e.g. slash-command /gpu set).
    """
    global _config_cache, _config_cache_stale
    with _config_lock:
        _config_cache.clear()
        _config_cache_stale = True


def is_gpu_monitoring_enabled() -> bool:
    """Return True if GPU monitoring is configured, enabled, and has a known provider."""
    cfg = _get_gpu_config()
    if not (cfg["enabled"] and cfg["endpoint"].strip()):
        return False
    return get_provider(cfg["provider"]) is not None


def get_gpu_endpoint() -> str:
    """Return the configured endpoint."""
    return _get_gpu_config()["endpoint"]


def get_gpu_provider() -> Optional[GpuProvider]:
    """Return the provider selected by config, or None if unknown."""
    return get_provider(_get_gpu_config()["provider"])


def fetch_gpu_metrics(
    endpoint: Optional[str] = None, timeout: float = 5.0
) -> list[GpuInfo]:
    """Fetch and parse GPU metrics via the configured provider."""
    provider = get_gpu_provider()
    if provider is None:
        return []
    if endpoint is None:
        endpoint = get_gpu_endpoint()
    return provider.fetch(endpoint, timeout=timeout)


def validate_gpu_endpoint(
    endpoint: str,
    provider_name: Optional[str] = None,
    timeout: float = 5.0,
) -> dict:
    """Validate an endpoint against a provider (config's provider by default)."""
    name = provider_name or _get_gpu_config()["provider"]
    provider = get_provider(name)
    if provider is None:
        known = ", ".join(sorted(p.name for p in list_providers())) or "(none)"
        return {
            "success": False,
            "gpu_count": 0,
            "gpus": [],
            "error": f"Unknown GPU provider: {name!r}. Known providers: {known}",
        }
    return provider.validate(endpoint, timeout=timeout)


class GpuMetricsCache:
    """Thread-safe cache for GPU metrics with background refresh."""

    def __init__(
        self,
        provider: GpuProvider,
        endpoint: str,
        refresh_interval: float = _DEFAULT_REFRESH_INTERVAL,
    ):
        self._provider = provider
        self._endpoint = endpoint
        self._refresh_interval = refresh_interval
        self._gpus: list[GpuInfo] = []
        self._last_fetch_time: float = 0
        self._lock = threading.Lock()
        self._refresh_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._start_background_refresh()

    @property
    def provider(self) -> GpuProvider:
        return self._provider

    @property
    def endpoint(self) -> str:
        return self._endpoint

    @property
    def refresh_interval(self) -> float:
        return self._refresh_interval

    def _start_background_refresh(self):
        if self._refresh_thread and self._refresh_thread.is_alive():
            return
        self._stop_event.clear()
        self._refresh_thread = threading.Thread(
            target=self._refresh_loop,
            daemon=True,
            name=f"gpu-metrics-cache-{self._provider.name}",
        )
        self._refresh_thread.start()

    def _refresh_loop(self):
        while not self._stop_event.is_set():
            try:
                gpus = self._provider.fetch(self._endpoint)
                now = time.monotonic()
                with self._lock:
                    self._gpus = gpus
                    self._last_fetch_time = now
            except Exception as e:
                logger.debug("GPU metrics cache refresh failed: %s", e)
            self._stop_event.wait(self._refresh_interval)

    def get_cached(self) -> list[GpuInfo]:
        """Return the most recently fetched GPU data."""
        with self._lock:
            return list(self._gpus)

    def force_refresh(self) -> list[GpuInfo]:
        """Force an immediate refresh and return the result."""
        try:
            gpus = self._provider.fetch(self._endpoint)
            now = time.monotonic()
            with self._lock:
                self._gpus = gpus
                self._last_fetch_time = now
                return list(self._gpus)
        except Exception:
            return self.get_cached()

    def stop(self):
        """Stop the background refresh thread."""
        self._stop_event.set()
        if self._refresh_thread:
            self._refresh_thread.join(timeout=2.0)


# Module-level singleton cache — lazily created on first call to
# get_gpu_cache().  Don't start it at import time: importing this
# module must not spawn threads or hit the network.
_gpu_cache: Optional[GpuMetricsCache] = None
_cache_lock = threading.Lock()


def get_gpu_cache() -> Optional[GpuMetricsCache]:
    """Get (or lazily create) the module-level GPU metrics cache.

    Returns None when GPU monitoring is disabled.  Recreates the cache
    if the configured provider, endpoint, or refresh interval has
    changed since the last call.
    """
    global _gpu_cache

    if not is_gpu_monitoring_enabled():
        stop_gpu_cache()
        return None

    cfg = _get_gpu_config()
    provider = get_provider(cfg["provider"])

    with _cache_lock:
        if (
            _gpu_cache is None
            or _gpu_cache.provider.name != provider.name
            or _gpu_cache.endpoint != cfg["endpoint"]
            or _gpu_cache.refresh_interval != cfg["refresh_interval"]
        ):
            if _gpu_cache is not None:
                _gpu_cache.stop()
            _gpu_cache = GpuMetricsCache(
                provider=provider,
                endpoint=cfg["endpoint"],
                refresh_interval=cfg["refresh_interval"],
            )
        return _gpu_cache


def stop_gpu_cache() -> None:
    """Stop and release the module-level GPU metrics cache, if any."""
    global _gpu_cache
    with _cache_lock:
        if _gpu_cache is not None:
            _gpu_cache.stop()
            _gpu_cache = None


# ─── Terminal Display ─────────────────────────────────────────────────────────


def _bar_glyphs(value: Optional[float], width: int) -> str:
    """Render the filled/empty bar glyphs with no color applied."""
    if value is None:
        return "?" * width
    filled = min(width, max(0, int(value / 100 * width)))
    return "█" * filled + "░" * (width - filled)


def _level_bar(
    value: Optional[float],
    width: int = 20,
    color_fn=None,
    code: Optional[str] = None,
) -> str:
    """Render a single-color progress bar (no value judgment).

    Use for metrics where "high" isn't "bad" — e.g. GPU utilization,
    which users *want* to be high.  ``code`` is an ANSI escape string
    (e.g. ``Colors.CYAN``); when omitted, the bar is uncolored.
    """
    bar = _bar_glyphs(value, width)
    if color_fn and code and value is not None:
        return color_fn(bar, code)
    return bar


def _saturation_bar(
    value: Optional[float], width: int = 20, color_fn=None
) -> str:
    """Render a threshold-colored bar (green → yellow → red).

    Use for metrics where "high" means "approaching a hard limit" —
    e.g. GPU memory usage (OOM risk).  ``color_fn`` follows the
    ``hermes_cli.colors.color`` contract: codes are ANSI escape strings.
    """
    bar = _bar_glyphs(value, width)
    if color_fn and value is not None:
        from hermes_cli.colors import Colors
        if value >= 90:
            return color_fn(bar, Colors.RED)
        if value >= 60:
            return color_fn(bar, Colors.YELLOW)
        return color_fn(bar, Colors.GREEN)
    return bar


def _temp_indicator(temp: Optional[float], color_fn=None) -> str:
    """Render temperature with color-coded indicator.

    ``color_fn`` follows the ``hermes_cli.colors.color`` contract (see above).
    """
    if temp is None:
        return "?"
    label = f"{temp:.0f}C"
    if color_fn:
        from hermes_cli.colors import Colors
        if temp >= 85:
            return color_fn(label, Colors.RED)
        if temp >= 70:
            return color_fn(label, Colors.YELLOW)
        return color_fn(label, Colors.GREEN)
    return label


def display_gpu_metrics(
    gpus: Optional[list] = None,
    endpoint: Optional[str] = None,
    width: int = 60,
) -> str:
    """Render a terminal-friendly GPU status display.

    Args:
        gpus: Pre-fetched GPUs. If None, fetches via the configured provider.
        endpoint: Override endpoint. Falls back to config.
        width: Terminal width for bar rendering.

    Returns:
        Formatted string ready for display.
    """
    from hermes_cli.colors import Colors, color

    provider = get_gpu_provider()
    provider_label = provider.label if provider else "GPU"
    if endpoint is None:
        endpoint = get_gpu_endpoint()
    if gpus is None:
        gpus = fetch_gpu_metrics(endpoint)

    if not gpus:
        return color(
            f"  No GPU data available. Is the {provider_label} running?",
            Colors.RED,
        )

    heading = f" GPU Status ({provider_label}) "
    lines = [
        color("┌" + "─" * (width - 2) + "┐", Colors.CYAN),
        color("│" + heading.center(width - 2) + "│", Colors.CYAN),
        color("└" + "─" * (width - 2) + "┘", Colors.CYAN),
        "",
    ]

    for gpu in gpus:
        lines.append(color(f"  GPU {gpu.gpu_id}: {gpu.short_model()}", Colors.BOLD))
        if gpu.uuid:
            lines.append(f"    UUID: {gpu.uuid}")
        if gpu.pci_bus_id:
            lines.append(f"    PCI:  {gpu.pci_bus_id}")
        if gpu.hostname:
            lines.append(f"    Host: {gpu.hostname}")
        lines.append("")

        util_bar = _level_bar(
            gpu.gpu_util, width=width - 20, color_fn=color, code=Colors.CYAN
        )
        if gpu.gpu_util is not None:
            lines.append(f"    GPU Util:  [{util_bar}] {gpu.gpu_util:.0f}%")
        else:
            lines.append(f"    GPU Util:  [{util_bar}] ?%")

        lines.append(f"    Temp:      {_temp_indicator(gpu.gpu_temp, color_fn=color)}")

        if gpu.fb_total_mib:
            fb_bar = _saturation_bar(gpu.fb_used_pct, width=width - 24, color_fn=color)
            pct = gpu.fb_used_pct
            mem_str = (
                f"{gpu.fb_used_mib:.0f} / {gpu.fb_total_mib:.0f} MiB ({pct:.0f}%)"
                if pct is not None
                else "? MiB"
            )
            lines.append(f"    Memory:    [{fb_bar}] {mem_str}")
        else:
            used_str = f"{gpu.fb_used_mib:.0f} MiB" if gpu.fb_used_mib is not None else "? MiB"
            free_str = f"{gpu.fb_free_mib:.0f} MiB" if gpu.fb_free_mib is not None else "? MiB"
            lines.append(f"    Memory:    Used: {used_str}, Free: {free_str}")

        lines.append("")

    lines.append(color(f"  Source: {endpoint}", Colors.DIM))
    lines.append(color("─" * width, Colors.DIM))

    return "\n".join(lines)
