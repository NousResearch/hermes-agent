"""Tests for the GPU monitoring framework."""

import pytest
import httpx
from unittest.mock import patch, MagicMock

from hermes_cli.gpu import (
    GpuInfo,
    GpuMetricsCache,
    PrometheusProvider,
    display_gpu_metrics,
    fetch_gpu_metrics,
    get_gpu_cache,
    get_provider,
    is_gpu_monitoring_enabled,
    list_providers,
    stop_gpu_cache,
    validate_gpu_endpoint,
)
from hermes_cli.gpu.base import _parse_labels, _parse_prometheus_line
from hermes_cli.gpu.monitor import _level_bar, _saturation_bar, _temp_indicator, _invalidate_config_cache


def _mock_stream_response(text: str):
    """Return (mock_response, patched_httpx_stream) for use with httpx.stream().

    Usage:
        mock_resp, stream_patch = _mock_stream_response(sample_text)
        with stream_patch:
            gpus = provider.fetch(endpoint)
    """
    mock_response = MagicMock()
    mock_response.text = text
    mock_response.raise_for_status = MagicMock()
    mock_response.iter_bytes.return_value = [text.encode("utf-8")]
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=False)
    return mock_response, patch("httpx.stream", return_value=mock_response)


@pytest.fixture(autouse=True)
def _reset_gpu_cache_singleton():
    """Stop and release the module-level GPU cache between tests."""
    stop_gpu_cache()
    # Also reset the timed config cache so patched load_config() is picked up.
    _invalidate_config_cache()
    yield
    stop_gpu_cache()
    _invalidate_config_cache()


# ─── Prometheus Parsing Helpers ───────────────────────────────────────────────


class TestParseLabels:
    def test_empty_labels(self):
        assert _parse_labels("") == {}
        assert _parse_labels("{}") == {}

    def test_multiple_labels(self):
        result = _parse_labels('{gpu="0",UUID="GPU-abc",modelName="RTX 5090"}')
        assert result == {"gpu": "0", "UUID": "GPU-abc", "modelName": "RTX 5090"}


class TestParsePrometheusLine:
    _names = {"DCGM_FI_DEV_GPU_UTIL", "DCGM_FI_DEV_GPU_TEMP"}

    def test_metric_line_is_parsed(self):
        line = 'DCGM_FI_DEV_GPU_UTIL{gpu="0",UUID="GPU-abc"} 94'
        labels, value, name = _parse_prometheus_line(line, self._names)
        assert labels["gpu"] == "0"
        assert value == 94
        assert name == "DCGM_FI_DEV_GPU_UTIL"

    def test_skip_comments(self):
        assert _parse_prometheus_line("# HELP DCGM_FI_DEV_GPU_UTIL desc", self._names) is None
        assert _parse_prometheus_line("# TYPE DCGM_FI_DEV_GPU_UTIL gauge", self._names) is None

    def test_skip_unknown_metrics(self):
        assert _parse_prometheus_line('some_other_metric{gpu="0"} 42', self._names) is None

    def test_scientific_notation_value(self):
        _, value, _ = _parse_prometheus_line(
            'DCGM_FI_DEV_GPU_TEMP{gpu="0"} 3.5e+01', self._names
        )
        assert value == 35.0


# ─── GpuInfo Dataclass ────────────────────────────────────────────────────────


class TestGpuInfo:
    def test_fb_total_mib(self):
        assert GpuInfo(fb_used_mib=27387, fb_free_mib=4721).fb_total_mib == 32108

    def test_fb_total_mib_none_when_missing(self):
        assert GpuInfo(fb_used_mib=27387, fb_free_mib=None).fb_total_mib is None

    def test_fb_used_pct(self):
        assert GpuInfo(fb_used_mib=27387, fb_free_mib=4721).fb_used_pct == 85.3

    def test_fb_used_pct_none_when_total_zero(self):
        # Divide-by-zero guard: no framebuffer → no percentage.
        assert GpuInfo(fb_used_mib=0, fb_free_mib=0).fb_used_pct is None

    def test_short_model_strips_geforce_prefix(self):
        # "NVIDIA GeForce " must be stripped before the shorter "NVIDIA ".
        assert GpuInfo(model_name="NVIDIA GeForce RTX 5090").short_model() == "RTX 5090"

    def test_short_model_strips_nvidia_prefix(self):
        assert GpuInfo(model_name="NVIDIA Tesla V100").short_model() == "Tesla V100"

    def test_short_model_unknown(self):
        assert GpuInfo(model_name="").short_model() == "Unknown GPU"


# ─── Rendering Helpers ────────────────────────────────────────────────────────


class TestLevelBar:
    def test_half_fills_half(self):
        assert _level_bar(50, width=10) == "█████░░░░░"

    def test_none_value(self):
        assert _level_bar(None, width=10) == "??????????"

    def test_uses_fixed_color_regardless_of_value(self):
        # A level bar never transitions colors — high GPU util is good, not bad.
        from hermes_cli.colors import Colors

        def color_fn(text, code):
            return f"[{code}:{text}]"
        low = _level_bar(10, width=10, color_fn=color_fn, code=Colors.CYAN)
        high = _level_bar(95, width=10, color_fn=color_fn, code=Colors.CYAN)
        assert low.startswith(f"[{Colors.CYAN}:")
        assert high.startswith(f"[{Colors.CYAN}:")


class TestSaturationBar:
    def test_half_fills_half(self):
        assert _saturation_bar(50, width=10) == "█████░░░░░"

    def test_color_threshold_high(self):
        # color_fn receives an ANSI escape code (Colors.RED etc.), not a name.
        from hermes_cli.colors import Colors

        def color_fn(text, code):
            return f"[{code}:{text}]"
        assert f"[{Colors.RED}:" in _saturation_bar(95, width=10, color_fn=color_fn)
        assert f"[{Colors.YELLOW}:" in _saturation_bar(70, width=10, color_fn=color_fn)
        assert f"[{Colors.GREEN}:" in _saturation_bar(30, width=10, color_fn=color_fn)


class TestTempIndicator:
    def test_format(self):
        assert _temp_indicator(45) == "45C"

    def test_none(self):
        assert _temp_indicator(None) == "?"

    def test_color_threshold_hot(self):
        from hermes_cli.colors import Colors

        def color_fn(text, code):
            return f"[{code}:{text}]"
        assert _temp_indicator(90, color_fn=color_fn) == f"[{Colors.RED}:90C]"


# ─── Provider Registry ───────────────────────────────────────────────────────


class TestProviderRegistry:
    def test_dcgm_is_registered(self):
        dcgm = get_provider("dcgm")
        assert dcgm is not None
        assert dcgm.name == "dcgm"
        assert dcgm.default_endpoint.endswith(":9400/metrics")

    def test_unknown_provider_returns_none(self):
        assert get_provider("nonexistent-provider") is None

    def test_list_providers_includes_dcgm(self):
        assert any(p.name == "dcgm" for p in list_providers())

    def test_custom_prometheus_provider(self):
        # A minimal bespoke PrometheusProvider subclass to demonstrate that the
        # base class handles new exporters with just metric + label maps.
        class FakeProvider(PrometheusProvider):
            name = "fake"
            label = "Fake exporter"
            default_endpoint = "http://localhost:1234/metrics"
            GPU_ID_LABELS = ["gpu_id"]
            METRIC_MAP = {"fake_gpu_util_pct": "gpu_util"}
            LABEL_MAP = {"gpu_id": "gpu_id", "model": "model_name"}

        sample = 'fake_gpu_util_pct{gpu_id="0",model="FakeCard"} 42\n'
        _, stream_patch = _mock_stream_response(sample)
        with stream_patch:
            gpus = FakeProvider().fetch("http://localhost:1234/metrics")

        assert len(gpus) == 1
        assert gpus[0].gpu_id == "0"
        assert gpus[0].model_name == "FakeCard"
        assert gpus[0].gpu_util == 42


# ─── Config-Derived Flags ─────────────────────────────────────────────────────


class TestIsGpuMonitoringEnabled:
    def test_enabled_when_configured(self):
        with patch("hermes_cli.config.load_config", return_value={
            "gpu": {"provider": "dcgm", "endpoint": "http://example.com/metrics", "enabled": True}
        }):
            assert is_gpu_monitoring_enabled() is True

    def test_disabled_with_empty_endpoint(self):
        with patch("hermes_cli.config.load_config", return_value={
            "gpu": {"provider": "dcgm", "endpoint": "", "enabled": True}
        }):
            assert is_gpu_monitoring_enabled() is False

    def test_disabled_when_explicitly_off(self):
        with patch("hermes_cli.config.load_config", return_value={
            "gpu": {"provider": "dcgm", "endpoint": "http://example.com/metrics", "enabled": False}
        }):
            assert is_gpu_monitoring_enabled() is False

    def test_disabled_when_provider_unknown(self):
        with patch("hermes_cli.config.load_config", return_value={
            "gpu": {"provider": "bogus", "endpoint": "http://example.com/metrics", "enabled": True}
        }):
            assert is_gpu_monitoring_enabled() is False


# ─── Fetch via DCGM Provider ──────────────────────────────────────────────────


class TestFetchGpuMetrics:
    def test_successful_fetch(self):
        sample = """# HELP DCGM_FI_DEV_GPU_TEMP GPU temperature (in C).
# TYPE DCGM_FI_DEV_GPU_TEMP gauge
DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-b0516a31",modelName="NVIDIA GeForce RTX 5090",Hostname="test-host"} 48
DCGM_FI_DEV_GPU_UTIL{gpu="0",UUID="GPU-b0516a31",modelName="NVIDIA GeForce RTX 5090",Hostname="test-host"} 94
DCGM_FI_DEV_FB_FREE{gpu="0",UUID="GPU-b0516a31",modelName="NVIDIA GeForce RTX 5090",Hostname="test-host"} 4721
DCGM_FI_DEV_FB_USED{gpu="0",UUID="GPU-b0516a31",modelName="NVIDIA GeForce RTX 5090",Hostname="test-host"} 27387
"""
        _, stream_patch = _mock_stream_response(sample)
        with stream_patch:
            gpus = fetch_gpu_metrics(endpoint="http://test:9400/metrics")

        assert len(gpus) == 1
        gpu = gpus[0]
        assert gpu.gpu_id == "0"
        assert gpu.uuid == "GPU-b0516a31"
        assert gpu.model_name == "NVIDIA GeForce RTX 5090"
        assert gpu.hostname == "test-host"
        assert gpu.gpu_util == 94
        assert gpu.gpu_temp == 48
        assert gpu.fb_used_mib == 27387
        assert gpu.fb_free_mib == 4721

    def test_fetch_failure_returns_empty(self):
        with patch("httpx.stream", side_effect=Exception("Connection refused")):
            assert fetch_gpu_metrics(endpoint="http://test:9400/metrics") == []

    def test_multiple_gpus_grouped_by_id(self):
        sample = """DCGM_FI_DEV_GPU_UTIL{gpu="0",UUID="GPU-000"} 94
DCGM_FI_DEV_GPU_UTIL{gpu="1",UUID="GPU-111"} 45
DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-000"} 48
DCGM_FI_DEV_GPU_TEMP{gpu="1",UUID="GPU-111"} 52
"""
        _, stream_patch = _mock_stream_response(sample)
        with stream_patch:
            gpus = fetch_gpu_metrics(endpoint="http://test:9400/metrics")

        assert len(gpus) == 2
        assert [g.gpu_id for g in gpus] == ["0", "1"]
        assert [g.uuid for g in gpus] == ["GPU-000", "GPU-111"]

    def test_body_size_limit_truncates_large_response(self):
        from hermes_cli.gpu import base
        from hermes_cli.gpu.base import PrometheusProvider

        # Simulate a response exceeding the 10 MB cap.
        big_chunk = b"x" * (base._MAX_BODY_SIZE + 1024)
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_bytes.return_value = [big_chunk]
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        with patch("httpx.stream", return_value=mock_response):
            provider = get_provider("dcgm")
            gpus = provider.fetch("http://localhost:9400/metrics")
        # Truncation produces no parseable DCGM metrics, so the result is empty.
        assert gpus == []


# ─── Display GPU Metrics ──────────────────────────────────────────────────────


class TestDisplayGpuMetrics:
    def test_display_with_data(self):
        sample = """DCGM_FI_DEV_GPU_UTIL{gpu="0",UUID="GPU-000",modelName="RTX 5090"} 94
DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-000",modelName="RTX 5090"} 51
DCGM_FI_DEV_FB_USED{gpu="0",UUID="GPU-000",modelName="RTX 5090"} 27387
DCGM_FI_DEV_FB_FREE{gpu="0",UUID="GPU-000",modelName="RTX 5090"} 4721
"""
        _, stream_patch = _mock_stream_response(sample)
        with stream_patch:
            output = display_gpu_metrics(endpoint="http://test:9400/metrics", width=50)

        assert "GPU Status" in output
        assert "RTX 5090" in output
        assert "94%" in output
        assert "51C" in output

    def test_display_no_data(self):
        mock_response = MagicMock(text="")
        mock_response.raise_for_status = MagicMock()
        with patch("httpx.get", return_value=mock_response):
            output = display_gpu_metrics(endpoint="http://test:9400/metrics")
        assert "No GPU data available" in output


# ─── GpuMetricsCache ──────────────────────────────────────────────────────────


def _dcgm_cache(endpoint="http://localhost:9999/metrics"):
    return GpuMetricsCache(provider=get_provider("dcgm"), endpoint=endpoint)


class TestGpuMetricsCache:
    def test_starts_background_thread(self):
        cache = _dcgm_cache()
        try:
            assert cache._refresh_thread is not None
            assert cache._refresh_thread.is_alive()
        finally:
            cache.stop()

    def test_force_refresh_populates_cache(self):
        sample = (
            'DCGM_FI_DEV_GPU_UTIL{gpu="0",UUID="GPU-000"} 75\n'
            'DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-000"} 55\n'
        )
        _, stream_patch = _mock_stream_response(sample)
        cache = _dcgm_cache()
        try:
            with stream_patch:
                gpus = cache.force_refresh()
            assert len(gpus) == 1
            assert gpus[0].gpu_util == 75
            assert gpus[0].gpu_temp == 55
        finally:
            cache.stop()

    def test_force_refresh_returns_empty_on_failure(self):
        cache = _dcgm_cache()
        try:
            with patch("httpx.stream", side_effect=Exception("Connection refused")):
                assert cache.force_refresh() == []
        finally:
            cache.stop()

    def test_get_cached_returns_copy(self):
        # Readers must not observe mutations to the internal list.
        cache = _dcgm_cache()
        try:
            assert cache.get_cached() is not cache.get_cached()
        finally:
            cache.stop()

    def test_stop_cleans_up_thread(self):
        cache = _dcgm_cache()
        cache.stop()
        assert not cache._refresh_thread.is_alive()


# ─── Validate GPU Endpoint ────────────────────────────────────────────────────


class TestValidateGpuEndpoint:
    def test_valid_endpoint_single_gpu(self):
        sample = """DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-b0516a31",modelName="NVIDIA GeForce RTX 5090"} 48
DCGM_FI_DEV_GPU_UTIL{gpu="0",UUID="GPU-b0516a31",modelName="NVIDIA GeForce RTX 5090"} 94
DCGM_FI_DEV_FB_FREE{gpu="0",UUID="GPU-b0516a31",modelName="NVIDIA GeForce RTX 5090"} 4721
DCGM_FI_DEV_FB_USED{gpu="0",UUID="GPU-b0516a31",modelName="NVIDIA GeForce RTX 5090"} 27387
"""
        mock_response = MagicMock(text=sample)
        mock_response.raise_for_status = MagicMock()
        with patch("httpx.get", return_value=mock_response):
            result = validate_gpu_endpoint(
                "http://test:9400/metrics", provider_name="dcgm"
            )

        assert result["success"] is True
        assert result["gpu_count"] == 1
        assert result["error"] is None
        gpu = result["gpus"][0]
        assert gpu["gpu_id"] == "0"
        assert gpu["model"] == "NVIDIA GeForce RTX 5090"
        assert gpu["util"] == "94%"
        assert gpu["temp"] == "48C"
        assert gpu["memory"] == "27387/32108 MiB (85%)"

    def test_valid_endpoint_multiple_gpus(self):
        sample = """DCGM_FI_DEV_GPU_UTIL{gpu="0",UUID="GPU-000",modelName="RTX 4090"} 50
DCGM_FI_DEV_GPU_UTIL{gpu="1",UUID="GPU-111",modelName="RTX 4090"} 75
"""
        mock_response = MagicMock(text=sample)
        mock_response.raise_for_status = MagicMock()
        with patch("httpx.get", return_value=mock_response):
            result = validate_gpu_endpoint(
                "http://test:9400/metrics", provider_name="dcgm"
            )

        assert result["success"] is True
        assert result["gpu_count"] == 2

    def test_connection_refused(self):
        with patch("httpx.get", side_effect=httpx.ConnectError("Connection refused")):
            result = validate_gpu_endpoint(
                "http://localhost:9999/metrics", provider_name="dcgm"
            )
        assert result["success"] is False
        assert "Connection refused" in result["error"]

    def test_http_error(self):
        mock_response = MagicMock(status_code=500, text="Internal Server Error")
        exc = httpx.HTTPStatusError("500", request=MagicMock(), response=mock_response)
        with patch("httpx.get", side_effect=exc):
            result = validate_gpu_endpoint(
                "http://test:9400/metrics", provider_name="dcgm"
            )
        assert result["success"] is False
        assert "500" in result["error"]

    def test_no_metrics(self):
        mock_response = MagicMock(text="some_random_metric 42\n")
        mock_response.raise_for_status = MagicMock()
        with patch("httpx.get", return_value=mock_response):
            result = validate_gpu_endpoint(
                "http://test:9400/metrics", provider_name="dcgm"
            )
        assert result["success"] is False
        assert "no" in result["error"].lower() and "metrics" in result["error"].lower()

    def test_gpu_info_without_optional_fields(self):
        # Only util is present — no temp, no framebuffer — and the validator
        # must omit those keys rather than emitting junk values.
        mock_response = MagicMock(
            text='DCGM_FI_DEV_GPU_UTIL{gpu="0",UUID="GPU-000",modelName="RTX 5090"} 50\n'
        )
        mock_response.raise_for_status = MagicMock()
        with patch("httpx.get", return_value=mock_response):
            result = validate_gpu_endpoint(
                "http://test:9400/metrics", provider_name="dcgm"
            )

        assert result["success"] is True
        gpu = result["gpus"][0]
        assert gpu["util"] == "50%"
        assert "temp" not in gpu
        assert "memory" not in gpu

    def test_unknown_provider_name(self):
        result = validate_gpu_endpoint("http://anywhere", provider_name="bogus")
        assert result["success"] is False
        assert "Unknown GPU provider" in result["error"]


# ─── Singleton Cache Accessor ─────────────────────────────────────────────────


class TestGetGpuCacheSingleton:
    def test_returns_none_when_disabled(self):
        with patch("hermes_cli.config.load_config", return_value={
            "gpu": {"provider": "dcgm", "endpoint": "http://test:9400/metrics", "enabled": False}
        }):
            assert get_gpu_cache() is None

    def test_returns_cache_when_enabled(self):
        with patch("hermes_cli.config.load_config", return_value={
            "gpu": {"provider": "dcgm", "endpoint": "http://test:9400/metrics", "enabled": True, "refresh_interval": 3.0}
        }):
            cache = get_gpu_cache()
        assert cache is not None
        assert cache.endpoint == "http://test:9400/metrics"
        assert cache.refresh_interval == 3.0
        assert cache.provider.name == "dcgm"

    def test_recreates_cache_on_endpoint_change(self):
        with patch("hermes_cli.config.load_config", return_value={
            "gpu": {"provider": "dcgm", "endpoint": "http://old:9400/metrics", "enabled": True, "refresh_interval": 3.0}
        }):
            cache1 = get_gpu_cache()
        _invalidate_config_cache()

        with patch("hermes_cli.config.load_config", return_value={
            "gpu": {"provider": "dcgm", "endpoint": "http://new:9400/metrics", "enabled": True, "refresh_interval": 3.0}
        }):
            cache2 = get_gpu_cache()

        assert cache2.endpoint == "http://new:9400/metrics"
        assert cache2 is not cache1

    def test_recreates_cache_on_interval_change(self):
        with patch("hermes_cli.config.load_config", return_value={
            "gpu": {"provider": "dcgm", "endpoint": "http://test:9400/metrics", "enabled": True, "refresh_interval": 3.0}
        }):
            cache1 = get_gpu_cache()
        _invalidate_config_cache()

        with patch("hermes_cli.config.load_config", return_value={
            "gpu": {"provider": "dcgm", "endpoint": "http://test:9400/metrics", "enabled": True, "refresh_interval": 5.0}
        }):
            cache2 = get_gpu_cache()

        assert cache2.refresh_interval == 5.0
        assert cache2 is not cache1
