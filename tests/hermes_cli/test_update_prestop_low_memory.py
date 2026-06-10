"""Tests for the low-memory pre-stop guard in ``hermes update``.

Regression tests for #26770: on low-memory hosts (1–2 GB lightweight VMs)
keeping the gateway alive while pip's resolver runs OOM-kills it.  The
update flow now stops the current-profile gateway BEFORE the dep install
when available memory is below threshold, and lets the existing
auto-restart block bring it back up afterwards.

These tests verify the decision helper ``_should_prestop_gateway_for_update``
and the memory probe ``_available_memory_mb`` — both pure-Python and
exercisable without any real subprocess / systemd interaction.
"""

from unittest.mock import patch

import pytest

import hermes_cli.main as cli_main


# ---------------------------------------------------------------------------
# _available_memory_mb
# ---------------------------------------------------------------------------


class TestAvailableMemoryMb:
    def test_uses_psutil_when_available(self):
        class _FakeVM:
            available = 4096 * 1024 * 1024  # 4 GB

        with patch("psutil.virtual_memory", return_value=_FakeVM()):
            assert cli_main._available_memory_mb() == 4096

    def test_returns_none_when_psutil_raises_and_no_proc_meminfo(self, monkeypatch):
        # Force psutil import path to raise, and make /proc/meminfo unreadable.
        def _boom():
            raise RuntimeError("nope")

        with patch("psutil.virtual_memory", side_effect=_boom):
            # On macOS / Windows there is no /proc/meminfo so the fallback
            # naturally returns None.  On Linux runners we still want None
            # here, so monkeypatch ``open`` to raise FileNotFoundError when
            # /proc/meminfo is requested.
            real_open = open

            def _fake_open(path, *a, **kw):
                if path == "/proc/meminfo":
                    raise FileNotFoundError(path)
                return real_open(path, *a, **kw)

            monkeypatch.setattr("builtins.open", _fake_open)
            assert cli_main._available_memory_mb() is None

    def test_falls_back_to_proc_meminfo_on_linux(self, monkeypatch):
        # psutil unavailable; /proc/meminfo readable.
        def _boom():
            raise RuntimeError("no psutil")

        meminfo = (
            "MemTotal:        2048000 kB\n"
            "MemFree:          200000 kB\n"
            "MemAvailable:    1572864 kB\n"   # 1.5 GiB → 1536 MB
            "Buffers:           50000 kB\n"
        )

        from io import StringIO

        real_open = open

        def _fake_open(path, *a, **kw):
            if path == "/proc/meminfo":
                return StringIO(meminfo)
            return real_open(path, *a, **kw)

        with patch("psutil.virtual_memory", side_effect=_boom):
            monkeypatch.setattr("builtins.open", _fake_open)
            assert cli_main._available_memory_mb() == 1536


# ---------------------------------------------------------------------------
# _should_prestop_gateway_for_update
# ---------------------------------------------------------------------------


class TestShouldPrestopGatewayForUpdate:
    def test_never_prestop_when_running_inside_gateway(self, monkeypatch):
        """gateway_mode=True is the ``hermes update --gateway`` flow spawned
        by the gateway's /update command; pre-stopping would kill us."""
        monkeypatch.delenv("HERMES_UPDATE_PRESTOP_GATEWAY", raising=False)
        # Even with very low memory we must refuse when gateway_mode=True.
        with patch.object(cli_main, "_available_memory_mb", return_value=256):
            should, reason = cli_main._should_prestop_gateway_for_update(True)
        assert should is False
        assert "gateway" in reason.lower()

    def test_low_memory_triggers_prestop(self, monkeypatch):
        monkeypatch.delenv("HERMES_UPDATE_PRESTOP_GATEWAY", raising=False)
        monkeypatch.delenv("HERMES_UPDATE_PRESTOP_THRESHOLD_MB", raising=False)
        # 1024 MB available, default threshold 2048 → should stop.
        with patch.object(cli_main, "_available_memory_mb", return_value=1024):
            should, reason = cli_main._should_prestop_gateway_for_update(False)
        assert should is True
        assert "1024" in reason
        assert "2048" in reason

    def test_high_memory_skips_prestop(self, monkeypatch):
        monkeypatch.delenv("HERMES_UPDATE_PRESTOP_GATEWAY", raising=False)
        monkeypatch.delenv("HERMES_UPDATE_PRESTOP_THRESHOLD_MB", raising=False)
        with patch.object(cli_main, "_available_memory_mb", return_value=8192):
            should, reason = cli_main._should_prestop_gateway_for_update(False)
        assert should is False
        assert "8192" in reason

    def test_unknown_memory_skips_prestop(self, monkeypatch):
        """Probe failure → preserve current behaviour (no pre-stop)."""
        monkeypatch.delenv("HERMES_UPDATE_PRESTOP_GATEWAY", raising=False)
        with patch.object(cli_main, "_available_memory_mb", return_value=None):
            should, reason = cli_main._should_prestop_gateway_for_update(False)
        assert should is False
        assert "unknown" in reason.lower()

    @pytest.mark.parametrize("value", ["1", "true", "yes", "on", "TRUE", "Yes"])
    def test_env_force_on(self, monkeypatch, value):
        monkeypatch.setenv("HERMES_UPDATE_PRESTOP_GATEWAY", value)
        # Even with plenty of memory the override forces a stop.
        with patch.object(cli_main, "_available_memory_mb", return_value=16384):
            should, reason = cli_main._should_prestop_gateway_for_update(False)
        assert should is True
        assert "override" in reason.lower()

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "FALSE", "No"])
    def test_env_force_off(self, monkeypatch, value):
        monkeypatch.setenv("HERMES_UPDATE_PRESTOP_GATEWAY", value)
        # Even with very low memory the override skips the stop.
        with patch.object(cli_main, "_available_memory_mb", return_value=128):
            should, reason = cli_main._should_prestop_gateway_for_update(False)
        assert should is False
        assert "override" in reason.lower()

    def test_env_override_still_skips_inside_gateway(self, monkeypatch):
        """The gateway_mode guard wins over the env-var override — refusing
        to kill ourselves is non-negotiable."""
        monkeypatch.setenv("HERMES_UPDATE_PRESTOP_GATEWAY", "1")
        with patch.object(cli_main, "_available_memory_mb", return_value=256):
            should, _reason = cli_main._should_prestop_gateway_for_update(True)
        assert should is False

    def test_custom_threshold_env(self, monkeypatch):
        monkeypatch.delenv("HERMES_UPDATE_PRESTOP_GATEWAY", raising=False)
        # 3 GB available, threshold raised to 4 GB → stop.
        monkeypatch.setenv("HERMES_UPDATE_PRESTOP_THRESHOLD_MB", "4096")
        with patch.object(cli_main, "_available_memory_mb", return_value=3072):
            should, reason = cli_main._should_prestop_gateway_for_update(False)
        assert should is True
        assert "4096" in reason

    def test_malformed_threshold_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.delenv("HERMES_UPDATE_PRESTOP_GATEWAY", raising=False)
        monkeypatch.setenv("HERMES_UPDATE_PRESTOP_THRESHOLD_MB", "not-a-number")
        # 1 GB available, garbled threshold → default 2048 still kicks in.
        with patch.object(cli_main, "_available_memory_mb", return_value=1024):
            should, reason = cli_main._should_prestop_gateway_for_update(False)
        assert should is True
        assert "2048" in reason
