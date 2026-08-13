"""Tests for gateway.disk_status — the /api/status disk rollup (NS-656)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from gateway import disk_status
from gateway.disk_status import classify_disk_pressure, collect_disk_status

_GB = 1024  # MB per GB, for readable fixtures


class TestClassifyDiskPressure:
    def test_plentiful_disk_is_ok(self) -> None:
        # 40 GB free of 100 GB.
        assert classify_disk_pressure(40 * _GB, 100 * _GB) == "ok"

    def test_absolute_floor_is_critical_on_any_volume(self) -> None:
        # 200 MB free — below the 256 MB floor even on a huge, low-percent
        # volume would be impossible, so use a big volume mostly full.
        assert classify_disk_pressure(200, 500 * _GB) == "critical"

    def test_high_percent_with_low_headroom_is_critical(self) -> None:
        # 96% used, 800 MB free (< 1 GB headroom) on a 20 GB volume.
        assert classify_disk_pressure(800, 20 * _GB) == "critical"

    def test_high_percent_with_ample_headroom_is_not_critical(self) -> None:
        # 96% used but 20 GB free on a 500 GB volume — percent alone must
        # not trigger critical when absolute headroom is comfortable.
        assert classify_disk_pressure(20 * _GB, 500 * _GB) == "ok"

    def test_low_free_is_elevated(self) -> None:
        # 400 MB free of 4 GB (~90% used): below the 512 MB elevated floor,
        # above the 256 MB critical floor, and under the 95% critical
        # percent gate.
        assert classify_disk_pressure(400, 4 * _GB) == "elevated"

    def test_elevated_percent_band(self) -> None:
        # 88% used, 2.4 GB free of 20 GB — elevated percent gate with
        # headroom under 4 GB.
        assert classify_disk_pressure(2400, 20 * _GB) == "elevated"

    def test_elevated_percent_with_ample_headroom_is_ok(self) -> None:
        # 90% used but 50 GB free of 500 GB.
        assert classify_disk_pressure(50 * _GB, 500 * _GB) == "ok"

    def test_missing_sample_is_unknown(self) -> None:
        assert classify_disk_pressure(None, None) == "unknown"

    def test_malformed_sample_is_unknown(self) -> None:
        assert classify_disk_pressure("lots", 100) == "unknown"
        assert classify_disk_pressure(True, 100) == "unknown"
        assert classify_disk_pressure(-5, 100) == "unknown"
        assert classify_disk_pressure(100, 0) == "unknown"


class TestCollectDiskStatus:
    def test_reports_real_usage(self, tmp_path: Path) -> None:
        status = collect_disk_status(tmp_path)
        assert status["pressure"] in {"ok", "elevated", "critical"}
        assert isinstance(status["total_mb"], int) and status["total_mb"] > 0
        assert isinstance(status["free_mb"], int) and status["free_mb"] >= 0
        assert isinstance(status["used_percent"], float)
        assert 0.0 <= status["used_percent"] <= 100.0

    def test_unreadable_filesystem_degrades_to_unknown(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(_path):  # noqa: ANN001, ANN202
            raise OSError("statvfs failed")

        monkeypatch.setattr(shutil, "disk_usage", _boom)
        status = collect_disk_status(tmp_path)
        assert status == {
            "pressure": "unknown",
            "total_mb": None,
            "free_mb": None,
            "used_percent": None,
        }

    def test_zero_total_degrades_to_unknown(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = shutil._ntuple_diskusage(total=0, used=0, free=0)  # type: ignore[attr-defined]
        monkeypatch.setattr(shutil, "disk_usage", lambda _p: fake)
        status = collect_disk_status(tmp_path)
        assert status["pressure"] == "unknown"
        assert status["total_mb"] is None

    def test_synthetic_full_volume_is_critical(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 10 GB volume with 100 MB free — the OOF-2/OOF-107 state.
        total = 10 * 1024**3
        free = 100 * 1024**2
        fake = shutil._ntuple_diskusage(  # type: ignore[attr-defined]
            total=total, used=total - free, free=free
        )
        monkeypatch.setattr(shutil, "disk_usage", lambda _p: fake)
        status = collect_disk_status(tmp_path)
        assert status["pressure"] == "critical"
        assert status["total_mb"] == 10 * 1024
        assert status["free_mb"] == 100
        assert status["used_percent"] == 99.0

    def test_never_raises_even_without_home(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Default-home resolution failing must degrade, not raise.
        monkeypatch.setattr(
            disk_status.shutil,
            "disk_usage",
            lambda _p: (_ for _ in ()).throw(PermissionError("nope")),
        )
        assert collect_disk_status(None)["pressure"] == "unknown"
