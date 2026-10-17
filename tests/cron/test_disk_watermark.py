"""Read-only disk watermark classification for the cron-store doctor check."""

from __future__ import annotations

import pytest

from cron.store_health import classify_disk_watermark


@pytest.mark.parametrize(
    "free,total,expected",
    [
        (20 << 30, 100 << 30, "healthy"),
        (10 << 30, 50 << 30, "healthy"),
        (5 << 30, 100 << 30, "warning"),
        (20 << 30, 200 << 30, "warning"),  # >10 GiB absolute, but 10% < 15% fraction
        (3 << 30, 50 << 30, "warning"),   # >=3 GiB absolute and 6% >= 5%, but <15% fraction
        (2 << 30, 20 << 30, "critical"),
        (20 << 30, 500 << 30, "critical"),  # >3 GiB absolute, but 4% < 5% fraction
        (800 << 20, 20 << 30, "critical"),
        (None, 100 << 30, "unknown"),
        (100 << 30, 0, "unknown"),
        (101, 100, "unknown"),
        (-1, 100 << 30, "unknown"),
        (True, 100 << 30, "unknown"),
        (10 << 30, None, "unknown"),
        (10 << 30, "lots", "unknown"),
        (10 << 30, True, "unknown"),
        (10 << 30, -5, "unknown"),
    ],
)
def test_classify_disk_watermark(free, total, expected):
    assert classify_disk_watermark(free, total) == expected
