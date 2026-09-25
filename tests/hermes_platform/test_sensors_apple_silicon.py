"""Apple Silicon sensor decoding (pure) and a live read on the macOS lane."""

from __future__ import annotations

import platform
import struct
import time

import pytest

from hermes_platform.sensors.apple_silicon import (
    dvfs_table_mhz,
    energy_watts,
    open_sampler,
    residency_summary,
)


def test_residency_excludes_idle_and_weights_frequency_by_time():
    table = [600.0, 1200.0, 2400.0]
    active, freq = residency_summary([("IDLE", 50), ("V0P2", 25), ("V1P1", 0), ("V2P0", 25)], table)
    assert active == pytest.approx(0.5)
    # Half the busy time at the lowest step, half at the highest.
    assert freq == pytest.approx((table[0] + table[2]) / 2)


def test_residency_with_more_states_than_table_rows_clamps_to_the_top_step():
    _, freq = residency_summary([("OFF", 0), ("P1", 0), ("P2", 10)], [400.0])
    assert freq == 400.0


def test_fully_idle_or_empty_windows_report_no_frequency():
    assert residency_summary([("IDLE", 100)], [600.0]) == (0.0, None)
    assert residency_summary([], [600.0]) == (0.0, None)


def test_dvfs_tables_decode_hz_and_khz_encodings_to_the_same_mhz():
    freqs_mhz = [600, 1332, 3228]
    as_hz = b"".join(struct.pack("<II", f * 1_000_000, 800) for f in [0, *freqs_mhz])
    as_khz = b"".join(struct.pack("<II", f * 1_000, 800) for f in freqs_mhz)
    assert dvfs_table_mhz(as_hz) == dvfs_table_mhz(as_khz) == [float(f) for f in freqs_mhz]


def test_energy_counters_convert_to_watts_over_the_window():
    assert energy_watts(2_000, "mJ", 2.0) == pytest.approx(1.0)
    assert energy_watts(5_000_000_000, "nJ", 1.0) == pytest.approx(5.0)
    assert energy_watts(1, "furlongs", 1.0) is None
    assert energy_watts(1, "mJ", 0.0) is None


@pytest.mark.platforms("macos")
@pytest.mark.skipif(platform.machine() != "arm64", reason="IOReport SoC channels exist on Apple Silicon only")
def test_live_sample_stays_within_the_hardware_envelope():
    sampler = open_sampler()
    assert sampler is not None
    time.sleep(0.25)
    reading = sampler.sample()

    assert reading.interval_s > 0
    assert reading.clusters and reading.cores and reading.gpu is not None
    for domain in [*reading.clusters, *reading.cores, reading.gpu]:
        assert 0.0 <= domain.active <= 1.0
        table = sampler._dvfs[domain.kind]
        if domain.freq_mhz is not None:
            assert min(table) - 1 <= domain.freq_mhz <= max(table) + 1
    assert {"cpu", "gpu"} <= reading.power_w.keys()
    assert all(w >= 0 for w in reading.power_w.values())
    assert reading.temps_c and all(0 < c < 150 for c in reading.temps_c.values())
