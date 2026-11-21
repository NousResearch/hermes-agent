"""The local runtime's VRAM budget must see the WHOLE card pool, not just card 0.

nvidia-smi emits one CSV row per visible GPU. ``_cached_nvidia_gpu_query`` (the
single shared nvidia-smi read that feeds both the budget probe and the hardware
endpoint) must sum every row — otherwise a multi-GPU machine is budgeted off one
card and spills models to CPU that the combined pool would fit.

Behavior contract, proven red on base / green on the fix:

* 2 identical 24 GiB cards, 22 GiB free each  ->  total 48 GiB / free 44 GiB
  (base reported only 24 GiB / 22 GiB — the spill decision).
* per-card fields (name, pci id, utilization) stay from card 0.
* a single-card host still reads correctly (no regression).
"""
import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.local_runtime import hardware


@pytest.fixture(autouse=True)
def _clear_gpu_query_cache():
    """Each query is module-cached; start every test with a cold cache."""
    hardware._gpu_query_cache = None
    yield
    hardware._gpu_query_cache = None


def _fake_run(stdout: str):
    """A subprocess.run stand-in that yields the given CSV and never shells out."""
    def _run(argv, *a, **k):
        return SimpleNamespace(returncode=0, stdout=stdout)
    return _run


def _query(monkeypatch, csv_text: str) -> dict | None:
    monkeypatch.setattr(hardware, "_nvidia_smi_path", lambda: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(hardware.subprocess, "run", _fake_run(csv_text))
    hardware._gpu_query_cache = None  # force a fresh (uncached) read
    return hardware._cached_nvidia_gpu_query()


def test_multi_gpu_budget_sums_all_cards(monkeypatch):
    # Two 24576 MiB cards, 22528 MiB free each, 2048 MiB used each.
    csv_text = (
        "24576, 22528, NVIDIA GeForce RTX 3090, 0x1e04, 2048, 3\n"
        "24576, 22528, NVIDIA GeForce RTX 3090, 0x1e07, 2048, 7\n"
    )
    q = _query(monkeypatch, csv_text)
    assert q is not None
    assert q["total_bytes"] == (24576 * 2) << 20        # 48 GiB, not 24
    assert q["free_bytes"] == (22528 * 2) << 20         # 44 GiB, not 22
    assert q["used_bytes"] == (2048 * 2) << 20


def test_per_card_identity_stays_from_card_zero(monkeypatch):
    csv_text = (
        "24576, 22528, NVIDIA GeForce RTX 3090, 0x1e04, 2048, 3\n"
        "24576, 22528, NVIDIA GeForce RTX 3090, 0x1e07, 2048, 7\n"
    )
    q = _query(monkeypatch, csv_text)
    assert q is not None
    assert q["gpu_name"] == "NVIDIA GeForce RTX 3090"
    assert q["gpu_pci_id"] == 0x1e04          # card 0's identity, not card 1's
    assert q["gpu_util_percent"] == 3         # card 0's utilization


def test_single_gpu_still_reads_correctly(monkeypatch):
    # One card — the pre-existing path must be unchanged (no regression).
    csv_text = "24576, 22528, NVIDIA GeForce RTX 3090, 0x1e04, 2048, 3\n"
    q = _query(monkeypatch, csv_text)
    assert q is not None
    assert q["total_bytes"] == 24576 << 20
    assert q["free_bytes"] == 22528 << 20
    assert q["used_bytes"] == 2048 << 20


def test_row_count_matches_cards(monkeypatch):
    # Sanity: the CSV row count is what drives the sum; 3 cards -> 3x.
    rows = [f"24576, 22528, NVIDIA GeForce RTX 3090, 0x1e0{i}, 2048, {i}" for i in range(3)]
    q = _query(monkeypatch, "\n".join(rows) + "\n")
    assert q is not None
    assert q["total_bytes"] == (24576 * 3) << 20
