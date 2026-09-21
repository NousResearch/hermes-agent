"""_nvidia_vram() must total every GPU row nvidia-smi prints. Tensor-split
engines (llama.cpp) address the summed VRAM of all cards, but the probe used
to read only GPU 0's CSV row — a 2x24GiB rig was budgeted at one card, so the
context ladder capped at the single-card rung and long contexts failed at
runtime with the card half idle. A card reporting a per-field "N/A" must skip
only its own row, never discard the healthy rows already totaled."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import hermes_cli.local_runtime.hardware as hardware


def _fake_smi(monkeypatch, stdout: str, returncode: int = 0):
    monkeypatch.setattr(hardware, "_nvidia_smi_path", lambda: "/fake/nvidia-smi")
    monkeypatch.setattr(
        hardware.subprocess, "run",
        lambda *a, **k: SimpleNamespace(returncode=returncode, stdout=stdout))


def test_nvidia_vram_totals_all_gpu_rows(monkeypatch):
    _fake_smi(monkeypatch, "24576, 24000\n24576, 23800\n")
    total, free = hardware._nvidia_vram()
    assert total == 49152 << 20
    assert free == 47800 << 20


def test_nvidia_vram_single_gpu_is_unchanged(monkeypatch):
    _fake_smi(monkeypatch, "24576, 24000\n")
    total, free = hardware._nvidia_vram()
    assert total == 24576 << 20
    assert free == 24000 << 20


def test_nvidia_vram_keeps_good_rows_when_one_card_reports_na(monkeypatch):
    # smi emits "N/A" per field, so a failing/driver-mismatched card produces a
    # two-field row — that card is skipped, the healthy cards keep their budget.
    _fake_smi(monkeypatch, "24576, 24000\nN/A, N/A\n24576, 23800\n")
    total, free = hardware._nvidia_vram()
    assert total == 49152 << 20
    assert free == 47800 << 20


def test_nvidia_vram_skips_malformed_rows(monkeypatch):
    # Covers short rows and the quoted-CSV shape in addition to per-field N/A above.
    _fake_smi(monkeypatch, "24576, 24000\n[N/A]\n\"24576\", \"24000\"\n24576, 23800\n")
    total, free = hardware._nvidia_vram()
    assert total == 49152 << 20
    assert free == 47800 << 20


def test_nvidia_vram_all_zero_reports_no_device(monkeypatch):
    _fake_smi(monkeypatch, "0, 0\n0, 0\n")
    assert hardware._nvidia_vram() is None


def test_nvidia_vram_smi_failure_returns_none(monkeypatch):
    _fake_smi(monkeypatch, "", returncode=1)
    assert hardware._nvidia_vram() is None
