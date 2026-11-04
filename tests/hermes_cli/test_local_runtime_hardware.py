"""Behavior contracts for hermes_cli.local_runtime.hardware VRAM accounting.

Regression for the multi-GPU budget bug: ``_nvidia_vram()`` read only the FIRST
nvidia-smi CSV row (``splitlines()[0]``), so a 2x3090 box (~48 GiB combined)
was budgeted as 24 GiB (card 0 only). The fit checker then judged a model that
actually fits as "too big" and silently spilled its FFN weights to CPU RAM,
cutting decode throughput dramatically (measured ~2x on this machine).

The contract: nvidia-smi emits one CSV row per visible GPU and llama.cpp may
place weights across the whole box, so the reported (total, free) VRAM is the
sum over every card, in bytes (MiB << 20, matching the module's convention).
"""

import types

import pytest

from hermes_cli.local_runtime import hardware


def _patched(monkeypatch, csv: str):
    """Make _nvidia_vram see a fixed nvidia-smi multi-row CSV output."""
    def _fake_run(cmd, *a, **k):
        assert any("memory.total" in c for c in cmd), "expected nvidia-smi VRAM query"
        return types.SimpleNamespace(returncode=0, stdout=csv)

    monkeypatch.setattr(hardware, "_nvidia_smi_path", lambda: "/fake/nvidia-smi")
    monkeypatch.setattr(hardware.subprocess, "run", _fake_run, raising=False)


def test_vram_sums_all_visible_gpus(monkeypatch):
    """Two 24 GiB cards must report the combined ~48 GiB total (bytes)."""
    _patched(monkeypatch, "24576,20000\n24576,18000\n")
    total_b, free_b = hardware._nvidia_vram()
    assert total_b == (49152) << 20          # 24576 + 24576 MiB, in bytes
    assert free_b == (38000) << 20           # 20000 + 18000 MiB, in bytes


def test_vram_single_gpu_unchanged(monkeypatch):
    """Single-card systems behave exactly as before."""
    _patched(monkeypatch, "24576,20000\n")
    total_b, free_b = hardware._nvidia_vram()
    assert total_b == (24576) << 20
    assert free_b == (20000) << 20


def test_vram_none_when_no_nvidia_smi(monkeypatch):
    """No nvidia-smi on PATH -> no VRAM signal (caller falls back)."""
    monkeypatch.setattr(hardware, "_nvidia_smi_path", lambda: None)
    assert hardware._nvidia_vram() is None
