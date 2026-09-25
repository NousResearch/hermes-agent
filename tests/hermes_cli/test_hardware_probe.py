"""_nvidia_vram() must total every GPU row the CUDA runtime can see.

Tensor-split engines (llama.cpp) address the summed VRAM of all VISIBLE cards, so
the probe totals every admitted row — reading only GPU 0 budgets a 2x24GiB rig at
one card. Admission follows CUDA_VISIBLE_DEVICES exactly (the managed llama-server
inherits it via server_child_env) because NVML ignores the mask: an unfiltered SMI
query lists every card, and summing masked-away rows would budget VRAM the child
cannot allocate. A card reporting a per-field "N/A" must skip only its own row,
never discard the healthy rows already totaled."""

from __future__ import annotations

from types import SimpleNamespace

import hermes_cli.local_runtime.hardware as hardware

# nvidia-smi --query-gpu=index,uuid,memory.total,memory.free --format=csv,noheader,nounits
_TWO_CARDS = "0, GPU-aa-00, 24576, 24000\n1, GPU-aa-01, 24576, 23800\n"
_THREE_CARDS = ("0, GPU-aa-00, 24576, 24000\n"
                "1, GPU-aa-01, 24576, 23800\n"
                "2, GPU-aa-02, 8192, 8000\n")


def _fake_smi(monkeypatch, stdout: str, returncode: int = 0, mask: str | None = ...):
    monkeypatch.setattr(hardware, "_nvidia_smi_path", lambda: "/fake/nvidia-smi")
    monkeypatch.setattr(
        hardware.subprocess, "run",
        lambda *a, **k: SimpleNamespace(returncode=returncode, stdout=stdout))
    # ... means "unset": the mask variable itself is absent from the environment.
    if mask is ...:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)


def test_nvidia_vram_unset_mask_totals_all_gpu_rows(monkeypatch):
    _fake_smi(monkeypatch, _TWO_CARDS)
    total, free = hardware._nvidia_vram()
    assert total == 49152 << 20
    assert free == 47800 << 20


def test_nvidia_vram_single_gpu_is_unchanged(monkeypatch):
    _fake_smi(monkeypatch, "0, GPU-aa-00, 24576, 24000\n")
    total, free = hardware._nvidia_vram()
    assert total == 24576 << 20
    assert free == 24000 << 20


def test_nvidia_vram_single_index_mask_admits_only_that_row(monkeypatch):
    # 2x24GiB host launched with CUDA_VISIBLE_DEVICES=0: the child can allocate
    # GPU 0 only, so the full-rig 48GiB sum here would pick models that fail at load.
    _fake_smi(monkeypatch, _TWO_CARDS, mask="0")
    total, free = hardware._nvidia_vram()
    assert total == 24576 << 20
    assert free == 24000 << 20


def test_nvidia_vram_reordered_multi_index_mask_sums_only_listed_rows(monkeypatch):
    # Mask order is the runtime's device order, not an admission order: rows 0 and 2
    # are admitted whichever way the mask lists them, row 1 never is.
    _fake_smi(monkeypatch, _THREE_CARDS, mask="2,0")
    total, free = hardware._nvidia_vram()
    assert total == (24576 + 8192) << 20
    assert free == (24000 + 8000) << 20


def test_nvidia_vram_uuid_mask_admits_matching_rows(monkeypatch):
    # CUDA accepts abbreviated GPU-UUIDs; a truncated prefix must still match its
    # row (case-insensitively — SMI and shell conventions differ on hex case).
    _fake_smi(monkeypatch, _TWO_CARDS, mask="GPU-AA-01")
    total, free = hardware._nvidia_vram()
    assert total == 24576 << 20
    assert free == 23800 << 20


def test_nvidia_vram_empty_mask_hides_every_device(monkeypatch):
    # CUDA semantics: an empty value hides all devices from the runtime — the
    # probe must report no NVIDIA budget, not the full-rig sum.
    _fake_smi(monkeypatch, _TWO_CARDS, mask="")
    assert hardware._nvidia_vram() is None


def test_nvidia_vram_unmappable_mask_fails_closed(monkeypatch):
    # MIG instance UUIDs name partitions no SMI row identifies (and malformed
    # tokens likewise): the visible set is unknown, so admit nothing rather than
    # risk a budget beyond what the masked runtime can allocate.
    _fake_smi(monkeypatch, _TWO_CARDS, mask="MIG-GPU-aa-01/0")
    assert hardware._nvidia_vram() is None


def test_nvidia_vram_mask_referencing_absent_row_sums_what_exists(monkeypatch):
    # A stale mask (e.g. index 7 on a 2-card host) admits no row; the probe
    # correctly finds no budget rather than falling back to the full rig.
    _fake_smi(monkeypatch, _TWO_CARDS, mask="7")
    assert hardware._nvidia_vram() is None


def test_nvidia_vram_keeps_good_rows_when_one_card_reports_na(monkeypatch):
    # smi emits "N/A" per field, so a failing/driver-mismatched card produces a
    # four-field row — that card is skipped, the healthy cards keep their budget.
    _fake_smi(monkeypatch, "0, GPU-aa-00, 24576, 24000\n1, GPU-aa-01, N/A, N/A\n2, GPU-aa-02, 24576, 23800\n")
    total, free = hardware._nvidia_vram()
    assert total == 49152 << 20
    assert free == 47800 << 20


def test_nvidia_vram_skips_malformed_rows(monkeypatch):
    # Covers short rows and the quoted-CSV shape in addition to per-field N/A above.
    _fake_smi(monkeypatch, "24576, 24000\n[N/A]\n\"24576\", \"24000\"\n0, GPU-aa-00, 24576, 23800\n")
    total, free = hardware._nvidia_vram()
    assert total == 24576 << 20
    assert free == 23800 << 20


def test_nvidia_vram_partial_row_contributes_nothing(monkeypatch):
    # smi reports "N/A" per field, not per row: "24576, N/A" must not bump
    # total_mib before its free parse fails (and "N/A, 24000" the reverse) —
    # a row commits to both aggregates or to neither, keeping the budget
    # internally consistent.
    _fake_smi(monkeypatch, "0, GPU-aa-00, 24576, N/A\n1, GPU-aa-01, N/A, 24000\n2, GPU-aa-02, 24576, 24000\n")
    total, free = hardware._nvidia_vram()
    assert total == 24576 << 20
    assert free == 24000 << 20


def test_nvidia_vram_all_zero_reports_no_device(monkeypatch):
    _fake_smi(monkeypatch, "0, GPU-aa-00, 0, 0\n1, GPU-aa-01, 0, 0\n")
    assert hardware._nvidia_vram() is None


def test_nvidia_vram_smi_failure_returns_none(monkeypatch):
    _fake_smi(monkeypatch, "", returncode=1)
    assert hardware._nvidia_vram() is None
