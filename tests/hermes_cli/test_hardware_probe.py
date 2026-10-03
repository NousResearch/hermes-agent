"""_nvidia_vram() must total every GPU row the CUDA runtime can see.

Tensor-split engines (llama.cpp) address the summed VRAM of all VISIBLE cards, so
the shared nvidia-smi query totals every admitted row — reading only GPU 0 budgets
a 2x24GiB rig at one card. Admission follows CUDA_VISIBLE_DEVICES exactly (the
managed llama-server inherits it via server_child_env) because NVML ignores the
mask: an unfiltered SMI query lists every card, and summing masked-away rows would
budget VRAM the child cannot allocate. A card reporting a per-field "N/A" must skip
only its own row, never discard the healthy rows already totaled."""

from __future__ import annotations

from types import SimpleNamespace

import hermes_cli.local_runtime.hardware as hardware

# nvidia-smi --query-gpu=index,uuid,memory.total,memory.free,name,pci.device_id,
#             memory.used,utilization.gpu --format=csv,noheader,nounits
_ROW = "{i}, GPU-aa-0{i}, {total}, {free}, NVIDIA Test GPU, 0x10DE, {used}, {util}\n"
_TWO_CARDS = _ROW.format(i=0, total=24576, free=24000, used=576, util=5) + \
    _ROW.format(i=1, total=24576, free=23800, used=776, util=7)
_THREE_CARDS = _TWO_CARDS + _ROW.format(i=2, total=8192, free=8000, used=192, util=0)


def _fake_smi(monkeypatch, stdout: str, returncode: int = 0, mask: str | None = ...):
    monkeypatch.setattr(hardware, "_nvidia_smi_path", lambda: "/fake/nvidia-smi")
    monkeypatch.setattr(
        hardware.subprocess, "run",
        lambda *a, **k: SimpleNamespace(returncode=returncode, stdout=stdout))
    monkeypatch.setattr(hardware, "_gpu_query_cache", None)
    # ... means "unset": the mask variable itself is absent from the environment.
    if mask is ...:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)


def _vram(monkeypatch, stdout: str, **kwargs) -> tuple[int, int] | None:
    _fake_smi(monkeypatch, stdout, **kwargs)
    vram = hardware._nvidia_vram()
    return None if vram is None else (vram[0], vram[1])


def test_nvidia_vram_unset_mask_totals_all_gpu_rows(monkeypatch):
    total, free = _vram(monkeypatch, _TWO_CARDS)
    assert total == 49152 << 20
    assert free == 47800 << 20


def test_nvidia_vram_single_gpu_is_unchanged(monkeypatch):
    total, free = _vram(monkeypatch, _ROW.format(i=0, total=24576, free=24000, used=576, util=5))
    assert total == 24576 << 20
    assert free == 24000 << 20


def test_nvidia_vram_single_index_mask_admits_only_that_row(monkeypatch):
    # 2x24GiB host launched with CUDA_VISIBLE_DEVICES=0: the child can allocate
    # GPU 0 only, so the full-rig 48GiB sum here would pick models that fail at load.
    total, free = _vram(monkeypatch, _TWO_CARDS, mask="0")
    assert total == 24576 << 20
    assert free == 24000 << 20


def test_nvidia_vram_reordered_multi_index_mask_sums_only_listed_rows(monkeypatch):
    # Mask order is the runtime's device order, not an admission order: rows 0 and 2
    # are admitted whichever way the mask lists them, row 1 never is.
    total, free = _vram(monkeypatch, _THREE_CARDS, mask="2,0")
    assert total == (24576 + 8192) << 20
    assert free == (24000 + 8000) << 20


def test_nvidia_vram_uuid_mask_admits_matching_rows(monkeypatch):
    # CUDA accepts abbreviated GPU-UUIDs; a truncated prefix must still match its
    # row (case-insensitively — SMI and shell conventions differ on hex case).
    total, free = _vram(monkeypatch, _TWO_CARDS, mask="GPU-AA-01")
    assert total == 24576 << 20
    assert free == 23800 << 20


def test_nvidia_vram_empty_mask_hides_every_device(monkeypatch):
    # CUDA semantics: an empty value hides all devices from the runtime — the
    # probe must report no NVIDIA budget, not the full-rig sum.
    assert _vram(monkeypatch, _TWO_CARDS, mask="") is None


def test_nvidia_vram_unmappable_mask_fails_closed(monkeypatch):
    # MIG instance UUIDs name partitions no SMI row identifies (and malformed
    # tokens likewise): the visible set is unknown, so admit nothing rather than
    # risk a budget beyond what the masked runtime can allocate.
    assert _vram(monkeypatch, _TWO_CARDS, mask="MIG-GPU-aa-01/0") is None


def test_nvidia_vram_mask_referencing_absent_row_sums_what_exists(monkeypatch):
    # A stale mask (e.g. index 7 on a 2-card host) admits no row; the probe
    # correctly finds no budget rather than falling back to the full rig.
    assert _vram(monkeypatch, _TWO_CARDS, mask="7") is None


def test_nvidia_vram_keeps_good_rows_when_one_card_reports_na(monkeypatch):
    # smi emits "N/A" per field, so a failing/driver-mismatched card produces a
    # full-width row with dead memory fields — that card is skipped, the healthy
    # cards keep their budget.
    total, free = _vram(monkeypatch,
                        "0, GPU-aa-00, 24576, 24000, NVIDIA Test GPU, 0x10DE, 576, 5\n"
                        "1, GPU-aa-01, N/A, N/A, NVIDIA Test GPU, N/A, N/A, N/A\n"
                        "2, GPU-aa-02, 24576, 23800, NVIDIA Test GPU, 0x10DE, 776, 7\n")
    assert total == 49152 << 20
    assert free == 47800 << 20


def test_nvidia_vram_skips_malformed_rows(monkeypatch):
    # Covers short rows and the quoted-CSV shape in addition to per-field N/A above.
    total, free = _vram(monkeypatch,
                        "24576, 24000\n[N/A]\n\"24576\", \"24000\"\n"
                        "0, GPU-aa-00, 24576, 23800, NVIDIA Test GPU, 0x10DE, 776, 7\n")
    assert total == 24576 << 20
    assert free == 23800 << 20


def test_nvidia_vram_partial_row_contributes_nothing(monkeypatch):
    # smi reports "N/A" per field, not per row: "24576, N/A" must not bump
    # the total before its free parse fails (and "N/A, 24000" the reverse) —
    # a row commits to every aggregate or to none, keeping the budget
    # internally consistent.
    total, free = _vram(monkeypatch,
                        "0, GPU-aa-00, 24576, N/A, NVIDIA Test GPU, 0x10DE, 576, 5\n"
                        "1, GPU-aa-01, N/A, 24000, NVIDIA Test GPU, 0x10DE, 576, 5\n"
                        "2, GPU-aa-02, 24576, 24000, NVIDIA Test GPU, 0x10DE, 576, 5\n")
    assert total == 24576 << 20
    assert free == 24000 << 20


def test_nvidia_vram_all_zero_reports_no_device(monkeypatch):
    assert _vram(monkeypatch,
                 "0, GPU-aa-00, 0, 0, NVIDIA Test GPU, 0x10DE, 0, 0\n"
                 "1, GPU-aa-01, 0, 0, NVIDIA Test GPU, 0x10DE, 0, 0\n") is None


def test_nvidia_vram_smi_failure_returns_none(monkeypatch):
    assert _vram(monkeypatch, "", returncode=1) is None


def test_shared_query_name_util_follow_the_admitted_rows(monkeypatch):
    # The statusbar reads gpu_name/gpu_util_percent/used_bytes from the same
    # folded rows: identical cards collapse to "name xN", utilization is the
    # admitted-row mean, and a mask narrows the view the runtime actually sees.
    _fake_smi(monkeypatch, _TWO_CARDS)
    query = hardware._cached_nvidia_gpu_query()
    assert query["gpu_name"] == "NVIDIA Test GPU x2"
    assert query["gpu_util_percent"] == 6  # mean(5, 7)
    assert query["used_bytes"] == (576 + 776) << 20
    assert query["gpu_pci_id"] is None  # a PCI ID identifies one card, not a rig

    _fake_smi(monkeypatch, _TWO_CARDS, mask="1")
    query = hardware._cached_nvidia_gpu_query()
    assert query["gpu_name"] == "NVIDIA Test GPU"
    assert query["gpu_util_percent"] == 7
    assert query["gpu_pci_id"] == 0x10DE


def test_shared_query_distinct_model_names_are_both_kept(monkeypatch):
    stdout = ("0, GPU-aa-00, 24576, 24000, NVIDIA RTX 3090, 0x10DE, 576, 5\n"
              "1, GPU-aa-01, 8192, 8000, NVIDIA RTX 4060, 0x10DE, 192, 0\n")
    _fake_smi(monkeypatch, stdout)
    query = hardware._cached_nvidia_gpu_query()
    assert query["gpu_name"] == "NVIDIA RTX 3090 + NVIDIA RTX 4060"
    assert query["gpu_util_percent"] == 2  # mean(5, 0) rounds down
