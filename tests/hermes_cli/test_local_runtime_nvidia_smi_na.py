"""One ``[N/A]`` field from nvidia-smi must not void the whole GPU query.

Windows WDDM consumer drivers answer ``[N/A]`` for ``utilization.gpu`` (some
also for ``memory.used``). The consolidated query introduced by #127209
parsed the CSV row with ``int()`` on every field inside one suppression block,
so a single unreadable metric made ``_cached_nvidia_gpu_query()`` return
``None``: ``bootstrap._detect_gpu_vendor()`` then returned ``None`` and
``select_backend(None)`` picked the CPU llama.cpp build, while the budget
probe lost its VRAM — and the 4 s cache served that ``None`` to every reader.
Found in the review of #127209 (thanks @kshitijk4poor).

The stub here is a real executable resolved through the normal PATH ladder,
answering each ``--query-gpu`` field in the order requested and printing
``[N/A]`` for the fields named in ``$NA_FIELDS`` — the same repro as the
review. POSIX-gated because the stub is a ``/bin/sh`` script; the parsing
under test is host-independent.
"""

from __future__ import annotations

import pytest


# nvidia-smi's own per-field "no readout" answer, e.g. WDDM utilization.
_NA = "[N/A]"

_STUB = """\
#!/bin/sh
# Regression stub: answers each --query-gpu field in the order requested.
# Fields named in $NA_FIELDS answer "[N/A]" — per-field, like real drivers.
q=
for arg in "$@"; do
  case $arg in
    --query-gpu=*) q=${arg#--query-gpu=} ;;
  esac
done
out=
oldifs=$IFS
IFS=,
for field in $q; do
  case $field in
    memory.total) val=8192 ;;
    memory.free) val=4096 ;;
    name) val="NVIDIA GeForce RTX 4060 Ti" ;;
    pci.device_id) val=0x250410DE ;;
    memory.used) val=4096 ;;
    utilization.gpu) val=12 ;;
    *) val=0 ;;
  esac
  case ",$NA_FIELDS," in
    *",$field,"*) val="[N/A]" ;;
  esac
  out="$out, $val"
done
IFS=$oldifs
printf '%s\\n' "${out#, }"
"""

# Column order the query requests (hardware._cached_nvidia_gpu_query).
_NUMERIC_FIELDS = ("memory.total", "memory.free", "pci.device_id", "memory.used", "utilization.gpu")
_FIELD_TO_KEY = {
    "memory.total": "total_bytes",
    "memory.free": "free_bytes",
    "pci.device_id": "gpu_pci_id",
    "memory.used": "used_bytes",
    "utilization.gpu": "gpu_util_percent",
}
_HEALTHY_QUERY = {
    "gpu_name": "NVIDIA GeForce RTX 4060 Ti",
    "total_bytes": 8192 << 20,
    "free_bytes": 4096 << 20,
    "used_bytes": 4096 << 20,
    "gpu_util_percent": 12,
    "gpu_pci_id": 0x250410DE,
}


def _plant_smi_stub(monkeypatch, tmp_path, na_fields: str) -> None:
    """A real nvidia-smi executable on a stripped PATH; $NA_FIELDS answer [N/A]."""
    from hermes_cli.local_runtime import hardware

    bin_dir = tmp_path / "stub-bin"
    bin_dir.mkdir()
    stub = bin_dir / "nvidia-smi"
    stub.write_text(_STUB, encoding="utf-8")
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setenv("NA_FIELDS", na_fields)
    # Resolve through the real ladder (PATH first) and start each test cold,
    # so a stale cache from another test can't mask the spawn.
    monkeypatch.setattr(hardware, "_smi_path_cache", None)
    monkeypatch.setattr(hardware, "_gpu_query_cache", None)


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("na_field", _NUMERIC_FIELDS)
def test_one_na_field_does_not_void_the_query(monkeypatch, tmp_path, na_field):
    """Each field parses on its own: the [N/A] one reads None, the rest survive."""
    from hermes_cli.local_runtime import bootstrap, hardware

    _plant_smi_stub(monkeypatch, tmp_path, na_field)

    query = hardware._cached_nvidia_gpu_query()
    assert query is not None, "one unreadable metric must not kill the whole read"
    assert query[_FIELD_TO_KEY[na_field]] is None

    expected = dict(_HEALTHY_QUERY)
    expected.pop(_FIELD_TO_KEY[na_field])
    assert {k: query[k] for k in expected} == expected

    # Backend selection keeps CUDA: vendor detection only needs the name.
    vendor = bootstrap._detect_gpu_vendor()
    assert vendor is not None and vendor.startswith("nvidia")

    # The budget probe keeps VRAM unless total/free is the unreadable field.
    vram = hardware._nvidia_vram()
    if na_field in ("memory.total", "memory.free"):
        assert vram is None
    else:
        assert vram is not None
        assert vram[:2] == (8192 << 20, 4096 << 20)


@pytest.mark.platforms("posix")
@pytest.mark.parametrize(
    "na_fields",
    ["utilization.gpu", "memory.used", "utilization.gpu,memory.used"],
    ids=["util-na", "used-na", "wddm-both-na"],
)
def test_wddm_na_readouts_keep_cuda_and_vram(monkeypatch, tmp_path, na_fields):
    """The WDDM shape from the review: util (and/or used) [N/A] must not demote
    backend selection to the CPU build or void the VRAM budget."""
    from hermes_cli.local_runtime import bootstrap, hardware

    _plant_smi_stub(monkeypatch, tmp_path, na_fields)

    vendor = bootstrap._detect_gpu_vendor()
    assert vendor is not None and vendor.startswith("nvidia")
    assert hardware._nvidia_vram() is not None

    query = hardware._cached_nvidia_gpu_query()  # TTL hit on the same read
    assert query is not None
    for field in ("utilization.gpu", "memory.used"):
        if field in na_fields.split(","):
            assert query[_FIELD_TO_KEY[field]] is None


@pytest.mark.platforms("posix")
def test_na_name_keeps_vram_but_defers_vendor(monkeypatch, tmp_path):
    """An [N/A] name can't confirm the vendor (select_backend's fallback ladder
    decides), but it must not erase the memory readouts the budget needs."""
    from hermes_cli.local_runtime import bootstrap, hardware

    _plant_smi_stub(monkeypatch, tmp_path, "name")

    assert bootstrap._detect_gpu_vendor() is None
    vram = hardware._nvidia_vram()
    assert vram is not None
    assert vram[:3] == (8192 << 20, 4096 << 20, "")  # name unreadable, VRAM intact
