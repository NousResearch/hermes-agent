"""Verify N1X recommendations use PCI identity rather than driver display names.

Simulated nvidia-smi output feeds the production memory probe, HardwareBudget and
catalog recommender. N1X IDs must still qualify after a name change; conflicting
IDs must not qualify through a matching name. Unavailable IDs retain the existing
name fallback without changing memory budgets or recipe eligibility.

Only the hardware responses are simulated. These are functional regression tests,
not physical-device validation or performance benchmarks.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from hermes_cli.local_runtime import catalog, hardware


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("pci_id, name, matches", [
    ("0x2E0010DE", "Renamed, GPU", True),
    ("0x2e0310de", "", True),
    ("0x2E0610DE", "Renamed GPU", True),
    ("0x2E1310DE", "Renamed GPU", True),
    ("0x2E2A10DE", "Renamed GPU", True),
    ("0x2E3F10DE", "Renamed GPU", True),
    ("0x2DFF10DE", "NVIDIA RTX Spark N1X", False),
    ("0x2E4010DE", "NVIDIA RTX Spark N1X", False),
    ("0x2E8610DE", "NVIDIA RTX Spark N1X", False),
    ("0x2E031002", "NVIDIA RTX Spark N1X", False),
    ("0x00000000", "NVIDIA RTX Spark N1X", False),
    ("-0x2E0310DE", "NVIDIA RTX Spark N1X", False),
    ("0x12E0310DE", "NVIDIA RTX Spark N1X", False),
    ("N/A", "NVIDIA RTX Spark N1X (updated description)", True),
    ("[Not Supported]", "NVIDIA RTX Spark N1X", True),
    ("malformed", "Renamed GPU", False),
    ("0x2E0310DE00", "Renamed GPU", False),
])
def test_pci_identity_controls_recommendations_without_changing_memory(
        monkeypatch, pci_id, name, matches):
    calls = []
    monkeypatch.setattr(hardware, "_nvidia_smi_path", lambda: "nvidia-smi")
    # Shared cached query: reset so each parametrized case makes its own spawn.
    monkeypatch.setattr(hardware, "_gpu_query_cache", None)
    monkeypatch.setattr(hardware, "_ram_bytes", lambda: (64 << 30, 22 << 30))
    monkeypatch.setattr(hardware, "_device_pool_view", lambda: (48 << 30, True))

    def run(argv, **kwargs):
        calls.append(argv)
        assert argv[1] == "--query-gpu=memory.total,memory.free,name,pci.device_id,memory.used,utilization.gpu"
        # A second adapter must not supply identity for the first adapter's budget.
        output = (f'32704, 31423, "{name}", {pci_id}, 2048, 7\n'
                  '32704, 31423, NVIDIA RTX Spark N1X, 0x2E0310DE, 2048, 7\n')
        return SimpleNamespace(returncode=0, stdout=output)

    monkeypatch.setattr(hardware.subprocess, "run", run)
    budget = hardware.probe_budget(planning=True)
    reference = replace(budget, gpu_name="NVIDIA RTX Spark N1X", gpu_pci_id=None)
    generic = replace(budget, gpu_name="", gpu_pci_id=None)
    expected = reference if matches else generic
    assert catalog.recommended_entry(budget) == catalog.recommended_entry(expected)
    entry = next(e for e in catalog.CATALOG if e.id == "qwen3.8-27b")
    variant = entry.variants[0]
    assert catalog.predicted_decode_tok_s(entry, variant, reference) != (
        catalog.predicted_decode_tok_s(entry, variant, generic))
    assert catalog.predicted_decode_tok_s(entry, variant, budget) == (
        catalog.predicted_decode_tok_s(entry, variant, expected))
    # Identity never overrides platform/recipe/placement eligibility.
    for backend in ("cpu", "vulkan"):
        assert catalog.predicted_decode_tok_s(entry, variant, budget, backend=backend) == (
            catalog.predicted_decode_tok_s(entry, variant, generic, backend=backend))
    assert catalog.predicted_decode_tok_s(entry, variant, budget, spilled=True) == (
        catalog.predicted_decode_tok_s(entry, variant, generic, spilled=True))
    assert budget.total_device_bytes == 48 << 30
    assert budget.usable_vram_bytes == int((48 << 30) * .8)
    assert budget.gpu_name == name
    assert len(calls) == 1


@pytest.mark.parametrize("integrated", [False, True])
def test_unavailable_pci_id_preserves_memory_and_name_fallback(monkeypatch, integrated):
    calls = []
    monkeypatch.setattr(hardware, "_nvidia_smi_path", lambda: "nvidia-smi")
    monkeypatch.setattr(hardware, "_gpu_query_cache", None)
    monkeypatch.setattr(hardware, "_ram_bytes", lambda: (64 << 30, 22 << 30))
    monkeypatch.setattr(hardware, "_device_pool_view", lambda: (48 << 30, integrated))
    name = "NVIDIA RTX Spark N1X"

    def run(argv, **kwargs):
        calls.append(argv)
        assert argv[1] == "--query-gpu=memory.total,memory.free,name,pci.device_id,memory.used,utilization.gpu"
        return SimpleNamespace(returncode=0, stdout=f"32704, 31423, {name}, N/A, 2048, 7\n")

    monkeypatch.setattr(hardware.subprocess, "run", run)
    budget = hardware.probe_budget(planning=True)
    assert budget.gpu_name == name
    assert budget.gpu_pci_id is None
    assert budget.uma == integrated
    total = 48 << 30 if integrated else 32704 << 20
    usable = int(total * .8) if integrated else total - max(2 << 30, int(total * .09))
    assert budget.total_device_bytes == total
    assert budget.usable_vram_bytes == usable
    assert len(calls) == 1
