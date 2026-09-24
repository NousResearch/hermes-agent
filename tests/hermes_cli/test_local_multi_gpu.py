"""Multi-GPU enumeration for the managed local runtime (issue #120573).

On multi-GPU boxes nvidia-smi prints one row per card, but the budget probe
and the /hardware facts only read the FIRST row — whatever enumerates as
GPU 0 (often the weaker card) prices every catalog row and names the pane.
Test doubles stand in for nvidia-smi (this box has no NVIDIA GPU).
"""
from __future__ import annotations

import pytest

import hermes_cli.local_runtime.hardware as hw
from hermes_cli.web_routers import local_models as lm

WEAK = "NVIDIA GeForce RTX 3050"
STRONG = "NVIDIA GeForce RTX 4090"

# Combined per-GPU query: index, name, total MiB, free MiB, used MiB, util %.
SMI_ALL = (
    f"0, {WEAK}, 4096, 512, 3584, 12\n"
    f"1, {STRONG}, 24564, 23000, 1564, 3\n"
)
# Legacy single-purpose queries (pre-fix code paths).
SMI_MEM = "4096, 512\n24564, 23000\n"
SMI_FACTS = f"{WEAK}, 12, 3584\n{STRONG}, 3, 1564\n"


class _Completed:
    def __init__(self, stdout: str):
        self.returncode = 0
        self.stdout = stdout


def _fake_smi(monkeypatch, rows: str = SMI_ALL):
    monkeypatch.setattr(hw, "_nvidia_smi_path", lambda: "nvidia-smi")
    monkeypatch.setattr(lm.hardware, "_nvidia_smi_path", lambda: "nvidia-smi")

    def fake_run(argv, **kw):
        query = next(a.split("=", 1)[1] for a in argv if a.startswith("--query-gpu="))
        if "index" in query:
            return _Completed(rows)
        if "memory.total" in query:
            return _Completed(SMI_MEM)
        return _Completed(SMI_FACTS)

    monkeypatch.setattr(hw.subprocess, "run", fake_run)


def _discrete_machine(monkeypatch, rows):
    """nvidia-smi double + the driver API saying 'discrete', RAM pinned."""
    _fake_smi(monkeypatch, rows)
    monkeypatch.setattr(hw, "_device_pool_view", lambda: (1 << 40, False))
    monkeypatch.setattr(hw, "_ram_bytes", lambda: (32 << 30, 16 << 30))


def test_lists_every_gpu(monkeypatch):
    """All installed dedicated GPUs are enumerated, not just GPU 0."""
    _fake_smi(monkeypatch)
    gpus = hw.list_nvidia_gpus()
    assert [g.name for g in gpus] == [WEAK, STRONG]
    assert [g.total_mib for g in gpus] == [4096, 24564]
    assert [g.free_mib for g in gpus] == [512, 23000]


WEAK_MIB, STRONG_MIB = 8192, 24564


def test_budget_reserves_per_card_not_once_on_the_sum(monkeypatch):
    """The reserve is a per-card cost, so admission can't hand out bytes no card yields.

    Reserving once after collapsing the cards (a 24 + 8 GiB box) advertises 29.12 GiB
    while per-card reserves leave 27.84 GiB — a footprint in that band used to be admitted
    with no realizable split behind it.
    """
    rows = (f"0, {WEAK}, {WEAK_MIB}, 512, 7680, 40\n"
            f"1, {STRONG}, {STRONG_MIB}, {STRONG_MIB}, 0, 1\n")
    _discrete_machine(monkeypatch, rows)

    def card(mib, free_mib, *, planning):
        total = mib << 20
        return max(0, (total if planning else free_mib << 20)
                   - max(hw._MARGIN_FLOOR, int(total * hw._MARGIN_FRACTION)))

    def collapsed():
        total = (WEAK_MIB + STRONG_MIB) << 20
        return total - max(hw._MARGIN_FLOOR, int(total * hw._MARGIN_FRACTION))

    planning = hw.probe_budget(planning=True)
    assert planning.uma is False
    assert planning.usable_vram_bytes == (card(WEAK_MIB, 512, planning=True)
                                          + card(STRONG_MIB, STRONG_MIB, planning=True))
    assert planning.usable_vram_bytes < collapsed()      # the false-admission band is gone
    assert planning.total_device_bytes == (WEAK_MIB + STRONG_MIB) << 20

    live = hw.probe_budget(planning=False)
    # The nearly-full card reserves down to nothing rather than dragging the sum below zero.
    assert card(WEAK_MIB, 512, planning=False) == 0
    assert live.usable_vram_bytes == (card(WEAK_MIB, 512, planning=False)
                                      + card(STRONG_MIB, STRONG_MIB, planning=False))


def test_hardware_facts_name_biggest_gpu_and_list_all(monkeypatch):
    """The pane names the strongest card and exposes the full list."""
    _fake_smi(monkeypatch)
    facts = lm._nvidia_smi_facts()
    assert facts["gpu_name"] == STRONG
    assert [g["name"] for g in facts["gpus"]] == [WEAK, STRONG]


def test_hardware_route_exposes_gpu_list(tmp_path, monkeypatch):
    """The /hardware payload carries the full list, biggest card first-named."""
    from fastapi.testclient import TestClient

    _fake_smi(monkeypatch)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli import web_server

    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    data = client.get("/api/local-models/hardware").json()
    assert [g["name"] for g in data["gpus"]] == [WEAK, STRONG]
    assert data["gpu_name"] == STRONG
