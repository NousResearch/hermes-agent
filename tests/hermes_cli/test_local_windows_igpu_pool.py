"""An integrated GPU on Windows is budgeted from what Windows lets it allocate, never all of RAM.

Windows caps an integrated GPU's allocations at its GPU memory (Task Manager; dedicated plus
shared): 18 GB of 31.5 on an Intel Arc B390. Budgeting all of RAM there showed "31.5 GB GPU memory"
and recommended a 27B the GPU cannot hold. The cap holds only when a GPU engine runs the model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import hermes_cli.local_runtime.hardware as hw
from hermes_cli.local_runtime import binaries
from hermes_cli.local_runtime.binaries import Engine
from hermes_cli.local_runtime.catalog import CATALOG, select_variant
from hermes_cli.local_runtime.devices import GGML_DEVICE_IGPU
from hermes_platform.host import gpu_adapters
from hermes_platform.host.gpu_adapters import Adapter

GIB = 1 << 30
RAM = int(31.5 * GIB)
B390 = Adapter("Intel(R) Arc(TM) B390 GPU", 0x8086, 128 << 20, 18 * GIB - (128 << 20), software=False)
ADRENO = Adapter("Qualcomm(R) Adreno(TM) X1-85 GPU", 0x4D4F4351, 0, 15 * GIB, software=False)
SOFTWARE = Adapter("Microsoft Basic Render Driver", 0x1414, 0, 16 * GIB, software=True)
# ggml-vulkan sums every heap of an integrated GPU, so the engine reports more than Windows grants.
ENGINE_B390 = {"description": B390.description, "type": GGML_DEVICE_IGPU, "total": 34 * GIB, "free": 30 * GIB}


def _machine(monkeypatch, adapters, device, *, installed="vulkan", planned="vulkan"):
    """``installed`` is the engine on disk (None before setup); ``planned`` is what setup installs."""
    monkeypatch.setattr(hw, "_nvidia_vram", lambda: None)
    monkeypatch.setattr(hw, "_device_pool_view", lambda: None)
    monkeypatch.setattr(hw, "_ram_bytes", lambda: (RAM, 20 * GIB))
    monkeypatch.setattr(hw, "_accelerator_device", lambda **_: device)
    monkeypatch.setattr(gpu_adapters, "windows_gpu_adapters", lambda: tuple(adapters))
    engine = None if installed is None else Engine(installed, "b11370", Path("llama-server"))
    monkeypatch.setattr(hw, "_configured_engine", lambda: engine)
    monkeypatch.setattr(binaries, "resolve_backend", lambda *_, **__: planned)


@pytest.mark.parametrize(("device", "installed"), [(ENGINE_B390, "vulkan"), (None, None)],
                         ids=["engine-named", "before-the-engine"])
def test_an_integrated_gpu_gets_what_windows_lets_it_allocate(monkeypatch, device, installed):
    _machine(monkeypatch, [B390, SOFTWARE], device, installed=installed)

    capacity = hw.probe_budget(planning=True)
    live = hw.probe_budget(planning=False)

    assert capacity.uma and capacity.total_device_bytes == B390.memory_bytes
    assert live.usable_vram_bytes <= capacity.usable_vram_bytes
    qwen_27b = next(e for e in CATALOG if e.id == "qwen3.8-27b")
    assert select_variant(qwen_27b, capacity) is None


def test_the_engine_named_adapter_wins_among_several(monkeypatch):
    discrete = Adapter("Intel(R) Arc(TM) A380 Graphics", 0x8086, 6 * GIB, 15 * GIB, software=False)

    _machine(monkeypatch, [discrete, B390, SOFTWARE], ENGINE_B390)
    assert hw.probe_budget(planning=True).total_device_bytes == B390.memory_bytes

    _machine(monkeypatch, [discrete, B390, SOFTWARE], None)  # ambiguous without the engine's answer
    assert hw.probe_budget(planning=True).total_device_bytes == RAM


@pytest.mark.parametrize(("adapter", "installed", "planned"), [
    (B390, "cpu", "cpu"),     # local_runtime.backend: cpu
    (ADRENO, None, "cpu"),    # a Snapdragon: setup installs the CPU build
], ids=["cpu-engine-installed", "cpu-engine-planned"])
def test_a_cpu_engine_keeps_the_ram_budget(monkeypatch, adapter, installed, planned):
    _machine(monkeypatch, [adapter, SOFTWARE], None, installed=installed, planned=planned)
    assert hw.probe_budget(planning=True).total_device_bytes == RAM


@pytest.mark.parametrize("adapters", [
    [SOFTWARE],                                                                 # no GPU: CPU inference
    [Adapter("AMD Radeon(TM) 8060S Graphics", 0x1002, 96 * GIB, 16 * GIB, software=False), SOFTWARE],
    [],                                                                         # DXGI did not answer
], ids=["cpu-only", "allowance-above-ram", "no-dxgi"])
def test_the_pool_never_grows_past_ram(monkeypatch, adapters):
    _machine(monkeypatch, adapters, None)
    assert hw.probe_budget(planning=True).total_device_bytes == RAM
