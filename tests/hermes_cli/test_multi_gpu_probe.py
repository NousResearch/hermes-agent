"""Multi-GPU nvidia-smi enumeration + llama.cpp tensor-split (issue #107960).

Every probe here mocks subprocess / `_nvidia_smi_path`. Never call a real GPU.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import hermes_cli.local_runtime.hardware as hw

GIB = 1 << 30


class _Smi:
    def __init__(self, stdout: str, returncode: int = 0):
        self.stdout = stdout
        self.stderr = ""
        self.returncode = returncode


def _margin(total_bytes: int) -> int:
    return max(hw._MARGIN_FLOOR, int(total_bytes * hw._MARGIN_FRACTION))


def _install_smi(monkeypatch, memory_csv: str, facts_csv: str = ""):
    """Route both hardware and local-models nvidia-smi calls through mocked CSV."""
    monkeypatch.setattr(hw, "_smi_path_cache", None)
    monkeypatch.setattr(hw, "_pool_probe_cache", None)
    monkeypatch.setattr(hw, "_nvidia_smi_path", lambda: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(hw, "_device_pool_view", lambda: None)
    monkeypatch.setattr(hw, "_ram_bytes", lambda: (64 * GIB, 32 * GIB))

    def fake_run(argv, **kwargs):
        joined = " ".join(str(a) for a in argv)
        if "memory.total,memory.free" in joined:
            return _Smi(memory_csv)
        if "name,utilization.gpu,memory.used" in joined:
            return _Smi(facts_csv)
        return _Smi("", 1)

    monkeypatch.setattr(hw.subprocess, "run", fake_run)
    return fake_run


def _two_16gib_csv() -> str:
    # 16384 MiB == 16 GiB
    return "16384, 15000\n16384, 14000\n"


# ── 1) Detection ─────────────────────────────────────────────


def test_two_16gib_cards_planning_sums_per_card_margin(monkeypatch):
    """Dual 16 GiB cards must price as ~32 GiB, with WDDM margin applied per card.

    MUST fail on current main: `_nvidia_vram` only reads splitlines()[0].
    """
    _install_smi(monkeypatch, _two_16gib_csv())
    b = hw.probe_budget(planning=True)
    card = 16384 << 20
    margin = _margin(card)
    assert b.uma is False
    assert b.total_device_bytes == 2 * card
    assert b.usable_vram_bytes == 2 * max(0, card - margin)
    # Per-card floor (2 GiB × 2), not one margin on the summed 32 GiB.
    single_on_sum = max(hw._MARGIN_FLOOR, int((2 * card) * hw._MARGIN_FRACTION))
    assert b.usable_vram_bytes == 2 * card - 2 * margin
    assert 2 * margin != single_on_sum


def test_two_cards_live_sums_per_card_free_minus_margin(monkeypatch):
    _install_smi(monkeypatch, _two_16gib_csv())
    b = hw.probe_budget(planning=False)
    card = 16384 << 20
    free0, free1 = 15000 << 20, 14000 << 20
    margin = _margin(card)
    assert b.uma is False
    assert b.usable_vram_bytes == max(0, free0 - margin) + max(0, free1 - margin)


# ── 2) Fail-open: one card ───────────────────────────────────


def test_single_card_matches_today_discrete_math(monkeypatch):
    _install_smi(monkeypatch, "16384, 15000\n")
    b = hw.probe_budget(planning=True)
    card = 16384 << 20
    margin = _margin(card)
    assert b.uma is False
    assert b.total_device_bytes == card
    assert b.usable_vram_bytes == card - margin
    assert getattr(b, "tensor_split", None) is None


# ── 3) Uneven tensor-split ───────────────────────────────────


def test_uneven_cards_ratio_aware_tensor_split_in_preset(tmp_path, monkeypatch):
    """24 GiB + 8 GiB must NOT emit 0.5,0.5; generate_presets writes the INI key."""
    # 24576 MiB + 8192 MiB
    _install_smi(monkeypatch, "24576, 24000\n8192, 8000\n")
    b = hw.probe_budget(planning=True)
    t24, t8 = 24576 << 20, 8192 << 20
    u24 = max(0, t24 - _margin(t24))
    u8 = max(0, t8 - _margin(t8))
    assert b.uma is False
    assert b.total_device_bytes == t24 + t8
    assert b.usable_vram_bytes == u24 + u8
    ratios = b.tensor_split
    assert ratios is not None and len(ratios) == 2
    assert abs(ratios[0] - u24 / (u24 + u8)) < 1e-9
    assert abs(ratios[1] - u8 / (u24 + u8)) < 1e-9
    assert abs(ratios[0] - 0.5) > 0.1
    assert abs(sum(ratios) - 1.0) < 1e-9

    import hermes_cli.local_runtime.presets as presets_mod
    from hermes_cli.local_runtime.estimator import LayerKind, ModelProfile

    mdir = tmp_path / "models"
    mdir.mkdir()
    (mdir / "tiny-dense.gguf").write_bytes(b"GGUF" + b"\x00" * 64)

    class _Header:
        sampling_defaults: dict = {}

    profile = ModelProfile(
        name="tiny-dense", weights_bytes=2 * GIB, embd_table_bytes=0,
        n_ctx_train=131072, layers=[(LayerKind.FULL, 512)] * 4)
    monkeypatch.setattr(presets_mod, "read_gguf_header", lambda p: _Header())
    monkeypatch.setattr(presets_mod, "profile_from_gguf", lambda h: profile)

    preset = tmp_path / "presets.ini"
    presets_mod.generate_presets(mdir, b, preset)
    ini = preset.read_text(encoding="utf-8")
    assert "tensor-split" in ini
    line = next(ln for ln in ini.splitlines() if ln.startswith("tensor-split"))
    raw = line.split("=", 1)[1].strip()
    parsed = [float(x) for x in raw.split(",")]
    assert len(parsed) == 2
    assert abs(parsed[0] - 0.5) > 0.1
    assert abs(parsed[0] - ratios[0]) < 1e-3
    assert abs(sum(parsed) - 1.0) < 1e-3


# ── 4) UMA fail-open ─────────────────────────────────────────


def test_uma_path_ignores_extra_smi_lines_no_tensor_split(tmp_path, monkeypatch):
    """Unified-pool classification must not absorb extra discrete smi rows."""
    uma_smi = 16320
    uma_pool = 46464 << 20
    _install_smi(monkeypatch, f"{uma_smi}, 14848\n16384, 15000\n")
    monkeypatch.setattr(hw, "_device_pool_view", lambda: (uma_pool, True))
    monkeypatch.setattr(hw, "_ram_bytes", lambda: (48 * GIB, 32 * GIB))
    b = hw.probe_budget(planning=True)
    assert b.uma is True
    assert b.total_device_bytes == uma_pool
    assert b.usable_vram_bytes == int(uma_pool * (1 - hw._UMA_HEADROOM_FRACTION))
    assert getattr(b, "tensor_split", None) is None

    import hermes_cli.local_runtime.presets as presets_mod
    from hermes_cli.local_runtime.estimator import LayerKind, ModelProfile

    mdir = tmp_path / "models"
    mdir.mkdir()
    (mdir / "tiny-dense.gguf").write_bytes(b"GGUF" + b"\x00" * 64)

    class _Header:
        sampling_defaults: dict = {}

    profile = ModelProfile(
        name="tiny-dense", weights_bytes=2 * GIB, embd_table_bytes=0,
        n_ctx_train=131072, layers=[(LayerKind.FULL, 512)] * 4)
    monkeypatch.setattr(presets_mod, "read_gguf_header", lambda p: _Header())
    monkeypatch.setattr(presets_mod, "profile_from_gguf", lambda h: profile)
    preset = tmp_path / "presets.ini"
    presets_mod.generate_presets(mdir, b, preset)
    assert "tensor-split" not in preset.read_text(encoding="utf-8")


# ── 5) API gpus[] ────────────────────────────────────────────


@pytest.fixture
def hardware_client(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli import web_server

    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    return client


def test_hardware_api_exposes_gpus_array(hardware_client, monkeypatch):
    facts = "NVIDIA GeForce RTX 5070 Ti, 10, 2048\nNVIDIA GeForce RTX 5060 Ti, 5, 1024\n"
    fake_run = _install_smi(monkeypatch, _two_16gib_csv(), facts)
    from hermes_cli.web_routers import local_models as lm

    monkeypatch.setattr(lm.subprocess, "run", fake_run)
    monkeypatch.setattr(lm.hardware, "_nvidia_smi_path", lambda: "/usr/bin/nvidia-smi")

    data = hardware_client.get("/api/local-models/hardware").json()
    assert data["gpu_name"] == "NVIDIA GeForce RTX 5070 Ti"
    assert data["gpu_util_percent"] == 10
    assert data["vram_used_bytes"] == (2048 + 1024) << 20
    gpus = data["gpus"]
    assert len(gpus) == 2
    assert gpus[0]["name"] == "NVIDIA GeForce RTX 5070 Ti"
    assert gpus[0]["gpu_util_percent"] == 10
    assert gpus[0]["vram_used_bytes"] == 2048 << 20
    assert gpus[1]["name"] == "NVIDIA GeForce RTX 5060 Ti"
    assert gpus[1]["gpu_util_percent"] == 5
    assert gpus[1]["vram_used_bytes"] == 1024 << 20
    assert gpus[0]["vram_total_bytes"] == 16384 << 20
    assert gpus[1]["vram_total_bytes"] == 16384 << 20


# ── parse fail-open helpers (small; lock the skip rules) ─────


def test_malformed_smi_lines_are_skipped(monkeypatch):
    _install_smi(monkeypatch, "16384, 15000\nnot-a-gpu\n8\n16384, 14000\n")
    b = hw.probe_budget(planning=True)
    card = 16384 << 20
    assert b.total_device_bytes == 2 * card
    assert b.usable_vram_bytes == 2 * max(0, card - _margin(card))
