"""Residency admission is priced against the card, not a constant count.

Residency was bounded by ``models_max`` alone, so a second model was admitted against an
already-full card; on Windows/WDDM the over-commit is paged instead of refused and that child
decodes at a third of its speed for the rest of its life. The cap handed to the router is now
derived from the hardware budget, with the configured value as a ceiling. The budgets and model
sizes here are injected — the logic must hold on a machine with no GPU at all.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.local_runtime import presets
from hermes_cli.local_runtime.estimator import HardwareBudget, ModelProfile

GIB = 1 << 30

# The card from the report: RTX 3070 Laptop, 8 GiB, desktop co-residents reserved away. A 6 GiB
# model plus runtime overhead fits once, never twice.
TIGHT = HardwareBudget(usable_vram_bytes=6 * GIB, total_device_bytes=8 * GIB,
                       ram_available_bytes=8 * GIB)
# A 48 GiB card: four 6 GiB models fit with room to spare.
ROOMY = HardwareBudget(usable_vram_bytes=48 * GIB, total_device_bytes=48 * GIB,
                       ram_available_bytes=64 * GIB)


def _staged(tmp_path, monkeypatch, weights: dict[str, int]):
    """A models dir of touched GGUFs whose parsed weights are the given bytes (keyed by stem)."""
    mdir = tmp_path / "models"
    mdir.mkdir(parents=True, exist_ok=True)
    for stem in weights:
        (mdir / f"{stem}.gguf").touch()
    monkeypatch.setattr(presets, "read_gguf_header",
                        lambda p: SimpleNamespace(path=p, sampling_defaults={}))
    monkeypatch.setattr(presets, "profile_from_gguf", lambda h: ModelProfile(
        name=Path(h.path).stem, weights_bytes=weights[Path(h.path).stem],
        embd_table_bytes=0, n_ctx_train=65536, layers=[]))
    return mdir


NO_DEVICE = HardwareBudget(usable_vram_bytes=0, total_device_bytes=0, ram_available_bytes=8 * GIB)


@pytest.mark.parametrize("weights,budget,configured,expected", [
    # The report's exact shape: 6 GiB and 3.9 GiB staged on an 8 GiB card. Admitting both pages
    # the second one; the cap must say one so llama.cpp evicts before the load.
    ({"nine-b": 6 * GIB, "gemma-e4b": int(3.9 * GIB)}, TIGHT, 4, 1),
    # No regression for big cards: the budget raises nothing, and 4 residents fit.
    ({"a": 6 * GIB, "b": 6 * GIB}, ROOMY, 4, 4),
    # ``models_max`` stays a ceiling — the user's knob still means what it said.
    ({"a": 1 * GIB}, ROOMY, 1, 1),
    # No usable device memory (no probe) must leave the count exactly as configured.
    ({"a": 6 * GIB}, NO_DEVICE, 4, 4),
    # A model the physics check refuses outright never loads, so it must not force evictions
    # of the models that do (200 GiB against 48 + 64 GiB is refused, not merely spilled).
    ({"tiny": 2 * GIB, "huge": 200 * GIB}, ROOMY, 4, 4),
])
def test_residency_cap_is_priced_against_the_card(tmp_path, monkeypatch, weights, budget, configured, expected):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    mdir = _staged(tmp_path, monkeypatch, weights)
    assert presets.admitted_residency_count(mdir, budget, configured) == expected


def _boom(**_kw):
    raise OSError("nvidia-smi vanished mid-session")


@pytest.mark.parametrize("configured,probe,expected", [
    (4, lambda **kw: TIGHT, 1),
    (1, lambda **kw: TIGHT, 1),
    # The cap is policy, not a prerequisite: a probe failure must fall back to the config.
    (4, _boom, 4),
])
def test_boot_hands_the_router_the_derived_cap(tmp_path, monkeypatch, configured, probe, expected):
    """The wiring, not just the math: what ``ensure_local_runtime`` passes as ``--models-max``."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    import hermes_cli.local_runtime.bootstrap as bs
    from hermes_cli.local_runtime import binaries, hardware, supervisor
    from hermes_cli.local_runtime import endpoint

    mdir = _staged(tmp_path, monkeypatch, {"nine-b": 6 * GIB})
    monkeypatch.setattr(bs, "models_dir", lambda: mdir)
    monkeypatch.setattr(hardware, "probe_budget", probe)
    monkeypatch.setattr(bs, "_SUPERVISOR", None)
    monkeypatch.setattr(bs, "_generate_presets", lambda mdir, preset_path: None)
    monkeypatch.setattr(bs, "_detect_gpu_vendor", lambda: None)
    monkeypatch.setattr(binaries, "installed_tags", lambda: ["build-1"])
    monkeypatch.setattr(binaries, "default_tag", lambda: "build-1")
    monkeypatch.setattr(binaries, "select_backend", lambda vendor: "cpu")
    monkeypatch.setattr(binaries, "ensure_runtime_installed",
                        lambda tag, backend: tmp_path / "install")
    monkeypatch.setattr(endpoint, "_state_endpoint", lambda: None)

    captured: dict = {}

    class FakeSupervisor:
        def __init__(self, install_dir, models_dir, **kwargs):
            captured.update(kwargs)
            self.proc = None
            self.base_url = "http://127.0.0.1:1/v1"

        def start(self) -> None:
            return None

    monkeypatch.setattr(supervisor, "LlamaServerSupervisor", FakeSupervisor)
    config = {"local_runtime": {"enabled": True, "models_max": configured}}
    assert bs.ensure_local_runtime(config) is not None
    assert captured["models_max"] == expected
