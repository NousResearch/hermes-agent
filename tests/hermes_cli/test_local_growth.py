"""In-session growth contracts (growth.py + the presets override seam).

The live half of the window ladder: grow before compress, overrides
persist across boots, physics re-checked every boot, growth state dies
with the model."""

from __future__ import annotations

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def test_overrides_roundtrip_and_clear(hermes_home):
    from hermes_cli.local_runtime.growth import (
        clear_window_override,
        load_window_overrides,
        save_window_override,
    )

    assert load_window_overrides() == {}
    save_window_override("model-a", 98304)
    save_window_override("model-b", 262144)
    assert load_window_overrides() == {"model-a": 98304, "model-b": 262144}
    clear_window_override("model-a")
    assert load_window_overrides() == {"model-b": 262144}
    # Clearing a missing key is a no-op, not an error.
    clear_window_override("never-existed")


def test_corrupt_overrides_read_as_empty(hermes_home):
    from hermes_cli.local_runtime.growth import (
        load_window_overrides,
        window_overrides_path,
    )

    path = window_overrides_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert load_window_overrides() == {}


def test_growth_declines_foreign_endpoints(hermes_home):
    """Only the server THIS process supervises grows — a detected external
    server or another process's endpoint returns None untouched."""
    from hermes_cli.local_runtime.growth import maybe_grow_window

    grown = maybe_grow_window(
        "some-model", base_url="http://127.0.0.1:9999/v1",
        session_tokens=100_000, current_window=65536)
    assert grown is None


def _patch_managed_growth(monkeypatch, tmp_path, *, cap, next_window, budget=None):
    import hermes_cli.local_runtime.bootstrap as boot
    import hermes_cli.local_runtime.context_policy as policy
    import hermes_cli.local_runtime.growth as growth
    import hermes_cli.local_runtime.hardware as hardware

    class _Supervisor:
        def is_idle(self, model_id):
            return True

    gguf = tmp_path / "model-a.gguf"
    gguf.write_bytes(b"GGUF")
    profile = _tiny_profile("model-a")
    monkeypatch.setattr(boot, "get_supervisor", lambda: _Supervisor())
    monkeypatch.setattr(boot, "staged_models", lambda: [gguf])
    monkeypatch.setattr(boot, "refresh_local_runtime", lambda: True)
    monkeypatch.setattr(growth, "is_managed_endpoint", lambda base_url: True)
    monkeypatch.setattr("hermes_cli.local_runtime.gguf.read_gguf_header", lambda path: object())
    monkeypatch.setattr("hermes_cli.local_runtime.estimator.profile_from_gguf", lambda header: profile)
    monkeypatch.setattr(hardware, "probe_budget", lambda planning: budget or object())
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {
        "local_runtime": {"launch_overrides": {"model-a": {"ctx-size": cap}}},
    })
    if next_window is not None:
        monkeypatch.setattr(policy, "growth_decision", lambda *args, **kwargs: type("Decision", (), {
            "action": "grow", "next_window": next_window, "reason": "test growth",
        })())
    return growth


def test_growth_stops_at_configured_context_cap(hermes_home, tmp_path, monkeypatch):
    growth = _patch_managed_growth(monkeypatch, tmp_path, cap=65536, next_window=98304)
    writes = []
    refreshes = []
    monkeypatch.setattr(growth, "save_window_override", lambda *args: writes.append(args))
    monkeypatch.setattr("hermes_cli.local_runtime.bootstrap.refresh_local_runtime",
                        lambda: refreshes.append(True) or True)

    result = growth.maybe_grow_window(
        "model-a", base_url="http://127.0.0.1:8080/v1",
        session_tokens=60_000, current_window=65536)

    assert result is None
    assert writes == []
    assert refreshes == []


def test_growth_clamps_next_rung_to_configured_context_cap(hermes_home, tmp_path, monkeypatch):
    growth = _patch_managed_growth(monkeypatch, tmp_path, cap=81920, next_window=98304)
    writes = []
    monkeypatch.setattr(growth, "save_window_override", lambda *args: writes.append(args))

    result = growth.maybe_grow_window(
        "model-a", base_url="http://127.0.0.1:8080/v1",
        session_tokens=60_000, current_window=65536)

    assert result == 81920
    assert writes == [("model-a", 81920)]


def test_growth_prices_configured_cap_instead_of_larger_native_rung(
        hermes_home, tmp_path, monkeypatch):
    from hermes_cli.local_runtime.estimator import HardwareBudget, ctx_bytes

    cap = 81920
    profile = _tiny_profile("model-a")
    capped_bytes = profile.weights_bytes + ctx_bytes(profile, cap)
    native_rung_bytes = profile.weights_bytes + ctx_bytes(profile, 98304)
    assert capped_bytes < native_rung_bytes
    budget = HardwareBudget(
        usable_vram_bytes=capped_bytes,
        total_device_bytes=capped_bytes,
        ram_available_bytes=0,
    )
    growth = _patch_managed_growth(
        monkeypatch, tmp_path, cap=cap, next_window=None, budget=budget)
    writes = []
    refreshes = []
    monkeypatch.setattr(growth, "save_window_override", lambda *args: writes.append(args))
    monkeypatch.setattr("hermes_cli.local_runtime.bootstrap.refresh_local_runtime",
                        lambda: refreshes.append(True) or True)

    result = growth.maybe_grow_window(
        "model-a", base_url="http://127.0.0.1:8080/v1",
        session_tokens=60_000, current_window=65536)

    assert result == cap
    assert writes == [("model-a", cap)]
    assert refreshes == [True]


def test_occupancy_confirmed_skips_gate_one():
    """The agent's compression gate IS the occupancy signal: when it fired,
    growth must not re-derive its own edge and hold. Decision-table check
    with a synthetic profile."""
    from hermes_cli.local_runtime.context_policy import growth_decision
    from hermes_cli.local_runtime.estimator import (
        HardwareBudget,
        LayerKind,
        ModelProfile,
    )

    gib = 1 << 30
    profile = ModelProfile(
        name="m", weights_bytes=2 * gib, embd_table_bytes=0,
        n_ctx_train=262144,
        layers=[(LayerKind.FULL, 4096)] * 16 + [(LayerKind.RECURRENT, 0)] * 48)
    budget = HardwareBudget(usable_vram_bytes=26 * gib,
                            total_device_bytes=32 * gib,
                            ram_available_bytes=64 * gib)

    # Hermes' threshold (e.g. 80% of window) can sit BELOW the ladder's 85%
    # occupancy gate: 78K of a 96K window is 81%.
    kwargs = dict(current_window=98304, session_tokens=78_000,
                  measured_decode_tok_s=None, server_idle=True)
    ungated = growth_decision(profile, budget, **kwargs)
    assert ungated.action == "hold", "sanity: below the ladder's own gate"

    confirmed = growth_decision(profile, budget, occupancy_confirmed=True, **kwargs)
    assert confirmed.action == "grow"
    assert confirmed.next_window and confirmed.next_window > 98304


def _stage_fake_gguf(mdir, name):
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / f"{name}.gguf").write_bytes(b"GGUF" + b"\x00" * 64)


def _header_stub(sampling: dict | None = None):
    """A read_gguf_header stand-in for tests that monkeypatch the reader:
    just enough surface for preset generation (sampling ladder included)."""

    class _Stub:
        sampling_defaults = dict(sampling or {})

    return _Stub()


def _tiny_profile(model_id: str):
    from hermes_cli.local_runtime.estimator import LayerKind, ModelProfile

    gib = 1 << 30
    return ModelProfile(
        name=model_id, weights_bytes=2 * gib, embd_table_bytes=0,
        n_ctx_train=131072,
        layers=[(LayerKind.FULL, 512)] * 4)


def test_preset_generation_for_catalog_model_with_mmproj(hermes_home, tmp_path, monkeypatch):
    """generate_presets must survive a model that IS in the catalog and
    carries a vision projector — this executes the find_entry_for_model +
    mmproj overhead branch that synthetic test models skip. Regression:
    the branch once treated the (entry, variant) tuple as the entry and
    crashed every real boot into the stock-fit fallback."""
    import hermes_cli.local_runtime.presets as presets_mod

    from hermes_cli.local_runtime.catalog import CATALOG
    from hermes_cli.local_runtime.estimator import HardwareBudget

    # A real catalog id with an mmproj (the recommended row has one).
    entry = next(e for e in CATALOG if e.mmproj is not None)
    variant = entry.variants[-1]
    mdir = tmp_path / "models"
    _stage_fake_gguf(mdir, variant.model_id)

    monkeypatch.setattr(presets_mod, "read_gguf_header", lambda p: _header_stub())
    monkeypatch.setattr(presets_mod, "profile_from_gguf",
                        lambda h: _tiny_profile(variant.model_id))

    gib = 1 << 30
    budget = HardwareBudget(usable_vram_bytes=24 * gib,
                            total_device_bytes=24 * gib,
                            ram_available_bytes=64 * gib)
    entries = presets_mod.generate_presets(mdir, budget, tmp_path / "p.ini")
    assert len(entries) == 1
    assert entries[0].refusal is None
    assert entries[0].window > 0


def test_preset_restores_grown_window_capped_at_native(hermes_home, tmp_path, monkeypatch):
    """A persisted override lifts the preset window; an absurd override is
    capped at native. GGUF parsing is stubbed — the contract under test is
    the override plumbing, not the reader."""
    import hermes_cli.local_runtime.presets as presets_mod

    from hermes_cli.local_runtime.estimator import HardwareBudget
    from hermes_cli.local_runtime.growth import save_window_override

    mdir = tmp_path / "models"
    _stage_fake_gguf(mdir, "tiny-dense")
    monkeypatch.setattr(presets_mod, "read_gguf_header", lambda p: _header_stub())
    monkeypatch.setattr(presets_mod, "profile_from_gguf",
                        lambda h: _tiny_profile("tiny-dense"))

    gib = 1 << 30
    budget = HardwareBudget(usable_vram_bytes=24 * gib,
                            total_device_bytes=24 * gib,
                            ram_available_bytes=64 * gib)
    preset = tmp_path / "presets.ini"

    baseline = presets_mod.generate_presets(mdir, budget, preset)[0]
    assert baseline.window == 131072  # tiny model: native from the start

    # Override above native must cap at native, not exceed it.
    save_window_override("tiny-dense", 10_000_000)
    capped = presets_mod.generate_presets(mdir, budget, preset)[0]
    assert capped.window == 131072


def test_preset_ignores_override_below_launch_window(hermes_home, tmp_path, monkeypatch):
    """Overrides only ever RAISE the window (growth is monotone); a stale
    smaller override never shrinks a launch decision."""
    import hermes_cli.local_runtime.presets as presets_mod

    from hermes_cli.local_runtime.estimator import HardwareBudget
    from hermes_cli.local_runtime.growth import save_window_override

    mdir = tmp_path / "models"
    _stage_fake_gguf(mdir, "tiny-dense")
    monkeypatch.setattr(presets_mod, "read_gguf_header", lambda p: _header_stub())
    monkeypatch.setattr(presets_mod, "profile_from_gguf",
                        lambda h: _tiny_profile("tiny-dense"))
    save_window_override("tiny-dense", 65536)

    gib = 1 << 30
    budget = HardwareBudget(usable_vram_bytes=24 * gib,
                            total_device_bytes=24 * gib,
                            ram_available_bytes=64 * gib)
    entry = presets_mod.generate_presets(mdir, budget, tmp_path / "p.ini")[0]
    assert entry.window == 131072


def test_launch_overrides_survive_preset_regeneration(hermes_home, tmp_path, monkeypatch):
    """User launch limits are reapplied after every generated-policy rewrite."""
    import configparser

    import hermes_cli.local_runtime.presets as presets_mod
    from hermes_cli.local_runtime.estimator import HardwareBudget

    mdir = tmp_path / "models"
    _stage_fake_gguf(mdir, "tiny-dense")
    monkeypatch.setattr(presets_mod, "read_gguf_header", lambda p: _header_stub())
    monkeypatch.setattr(presets_mod, "profile_from_gguf",
                        lambda h: _tiny_profile("tiny-dense"))
    gib = 1 << 30
    budget = HardwareBudget(usable_vram_bytes=24 * gib,
                            total_device_bytes=24 * gib,
                            ram_available_bytes=64 * gib)
    overrides = {
        "tiny-dense": {"ctx-size": 65536, "batch-size": 512, "ubatch-size": 512},
    }
    preset = tmp_path / "presets.ini"

    for _ in range(2):
        entry = presets_mod.generate_presets(
            mdir, budget, preset, launch_overrides=overrides)[0]
        ini = configparser.ConfigParser()
        ini.read(preset)
        assert entry.window == 65536
        assert ini["tiny-dense"]["ctx-size"] == "65536"
        assert ini["tiny-dense"]["batch-size"] == "512"
        assert ini["tiny-dense"]["ubatch-size"] == "512"


def test_context_cap_below_floor_warns_and_uses_floor(caplog):
    from hermes_cli.local_runtime.context_policy import FLOOR
    from hermes_cli.local_runtime.presets import effective_ctx_cap

    configured = FLOOR // 2
    cap = effective_ctx_cap(
        {"model-a": {"ctx-size": configured}}, "model-a", native_window=FLOOR * 2)

    assert cap == FLOOR
    assert (
        f"local_runtime.launch_overrides.model-a.ctx-size={configured} "
        f"raised to minimum {FLOOR}"
    ) in caplog.text


def test_preset_restores_grown_window_midladder(hermes_home, tmp_path, monkeypatch):
    """The real growth shape: launch at a lower rung, override to a middle
    rung -> the preset window follows the override."""
    import hermes_cli.local_runtime.presets as presets_mod

    from hermes_cli.local_runtime.estimator import HardwareBudget, LayerKind, ModelProfile
    from hermes_cli.local_runtime.growth import save_window_override

    gib = 1 << 30
    # Expensive dense KV so the launch decision lands BELOW native on this
    # budget: 60 layers x 4 KiB/tok f16 -> q8 ~= 120 KiB/tok.
    profile = ModelProfile(
        name="big-dense", weights_bytes=20 * gib, embd_table_bytes=0,
        n_ctx_train=262144,
        layers=[(LayerKind.FULL, 4096)] * 60)
    mdir = tmp_path / "models"
    _stage_fake_gguf(mdir, "big-dense")
    monkeypatch.setattr(presets_mod, "read_gguf_header", lambda p: _header_stub())
    monkeypatch.setattr(presets_mod, "profile_from_gguf", lambda h: profile)

    budget = HardwareBudget(usable_vram_bytes=28 * gib,
                            total_device_bytes=32 * gib,
                            ram_available_bytes=128 * gib)
    baseline = presets_mod.generate_presets(mdir, budget, tmp_path / "a.ini")[0]
    assert baseline.window < 262144, "sanity: launch below native"

    grown = baseline.window * 2
    save_window_override("big-dense", grown)
    restored = presets_mod.generate_presets(mdir, budget, tmp_path / "b.ini")[0]
    assert restored.window >= grown, "override must lift the launch window"


def test_sampling_ladder_file_beats_catalog_beats_nothing(hermes_home, tmp_path, monkeypatch):
    """The sampling deference ladder: the GGUF's own general.sampling.*
    wins per key, catalog fills only what the file left silent, and a
    model with neither gets no sampling keys at all (llama.cpp defaults).
    Policy keys (ctx-size, cache types) must never be displaced."""
    import configparser

    import hermes_cli.local_runtime.presets as presets_mod
    from hermes_cli.local_runtime.catalog import CATALOG
    from hermes_cli.local_runtime.estimator import HardwareBudget

    # A real catalog entry WITH catalog sampling, staged on disk.
    entry = next(e for e in CATALOG if e.sampling)
    variant = entry.variants[-1]
    mdir = tmp_path / "models"
    _stage_fake_gguf(mdir, variant.model_id)
    _stage_fake_gguf(mdir, "off-catalog-model")

    gib = 1 << 30
    budget = HardwareBudget(usable_vram_bytes=64 * gib,
                            total_device_bytes=64 * gib,
                            ram_available_bytes=64 * gib)
    # The catalog model's file carries temp; catalog must fill the rest
    # but NOT displace the file's value. The off-catalog file carries none.
    def fake_header(path):
        if variant.model_id in str(path):
            return _header_stub({"temp": "0.42"})
        return _header_stub()

    monkeypatch.setattr(presets_mod, "read_gguf_header", fake_header)
    monkeypatch.setattr(presets_mod, "profile_from_gguf",
                        lambda h: _tiny_profile("x"))

    out = tmp_path / "presets.ini"
    presets_mod.generate_presets(mdir, budget, out)
    ini = configparser.ConfigParser()
    ini.read(out)

    sec = ini[variant.model_id]
    assert sec["temp"] == "0.42", "file's own sampling must win per key"
    for k, v in entry.sampling.items():
        if k != "temp":
            assert sec[k] == v, f"catalog must fill the silent key {k}"
    assert "ctx-size" in sec, "policy keys survive the ladder"

    off = ini["off-catalog-model"]
    assert "temp" not in off and "top-p" not in off, (
        "no file keys + no catalog entry = llama.cpp defaults, not ours")
