"""Per-model limits survive every preset writer and use the posture they price."""

from dataclasses import replace
from contextlib import nullcontext
import struct
import threading
from types import SimpleNamespace

import pytest

from hermes_cli.config import atomic_config_write
from hermes_cli.local_runtime import bootstrap, catalog, growth, hardware, presets
from hermes_cli.local_runtime.context_policy import RUNTIME_OVERHEAD_BYTES
from hermes_cli.local_runtime.estimator import HardwareBudget, footprint_bytes, profile_from_gguf
from hermes_cli.local_runtime.gguf import read_gguf_header

QWEN = "Qwen3.6-35B-A3B-UD-Q4_K_M"
CAPACITY = HardwareBudget(64 << 30, 64 << 30, 64 << 30)


def _stage(path, native):
    """Metadata-only GGUF: real reader, vocabulary and window-scaled compute pricing."""
    metadata = {
        "general.architecture": "qwen4exp",
        "qwen4exp.context_length": native,
        "qwen4exp.block_count": 2,
        "qwen4exp.attention.head_count": 4,
        "qwen4exp.attention.head_count_kv": 2,
        "qwen4exp.attention.key_length": 64,
        "qwen4exp.attention.value_length": 64,
        "qwen4exp.vocab_size": 10000,
        "general.sampling.top_k": 17,
    }

    def string(value):
        raw = value.encode()
        return struct.pack("<Q", len(raw)) + raw

    data = b"GGUF" + struct.pack("<IQQ", 3, 0, len(metadata))
    for key, value in metadata.items():
        kind, raw = ((8, string(value)) if isinstance(value, str) else
                     (4, struct.pack("<I", value)))
        data += string(key) + struct.pack("<I", kind) + raw
    path.write_bytes(data)


@pytest.fixture
def staged(tmp_path, monkeypatch):
    root = tmp_path / "home"
    profile = root / "profiles" / "coder"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    atomic_config_write(profile / "config.yaml", {"local_runtime": {"model_settings": {
        QWEN: {"context_cap": 262144, "batch_posture": "auto"}}}})
    mdir = bootstrap.models_dir()
    assert mdir == root / "models"
    mdir.mkdir()
    monkeypatch.setattr(hardware, "probe_budget", lambda **kwargs: CAPACITY)
    monkeypatch.setattr(bootstrap, "_launch_budget", lambda *args, **kwargs: CAPACITY)
    return root, mdir


def _configure(root, model_id, settings):
    atomic_config_write(root / "config.yaml", {
        "local_runtime": {"model_settings": {model_id: settings}}})


class _Router:
    def __init__(self, path):
        self.preset_path = path
        self.models_max = 1
        self.proc = SimpleNamespace(poll=lambda: None)
        self._refit_usable = None
        self._lifecycle_lock = threading.RLock()
        self.reloads = 0

    def models(self, timeout_s):
        return {}

    def reload_presets(self):
        self.reloads += 1


@pytest.mark.parametrize("model_id,native", [
    (QWEN, 262144), (QWEN, 16384), ("off-catalog-model", 262144),
])
def test_limits_and_batch_footprint_survive_every_regeneration(staged, monkeypatch, model_id, native):
    root, mdir = staged
    gguf = mdir / f"{model_id}.gguf"
    control = mdir / "control-model.gguf"
    _stage(gguf, native)
    _stage(control, 262144)
    entry = catalog.entry_for_model(model_id)
    companion_bytes = 0
    if entry is not None:
        projector = bootstrap.assets_dir() / entry.mmproj.local_name
        projector.parent.mkdir()
        projector.touch()
        companion_bytes = entry.mmproj.size_bytes
    path = root / "presets.ini"
    bootstrap._generate_presets(mdir, path)
    automatic = path.read_text(encoding="utf-8")
    control_before = presets.read_preset_decisions(path)["control-model"]
    _configure(root, model_id, {"context_cap": 32768, "batch_posture": "backend-default"})
    growth.save_window_override(model_id, 10_000_000)
    window = min(native, 32768)

    def assert_policy():
        target = presets.read_preset_decisions(path)[model_id]
        assert target.window == window and not target.spilled
        assert "batch-size" not in target.keys and "ubatch-size" not in target.keys
        assert target.keys["cache-type-k"] == target.keys["cache-type-v"] == "q8_0"
        assert target.keys["top-k"] == "17"
        if entry is not None:
            assert target.keys["mmproj"] == str(projector)
            assert target.keys["spec-type"] == "draft-mtp"
            assert target.keys["spec-draft-n-max"] == str(entry.mtp_draft_depth)
            assert target.keys["backend-sampling"] == "on"
            assert target.keys["spec-draft-backend-sampling"] == "on"
        assert presets.read_preset_decisions(path)["control-model"] == control_before

    for regenerate in (
        lambda: bootstrap._generate_presets(mdir, path),
        lambda: presets.generate_presets(mdir, CAPACITY, path),
        lambda: bootstrap.refit_idle_presets(_Router(path)),
    ):
        # Reproduce the previous large automatic policy before each owning writer runs.
        path.write_text(automatic, encoding="utf-8")
        regenerate()
        assert_policy()

    # A live-fit fallback must honor the smaller ceiling even if no resident rung fits now.
    busy = HardwareBudget(0, 64 << 30, 64 << 30)
    for requested in (None, 0, 10_000_000):
        target = presets.preset_for_model(gguf, CAPACITY, set(), requested_window=requested, live=busy)
        assert target.window == window and not target.spilled
        assert "ubatch-size" not in target.keys and "batch-size" not in target.keys

    # Prices must use the same 512 microbatch, MTP context count and capped window as the INI.
    profile = profile_from_gguf(read_gguf_header(gguf))
    contexts = 2 if entry is not None else 1
    priced = replace(profile, kv_scale=1.2 if entry is not None else 1.0,
                     window_compute_per_token=46 * 512 * contexts)
    overhead = RUNTIME_OVERHEAD_BYTES + companion_bytes + 512 * profile.n_vocab * 4 * contexts
    expected = footprint_bytes(priced, window, overhead_bytes=overhead)
    assert presets.resident_footprint(gguf, CAPACITY, window, preset_path=path) == expected
    assert presets._launch_footprint(gguf, CAPACITY) == expected
    refusal = presets.preset_for_model(gguf, HardwareBudget(0, 0, 0), set())
    assert refusal.refusal and refusal.keys is None


def test_growth_stops_at_the_configured_ceiling_without_saving_or_refreshing(staged, monkeypatch):
    root, mdir = staged
    gguf = mdir / f"{QWEN}.gguf"
    _stage(gguf, 262144)
    _configure(root, QWEN, {"context_cap": 32768, "batch_posture": "backend-default"})
    monkeypatch.setattr(bootstrap, "get_supervisor", lambda: SimpleNamespace(is_idle=lambda _: True))
    monkeypatch.setattr(growth, "is_managed_endpoint", lambda _: True)
    monkeypatch.setattr(bootstrap, "refresh_local_runtime", lambda: pytest.fail("refreshed past cap"))
    assert growth.maybe_grow_window(QWEN, base_url="http://127.0.0.1:1/v1",
                                   session_tokens=32768, current_window=32768) is None
    assert growth.load_window_overrides() == {}


@pytest.mark.parametrize("settings", [
    None, [], {"context_cap": None}, {"context_cap": 0}, {"context_cap": -1},
    {"context_cap": True}, {"context_cap": "32768"}, {"batch_posture": "unknown"},
    {"context_caps": 32768},
])
def test_invalid_explicit_model_settings_are_rejected(staged, settings):
    root, mdir = staged
    gguf = mdir / f"{QWEN}.gguf"
    _stage(gguf, 262144)
    _configure(root, QWEN, settings)
    with pytest.raises(ValueError):
        presets.preset_for_model(gguf, CAPACITY, set())


@pytest.mark.parametrize("prior_ini", [False, True])
@pytest.mark.parametrize("malformed_yaml", [False, True])
def test_invalid_owner_policy_blocks_real_boot_before_router_construction(
        staged, monkeypatch, prior_ini, malformed_yaml):
    from hermes_cli.local_runtime import binaries, endpoint, supervisor

    root, mdir = staged
    _stage(mdir / f"{QWEN}.gguf", 262144)
    path = binaries.runtimes_root() / "presets.ini"
    if prior_ini:
        presets.generate_presets(mdir, CAPACITY, path)
    previous = path.read_bytes() if prior_ini else None
    if malformed_yaml:
        (root / "config.yaml").write_text("local_runtime: [", encoding="utf-8")
    else:
        _configure(root, QWEN, {"context_cap": "32768"})
    constructed = []
    monkeypatch.setattr(supervisor, "LlamaServerSupervisor", lambda *args, **kwargs: constructed.append(kwargs))
    monkeypatch.setattr(binaries, "installed_engine", lambda _: SimpleNamespace(
        binary=root / "llama-server", backend="cpu", tag="test"))
    monkeypatch.setattr(endpoint, "_state_endpoint", lambda: None)
    monkeypatch.setattr(bootstrap, "_SUPERVISOR", None)
    monkeypatch.setattr(bootstrap, "_SERVING_ENGINE", None)
    monkeypatch.setattr(bootstrap, "adopt_legacy_models", list)
    monkeypatch.setattr(bootstrap, "_cross_process_boot_lock", nullcontext)
    assert bootstrap.ensure_local_runtime({"local_runtime": {"enabled": True}}) is None
    assert constructed == []
    assert (path.read_bytes() if path.exists() else None) == previous


@pytest.mark.parametrize("prior_ini", [False, True])
@pytest.mark.parametrize("failure", ["probe", "write"])
@pytest.mark.parametrize("configured", [False, True])
def test_valid_owner_limits_block_boot_fallback_on_generation_failure(
        staged, monkeypatch, prior_ini, failure, configured):
    from hermes_cli.local_runtime import binaries, endpoint, supervisor

    root, mdir = staged
    _stage(mdir / f"{QWEN}.gguf", 262144)
    path = binaries.runtimes_root() / "presets.ini"
    if prior_ini:
        presets.generate_presets(mdir, CAPACITY, path)
    previous = path.read_bytes() if prior_ini else None
    if configured:
        _configure(root, QWEN, {"context_cap": 32768, "batch_posture": "backend-default"})

    def fail(*args, **kwargs):
        raise OSError(f"{failure} unavailable")

    if failure == "probe":
        monkeypatch.setattr(hardware, "probe_budget", fail)
    else:
        monkeypatch.setattr(presets, "write_presets", fail)
    constructed = []
    monkeypatch.setattr(supervisor, "LlamaServerSupervisor", lambda *args, **kwargs: constructed.append(kwargs))
    monkeypatch.setattr(binaries, "installed_engine", lambda _: SimpleNamespace(
        binary=root / "llama-server", backend="cpu", tag="test"))
    monkeypatch.setattr(endpoint, "_state_endpoint", lambda: None)
    monkeypatch.setattr(bootstrap, "_SUPERVISOR", None)
    monkeypatch.setattr(bootstrap, "_SERVING_ENGINE", None)
    monkeypatch.setattr(bootstrap, "adopt_legacy_models", list)
    monkeypatch.setattr(bootstrap, "_cross_process_boot_lock", nullcontext)
    assert bootstrap.ensure_local_runtime({"local_runtime": {"enabled": True}}) is None
    if configured:
        assert constructed == []
    else:
        assert len(constructed) == 1
        assert constructed[0]["preset_path"] == (path if prior_ini else None)
    assert (path.read_bytes() if path.exists() else None) == previous


def test_invalid_idle_refit_leaves_admitted_policy_untouched(staged):
    from hermes_cli.local_runtime.model_settings import InvalidModelSettings

    root, mdir = staged
    _stage(mdir / f"{QWEN}.gguf", 262144)
    path = root / "presets.ini"
    presets.generate_presets(mdir, CAPACITY, path)
    previous = path.read_bytes()
    _configure(root, QWEN, {"batch_posture": "unknown"})
    router = _Router(path)
    with pytest.raises(InvalidModelSettings):
        bootstrap.refit_idle_presets(router)
    assert path.read_bytes() == previous
    assert router.reloads == 0 and router._refit_usable is None


@pytest.mark.parametrize("new_cap", [32768, 262144])
def test_loaded_residency_ignores_edited_policy_and_preserves_live_reserve(staged, monkeypatch, new_cap):
    root, mdir = staged
    gguf = mdir / f"{QWEN}.gguf"
    _stage(gguf, 262144)
    path = root / "admitted.ini"
    current = 65536
    _configure(root, QWEN, {"context_cap": current, "batch_posture": "backend-default"})
    presets.generate_presets(mdir, CAPACITY, path)
    admitted = path.read_bytes()
    profile = profile_from_gguf(read_gguf_header(gguf))
    priced = replace(profile, kv_scale=1.2, window_compute_per_token=46 * 512 * 2)
    overhead = RUNTIME_OVERHEAD_BYTES + 512 * profile.n_vocab * 4 * 2
    actual_own = footprint_bytes(priced, current, overhead_bytes=overhead)
    _configure(root, QWEN, {"context_cap": new_cap, "batch_posture": "auto"})
    assert presets.resident_footprint(gguf, CAPACITY, current, preset_path=path) == actual_own
    assert presets.resident_footprint(gguf, CAPACITY, current * 2, preset_path=path) > actual_own

    credits, live_budgets = [], []
    others = 58 << 30
    monkeypatch.setattr(hardware, "_nvidia_vram", lambda: (
        CAPACITY.total_device_bytes, CAPACITY.total_device_bytes - others - actual_own, "test", None))

    def launch_budget(capacity, own_bytes=0):
        credits.append(own_bytes)
        live = hardware.launch_budget(capacity, own_bytes=own_bytes)
        live_budgets.append(live)
        return live

    monkeypatch.setattr(bootstrap, "_launch_budget", launch_budget)
    monkeypatch.setattr(bootstrap, "get_supervisor", lambda: SimpleNamespace(
        is_idle=lambda _: True, preset_path=path))
    monkeypatch.setattr(growth, "is_managed_endpoint", lambda _: True)
    monkeypatch.setattr(bootstrap, "refresh_local_runtime", lambda: pytest.fail("spent other programs' reserve"))
    assert growth.maybe_grow_window(QWEN, base_url="http://127.0.0.1:1/v1",
                                   session_tokens=current, current_window=current) is None
    if new_cap > current:
        assert credits == [actual_own]
        assert live_budgets[0].usable_vram_bytes == (
            CAPACITY.total_device_bytes - others - hardware._LAUNCH_HEADROOM)
    else:
        assert credits == []
    assert growth.load_window_overrides() == {} and path.read_bytes() == admitted
    path.unlink()
    assert presets.resident_footprint(gguf, CAPACITY, current, preset_path=path) is None
