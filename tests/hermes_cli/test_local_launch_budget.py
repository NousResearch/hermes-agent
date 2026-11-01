"""Launch windows are sized beside what other programs hold on the card, not from capacity alone.

On a 32 GiB card, the capacity plan booted Qwen3.8 27B at 216K. That fit only while other
programs held about 3.4 GiB or less. With 4.5-6.3 GiB held, Windows paged part of the model to
host memory without an error, and decode fell from ~90 to ~24 tok/s. The window must shrink to
what fits now, and the weights must stay on the GPU.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from hermes_cli.local_runtime import hardware
from hermes_cli.local_runtime.estimator import HardwareBudget

GIB = 1 << 30
# The measured card: RTX 5090, 31.84 GiB total, capacity = total minus the 9% margin.
CARD_TOTAL = int(31.84 * GIB)
CARD_CAPACITY = HardwareBudget(
    usable_vram_bytes=CARD_TOTAL - max(hardware._MARGIN_FLOOR, int(CARD_TOTAL * hardware._MARGIN_FRACTION)),
    total_device_bytes=CARD_TOTAL, ram_available_bytes=256 * GIB, uma=False)
MODEL_ID = "Qwen3.8-27B-UD-Q4_K_M"


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    return tmp_path


def _card(monkeypatch, others_gib: float, *, total: int = CARD_TOTAL):
    monkeypatch.setattr(hardware, "_nvidia_vram",
                        lambda: (total, total - int(others_gib * GIB), "NVIDIA GeForce RTX 5090", None))


@pytest.fixture
def capacity_probe(monkeypatch):
    """Launch decisions run through a server bounce, so they price the card's capacity
    (``planning=True``), never raw free memory, which still counts the outgoing server. Free
    memory enters only through ``launch_budget``. Every call is recorded, so a probe inside a
    path that swallows exceptions still fails the test."""
    calls = []

    def probe(**kwargs):
        calls.append(kwargs)
        return CARD_CAPACITY

    monkeypatch.setattr(hardware, "probe_budget", probe)
    yield calls
    assert all(call == {"planning": True} for call in calls), calls


# ── launch_budget ────────────────────────────────────────────


def test_launch_budget_subtracts_other_programs_plus_headroom(monkeypatch):
    _card(monkeypatch, 5.4)
    budget = hardware.launch_budget(CARD_CAPACITY)
    assert budget.usable_vram_bytes == CARD_TOTAL - int(5.4 * GIB) - hardware._LAUNCH_HEADROOM
    assert budget.total_device_bytes == CARD_TOTAL and budget.uma is False


def test_launch_budget_reads_free_memory_on_a_vulkan_card(monkeypatch, tmp_path):
    """Behind Vulkan/HIP (no nvidia-smi) the engine's own device probe reports free memory, and each
    launch asks again: the cached capacity probe predates programs that started since."""
    from hermes_cli.local_runtime import devices

    held_gib = [0.0]
    monkeypatch.setattr(hardware, "_nvidia_vram", lambda: None)
    monkeypatch.setattr(hardware, "_configured_engine",
                        lambda: SimpleNamespace(backend="vulkan", binary=tmp_path / "llama-server"))
    monkeypatch.setattr(hardware, "_accelerator_cache", {})
    monkeypatch.setattr(devices, "probe_devices", lambda directory, backend: [{
        "description": "AMD Radeon RX 9060 XT", "type": 1, "total": CARD_TOTAL,
        "free": CARD_TOTAL - int(held_gib[0] * GIB)}])
    hardware._accelerator_device()  # capacity cache warmed while the card was idle
    held_gib[0] = 5.4

    budget = hardware.launch_budget(CARD_CAPACITY)
    assert budget.usable_vram_bytes == CARD_TOTAL - int(5.4 * GIB) - hardware._LAUNCH_HEADROOM


def test_launch_budget_never_exceeds_capacity(monkeypatch):
    _card(monkeypatch, 0.0)
    assert hardware.launch_budget(CARD_CAPACITY).usable_vram_bytes == CARD_CAPACITY.usable_vram_bytes


def test_launch_budget_counts_own_server_as_free(monkeypatch):
    """The managed server exits before the next instance loads; counting it as another program
    once pinned a fitting model's weights to the CPU."""
    _card(monkeypatch, 5.0 + 24.6)  # 5 GiB of other programs plus our own 24.6 GiB model
    budget = hardware.launch_budget(CARD_CAPACITY, own_bytes=int(24.6 * GIB))
    assert budget.usable_vram_bytes == pytest.approx(
        CARD_TOTAL - 5 * GIB - hardware._LAUNCH_HEADROOM, abs=1 << 20)


def test_launch_budget_skips_nvidia_unified_memory_and_failed_probes(monkeypatch):
    """An NVIDIA unified-memory pool sits partly behind a carve-out RAM readings cannot see."""
    uma = replace(CARD_CAPACITY, uma=True)
    monkeypatch.setattr(hardware, "_nvidia_vram", lambda: pytest.fail("UMA must not probe"))
    monkeypatch.setattr(hardware, "_windows_memory_status", lambda: None)  # Windows narrows by commit
    monkeypatch.setattr(hardware, "_ram_reading", lambda: pytest.fail("carve-out pools must not read RAM"))
    assert hardware.launch_budget(uma) is None
    monkeypatch.setattr(hardware, "_nvidia_vram", lambda: None)
    assert hardware.launch_budget(CARD_CAPACITY) is None


# ── unified memory budgeted from system RAM ──────────────────

MAC_TOTAL = 18 * GIB
MAC_CAPACITY = HardwareBudget(usable_vram_bytes=int(MAC_TOTAL * (1 - hardware._UMA_HEADROOM_FRACTION)),
                              total_device_bytes=MAC_TOTAL, ram_available_bytes=0, uma=True,
                              platform="darwin", ram_backed=True)


def _free_ram(monkeypatch, avail_gib: float, *, measured: bool = True, total: int = MAC_TOTAL):
    monkeypatch.setattr(hardware, "_ram_reading", lambda: (total, int(avail_gib * GIB), measured))
    monkeypatch.setattr(hardware, "_nvidia_vram", lambda: pytest.fail("RAM-backed UMA must not probe NVIDIA"))
    monkeypatch.setattr(hardware, "_accelerator_device",
                        lambda **_: pytest.fail("RAM-backed UMA must not probe the engine's device"))


def test_ram_backed_launch_budget_is_free_ram_less_headroom(monkeypatch):
    """Measured on an 18 GiB Mac: capacity (80% of total) granted a 27B 221K while ~5 GiB was free,
    and the load drove free memory to 5% before failing."""
    _free_ram(monkeypatch, 5.0)
    budget = hardware.launch_budget(MAC_CAPACITY)
    assert budget.uma is True and budget.ram_backed is True
    assert budget.usable_vram_bytes == 5 * GIB - hardware._LAUNCH_HEADROOM


def test_ram_backed_launch_budget_never_exceeds_capacity_and_counts_own_server(monkeypatch):
    _free_ram(monkeypatch, 17.5)
    assert hardware.launch_budget(MAC_CAPACITY).usable_vram_bytes == MAC_CAPACITY.usable_vram_bytes
    _free_ram(monkeypatch, 2.0)
    budget = hardware.launch_budget(MAC_CAPACITY, own_bytes=6 * GIB)
    assert budget.usable_vram_bytes == 8 * GIB - hardware._LAUNCH_HEADROOM


def test_ram_backed_launch_budget_bottoms_out_at_zero(monkeypatch):
    _free_ram(monkeypatch, 0.25)
    assert hardware.launch_budget(MAC_CAPACITY, own_bytes=256 << 20).usable_vram_bytes == 0


def test_a_guessed_free_ram_figure_keeps_the_capacity_plan(monkeypatch):
    """vm_stat failing leaves half of total as a guess; a guess must not size a launch."""
    _free_ram(monkeypatch, 9.0, measured=False)
    assert hardware.launch_budget(MAC_CAPACITY) is None
    _free_ram(monkeypatch, 0.0, measured=False, total=0)  # sysctl failed too
    assert hardware.launch_budget(MAC_CAPACITY) is None


@pytest.mark.platforms("macos")
def test_mac_ram_reading_flags_the_vm_stat_fallback(monkeypatch):
    page = 16384

    def stdout(*argv):
        if argv[0].endswith("sysctl"):
            return str(MAC_TOTAL)
        return vm_stat[0]

    monkeypatch.setattr(hardware, "_stdout", stdout)
    vm_stat = [f"Mach Virtual Memory Statistics: (page size of {page} bytes)\n"
               f"Pages free:                  {GIB // page}.\n"
               f"Pages inactive:              {2 * GIB // page}.\n"]
    assert hardware._ram_reading() == (MAC_TOTAL, 3 * GIB, True)
    assert hardware._ram_bytes() == (MAC_TOTAL, 3 * GIB)
    vm_stat[0] = ""  # vm_stat missing or unparseable
    assert hardware._ram_reading() == (MAC_TOTAL, MAC_TOTAL // 2, False)


def _no_discrete_gpu(monkeypatch):
    monkeypatch.setattr(hardware, "_nvidia_vram", lambda: None)
    monkeypatch.setattr(hardware, "_device_pool_view", lambda: None)
    monkeypatch.setattr(hardware, "_accelerator_device", lambda **_: None)
    monkeypatch.setattr(hardware, "_ram_bytes", lambda: (32 * GIB, 6 * GIB))
    monkeypatch.setattr(hardware, "_ram_reading", lambda: (32 * GIB, 6 * GIB, True))


@pytest.mark.platforms("linux")
def test_linux_igpu_budget_is_ram_backed(monkeypatch):
    """A Linux integrated GPU (or Asahi) builds its budget from RAM, so it gets the launch-time
    reading too, not just macOS."""
    _no_discrete_gpu(monkeypatch)
    capacity = hardware.probe_budget(planning=True)
    assert capacity.uma and capacity.ram_backed
    assert hardware.launch_budget(capacity).usable_vram_bytes == 5 * GIB


@pytest.mark.platforms("windows")
def test_windows_igpu_pool_is_ram_backed(monkeypatch):
    _no_discrete_gpu(monkeypatch)
    monkeypatch.setattr(hardware, "_windows_igpu_bytes", lambda device: 18 * GIB)
    pool = hardware.probe_budget(planning=True)
    assert pool.ram_backed and pool.total_device_bytes == 18 * GIB


def test_nvidia_unified_pool_budget_is_not_ram_backed(monkeypatch):
    monkeypatch.setattr(hardware, "_ram_bytes", lambda: (128 * GIB, 100 * GIB))
    monkeypatch.setattr(hardware, "_nvidia_vram", lambda: (16 * GIB, 15 * GIB, "NVIDIA GB10", None))
    monkeypatch.setattr(hardware, "_device_pool_view", lambda: (120 * GIB, True))
    capacity = hardware.probe_budget(planning=True)
    assert capacity.uma and not capacity.ram_backed
    assert hardware.launch_budget(capacity) is None


# ── launch windows for the real catalog 27B ─────────────────


def _stage_27b(hermes_home, monkeypatch):
    """The real catalog entry (mmproj staged), priced from its catalog profile."""
    from hermes_cli.local_runtime import bootstrap, catalog, presets

    entry = catalog.catalog_by_id()["qwen3.8-27b"]
    variant = next(v for v in entry.variants if v.model_id == MODEL_ID)
    mdir = hermes_home / "models"
    mdir.mkdir(parents=True, exist_ok=True)
    # First: assets_dir() resolves under models_dir(), so the projector lands in this test's tree.
    monkeypatch.setattr(bootstrap, "models_dir", lambda: mdir)
    gguf = mdir / f"{MODEL_ID}.gguf"
    gguf.write_bytes(b"GGUF" + b"\x00" * 64)
    asset = bootstrap.assets_dir() / entry.mmproj.local_name
    asset.parent.mkdir(parents=True, exist_ok=True)
    asset.touch()
    profile = entry.profile(variant)
    monkeypatch.setattr(presets, "read_gguf_header", lambda p: SimpleNamespace(sampling_defaults={}))
    monkeypatch.setattr(presets, "profile_from_gguf", lambda h: profile)
    monkeypatch.setattr(bootstrap, "staged_models", lambda: [gguf])
    monkeypatch.setattr(bootstrap, "staged_in", lambda d: [gguf])
    return mdir, gguf


@pytest.mark.parametrize("others_gib,window_k", [
    (0.0, 216),   # quiet card: capacity plan unchanged
    (3.0, 216),
    (3.6, 144),   # measured: 216K ran with 0.7 GiB spare here — too close to keep
    (4.5, 144),   # measured: 216K demoted to ~24 tok/s here
    (5.4, 144),
    (6.5, 96),    # measured smoke run held 6.3 GiB at peak
    (8.5, 64),
])
def test_27b_window_follows_other_programs(hermes_home, monkeypatch, others_gib, window_k):
    from hermes_cli.local_runtime import presets

    _mdir, gguf = _stage_27b(hermes_home, monkeypatch)
    _card(monkeypatch, others_gib)
    entry = presets.preset_for_model(gguf, CARD_CAPACITY, set(), requested_window=0,
                                     live=hardware.launch_budget(CARD_CAPACITY))
    assert entry.window == window_k * 1024
    assert entry.spilled is False and "override-tensor" not in entry.keys
    assert entry.keys["ctx-size"] == str(window_k * 1024)
    assert entry.keys["mmproj"].endswith("mmproj-Qwen3.8-27B-BF16.gguf")


def test_busy_card_holds_the_floor_and_keeps_weights_on_the_gpu(hermes_home, monkeypatch):
    """A live reading may still include memory about to be freed; it never moves weights."""
    from hermes_cli.local_runtime import presets

    _mdir, gguf = _stage_27b(hermes_home, monkeypatch)
    _card(monkeypatch, 14.0)
    entry = presets.preset_for_model(gguf, CARD_CAPACITY, set(), requested_window=0,
                                     live=hardware.launch_budget(CARD_CAPACITY))
    assert entry.window == 64 * 1024
    assert entry.spilled is False and "override-tensor" not in entry.keys


def test_a_capacity_plan_that_spills_is_left_alone(hermes_home, monkeypatch):
    """Small cards spill by design; a live reading must not rearrange that placement."""
    from hermes_cli.local_runtime import presets

    _mdir, gguf = _stage_27b(hermes_home, monkeypatch)
    small = HardwareBudget(usable_vram_bytes=10 * GIB, total_device_bytes=12 * GIB,
                           ram_available_bytes=64 * GIB)
    _card(monkeypatch, 3.0, total=12 * GIB)
    capacity_only = presets.preset_for_model(gguf, small, set(), requested_window=0)
    with_live = presets.preset_for_model(gguf, small, set(), requested_window=0,
                                         live=hardware.launch_budget(small))
    assert capacity_only.spilled and with_live == capacity_only


def test_recommendations_still_price_against_capacity(hermes_home, monkeypatch):
    """The catalog and picker describe the card, not this minute's desktop."""
    from hermes_cli.local_runtime import catalog

    _card(monkeypatch, 8.5)
    entry = catalog.catalog_by_id()["qwen3.8-27b"]
    variant = next(v for v in entry.variants if v.model_id == MODEL_ID)
    assert entry.launch_plan(variant, CARD_CAPACITY).decision.window == 216 * 1024


# ── boot wiring ──────────────────────────────────────────────


def test_boot_writes_the_launch_window(hermes_home, monkeypatch, capacity_probe):
    from hermes_cli.local_runtime import bootstrap, presets

    mdir, _gguf = _stage_27b(hermes_home, monkeypatch)
    _card(monkeypatch, 5.4)
    path = hermes_home / "presets.ini"
    assert bootstrap._generate_presets(mdir, path) == path
    assert presets.read_preset_decisions(path)[MODEL_ID].window == 144 * 1024


def test_boot_keeps_capacity_when_the_probe_fails(hermes_home, monkeypatch, capacity_probe):
    """The probe is policy, not a prerequisite: a failure must never block a boot."""
    from hermes_cli.local_runtime import bootstrap, presets

    mdir, _gguf = _stage_27b(hermes_home, monkeypatch)

    def boom():
        raise OSError("nvidia-smi vanished")

    monkeypatch.setattr(hardware, "_nvidia_vram", boom)
    path = hermes_home / "presets.ini"
    assert bootstrap._generate_presets(mdir, path) == path
    assert presets.read_preset_decisions(path)[MODEL_ID].window == 216 * 1024


# ── re-planning while nothing is loaded ──────────────────────


class _Router:
    def __init__(self, preset_path, statuses):
        self.preset_path = preset_path
        self.statuses = statuses
        self.reloads = 0
        self.models_max = 1
        self.proc = SimpleNamespace(poll=lambda: None)
        self._refit_usable = None
        import threading

        self._lifecycle_lock = threading.RLock()

    def models(self, timeout_s=30):
        return dict(self.statuses)

    def reload_presets(self):
        self.reloads += 1


def _booted(hermes_home, monkeypatch, others_gib):
    from hermes_cli.local_runtime import bootstrap

    mdir, _gguf = _stage_27b(hermes_home, monkeypatch)
    _card(monkeypatch, others_gib)
    path = hermes_home / "presets.ini"
    bootstrap._generate_presets(mdir, path)
    return path


def test_idle_router_gets_windows_for_the_current_desktop(hermes_home, monkeypatch, capacity_probe):
    from hermes_cli.local_runtime import bootstrap, presets

    path = _booted(hermes_home, monkeypatch, 6.5)
    assert presets.read_preset_decisions(path)[MODEL_ID].window == 96 * 1024
    router = _Router(path, {MODEL_ID: "unloaded"})
    _card(monkeypatch, 2.0)  # the user closed the heavy apps
    assert bootstrap.refit_idle_presets(router) is True
    assert router.reloads == 1
    assert presets.read_preset_decisions(path)[MODEL_ID].window == 216 * 1024
    # Nothing changed since: no rewrite, no reload.
    assert bootstrap.refit_idle_presets(router) is False and router.reloads == 1


def test_a_replan_keeps_the_draft_the_serving_engine_loads(hermes_home, monkeypatch, capacity_probe):
    """The idle re-plan decides for the engine the server runs, as boot did: a staged DFlash draft
    stays in the preset, not the MTP fallback an unknown engine gets."""
    from hermes_cli.local_runtime import bootstrap, catalog

    mdir, _gguf = _stage_27b(hermes_home, monkeypatch)
    draft = catalog.catalog_by_id()["qwen3.8-27b"].draft
    (bootstrap.assets_dir() / draft.local_name).touch()
    engine = SimpleNamespace(tag="b11671")
    monkeypatch.setattr(bootstrap, "_SERVING_ENGINE", engine)
    _card(monkeypatch, 2.0)
    path = hermes_home / "presets.ini"
    bootstrap._generate_presets(mdir, path, engine_tag=engine.tag)
    assert "spec-type = draft-dflash" in path.read_text(encoding="utf-8-sig")

    _card(monkeypatch, 0.0)  # the user closed the heavy apps
    bootstrap.refit_idle_presets(_Router(path, {MODEL_ID: "unloaded"}))

    text = path.read_text(encoding="utf-8-sig")
    assert "spec-type = draft-dflash" in text and "parallel = 1" in text


@pytest.mark.parametrize("status", ["loaded", "loading", "sleeping"])
def test_never_rewrites_while_a_model_holds_memory(hermes_home, monkeypatch, capacity_probe, status):
    """The router unloads a loaded model whose launch flags change on reload."""
    from hermes_cli.local_runtime import bootstrap

    path = _booted(hermes_home, monkeypatch, 6.5)
    before = path.read_text(encoding="utf-8-sig")
    router = _Router(path, {MODEL_ID: status})
    _card(monkeypatch, 2.0)
    assert bootstrap.refit_idle_presets(router) is False
    assert router.reloads == 0 and path.read_text(encoding="utf-8-sig") == before


def test_small_changes_in_free_memory_do_not_replan(hermes_home, monkeypatch, capacity_probe):
    from hermes_cli.local_runtime import bootstrap, presets

    path = _booted(hermes_home, monkeypatch, 5.4)
    router = _Router(path, {MODEL_ID: "unloaded"})
    assert bootstrap.refit_idle_presets(router) is False  # same reading as boot: same file
    monkeypatch.setattr(presets, "plan_presets", lambda *a, **k: pytest.fail("re-planned"))
    _card(monkeypatch, 5.5)
    assert bootstrap.refit_idle_presets(router) is False


# ── growth ───────────────────────────────────────────────────


def test_growth_counts_the_growing_model_as_free(hermes_home, monkeypatch, capacity_probe):
    """The grown instance loads after the current one exits. Counting the model's own memory as
    held would veto every rung."""
    from hermes_cli.local_runtime import bootstrap, growth, presets

    mdir, gguf = _stage_27b(hermes_home, monkeypatch)
    monkeypatch.setattr(bootstrap, "get_supervisor", lambda: SimpleNamespace(is_idle=lambda m: True))
    monkeypatch.setattr(growth, "is_managed_endpoint", lambda url: True)
    from hermes_cli.local_runtime import estimator, gguf as gguf_mod

    monkeypatch.setattr(gguf_mod, "read_gguf_header", presets.read_gguf_header)
    monkeypatch.setattr(estimator, "profile_from_gguf", presets.profile_from_gguf)
    own = presets.resident_footprint(gguf, CARD_CAPACITY, 144 * 1024)
    _card(monkeypatch, 2.0 + own / GIB)  # 2 GiB of other programs plus our loaded 144K model
    path = hermes_home / "presets.ini"

    def refresh():
        presets.generate_presets(mdir, CARD_CAPACITY, path)
        return True

    monkeypatch.setattr(bootstrap, "refresh_local_runtime", refresh)
    real_read = presets.read_preset_decisions
    monkeypatch.setattr(presets, "read_preset_decisions", lambda p=None: real_read(path))
    grown = growth.maybe_grow_window(MODEL_ID, base_url="http://127.0.0.1:1/v1",
                                     session_tokens=140 * 1024, current_window=144 * 1024)
    assert grown == 216 * 1024


def test_growth_refuses_a_rung_other_programs_leave_no_room_for(hermes_home, monkeypatch, capacity_probe):
    from hermes_cli.local_runtime import bootstrap, growth, presets

    _mdir, gguf = _stage_27b(hermes_home, monkeypatch)
    monkeypatch.setattr(bootstrap, "get_supervisor", lambda: SimpleNamespace(is_idle=lambda m: True))
    monkeypatch.setattr(growth, "is_managed_endpoint", lambda url: True)
    from hermes_cli.local_runtime import estimator, gguf as gguf_mod

    monkeypatch.setattr(gguf_mod, "read_gguf_header", presets.read_gguf_header)
    monkeypatch.setattr(estimator, "profile_from_gguf", presets.profile_from_gguf)
    own = presets.resident_footprint(gguf, CARD_CAPACITY, 144 * 1024)
    _card(monkeypatch, 5.4 + own / GIB)
    monkeypatch.setattr(bootstrap, "refresh_local_runtime", lambda: pytest.fail("bounced"))
    assert growth.maybe_grow_window(MODEL_ID, base_url="http://127.0.0.1:1/v1",
                                    session_tokens=140 * 1024, current_window=144 * 1024) is None
    assert not growth.load_window_overrides()


# ── unified memory: refusal and re-planning ─────────────────

# A 36 GiB Mac: capacity grants the catalog 27B its 216K window from total RAM alone.
BIG_MAC = HardwareBudget(usable_vram_bytes=int(36 * GIB * (1 - hardware._UMA_HEADROOM_FRACTION)),
                         total_device_bytes=36 * GIB, ram_available_bytes=0, uma=True,
                         platform="darwin", ram_backed=True)


@pytest.fixture
def mac_probe(monkeypatch):
    monkeypatch.setattr(hardware, "probe_budget", lambda **kwargs: BIG_MAC)


def _mac_free(monkeypatch, free_gib: float):
    monkeypatch.setattr(hardware, "_ram_reading", lambda: (36 * GIB, int(free_gib * GIB), True))


def test_unified_memory_window_follows_free_ram(hermes_home, monkeypatch):
    from hermes_cli.local_runtime import presets

    _mdir, gguf = _stage_27b(hermes_home, monkeypatch)
    assert presets.preset_for_model(gguf, BIG_MAC, set(), requested_window=0).window == 216 * 1024
    _mac_free(monkeypatch, 24.0)
    entry = presets.preset_for_model(gguf, BIG_MAC, set(), requested_window=0,
                                     live=hardware.launch_budget(BIG_MAC))
    assert entry.refusal is None and entry.window == 96 * 1024


def test_unified_memory_refuses_weights_free_ram_cannot_hold(hermes_home, monkeypatch):
    """There is no CPU to fall back to: holding the 64K floor here launched a load that failed."""
    from hermes_cli.local_runtime import presets

    _mdir, gguf = _stage_27b(hermes_home, monkeypatch)
    _mac_free(monkeypatch, 10.0)
    entry = presets.preset_for_model(gguf, BIG_MAC, set(), requested_window=0,
                                     live=hardware.launch_budget(BIG_MAC))
    assert entry.window == 0 and entry.keys is None
    assert "free now" in entry.refusal and "close other apps" in entry.refusal
    rendered = presets.render_presets([entry])
    assert "ctx-size" not in rendered and f"[{MODEL_ID}]" not in rendered


def test_a_discrete_card_still_holds_the_floor_rather_than_refusing(hermes_home, monkeypatch):
    """Only RAM-backed budgets refuse: a busy card's reading may include memory about to be freed."""
    from hermes_cli.local_runtime import presets

    _mdir, gguf = _stage_27b(hermes_home, monkeypatch)
    _card(monkeypatch, 28.0)
    entry = presets.preset_for_model(gguf, CARD_CAPACITY, set(), requested_window=0,
                                     live=hardware.launch_budget(CARD_CAPACITY))
    assert entry.refusal is None and entry.window == 64 * 1024


def test_idle_replan_admits_a_refused_model_once_ram_frees(hermes_home, monkeypatch, mac_probe):
    from hermes_cli.local_runtime import bootstrap, presets

    mdir, _gguf = _stage_27b(hermes_home, monkeypatch)
    _mac_free(monkeypatch, 10.0)
    path = hermes_home / "presets.ini"
    bootstrap._generate_presets(mdir, path)
    assert presets.read_preset_decisions(path)[MODEL_ID].refusal
    router = _Router(path, {})
    _mac_free(monkeypatch, 30.0)  # the user quit the heavy apps: free RAM now exceeds capacity
    assert bootstrap.refit_idle_presets(router) is True and router.reloads == 1
    assert presets.read_preset_decisions(path)[MODEL_ID].window == 216 * 1024


def test_free_ram_drift_does_not_replan_every_pass(hermes_home, monkeypatch, mac_probe):
    """Free RAM moves by hundreds of MiB between readings; that must not re-read every header."""
    from hermes_cli.local_runtime import bootstrap, presets

    mdir, _gguf = _stage_27b(hermes_home, monkeypatch)
    _mac_free(monkeypatch, 24.0)
    path = hermes_home / "presets.ini"
    bootstrap._generate_presets(mdir, path)
    router = _Router(path, {MODEL_ID: "unloaded"})
    assert bootstrap.refit_idle_presets(router) is False  # same reading as boot: same file
    real_plan = presets.plan_presets
    monkeypatch.setattr(presets, "plan_presets", lambda *a, **k: pytest.fail("re-planned"))
    for drift_gib in (24.6, 23.4, 24.9):  # past the discrete-card step, inside the UMA one
        assert bootstrap._REFIT_STEP_BYTES < abs(drift_gib - 24) * GIB < bootstrap._UMA_REFIT_STEP_BYTES
        _mac_free(monkeypatch, drift_gib)
        assert bootstrap.refit_idle_presets(router) is False
    monkeypatch.setattr(presets, "plan_presets", real_plan)
    _mac_free(monkeypatch, 30.0)
    assert bootstrap.refit_idle_presets(router) is True


# ── budget source ────────────────────────────────────────────


def test_residency_cap_prices_capacity(hermes_home, monkeypatch, capacity_probe):
    """The residency cap is also decided before a server bounce. It swallows probe errors to keep
    boots unblocked, so the recorded call is the proof it priced capacity."""
    from hermes_cli.local_runtime import bootstrap

    mdir, _gguf = _stage_27b(hermes_home, monkeypatch)
    bootstrap._admitted_models_max(mdir, 4)
    assert capacity_probe == [{"planning": True}]
