"""Launch windows are sized beside what other programs hold on the card, not from capacity alone.

On a 32 GiB card, the capacity plan booted Qwen3.8 27B at 216K. That fit only while other
programs held about 3.4 GiB or less. With 4.5-6.3 GiB held, Windows paged part of the model to
host memory without an error, and decode fell from ~90 to ~24 tok/s. The window must shrink to
what fits now, and the weights must stay on the GPU.
"""
from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
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
                        lambda: (total, total - int(others_gib * GIB), "NVIDIA GeForce RTX 5090"))


# ── launch_budget ────────────────────────────────────────────


def test_launch_budget_subtracts_other_programs_plus_headroom(monkeypatch):
    _card(monkeypatch, 5.4)
    budget = hardware.launch_budget(CARD_CAPACITY)
    assert budget.usable_vram_bytes == CARD_TOTAL - int(5.4 * GIB) - hardware._LAUNCH_HEADROOM
    assert budget.total_device_bytes == CARD_TOTAL and budget.uma is False


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


def test_launch_budget_skips_unified_memory_and_failed_probes(monkeypatch):
    uma = replace(CARD_CAPACITY, uma=True)
    monkeypatch.setattr(hardware, "_nvidia_vram", lambda: pytest.fail("UMA must not probe"))
    assert hardware.launch_budget(uma) is None
    monkeypatch.setattr(hardware, "_nvidia_vram", lambda: None)
    assert hardware.launch_budget(CARD_CAPACITY) is None


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


def test_boot_writes_the_launch_window(hermes_home, monkeypatch):
    from hermes_cli.local_runtime import bootstrap, presets

    mdir, _gguf = _stage_27b(hermes_home, monkeypatch)
    monkeypatch.setattr(hardware, "probe_budget", lambda **kw: CARD_CAPACITY)
    _card(monkeypatch, 5.4)
    path = hermes_home / "presets.ini"
    assert bootstrap._generate_presets(mdir, path) == path
    assert presets.read_preset_decisions(path)[MODEL_ID].window == 144 * 1024


def test_boot_keeps_capacity_when_the_probe_fails(hermes_home, monkeypatch):
    """The probe is policy, not a prerequisite: a failure must never block a boot."""
    from hermes_cli.local_runtime import bootstrap, presets

    mdir, _gguf = _stage_27b(hermes_home, monkeypatch)
    monkeypatch.setattr(hardware, "probe_budget", lambda **kw: CARD_CAPACITY)

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
    monkeypatch.setattr(hardware, "probe_budget", lambda **kw: CARD_CAPACITY)
    _card(monkeypatch, others_gib)
    path = hermes_home / "presets.ini"
    bootstrap._generate_presets(mdir, path)
    return path


def test_idle_router_gets_windows_for_the_current_desktop(hermes_home, monkeypatch):
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


@pytest.mark.parametrize("status", ["loaded", "loading", "sleeping"])
def test_never_rewrites_while_a_model_holds_memory(hermes_home, monkeypatch, status):
    """The router unloads a loaded model whose launch flags change on reload."""
    from hermes_cli.local_runtime import bootstrap

    path = _booted(hermes_home, monkeypatch, 6.5)
    before = path.read_text(encoding="utf-8")
    router = _Router(path, {MODEL_ID: status})
    _card(monkeypatch, 2.0)
    assert bootstrap.refit_idle_presets(router) is False
    assert router.reloads == 0 and path.read_text(encoding="utf-8") == before


def test_small_changes_in_free_memory_do_not_replan(hermes_home, monkeypatch):
    from hermes_cli.local_runtime import bootstrap, presets

    path = _booted(hermes_home, monkeypatch, 5.4)
    router = _Router(path, {MODEL_ID: "unloaded"})
    assert bootstrap.refit_idle_presets(router) is False  # same reading as boot: same file
    monkeypatch.setattr(presets, "plan_presets", lambda *a, **k: pytest.fail("re-planned"))
    _card(monkeypatch, 5.5)
    assert bootstrap.refit_idle_presets(router) is False


# ── growth ───────────────────────────────────────────────────


def test_growth_counts_the_growing_model_as_free(hermes_home, monkeypatch):
    """The grown instance loads after the current one exits. Counting the model's own memory as
    held would veto every rung."""
    from hermes_cli.local_runtime import bootstrap, growth, presets

    mdir, gguf = _stage_27b(hermes_home, monkeypatch)
    monkeypatch.setattr(hardware, "probe_budget", lambda **kw: CARD_CAPACITY)
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


def test_growth_refuses_a_rung_other_programs_leave_no_room_for(hermes_home, monkeypatch):
    from hermes_cli.local_runtime import bootstrap, growth, presets

    _mdir, gguf = _stage_27b(hermes_home, monkeypatch)
    monkeypatch.setattr(hardware, "probe_budget", lambda **kw: CARD_CAPACITY)
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


# ── budget-source contract ───────────────────────────────────

_RUNTIME = Path(hardware.__file__).parent


@pytest.mark.parametrize("module", ["bootstrap.py", "growth.py", "presets.py"])
def test_launch_paths_read_capacity_and_narrow_only_through_launch_budget(module):
    """Launch decisions run through a server bounce, so the raw live budget (free memory with the
    outgoing server still resident) must never price them. Free memory enters only through
    ``launch_budget``, which counts the server's own memory as free and never moves weights."""
    tree = ast.parse((_RUNTIME / module).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", "")) == "probe_budget":
            planning = [k for k in node.keywords if k.arg == "planning"]
            assert planning and isinstance(planning[0].value, ast.Constant) and planning[0].value.value is True, (
                f"{module}:{node.lineno} calls probe_budget without planning=True")
