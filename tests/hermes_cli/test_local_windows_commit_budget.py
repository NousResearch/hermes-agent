"""Windows unified memory: the model and its prompt cache must fit in commit.

Windows charges every GPU allocation to commit (RAM Windows can see plus the page file). On an
RTX Spark the GPU carve-out is invisible to Windows, so the allocator pool overstates what Windows
will hand out. Measured on a 64 GB unit at the shipped 32/32 split: 61.2 GiB commit limit, 48.4 GiB
free at idle; the 27B at a 256K window took 37.5 GiB of commit, and llama-server's default 8 GiB
prompt cache brought free commit down to ~1.4 GiB with a busy desktop.

Free commit is a live reading, so it only narrows the launch window and sizes the cache. Capacity,
the physics check and the catalog's recommendation stay on the pool: a busy desktop at boot must
not refuse a model that fits.
"""

from types import SimpleNamespace

import pytest

from hermes_cli.local_runtime import hardware as hw
from hermes_cli.local_runtime import presets
from hermes_cli.local_runtime.estimator import HardwareBudget, ModelProfile

G = 1 << 30


def _stat(ram_visible, commit_limit, commit_free):
    return SimpleNamespace(ullTotalPhys=int(ram_visible * G), ullAvailPhys=int(ram_visible * G / 2),
                           ullTotalPageFile=int(commit_limit * G), ullAvailPageFile=int(commit_free * G))


@pytest.fixture
def spark(monkeypatch):
    """An RTX Spark at 32/32: Windows sees 30.2 GiB, the CUDA pool is 50.6 GiB."""
    state = {"stat": _stat(30.2, 61.2, 48.4)}
    monkeypatch.setattr(hw, "_windows_memory_status", lambda: state["stat"])
    monkeypatch.setattr(hw, "_unified_pool_bytes", lambda smi_total, ram_total: int(50.6 * G))
    monkeypatch.setattr(hw, "_nvidia_vram", lambda: (32 * G, 31 * G, "NVIDIA RTX Spark N1X", None))
    monkeypatch.setattr(hw, "_ram_bytes", lambda: (int(30.2 * G), int(15 * G)))
    return state


def test_commit_room_leaves_a_desktop_margin(spark):
    assert hw._windows_commit_room() == pytest.approx((48.4 - 2.718) * G, rel=1e-3)  # 9% of visible RAM


def test_the_managed_server_counts_as_free_before_its_relaunch(spark):
    before = hw._windows_commit_room()
    spark["stat"] = _stat(30.2, 61.2, 48.4 - 37.5)
    assert hw._windows_commit_room(own_bytes=int(37.5 * G)) == pytest.approx(before, rel=1e-6)


def test_off_windows_has_no_commit_room(monkeypatch):
    monkeypatch.setattr(hw, "_windows_memory_status", lambda: None)
    assert hw._windows_commit_room() is None
    capacity = HardwareBudget(int(16 * G), int(50 * G), 0, uma=True)
    assert hw.launch_budget(capacity) is None


def test_capacity_stays_on_the_pool_however_busy_the_desktop(spark):
    idle = hw.probe_budget(planning=True)
    spark["stat"] = _stat(30.2, 61.2, 12.0)
    busy = hw.probe_budget(planning=True)
    assert busy.usable_vram_bytes == idle.usable_vram_bytes
    assert busy.commit_room_bytes is None


def test_launch_budget_narrows_to_free_commit(spark):
    capacity = hw.probe_budget(planning=True)
    spark["stat"] = _stat(22.2, 44.4, 31.6)  # 40/24 split: 8 GiB less RAM, a smaller page file
    live = hw.launch_budget(capacity)
    assert live.commit_room_bytes == hw._windows_commit_room()
    assert live.usable_vram_bytes == int(live.commit_room_bytes * 0.8) < capacity.usable_vram_bytes


def test_pool_binds_when_commit_is_plentiful(spark):
    spark["stat"] = _stat(94, 120, 106)  # 128 GB unit at 32/96
    capacity = hw.probe_budget(planning=True)
    live = hw.launch_budget(capacity)
    assert live.usable_vram_bytes == capacity.usable_vram_bytes == int(50.6 * G * 0.8)


def test_prompt_cache_takes_what_the_models_leave():
    capacity = HardwareBudget(int(24 * G), 0, 0, uma=True)
    live = HardwareBudget(0, 0, 0, uma=True, commit_room_bytes=int(45.7 * G))
    # One 20 GiB model fits the capacity once: 45.7 - 25 = 20.7 GiB left, capped at the default.
    assert hw.prompt_cache_bytes(capacity, live, 20 * G) == 8 * G
    # 31 GiB estimated -> ~38.75 GiB charged: ~6.9 GiB left for the cache.
    assert hw.prompt_cache_bytes(capacity, live, 31 * G) == pytest.approx(45.7 * G - 31 * G / 0.8, rel=1e-6)
    assert hw.prompt_cache_bytes(capacity, live, int(36.5 * G)) == 0  # under 512 MiB: off
    assert hw.prompt_cache_bytes(capacity, None, 10 * G) is None  # engine default without a live reading


def test_co_resident_models_share_the_cache_room():
    capacity = HardwareBudget(int(30 * G), 0, 0, uma=True)
    live = HardwareBudget(0, 0, 0, uma=True, commit_room_bytes=int(45.7 * G))
    # Two 14 GiB models fit the capacity: each charges 17.5 GiB, and 10.7 GiB remain for two caches.
    each = hw.prompt_cache_bytes(capacity, live, 14 * G)
    assert each == pytest.approx((45.7 - 2 * 17.5) * G / 2, rel=1e-6)
    assert 2 * (each + 17.5 * G) <= 45.7 * G


def _staged(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    gguf = tmp_path / "model.gguf"
    gguf.touch()
    monkeypatch.setattr(presets, "read_gguf_header", lambda p: SimpleNamespace(path=p, sampling_defaults={}))
    monkeypatch.setattr(presets, "profile_from_gguf", lambda h: ModelProfile(
        name="model", weights_bytes=8 * G, embd_table_bytes=0, n_ctx_train=65536, layers=[]))
    return gguf


def test_preset_writes_the_cache_size_from_live_commit(tmp_path, monkeypatch):
    gguf = _staged(tmp_path, monkeypatch)
    capacity = HardwareBudget(int(16 * G), int(50 * G), 0, uma=True)
    live = HardwareBudget(int(16 * G), int(50 * G), 0, uma=True, commit_room_bytes=int(20 * G))
    entry = presets.preset_for_model(gguf, capacity, set(), live=live)
    assert 0 < int(entry.keys["cache-ram"]) <= 8192
    assert "cache-ram" not in presets.preset_for_model(gguf, capacity, set()).keys


def test_a_busy_desktop_narrows_but_never_refuses(tmp_path, monkeypatch):
    gguf = _staged(tmp_path, monkeypatch)
    capacity = HardwareBudget(int(16 * G), int(50 * G), 0, uma=True)
    live = HardwareBudget(int(1 * G), int(50 * G), 0, uma=True, commit_room_bytes=int(1.25 * G))
    entry = presets.preset_for_model(gguf, capacity, set(), live=live)
    assert entry.refusal is None and entry.window > 0
    assert entry.keys["cache-ram"] == "0"
