"""GPU selection for the managed local runtime (#134227).

A machine can hold a display card plus a dedicated compute card (the reporter's
topology: RTX 2060 driving the monitors, RTX 3070 bought for models). The
budget probe read only the FIRST nvidia-smi row and the spawn pinned nothing,
so pricing followed the display card while the engine placed layers wherever
the backend's default order said — two answers that can disagree.

Selection must be ONE resolver consumed by BOTH sides: the card the budget
prices is the card the child is pinned to. ``auto`` prices the largest card
(llama.cpp's own placement rule among discrete devices) and pins nothing, so
multi-card split behaviour is unchanged until the user asks for a card.

Simulated nvidia-smi output feeds the production parser; these are functional
regression tests, not physical-device validation.
"""

from types import SimpleNamespace

from hermes_cli.local_runtime import hardware as hw

GIB = 1 << 30
MIB = 1 << 20

# The reporter's pair from #134227: display card first (smi/PCI order), compute card second.
ROWS_2060_3070 = [
    # memory.total, memory.free, name, pci.device_id, memory.used, utilisation
    "6144, 3040, NVIDIA GeForce RTX 2060, 0x1F0810DE, 3104, 6",
    "8192, 8165, NVIDIA GeForce RTX 3070, 0x248410DE, 27, 0",
]
NAME_2060 = "NVIDIA GeForce RTX 2060"
NAME_3070 = "NVIDIA GeForce RTX 3070"


def _smi(monkeypatch, rows):
    """Feed the real parser: the shared cached query spawns nvidia-smi with these rows."""
    monkeypatch.setattr(hw, "_nvidia_smi_path", lambda: "nvidia-smi")
    monkeypatch.setattr(hw, "_gpu_query_cache", None)

    def run(argv, **kwargs):
        assert argv[1].startswith("--query-gpu=memory.total")
        return SimpleNamespace(returncode=0, stdout="\n".join(rows))

    monkeypatch.setattr(hw.subprocess, "run", run)


def _select(monkeypatch, value):
    monkeypatch.setattr(hw, "_configured_gpu_selection", lambda: value)


def _no_env_pin(monkeypatch):
    for key in ("CUDA_VISIBLE_DEVICES", "GGML_VK_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        monkeypatch.delenv(key, raising=False)


# ── the shared query exposes every card ──────────────────────


def test_shared_query_parses_every_card_row(monkeypatch):
    _smi(monkeypatch, ROWS_2060_3070)
    query = hw._cached_nvidia_gpu_query()
    cards = query["cards"]
    assert len(cards) == 2
    assert cards[0]["gpu_name"] == NAME_2060 and cards[0]["total_bytes"] == 6144 * MIB
    assert cards[1]["gpu_name"] == NAME_3070 and cards[1]["total_bytes"] == 8192 * MIB
    assert cards[1]["free_bytes"] == 8165 * MIB
    # The single-card top-level view stays the FIRST row — pre-selection callers
    # (statusbar facts) keep their historical answer byte-for-byte.
    assert query["gpu_name"] == NAME_2060
    assert query["total_bytes"] == 6144 * MIB


def test_nvidia_cards_attaches_smi_index(monkeypatch):
    _smi(monkeypatch, ROWS_2060_3070)
    cards = hw.nvidia_cards()
    assert [(c["index"], c["gpu_name"]) for c in cards] == [(0, NAME_2060), (1, NAME_3070)]


def test_single_card_query_keeps_legacy_shape(monkeypatch):
    _smi(monkeypatch, ROWS_2060_3070[:1])
    query = hw._cached_nvidia_gpu_query()
    assert [c["index"] for c in query["cards"]] == [0]
    cards = hw.nvidia_cards()
    assert len(cards) == 1 and cards[0]["gpu_name"] == NAME_2060


def test_no_smi_means_no_cards(monkeypatch):
    monkeypatch.setattr(hw, "_nvidia_smi_path", lambda: None)
    monkeypatch.setattr(hw, "_gpu_query_cache", None)
    assert hw.nvidia_cards() == []
    assert hw.selected_nvidia_card() is None


# ── selection: one resolver ───────────────────────────────────


def test_auto_selects_the_largest_card(monkeypatch):
    _smi(monkeypatch, ROWS_2060_3070)
    _select(monkeypatch, "auto")
    _no_env_pin(monkeypatch)
    card = hw.selected_nvidia_card()
    assert card is not None and card["gpu_name"] == NAME_3070 and card["index"] == 1


def test_auto_tie_prefers_the_first_smi_row(monkeypatch):
    """Equal cards must resolve deterministically to smi row 0 — the N1X identity
    tests feed two equal-size rows and expect the FIRST row's identity."""
    _smi(monkeypatch, [
        "16384, 16000, NVIDIA GeForce RTX 5070 Ti, 0x2E0010DE, 384, 1",
        "16384, 16000, NVIDIA GeForce RTX 5060 Ti, 0x2E0310DE, 384, 1",
    ])
    _select(monkeypatch, "auto")
    _no_env_pin(monkeypatch)
    assert hw.selected_nvidia_card()["gpu_name"] == "NVIDIA GeForce RTX 5070 Ti"


def test_index_pin_selects_that_card(monkeypatch):
    _smi(monkeypatch, ROWS_2060_3070)
    _no_env_pin(monkeypatch)
    _select(monkeypatch, "1")
    assert hw.selected_nvidia_card()["gpu_name"] == NAME_3070
    _select(monkeypatch, "0")
    assert hw.selected_nvidia_card()["gpu_name"] == NAME_2060


def test_out_of_range_pin_falls_back_to_largest(monkeypatch):
    _smi(monkeypatch, ROWS_2060_3070)
    _select(monkeypatch, "3")
    _no_env_pin(monkeypatch)
    # A stale pin (card removed) must never block budgeting — fall back to auto.
    assert hw.selected_nvidia_card()["gpu_name"] == NAME_3070


def test_garbage_pin_falls_back_to_largest(monkeypatch):
    _smi(monkeypatch, ROWS_2060_3070)
    _select(monkeypatch, "not-a-gpu")
    _no_env_pin(monkeypatch)
    assert hw.selected_nvidia_card()["gpu_name"] == NAME_3070


def test_user_env_pin_is_respected_under_auto(monkeypatch):
    """A user who exports CUDA_VISIBLE_DEVICES=1 before Hermes has already chosen;
    pricing must follow that card or it prices a card the engine cannot see."""
    _smi(monkeypatch, ROWS_2060_3070)
    _select(monkeypatch, "auto")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert hw.selected_nvidia_card()["gpu_name"] == NAME_2060
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert hw.selected_nvidia_card()["gpu_name"] == NAME_3070


def test_multi_card_env_pin_reads_as_auto(monkeypatch):
    """CVD="0,1" leaves both cards visible — the engine places layers itself, so
    pricing follows the auto rule (largest), not the first listed index."""
    _smi(monkeypatch, ROWS_2060_3070)
    _select(monkeypatch, "auto")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    assert hw.selected_nvidia_card()["gpu_name"] == NAME_3070


def test_config_pin_overrides_env_pin(monkeypatch):
    _smi(monkeypatch, ROWS_2060_3070)
    _select(monkeypatch, "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert hw.selected_nvidia_card()["gpu_name"] == NAME_3070


def test_single_card_machine_pins_nothing(monkeypatch):
    _smi(monkeypatch, ROWS_2060_3070[:1])
    _select(monkeypatch, "auto")
    _no_env_pin(monkeypatch)
    card = hw.selected_nvidia_card()
    assert card is not None and card["index"] == 0
    assert hw.gpu_pin_env(backend="cuda") == {}


# ── pricing follows the selection ─────────────────────────────


def _budget(monkeypatch, rows, selection):
    _smi(monkeypatch, rows)
    _select(monkeypatch, selection)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("GGML_VK_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(hw, "_device_pool_view", lambda: (0, False))
    monkeypatch.setattr(hw, "_ram_bytes", lambda: (64 * GIB, 32 * GIB))
    return hw.probe_budget(planning=True)


def test_probe_budget_prices_the_pinned_card(monkeypatch):
    budget3070 = _budget(monkeypatch, ROWS_2060_3070, "1")
    assert budget3070.gpu_name == NAME_3070
    assert budget3070.total_device_bytes == 8192 * MIB
    budget2060 = _budget(monkeypatch, ROWS_2060_3070, "0")
    assert budget2060.gpu_name == NAME_2060
    assert budget2060.total_device_bytes == 6144 * MIB


def test_probe_budget_auto_prices_the_largest_card(monkeypatch):
    budget = _budget(monkeypatch, ROWS_2060_3070, "auto")
    assert budget.gpu_name == NAME_3070
    assert budget.total_device_bytes == 8192 * MIB


def test_single_card_budget_is_unchanged(monkeypatch):
    """One card, no selection: byte-identical to the pre-selection answer."""
    budget = _budget(monkeypatch, ROWS_2060_3070[:1], "auto")
    assert budget.gpu_name == NAME_2060
    assert budget.total_device_bytes == 6144 * MIB
    margin = max(hw._MARGIN_FLOOR, int(6144 * MIB * hw._MARGIN_FRACTION))
    assert budget.usable_vram_bytes == 6144 * MIB - margin


def test_launch_budget_reads_the_selected_cards_free(monkeypatch):
    _smi(monkeypatch, ROWS_2060_3070)
    _select(monkeypatch, "1")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("GGML_VK_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(hw, "_ram_bytes", lambda: (64 * GIB, 32 * GIB))
    monkeypatch.setattr(hw, "_device_pool_view", lambda: (0, False))
    capacity = hw.probe_budget(planning=True)
    live = hw.launch_budget(capacity)
    assert live is not None
    # usable is capped by what the 3070 has FREE (8165 MiB) less launch headroom.
    assert live.usable_vram_bytes <= 8165 * MIB - hw._LAUNCH_HEADROOM + 1


# ── the spawn-side pin ────────────────────────────────────────


def test_gpu_pin_env_pins_index_and_bus_order(monkeypatch):
    _smi(monkeypatch, ROWS_2060_3070)
    _select(monkeypatch, "1")
    _no_env_pin(monkeypatch)
    env = hw.gpu_pin_env(backend="cuda")
    # CUDA enumerates fastest-first by default, NOT in smi/PCI order — pinning by
    # smi index without fixing the order silently selects the wrong card.
    assert env == {"CUDA_VISIBLE_DEVICES": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}


def test_gpu_pin_env_auto_is_empty(monkeypatch):
    _smi(monkeypatch, ROWS_2060_3070)
    _select(monkeypatch, "auto")
    _no_env_pin(monkeypatch)
    assert hw.gpu_pin_env(backend="cuda") == {}


def test_gpu_pin_env_vulkan_and_hip_use_their_variables(monkeypatch):
    _smi(monkeypatch, ROWS_2060_3070)
    _select(monkeypatch, "1")
    _no_env_pin(monkeypatch)
    assert hw.gpu_pin_env(backend="vulkan") == {"GGML_VK_VISIBLE_DEVICES": "1"}
    assert hw.gpu_pin_env(backend="hip") == {"HIP_VISIBLE_DEVICES": "1"}


def test_gpu_pin_env_survives_a_missing_query(monkeypatch):
    monkeypatch.setattr(hw, "_nvidia_smi_path", lambda: None)
    monkeypatch.setattr(hw, "_gpu_query_cache", None)
    _select(monkeypatch, "1")
    _no_env_pin(monkeypatch)
    assert hw.gpu_pin_env(backend="cuda") == {}
