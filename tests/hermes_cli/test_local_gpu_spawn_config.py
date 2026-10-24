"""The managed server is pinned to the selected GPU at spawn (#134227).

Selection is one resolver (``hardware.selected_nvidia_card``); pricing consumes
it (test_local_gpu_selection.py) and so must the spawn, or the catalog prices a
card the engine never touches. The pin rides the child environment —
backend-specific ``*_VISIBLE_DEVICES`` — resolved at spawn time from the live
smi query, so a card removed between sessions pins nothing instead of a wrong
card. ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` accompanies every CUDA pin: CUDA's
default enumeration is fastest-first, not smi/PCI order.

Simulated hardware and a fake spawner; no physical GPU is touched.
"""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from hermes_cli.local_runtime import supervisor

UUID0 = "GPU-00000000-0000-0000-0000-000000000000"
UUID1 = "GPU-11111111-0000-0000-0000-000000000001"

CARDS = [
    {"index": 0, "uuid": UUID0, "id": UUID0, "gpu_name": "NVIDIA GeForce RTX 2060", "total_bytes": 6 << 30,
     "free_bytes": 3 << 30, "used_bytes": 3 << 30, "gpu_util_percent": 6, "gpu_pci_id": None},
    {"index": 1, "uuid": UUID1, "id": UUID1, "gpu_name": "NVIDIA GeForce RTX 3070", "total_bytes": 8 << 30,
     "free_bytes": 8 << 30, "used_bytes": 0, "gpu_util_percent": 0, "gpu_pci_id": None},
]


def _spawnable(tmp_path, monkeypatch, cards, selection, backend="cuda"):
    """A supervisor whose spawn is captured instead of executed."""
    from hermes_cli.local_runtime import hardware

    monkeypatch.setattr(hardware, "nvidia_cards", lambda: list(cards))
    monkeypatch.setattr(hardware, "accelerator_cards", lambda **kw: [dict(c, name=f"Vulkan{c['index']}") for c in cards])
    monkeypatch.setattr(hardware, "_configured_gpu_selection", lambda: selection)
    monkeypatch.setattr(supervisor, "runtimes_root", lambda: tmp_path / "runtime")
    monkeypatch.setattr(supervisor, "_direct_io_args", lambda exe: [])
    spawns = []
    monkeypatch.setattr(supervisor, "spawn_server", lambda argv, **kwargs: (
        spawns.append({"argv": argv, **kwargs}) or SimpleNamespace(pid=123, poll=lambda: 0), None,
    ))
    return spawns


def _sup(tmp_path):
    return supervisor.LlamaServerSupervisor(tmp_path / "engine" / "llama-server",
                                            tmp_path / "models", port=19002, backend="cuda")


def _child_env(spawns):
    assert len(spawns) == 1
    return spawns[0]["env"]


def test_spawn_pins_selected_card_on_cuda(monkeypatch, tmp_path):
    spawns = _spawnable(tmp_path, monkeypatch, CARDS, "1")
    sup = _sup(tmp_path)
    monkeypatch.setattr(sup, "_write_state", lambda: None)
    try:
        sup._spawn()
    finally:
        sup.stop()
    env = _child_env(spawns)
    assert env["CUDA_VISIBLE_DEVICES"] == UUID1


def test_spawn_auto_pins_the_card_it_prices(monkeypatch, tmp_path):
    spawns = _spawnable(tmp_path, monkeypatch, CARDS, "auto")
    sup = _sup(tmp_path)
    monkeypatch.setattr(sup, "_write_state", lambda: None)
    try:
        sup._spawn()
    finally:
        sup.stop()
    env = _child_env(spawns)
    assert env["CUDA_VISIBLE_DEVICES"] == UUID1


def test_spawn_refuses_probe_miss_for_explicit_choice(monkeypatch, tmp_path):
    spawns = _spawnable(tmp_path, monkeypatch, [], "1")
    sup = _sup(tmp_path)
    with pytest.raises(ValueError, match="unavailable"):
        sup._spawn()
    assert spawns == []


def test_spawn_uses_vulkan_variable_on_vulkan_backend(monkeypatch, tmp_path):
    spawns = _spawnable(tmp_path, monkeypatch, CARDS, "1", backend="vulkan")
    sup = supervisor.LlamaServerSupervisor(tmp_path / "engine" / "llama-server",
                                           tmp_path / "models", port=19002, backend="vulkan")
    monkeypatch.setattr(sup, "_write_state", lambda: None)
    try:
        sup._spawn()
    finally:
        sup.stop()
    env = _child_env(spawns)
    assert "--device" in spawns[0]["argv"]
    assert spawns[0]["argv"][-4:] == ["--device", "Vulkan1", "--split-mode", "none"]
    assert "GGML_VK_VISIBLE_DEVICES" not in env
    assert "CUDA_VISIBLE_DEVICES" not in env


def test_spawn_pin_applies_on_restart_and_does_not_borrow_new_profiles_config(monkeypatch, tmp_path):
    from hermes_cli.local_runtime import hardware
    spawns = _spawnable(tmp_path, monkeypatch, CARDS, "1")
    sup = _sup(tmp_path)
    monkeypatch.setattr(sup, "_write_state", lambda: None)
    try:
        sup._spawn()
        monkeypatch.setattr(hardware, "_configured_gpu_selection", lambda: "0")
        sup._spawn()
    finally:
        sup.stop()
    assert len(spawns) == 2
    assert all(s["env"]["CUDA_VISIBLE_DEVICES"] == UUID1 for s in spawns)


# ── the config knob ──────────────────────────────────────────


def test_local_runtime_gpu_default_is_auto(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli import config as config_mod
    from hermes_cli.local_runtime import hardware

    assert config_mod.DEFAULT_CONFIG["local_runtime"]["gpu"] == "auto"
    # The resolver reads the same key from a real (empty) config.
    assert hardware._configured_gpu_selection() == "auto"


def test_gpu_setting_round_trips_through_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli import config as config_mod

    config = config_mod.load_config()
    config.setdefault("local_runtime", {})["gpu"] = "1"
    config_mod.save_config(config)
    reread = config_mod.load_config_readonly()
    assert reread["local_runtime"]["gpu"] == "1"
    assert config_mod.validate_config_structure(reread) == []


def test_bogus_gpu_pin_still_validates(tmp_path, monkeypatch):
    """Validation is structural: a numeric-or-auto pin is honoured at runtime by
    falling back to auto (hardware tests); config must accept the literal."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli import config as config_mod

    config = config_mod.load_config()
    config.setdefault("local_runtime", {})["gpu"] = "banana"
    config_mod.save_config(config)
    assert config_mod.validate_config_structure(config_mod.load_config_readonly()) == []


# ── the /hardware endpoint exposes the card list + selection ──


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli import web_server
    from hermes_cli.local_runtime import catalog

    monkeypatch.setattr(catalog, "refresh_catalog_soon", lambda: None)
    test_client = TestClient(web_server.app)
    test_client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    return test_client


def test_hardware_endpoint_lists_cards_and_selection(client, monkeypatch):
    from hermes_cli.local_runtime import hardware
    from hermes_cli.web_routers import local_models as router_mod

    cards = [dict(c) for c in CARDS]
    monkeypatch.setattr(hardware, "nvidia_cards", lambda: cards)
    monkeypatch.setattr(hardware, "_configured_gpu_selection", lambda: "auto")
    monkeypatch.setattr(router_mod.hardware, "nvidia_cards", lambda: cards)
    monkeypatch.setattr(router_mod.hardware, "_configured_gpu_selection", lambda: "auto")
    monkeypatch.setattr(router_mod.hardware, "probe_budget", lambda planning=False: SimpleNamespace(
        uma=False, total_device_bytes=8 << 30, usable_vram_bytes=6 << 30))
    monkeypatch.setattr(router_mod.hardware, "_ram_bytes", lambda: (64 << 30, 32 << 30))
    monkeypatch.setattr(router_mod, "_nvidia_smi_facts", dict)

    r = client.get("/api/local-models/hardware")
    assert r.status_code == 200
    data = r.json()
    assert [c["index"] for c in data["gpus"]] == [0, 1]
    assert data["gpus"][1]["name"] == "NVIDIA GeForce RTX 3070"
    assert data["gpu"] == "auto"
    assert data["gpu_selected"] == 1  # auto = largest card


def test_gpu_selection_endpoint_rejects_garbage(client):
    r = client.post("/api/local-models/gpu", json={"gpu": "banana"})
    assert r.status_code == 422


def test_gpu_selection_endpoint_writes_config_and_bounces(client, monkeypatch):
    from hermes_cli.local_runtime import hardware
    from hermes_cli.web_routers import local_models as router_mod

    monkeypatch.setattr(router_mod.hardware, "nvidia_cards", lambda: [dict(c) for c in CARDS])
    monkeypatch.setattr(hardware, "nvidia_cards", lambda: [dict(c) for c in CARDS])
    monkeypatch.setattr(hardware, "_configured_gpu_selection", lambda: "1")
    monkeypatch.setattr(router_mod.hardware, "_configured_gpu_selection", lambda: "1")
    bounces = []
    monkeypatch.setattr(router_mod.bootstrap, "get_supervisor", lambda: SimpleNamespace(is_idle=lambda: True))
    monkeypatch.setattr(router_mod.bootstrap, "refresh_local_runtime", lambda: bounces.append(1) or True)
    r = client.post("/api/local-models/gpu", json={"gpu": "1"})
    assert r.status_code == 200
    assert r.json()["gpu"] == UUID1
    assert bounces == [1]
    # Durable: the key landed in the on-disk config.
    from hermes_cli import config as config_mod
    assert config_mod.load_config_readonly()["local_runtime"]["gpu"] == UUID1
