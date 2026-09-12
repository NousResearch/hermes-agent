"""Contract tests for the local-models dashboard routes (Rollout 4).

Real FastAPI TestClient against the real router; the runtime pieces
underneath are exercised against temp HERMES_HOME (autouse fixture). Network
downloads are stubbed at the urllib boundary — never live."""

from __future__ import annotations

import io
import json
import struct
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli import web_server

    test_client = TestClient(web_server.app)
    # Same auth pattern as the git-route tests: present the session token.
    test_client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    return test_client


def test_local_models_routes_require_auth(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli import web_server

    unauth = TestClient(web_server.app)
    assert unauth.get("/api/local-models/status").status_code == 401


def _write_fake_gguf(path: Path, size: int = 1024) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF" + b"\x00" * size)


def _gguf_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _write_binary_gguf(path: Path, *, metadata: dict[str, int | str] | None = None,
                       tensors: list[tuple[str, list[int], int]] | None = None) -> None:
    """Real GGUF header/table fixture without allocating the tensor payload."""
    metadata = metadata or {}
    tensors = tensors or []
    body = [b"GGUF", struct.pack("<IQQ", 3, len(tensors), len(metadata))]
    for key, value in metadata.items():
        body.append(_gguf_string(key))
        if isinstance(value, str):
            body.extend((struct.pack("<I", 8), _gguf_string(value)))
        else:
            body.extend((struct.pack("<I", 10), struct.pack("<Q", value)))
    for name, dims, tensor_type in tensors:
        body.extend((
            _gguf_string(name), struct.pack("<I", len(dims)),
            struct.pack(f"<{len(dims)}Q", *dims), struct.pack("<I", tensor_type),
            struct.pack("<Q", 0),
        ))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(body))


def _write_split_ple_fixture(mdir: Path, *, name: str = "PLE-Q4") -> tuple[str, int]:
    """Qwen-shaped split: shard one has metadata only; the lookup table is in shard two."""
    from hermes_cli.local_runtime.gguf import AUTOMATIC_LAZY_TENSOR_THRESHOLD_BYTES

    first = mdir / f"{name}-00001-of-00002.gguf"
    second = mdir / f"{name}-00002-of-00002.gguf"
    lazy_bytes = AUTOMATIC_LAZY_TENSOR_THRESHOLD_BYTES + 1
    _write_binary_gguf(first, metadata={"general.architecture": "toy"})
    _write_binary_gguf(second, tensors=[("per_layer_token_embd.weight", [lazy_bytes], 24)])
    return name, lazy_bytes


# ── status ───────────────────────────────────────────────────


def test_status_shape_and_defaults(client):
    r = client.get("/api/local-models/status")
    assert r.status_code == 200
    data = r.json()
    # Contract: every key the pane's first paint needs, present and typed.
    assert isinstance(data["enabled"], bool)
    assert isinstance(data["tag"], str) and data["tag"].startswith("b")
    assert isinstance(data["runtime_installed"], bool)
    assert isinstance(data["server_running"], bool)
    assert isinstance(data["models"], list)


def test_status_lists_staged_models_with_labels(client, tmp_path):
    from hermes_cli.local_runtime.bootstrap import models_dir

    _write_fake_gguf(models_dir() / "Some-Model.gguf", size=2048)
    data = client.get("/api/local-models/status").json()
    ids = [m["id"] for m in data["models"]]
    assert "Some-Model" in ids
    row = data["models"][ids.index("Some-Model")]
    assert row["size_bytes"] > 0
    assert row["size_label"].endswith("GB")


def test_status_treats_a_truncated_gguf_as_unknown_instead_of_500(client):
    """Status polls must survive arbitrary third-party files, including struct.error from a short header."""
    from hermes_cli.local_runtime.bootstrap import models_dir

    path = models_dir() / "truncated.gguf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF")

    response = client.get("/api/local-models/status")

    assert response.status_code == 200
    row = next(model for model in response.json()["models"] if model["id"] == "truncated")
    assert row["lookup_placement"] == "unknown"
    assert client.post("/api/local-models/activate", json={"model_id": "truncated"}).status_code == 422


def test_status_inspects_a_complete_split_ple_before_the_server_starts(client, monkeypatch):
    """The model card gets the exact table placement from the real shard set, not a catalog guess."""
    from hermes_cli.local_runtime import binaries
    from hermes_cli.local_runtime.bootstrap import models_dir
    from hermes_cli.web_routers import local_models

    model_id, lazy_bytes = _write_split_ple_fixture(models_dir())
    monkeypatch.setattr(local_models, "_runtime_section", lambda: {"tag": "b10679"})
    monkeypatch.setattr(binaries, "installed_tags", lambda: ["b10679"])

    models = client.get("/api/local-models/status").json()["models"]
    row = next(model for model in models if model["id"] == model_id)

    assert row["lookup_placement"] == "disk-backed"
    assert row["lookup_table_bytes"] == lazy_bytes
    assert row["disk_backed_lookup_bytes"] == lazy_bytes
    assert row["disk_backed_lookup_label"].endswith("GB")


def test_status_requires_a_new_engine_for_an_oversized_split_ple(client, monkeypatch):
    from hermes_cli.local_runtime import binaries
    from hermes_cli.local_runtime.bootstrap import models_dir
    from hermes_cli.web_routers import local_models

    model_id, lazy_bytes = _write_split_ple_fixture(models_dir())
    monkeypatch.setattr(local_models, "_runtime_section", lambda: {"tag": "b10679"})
    monkeypatch.setattr(binaries, "installed_tags", lambda: ["b10678"])

    models = client.get("/api/local-models/status").json()["models"]
    row = next(model for model in models if model["id"] == model_id)

    assert row["lookup_placement"] == "requires-engine-update"
    assert row["lookup_table_bytes"] == lazy_bytes
    assert row["required_engine"] == "b10679"
    assert "disk_backed_lookup_bytes" not in row


@pytest.mark.parametrize("tag", ["b10679-old", "not-a-build-10679", "b" + "9" * 10_000])
def test_catalog_engine_gate_fails_closed_for_malformed_active_tags(tag, monkeypatch):
    from hermes_cli.local_runtime import binaries
    from hermes_cli.web_routers import local_models

    monkeypatch.setattr(binaries, "active_tag", lambda section: tag)

    assert local_models._engine_too_old("b10679") is True


def test_catalog_engine_gate_blocks_activate_advanced_and_gateway_routes(client, monkeypatch):
    """An API caller cannot bypass a known catalog architecture's min_engine through alternate UI paths."""
    from hermes_cli.local_runtime import bootstrap, catalog
    from hermes_cli.web_routers import local_models

    model_id = "Catalog-Architecture-Gate"
    _write_binary_gguf(bootstrap.models_dir() / f"{model_id}.gguf")
    entry = SimpleNamespace(display_name="Catalog-gated Model", min_engine="b10679")
    monkeypatch.setattr(catalog, "entry_for_model", lambda candidate: entry if candidate == model_id else None)
    monkeypatch.setattr(local_models, "_serving_engine_tag", lambda: "b10678")

    activate = client.post("/api/local-models/activate", json={"model_id": model_id})
    advanced = client.post("/api/local-models/advanced/plan", json={"model_id": model_id})
    gateway = client.post("/api/local-models/gateway-routes", json={
        "alias": "catalog-gated", "model_id": model_id, "mode": "agent",
    })

    for response in (activate, advanced, gateway):
        assert response.status_code == 409
        assert "b10679" in response.json()["detail"]


def test_status_uses_the_next_boot_tag_when_a_state_file_is_corrupt_or_dead(client, monkeypatch):
    """Atomic writes make a corrupt leftover non-evidence of a live old server, so bootstrap can heal it."""
    from hermes_cli.local_runtime import binaries
    from hermes_cli.local_runtime.bootstrap import models_dir
    from hermes_cli.local_runtime.supervisor import state_path
    from hermes_cli.web_routers import local_models

    model_id, _ = _write_split_ple_fixture(models_dir(), name="Corrupt-State-PLE")
    monkeypatch.setattr(local_models, "_runtime_section", lambda: {"tag": "b10679"})
    monkeypatch.setattr(binaries, "installed_tags", lambda: ["b10679"])
    state_path().parent.mkdir(parents=True, exist_ok=True)
    state_path().write_text("{not json", encoding="utf-8")

    row = next(model for model in client.get("/api/local-models/status").json()["models"]
               if model["id"] == model_id)
    assert row["lookup_placement"] == "disk-backed"
    assert row["disk_backed_lookup_bytes"] > 0


def test_proven_running_engine_wins_over_a_newer_configured_target(client, monkeypatch):
    """Installing b10679 must not relabel an already-running b10678 server as lazy-capable."""
    from hermes_cli.local_runtime import binaries
    from hermes_cli.local_runtime.bootstrap import models_dir
    from hermes_cli.web_routers import local_models

    model_id, _ = _write_split_ple_fixture(models_dir())
    monkeypatch.setattr(local_models, "_runtime_section", lambda: {"tag": "b10679"})
    monkeypatch.setattr(binaries, "installed_tags", lambda: ["b10679", "b10678"])
    monkeypatch.setattr(
        local_models, "_state_endpoint", lambda: {"base_url": "http://127.0.0.1:1/v1", "engine_tag": "b10678"}
    )

    row = next(model for model in client.get("/api/local-models/status").json()["models"] if model["id"] == model_id)

    assert row["lookup_placement"] == "requires-engine-update"
    assert row["required_engine"] == "b10679"


def test_exact_four_gib_lookup_is_resident_and_needs_no_engine_update(client, monkeypatch):
    """llama.cpp's automatic path is strict: exactly 4 GiB stays ordinary model memory."""
    from hermes_cli.local_runtime import binaries
    from hermes_cli.local_runtime.bootstrap import models_dir
    from hermes_cli.local_runtime.gguf import AUTOMATIC_LAZY_TENSOR_THRESHOLD_BYTES
    from hermes_cli.web_routers import local_models

    path = models_dir() / "exact-threshold.gguf"
    _write_binary_gguf(
        path, metadata={"general.architecture": "toy"},
        tensors=[("per_layer_token_embd.weight", [AUTOMATIC_LAZY_TENSOR_THRESHOLD_BYTES], 24)],
    )
    monkeypatch.setattr(local_models, "_runtime_section", lambda: {"tag": "b10678"})
    monkeypatch.setattr(binaries, "installed_tags", lambda: ["b10678"])
    monkeypatch.setattr(local_models, "_ensure_server", lambda *args, **kwargs: None)
    monkeypatch.setattr(local_models, "_assign_default", lambda *args, **kwargs: None)

    row = next(model for model in client.get("/api/local-models/status").json()["models"] if model["id"] == "exact-threshold")
    assert row["lookup_placement"] == "resident"
    assert row["lookup_table_bytes"] == AUTOMATIC_LAZY_TENSOR_THRESHOLD_BYTES
    assert "required_engine" not in row
    assert "disk_backed_lookup_bytes" not in row

    # No large lazy table means the old engine may proceed through activation.
    assert client.post("/api/local-models/activate", json={"model_id": "exact-threshold"}).status_code == 200


def test_activation_blocks_an_oversized_ple_on_an_old_engine(client, monkeypatch):
    from hermes_cli.local_runtime import binaries
    from hermes_cli.local_runtime.bootstrap import models_dir
    from hermes_cli.web_routers import local_models

    model_id, _ = _write_split_ple_fixture(models_dir())
    monkeypatch.setattr(local_models, "_runtime_section", lambda: {"tag": "b10679"})
    monkeypatch.setattr(binaries, "installed_tags", lambda: ["b10678"])
    monkeypatch.setattr(local_models, "_ensure_server", lambda *args, **kwargs: pytest.fail("must not start"))

    response = client.post("/api/local-models/activate", json={"model_id": model_id})

    assert response.status_code == 409
    assert "b10679" in response.json()["detail"]


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/api/local-models/advanced/plan", {"model_id": "PLE-Q4"}),
        ("/api/local-models/gateway-routes", {"alias": "ple", "model_id": "PLE-Q4", "mode": "agent"}),
    ],
)
def test_advanced_and_gateway_paths_block_an_oversized_ple_on_an_old_engine(
    client, monkeypatch, path, body
):
    from hermes_cli.local_runtime import binaries
    from hermes_cli.local_runtime.bootstrap import models_dir
    from hermes_cli.web_routers import local_models

    _write_split_ple_fixture(models_dir())
    monkeypatch.setattr(local_models, "_runtime_section", lambda: {"tag": "b10679"})
    monkeypatch.setattr(binaries, "installed_tags", lambda: ["b10678"])

    response = client.post(path, json=body)

    assert response.status_code == 409
    assert "b10679" in response.json()["detail"]


def test_status_reports_disk_backed_lookup_separately_from_spill(client, monkeypatch):
    """The API must not turn llama.cpp's mmap lookup path into ordinary host spill."""
    from hermes_cli.local_runtime import presets
    from hermes_cli.local_runtime.presets import PresetEntry
    from hermes_cli.web_routers import local_models

    model_id = "ple-mmap-model"
    lazy_bytes = 28_800_138_240
    monkeypatch.setattr(
        local_models, "_state_endpoint", lambda: {"base_url": "http://127.0.0.1:1/v1", "engine_tag": "b10679"}
    )
    monkeypatch.setattr(
        presets, "read_preset_decisions",
        lambda: {model_id: PresetEntry(model_id, 65536, spilled=False, lazy_table_bytes=lazy_bytes)},
    )

    def router_response(running, route, **kwargs):
        if route == "/models":
            return {"data": [{"id": model_id, "status": {"value": "loaded"}}]}
        assert route == f"/props?model={model_id}"
        return {"default_generation_settings": {"n_ctx": 65536}}

    monkeypatch.setattr(local_models, "_router_request", router_response)
    placement = client.get("/api/local-models/status").json()["placement"][model_id]
    assert placement["spilled"] is False
    assert placement["disk_backed_lookup_bytes"] == lazy_bytes
    assert placement["disk_backed_lookup_label"].endswith("GB")


def test_catalog_prices_ple_against_the_active_engine(client, monkeypatch):
    """A configured new tag cannot enable lazy pricing while an old build still boots."""
    from hermes_cli.local_runtime import binaries, catalog, hardware
    from hermes_cli.local_runtime.estimator import HardwareBudget
    from hermes_cli.web_routers import local_models

    gib = 1 << 30
    # 96 GiB discrete planning shape: PLE core weights fit, but total file bytes must spill.
    budget = HardwareBudget(usable_vram_bytes=96 * gib - int(96 * gib * 0.09),
                            total_device_bytes=96 * gib, ram_available_bytes=64 * gib)
    monkeypatch.setattr(hardware, "probe_budget", lambda **kwargs: budget)
    monkeypatch.setattr(catalog, "refresh_catalog_soon", lambda: None)
    monkeypatch.setattr(local_models, "_runtime_section", lambda: {"tag": "b10679"})

    monkeypatch.setattr(binaries, "installed_tags", lambda: ["b10678"])
    old_rows = client.get("/api/local-models/catalog").json()["models"]
    old = next(row for row in old_rows if row["id"] == "qwen3.8-flash-next")
    assert old["needs_engine"] is True
    assert old["spilled"] is True
    assert "disk_backed_lookup_bytes" not in old

    monkeypatch.setattr(binaries, "installed_tags", lambda: ["b10679"])
    new_rows = client.get("/api/local-models/catalog").json()["models"]
    new = next(row for row in new_rows if row["id"] == "qwen3.8-flash-next")
    assert new["needs_engine"] is False
    assert new["spilled"] is False
    assert new["disk_backed_lookup_bytes"] == 28_800_138_240
    assert "disk-backed lookup table" in new["fit_summary"]
    assert "fully on your GPU" not in new["quant_reason"]


def test_qwen_download_requires_the_lazy_lookup_engine(client, monkeypatch):
    """The install path must not stage Flash Next on a build that lacks its automatic PLE mmap path."""
    from hermes_cli.local_runtime import binaries
    from hermes_cli.web_routers import local_models

    monkeypatch.setattr(local_models, "_runtime_section", lambda: {"tag": "b10679"})
    monkeypatch.setattr(binaries, "installed_tags", lambda: ["b10678"])

    response = client.post("/api/local-models/download", json={"model_id": "qwen3.8-flash-next"})

    assert response.status_code == 409
    assert "b10679" in response.json()["detail"]


def test_exact_qwen_variant_download_cannot_bypass_the_lazy_lookup_engine(client, monkeypatch):
    """A future catalog quant picker posts an exact variant id, so it must retain the family gate."""
    from hermes_cli.local_runtime import binaries
    from hermes_cli.web_routers import local_models

    monkeypatch.setattr(local_models, "_runtime_section", lambda: {"tag": "b10679"})
    monkeypatch.setattr(binaries, "installed_tags", lambda: ["b10678"])

    response = client.post("/api/local-models/download", json={"model_id": "Qwen3.8-Flash-Next-UD-Q4_K_XL"})

    assert response.status_code == 409
    assert "b10679" in response.json()["detail"]


def test_advanced_plan_uses_the_active_engine_for_sideloaded_models(monkeypatch):
    """Sideloaded files have no catalog min-engine gate, so their parser must receive the boot tag."""
    from pathlib import Path

    from hermes_cli.local_runtime import binaries, bootstrap, estimator, gguf, hardware
    from hermes_cli.local_runtime.estimator import HardwareBudget, ModelProfile
    from hermes_cli.web_routers import local_models

    path = Path("Sideloaded.gguf")
    seen = {}
    profile = ModelProfile(
        name="Sideloaded", weights_bytes=12 << 30, embd_table_bytes=0,
        n_ctx_train=65536, layers=[], lazy_table_bytes=8 << 30,
    )

    def profile_from_header(_, *, engine_tag=None):
        seen["engine_tag"] = engine_tag
        return profile

    monkeypatch.setattr(bootstrap, "staged_models", lambda: [path])
    monkeypatch.setattr(gguf, "read_gguf_header", lambda _: object())
    # This test isolates launch-profile engine propagation; the synthetic path
    # is not a real GGUF for the separate placement-inspection guard.
    monkeypatch.setattr(local_models, "_lookup_inspection", lambda _: {"lookup_placement": "none"})
    monkeypatch.setattr(estimator, "profile_from_gguf", profile_from_header)
    monkeypatch.setattr(binaries, "installed_tags", lambda: ["b10678"])
    monkeypatch.setattr(local_models, "_runtime_section", lambda: {"tag": "b10679"})
    monkeypatch.setattr(
        hardware, "probe_budget",
        lambda **kwargs: HardwareBudget(usable_vram_bytes=6 << 30, total_device_bytes=6 << 30,
                                        ram_available_bytes=0))

    local_models._advanced_plan("Sideloaded", {})
    assert binaries.active_tag({"tag": "b10679"}) == "b10678"
    assert seen["engine_tag"] == "b10678"


# ── hardware ─────────────────────────────────────────────────


def test_hardware_plain_facts(client):
    data = client.get("/api/local-models/hardware").json()
    assert isinstance(data["uma"], bool)
    assert data["ram_total_bytes"] > 0
    assert data["vram_total_bytes"] >= 0
    # GPU fields are None-able (non-NVIDIA machines) but must exist.
    assert "gpu_name" in data and "gpu_util_percent" in data and "vram_used_bytes" in data


# ── catalog ──────────────────────────────────────────────────


def test_catalog_prices_every_entry_for_this_machine(client):
    data = client.get("/api/local-models/catalog").json()
    assert len(data["models"]) >= 3
    for row in data["models"]:
        # The three user questions, answered on every row:
        assert row["size_label"].endswith("GB")            # how big
        assert isinstance(row["fits"], bool)               # will it fit
        assert row["fit_summary"]                          # what shape
        if row["fits"]:
            assert row["start_window"] >= 1
            assert row["start_window_label"].endswith("K")
        else:
            assert "memory" in row["fit_summary"].lower()
        assert isinstance(row["downloaded"], bool)


def test_catalog_never_hides_unaffordable_models(client, monkeypatch):
    """Unaffordable entries stay visible with a plain reason — hiding them
    is how users conclude the feature is broken."""
    from hermes_cli.local_runtime.estimator import HardwareBudget

    tiny = HardwareBudget(usable_vram_bytes=1 << 30, total_device_bytes=1 << 30,
                          ram_available_bytes=1 << 30)
    monkeypatch.setattr("hermes_cli.local_runtime.hardware.probe_budget",
                        lambda **kw: tiny)
    data = client.get("/api/local-models/catalog").json()
    from hermes_cli.local_runtime.catalog import CATALOG

    assert len(data["models"]) == len(CATALOG)
    refused = [m for m in data["models"] if not m["fits"]]
    assert refused, "a 1 GiB machine must refuse the 20 GB models"
    for row in refused:
        assert row["fit_detail"] or row["fit_summary"]


# ── downloads ────────────────────────────────────────────────


def test_download_unknown_model_404s(client):
    r = client.post("/api/local-models/download", json={"model_id": "nope"})
    assert r.status_code == 404


def test_download_short_of_server_length_errors_and_cleans_up(client, monkeypatch):
    """Catalog sizes are advisory (upstream re-uploads may make them
    stale — a mismatch against the CATALOG must not fail a download).
    The server's own declared length is the only completeness check:
    fewer bytes than the server promised means a dropped connection, so
    the job errors and nothing is staged."""

    class FakeResponse(io.BytesIO):
        # Body is 17 bytes; the server promises 32 — a truncated stream.
        headers = {"Content-Length": "32"}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("urllib.request.urlopen",
                        lambda *a, **k: FakeResponse(b"not the real body"))

    # Pin a generous budget: variant selection prices against the machine
    # running the test, and a GPU-less CI runner honestly refuses every
    # build (409) — this test is about the download path, not selection.
    from hermes_cli.local_runtime.estimator import HardwareBudget

    budget = HardwareBudget(usable_vram_bytes=64 << 30,
                            total_device_bytes=64 << 30,
                            ram_available_bytes=64 << 30)
    monkeypatch.setattr("hermes_cli.local_runtime.hardware.probe_budget",
                        lambda **kw: budget)

    from hermes_cli.local_runtime.catalog import CATALOG

    entry_id = CATALOG[0].id
    r = client.post("/api/local-models/download", json={"model_id": entry_id})
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    assert job_id

    deadline = time.time() + 10
    status = None
    while time.time() < deadline:
        status = client.get(f"/api/local-models/jobs/{job_id}").json()
        if status["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert status is not None and status["status"] == "error"
    assert "bytes" in status["error"].lower()

    from hermes_cli.local_runtime.bootstrap import models_dir

    assert not (models_dir() / f"{entry_id}.gguf").exists()
    assert not (models_dir() / f"{entry_id}.part").exists()


def test_download_already_downloaded_short_circuits(client, monkeypatch):
    from hermes_cli.local_runtime.bootstrap import models_dir
    from hermes_cli.local_runtime.catalog import CATALOG, select_variant
    from hermes_cli.local_runtime.estimator import HardwareBudget

    # Pin the budget so the selected variant is deterministic in the test.
    budget = HardwareBudget(usable_vram_bytes=64 << 30, total_device_bytes=64 << 30,
                            ram_available_bytes=64 << 30)
    monkeypatch.setattr("hermes_cli.local_runtime.hardware.probe_budget",
                        lambda **kw: budget)
    choice = select_variant(CATALOG[0], budget)
    assert choice is not None
    _write_fake_gguf(models_dir() / choice.variant.files[0].local_name)
    r = client.post("/api/local-models/download", json={"model_id": CATALOG[0].id})
    assert r.status_code == 200
    assert r.json()["already_downloaded"] is True


def test_delete_model(client):
    from hermes_cli.local_runtime.bootstrap import models_dir

    _write_fake_gguf(models_dir() / "Doomed.gguf")
    assert client.delete("/api/local-models/models/Doomed").status_code == 200
    assert not (models_dir() / "Doomed.gguf").exists()
    assert client.delete("/api/local-models/models/Doomed").status_code == 404


def test_gateway_route_publish_and_unpublish_is_profile_config_only(client):
    """Publishing records a typed alias but never restarts a gateway from a settings request."""
    from hermes_cli.local_runtime.bootstrap import models_dir

    _write_fake_gguf(models_dir() / "Gateway-Model.gguf")
    created = client.post("/api/local-models/gateway-routes", json={
        "alias": "my-local", "model_id": "Gateway-Model", "mode": "raw"})
    assert created.status_code == 200
    assert created.json()["restart_required"] is True
    listed = client.get("/api/local-models/gateway-routes")
    assert listed.status_code == 200
    assert listed.json()["routes"] == [{"alias": "my-local", "model_id": "Gateway-Model", "mode": "raw"}]
    assert client.delete("/api/local-models/gateway-routes/my-local").status_code == 200
    assert client.get("/api/local-models/gateway-routes").json()["routes"] == []


def test_gateway_route_rejects_alias_collision_between_agent_and_raw(client):
    from hermes_cli.local_runtime.bootstrap import models_dir

    _write_fake_gguf(models_dir() / "Gateway-Model.gguf")
    assert client.post("/api/local-models/gateway-routes", json={
        "alias": "shared", "model_id": "Gateway-Model", "mode": "agent"}).status_code == 200
    duplicate = client.post("/api/local-models/gateway-routes", json={
        "alias": "shared", "model_id": "Gateway-Model", "mode": "raw"})
    assert duplicate.status_code == 409


# ── runtime install ──────────────────────────────────────────


def test_runtime_install_rejects_impossible_combo(client, monkeypatch):
    """Impossible platform/backend combos fail the POST itself with the
    resolver's honest message — not a background job that dies silently.
    (win-arm64-vulkan; the old cuda case became real upstream at ~b1036x.)"""
    monkeypatch.setattr(
        "hermes_cli.local_runtime.binaries._host_os_arch", lambda: ("win", "arm64"))
    r = client.post("/api/local-models/runtime/install", json={"backend": "vulkan"})
    assert r.status_code == 400
    assert "arm64" in r.json()["detail"]


def test_explicit_compatible_engine_install_becomes_the_boot_target(client, monkeypatch):
    """A model-card update click must persist b10679, not reinstall the user's old pinned tag."""
    from types import SimpleNamespace

    from hermes_cli import config as config_mod
    from hermes_cli.local_runtime import binaries, bootstrap

    monkeypatch.setattr(binaries, "resolve_assets", lambda tag, backend: SimpleNamespace(assets=["runtime.zip"]))
    monkeypatch.setattr(binaries, "ensure_runtime_installed", lambda *args, **kwargs: None)
    monkeypatch.setattr(binaries, "installed_tags", lambda: ["b10678"])
    monkeypatch.setattr(binaries, "prune_old_tags", lambda *args, **kwargs: None)
    monkeypatch.setattr(bootstrap, "get_supervisor", lambda: None)

    response = client.post("/api/local-models/runtime/install", json={"backend": "cpu", "tag": "b10679"})

    assert response.status_code == 200
    assert response.json()["tag"] == "b10679"
    job_id = response.json()["job_id"]
    deadline = time.time() + 10
    status = None
    while time.time() < deadline:
        status = client.get(f"/api/local-models/jobs/{job_id}").json()
        if status["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert status is not None and status["status"] == "done", status and status.get("error")
    assert config_mod.load_config()["local_runtime"]["tag"] == "b10679"

    invalid = client.post("/api/local-models/runtime/install", json={"backend": "cpu", "tag": "not-a-build"})
    assert invalid.status_code == 422


def test_engine_update_restarts_an_adopted_old_server(monkeypatch):
    """A user-clicked upgrade must switch a persisted server too, not only an in-process supervisor."""
    from hermes_cli.local_runtime import bootstrap
    from hermes_cli.web_routers import local_models

    job = {}
    monkeypatch.setattr(bootstrap, "get_supervisor", lambda: None)
    monkeypatch.setattr(
        local_models, "_state_endpoint", lambda: {"base_url": "http://127.0.0.1:1/v1", "engine_tag": "b10678"}
    )
    restarted = {}
    monkeypatch.setattr(bootstrap, "refresh_local_runtime", lambda: restarted.setdefault("yes", True))

    assert local_models._restart_on_new_tag(job, "b10679") is True
    assert job["phase"] == "restarting"
    assert restarted["yes"] is True


def test_job_poll_unknown_404s(client):
    assert client.get("/api/local-models/jobs/deadbeef").status_code == 404


def test_eject_without_supervisor_is_not_a_500(client, monkeypatch):
    """Eject on an ADOPTED server (no in-process supervisor — the shape
    every backend restart produces, since boot adopts the running server
    via the state file) must route through the persisted endpoint, not
    crash. Regression: _state_endpoint was only imported inside the
    status route, so eject raised NameError -> 500 for every adopted-
    server session."""
    monkeypatch.setattr(
        "hermes_cli.local_runtime.bootstrap.get_supervisor", lambda: None)
    # No running server either: the route must answer 409 (no server),
    # never a NameError 500.
    monkeypatch.setattr(
        "hermes_cli.web_routers.local_models._state_endpoint", lambda: None)
    r = client.post("/api/local-models/eject", json={"model_id": "anything"})
    assert r.status_code == 409, (r.status_code, r.text)


def test_download_tolerates_stale_catalog_size(client, monkeypatch):
    """Upstream re-uploads make catalog sizes stale; a download whose
    delivered bytes are self-consistent with the SERVER's declared length
    must succeed even when the catalog said something else. (This is the
    tolerance the sha removal was for — being out of date must not break
    downloads.)"""

    body = b"x" * 48  # server-consistent: Content-Length == body length

    class FakeResponse(io.BytesIO):
        headers = {"Content-Length": str(len(body))}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("urllib.request.urlopen",
                        lambda *a, **k: FakeResponse(body))

    from hermes_cli.local_runtime.estimator import HardwareBudget

    budget = HardwareBudget(usable_vram_bytes=64 << 30,
                            total_device_bytes=64 << 30,
                            ram_available_bytes=64 << 30)
    monkeypatch.setattr("hermes_cli.local_runtime.hardware.probe_budget",
                        lambda **kw: budget)
    # Keep the post-download server bounce out of this unit.
    monkeypatch.setattr(
        "hermes_cli.local_runtime.bootstrap.refresh_local_runtime",
        lambda: False)

    from hermes_cli.local_runtime.catalog import CATALOG

    # Catalog size for this entry is in the tens of GB — wildly stale
    # versus our 48-byte body. The download must still land.
    entry_id = CATALOG[0].id
    r = client.post("/api/local-models/download", json={"model_id": entry_id})
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    deadline = time.time() + 10
    status = None
    while time.time() < deadline:
        status = client.get(f"/api/local-models/jobs/{job_id}").json()
        if status["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert status is not None and status["status"] == "done", status.get("error")
