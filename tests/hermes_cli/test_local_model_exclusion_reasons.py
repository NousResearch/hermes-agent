"""A GGUF the managed runtime refuses must say WHY, everywhere the user meets it.

The router's listing is preset-only, so a refused file is simply absent from ``/models`` and
every door the user clicks (Local Models row, /model switch) used to answer the same generic
"not found in this provider's model listing". The launch decision already knew the reason; these
tests pin it to the surfaces.

No NVIDIA GPU needed: every refusal here is driven by an injected HardwareBudget and a synthetic
header, so the estimator's physics and the reader's parse errors are exercised directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    """The real router under the real web app, same as test_local_models_routes."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir(parents=True, exist_ok=True)
    from hermes_cli import web_server

    test_client = TestClient(web_server.app)
    test_client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    return test_client


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _stage(home: Path, name: str) -> Path:
    mdir = home / "models"
    mdir.mkdir(exist_ok=True)
    path = mdir / f"{name}.gguf"
    path.write_bytes(b"GGUF" + b"\x00" * 64)
    return path


def _preset_ini(runtimes_root: Path) -> Path:
    return runtimes_root / "presets.ini"


def _decisions(runtimes_root: Path) -> dict:
    from hermes_cli.local_runtime import presets

    return presets.read_preset_decisions(_preset_ini(runtimes_root))


# ── the reader's own words ───────────────────────────────────


def test_unreadable_header_records_a_reason_not_silence(hermes_home, monkeypatch):
    """A corrupt/truncated GGUF is the Discord case that took a diagnostics bundle to diagnose.
    It must come back as a refusal string the user can act on, not a dropped model."""
    import hermes_cli.local_runtime.presets as presets

    _stage(hermes_home, "Truncated-Model")
    monkeypatch.setattr(presets, "read_gguf_header", _truncated_header)
    entries = presets.generate_presets(hermes_home / "models", _budget(24, 64),
                                       _preset_ini(_runtimes_root(hermes_home)))
    by_id = {e.model_id: e for e in entries}
    refusal = by_id["Truncated-Model"].refusal
    assert refusal, "an unreadable header produced no recorded reason"
    # Actionable: names the cause and the remedy.
    assert "incomplete" in refusal or "re-download" in refusal
    assert by_id["Truncated-Model"].keys is None  # never autoloaded


def test_unknown_tensor_type_names_the_type_and_the_remedy(hermes_home):
    """`unknown ggml tensor type 39` (MXFP4) is a quant this engine can't read — the message must
    say which type and what to do, not just 'unknown'."""
    import struct

    from hermes_cli.local_runtime.gguf import read_gguf_header

    path = hermes_home / "mxfp4.gguf"
    # 1 tensor of ggml type 39 (MXFP4_MOE): unsupported by the shipped type table.
    # Tensor record: name (length-prefixed), n_dims, dims, type, offset.
    name = b"tok_embd.weight"
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1, 0)
                     + struct.pack("<Q", len(name)) + name
                     + struct.pack("<I", 1) + struct.pack("<Q", 4)
                     + struct.pack("<I", 39) + b"\x00" * 8)
    with pytest.raises(ValueError) as exc:
        read_gguf_header(path)
    message = str(exc.value)
    assert "39" in message                       # the actual type id
    assert "quant" in message or "engine" in message   # the remedy
    assert "path" not in message                 # no filesystem path leaks to the UI


def test_not_a_gguf_file_says_what_to_do(hermes_home):
    from hermes_cli.local_runtime.gguf import read_gguf_header

    path = hermes_home / "nope.gguf"
    path.write_bytes(b"NOPE" + b"\x00" * 32)
    with pytest.raises(ValueError) as exc:
        read_gguf_header(path)
    assert "not a GGUF" in str(exc.value)
    assert "re-download" in str(exc.value)


def test_truncated_tensor_table_is_a_stated_reason(hermes_home):
    """A short buffer used to escape as struct.error — the model vanished from every list with
    no reason recorded at all."""
    from hermes_cli.local_runtime.gguf import read_gguf_header

    path = hermes_home / "short.gguf"
    path.write_bytes(b"GGUF" + b"\x00" * 8)   # header claims more than the file holds
    with pytest.raises(ValueError) as exc:
        read_gguf_header(path)
    assert "incomplete" in str(exc.value)


# ── the physics refusal keeps its numbers ────────────────────


def test_physics_refusal_states_the_shortfall():
    from hermes_cli.local_runtime.estimator import LayerKind, ModelProfile, physics_check

    profile = ModelProfile("Too-Big", 60 << 30, 0, 262144, [(LayerKind.FULL, 512)] * 8)
    refusal = physics_check(profile, _budget(24, 16), 64 * 1024)
    assert refusal is not None
    # The two numbers that make it actionable: what's needed, what exists.
    assert "GiB" in refusal.message
    assert "needs" in refusal.message and "available" in refusal.message
    assert "quant" in refusal.message


# ── the Local Models row ─────────────────────────────────────


def test_status_row_carries_the_refusal(client, hermes_home, monkeypatch):
    """The row the user looks at says why the model is excluded."""
    import hermes_cli.local_runtime.presets as presets
    from hermes_cli.local_runtime import bootstrap
    from hermes_cli.local_runtime.binaries import runtimes_root

    _stage(hermes_home, "Unknown-Quant-Model")
    monkeypatch.setattr(presets, "read_gguf_header", _unknown_tensor_header)
    presets.generate_presets(bootstrap.models_dir(), _budget(24, 64),
                             _preset_ini(runtimes_root()))

    data = client.get("/api/local-models/status").json()
    row = data["models"][0]
    assert row["id"] == "Unknown-Quant-Model"
    assert row["servable"] is False
    assert "tensor type 39" in row["refusal"]


def test_status_row_marks_a_servable_model(client, hermes_home, monkeypatch):
    """No recorded refusal -> servable (the old shape, unchanged)."""
    _stage(hermes_home, "Plain-Model")
    data = client.get("/api/local-models/status").json()
    row = data["models"][0]
    assert row["servable"] is True
    assert row.get("refusal") is None


# ── the /model switch (the 400 the user actually hits) ────────


def test_switch_failure_carries_the_exclusion_reason(hermes_home, monkeypatch):
    """Desktop 'Use' -> /model switch: the 400 must append the launch decision's reason."""
    import hermes_cli.local_runtime.presets as presets
    from hermes_cli import models as _m
    from hermes_cli import models_validate
    from hermes_cli.local_runtime import bootstrap
    from hermes_cli.local_runtime.binaries import runtimes_root

    _stage(hermes_home, "Qwen3.6-14B-MXFP4_MOE")
    monkeypatch.setattr(presets, "read_gguf_header", _unknown_tensor_header)
    presets.generate_presets(bootstrap.models_dir(), _budget(24, 64),
                             _preset_ini(runtimes_root()))
    # The router serves everything it admitted — this one isn't among them.
    monkeypatch.setattr(_m, "fetch_api_models",
                        lambda *a, **kw: ["Some-Other-Model"])

    verdict = models_validate.validate_requested_model(
        "Qwen3.6-14B-MXFP4_MOE", "llamacpp", base_url="http://127.0.0.1:18434/v1")
    assert verdict["accepted"] is False
    message = verdict["message"]
    assert "not found in this provider's model listing" in message
    # The reason rides along — this is the whole point.
    assert "tensor type 39" in message
    assert "re-download" in message or "quant" in message


def test_cloud_providers_keep_the_generic_message(hermes_home, monkeypatch):
    """The appended reason is local-runtime-only: a cloud miss stays as it was."""
    from hermes_cli import models as _m
    from hermes_cli import models_validate

    monkeypatch.setattr(_m, "fetch_api_models", lambda *a, **kw: ["claude-opus-4.6"])
    verdict = models_validate.validate_requested_model("claude-sonnet-4.5", "anthropic")
    assert verdict["accepted"] is False
    assert "tensor type" not in verdict["message"]
    assert "excluded" not in verdict["message"]


def test_a_recorded_refusal_never_leaks_to_another_provider(hermes_home, monkeypatch):
    """The gate is per-provider: a local GGUF's exclusion must not decorate an unrelated
    provider's rejection (a user who has both a refused local model and a bad cloud id)."""
    import hermes_cli.local_runtime.presets as presets
    from hermes_cli import models as _m
    from hermes_cli import models_validate
    from hermes_cli.local_runtime import bootstrap
    from hermes_cli.local_runtime.binaries import runtimes_root

    _stage(hermes_home, "Refused-Local-Model")
    monkeypatch.setattr(presets, "read_gguf_header", _unknown_tensor_header)
    presets.generate_presets(bootstrap.models_dir(), _budget(24, 64),
                             _preset_ini(runtimes_root()))

    monkeypatch.setattr(_m, "fetch_api_models", lambda *a, **kw: ["claude-opus-4.6"])
    verdict = models_validate.validate_requested_model("claude-sonnet-4.5", "anthropic")
    assert "Refused-Local-Model" not in verdict["message"]
    assert "tensor type" not in verdict["message"]


# ── the ownership refusal ────────────────────────────────────


def test_stale_server_record_is_named_in_the_ownership_refusal(client, hermes_home, monkeypatch):
    """PID alive but no longer the recorded llama-server (the macOS PID-reuse case): the refusal
    says the record is stale, not 'another Hermes process owns this server'."""
    from fastapi import HTTPException

    from hermes_cli.local_runtime import recovery, supervisor
    from hermes_cli.web_routers import local_models

    monkeypatch.setattr(supervisor, "runtimes_root", lambda: hermes_home / "runtimes")
    state = {"pid": 12345, "create_time": 1.0, "owner_pid": 999, "owner_create_time": 1.0,
             "executable": str(hermes_home / "llama-server.exe"),
             "base_url": "http://127.0.0.1:18434/v1", "api_key": "k"}
    supervisor.state_path().parent.mkdir(parents=True, exist_ok=True)
    supervisor.state_path().write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr(local_models.bootstrap, "get_supervisor", lambda: None)
    monkeypatch.setattr(local_models, "_state_endpoint", lambda: None)
    # The recorded PID is alive but is NOT the recorded process.
    monkeypatch.setattr(recovery, "recorded_process", lambda s: None)
    monkeypatch.setattr(recovery, "_owner_is_dead", lambda s: True)
    monkeypatch.setattr("hermes_cli.local_runtime.recovery.stop_recorded_orphan", lambda: False)

    with pytest.raises(HTTPException) as exc:
        local_models._terminate_state_pid()
    assert exc.value.status_code == 409
    assert "stale" in exc.value.detail
    assert "12345" in exc.value.detail


# ── helpers ──────────────────────────────────────────────────


def _runtimes_root(home: Path) -> Path:
    from hermes_cli.local_runtime.binaries import runtimes_root

    return runtimes_root()


def _budget(vram_gib: int, ram_gib: int):
    from hermes_cli.local_runtime.estimator import HardwareBudget

    return HardwareBudget(vram_gib << 30, vram_gib << 30, ram_gib << 30)


def _truncated_header(path):
    raise ValueError("the file ends before its tensor table (unpack requires a buffer of "
                     "20 bytes) — the download is incomplete; delete and re-download it")


def _unknown_tensor_header(path):
    raise ValueError("unknown ggml tensor type 39 — this quant is too new for this build; "
                     "try a standard Q4_K_M/Q8_0 build, or update the local engine")
