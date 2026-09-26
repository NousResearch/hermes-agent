"""local_runtime.executable_path / model_dirs / model_overrides, each observed where it lands:
the supervisor's binary, the staged list, the preset INI."""

from __future__ import annotations

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _stage(mdir, name):
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / f"{name}.gguf").write_bytes(b"GGUF" + b"\x00" * 64)


def _tiny_presets(monkeypatch):
    import hermes_cli.local_runtime.presets as presets
    from hermes_cli.local_runtime.estimator import HardwareBudget, LayerKind, ModelProfile

    gib = 1 << 30
    monkeypatch.setattr(presets, "read_gguf_header", lambda p: type("H", (), {"sampling_defaults": {}})())
    monkeypatch.setattr(presets, "profile_from_gguf", lambda h: ModelProfile(
        name="tiny", weights_bytes=2 * gib, embd_table_bytes=0, n_ctx_train=131072,
        layers=[(LayerKind.FULL, 512)] * 4))
    return presets, HardwareBudget(usable_vram_bytes=24 * gib, total_device_bytes=24 * gib,
                                   ram_available_bytes=64 * gib)


def test_executable_path_is_the_supervised_binary(hermes_home, tmp_path, monkeypatch):
    import hermes_cli.local_runtime.bootstrap as boot

    _stage(hermes_home / "models", "tiny-dense")
    seen = {}

    class FakeSupervisor:
        base_url = "http://127.0.0.1:1/v1"

        def __init__(self, binary, models_dir, **kw):
            seen["binary"] = binary

        def start(self):
            pass

    monkeypatch.setattr(boot, "_SUPERVISOR", None)
    monkeypatch.setattr("hermes_cli.local_runtime.endpoint._state_endpoint", lambda: None)
    monkeypatch.setattr("hermes_cli.local_runtime.supervisor.LlamaServerSupervisor", FakeSupervisor)
    monkeypatch.setattr("hermes_cli.local_runtime.binaries.installed_engine", lambda backend: None)
    monkeypatch.setattr(boot, "_generate_presets", lambda *a, **k: None)
    monkeypatch.setattr(boot, "_admitted_models_max", lambda *a, **k: 1)
    monkeypatch.setattr(boot, "_start_idle_sweeper", lambda sup: None)
    exe = tmp_path / "bin" / "llama-server.exe"

    sup = boot.ensure_local_runtime({"local_runtime": {"enabled": True, "executable_path": str(exe)}})
    assert isinstance(sup, FakeSupervisor)
    assert seen["binary"] == exe


def test_model_dirs_are_staged_and_policed(hermes_home, tmp_path, monkeypatch):
    from hermes_cli.local_runtime.bootstrap import staged_model_ids

    extra = tmp_path / "shared-models"
    _stage(extra, "tiny-dense")
    (hermes_home / "config.yaml").write_text(
        f"local_runtime:\n  model_dirs:\n    - {extra.as_posix()}\n", encoding="utf-8")
    assert "tiny-dense" in staged_model_ids()

    presets, budget = _tiny_presets(monkeypatch)
    (entry,) = presets.generate_presets(hermes_home / "models", budget, tmp_path / "p.ini", extra_dirs=[extra])
    assert entry.keys["model"] == str(extra / "tiny-dense.gguf")


def test_model_overrides_land_in_the_preset_ini(hermes_home, tmp_path, monkeypatch):
    presets, budget = _tiny_presets(monkeypatch)
    mdir = tmp_path / "models"
    _stage(mdir, "tiny-dense")
    ini = tmp_path / "p.ini"
    presets.generate_presets(mdir, budget, ini,
                             overrides={"tiny-dense": {"ctx-size": 4096, "flash-attn": "on"}})
    decision = presets.read_preset_decisions(ini)["tiny-dense"]
    assert decision.window == 4096
    assert decision.keys["flash-attn"] == "on"
