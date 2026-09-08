"""Every staged model must launch with a policy decision, never stock fit.

The managed server autoloads any GGUF in its models dir; a model missing
from the preset INI loads with llama-server defaults (f16 KV at max
context, no placement) — on Windows/WDDM that silently demotes VRAM and
decodes at a crawl. Boot must therefore refuse to adopt a running server
whose presets predate the staged set."""

from __future__ import annotations

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _stage(home, name):
    mdir = home / "models"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / f"{name}.gguf").write_bytes(b"GGUF" + b"\x00" * 32)


def _write_presets(home, *model_ids):
    pdir = home / "runtimes" / "llamacpp"
    pdir.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f"[{m}]\nctx-size = 65536\n" for m in model_ids)
    (pdir / "presets.ini").write_text(body, encoding="utf-8")


def test_boot_passes_tensor_placement_to_preset_generation(tmp_path, monkeypatch):
    import hermes_cli.local_runtime.bootstrap as boot
    import hermes_cli.local_runtime.hardware as hardware
    import hermes_cli.local_runtime.presets as presets

    captured = {}
    monkeypatch.setattr(hardware, "probe_budget", lambda planning=False: "budget")

    def fake_generate(models_dir, budget, preset_path, **kwargs):
        captured.update(models_dir=models_dir, budget=budget,
                        preset_path=preset_path, kwargs=kwargs)
        return []

    monkeypatch.setattr(presets, "generate_presets", fake_generate)
    preset_path = tmp_path / "presets.ini"

    assert boot._generate_presets(tmp_path, preset_path, tensor_placement="auto") == preset_path
    assert captured["kwargs"] == {"tensor_placement": "auto"}


def test_invalid_tensor_placement_is_rejected_before_runtime_boot(monkeypatch):
    from hermes_cli.local_runtime import bootstrap

    with pytest.raises(ValueError, match="unsupported tensor placement"):
        bootstrap.ensure_local_runtime(
            {"local_runtime": {"enabled": True, "tensor_placement": "invalid"}})


def test_disabled_runtime_ignores_tensor_placement_until_enabled(monkeypatch):
    from hermes_cli.local_runtime import bootstrap

    assert bootstrap.ensure_local_runtime(
        {"local_runtime": {"enabled": False, "tensor_placement": "invalid"}}) is None


def test_presets_stale_when_a_staged_model_has_no_section(hermes_home):
    from hermes_cli.local_runtime.bootstrap import _presets_stale

    _stage(hermes_home, "model-a")
    _stage(hermes_home, "model-b")
    _write_presets(hermes_home, "model-a")
    assert _presets_stale() is True


def test_presets_current_when_every_staged_model_is_covered(hermes_home):
    from hermes_cli.local_runtime.bootstrap import _presets_stale

    _stage(hermes_home, "model-a")
    _write_presets(hermes_home, "model-a")
    assert _presets_stale() is False


def test_no_models_is_never_stale(hermes_home):
    from hermes_cli.local_runtime.bootstrap import _presets_stale

    _write_presets(hermes_home, "model-a")
    assert _presets_stale() is False


def test_boot_replaces_incumbent_with_stale_presets(hermes_home, monkeypatch):
    """ensure_local_runtime must not adopt a running server whose presets
    miss a staged model — it stops it and boots fresh (boot itself is
    stubbed; the contract under test is the adopt/replace decision)."""
    import hermes_cli.local_runtime.bootstrap as boot

    _stage(hermes_home, "model-a")
    _stage(hermes_home, "model-b")
    _write_presets(hermes_home, "model-a")

    stopped = {}
    monkeypatch.setattr(
        "hermes_cli.local_runtime.endpoint._state_endpoint",
        lambda: {"base_url": "http://127.0.0.1:18434/v1", "pid": 12345})
    monkeypatch.setattr(boot, "_stop_state_server",
                        lambda state: stopped.setdefault("pid", state["pid"]))

    sentinel = object()

    def fake_boot(*a, **k):
        raise _BootReached()

    class _BootReached(Exception):
        pass

    # Fail fast once boot proper begins — reaching it IS the assertion.
    monkeypatch.setattr(
        "hermes_cli.local_runtime.binaries.ensure_runtime_installed", fake_boot)

    result = boot.ensure_local_runtime({"local_runtime": {"enabled": True}})
    assert stopped.get("pid") == 12345, "stale incumbent was not stopped"
    # Boot proceeded past adoption (our fake raised inside the try block,
    # which ensure_local_runtime swallows into a None return).
    assert result is None or result is sentinel


def test_refresh_bounces_an_adopted_server(hermes_home, monkeypatch):
    """refresh_local_runtime with no in-process supervisor but a running
    state-file server (the post-restart shape) must stop that server and
    boot fresh — NOT silently no-op. Regression: the no-op meant every
    download/delete after a backend restart left the router serving a
    stale model catalog, and picking the new model failed with
    'not found in this provider's model listing'."""
    import hermes_cli.local_runtime.bootstrap as boot

    stopped = {}
    monkeypatch.setattr(boot, "_SUPERVISOR", None)
    monkeypatch.setattr(
        "hermes_cli.local_runtime.endpoint._state_endpoint",
        lambda: {"base_url": "http://127.0.0.1:18434/v1", "pid": 4242})
    monkeypatch.setattr(boot, "_stop_state_server",
                        lambda state: stopped.setdefault("pid", state["pid"]))
    booted = {}
    monkeypatch.setattr(boot, "ensure_local_runtime",
                        lambda cfg, force=False: booted.setdefault("force", force) or object())

    assert boot.refresh_local_runtime() is True
    assert stopped.get("pid") == 4242, "adopted server was not stopped"
    assert booted.get("force") is True, "fresh boot did not follow the stop"


def test_refresh_no_server_anywhere_is_a_noop(hermes_home, monkeypatch):
    import hermes_cli.local_runtime.bootstrap as boot

    monkeypatch.setattr(boot, "_SUPERVISOR", None)
    monkeypatch.setattr(
        "hermes_cli.local_runtime.endpoint._state_endpoint", lambda: None)
    assert boot.refresh_local_runtime() is False
