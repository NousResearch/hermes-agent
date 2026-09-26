"""RED first for #123784: PM rebuilds drop memory-provider extras.

Two gaps on unfixed main:
1. An explicit sync never unions the configured memory provider's PM extra,
   so a provider enabled via config (not via a recorded setup sync) rebuilds
   without its runtime dep.
2. Hindsight local_embedded mode reports dependencies_installed=True when the
   PM ledger is current even though the embedded `hindsight` runtime is gone.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace


def _stub_memory_provider(monkeypatch, tmp_path, *, provider, extra):
    from pathlib import Path

    candidate = Path(str(tmp_path)) / "provider"
    candidate.mkdir(parents=True, exist_ok=True)
    (candidate / "plugin.yaml").write_text(f"name: {provider}\nextra: {extra}\n")
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"memory": {"provider": provider}},
    )
    monkeypatch.setattr(
        "plugins.memory.find_provider_dir",
        lambda name: candidate if name == provider else None,
    )
    return candidate


def test_explicit_sync_unions_configured_memory_extra(tmp_path, monkeypatch):
    """Explicit builds must carry the configured provider's extra (Ask #1)."""
    import pm.extras as extras_mod
    from pm import install as install_mod
    from pm.package import InstallError
    import pm.receipt as receipt_mod

    assert hasattr(extras_mod, "configured_memory_extras"), "configured_memory_extras missing"
    _stub_memory_provider(monkeypatch, tmp_path, provider="supermemory", extra="supermemory")
    monkeypatch.setattr(extras_mod, "_PLATFORM_GATES", {})

    assert extras_mod.configured_memory_extras() == ["supermemory"]

    seen = {}

    def fake_policy(requested, *, repair):
        seen["extras"] = requested
        raise InstallError("venv", "stop after policy")

    monkeypatch.setattr(install_mod, "_feature_policy", fake_policy)
    monkeypatch.setattr(receipt_mod, "begin", lambda kind: "tok")
    monkeypatch.setattr(receipt_mod, "record_step", lambda *a, **k: None)
    monkeypatch.setattr(receipt_mod, "finalize", lambda *a, **k: None)

    import pytest

    with pytest.raises(InstallError):
        install_mod.sync_venv(["all"], explicit=True)
    assert seen["extras"] == ["all", "supermemory"]

    with pytest.raises(InstallError):
        install_mod.sync_venv(None, explicit=True)
    assert seen["extras"] == ["supermemory"]

    # Lazy and repair syncs never gain features.
    with pytest.raises(InstallError):
        install_mod.sync_venv(["all"])
    assert seen["extras"] == ["all"]

    with pytest.raises(InstallError):
        install_mod.sync_venv(None, repair=True)
    assert seen["extras"] is None


def test_hindsight_embedded_missing_runtime_is_detectable(tmp_path, monkeypatch):
    """local_embedded + missing `hindsight` import must read as not-installed (Ask #2)."""
    import importlib.util

    import pm
    from hermes_cli import web_server_memory as wsm

    home = tmp_path / "home"
    (home / "hindsight").mkdir(parents=True)
    (home / "hindsight" / "config.json").write_text('{"mode": "local_embedded"}')
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: home)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"memory": {"provider": "hindsight"}})
    monkeypatch.setattr(wsm, "_load_memory_provider", lambda name: SimpleNamespace())
    monkeypatch.setattr(
        wsm,
        "_normalize_memory_provider_schema",
        lambda name, provider: [],
    )
    monkeypatch.setattr(
        "plugins.memory.discover_memory_providers",
        lambda: [("hindsight", "desc", True)],
        raising=False,
    )
    # The PM ledger is current (member present) — the old code reports True here.
    monkeypatch.setattr(pm, "venv_is_current", lambda **kwargs: True)
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: None if name == "hindsight" else real_find_spec(name),
    )

    info = wsm._memory_provider_setup_info("hindsight")
    assert info["dependencies_installed"] is False
