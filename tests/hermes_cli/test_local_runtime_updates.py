"""Engine-update contracts (Rollout 4 follow-up):

- default tag flows from DEFAULT_CONFIG unless the user pinned;
- boot serves what is INSTALLED, never downloads (the ladder);
- update_available only when the local engine is enabled AND installed
  AND the configured tag is missing on disk;
- the update itself is a button-driven job, and prune keeps N-1.
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _install_fake_tag(home, tag: str, backend: str = "cuda") -> None:
    d = home / "runtimes" / "llamacpp" / tag / backend
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({
        "tag": tag, "backend": backend, "assets": {},
        "verified_version": f"version: {tag.lstrip('b')}",
    }), encoding="utf-8")
    # server_binary() looks for the executable name per-OS; give it both.
    (d / "llama-server.exe").write_bytes(b"MZ fake")
    (d / "llama-server").write_bytes(b"\x7fELF fake")


def test_installed_tags_newest_first(hermes_home):
    from hermes_cli.local_runtime.binaries import installed_tags

    assert installed_tags() == []
    _install_fake_tag(hermes_home, "b10290")
    _install_fake_tag(hermes_home, "b10412")
    assert installed_tags() == ["b10412", "b10290"]


def test_default_tag_flows_from_default_config(hermes_home):
    """Unpinned users inherit the Hermes-release default (deep-merge);
    the shipped default must be a plausible rolling tag."""
    from hermes_cli.config import load_config
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    default_tag = DEFAULT_CONFIG["local_runtime"]["tag"]
    assert default_tag.startswith("b") and default_tag.lstrip("b").isdigit()
    assert load_config()["local_runtime"]["tag"] == default_tag


def test_update_available_requires_enabled_and_installed(hermes_home, monkeypatch):
    """The flag's truth table: enabled+installed+configured-missing only."""
    from fastapi.testclient import TestClient

    from hermes_cli import web_server

    client = TestClient(web_server.app)
    # Same auth pattern as the other local-models route tests.
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN

    def status():
        r = client.get("/api/local-models/status")
        assert r.status_code == 200, r.text
        return r.json()

    import hermes_cli.web_routers.local_models as lm

    # Case 1: enabled, configured newer than installed -> update available.
    monkeypatch.setattr(lm, "_runtime_section",
                        lambda: {"enabled": True, "tag": "b10412"})
    _install_fake_tag(hermes_home, "b10290")
    s = status()
    assert s["update_available"] is True
    assert s["configured_tag"] == "b10412"
    assert s["tag"] == "b10290"          # serving what's installed

    # Case 2: configured tag installed -> no update.
    _install_fake_tag(hermes_home, "b10412")
    s = status()
    assert s["update_available"] is False
    assert s["tag"] == "b10412"

    # Case 3: disabled -> never flagged, even with a mismatch.
    monkeypatch.setattr(lm, "_runtime_section",
                        lambda: {"enabled": False, "tag": "b10999"})
    assert status()["update_available"] is False


def test_boot_never_downloads_missing_tag(hermes_home, monkeypatch):
    """The ladder: configured-but-not-installed serves the newest installed
    tag; nothing installed means no boot (and NO download either way)."""
    from hermes_cli.local_runtime import bootstrap

    calls = []
    monkeypatch.setattr(
        "hermes_cli.local_runtime.binaries.ensure_runtime_installed",
        lambda tag, backend, **kw: calls.append(tag) or (_ for _ in ()).throw(
            AssertionError("boot must not reach install for missing tags")))

    # Nothing installed: returns None before any install attempt.
    cfg = {"local_runtime": {"enabled": True, "tag": "b10412"}}
    assert bootstrap.ensure_local_runtime(cfg) is None
    assert calls == []


def test_prune_keeps_n_minus_one(hermes_home):
    from hermes_cli.local_runtime.binaries import installed_tags, prune_old_tags

    for tag in ("b10100", "b10200", "b10290"):
        _install_fake_tag(hermes_home, tag)
    prune_old_tags(["b10290", "b10200"])
    assert installed_tags() == ["b10290", "b10200"]
    # downloads/ cache dir must survive pruning when present.
    downloads = hermes_home / "runtimes" / "llamacpp" / "downloads"
    downloads.mkdir(exist_ok=True)
    prune_old_tags(["b10290"])
    assert downloads.exists()
