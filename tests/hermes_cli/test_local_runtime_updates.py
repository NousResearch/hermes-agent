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


@pytest.mark.parametrize("pin", [None, "b10679", "b10412"])
def test_preferred_b10964_update_offer_respects_explicit_pins(hermes_home, pin):
    """Existing unpinned installs get the shipped upgrade; user pins win."""
    from fastapi.testclient import TestClient

    from hermes_cli import web_server
    from hermes_cli.config import load_config

    runtime = {"enabled": True}
    if pin is not None:
        runtime["tag"] = pin
    (hermes_home / "config.yaml").write_text(
        json.dumps({"local_runtime": runtime}), encoding="utf-8")
    _install_fake_tag(hermes_home, "b10679")

    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    response = client.get("/api/local-models/status")
    assert response.status_code == 200, response.text
    status = response.json()
    expected_tag = pin or "b10964"
    assert load_config()["local_runtime"]["tag"] == expected_tag
    assert status["configured_tag"] == expected_tag
    assert status["tag"] == "b10679"  # An offer must not replace the installed engine.
    assert status["update_available"] is (expected_tag != "b10679")


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


def test_prune_unlinks_a_symlinked_runtime_entry(hermes_home, tmp_path):
    """A symlinked entry is not a runtime tag: rmtree would skip it silently."""
    from hermes_cli.local_runtime.binaries import prune_old_tags, runtimes_root

    _install_fake_tag(hermes_home, "b10100")
    target = tmp_path / "elsewhere"
    target.mkdir()
    (target / "marker.txt").write_text("keep", encoding="utf-8")
    link = runtimes_root() / "b10999"
    link.symlink_to(target, target_is_directory=True)

    prune_old_tags(["b10100"])

    assert not link.is_symlink() and not link.exists(), "the link itself is gone"
    assert (target / "marker.txt").exists(), "the link target is never chased"
    assert runtimes_root().joinpath("b10100").is_dir(), "kept tags survive"

def test_prune_unlinks_a_dangling_symlinked_runtime_entry(hermes_home, tmp_path):
    """A link whose target is gone is pruned too: is_dir() can never classify it.

    ``prune_old_tags`` has no age policy — it prunes every entry that is not in ``keep``
    — but it decided *what* an entry is with ``is_dir()``, which follows the link. A
    dangling entry therefore looked like neither a tag nor an error anyone reports: it
    was skipped, and the silent leak this branch exists to close stayed open for exactly
    the entries whose target disappeared first.
    """
    from hermes_cli.local_runtime.binaries import prune_old_tags, runtimes_root

    _install_fake_tag(hermes_home, "b10100")
    target = tmp_path / "elsewhere"
    target.mkdir()
    link = runtimes_root() / "b10999"
    link.symlink_to(target, target_is_directory=True)
    target.rmdir()  # the entry now dangles
    assert link.is_symlink() and not link.exists()
    assert not link.is_dir(), "the precondition: is_dir() cannot see a dangling link"
    stray = runtimes_root() / "notes.txt"
    stray.write_text("not a runtime tag", encoding="utf-8")

    prune_old_tags(["b10100"])

    assert not link.is_symlink(), "the dangling runtime link is pruned, not skipped"
    assert runtimes_root().joinpath("b10100").is_dir(), "kept tags survive"
    assert stray.exists(), "a plain file is still not a runtime tag"
