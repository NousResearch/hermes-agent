"""Layout discovery must permit manifest reads without permitting mutations."""
from __future__ import annotations

import os

import pytest

from tests.home_io_guard import HomeIOGuard


@pytest.mark.parametrize("contents", [None, "{}", "not-json"])
def test_real_payload_discovery_can_read_a_manifest_beside_an_in_home_checkout(tmp_path, monkeypatch, contents):
    from pm.environments import _payload_manifest

    home = tmp_path / "protected-home"
    checkout = home / "hermes-agent"
    checkout.mkdir(parents=True)
    if contents is not None:
        (home / "manifest.json").write_text(contents, encoding="utf-8")
    guard = HomeIOGuard(lambda: [home])
    with monkeypatch.context() as patcher:
        guard.install(patcher)
        assert _payload_manifest(checkout) is None


def test_pytest_checkout_guard_precedes_gitdir_marker_discovery(tmp_path, monkeypatch):
    from hermes_cli import _early_recovery

    def forbidden_marker_probe(root):
        raise AssertionError("must not inspect the live gitdir under pytest")

    monkeypatch.setattr(_early_recovery, "_pytest_owns_live_checkout", lambda root: True)
    monkeypatch.setattr(_early_recovery, "interrupted_pull_marker", forbidden_marker_probe)
    assert _early_recovery.restore_interrupted_pull(tmp_path) is False


@pytest.mark.parametrize("present", [False, True])
def test_payload_manifest_read_uses_guarded_io_open(tmp_path, monkeypatch, present):
    home = tmp_path / "protected-home"
    home.mkdir()
    manifest = home / "manifest.json"
    if present:
        manifest.write_text('{"venv": "venv"}', encoding="utf-8")
    guard = HomeIOGuard(lambda: [home])
    with monkeypatch.context() as patcher:
        guard.install(patcher)
        if present:
            assert manifest.read_text(encoding="utf-8") == '{"venv": "venv"}'
            with open(manifest, encoding="utf-8") as stream:
                assert stream.read() == '{"venv": "venv"}'
            fd = os.open(manifest, os.O_RDONLY)
            os.close(fd)
        else:
            with pytest.raises(FileNotFoundError):
                manifest.read_text(encoding="utf-8")
        assert manifest.is_file() is present


@pytest.mark.parametrize("operation", ["write", "append", "update", "os-open", "unlink", "replace"])
def test_payload_manifest_mutations_remain_refused(tmp_path, monkeypatch, operation):
    home = tmp_path / "protected-home"
    home.mkdir()
    manifest = home / "manifest.json"
    manifest.write_text("unchanged", encoding="utf-8")
    source = tmp_path / "replacement.json"
    source.write_text("replacement", encoding="utf-8")
    guard = HomeIOGuard(lambda: [home])
    with monkeypatch.context() as patcher:
        guard.install(patcher)
        with pytest.raises(AssertionError, match="REAL hermes home"):
            if operation == "write":
                manifest.write_text("changed", encoding="utf-8")
            elif operation == "append":
                with open(manifest, "a", encoding="utf-8"):
                    pass
            elif operation == "update":
                with open(manifest, "r+", encoding="utf-8"):
                    pass
            elif operation == "os-open":
                fd = os.open(manifest, os.O_WRONLY)
                os.close(fd)
            elif operation == "unlink":
                manifest.unlink()
            else:
                source.replace(manifest)
    assert manifest.read_text(encoding="utf-8") == "unchanged"
    assert source.read_text(encoding="utf-8") == "replacement"


@pytest.mark.parametrize("relative", ["config.yaml", ".env", "state.db", "manifest.json.bak", "nested/manifest.json"])
def test_manifest_exemption_does_not_allow_other_home_reads(tmp_path, monkeypatch, relative):
    home = tmp_path / "protected-home"
    target = home / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("protected", encoding="utf-8")
    guard = HomeIOGuard(lambda: [home])
    with monkeypatch.context() as patcher:
        guard.install(patcher)
        with pytest.raises(AssertionError, match="REAL hermes home"):
            target.read_text(encoding="utf-8")
