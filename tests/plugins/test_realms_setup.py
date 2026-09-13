"""Setup failures stay actionable without authorizing host execution."""
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.linux_only
def test_missing_driver_refuses_realm_selection_and_explains_recovery(tmp_path):
    load = runpy.run_path(str(ROOT / "plugins/hermes-realms/realms/_binding.py"))["load_runtime"]
    service = load("integration").RealmIntegration(tmp_path / "requested")
    owner = service.bind(session_id="setup-owner", task_id="setup-task")
    service.owners.set_mode(owner, "ask")

    with pytest.raises(ValueError, match="hermes realms install-driver"):
        service.command("on", session_id="setup-owner", task_id="setup-task")
    assert service.owners.mode(owner, "realm") == "ask"
    assert service.status(owner)["setup"]["ready"] is False

    # Already-enabled profiles fail closed too, with an actual recovery command.
    service.owners.set_mode(owner, "realm")
    result = service.pre_tool(
        tool_name="terminal", args={"command": "true"},
        session_id="setup-owner", task_id="setup-task",
    )
    assert result["action"] == "block"
    assert "hermes realms install-driver" in result["message"]
    assert "host fallback is disabled" in result["message"]
    assert service.manager.list() == []
    assert not service.driver_executable.exists()


@pytest.mark.linux_only
@pytest.mark.integration
def test_setup_installs_repairs_and_executes_pinned_driver_in_own_profile(tmp_path, monkeypatch):
    import hashlib
    import os
    import subprocess
    import urllib.request

    plugin = ROOT / "plugins/hermes-realms"
    load = runpy.run_path(str(plugin / "realms/_binding.py"))["load_runtime"]
    installer = load("install_driver")
    home = tmp_path / "requested"
    assert load("manager").Manager(home).doctor()["ok"] is False
    ambient = tmp_path / "ambient"
    monkeypatch.setenv("HERMES_HOME", str(ambient))
    archive = tmp_path / "release.tar.gz"
    with urllib.request.urlopen(installer.URL, timeout=60) as response:
        archive.write_bytes(response.read())
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == installer.ARCHIVE_SHA256

    def no_download(*args, **kwargs):
        raise AssertionError("An approved archive or valid installation needs no download")

    monkeypatch.setattr(urllib.request, "urlopen", no_download)
    target = load("config").driver_path(home)
    assert installer.install(home=home, archive=archive) == str(target)
    result = subprocess.run([str(target), "--version"], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0 and installer.VERSION in result.stdout
    receipt = target.stat()
    assert installer.install(home=home) == str(target)
    assert target.stat() == receipt

    setup = runpy.run_path(str(plugin / "setup.py"))
    described = setup["describe"](home)
    assert described["ready"] is False, 'binary verification is not completed setup'
    assert installer.VERSION in described["summary"]
    assert installer.URL in " ".join(described["details"])
    setup["run"](home)  # Idempotent setup must also avoid downloading.
    assert setup["describe"](home)["ready"] is True
    directory_mode = target.parent.stat().st_mode
    try:
        target.parent.chmod(0o500)
        with pytest.raises(PermissionError):
            setup["run"](home)
        described = setup["describe"](home)
        assert not described["ready"], "Failed receipt invalidation must not report ready"
        assert "permissions" in " ".join(described["details"])
    finally:
        target.parent.chmod(directory_mode)
    config = home / "config.yaml"
    config.write_text('plugins:\n  realms:\n    size: invalid\n', encoding="utf-8")
    with pytest.raises(ValueError, match="size"):
        setup["run"](home)
    assert not target.with_name(".realms-setup.json").exists()
    config.unlink()
    assert not setup["describe"](home)["ready"]
    setup["run"](home)
    assert setup["describe"](home)["ready"]

    # Repair actual corruption/missing execute permission/symlinks, not only a mock archive.
    other = tmp_path / "unrelated"
    for damage in ("bytes", "mode", "symlink"):
        target.unlink()
        if damage == "symlink":
            other.write_text("untouched", encoding="utf-8")
            target.symlink_to(other)
        elif damage == "bytes":
            target.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            target.chmod(0o700)
        else:
            installer.install(home=home, archive=archive)
            target.chmod(0o600)
        assert not setup["describe"](home)["ready"]
        installer.install(home=home, archive=archive)
        assert not setup["describe"](home)["ready"]
        setup["run"](home)
        assert setup["describe"](home)["ready"]
    assert other.read_text(encoding="utf-8") == "untouched"
    redirected = tmp_path / "redirected"
    foreign = tmp_path / "foreign-profile"
    redirected.mkdir()
    foreign.mkdir()
    (redirected / "plugin-data").symlink_to(foreign, target_is_directory=True)
    with pytest.raises(ValueError, match="profile"):
        installer.install(home=redirected, archive=archive)
    assert list(foreign.iterdir()) == []
    with monkeypatch.context() as relative:
        relative.chdir(tmp_path)
        exported = installer.install(archive=archive, target=Path("exported-driver"))
        assert Path(exported).is_absolute()
        assert installer.execution_verified(exported)
    before = target.read_bytes()
    archive.write_bytes(b"untrusted archive")
    with pytest.raises(ValueError, match="checksum mismatch"):
        installer.install_archive(archive, target)
    assert target.read_bytes() == before
    assert not ambient.exists()
    assert os.access(target, os.X_OK)
