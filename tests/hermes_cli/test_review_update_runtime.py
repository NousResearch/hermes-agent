"""Source update safety at the bootstrap and completion boundaries."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from hermes_cli import source_completion, venv_sync


def test_pre_pm_version_reads_checkout_stamp_without_importing_pm():
    root = Path(__file__).resolve().parents[2]
    code = (
        "import builtins\n"
        "original = builtins.__import__\n"
        "def restricted(name, *args, **kwargs):\n"
        "    if name == 'pm' or name.startswith('pm.'):\n"
        "        raise ModuleNotFoundError(\"No module named 'pm'\", name='pm')\n"
        "    return original(name, *args, **kwargs)\n"
        "builtins.__import__ = restricted\n"
        "from hermes_cli import __version__\n"
        "print(__version__)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True,
                            text=True, encoding="utf-8", timeout=20)
    assert result.returncode == 0, result.stderr
    stamp = root / "install-stamp.json"
    expected = json.loads(stamp.read_text(encoding="utf-8-sig")).get("baseVersion") if stamp.exists() else None
    assert result.stdout.strip() == (expected or "0.0.0")


@pytest.mark.platforms("posix")
def test_foreign_owned_venv_file_refused_before_sync(tmp_path, monkeypatch):
    from pm import environments

    checkout = tmp_path / "checkout"
    installer = checkout / "venv/lib/python3.14/site-packages/pkg.dist-info/INSTALLER"
    installer.parent.mkdir(parents=True)
    installer.write_text("pip\n", encoding="utf-8")
    monkeypatch.setattr(environments, "selected_venv", lambda root: checkout / "venv")
    original_lstat = Path.lstat
    foreign_uid = os.geteuid() + 1

    def stat(path, *args, **kwargs):
        if path == installer:
            return SimpleNamespace(st_uid=foreign_uid)
        return original_lstat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", stat)
    with pytest.raises(RuntimeError, match="INSTALLER.*owned by uid"):
        venv_sync.refuse_foreign_owned_venv(checkout)


def test_completed_maintenance_survives_stamp_io_error(tmp_path, monkeypatch, capsys):
    from hermes_cli import source_build, source_stamp, update_cmd_maint

    monkeypatch.setattr(venv_sync, "publish_launchers", lambda root: None)
    monkeypatch.setattr(source_build, "build_update_products", lambda root, *, desktop: None)
    monkeypatch.setattr(update_cmd_maint, "_run_post_update_maintenance", lambda **kwargs: True)
    monkeypatch.setattr(source_stamp, "write_source_stamp", lambda root: (_ for _ in ()).throw(OSError("readonly")))
    assert source_completion.complete_source_checkout(tmp_path, desktop=False, assume_yes=True)
    assert "completed, but the install stamp" in capsys.readouterr().err


def test_sealed_stamp_reader_honors_external_install_root(tmp_path, monkeypatch):
    from pm import paths

    stamped = tmp_path / "payload"
    stamped.mkdir()
    (tmp_path / "install-stamp.json").write_text(json.dumps({"updateMechanism": "external"}), encoding="utf-8")
    monkeypatch.setenv("HERMES_INSTALL_ROOT", str(tmp_path))
    # The executing tree may be mapped into a wrapper-owned installation root.
    monkeypatch.setattr(paths, "repo_root", lambda: stamped)
    monkeypatch.setattr(paths, "install_root", lambda: tmp_path)
    assert venv_sync._is_sealed(stamped) is True


def test_developer_checkout_skips_managed_runtime_warning(tmp_path, monkeypatch):
    import pm

    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)
    monkeypatch.setattr(pm, "activate", lambda: pytest.fail("dev checkout activated managed runtime"))
    assert venv_sync.check_runtime(checkout) is None
