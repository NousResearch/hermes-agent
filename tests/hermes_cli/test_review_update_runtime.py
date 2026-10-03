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


@pytest.mark.platforms("posix")
def test_foreign_owned_node_modules_dir_refused_before_sync(tmp_path, monkeypatch):
    """A root-owned cache entry under an npm workspace must refuse the update
    before `npm ci` reaches it (the EACCES-rmdir wall: serve+gateway crash-loop,
    updater completion aborts, and npm's advice is to run as root, which poisons
    the worktree further)."""
    from pm import environments

    checkout = tmp_path / "checkout"
    vitest = checkout / "apps/desktop/node_modules/.vite/vitest"
    vitest.mkdir(parents=True)
    (vitest / "results.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(environments, "selected_venv", lambda root: checkout / "venv")
    original_lstat = Path.lstat
    foreign_uid = os.geteuid() + 1

    def stat(path, *args, **kwargs):
        if path == vitest:
            return SimpleNamespace(st_uid=foreign_uid)
        return original_lstat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", stat)
    with pytest.raises(RuntimeError, match=r"\.vite/vitest.*owned by uid"):
        venv_sync.refuse_foreign_owned_venv(checkout)


@pytest.mark.platforms("posix")
def test_foreign_owned_runtime_tree_refused_before_sync(tmp_path, monkeypatch):
    """A root-owned generation under the managed runtime tree must refuse the
    update (root-run installs publish root-owned generation state)."""
    from pm import environments

    checkout = tmp_path / "checkout"
    generation = checkout / ".hermes-runtime/python/generation-1"
    generation.mkdir(parents=True)
    monkeypatch.setattr(environments, "selected_venv", lambda root: checkout / "venv")
    original_lstat = Path.lstat
    foreign_uid = os.geteuid() + 1

    def stat(path, *args, **kwargs):
        if path == generation:
            return SimpleNamespace(st_uid=foreign_uid)
        return original_lstat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", stat)
    with pytest.raises(RuntimeError, match=r"generation-1.*owned by uid"):
        venv_sync.refuse_foreign_owned_venv(checkout)


@pytest.mark.platforms("posix")
def test_clean_checkout_passes_ownership_guard(tmp_path, monkeypatch):
    """A fully hermes-owned checkout (venv + node_modules + runtime trees) must
    not trip the guard."""
    from pm import environments

    checkout = tmp_path / "checkout"
    for sub in ("venv/lib/python3.14/site-packages",
                "apps/desktop/node_modules/.vite/vitest",
                ".hermes-runtime/python/generation-1/lib/python3.11"):
        (checkout / sub).mkdir(parents=True)
    (checkout / "venv/lib/python3.14/site-packages/pkg.dist-info").mkdir()
    (checkout / "apps/desktop/node_modules/.vite/vitest/results.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(environments, "selected_venv", lambda root: checkout / "venv")
    venv_sync.refuse_foreign_owned_venv(checkout)  # must not raise


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
