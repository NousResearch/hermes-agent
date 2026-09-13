"""Update retries use the same dependency transaction as a newly pulled tree."""

import json
import subprocess
import venv
from types import SimpleNamespace

import pytest

import pm
from hermes_cli import main, update_cmd


def test_current_checkout_dependency_failure_prevents_completion(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)

    monkeypatch.setattr(update_cmd, "_print_verified_update_completion", lambda *a: pytest.fail("reported completion"))

    def fail_sync(*args, **kwargs):
        raise pm.InstallError("venv", "dependency conflict")

    monkeypatch.setattr(pm, "sync_venv", fail_sync)
    with pytest.raises(pm.InstallError, match="dependency conflict"):
        update_cmd._repair_current_checkout(
            assume_yes=True, gateway_mode=False, pre_update_snapshot_id=None,
            had_desktop_app_before_update=False, upstream_checked=True,
        )


def test_build_runs_in_selected_python_and_propagates_failure(tmp_path, monkeypatch):
    from hermes_cli.update_cmd_maint import _prepare_updated_checkout
    from hermes_cli.runtime_paths import install_state_dir, runtime_facts_path
    from hermes_constants import venv_python_path

    root = tmp_path / "checkout"
    package = root / "hermes_cli"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "source_build.py").write_text(
        "import json, os, pathlib, sys\n"
        "pathlib.Path('build-process.json').write_text(json.dumps({"
        "'python': sys.executable, 'argv': sys.argv[1:], 'path': sys.path}))\n"
        "raise SystemExit(23)\n"
    )
    calls = []
    selected = install_state_dir(root) / "environments/selected/venv"
    # A stale repo-local venv must not win over PM's selected generation.
    venv.EnvBuilder(with_pip=False).create(root / "venv")

    def sync(*args, **kwargs):
        calls.append((args, kwargs))
        # Publication creates the interpreter the next process must use.
        venv.EnvBuilder(with_pip=False).create(selected)
        pm.Facts(runtime_facts_path(root)).record_state("venv", "prepared", [], environment=selected)

    monkeypatch.setattr(pm, "sync_venv", sync)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "obsolete-deps"))
    with pytest.raises(subprocess.CalledProcessError) as error:
        _prepare_updated_checkout(root, desktop=True)
    assert error.value.returncode == 23
    assert calls == [((), {"explicit": True, "project_root": root})]
    record = json.loads((root / "build-process.json").read_text())
    assert record["python"] == str(venv_python_path(selected))
    assert record["argv"] == ["--source", str(root), "--desktop"]
    assert str(tmp_path / "obsolete-deps") not in record["path"]
    assert not (root / ".update-incomplete").exists()
    assert not (root / ".lazy-refresh-incomplete").exists()


@pytest.mark.parametrize("failure", [
    pm.InstallError("venv", "conflict"),
    subprocess.CalledProcessError(23, ["python", "-m", "hermes_cli.source_build"]),
])
def test_command_reports_failed_preparation_and_releases_lock(tmp_path, monkeypatch, capsys, failure):
    from hermes_cli import update_lock, update_receipt

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(main, "_update_preflight_handled", lambda args: False)
    monkeypatch.setattr(main, "_install_hangup_protection", lambda **kw: None)
    finalized = []
    monkeypatch.setattr(main, "_finalize_update_output", finalized.append)

    def fail(args, gateway_mode):
        update_receipt.begin_update_receipt()
        raise failure

    monkeypatch.setattr(update_cmd, "_cmd_update_impl", fail)
    with pytest.raises(SystemExit) as error:
        main.cmd_update(SimpleNamespace(gateway=True))
    assert error.value.code == 1
    receipt = update_receipt.read_latest_receipt()
    assert receipt is not None
    assert receipt["exit_code"] == 1
    assert receipt["outcome"] == "failed"
    assert (tmp_path / ".update_exit_code").read_text().strip() == "1"
    assert finalized == [None]
    lock = update_lock.UpdateLock()
    assert lock.acquire()
    lock.release()
    assert "Update failed" in capsys.readouterr().out
