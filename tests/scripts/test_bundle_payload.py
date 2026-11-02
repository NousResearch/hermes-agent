"""Shared bundle operations use real filesystem and entrypoint contracts."""
from __future__ import annotations

import json
import os
import subprocess
import sys


import pytest

from scripts.bundles.payload import relativize_links, snapshot
from scripts.build.inputs import project_entries
from scripts.build.agent import plant_surfaces
from scripts.build.launchers import posix_launcher
from scripts.bundles.desktop import release_version


def test_release_version_comes_from_the_release_identity_not_the_checkout(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nversion="0.0.0"\n', encoding="utf-8")
    assert release_version(tmp_path, "v1.2.3") == "1.2.3"
    assert release_version(tmp_path, "v1.2.4") == "1.2.4"


def test_snapshot_carries_revision_bytes_regardless_of_builder_eol_config(tmp_path):
    # Git for Windows ships core.autocrlf=true system-wide; set it on the
    # fixture repo so every platform exercises the same builder config.
    repo = tmp_path / "repo"
    repo.mkdir()
    git = ["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.test"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "config", "core.autocrlf", "true"], cwd=repo, check=True)
    (repo / ".gitattributes").write_bytes(b"*.ps1 text eol=crlf\n")
    (repo / "plain.json").write_bytes(b'{\n"a": 1\n}\n')
    (repo / "script.ps1").write_bytes(b"one\ntwo\n")
    subprocess.run([*git, "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run([*git, "commit", "-qm", "fixture"], cwd=repo, check=True)
    out = tmp_path / "out"
    snapshot(repo, "HEAD", out)
    assert (out / "plain.json").read_bytes() == b'{\n"a": 1\n}\n'
    # Committed eol attributes still apply: only builder config is ignored.
    assert (out / "script.ps1").read_bytes() == b"one\r\ntwo\r\n"


def test_snapshot_replaces_a_previous_payload_holding_read_only_files(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "mod.py").write_text("value = 1\n", encoding="utf-8")
    git = ["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.test"]
    subprocess.run([*git, "add", "."], cwd=repo, check=True)
    subprocess.run([*git, "commit", "-qm", "fixture"], cwd=repo, check=True)
    out = tmp_path / "out"
    snapshot(repo, "HEAD", out)
    # Baked bytecode is chmodded read-only; Windows refuses to unlink it.
    stale = out / "__pycache__" / "mod.cpython.pyc"
    stale.parent.mkdir()
    stale.write_bytes(b"baked")
    stale.chmod(0o444)
    snapshot(repo, "HEAD", out)
    assert not stale.parent.exists()
    assert (out / "mod.py").read_text(encoding="utf-8") == "value = 1\n"


def test_wrapper_rejects_unresolved_template_fields(tmp_path, monkeypatch):
    from scripts.build import launchers

    template = tmp_path / "launcher_wrapper.py"
    template.write_text('entry = "__HERMES_ENTRY_MODULE__"\nmissing = "__HERMES_NEW__"\n', encoding="utf-8")
    monkeypatch.setattr(launchers, "__file__", str(tmp_path / "payload.py"))
    with pytest.raises(ValueError, match="__HERMES_NEW__"):
        launchers.render_wrapper("entry:run", "../app", "../venv/Lib/site-packages")


@pytest.mark.platforms("posix")
def test_relocation_preserves_sibling_and_framework_links(tmp_path):
    root = tmp_path / "payload"
    store = root / "tools/python/bin"
    venv = root / "venv/bin"
    store.mkdir(parents=True)
    venv.mkdir(parents=True)
    (store / "python3").write_text("interpreter", encoding="utf-8")
    (venv / "python").symlink_to("/builder/tools/python/bin/python3")
    (venv / "python3").symlink_to("python")
    framework = root / "tools/framework"
    framework.symlink_to("python/bin/python3")
    assert relativize_links(root) == 1
    assert (venv / "python3").read_text(encoding="utf-8-sig") == "interpreter"
    assert os.readlink(venv / "python3") == "python"
    assert os.readlink(framework) == "python/bin/python3"
    assert relativize_links(root) == 0
    (venv / "bad").symlink_to("/usr/bin/python")
    with pytest.raises(ValueError, match="escapes payload"):
        relativize_links(root)


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("target", ["linux-x64", "linux-arm64-bionic"])
def test_both_launchers_keep_active_home_and_call_declared_function(tmp_path, target):
    root = tmp_path / "payload with spaces"
    (root / "bin").mkdir(parents=True)
    (root / "app").mkdir()
    (root / "python").symlink_to(sys.executable)
    (root / "app/entry.py").write_text("import json,os,sys\ndef run():\n print(json.dumps([os.environ['HERMES_HOME'],sys.argv[1:]])); return 7\n", encoding="utf-8")
    launcher = root / "bin/custom"
    launcher.write_text(posix_launcher("custom", "entry:run", python="python", repo="app", site="deps", target=target), encoding="utf-8")
    result = subprocess.run(["sh", str(launcher), "two words", "$(nope)", ""], cwd=tmp_path,
                            env={**os.environ, "HERMES_HOME": str(tmp_path / "custom/profiles/memory")}, capture_output=True, text=True, check=False)
    assert result.returncode == 7, result.stderr
    assert json.loads(result.stdout) == [str(tmp_path / "custom/profiles/memory"), ["two words", "$(nope)", ""]]
