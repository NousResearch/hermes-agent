"""Run generated launchers with a real interpreter and captured entrypoint."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scripts.build.launchers import write_launchers


@pytest.mark.platforms("posix")
def test_launchers_forward_arguments_and_export_payload_environment(tmp_path):
    payload = tmp_path / "payload with spaces"
    (payload / "venv/bin").mkdir(parents=True)
    (payload / "venv/bin/python").symlink_to(sys.executable)
    (payload / "app").mkdir()
    (payload / "app/capture_entry.py").write_text(
        "import json, os, sys\n"
        "def main():\n"
        "    print(json.dumps({'argv': sys.argv[1:], 'env': {k: os.environ.get(k) for k in "
        "['HERMES_NODE', 'HERMES_PYTHON', 'HERMES_RUNTIME_DIR', 'PYTHONPATH', 'PYTHONHOME', 'PYTHONPYCACHEPREFIX']}}))\n",
        encoding="utf-8",
    )
    (tmp_path / "json.py").write_text("raise RuntimeError('cwd shadowed stdlib')\n", encoding="utf-8")
    entries = {name: "capture_entry:main" for name in ("hermes", "hermes-agent", "hermes-acp")}
    write_launchers(payload, entries, python="venv/bin/python", repo="app",
                    site="venv/lib/python3.14/site-packages", target="linux-arm64-bionic")
    bin_dir = tmp_path / "prefix/bin"
    bin_dir.mkdir(parents=True)
    for name in entries:
        link = bin_dir / name
        link.symlink_to(os.path.relpath(payload / "bin" / name, bin_dir))
        result = subprocess.run(
            [shutil.which("sh"), str(link), "one two", "$(not-executed)", ""],
            cwd=tmp_path, check=True, capture_output=True, text=True,
            env={**os.environ, "PYTHONHOME": "/does-not-exist", "PYTHONPATH": "/foreign"},
        )
        data = json.loads(result.stdout)
        assert data["argv"] == ["one two", "$(not-executed)", ""]
        env = data["env"]
        assert Path(env["HERMES_PYTHON"]).resolve() == Path(sys.executable).resolve()
        assert Path(env["HERMES_NODE"]) == payload / "tools/node/data/data/com.termux/files/usr/bin/node"
        assert Path(env["HERMES_RUNTIME_DIR"]) == payload / "tools"
        assert env["PYTHONPATH"].split(os.pathsep)[0] == str(payload / "app")
        assert env["PYTHONHOME"] is None
        assert not Path(env["PYTHONPYCACHEPREFIX"]).is_relative_to(payload)


@pytest.mark.platforms("posix")
def test_postinst_refuses_foreign_path_and_prerm_preserves_it(tmp_path):
    from scripts.termux.launchers import write_maintainer_scripts

    control = tmp_path / "DEBIAN"
    control.mkdir()
    write_maintainer_scripts(control, ["hermes"])
    prefix = tmp_path / "prefix"
    bin_dir = prefix / "bin"
    bin_dir.mkdir(parents=True)
    link = bin_dir / "hermes"
    env = {**os.environ, "PREFIX": str(prefix)}
    command = [shutil.which("sh"), str(control / "postinst"), "configure"]
    subprocess.run(command, check=True, env=env)
    assert os.readlink(link) == "../lib/hermes-agent/bin/hermes"
    subprocess.run(command, check=True, env=env)
    link.unlink()
    link.write_text("foreign launcher", encoding="utf-8")
    assert subprocess.run(command, env=env).returncode != 0
    subprocess.run([shutil.which("sh"), str(control / "prerm"), "remove"], check=True, env=env)
    assert link.read_text(encoding="utf-8") == "foreign launcher"


@pytest.mark.platforms("posix")
def test_verifier_stops_detached_descendants_before_cleanup(tmp_path):
    import psutil
    from scripts.termux.validate_installed import stop_child_tree

    code = (
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'], "
        "start_new_session=True)\n"
        "print(child.pid, flush=True)\n"
        "time.sleep(120)\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", code], stdout=subprocess.PIPE, text=True,
        cwd=tmp_path, start_new_session=True,
    )
    descendant = psutil.Process(int(child.stdout.readline()))
    try:
        stop_child_tree(child)
        assert child.poll() is not None
        assert not descendant.is_running() or descendant.status() == psutil.STATUS_ZOMBIE
    finally:
        if child.poll() is None:
            child.kill()
        if descendant.is_running() and descendant.status() != psutil.STATUS_ZOMBIE:
            descendant.kill()
        child.wait(timeout=10)
        child.stdout.close()
