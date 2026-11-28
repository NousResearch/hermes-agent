"""Undecodable compose probes still refuse builds over unknown or live state."""

from pathlib import Path
import json
import os
import shlex
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
DRIVER = """
import json, shlex, subprocess, sys
from pathlib import Path
from agent.verify.recipes import Recipe
from agent.verify.runner import run_verify
root = Path(sys.argv[1])
argv = [sys.executable, str(root / 'build.py')]
command = subprocess.list2cmdline(argv) if sys.platform == 'win32' else shlex.join(argv)
recipe = Recipe(name='compose decode fixture', kind='compose', build=[command])
result = run_verify(root, recipe, phases=('build',), skip_start=True)
print(json.dumps(result.to_dict()))
"""


def _fake_docker(directory, calls, stdout, stderr, exit_code):
    script = (
        "import json, sys\n"
        "from pathlib import Path\n"
        f"Path({str(calls)!r}).write_text(json.dumps(sys.argv[1:]), encoding='utf-8')\n"
        f"sys.stdout.buffer.write({stdout!r})\n"
        f"sys.stderr.buffer.write({stderr!r})\n"
        f"sys.exit({exit_code})\n"
    )
    if sys.platform == "win32":
        # A renamed Python script is not a PE executable. Use the same launcher
        # library already declared in the Windows test dependency group.
        from distlib.scripts import ScriptMaker

        class DockerScriptMaker(ScriptMaker):
            def _get_script_text(self, entry):
                return script

        maker = DockerScriptMaker(None, str(directory), add_launchers=True)
        maker.executable = sys.executable
        maker.variants = {""}
        maker.make("docker = fixture:main", {"interpreter_args": ["-I"]})
    else:
        launcher = directory / "docker"
        launcher.write_text(
            f"#!/bin/sh\nexec {shlex.quote(sys.executable)} -I -c {shlex.quote(script)} \"$@\"\n",
            encoding="utf-8",
        )
        launcher.chmod(0o755)


@pytest.mark.platforms("windows", "posix")
@pytest.mark.parametrize(
    "stdout,stderr,exit_code,reason",
    [
        pytest.param(b"", b"\xff\xfe daemon-error\n", 1, "failed (exit 1)", id="undecodable-stderr"),
        pytest.param(b"\xff\xfe live-container\n", b"", 0, "running container(s)", id="undecodable-stdout"),
    ],
)
def test_compose_decode_failure_refuses_mutating_build(tmp_path, stdout, stderr, exit_code, reason):
    bindir = tmp_path / "stub bin"
    bindir.mkdir()
    empty_path = tmp_path / "empty path"
    empty_path.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    calls = tmp_path / "probe-args.json"
    marker = tmp_path / "build-ran"
    _fake_docker(bindir, calls, stdout, stderr, exit_code)
    (tmp_path / "build.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('built')\n",
        encoding="utf-8",
    )

    # Set the actual reader's UTF-8 mode, independently of the runner's locale.
    # PATH contains only disposable directories, so no real Docker is invoked.
    child = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", DRIVER, str(tmp_path)],
        cwd=ROOT,
        env={**os.environ, "PATH": os.pathsep.join((str(bindir), str(empty_path)))},
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert json.loads(calls.read_text(encoding="utf-8")) == [
        "compose", "ps", "--status", "running", "--format", "{{.Name}}",
    ]
    result = json.loads(child.stdout)
    assert not result["ok"]
    assert len(result["phases"]) == 1 and result["phases"][0]["exitCode"] == 1
    refusal = result["phases"][0]["outputTail"]
    assert "Refusing to run" in refusal and reason in refusal and "\ufffd" in refusal
    assert not marker.exists(), "a failed or nonempty liveness probe must not permit a build"
