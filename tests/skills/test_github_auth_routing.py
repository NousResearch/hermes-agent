"""Execute GitHub reference setup and API gates with isolated auth fixtures.

Git-only setup must stay usable. API routes must reject Git-store-only or
missing API authentication. These are document contracts, not live API tests.
"""
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

REFERENCES = Path(__file__).resolve().parents[2] / "skills/software-development/github/references"
DOCUMENTS = ("pr-workflow.md", "repo-management.md")


def _blocks(document):
    return re.findall(r"```bash\n(.*?)\n```", (REFERENCES / document).read_text(), re.S)


def _environment(tmp_path, method):
    home = tmp_path / "home"
    helper = home / "skills/github/github-auth/scripts/gh-env.sh"
    helper.parent.mkdir(parents=True)
    helper.write_text(f"export GH_AUTH_METHOD={method} GH_USER=fixture\n")
    bindir = home / "bin"
    bindir.mkdir()
    for name in ("gh", "curl"):
        command = bindir / name
        command.write_text('#!/bin/sh\nprintf "api-call\\n" >> "$CALLS_LOG"\nprintf \'{"login":"fixture"}\\n\'\n')
        command.chmod(0o700)
    (bindir / "python").symlink_to(sys.executable)
    env = {"HOME": str(home), "HERMES_HOME": str(home),
           "PATH": str(bindir) + os.pathsep + "/usr/bin:/bin",
           "CALLS_LOG": str(home / "api-calls"), "AUTH": method}
    return env


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("document", DOCUMENTS)
@pytest.mark.parametrize("method", ("git", "none", "gh", "curl"))
def test_general_detection_allows_local_git(tmp_path, document, method):
    env = _environment(tmp_path, method)
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "--quiet", str(repo)], env=env, check=True)
    setup = _blocks(document)[0]
    result = subprocess.run(["/bin/bash", "-c", setup + '\ngit -C "$1" rev-parse --is-inside-work-tree',
                             "fixture", str(repo)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "true", result.stdout
    if method in ("git", "none"):
        assert not Path(env["CALLS_LOG"]).exists(), "Git-only setup invoked an API command"


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("document", DOCUMENTS)
@pytest.mark.parametrize("method", ("git", "none", "gh", "curl"))
def test_api_gate_rejects_non_api_authentication(tmp_path, document, method):
    env = _environment(tmp_path, method)
    gates = [block for block in _blocks(document) if 'case "$AUTH"' in block and "exit 1" in block]
    assert gates, "No fail-closed API gate documented"
    result = subprocess.run(["/bin/bash", "-c", gates[0] + '\nprintf "API_ROUTE_ALLOWED\\n"'],
                            env=env, capture_output=True, text=True)
    if method in ("git", "none"):
        assert result.returncode == 1
        assert "API_ROUTE_ALLOWED" not in result.stdout
        assert not Path(env["CALLS_LOG"]).exists(), "Rejected gate invoked an API command"
    else:
        assert result.returncode == 0, result.stderr
        assert "API_ROUTE_ALLOWED" in result.stdout
