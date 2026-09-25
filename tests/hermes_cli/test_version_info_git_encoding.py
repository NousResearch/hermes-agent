"""Exercise native Windows pipe decoding with UTF-8 mode disabled."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from hermes_cli.version_info import _run_git


@pytest.mark.parametrize("failure", [OSError("missing git"), subprocess.TimeoutExpired("git", 3)])
def test_git_probe_preserves_failure_fallback(tmp_path, monkeypatch, failure):
    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(subprocess, "run", fail)
    assert _run_git(tmp_path, "status") is None


def test_git_probe_preserves_empty_and_failed_output(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)
    assert _run_git(tmp_path, "tag", "--list") is None
    assert _run_git(tmp_path, "rev-parse", "--verify", "refs/heads/missing") is None
    blob = tmp_path / "legacy.txt"
    blob.write_bytes(b"legacy: \xff\n")
    oid = _run_git(tmp_path, "hash-object", "-w", str(blob))
    assert oid
    assert _run_git(tmp_path, "cat-file", "blob", oid) == "legacy: \ufffd"


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("dirty", [False, True])
def test_version_identity_decodes_git_utf8_without_utf8_mode(tmp_path, dirty):
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull}

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=repo, env=env, check=True,
            capture_output=True, encoding="utf-8", timeout=10,
        ).stdout.strip()

    git("init", "-q")
    git("config", "user.name", "Hermes Test")
    git("config", "user.email", "hermes@example.invalid")
    git("config", "core.quotePath", "false")
    branch = "feature/\u4e81"
    git("checkout", "-qb", branch)
    project = '[project]\nname = "fixture"\nversion = "1.2.3"\ndescription = "Hermes \u2014 \u4e81"\n'
    (repo / "pyproject.toml").write_text(project, encoding="utf-8")
    tracked = repo / "\u4e81.txt"
    tracked.write_text("release\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-qm", "release")
    git("tag", "v2026.1.2")
    git("commit", "--allow-empty", "-qm", "after release")
    if dirty:
        tracked.write_text("changed\n", encoding="utf-8")
    commit = git("rev-parse", "HEAD")
    code_root = Path(__file__).resolve().parents[2]
    probe = f"""
import json, sys
from dataclasses import asdict
from pathlib import Path
sys.path.insert(0, {str(code_root)!r})
from hermes_cli import version_info
version_info._resolve_stamp_file = lambda: None
version_info._resolve_repo_dir = lambda: Path({str(repo)!r})
print(json.dumps(asdict(version_info.get_version_info())))
"""
    # Do not emulate another OS or mock subprocess.run: this invokes real git
    # through the public identity resolver and the host's native locale codec.
    result = subprocess.run(
        [sys.executable, "-X", "utf8=0", "-c", probe],
        cwd=repo, env=env, capture_output=True, encoding="utf-8", timeout=30,
    )
    assert result.returncode == 0, result.stderr
    info = json.loads(result.stdout)
    assert info["base_version"] == "1.2.3", result.stderr
    assert info["branch"] == branch, result.stderr
    assert info["commit"] == commit
    assert info["distance"] == 1
    assert info["dirty"] is dirty
    assert result.stderr == ""
