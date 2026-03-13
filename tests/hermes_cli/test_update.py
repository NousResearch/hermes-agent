import argparse
import importlib
import subprocess
from pathlib import Path

import pytest


class _RunRecorder:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def __call__(self, cmd, **kwargs):
        key = tuple(cmd)
        self.calls.append(list(cmd))
        if key not in self.responses:
            pytest.fail(f"Unexpected command: {cmd}")

        result = self.responses[key]
        if isinstance(result, Exception):
            raise result

        if kwargs.get("check") and result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode,
                cmd,
                output=result.stdout,
                stderr=result.stderr,
            )
        return result


def _completed(cmd, stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)


def _load_main(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    import hermes_cli.main as main

    return importlib.reload(main)


def test_cmd_update_uses_tracking_upstream_remote(monkeypatch, tmp_path):
    main = _load_main(monkeypatch, tmp_path)
    project_root = tmp_path / "repo"
    (project_root / ".git").mkdir(parents=True)
    monkeypatch.setattr(main, "PROJECT_ROOT", project_root)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv" if name == "uv" else None)

    import tools.skills_sync as skills_sync
    import hermes_cli.config as config

    monkeypatch.setattr(
        skills_sync,
        "sync_skills",
        lambda quiet=True: {"copied": [], "updated": [], "user_modified": [], "cleaned": []},
    )
    monkeypatch.setattr(config, "get_missing_env_vars", lambda required_only=True: [])
    monkeypatch.setattr(config, "get_missing_config_fields", lambda: [])
    monkeypatch.setattr(config, "check_config_version", lambda: (1, 1))
    monkeypatch.setattr(
        config,
        "migrate_config",
        lambda interactive=True, quiet=False: {"env_added": [], "config_added": []},
    )

    responses = {
        ("git", "rev-parse", "--abbrev-ref", "HEAD"): _completed(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stdout="fix/discord-attachment-support\n",
        ),
        ("git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"): _completed(
            ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
            stdout="fork/fix/discord-attachment-support\n",
        ),
        ("git", "fetch", "fork"): _completed(["git", "fetch", "fork"]),
        (
            "git",
            "rev-list",
            "HEAD..fork/fix/discord-attachment-support",
            "--count",
        ): _completed(
            ["git", "rev-list", "HEAD..fork/fix/discord-attachment-support", "--count"],
            stdout="1\n",
        ),
        ("git", "pull", "fork", "fix/discord-attachment-support"): _completed(
            ["git", "pull", "fork", "fix/discord-attachment-support"]
        ),
        ("/usr/bin/uv", "pip", "install", "-e", ".", "--quiet"): _completed(
            ["/usr/bin/uv", "pip", "install", "-e", ".", "--quiet"]
        ),
        ("systemctl", "--user", "is-active", "hermes-gateway"): _completed(
            ["systemctl", "--user", "is-active", "hermes-gateway"],
            stdout="inactive\n",
        ),
    }
    runner = _RunRecorder(responses)
    monkeypatch.setattr(subprocess, "run", runner)

    main.cmd_update(argparse.Namespace())

    assert ["git", "fetch", "fork"] in runner.calls
    assert ["git", "pull", "fork", "fix/discord-attachment-support"] in runner.calls


def test_cmd_update_falls_back_to_origin_when_branch_has_no_upstream(monkeypatch, tmp_path):
    main = _load_main(monkeypatch, tmp_path)
    project_root = tmp_path / "repo"
    (project_root / ".git").mkdir(parents=True)
    monkeypatch.setattr(main, "PROJECT_ROOT", project_root)

    responses = {
        ("git", "rev-parse", "--abbrev-ref", "HEAD"): _completed(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stdout="feature/local-only\n",
        ),
        ("git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"): _completed(
            ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
            stderr="fatal: no upstream configured\n",
            returncode=128,
        ),
        ("git", "fetch", "origin"): _completed(["git", "fetch", "origin"]),
        ("git", "rev-list", "HEAD..origin/feature/local-only", "--count"): _completed(
            ["git", "rev-list", "HEAD..origin/feature/local-only", "--count"],
            stdout="0\n",
        ),
    }
    runner = _RunRecorder(responses)
    monkeypatch.setattr(subprocess, "run", runner)

    main.cmd_update(argparse.Namespace())

    assert ["git", "fetch", "origin"] in runner.calls
    assert ["git", "rev-list", "HEAD..origin/feature/local-only", "--count"] in runner.calls
