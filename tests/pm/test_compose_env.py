"""Regression tests for managed PATH growth in issue #126861."""

import json
import os
from pathlib import Path

import pytest

from hermes_platform.resolver import LookupContext, locate_command
from pm.package import Runner, compose_env


def test_compose_env_repeated_application_preserves_unmanaged_path_entries(tmp_path):
    node_dir = str(tmp_path / "managed-node")
    npm_dir = str(tmp_path / "managed-npm")
    caller_dir = str(tmp_path / "caller")
    unmanaged = [caller_dir, caller_dir, "", r"relative\bin", ""]
    diffs = [
        {"PATH": [node_dir], "PRECEDENCE": "dependency"},
        {"PATH": [npm_dir], "PRECEDENCE": "dependent"},
    ]
    base = {
        "Path": os.pathsep.join([npm_dir, node_dir, *unmanaged]),
        "CALLER_VALUE": "preserved",
    }

    first = compose_env(diffs, base=base)
    second = compose_env(diffs, base=first)
    expected_path = [npm_dir, node_dir, *unmanaged]

    assert first["Path"].split(os.pathsep) == expected_path
    assert second["Path"].split(os.pathsep) == expected_path
    assert "PATH" not in second
    assert second["CALLER_VALUE"] == "preserved"
    assert second["PRECEDENCE"] == "dependent"


@pytest.mark.platforms("windows")
def test_compose_env_repeated_managed_path_runs_offline_npm_lifecycle(tmp_path):
    context = LookupContext(path=os.environ.get("PATH", ""), pathext=os.environ.get("PATHEXT"))
    node_resolution = locate_command("node", context)
    npm_resolution = locate_command("npm", context)
    if not node_resolution.command or not npm_resolution.command:
        pytest.skip("Node.js and npm are required for the native Windows lifecycle check")

    node_binary = Path(node_resolution.command[0])
    npm_binary = Path(npm_resolution.command[0])
    node_dir = str(node_binary.parent)
    npm_dir = str(npm_binary.parent)
    managed_dirs = list(dict.fromkeys((npm_dir, node_dir)))
    managed_diffs = [{"PATH": [node_dir]}, {"PATH": [npm_dir]}]
    caller_dir = str(tmp_path / "caller")
    unmanaged = [caller_dir, caller_dir, "", r"relative\bin"]
    original_path = os.pathsep.join([*managed_dirs, *managed_dirs, *unmanaged])
    base_env = {
        key: os.environ[key]
        for key in ("SystemRoot", "SystemDrive", "ComSpec", "PATHEXT")
        if key in os.environ
    }
    base_env.update({
        "PATH": original_path,
        "TEMP": str(tmp_path),
        "TMP": str(tmp_path),
        "USERPROFILE": str(tmp_path),
        "APPDATA": str(tmp_path),
        "LOCALAPPDATA": str(tmp_path),
        "npm_config_userconfig": str(tmp_path / "user.npmrc"),
        "npm_config_globalconfig": str(tmp_path / "global.npmrc"),
        "npm_config_cache": str(tmp_path / "npm-cache"),
        "npm_config_offline": "true",
        "npm_config_update_notifier": "false",
        "npm_config_fund": "false",
        "npm_config_audit": "false",
    })
    version_runner = Runner("npm", base_env)
    node_version = version_runner.run(
        [str(node_binary), "--version"], capture_output=True, text=True, timeout=30,
    )
    npm_version = version_runner.run(
        [str(npm_binary), "--version"], capture_output=True, text=True, timeout=30,
    )
    assert node_version.returncode == 0, node_version.stderr
    assert npm_version.returncode == 0, npm_version.stderr
    assert node_version.stdout.strip().startswith("v")
    assert npm_version.stdout.strip()

    prefix_length = len(os.pathsep.join(managed_dirs)) + len(os.pathsep)
    iterations = max(1, (12_000 - len(original_path)) // prefix_length + 1)

    environment = base_env
    for _ in range(iterations):
        environment = compose_env(managed_diffs, base=environment)

    project = tmp_path / "project"
    project.mkdir()
    (project / "package.json").write_text(
        json.dumps({
            "name": "hermes-path-proof",
            "version": "1.0.0",
            "private": True,
            "scripts": {"smoke": 'node -e "process.exit(0)"'},
        }),
        encoding="utf-8",
    )
    result = Runner("npm", environment).run(
        [str(npm_binary), "run", "smoke", "--offline"],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=45,
    )

    assert result.returncode == 0, result.stdout + result.stderr
