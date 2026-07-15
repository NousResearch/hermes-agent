"""Installed-wheel E2E for the production-shaped canary runtime contract."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tomllib
import venv
import zipfile
from pathlib import Path

import pytest

from gateway.canonical_capability_canary_runtime import (
    CAPABILITY_CREDENTIAL_BINDINGS,
    CAPABILITY_OBSERVER_UNIT,
    CAPABILITY_PRE_CLEANUP_STOP_ORDER,
    CAPABILITY_START_ORDER,
    CAPABILITY_STOP_ORDER,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
@pytest.mark.skipif(os.name == "nt", reason="canary runtime is Linux-only")
def test_installed_wheel_exposes_capability_runtime_and_verifier(tmp_path):
    source = tmp_path / "source"
    shutil.copytree(
        REPO_ROOT,
        source,
        ignore=shutil.ignore_patterns(
            ".git", ".venv", "venv", "build", "dist", "node_modules",
            "__pycache__", "*.pyc",
        ),
    )
    wheel_dir = tmp_path / "wheel"
    built = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(wheel_dir), "."],
        cwd=source,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert built.returncode == 0, built.stderr
    wheels = list(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as archive:
        packaged = set(archive.namelist())
    assert {
        "gateway/canonical_capability_canary_runtime.py",
        "gateway/canonical_capability_canary_e2e.py",
        "gateway/production_runtime_dependencies.py",
        "gateway/discord_connector_bootstrap.py",
        "gateway/discord_connector_protocol.py",
        "gateway/discord_connector_service.py",
        "gateway/relay/discord_connector_transport.py",
        "gateway/mac_ops_edge_client.py",
        "gateway/mac_ops_edge_protocol.py",
        "gateway/mac_ops_edge_service.py",
        "tools/mac_ops_edge_tool.py",
    } <= packaged

    environment = tmp_path / "venv"
    venv.create(environment, with_pip=True)
    python = environment / "bin/python"
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    bootstrap = [
        requirement.split(";", 1)[0]
        for requirement in project["project"]["dependencies"]
        if requirement.split("==", 1)[0].split("[", 1)[0].casefold()
        in {"cryptography", "pyyaml"}
    ]
    subprocess.run(
        [str(python), "-m", "pip", "install", "-q", *bootstrap],
        check=True,
        timeout=300,
    )
    subprocess.run(
        [
            str(python), "-m", "pip", "install", "-q", "--no-deps",
            "--force-reinstall", str(wheels[0]),
        ],
        check=True,
        timeout=300,
    )
    packaged_origin = subprocess.run(
        [
            str(python),
            "-B",
            "-I",
            "-c",
            (
                "from gateway import production_runtime_dependencies as m; "
                "print(m.__file__)"
            ),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env={
            "HOME": str(tmp_path / "empty-home"),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    assert packaged_origin.returncode == 0, packaged_origin.stderr
    assert packaged_origin.stdout.strip().endswith(
        "/site-packages/gateway/production_runtime_dependencies.py"
    )
    packaged_cli = subprocess.run(
        [
            str(python),
            "-B",
            "-I",
            "-m",
            "gateway.production_runtime_dependencies",
            "--help",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env={
            "HOME": str(tmp_path / "empty-home"),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    assert packaged_cli.returncode == 0, packaged_cli.stderr
    assert "{prepare,install,build-manifest,verify}" in packaged_cli.stdout
    completed = subprocess.run(
        [
            str(python), "-B", "-I", "-m",
            "gateway.canonical_capability_canary_runtime", "contract",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env={
            "HOME": str(tmp_path / "empty-home"),
            "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    assert completed.returncode == 0, completed.stderr
    value = json.loads(completed.stdout)
    assert value["normal_gateway_loop"] is True
    assert value["model_semantic_authority"] is True
    assert value["model"] == "gpt-5.6-sol"
    assert "mac_ops" in value["toolsets"]
    assert value["codex_refresh_token_leased"] is False
    assert value["direct_discord_in_gateway"] is False
    assert value["discord_dm_enabled"] is False
    assert value["goal_judge_enabled"] is False
    assert value["goal_continuations_enabled"] is False
    assert value["mcp_auto_discovery_enabled"] is False
    assert value["gateway_event_hooks_enabled"] is False
    assert value["shell_hooks_enabled"] is False
    assert value["plugin_middleware_enabled"] is False
    assert value["plugin_allowlist"] == ["muncho_canary_evidence"]
    assert value["browser_identity"] == "dedicated_create_only_principal"
    assert value["browser_gateway_access"] == (
        "authenticated_af_unix_controller_only"
    )
    assert value["browser_controller_readiness"] == (
        "real_agent_browser_command_round_trip"
    )
    assert value["browser_sandbox"] == "unprivileged_user_namespace_required"
    assert value["terminal_gateway_access"] == (
        "authenticated_af_unix_isolated_worker_only"
    )
    assert value["terminal_network_access"] is False
    assert value["terminal_credentials_available"] is False
    assert value["workspace_policy"] == (
        "ephemeral_isolated_worker_lease_no_host_projection"
    )
    assert value["start_order"] == list(CAPABILITY_START_ORDER)
    assert value["stop_order"] == list(CAPABILITY_STOP_ORDER)
    assert list(CAPABILITY_PRE_CLEANUP_STOP_ORDER) == list(
        reversed(
            [
                unit
                for unit in value["start_order"]
                if unit != CAPABILITY_OBSERVER_UNIT
            ]
        )
    )
    assert value["stop_order"] == [
        *CAPABILITY_PRE_CLEANUP_STOP_ORDER,
        CAPABILITY_OBSERVER_UNIT,
    ]
    assert value["credential_bindings"] == list(CAPABILITY_CREDENTIAL_BINDINGS)
