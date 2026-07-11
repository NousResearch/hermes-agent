import json
import os
import subprocess
from pathlib import Path

import pytest

from agent.claude_cli_boundary import (
    attest_claude_max_auth,
    create_exact_env_cli_wrapper,
)
from agent.claude_workspace_terminal import build_workspace_seatbelt_profile


def _write_probe(path: Path, payload: dict | None = None) -> None:
    if payload is None:
        body = """#!/usr/bin/env python3
import json, os
print(json.dumps(dict(os.environ), sort_keys=True))
"""
    else:
        body = """#!/usr/bin/env python3
import json
print(json.dumps(%s))
""" % repr(payload)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o700)


def test_wrapper_gives_actual_child_an_exact_environment(tmp_path):
    probe = tmp_path / "probe.py"
    _write_probe(probe)
    exact = {
        "HOME": str(tmp_path / "home"),
        "PATH": "/usr/bin:/bin",
        "USER": "worker",
        "LOGNAME": "worker",
        "DISABLE_TELEMETRY": "1",
    }
    wrapper = create_exact_env_cli_wrapper(probe, exact, tmp_path / "wrappers")
    ambient = {
        **os.environ,
        "ANTHROPIC_API_KEY": "sentinel-paid-key",
        "ANTHROPIC_AUTH_TOKEN": "sentinel-auth-token",
        "OPENAI_API_KEY": "sentinel-openai-key",
        "AMBIENT_SENTINEL_SECRET": "must-not-cross",
    }

    result = subprocess.run(
        [str(wrapper)],
        env=ambient,
        capture_output=True,
        text=True,
        check=True,
    )
    child_env = json.loads(result.stdout)

    assert {key: child_env.get(key) for key in exact} == exact
    assert "ANTHROPIC_API_KEY" not in child_env
    assert "ANTHROPIC_AUTH_TOKEN" not in child_env
    assert "AMBIENT_SENTINEL_SECRET" not in child_env
    assert wrapper.stat().st_mode & 0o077 == 0


@pytest.mark.skipif(os.uname().sysname != "Darwin", reason="macOS sandbox-exec")
def test_wrapper_process_boundary_blocks_symlink_write_escape(tmp_path):
    workspace = tmp_path / "worktree"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("outside-secret", encoding="utf-8")
    (workspace / "escape").symlink_to(outside, target_is_directory=True)
    probe = workspace / "probe.sh"
    probe.write_text(
        "#!/bin/sh\ntouch allowed.txt\ntouch escape/escaped.txt\n"
        "cat escape/secret.txt > leaked.txt\n",
        encoding="utf-8",
    )
    probe.chmod(0o700)
    profile = build_workspace_seatbelt_profile(
        workspace=workspace,
        host_home=tmp_path / "host",
        allow_network=True,
        readable_roots=[workspace],
    )
    wrapper = create_exact_env_cli_wrapper(
        probe,
        {"HOME": str(workspace), "PATH": "/usr/bin:/bin"},
        tmp_path / "wrappers",
        sandbox_profile=profile,
    )

    subprocess.run([str(wrapper)], cwd=workspace, check=False)

    assert (workspace / "allowed.txt").exists()
    assert not (outside / "escaped.txt").exists()
    assert not (workspace / "leaked.txt").read_text(encoding="utf-8")


def test_cli_profile_keeps_network_while_terminal_profile_denies_it(tmp_path):
    workspace = tmp_path / "worktree"
    workspace.mkdir()
    cli_profile = build_workspace_seatbelt_profile(
        workspace=workspace,
        host_home=tmp_path / "host",
        allow_network=True,
        restrict_reads=False,
    )
    terminal_profile = build_workspace_seatbelt_profile(
        workspace=workspace,
        host_home=tmp_path / "host",
        allow_network=False,
    )

    assert "(deny network*)" not in cli_profile
    assert "(deny network*)" in terminal_profile


def test_auth_attestation_requires_first_party_max_on_same_wrapper(tmp_path):
    good = tmp_path / "good.py"
    _write_probe(
        good,
        {
            "loggedIn": True,
            "authMethod": "claude.ai",
            "apiProvider": "firstParty",
            "subscriptionType": "max",
        },
    )
    wrapper = create_exact_env_cli_wrapper(
        good,
        {"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
        tmp_path / "good-wrapper",
    )

    attestation = attest_claude_max_auth(wrapper, cache_ttl_seconds=0)

    assert attestation.subscription_type == "max"
    assert attestation.included_usage is True


@pytest.mark.parametrize(
    "override",
    [
        {"loggedIn": False},
        {"authMethod": "apiKey"},
        {"apiProvider": "thirdParty"},
        {"subscriptionType": "pro"},
    ],
)
def test_auth_attestation_fails_closed_for_non_max_routes(tmp_path, override):
    payload = {
        "loggedIn": True,
        "authMethod": "claude.ai",
        "apiProvider": "firstParty",
        "subscriptionType": "max",
    }
    payload.update(override)
    probe = tmp_path / "bad.py"
    _write_probe(probe, payload)
    wrapper = create_exact_env_cli_wrapper(
        probe,
        {"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
        tmp_path / "bad-wrapper",
    )

    with pytest.raises(RuntimeError, match="Claude Max attestation failed"):
        attest_claude_max_auth(wrapper, cache_ttl_seconds=0)
