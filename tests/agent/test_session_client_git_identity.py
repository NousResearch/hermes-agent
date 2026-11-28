"""Regression for #155010: a declared CLI Git identity reaches execute_code via client_env."""

import os
import subprocess
import sys

from agent.session_client_env import (
    accept_client_env, client_env_rules, client_env_scope, collect_client_env,
)
from tools.code_execution_env import _build_child_env


IDENTITY = {
    "GIT_AUTHOR_NAME": "Agent",
    "GIT_AUTHOR_EMAIL": "agent@example.com",
    "GIT_AUTHOR_DATE": "2026-11-13T20:05:00+08:00",
    "GIT_COMMITTER_NAME": "Agent",
    "GIT_COMMITTER_EMAIL": "agent@example.com",
    "GIT_COMMITTER_DATE": "2026-11-13T20:05:00+08:00",
}


def test_local_cli_git_identity_survives_client_policy_and_code_child(tmp_path, monkeypatch):
    """The client, gateway, and code child must agree on all six declared identity names."""
    config = {"terminal": {"client_env_passthrough": [*IDENTITY, "HTTP_AUTHORIZATION"]}}
    source = {"PATH": os.environ.get("PATH", "/usr/bin"), **IDENTITY,
              "HTTP_AUTHORIZATION": "Bearer fake"}
    rules = client_env_rules(config)
    sent = collect_client_env(source, config)
    accepted = accept_client_env(sent, config)
    assert {name: accepted.get(name) for name in IDENTITY} == IDENTITY
    assert "HTTP_AUTHORIZATION" not in accepted

    # The server process does not have the client's Git identity in its own environment.
    for name in IDENTITY:
        monkeypatch.delenv(name, raising=False)
    with client_env_scope(accepted, local=True, rules=rules):
        child = _build_child_env(rpc_endpoint="unix:///nonexistent.sock", rpc_token="token",
                                 tmpdir=str(tmp_path), child_python=sys.executable)
    assert {name: child.get(name) for name in IDENTITY} == IDENTITY

    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (("init", "-q"), ("config", "user.name", "Repo Default"),
                 ("config", "user.email", "repo@example.com")):
        subprocess.run(["git", "-C", str(repo), *args], env=child, check=True, capture_output=True)
    for role in ("AUTHOR", "COMMITTER"):
        ident = subprocess.run(["git", "-C", str(repo), "var", f"GIT_{role}_IDENT"],
                               env=child, check=True, capture_output=True, text=True)
        assert ident.stdout.startswith("Agent <agent@example.com> "), ident.stdout
