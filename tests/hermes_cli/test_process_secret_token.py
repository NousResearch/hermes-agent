"""Process bootstrap credentials cross profiles only with operator authority."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


_CHILD = r'''
import json, os, sys
from pathlib import Path
from hermes_cli import env_loader
from agent.secret_sources import registry
from agent.secret_sources.onepassword import OnePasswordSource
from agent.secret_sources.base import SecretSource, FetchResult, source_child_env
import subprocess

root = Path(sys.argv[1])
case = sys.argv[2]
home = root / "profile"
registry._reset_registry_for_tests()
registry.list_sources()
if case == "plugin_replacement":
    registry.register_source(OnePasswordSource(), replace=True, scope=str(home))
if case == "global_plugin":
    registry.register_source(OnePasswordSource(), replace=True)
if case == "subclass":
    class Replacement(OnePasswordSource):
        pass
    registry.register_source(Replacement(), replace=True, scope=str(home))
if case == "launch_residue":
    env_loader._load_dotenv_with_fallback(root / "launch.env", override=True)
    (root / "launch.env").unlink()
if case == "source_residue":
    env_loader._SOURCE_SUPPLIED_NAMES.add("OP_SERVICE_ACCOUNT_TOKEN")
if case == "redirect_policy":
    env_loader._load_dotenv_with_fallback(root / "redirect.env", override=True)
if case == "switch_before_fetch":
    (root / "op").unlink()
    (root / "op").symlink_to(root / "alternative")
if case == "switch_at_exec":
    import agent.secret_sources.onepassword as op
    original_run = op.run_cli
    def switch(argv, **kwargs):
        (root / "op").unlink()
        (root / "op").symlink_to(root / "alternative")
        return original_run(argv, **kwargs)
    op.run_cli = switch
if case == "concurrent_sources":
    import agent.secret_sources.bitwarden as bw
    bw.find_bws = lambda **kwargs: root / "bws"
    class Observer(SecretSource):
        name = "observer"
        token_env_key = "token_env"
        default_token_env = "OP_SERVICE_ACCOUNT_TOKEN"
        def fetch(self, cfg, home_path):
            subprocess.run([sys.executable, str(root / "observer")],
                           env=source_child_env(), check=True, capture_output=True)
            return FetchResult()
    registry.register_source(Observer(), scope=str(home))

original_apply = registry.apply_all
common = {}
def capture(cfg, path, environ=None):
    common.update(environ)
    return original_apply(cfg, path, environ=environ)
registry.apply_all = capture
before = dict(os.environ)
values = env_loader.hydrate_profile_secret_sources(home)
snapshot = env_loader.get_secret_source_values(home)
assert dict(os.environ) == before
from agent import secret_scope
from hermes_constants import set_hermes_home_override, reset_hermes_home_override
secret_scope.set_multiplex_active(True)
home_token = set_hermes_home_override(home)
scope = secret_scope.build_profile_secret_scope(home)
token = secret_scope.set_secret_scope(scope, profile_home=str(home))
try:
    from tools.environments.local import build_subprocess_env
    from tools.env_passthrough import register_env_passthrough
    # Explicit skill passthrough must still resolve through the routed scope.
    register_env_passthrough(["OP_SERVICE_ACCOUNT_TOKEN", "BWS_ACCESS_TOKEN"])
    tool_env = build_subprocess_env()
    secret = secret_scope.get_secret("OP_SERVICE_ACCOUNT_TOKEN")
finally:
    secret_scope.reset_secret_scope(token)
    reset_hermes_home_override(home_token)
print(json.dumps(dict(values=values, snapshot=snapshot, common=common,
                      scope=scope, secret=secret, tool_env=tool_env)))
'''


def _helper(path: Path, report: Path, *, mode="op"):
    path.write_text(
        f"#!{sys.executable}\n"
        "import json, os\n"
        f"with open({str(report)!r}, 'w') as f: json.dump(dict(os.environ), f)\n"
        + {"op": "print('vault-value' if os.environ.get('OP_SERVICE_ACCOUNT_TOKEN') else '')\n",
           "bws": "print('[]')\n", "command": "print('COMMAND_KEY=command-value')\n"}[mode]
    )
    path.chmod(0o755)


def _hydrate(tmp_path, case, *, dotenv=None, op_env=None):
    home = tmp_path / "profile"
    home.mkdir()
    operator = tmp_path / "operator"
    operator.mkdir()
    _helper(tmp_path / "trusted", tmp_path / "op.json")
    _helper(tmp_path / "alternative", tmp_path / "alternative.json")
    (tmp_path / "op").symlink_to(tmp_path / "trusted")
    op_cfg = dict(enabled=True, env={"VAULT_KEY": "op://vault/item/password"},
                  binary_path=str(tmp_path / "op"), cache_ttl_seconds=0)
    policy = {"secrets": {"onepassword": {
        "share_process_token": True, "binary_path": str(tmp_path / "op")}}}
    if case in ("missing_opt_in", "profile_opt_in", "redirect_policy"):
        policy = {}
    if case == "renamed_token":
        op_cfg["service_account_token_env"] = "RENAMED_TOKEN"
    if case == "alternative_executable":
        op_cfg["binary_path"] = str(tmp_path / "alternative")
    op_cfg["share_process_token"] = case != "missing_opt_in"
    cfg = {"secrets": {"onepassword": op_cfg}}
    if case == "concurrent_sources":
        _helper(tmp_path / "bws", tmp_path / "bws.json", mode="bws")
        _helper(tmp_path / "command", tmp_path / "command.json", mode="command")
        _helper(tmp_path / "observer", tmp_path / "observer.json", mode="command")
        cfg["secrets"].update(
            command={"enabled": True, "command": shlex.quote(str(tmp_path / "command"))},
            observer={"enabled": True},
            bitwarden={"enabled": True, "project_id": "test", "auto_install": False},
        )
    (home / "config.yaml").write_text(json.dumps(cfg))
    (operator / "config.yaml").write_text(json.dumps(policy))
    if dotenv is not None:
        (home / ".env").write_text(dotenv)
    if op_env is not None:
        (home / ".op.env").write_text(op_env)
    (tmp_path / "launch.env").write_text("OP_SERVICE_ACCOUNT_TOKEN=launch-residue\n")
    rogue = tmp_path / "rogue"
    rogue.mkdir()
    (rogue / "config.yaml").write_text(json.dumps({"secrets": {"onepassword": {
        "share_process_token": True, "binary_path": str(tmp_path / "op")}}}))
    (tmp_path / "redirect.env").write_text(f"HERMES_MANAGED_DIR={rogue}\n")
    child_env = dict(os.environ, HERMES_HOME=str(tmp_path / "launch"),
                     HERMES_MANAGED_DIR=str(operator), OP_SERVICE_ACCOUNT_TOKEN="process-token",
                     BWS_ACCESS_TOKEN="process-bitwarden", RENAMED_TOKEN="renamed-process-token")
    child_env.pop("OP_CONNECT_HOST", None)
    if case == "connect_process":
        child_env["OP_CONNECT_HOST"] = "https://connect.invalid"
    proc = subprocess.run([sys.executable, "-c", _CHILD, str(tmp_path), case],
                          env=child_env, capture_output=True, text=True, timeout=60, check=False)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    result["children"] = {p.stem: json.loads(p.read_text()) for p in tmp_path.glob("*.json")}
    return result


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("case", [
    "missing_opt_in", "profile_opt_in", "launch_residue", "source_residue",
    "renamed_token", "connect_process", "connect_profile", "alternative_executable",
    "plugin_replacement", "global_plugin", "subclass", "redirect_policy", "switch_before_fetch",
])
def test_process_token_denied(tmp_path, case):
    result = _hydrate(tmp_path, case, dotenv=(
        "OP_CONNECT_HOST=https://connect.invalid\n" if case == "connect_profile" else None))
    assert result["values"] == {}
    assert "OP_SERVICE_ACCOUNT_TOKEN" not in result["common"]
    for env in result["children"].values():
        assert "OP_SERVICE_ACCOUNT_TOKEN" not in env
        assert "BWS_ACCESS_TOKEN" not in env
        assert "RENAMED_TOKEN" not in env


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("case", ["trusted", "switch_at_exec", "concurrent_sources"])
def test_authorized_token_is_fetch_private(tmp_path, case):
    result = _hydrate(tmp_path, case)
    assert result["values"]["VAULT_KEY"] == "vault-value"
    assert result["snapshot"] == result["values"]
    assert result["children"]["op"]["OP_SERVICE_ACCOUNT_TOKEN"] == "process-token"
    assert "alternative" not in result["children"]
    for key in ("common", "snapshot", "scope", "tool_env"):
        assert "OP_SERVICE_ACCOUNT_TOKEN" not in result[key]
        assert "BWS_ACCESS_TOKEN" not in result[key]
    assert result["secret"] is None
    if case == "concurrent_sources":
        assert result["values"]["COMMAND_KEY"] == "command-value"
        assert "bws" not in result["children"]
        for name in ("command", "observer"):
            assert "OP_SERVICE_ACCOUNT_TOKEN" not in result["children"][name]
            assert "BWS_ACCESS_TOKEN" not in result["children"][name]


@pytest.mark.platforms("posix")
@pytest.mark.parametrize(("dotenv", "op_env", "expected"), [
    (None, "OP_SERVICE_ACCOUNT_TOKEN=op-file\n", "op-file"),
    ("OP_SERVICE_ACCOUNT_TOKEN=profile-file\n", "OP_SERVICE_ACCOUNT_TOKEN=op-file\n", "profile-file"),
    ("OP_SERVICE_ACCOUNT_TOKEN=\n", "OP_SERVICE_ACCOUNT_TOKEN=op-file\n", None),
])
def test_profile_credentials_precede_shared_token(tmp_path, dotenv, op_env, expected):
    result = _hydrate(tmp_path, "trusted", dotenv=dotenv, op_env=op_env)
    assert result["children"]["op"].get("OP_SERVICE_ACCOUNT_TOKEN") == expected
    assert result["values"] == ({"VAULT_KEY": "vault-value"} if expected else {})


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("case", ["plugin_replacement", "subclass", "alternative_executable", "renamed_token"])
def test_excluded_sources_keep_profile_credentials(tmp_path, case):
    token_name = "RENAMED_TOKEN" if case == "renamed_token" else "OP_SERVICE_ACCOUNT_TOKEN"
    result = _hydrate(tmp_path, case, op_env=f"{token_name}=profile-token\n")
    assert result["values"] == {"VAULT_KEY": "vault-value"}
    name = "alternative" if case == "alternative_executable" else "op"
    assert result["children"][name]["OP_SERVICE_ACCOUNT_TOKEN"] == "profile-token"


@pytest.mark.platforms("posix")
def test_other_sources_keep_profile_local_bootstrap(tmp_path):
    result = _hydrate(tmp_path, "concurrent_sources", dotenv=(
        "OP_SERVICE_ACCOUNT_TOKEN=profile-op\nBWS_ACCESS_TOKEN=profile-bws\n"))
    assert result["values"]["VAULT_KEY"] == "vault-value"
    assert result["values"]["COMMAND_KEY"] == "command-value"
    assert result["children"]["bws"]["BWS_ACCESS_TOKEN"] == "profile-bws"
    assert result["children"]["bws"]["OP_SERVICE_ACCOUNT_TOKEN"] == "profile-op"
    for name in ("command", "observer"):
        assert result["children"][name]["OP_SERVICE_ACCOUNT_TOKEN"] == "profile-op"
        assert result["children"][name]["BWS_ACCESS_TOKEN"] == "profile-bws"
