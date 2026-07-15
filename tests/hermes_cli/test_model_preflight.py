from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import model_preflight as mp
from hermes_cli.model_preflight import (
    ModelPreflightResult,
    evaluate_model_preflight,
    format_model_preflight,
)
from hermes_cli.subcommands.model import build_model_parser


def _inputs(tmp_path: Path) -> dict:
    home = tmp_path / "profiles" / "main-gateway-v2"
    source = tmp_path / "worktree"
    return {
        "profile_home": home,
        "config": {
            "model": {"provider": "openai-codex", "default": "gpt-5.6-sol"}
        },
        "runtime": {
            "pid": 17420,
            "gateway_state": "running",
            "argv": [str(source / "hermes_cli" / "main.py"), "gateway", "run"],
        },
        "process_command": (
            f"{source}/.venv/bin/python -m hermes_cli.main "
            "--profile main-gateway-v2 gateway run --replace"
        ),
        "auth": {
            "providers": {"openai-codex": {"tokens": {"access_token": "SECRET"}}},
            "credential_pool": {"openai-codex": [{"access_token": "SECRET"}]},
        },
        "current_source_root": source,
    }


def test_matching_profile_source_and_auth_pass(tmp_path):
    result = evaluate_model_preflight(**_inputs(tmp_path))

    assert result.status == "PASS"
    assert result.profile_match is True
    assert result.source_match is True
    assert result.auth_present is True
    assert result.gateway_pid == 17420
    assert result.configured_provider == "openai-codex"
    assert result.configured_model == "gpt-5.6-sol"
    assert result.blockers == ()


def test_explicit_other_profile_fails_closed(tmp_path):
    inputs = _inputs(tmp_path)
    inputs["process_command"] = inputs["process_command"].replace(
        "--profile main-gateway-v2", "--profile private-engineer"
    )

    result = evaluate_model_preflight(**inputs)

    assert result.status == "FAIL"
    assert result.profile_match is False
    assert "gateway_profile_mismatch" in result.blockers


def test_other_gateway_source_fails_closed(tmp_path):
    inputs = _inputs(tmp_path)
    inputs["runtime"]["argv"][0] = str(
        tmp_path / "inactive-source" / "hermes_cli" / "main.py"
    )

    result = evaluate_model_preflight(**inputs)

    assert result.status == "FAIL"
    assert result.source_match is False
    assert "gateway_source_mismatch" in result.blockers


def test_missing_gateway_is_inconclusive(tmp_path):
    inputs = _inputs(tmp_path)
    inputs["runtime"] = None
    inputs["process_command"] = ""

    result = evaluate_model_preflight(**inputs)

    assert result.status == "INCONCLUSIVE"
    assert result.gateway_pid is None
    assert "gateway_not_running" in result.blockers


def test_missing_codex_auth_fails_closed(tmp_path):
    inputs = _inputs(tmp_path)
    inputs["auth"] = {"providers": {}, "credential_pool": {}}

    result = evaluate_model_preflight(**inputs)

    assert result.status == "FAIL"
    assert result.auth_present is False
    assert "configured_provider_auth_missing" in result.blockers


def test_json_output_never_contains_credential_values(tmp_path):
    inputs = _inputs(tmp_path)
    inputs["process_command"] += " --api-key COMMAND_SECRET --token=INLINE_SECRET"
    result = evaluate_model_preflight(**inputs)

    payload = format_model_preflight(result, as_json=True)
    parsed = json.loads(payload)

    assert parsed["status"] == "PASS"
    assert "SECRET" not in payload
    assert "access_token" not in payload
    assert "COMMAND_SECRET" not in payload
    assert "INLINE_SECRET" not in payload
    assert "gateway_command" not in parsed


def test_anthropic_api_key_configuration_does_not_require_oauth_pool(tmp_path):
    inputs = _inputs(tmp_path)
    inputs["config"]["model"] = {
        "provider": "anthropic",
        "default": "claude-sonnet-4",
    }
    inputs["auth"] = {}

    result = evaluate_model_preflight(**inputs)

    assert result.status == "PASS"
    assert result.auth_present is True


def test_current_worktree_console_script_source_is_verified(tmp_path):
    inputs = _inputs(tmp_path)
    source = inputs["current_source_root"]
    inputs["runtime"]["argv"][0] = str(source / ".venv" / "bin" / "hermes")
    inputs["process_command"] = (
        f"{source}/.venv/bin/python {source}/.venv/bin/hermes "
        "--profile main-gateway-v2 gateway run"
    )

    result = evaluate_model_preflight(**inputs)

    assert result.status == "PASS"
    assert result.source_match is True
    assert result.gateway_source_root == str(source.resolve())


def test_collect_rejects_stale_runtime_pid(monkeypatch, tmp_path):
    home = tmp_path / "profiles" / "main-gateway-v2"
    home.mkdir(parents=True)
    (home / "auth.json").write_text(
        json.dumps({"credential_pool": {"openai-codex": [{"id": "present"}]}}),
        encoding="utf-8",
    )
    runtime = {
        "pid": 99999,
        "gateway_state": "running",
        "argv": [str(Path(mp.__file__).resolve().parents[1] / "hermes_cli" / "main.py")],
    }
    monkeypatch.setattr(mp, "get_hermes_home", lambda: home)
    monkeypatch.setattr(mp, "get_config_path", lambda: home / "config.yaml")
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"model": {"provider": "openai-codex", "default": "gpt-5.6-sol"}},
    )
    monkeypatch.setattr("gateway.status.read_runtime_status", lambda: runtime)
    monkeypatch.setattr(
        "gateway.status.get_runtime_status_running_pid", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "gateway.status._read_process_cmdline",
        lambda _pid: (_ for _ in ()).throw(AssertionError("stale PID must not be read")),
    )

    result = mp.collect_model_preflight()

    assert result.status == "INCONCLUSIVE"
    assert result.gateway_pid is None
    assert "gateway_not_running" in result.blockers


def test_model_parser_accepts_readonly_preflight_flags():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    handler = lambda _args: None
    build_model_parser(subparsers, cmd_model=handler)

    args = parser.parse_args(["model", "--preflight", "--json"])

    assert args.preflight is True
    assert args.json is True
    assert args.func is handler


def test_cmd_model_preflight_does_not_require_tty(monkeypatch, capsys):
    from hermes_cli import main

    result = ModelPreflightResult(
        profile_home="/profile/main-gateway-v2",
        config_path="/profile/main-gateway-v2/config.yaml",
        configured_provider="openai-codex",
        configured_model="gpt-5.6-sol",
        gateway_pid=17420,
        gateway_command="python -m hermes_cli.main --profile main-gateway-v2 gateway run",
        gateway_interpreter="python",
        gateway_source_root="/source/worktree",
        profile_match=True,
        source_match=True,
        auth_present=True,
        status="PASS",
        blockers=(),
    )
    monkeypatch.setattr(
        "hermes_cli.model_preflight.collect_model_preflight", lambda: result
    )
    monkeypatch.setattr(
        main,
        "_require_tty",
        lambda _name: (_ for _ in ()).throw(AssertionError("TTY must not be required")),
    )

    rc = main.cmd_model(SimpleNamespace(preflight=True, json=True, refresh=False))

    assert rc == 0
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"
