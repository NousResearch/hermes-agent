from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.model_preflight import ModelPreflightResult
from hermes_cli.model_transaction import (
    ConfigRevisionConflict,
    ModelPair,
    ModelSmokeEvidence,
    TransactionHooks,
    execute_model_transaction,
    format_model_transaction,
)


def preflight(tmp_path: Path, *, status: str = "PASS") -> ModelPreflightResult:
    home = tmp_path / "profiles" / "main-gateway-v2"
    return ModelPreflightResult(
        profile_home=str(home),
        config_path=str(home / "config.yaml"),
        configured_provider="openai-codex",
        configured_model="gpt-5.6-sol",
        gateway_pid=100,
        gateway_command="python -m hermes_cli.main --profile main-gateway-v2 gateway run",
        gateway_interpreter="/runtime/.venv/bin/python",
        gateway_source_root="/runtime/source",
        profile_match=True,
        source_match=True,
        auth_present=True,
        status=status,
        blockers=() if status == "PASS" else ("blocked",),
    )


def make_hooks(tmp_path: Path, *, target_smoke_ok: bool = True):
    state = {
        "config": {
            "display": {"theme": "dark"},
            "model": {
                "provider": "openai-codex",
                "default": "gpt-5.6-sol",
                "context_length": 1000,
            },
        },
        "writes": [],
        "restarts": [],
        "smokes": [],
    }

    def read_config():
        import copy

        return copy.deepcopy(state["config"])

    def write_config(expected, config):
        import copy

        if state["config"] != expected:
            raise ConfigRevisionConflict("config_revision_conflict")
        state["config"] = copy.deepcopy(config)
        state["writes"].append(copy.deepcopy(config))

    def restart(previous_pid):
        state["restarts"].append(previous_pid)
        return 100 + len(state["restarts"])

    def smoke(pair, interpreter, profile):
        state["smokes"].append((pair, interpreter, profile))
        ok = target_smoke_ok or pair.provider == "openai-codex"
        return ModelSmokeEvidence(
            provider=pair.provider if ok else "wrong-provider",
            model=pair.model if ok else "wrong-model",
            api_calls=1 if ok else 0,
            completed=ok,
        )

    hooks = TransactionHooks(
        collect_preflight=lambda: preflight(tmp_path),
        read_config=read_config,
        write_config=write_config,
        restart_gateway=restart,
        smoke_model=smoke,
    )
    return hooks, state


def test_preview_is_read_only_and_preserves_non_model_config(tmp_path):
    hooks, state = make_hooks(tmp_path)

    result = execute_model_transaction(
        ModelPair("openai-codex", "gpt-next"),
        confirm_profile="main-gateway-v2",
        apply=False,
        hooks=hooks,
    )

    assert result.status == "PREVIEW"
    assert result.previous == ModelPair("openai-codex", "gpt-5.6-sol")
    assert result.target == ModelPair("openai-codex", "gpt-next")
    assert state["writes"] == []
    assert state["restarts"] == []
    assert state["config"]["display"] == {"theme": "dark"}


def test_apply_of_current_pair_is_noop(tmp_path):
    hooks, state = make_hooks(tmp_path)

    result = execute_model_transaction(
        ModelPair("openai-codex", "gpt-5.6-sol"),
        confirm_profile="main-gateway-v2",
        apply=True,
        hooks=hooks,
    )

    assert result.status == "NOOP"
    assert state["writes"] == []
    assert state["restarts"] == []
    assert state["smokes"] == []


def test_wrong_confirmed_profile_blocks_before_write(tmp_path):
    hooks, state = make_hooks(tmp_path)

    result = execute_model_transaction(
        ModelPair("openai-codex", "gpt-next"),
        confirm_profile="private-engineer",
        apply=True,
        hooks=hooks,
    )

    assert result.status == "BLOCKED"
    assert "confirmed_profile_mismatch" in result.blockers
    assert state["writes"] == []


def test_non_pass_preflight_blocks_before_write(tmp_path):
    hooks, state = make_hooks(tmp_path)
    hooks.collect_preflight = lambda: preflight(tmp_path, status="INCONCLUSIVE")

    result = execute_model_transaction(
        ModelPair("openai-codex", "gpt-next"),
        confirm_profile="main-gateway-v2",
        apply=True,
        hooks=hooks,
    )

    assert result.status == "BLOCKED"
    assert "preflight_not_pass" in result.blockers
    assert state["writes"] == []


def test_success_requires_readback_new_pid_and_matching_api_evidence(tmp_path):
    hooks, state = make_hooks(tmp_path)
    target = ModelPair("openai-codex", "gpt-next")

    result = execute_model_transaction(
        target,
        confirm_profile="main-gateway-v2",
        apply=True,
        hooks=hooks,
    )

    assert result.status == "PASS"
    assert result.new_pid == 101
    assert result.smoke == ModelSmokeEvidence(
        provider="openai-codex", model="gpt-next", api_calls=1, completed=True
    )
    assert state["config"]["model"] == {
        "provider": "openai-codex",
        "default": "gpt-next",
        "context_length": 1000,
    }
    assert state["config"]["display"] == {"theme": "dark"}
    assert state["restarts"] == [100]


def test_failed_target_smoke_restores_previous_pair_and_verifies_rollback(tmp_path):
    hooks, state = make_hooks(tmp_path, target_smoke_ok=False)

    result = execute_model_transaction(
        ModelPair("broken-provider", "broken-model"),
        confirm_profile="main-gateway-v2",
        apply=True,
        hooks=hooks,
    )

    assert result.status == "ROLLED_BACK"
    assert result.rollback_verified is True
    assert state["config"]["model"]["provider"] == "openai-codex"
    assert state["config"]["model"]["default"] == "gpt-5.6-sol"
    assert state["config"]["model"]["context_length"] == 1000
    assert state["restarts"] == [100, 101]
    assert len(state["writes"]) == 2


def test_rollback_verification_failure_is_explicit(tmp_path):
    hooks, state = make_hooks(tmp_path, target_smoke_ok=False)
    hooks.smoke_model = lambda pair, interpreter, profile: ModelSmokeEvidence(
        provider="wrong", model="wrong", api_calls=0, completed=False
    )

    result = execute_model_transaction(
        ModelPair("broken-provider", "broken-model"),
        confirm_profile="main-gateway-v2",
        apply=True,
        hooks=hooks,
    )

    assert result.status == "ROLLBACK_FAILED"
    assert result.rollback_verified is False
    assert "rollback_verification_failed" in result.blockers
    assert state["config"]["model"]["provider"] == "openai-codex"


def test_target_smoke_timeout_rolls_back_and_reports_stable_code(tmp_path):
    hooks, state = make_hooks(tmp_path)
    calls = 0

    def timed_smoke(pair, interpreter, profile):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("model_smoke_timeout")
        return ModelSmokeEvidence(pair.provider, pair.model, 1, True)

    hooks.smoke_model = timed_smoke
    result = execute_model_transaction(
        ModelPair("slow-provider", "slow-model"),
        confirm_profile="main-gateway-v2",
        apply=True,
        hooks=hooks,
    )

    assert result.status == "ROLLED_BACK"
    assert result.blockers == ("model_smoke_timeout",)
    assert result.rollback_verified is True
    assert state["config"]["model"]["provider"] == "openai-codex"
    assert len(state["restarts"]) == 2


def test_preflight_config_drift_blocks_before_write(tmp_path):
    hooks, state = make_hooks(tmp_path)
    state["config"]["model"]["default"] = "externally-changed"

    result = execute_model_transaction(
        ModelPair("openai-codex", "gpt-next"),
        confirm_profile="main-gateway-v2",
        apply=True,
        hooks=hooks,
    )

    assert result.status == "BLOCKED"
    assert result.blockers == ("preflight_config_drift",)
    assert state["writes"] == []


def test_config_revision_conflict_does_not_clobber_external_edit(tmp_path):
    hooks, state = make_hooks(tmp_path)
    real_write = hooks.write_config

    def conflicting_write(expected, config):
        state["config"]["display"]["theme"] = "external"
        real_write(expected, config)

    hooks.write_config = conflicting_write
    result = execute_model_transaction(
        ModelPair("openai-codex", "gpt-next"),
        confirm_profile="main-gateway-v2",
        apply=True,
        hooks=hooks,
    )

    assert result.status == "FAILED"
    assert result.blockers == ("config_revision_conflict",)
    assert state["config"]["display"]["theme"] == "external"
    assert state["config"]["model"]["default"] == "gpt-5.6-sol"
    assert state["restarts"] == []


def test_rollback_conflict_never_overwrites_third_party_model_change(tmp_path):
    hooks, state = make_hooks(tmp_path)

    def conflicting_smoke(pair, interpreter, profile):
        state["config"]["model"]["provider"] = "third-party"
        state["config"]["model"]["default"] = "third-model"
        return ModelSmokeEvidence("wrong", "wrong", 0, False)

    hooks.smoke_model = conflicting_smoke
    result = execute_model_transaction(
        ModelPair("broken-provider", "broken-model"),
        confirm_profile="main-gateway-v2",
        apply=True,
        hooks=hooks,
    )

    assert result.status == "ROLLBACK_FAILED"
    assert "rollback_conflict" in result.blockers
    assert state["config"]["model"]["provider"] == "third-party"
    assert state["config"]["model"]["default"] == "third-model"
    assert len(state["writes"]) == 1


def test_apply_is_blocked_when_gateway_is_an_ancestor(tmp_path):
    hooks, state = make_hooks(tmp_path)
    hooks.restart_allowed = lambda: False

    result = execute_model_transaction(
        ModelPair("openai-codex", "gpt-next"),
        confirm_profile="main-gateway-v2",
        apply=True,
        hooks=hooks,
    )

    assert result.status == "BLOCKED"
    assert "transaction_restart_not_supervisor_safe" in result.blockers
    assert state["writes"] == []


def test_transaction_output_never_serializes_raw_config_or_auth(tmp_path):
    hooks, state = make_hooks(tmp_path)
    state["config"]["model"]["api_key"] = "TOP-SECRET-VALUE"
    state["config"]["model"]["base_url"] = "https://secret.example.invalid"

    result = execute_model_transaction(
        ModelPair("openai-codex", "gpt-next"),
        confirm_profile="main-gateway-v2",
        apply=False,
        hooks=hooks,
    )
    rendered = format_model_transaction(result, as_json=True)

    assert "TOP-SECRET-VALUE" not in rendered
    assert "secret.example.invalid" not in rendered
    assert "api_key" not in rendered
    assert "gateway_command" not in rendered


def test_parser_accepts_transaction_preview_and_apply_flags():
    import argparse

    from hermes_cli.subcommands.model import build_model_parser

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_model_parser(subparsers, cmd_model=lambda _args: None)

    args = parser.parse_args(
        [
            "model",
            "--provider",
            "openai-codex",
            "--model",
            "gpt-next",
            "--confirm-profile",
            "main-gateway-v2",
            "--apply-transaction",
            "--json",
        ]
    )

    assert args.transaction_provider == "openai-codex"
    assert args.transaction_model == "gpt-next"
    assert args.confirm_profile == "main-gateway-v2"
    assert args.apply_transaction is True
    assert args.json is True


def test_parser_rejects_non_positive_transaction_timeouts():
    import argparse

    from hermes_cli.subcommands.model import build_model_parser

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_model_parser(subparsers, cmd_model=lambda _args: None)

    with pytest.raises(SystemExit):
        parser.parse_args(["model", "--restart-timeout", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args(["model", "--smoke-timeout", "-1"])


def test_cmd_model_preview_bypasses_tty_and_does_not_write(monkeypatch, tmp_path, capsys):
    from hermes_cli import main

    hooks, state = make_hooks(tmp_path)
    monkeypatch.setattr(
        "hermes_cli.model_transaction.build_production_hooks", lambda **_kwargs: hooks
    )
    monkeypatch.setattr(
        main,
        "_require_tty",
        lambda _name: (_ for _ in ()).throw(AssertionError("TTY must not be required")),
    )

    rc = main.cmd_model(
        SimpleNamespace(
            transaction_provider="openai-codex",
            transaction_model="gpt-next",
            confirm_profile="main-gateway-v2",
            apply_transaction=False,
            preflight=False,
            json=True,
            refresh=False,
        )
    )

    assert rc == 0
    assert '"status": "PREVIEW"' in capsys.readouterr().out
    assert state["writes"] == []
    assert state["restarts"] == []


def test_production_config_write_uses_revision_cas(monkeypatch, tmp_path):
    from hermes_cli import model_transaction as transaction

    live_preflight = preflight(tmp_path)
    home = Path(live_preflight.profile_home)
    home.mkdir(parents=True)
    source = tmp_path / "runtime-source"
    source.mkdir()
    live_preflight = ModelPreflightResult(
        **{
            **live_preflight.__dict__,
            "gateway_source_root": str(source),
        }
    )
    state = {
        "config": {
            "model": {"provider": "openai-codex", "default": "gpt-5.6-sol"},
            "display": {"theme": "dark"},
        }
    }
    monkeypatch.setattr(
        "hermes_cli.model_preflight.collect_model_preflight",
        lambda: live_preflight,
    )
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config",
        lambda: copy.deepcopy(state["config"]),
    )

    def fake_save(config, **_kwargs):
        state["config"] = copy.deepcopy(config)

    monkeypatch.setattr("hermes_cli.config.save_config", fake_save)
    monkeypatch.setattr(
        transaction.subprocess,
        "run",
        lambda _command, **_kwargs: SimpleNamespace(
            returncode=0, stdout=f"{source}\n", stderr=""
        ),
    )
    hooks = transaction.build_production_hooks(config_lock_timeout=0.2)
    expected = copy.deepcopy(state["config"])
    candidate = copy.deepcopy(expected)
    candidate["model"]["default"] = "gpt-next"

    hooks.write_config(expected, candidate)
    assert state["config"]["model"]["default"] == "gpt-next"

    if os.name != "nt":
        import fcntl

        with (home / ".model-transaction.lock").open("a+") as held_lock:
            fcntl.flock(held_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with pytest.raises(TimeoutError, match="config_lock_timeout"):
                hooks.write_config(candidate, expected)
            fcntl.flock(held_lock.fileno(), fcntl.LOCK_UN)

    state["config"]["display"]["theme"] = "external"
    with pytest.raises(ConfigRevisionConflict):
        hooks.write_config(candidate, expected)
    assert state["config"]["display"]["theme"] == "external"


def test_production_restart_guard_requires_active_supervisor(monkeypatch, tmp_path):
    from hermes_cli import model_transaction as transaction

    live_preflight = preflight(tmp_path)
    source = tmp_path / "runtime-source-supervisor"
    source.mkdir()
    live_preflight = ModelPreflightResult(
        **{
            **live_preflight.__dict__,
            "gateway_pid": 987654,
            "gateway_source_root": str(source),
        }
    )
    monkeypatch.setattr(
        "hermes_cli.model_preflight.collect_model_preflight",
        lambda: live_preflight,
    )
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config",
        lambda: {"model": {"provider": "openai-codex", "default": "gpt-5.6-sol"}},
    )
    launchd_pid = {"value": 987654}

    def fake_run(command, **_kwargs):
        if "-c" in command:
            return SimpleNamespace(returncode=0, stdout=f"{source}\n", stderr="")
        return SimpleNamespace(
            returncode=0,
            stdout=f'{{\n\t"PID" = {launchd_pid["value"]};\n}}\n',
            stderr="",
        )

    plist = tmp_path / "gateway.plist"
    plist.write_text("installed", encoding="utf-8")
    monkeypatch.setattr(transaction.subprocess, "run", fake_run)
    monkeypatch.setattr(
        "hermes_cli.gateway.get_launchd_plist_path", lambda: plist
    )
    hooks = transaction.build_production_hooks()

    monkeypatch.setattr(
        "hermes_cli.service_manager.detect_service_manager", lambda: "none"
    )
    assert hooks.restart_allowed() is False

    monkeypatch.setattr(
        "hermes_cli.service_manager.detect_service_manager", lambda: "launchd"
    )
    assert hooks.restart_allowed() is True

    launchd_pid["value"] = 123456
    assert hooks.restart_allowed() is False

    systemd_pid = {"value": 987654}
    unit = tmp_path / "gateway.service"
    unit.write_text("installed", encoding="utf-8")
    monkeypatch.setattr(
        "hermes_cli.service_manager.detect_service_manager", lambda: "systemd"
    )
    monkeypatch.setattr(
        "hermes_cli.gateway._probe_systemd_service_running",
        lambda: (False, True),
    )
    monkeypatch.setattr(
        "hermes_cli.gateway.get_systemd_unit_path", lambda system=False: unit
    )
    monkeypatch.setattr(
        "hermes_cli.gateway._systemd_main_pid",
        lambda system=False: systemd_pid["value"],
    )
    assert hooks.restart_allowed() is True
    systemd_pid["value"] = 123456
    assert hooks.restart_allowed() is False

    monkeypatch.setattr(
        "hermes_cli.service_manager.detect_service_manager", lambda: "s6"
    )
    monkeypatch.setattr(
        "hermes_cli.service_manager.get_service_manager",
        lambda: SimpleNamespace(_supervised_pid=lambda _name: 987654),
    )
    assert hooks.restart_allowed() is True

    monkeypatch.setattr(
        "hermes_cli.service_manager.get_service_manager",
        lambda: SimpleNamespace(_supervised_pid=lambda _name: 123456),
    )
    assert hooks.restart_allowed() is False


def test_production_config_write_preserves_unowned_fields_with_real_save_config(
    monkeypatch, tmp_path
):
    from hermes_cli import model_transaction as transaction

    live_preflight = preflight(tmp_path)
    home = Path(live_preflight.profile_home)
    home.mkdir(parents=True)
    source = tmp_path / "runtime-source-real-save"
    source.mkdir()
    live_preflight = ModelPreflightResult(
        **{
            **live_preflight.__dict__,
            "gateway_source_root": str(source),
        }
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        """model:\n  provider: openai-codex\n  default: gpt-5.6-sol\n  api_key: ${MODEL_SECRET}\n  custom_flag: keep-me\ndisplay:\n  theme: dark\n""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "hermes_cli.model_preflight.collect_model_preflight",
        lambda: live_preflight,
    )
    monkeypatch.setattr(
        transaction.subprocess,
        "run",
        lambda _command, **_kwargs: SimpleNamespace(
            returncode=0, stdout=f"{source}\n", stderr=""
        ),
    )
    hooks = transaction.build_production_hooks(config_lock_timeout=0.2)
    original = hooks.read_config()
    candidate = copy.deepcopy(original)
    candidate["model"]["default"] = "gpt-next"

    hooks.write_config(original, candidate)
    readback = hooks.read_config()

    assert readback["model"]["provider"] == "openai-codex"
    assert readback["model"]["default"] == "gpt-next"
    assert readback["model"]["api_key"] == "${MODEL_SECRET}"
    assert readback["model"]["custom_flag"] == "keep-me"
    assert readback["display"] == {"theme": "dark"}


def test_production_smoke_uses_configured_defaults_and_active_interpreter(
    monkeypatch, tmp_path
):
    from hermes_cli import model_transaction as transaction

    live_preflight = preflight(tmp_path)
    source = tmp_path / "runtime-source"
    source.mkdir()
    live_preflight = ModelPreflightResult(
        **{
            **live_preflight.__dict__,
            "gateway_interpreter": "/active/venv/bin/python",
            "gateway_source_root": str(source),
        }
    )
    monkeypatch.setattr(
        "hermes_cli.model_preflight.collect_model_preflight",
        lambda: live_preflight,
    )
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config",
        lambda: {"model": {"provider": "openai-codex", "default": "gpt-next"}},
    )
    captured = {}

    def fake_run(command, **kwargs):
        if "-c" in command:
            return SimpleNamespace(returncode=0, stdout=f"{source}\n", stderr="")
        captured["command"] = command
        captured["cwd"] = kwargs["cwd"]
        usage_path = Path(command[command.index("--usage-file") + 1])
        usage_path.write_text(
            json.dumps(
                {
                    "provider": "openai-codex",
                    "model": "gpt-next",
                    "api_calls": 1,
                    "completed": True,
                    "failed": False,
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="OK", stderr="")

    monkeypatch.setattr(transaction.subprocess, "run", fake_run)
    hooks = transaction.build_production_hooks()

    evidence = hooks.smoke_model(
        ModelPair("openai-codex", "gpt-next"),
        "/active/venv/bin/python",
        "main-gateway-v2",
    )

    assert evidence.api_calls == 1
    assert captured["command"][0] == "/active/venv/bin/python"
    assert captured["cwd"] == str(source)
    assert "--oneshot" in captured["command"]
    assert "--model" not in captured["command"]
    assert "--provider" not in captured["command"]
