from types import SimpleNamespace
import subprocess

import cli


def test_split_local_hermes_command_allows_offline_config_and_gateway_stop():
    assert cli._split_local_hermes_command(
        "hermes config set model.provider deepseek"
    ) == [
        "config",
        "set",
        "model.provider",
        "deepseek",
    ]
    assert cli._split_local_hermes_command("hermes gateway stop") == [
        "gateway",
        "stop",
    ]


def test_split_local_hermes_command_rejects_general_shell_commands():
    assert cli._split_local_hermes_command("echo hermes gateway stop") is None
    assert cli._split_local_hermes_command("hermes gateway run") is None
    assert cli._split_local_hermes_command("hermes update") is None


def test_run_local_hermes_command_uses_python_module_without_shell(monkeypatch):
    calls = []

    def fake_run(argv, *, check):
        calls.append((argv, check))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    printed = []
    monkeypatch.setattr(cli, "_cprint", lambda msg: printed.append(msg))

    assert cli._run_local_hermes_command_from_prompt("hermes gateway stop")

    assert calls == [
        ([cli.sys.executable, "-m", "hermes_cli.main", "gateway", "stop"], False)
    ]
    assert any("hermes gateway stop" in msg for msg in printed)


def test_run_local_hermes_command_ignores_chat_text(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected run")),
    )

    assert not cli._run_local_hermes_command_from_prompt("please stop the gateway")
