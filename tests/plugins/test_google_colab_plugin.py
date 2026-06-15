from __future__ import annotations

import importlib
from pathlib import Path

from plugins.google_colab import core, register


class _FakeContext:
    def __init__(self) -> None:
        self.tools = {}
        self.commands = {}
        self.cli_commands = {}

    def register_tool(self, name, **kwargs):
        self.tools[name] = kwargs

    def register_command(self, name, **kwargs):
        self.commands[name] = kwargs

    def register_cli_command(self, name, **kwargs):
        self.cli_commands[name] = kwargs


def test_registers_tools_slash_and_cli_command() -> None:
    ctx = _FakeContext()
    register(ctx)

    assert "google_colab_status" in ctx.tools
    assert "google_colab_sessions" in ctx.tools
    assert "google_colab_run" in ctx.tools
    assert "google_colab_sft_template" in ctx.tools
    assert "colab" in ctx.commands
    assert "google-colab" in ctx.cli_commands


def test_status_reports_unavailable_without_colab(monkeypatch) -> None:
    monkeypatch.setattr(core, "_is_windows", lambda: False)
    monkeypatch.setattr(core, "_colab_exe", lambda: None)

    payload = core.status_payload({})

    assert payload["ok"] is False
    assert payload["backend"] == "unavailable"
    assert "colab executable" in payload["notes"][0]


def test_run_job_requires_confirmation(tmp_path: Path) -> None:
    script = tmp_path / "job.py"
    script.write_text("print('hello')\n", encoding="utf-8")

    result = core.run_job({"script_path": str(script), "gpu": "T4"})

    assert result["ok"] is False
    assert result["confirmation_required"] is True


def test_run_job_builds_colab_command(monkeypatch, tmp_path: Path) -> None:
    script = tmp_path / "job.py"
    script.write_text("print('hello')\n", encoding="utf-8")
    calls = []

    monkeypatch.setattr(core, "_is_windows", lambda: False)
    monkeypatch.setattr(core, "_colab_exe", lambda: "colab")

    def fake_run(command, *, timeout_seconds, cwd=None):
        calls.append((command, timeout_seconds, cwd))
        return {"ok": True, "exit_code": 0, "command": command, "stdout": "done", "stderr": ""}

    monkeypatch.setattr(core, "_run_command", fake_run)

    result = core.run_job(
        {
            "script_path": str(script),
            "args": ["--epochs", "1"],
            "gpu": "T4",
            "session_name": "hermes-sft",
            "auth": "adc",
            "confirmed": True,
            "timeout_seconds": 123,
        }
    )

    assert result["ok"] is True
    command = calls[0][0]
    assert command[:3] == ["colab", "--auth=adc", "run"]
    assert "--gpu" in command
    assert "T4" in command
    assert "-s" in command
    assert str(script) in command
    assert command[-2:] == ["--epochs", "1"]
    assert calls[0][1] == 123


def test_write_sft_template(tmp_path: Path) -> None:
    output = tmp_path / "sft.py"

    result = core.write_sft_template(
        {
            "output_path": str(output),
            "model_id": "Qwen/Qwen3-0.6B",
            "dataset_name": "trl-lib/Capybara",
            "max_steps": 5,
        }
    )

    assert result["ok"] is True
    source = output.read_text(encoding="utf-8")
    assert "SFTTrainer" in source
    assert "prepare_model_for_kbit_training" in source
    assert "HF_TOKEN" in source


def test_module_importable() -> None:
    assert importlib.import_module("plugins.google_colab.core") is core
