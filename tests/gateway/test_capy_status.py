from hermes_cli.commands import resolve_command
from gateway.capy_status import _short_command, render_capabilities
import sys


def test_capy_dashboard_command_registered():
    cmd = resolve_command("capy")
    assert cmd is not None
    assert cmd.name == "capy"
    assert cmd.gateway_only is True
    assert resolve_command("dashboard") is cmd
    assert resolve_command("health") is cmd


def test_capy_dashboard_redacts_command_output(monkeypatch):
    key = "TINKER" + "_API_KEY"
    monkeypatch.setenv(key, "super-secret-value")
    code, output = _short_command(
        [sys.executable, "-c", "import os; print(os.environ['TINKER' + '_API_KEY'])"]
    )
    assert code == 0
    assert "super-secret-value" not in output
    assert "[REDACTED]" in output


def test_capy_dashboard_redaction_safe_render():
    rendered = render_capabilities(
        {
            "hermes": {"version": "Hermes Agent vX", "provider": "openai-codex", "model": "gpt-5.5"},
            "session": {"session_id": "sess", "workspace": "/tmp/work"},
            "services": {"webui_local": True, "lmstudio": False, "comfyui": True},
            "rl": {
                "ready": False,
                "tinker_api_key": False,
                "wandb_api_key": False,
                "readiness_script": "/tmp/rl.py",
            },
            "local_stack": {
                "autonovel": "/tmp/autonovel",
                "comfyui": "/tmp/comfy",
                "piper_voice": "/tmp/piper.onnx",
                "lmstudio_url": "http://127.0.0.1:1234/v1",
            },
            "runbooks": {"visual_qa": "/tmp/visual.md", "provider_fallbacks": "/tmp/fallback.md"},
        }
    )
    assert "Capy capability dashboard" in rendered
    assert "gpt-5.5" in rendered
    assert "TINKER_API_KEY: missing" in rendered
    assert "[REDACTED]" not in rendered
