from __future__ import annotations

from types import SimpleNamespace

import auto_fix_deps
import ws_monitor


def test_redact_cmd_masks_model_download_tokens():
    cmd = [
        "comfy", "model", "download",
        "--set-hf-api-token", "hf_SECRET_TOKEN",
        "--set-civitai-api-token", "civitai_SECRET_TOKEN",
        "--filename", "model.safetensors",
    ]

    redacted = auto_fix_deps.redact_cmd(cmd)

    assert "hf_SECRET_TOKEN" not in redacted
    assert "civitai_SECRET_TOKEN" not in redacted
    assert redacted.count(auto_fix_deps.REDACTED_SECRET) == 2
    assert redacted[-2:] == ["--filename", "model.safetensors"]


def test_run_cmd_logs_redacted_tokens_but_executes_original(monkeypatch, capsys):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(auto_fix_deps.subprocess, "run", fake_run)

    cmd = [
        "comfy", "model", "download",
        "--set-hf-api-token", "hf_SECRET_TOKEN",
        "--set-civitai-api-token", "civitai_SECRET_TOKEN",
    ]
    code, _ = auto_fix_deps.run_cmd(cmd)

    assert code == 0
    assert captured["cmd"] == cmd
    stderr = capsys.readouterr().err
    assert "hf_SECRET_TOKEN" not in stderr
    assert "civitai_SECRET_TOKEN" not in stderr
    assert auto_fix_deps.REDACTED_SECRET in stderr


def test_redact_url_secrets_masks_token_query_values():
    url = "wss://cloud.comfy.org/ws?clientId=abc&token=sk_SECRET&other=value"

    redacted = ws_monitor.redact_url_secrets(url)

    assert "sk_SECRET" not in redacted
    assert "token=[REDACTED_SECRET]" in redacted
    assert "clientId=abc" in redacted
    assert "other=value" in redacted
