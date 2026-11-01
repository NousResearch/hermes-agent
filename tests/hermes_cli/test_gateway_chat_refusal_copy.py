"""One-shot configuration refusals retain the owner's repair guidance on both CLI outputs."""
import argparse
import json

import pytest


@pytest.mark.parametrize("output_format", ["text", "stream-json"])
@pytest.mark.parametrize("detail", [
    "No model configured; set model.default in config.yaml or pass -m <model>.",
    "Unsupported empty model; set model.default in config.yaml or pass -m <model>.",
])
def test_oneshot_model_refusal_explains_the_configuration_fix(monkeypatch, capsys, output_format, detail):
    from hermes_cli import gateway_chat
    from hermes_cli.gateway_client import rpc_error
    import hermes_cli.gateway_chat_startup as startup

    monkeypatch.setattr(gateway_chat, "_register_terminal_process", lambda: None)
    monkeypatch.setattr(startup, "ensure_launch_provider", lambda args: True)

    async def refused(args, emitter=None):
        raise rpc_error({
            "code": 4001,
            "message": "invalid_params",
            "data": {
                "reason": "invalid_params",
                "field": "model",
                "detail": detail,
            },
        })

    monkeypatch.setattr(gateway_chat, "run_gateway_chat", refused)
    args = argparse.Namespace(oneshot="Reply with OK", query=None, model=None, output_format=output_format)
    assert gateway_chat.launch_from_args(args) == 1
    out, err = capsys.readouterr()
    assert "invalid_params" in err and "model.default" in err and "-m <model>" in err
    if output_format == "stream-json":
        result = [json.loads(line) for line in out.splitlines() if line.strip()][-1]
        assert result["type"] == "result" and result["exit_code"] == 1
        assert "model.default" in result["error"] and "-m <model>" in result["error"]
    else:
        assert not out
