import argparse
import json

from hermes_cli import redact_cmd


def _args(**kwargs):
    defaults = {
        "redact_command": "config",
        "enable": False,
        "disable": False,
        "hosted_only": False,
        "all_endpoints": False,
        "fail_closed": False,
        "fail_open": False,
        "heuristics_only": False,
        "no_heuristics_only": False,
        "model": None,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_register_redact_parser_wires_expected_flags():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    redact_cmd.register_redact_parser(subparsers)

    args = parser.parse_args(
        [
            "redact",
            "config",
            "--enable",
            "--all-endpoints",
            "--fail-open",
            "--heuristics-only",
            "--model",
            "local-model",
        ]
    )

    assert args.command == "redact"
    assert args.redact_command == "config"
    assert args.enable is True
    assert args.all_endpoints is True
    assert args.fail_open is True
    assert args.heuristics_only is True
    assert args.model == "local-model"
    assert args.func is redact_cmd.cmd_redact


def test_cmd_redact_config_persists_nested_settings(monkeypatch, capsys):
    saved = {}
    current = {"security": {}}

    def load_config():
        return current

    def save_config(config):
        saved.clear()
        saved.update(config)

    monkeypatch.setattr("hermes_cli.config.load_config", load_config)
    monkeypatch.setattr("hermes_cli.config.save_config", save_config)
    monkeypatch.setattr(
        redact_cmd,
        "_runtime_config",
        lambda config=None: {
            "enabled": True,
            "provider": "rampart",
            "hosted_only": False,
            "fail_closed": False,
            "rampart": {"model": "local-rampart", "heuristics_only": True},
            "setup_ready": True,
        },
    )

    code = redact_cmd.cmd_redact(
        _args(
            enable=True,
            all_endpoints=True,
            fail_open=True,
            heuristics_only=True,
            model="local-rampart",
        )
    )

    assert code == 0
    section = saved["security"]["pii_redaction"]
    assert section["enabled"] is True
    assert section["hosted_only"] is False
    assert section["fail_closed"] is False
    assert section["provider"] == "rampart"
    assert section["rampart"] == {"heuristics_only": True, "model": "local-rampart"}

    out = capsys.readouterr().out
    assert "Saved local PII redaction config." in out
    assert "Enabled:        yes" in out
    assert "Hosted only:    no" in out
    assert "Fail closed:    no" in out
    assert "Heuristics only: yes" in out


def test_cmd_redact_config_show_does_not_save(monkeypatch, capsys):
    save_calls = []
    config = {
        "security": {
            "pii_redaction": {
                "enabled": False,
                "provider": "rampart",
                "hosted_only": True,
                "fail_closed": True,
                "rampart": {"heuristics_only": False},
            }
        }
    }

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
    monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg: save_calls.append(cfg))
    monkeypatch.setattr(redact_cmd, "_runtime_config", lambda config=None: config["security"]["pii_redaction"])

    code = redact_cmd.cmd_redact(_args())

    assert code == 0
    assert save_calls == []
    assert "Enabled:        no" in capsys.readouterr().out


def test_setup_ready_requires_transformers_for_model_backed_rampart(tmp_path):
    worker = tmp_path / "rampart_pii_worker.mjs"
    worker.write_text("worker", encoding="utf-8")
    rampart_dir = tmp_path / "node_modules" / "@nationaldesignstudio" / "rampart"
    rampart_dir.mkdir(parents=True)

    assert (
        redact_cmd._setup_ready(
            {
                "provider": "rampart",
                "rampart": {
                    "command": f"node {worker}",
                    "heuristics_only": False,
                },
            }
        )
        is False
    )

    transformers_dir = tmp_path / "node_modules" / "@huggingface" / "transformers"
    transformers_dir.mkdir(parents=True)
    assert (
        redact_cmd._setup_ready(
            {
                "provider": "rampart",
                "rampart": {
                    "command": f"node {worker}",
                    "heuristics_only": False,
                },
            }
        )
        is True
    )


def test_setup_ready_heuristics_only_still_requires_rampart_package(tmp_path):
    worker = tmp_path / "rampart_pii_worker.mjs"
    worker.write_text("worker", encoding="utf-8")
    status = {
        "provider": "rampart",
        "rampart": {
            "command": f"node {worker}",
            "heuristics_only": True,
        },
    }

    assert redact_cmd._setup_ready(status) is False

    rampart_dir = tmp_path / "node_modules" / "@nationaldesignstudio" / "rampart"
    rampart_dir.mkdir(parents=True)
    assert redact_cmd._setup_ready(status) is True


def test_cmd_redact_run_uses_runtime_redactor(monkeypatch, capsys):
    calls = []

    def fake_redact(text):
        calls.append(text)
        return "Email [EMAIL]\n", {"redacted": True, "texts_changed": 1}

    monkeypatch.setattr(redact_cmd, "_redact_text", fake_redact)

    code = redact_cmd.cmd_redact(argparse.Namespace(redact_command="run", text="Email alice@example.com", json=False))

    assert code == 0
    assert calls == ["Email alice@example.com"]
    captured = capsys.readouterr()
    assert captured.out == "Email [EMAIL]\n"
    assert captured.err == ""


def test_cmd_redact_run_json_reports_skipped(monkeypatch, capsys):
    monkeypatch.setattr(
        redact_cmd,
        "_redact_text",
        lambda text: (text, {"skipped": True, "skipped_reason": "disabled"}),
    )

    code = redact_cmd.cmd_redact(argparse.Namespace(redact_command="run", text="Email alice@example.com", json=True))

    assert code == 0
    captured = capsys.readouterr()
    assert "redaction skipped (disabled)" in captured.err
    payload = json.loads(captured.out)
    assert payload == {
        "text": "Email alice@example.com",
        "stats": {"skipped": True, "skipped_reason": "disabled"},
    }


def test_cmd_redact_run_runtime_failure_returns_error(monkeypatch, capsys):
    def fail(_text):
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(redact_cmd, "_redact_text", fail)

    code = redact_cmd.cmd_redact(argparse.Namespace(redact_command="run", text="Email alice@example.com", json=False))

    assert code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "hermes redact run: backend unavailable" in captured.err
