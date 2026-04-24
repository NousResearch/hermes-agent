import importlib.util
import json
import sys
from pathlib import Path


SKILL_REFS = Path(__file__).resolve().parents[2] / "skills" / "codex-bridge" / "references"


def load_reference_module(name):
    module_path = SKILL_REFS / f"{name}.py"
    sys.path.insert(0, str(SKILL_REFS))
    try:
        spec = importlib.util.spec_from_file_location(f"codex_bridge_skill_{name}", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module
    finally:
        try:
            sys.path.remove(str(SKILL_REFS))
        except ValueError:
            pass


def test_validator_rejects_unsafe_start_inputs(tmp_path):
    validator = load_reference_module("validator")

    for sandbox in ["danger-full-access", "network-only"]:
        try:
            validator.validate_start_input("hello", str(tmp_path), sandbox, "untrusted")
        except validator.ValidationError as exc:
            assert "sandbox" in str(exc) or "danger-full-access" in str(exc)
        else:
            raise AssertionError(f"expected {sandbox} to be rejected")

    try:
        validator.validate_start_input("hello", str(tmp_path), "read-only", "never")
    except validator.ValidationError as exc:
        assert "approval_policy" in str(exc)
    else:
        raise AssertionError("expected approval_policy=never to be rejected")

    try:
        validator.validate_start_input("", str(tmp_path), "read-only", "untrusted")
    except validator.ValidationError as exc:
        assert "prompt" in str(exc)
    else:
        raise AssertionError("expected empty prompt to be rejected")

    try:
        validator.validate_start_input("hello", str(tmp_path / "missing"), "read-only", "untrusted")
    except validator.ValidationError as exc:
        assert "cwd" in str(exc)
    else:
        raise AssertionError("expected missing cwd to be rejected")


def test_validator_requires_safe_start_output_contract():
    validator = load_reference_module("validator")

    valid = {
        "success": True,
        "protocol": {"mailbox": False, "transport": "app-server stdio"},
        "task": {
            "hermes_task_id": "codex-1",
            "codex_thread_id": "thread-1",
            "codex_turn_id": "turn-1",
        },
    }
    validator.validate_start_output(valid)

    invalid = dict(valid)
    invalid["protocol"] = {"mailbox": True, "transport": "app-server stdio"}
    try:
        validator.validate_start_output(invalid)
    except validator.ValidationError as exc:
        assert "mailbox" in str(exc)
    else:
        raise AssertionError("expected mailbox output to be rejected")

    invalid = dict(valid)
    invalid["protocol"] = {"mailbox": False, "transport": "mailbox"}
    try:
        validator.validate_start_output(invalid)
    except validator.ValidationError as exc:
        assert "app-server" in str(exc)
    else:
        raise AssertionError("expected non app-server transport to be rejected")


def test_cli_start_validates_and_emits_bridge_json(tmp_path, monkeypatch, capsys):
    cli = load_reference_module("cli")
    calls = []

    def fake_codex_bridge(**kwargs):
        calls.append(kwargs)
        return json.dumps(
            {
                "success": True,
                "protocol": {"mailbox": False, "transport": "app-server stdio"},
                "task": {
                    "hermes_task_id": "codex-abc",
                    "codex_thread_id": "thread-abc",
                    "codex_turn_id": "turn-abc",
                },
            }
        )

    monkeypatch.setattr(cli, "codex_bridge", fake_codex_bridge)

    exit_code = cli.main(["start", "--cwd", str(tmp_path), "--prompt", "Analyze tests"])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["task"]["hermes_task_id"] == "codex-abc"
    assert calls == [
        {
            "action": "start",
            "prompt": "Analyze tests",
            "cwd": str(tmp_path),
            "model": None,
            "sandbox": "read-only",
            "approval_policy": "untrusted",
            "codex_home": None,
        }
    ]


def test_cli_respond_maps_request_id_to_bridge_instruction(monkeypatch, capsys):
    cli = load_reference_module("cli")
    calls = []

    def fake_codex_bridge(**kwargs):
        calls.append(kwargs)
        return json.dumps({"success": True, "response": {"decision": kwargs["decision"]}})

    monkeypatch.setattr(cli, "codex_bridge", fake_codex_bridge)

    exit_code = cli.main(
        [
            "respond",
            "codex-abc",
            "--request-id",
            "approval-1",
            "--decision",
            "decline",
            "--answers",
            '{"q1": {"answers": ["yes"]}}',
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["response"] == {"decision": "decline"}
    assert calls == [
        {
            "action": "respond",
            "task_id": "codex-abc",
            "instruction": "approval-1",
            "decision": "decline",
            "answers": {"q1": {"answers": ["yes"]}},
        }
    ]


def test_cli_smoke_test_polls_until_completed_with_sentinel(tmp_path, monkeypatch, capsys):
    cli = load_reference_module("cli")
    calls = []

    def fake_codex_bridge(**kwargs):
        calls.append(kwargs)
        action = kwargs["action"]
        if action == "start":
            return json.dumps(
                {
                    "success": True,
                    "protocol": {"mailbox": False, "transport": "app-server stdio"},
                    "task": {
                        "hermes_task_id": "codex-smoke",
                        "codex_thread_id": "thread-smoke",
                        "codex_turn_id": "turn-smoke",
                    },
                }
            )
        return json.dumps(
            {
                "success": True,
                "task": {
                    "hermes_task_id": "codex-smoke",
                    "status": "completed",
                    "recent_events": [{"payload_summary": "assistant replied CODEX_ASYNC_OK"}],
                    "final_summary": None,
                },
            }
        )

    monkeypatch.setattr(cli, "codex_bridge", fake_codex_bridge)
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)

    exit_code = cli.main(
        [
            "smoke-test",
            "--cwd",
            str(tmp_path),
            "--wait",
            "3",
            "--timeout",
            "10",
            "--poll-interval",
            "0.01",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["success"] is True
    assert output["task_id"] == "codex-smoke"
    assert [call["action"] for call in calls] == ["start", "status"]
    assert "CODEX_ASYNC_OK" in calls[0]["prompt"]
