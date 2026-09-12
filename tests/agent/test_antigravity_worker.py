from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import threading
import time

import pytest


@pytest.fixture
def fake_agy(tmp_path: Path) -> Path:
    executable = tmp_path / "agy"
    executable.write_text(
        """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import stat
import sys
import time

argv = sys.argv[1:]
prompt = sys.stdin.read()
input_payload = json.loads(prompt.split("INPUT_JSON:\\n", 1)[1])
if "PID_FILE=" in prompt:
    target = input_payload["goal"].split("PID_FILE=", 1)[1].split()[0]
    Path(target).write_text(f"{os.getpid()}\\n{os.getcwd()}", encoding="utf-8")
if "SLEEP" in prompt:
    time.sleep(30)
if "MALFORMED" in prompt:
    print("not-json")
    raise SystemExit(0)
if "NONZERO" in prompt:
    print("model failed", file=sys.stderr)
    raise SystemExit(7)
if "OVERSIZE" in prompt:
    print(json.dumps({"status": "SUCCESS", "response": "x" * 4096}))
    raise SystemExit(0)
if "MODEL_ERROR" in prompt:
    print(json.dumps({"status": "ERROR", "response": "nope"}))
    raise SystemExit(0)
if "DENIED" in prompt:
    print(json.dumps({
        "status": "SUCCESS",
        "response": "unsafe",
        "tool_actions": [{"name": "write_file", "status": "DENIED"}],
    }))
    raise SystemExit(0)
if "PENDING" in prompt:
    print(json.dumps({
        "status": "SUCCESS",
        "response": "unsafe",
        "nested": {"toolAction": {"permissionStatus": "PENDING"}},
    }))
    raise SystemExit(0)

schema_path = None
if "--json-schema" in argv:
    schema_path = argv[argv.index("--json-schema") + 1]
response = json.dumps({"answer": "ok"}) if schema_path else "done"
envelope = {
    "conversation_id": "conversation-1",
    "status": "SUCCESS",
    "response": response,
    "usage": {"input_tokens": 3, "output_tokens": 2},
    "capture": {
        "argv": argv,
        "cwd": os.getcwd(),
        "cwd_mode": stat.S_IMODE(os.stat(os.getcwd()).st_mode),
        "stdin": prompt,
        "env_keys": sorted(os.environ),
        "home": os.environ.get("HOME"),
        "schema_path": schema_path,
        "schema": json.loads(Path(schema_path).read_text()) if schema_path else None,
    },
}
print(json.dumps(envelope))
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def make_worker(fake_agy: Path, **kwargs):
    from agent.antigravity_worker import AntigravityWorker

    return AntigravityWorker(command=str(fake_agy), **kwargs)


def test_success_is_sandboxed_and_pins_cli_contract(
    fake_agy: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("GEMINI_API_KEY", "paid-secret")
    monkeypatch.setenv("GOOGLE_API_KEY", "another-paid-secret")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "slack-secret")
    monkeypatch.setenv("HERMES_DASHBOARD_SESSION_TOKEN", "hermes-secret")
    monkeypatch.setenv("UNRELATED_SECRET", "also-secret")
    monkeypatch.setenv("AGY_CLI_HIDE_LOGO", "1")

    result = make_worker(fake_agy).run(
        goal="Summarize the supplied material", context="Only this context", output_schema=None
    )

    assert result.status == "success"
    assert result.response == "done"
    assert result.conversation_id == "conversation-1"
    assert result.usage == {"input_tokens": 3, "output_tokens": 2}
    assert result.exit_code == 0
    assert result.error_code is None
    assert result.duration_ms >= 0
    assert result.raw_envelope is not None

    capture = result.raw_envelope["capture"]
    assert capture["argv"] == [
        "--print",
        "--model", "gemini-3.8-flash-low",
        "--effort", "low",
        "--mode", "plan",
        "--sandbox",
        "--disable-slash-commands",
        "--output-format", "json",
        "--print-timeout", "120s",
    ]
    assert "Summarize the supplied material" in capture["stdin"]
    assert "Only this context" in capture["stdin"]
    assert "Summarize the supplied material" not in capture["argv"]
    assert capture["cwd_mode"] == 0o700
    assert not Path(capture["cwd"]).exists()
    assert "AGY_CLI_HIDE_LOGO" in capture["env_keys"]
    for forbidden in (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "SLACK_BOT_TOKEN",
        "HERMES_DASHBOARD_SESSION_TOKEN",
        "UNRELATED_SECRET",
        "HERMES_HOME",
    ):
        assert forbidden not in capture["env_keys"]


def test_json_schema_uses_absolute_temporary_path_and_validates_response(fake_agy: Path):
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string", "enum": ["ok"]}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    result = make_worker(fake_agy).run(goal="Return JSON", context="", output_schema=schema)

    assert result.status == "success"
    assert json.loads(result.response or "") == {"answer": "ok"}
    assert result.raw_envelope is not None
    capture = result.raw_envelope["capture"]
    schema_path = Path(capture["schema_path"])
    assert schema_path.is_absolute()
    assert capture["schema"] == schema
    assert not schema_path.exists()
    assert capture["argv"][-2:] == ["--json-schema", str(schema_path)]


def test_schema_mismatch_is_rejected(fake_agy: Path):
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
    }
    result = make_worker(fake_agy).run(goal="Return JSON", context="", output_schema=schema)
    assert result.status == "failed"
    assert result.error_code == "schema_validation_failed"
    assert result.response is None


@pytest.mark.parametrize(
    ("goal", "error_code", "exit_code"),
    [
        ("NONZERO", "nonzero_exit", 7),
        ("MALFORMED", "malformed_envelope", 0),
        ("MODEL_ERROR", "unsuccessful_status", 0),
        ("DENIED", "tool_action_blocked", 0),
        ("PENDING", "tool_action_blocked", 0),
    ],
)
def test_invalid_child_results_fail_closed(
    fake_agy: Path, goal: str, error_code: str, exit_code: int
):
    result = make_worker(fake_agy).run(goal=goal, context="", output_schema=None)
    assert result.status == "failed"
    assert result.error_code == error_code
    assert result.exit_code == exit_code
    assert result.response is None


def test_output_limit_is_enforced(fake_agy: Path):
    complete_output = (
        json.dumps({"status": "SUCCESS", "response": "x" * 4096}) + "\n"
    ).encode("utf-8")
    result = make_worker(fake_agy, max_output_bytes=512).run(
        goal="OVERSIZE", context="", output_schema=None
    )
    assert result.status == "failed"
    assert result.error_code == "output_too_large"
    assert result.response is None
    assert result.output_excerpt is not None
    assert len(result.output_excerpt.encode("utf-8")) <= 512
    assert result.output_sha256 == hashlib.sha256(complete_output).hexdigest()
    assert result.output_bytes == len(complete_output)


def test_input_limit_prevents_spawn(fake_agy: Path):
    result = make_worker(fake_agy, max_input_bytes=64).run(
        goal="x" * 100, context="", output_schema=None
    )
    assert result.status == "failed"
    assert result.error_code == "input_too_large"
    assert result.exit_code is None


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_monotonic_timeout_terminates_group_and_cleans_workspace(
    fake_agy: Path, tmp_path: Path
):
    pid_file = tmp_path / "timeout.pid"
    result = make_worker(fake_agy, timeout_seconds=2.0).run(
        goal=f"SLEEP PID_FILE={pid_file}", context="", output_schema=None
    )
    assert result.status == "failed"
    assert result.error_code == "timeout"
    assert result.duration_ms < 5000
    pid, workspace = pid_file.read_text(encoding="utf-8").splitlines()
    assert not _pid_is_alive(int(pid))
    assert not Path(workspace).exists()


def test_cancel_terminates_active_run_and_close_is_idempotent(
    fake_agy: Path, tmp_path: Path
):
    pid_file = tmp_path / "cancel.pid"
    worker = make_worker(fake_agy, timeout_seconds=10)
    holder = {}

    thread = threading.Thread(
        target=lambda: holder.setdefault(
            "result",
            worker.run(
                goal=f"SLEEP PID_FILE={pid_file}", context="", output_schema=None
            ),
        )
    )
    thread.start()
    deadline = time.monotonic() + 3
    while not pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert pid_file.exists()

    worker.cancel()
    thread.join(timeout=3)
    assert not thread.is_alive()
    result = holder["result"]
    assert result.status == "failed"
    assert result.error_code == "cancelled"
    pid, workspace = pid_file.read_text(encoding="utf-8").splitlines()
    assert not _pid_is_alive(int(pid))
    assert not Path(workspace).exists()
    worker.close()
    worker.close()


def test_result_is_frozen(fake_agy: Path):
    from dataclasses import FrozenInstanceError

    result = make_worker(fake_agy).run(goal="ok", context="", output_schema=None)
    with pytest.raises(FrozenInstanceError):
        setattr(result, "status", "changed")


def test_runtime_model_effort_timeout_and_safe_extra_args_are_configurable(fake_agy: Path):
    result = make_worker(
        fake_agy,
        model="gemini-test",
        effort="medium",
        timeout_seconds=17,
        extra_args=["--no-gui"],
    ).run(goal="Return text", context="", output_schema=None)

    assert result.status == "success"
    assert result.raw_envelope is not None
    argv = result.raw_envelope["capture"]["argv"]
    assert argv[argv.index("--model") + 1] == "gemini-test"
    assert argv[argv.index("--effort") + 1] == "medium"
    assert argv[argv.index("--print-timeout") + 1] == "17s"
    assert argv[-1] == "--no-gui"


def test_process_started_callback_runs_after_spawn(fake_agy: Path):
    observations = []
    result = make_worker(fake_agy).run(
        goal="Return text",
        context="",
        output_schema=None,
        on_process_started=lambda: observations.append("started"),
    )
    assert result.status == "success"
    assert observations == ["started"]


def test_process_started_callback_failure_terminates_child(fake_agy: Path):
    def fail_callback():
        raise RuntimeError("ledger unavailable")

    result = make_worker(fake_agy).run(
        goal="SLEEP",
        context="",
        output_schema=None,
        on_process_started=fail_callback,
    )
    assert result.status == "failed"
    assert result.error_code == "process_start_callback_failed"


def test_unsupported_json_schema_fails_closed_before_spawn(fake_agy: Path):
    called = []
    result = make_worker(fake_agy).run(
        goal="Return JSON",
        context="",
        output_schema={
            "$defs": {"answer": {"type": "string"}},
            "$ref": "#/$defs/answer",
        },
        on_process_started=lambda: called.append(True),
    )
    assert result.status == "failed"
    assert result.error_code == "invalid_schema"
    assert called == []


def test_dangerous_or_contract_overriding_extra_args_are_rejected(fake_agy: Path):
    for arg in (
        "--dangerously-skip-permissions",
        "--print",
        "--print=false",
        "--mode",
        "--sandbox=false",
        "--no-sandbox",
        "--output-format=text",
        "--disable-slash-commands=false",
        "--no-disable-slash-commands",
        "--json-schema=elsewhere.json",
        "--add-dir=/tmp/escape",
        "--continue",
        "--conversation=other",
    ):
        with pytest.raises(ValueError):
            make_worker(fake_agy, extra_args=[arg])


@pytest.mark.parametrize("extra_args", [None, False, "", 0, {}, ()])
def test_explicit_non_list_extra_args_are_rejected(fake_agy: Path, extra_args):
    with pytest.raises(ValueError, match="extra_args"):
        make_worker(fake_agy, extra_args=extra_args)
