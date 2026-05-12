import json
from io import StringIO
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

from agent.codex_app_server_client import CodexAppServerSubagent, codex_app_server_command_from_env


class FakeProc:
    def __init__(self, lines):
        self.stdin = StringIO()
        self.stdout = StringIO("\n".join(json.dumps(line) for line in lines) + "\n")
        self.stderr = StringIO("")
        self._code = None
        self.terminated = False

    def poll(self):
        return self._code

    def terminate(self):
        self.terminated = True
        self._code = 0

    def wait(self, timeout=None):
        self._code = 0
        return 0

    def kill(self):
        self._code = -9


def test_command_defaults_to_codex_app_server(monkeypatch):
    monkeypatch.delenv("HERMES_CODEX_APP_SERVER_COMMAND", raising=False)
    monkeypatch.delenv("CODEX_APP_SERVER_COMMAND", raising=False)
    monkeypatch.delenv("HERMES_CODEX_APP_SERVER_ARGS", raising=False)
    monkeypatch.delenv("CODEX_APP_SERVER_ARGS", raising=False)
    command, args = codex_app_server_command_from_env()
    assert command == "codex"
    assert args == ["app-server", "--listen", "stdio://"]


def test_codex_app_server_subagent_runs_thread_turn_and_extracts_final_message(tmp_path):
    lines = [
        {"id": 1, "result": {"protocolVersion": "0.1.0"}},
        {"id": 2, "result": {"thread": {"id": "thr_1"}, "model": "gpt-5.3-codex-spark"}},
        {"id": 3, "result": {"turn": {"id": "turn_1"}}},
        {"method": "turn/started", "params": {"threadId": "thr_1", "turn": {"id": "turn_1"}}},
        {"method": "item/agentMessage/delta", "params": {"delta": "partial"}},
        {"method": "item/completed", "params": {"item": {"type": "agentMessage", "text": "final answer"}}},
        {"method": "turn/completed", "params": {"threadId": "thr_1", "turn": {"id": "turn_1"}}},
    ]
    proc = FakeProc(lines)

    with patch("subprocess.Popen", return_value=proc) as popen:
        child = CodexAppServerSubagent(
            model="gpt-5.3-codex-spark",
            cwd=str(tmp_path),
            context="parent context",
            role="leaf",
            toolsets=["terminal"],
            command="codex",
            args=["app-server", "--listen", "stdio://"],
            timeout_seconds=3,
            reasoning_effort="high",
        )
        result = child.run_conversation(user_message="do the thing", task_id="sa-test")

    assert result["completed"] is True
    assert result["final_response"] == "final answer"
    assert result["api_calls"] == 1
    popen.assert_called_once()
    argv = popen.call_args.args[0]
    assert argv == ["codex", "app-server", "--listen", "stdio://"]
    written = [json.loads(line) for line in proc.stdin.getvalue().splitlines()]
    assert written[0]["method"] == "initialize"
    assert written[1]["method"] == "initialized"
    assert written[2]["method"] == "thread/start"
    assert written[2]["params"]["model"] == "gpt-5.3-codex-spark"
    assert written[2]["params"]["config"]["model_reasoning_effort"] == "high"
    assert written[2]["params"]["cwd"] == str(Path(tmp_path).resolve())
    assert written[3]["method"] == "turn/start"
    assert written[3]["params"]["threadId"] == "thr_1"
    assert written[3]["params"]["effort"] == "high"
    assert written[3]["params"]["input"][0]["type"] == "text"
    assert "parent context" in written[3]["params"]["input"][0]["text"]
    assert "do the thing" in written[3]["params"]["input"][0]["text"]


def test_codex_app_server_declines_approval_requests(tmp_path):
    lines = [
        {"id": 1, "result": {}},
        {"id": 2, "result": {"thread": {"id": "thr_1"}}},
        {"id": 3, "result": {"turn": {"id": "turn_1"}}},
        {"id": 99, "method": "item/commandExecution/requestApproval", "params": {"itemId": "x"}},
        {"method": "item/completed", "params": {"item": {"type": "agentMessage", "text": "done"}}},
        {"method": "turn/completed", "params": {"threadId": "thr_1", "turn": {"id": "turn_1"}}},
    ]
    proc = FakeProc(lines)
    with patch("subprocess.Popen", return_value=proc):
        child = CodexAppServerSubagent(
            model="gpt-5.3-codex-spark",
            cwd=str(tmp_path),
            context=None,
            role="leaf",
            toolsets=None,
            timeout_seconds=3,
        )
        result = child.run_conversation(user_message="run safe task")
    assert result["final_response"] == "done"
    written = [json.loads(line) for line in proc.stdin.getvalue().splitlines()]
    approval_response = [m for m in written if m.get("id") == 99][0]
    assert approval_response == {"id": 99, "result": {"decision": "decline"}}


def test_codex_app_server_declines_permissions_request_without_extra_access(tmp_path):
    lines = [
        {"id": 1, "result": {}},
        {"id": 2, "result": {"thread": {"id": "thr_1"}}},
        {"id": 3, "result": {"turn": {"id": "turn_1"}}},
        {"id": 99, "method": "item/permissions/requestApproval", "params": {"itemId": "x"}},
        {"method": "item/completed", "params": {"item": {"type": "agentMessage", "text": "done"}}},
        {"method": "turn/completed", "params": {"threadId": "thr_1", "turn": {"id": "turn_1"}}},
    ]
    proc = FakeProc(lines)
    with patch("subprocess.Popen", return_value=proc):
        child = CodexAppServerSubagent(
            model="gpt-5.3-codex-spark",
            cwd=str(tmp_path),
            context=None,
            role="leaf",
            toolsets=None,
            timeout_seconds=3,
        )
        result = child.run_conversation(user_message="run safe task")
    assert result["final_response"] == "done"
    written = [json.loads(line) for line in proc.stdin.getvalue().splitlines()]
    permissions_response = [m for m in written if m.get("id") == 99][0]
    assert permissions_response == {
        "id": 99,
        "result": {"permissions": {"fileSystem": None, "network": None}, "scope": "turn"},
    }


def test_codex_app_server_answers_user_input_request_with_empty_answer_map(tmp_path):
    lines = [
        {"id": 1, "result": {}},
        {"id": 2, "result": {"thread": {"id": "thr_1"}}},
        {"id": 3, "result": {"turn": {"id": "turn_1"}}},
        {"id": 99, "method": "item/tool/requestUserInput", "params": {"questions": []}},
        {"method": "item/completed", "params": {"item": {"type": "agentMessage", "text": "done"}}},
        {"method": "turn/completed", "params": {"threadId": "thr_1", "turn": {"id": "turn_1"}}},
    ]
    proc = FakeProc(lines)
    with patch("subprocess.Popen", return_value=proc):
        child = CodexAppServerSubagent(
            model="gpt-5.3-codex-spark",
            cwd=str(tmp_path),
            context=None,
            role="leaf",
            toolsets=None,
            timeout_seconds=3,
        )
        result = child.run_conversation(user_message="run safe task")
    assert result["final_response"] == "done"
    written = [json.loads(line) for line in proc.stdin.getvalue().splitlines()]
    user_input_response = [m for m in written if m.get("id") == 99][0]
    assert user_input_response == {"id": 99, "result": {"answers": {}}}


def test_codex_app_server_uses_hermes_codex_auth_as_temporary_codex_home(tmp_path, monkeypatch):
    proc = FakeProc([
        {"id": 1, "result": {}},
        {"id": 2, "result": {"thread": {"id": "thr_1"}}},
        {"id": 3, "result": {"turn": {"id": "turn_1"}}},
        {"method": "item/completed", "params": {"item": {"type": "agentMessage", "text": "done"}}},
        {"method": "turn/completed", "params": {"threadId": "thr_1", "turn": {"id": "turn_1"}}},
    ])
    fake_auth = ModuleType("hermes_cli.auth")
    tokens = {"access_token": "access", "refresh_token": "refresh", "account_id": "acct"}
    saved = {}
    fake_auth.resolve_codex_runtime_credentials = MagicMock(return_value={"api_key": "access"})
    fake_auth._read_codex_tokens = MagicMock(return_value={"tokens": tokens, "last_refresh": "now"})
    fake_auth._save_codex_tokens = MagicMock(side_effect=lambda t, last_refresh=None: saved.update({"tokens": t, "last_refresh": last_refresh}))
    monkeypatch.setitem(__import__("sys").modules, "hermes_cli.auth", fake_auth)
    monkeypatch.delenv("CODEX_HOME", raising=False)

    with patch("subprocess.Popen", return_value=proc) as popen:
        child = CodexAppServerSubagent(
            model="gpt-5.5",
            cwd=str(tmp_path),
            context=None,
            role="leaf",
            toolsets=None,
            timeout_seconds=3,
        )
        result = child.run_conversation(user_message="use stock auth")

    assert result["final_response"] == "done"
    env = popen.call_args.kwargs["env"]
    assert env["CODEX_HOME"]
    fake_auth.resolve_codex_runtime_credentials.assert_called_once()
    fake_auth._save_codex_tokens.assert_called_once()
    assert saved["tokens"] == tokens
    assert saved["last_refresh"] == "now"
