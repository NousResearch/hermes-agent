"""RED tests for the Antigravity `agy agentapi` transport.

These encode the real CLI contract observed on the Antigravity CLI:

1. Start:     agy agentapi new-conversation --model=... --title=... <prompt>
              -> {"response":{"newConversation":{"conversationId":"..."}}}
2. Follow-up: agy agentapi send-message --title=... <conversation_id> <content>
3. The model's reply lands asynchronously in
   ~/.gemini/antigravity-cli/brain/<conversation_id>/.system_generated/logs/transcript.jsonl
   as a DONE MODEL/PLANNER_RESPONSE line with a monotonically growing step_index.

The transport must poll that transcript and return only responses newer than the
baseline captured before the message was sent.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import agent.antigravity_agentapi_client as aac
import agent.copilot_acp_client as cac
from agent.copilot_acp_client import CopilotACPClient

CONVERSATION_ID = "e7730000-0000-4000-8000-000000000001"


def _transcript_path(home: Path, conversation_id: str = CONVERSATION_ID) -> Path:
    return (
        home
        / ".gemini"
        / "antigravity-cli"
        / "brain"
        / conversation_id
        / ".system_generated"
        / "logs"
        / "transcript.jsonl"
    )


def _append_transcript(home: Path, lines: list[dict], conversation_id: str = CONVERSATION_ID) -> None:
    path = _transcript_path(home, conversation_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(json.dumps(line) + "\n")


def _planner_response(step_index: int, content: str) -> dict:
    return {
        "step_index": step_index,
        "source": "MODEL",
        "type": "PLANNER_RESPONSE",
        "status": "DONE",
        "content": content,
    }


def _noise(step_index: int) -> list[dict]:
    """Transcript lines the transport must not mistake for a final response."""

    return [
        {"step_index": step_index, "source": "USER", "type": "USER_MESSAGE", "status": "DONE", "content": "ignore me"},
        {"step_index": step_index, "source": "MODEL", "type": "PLANNER_RESPONSE", "status": "IN_PROGRESS", "content": "thinking"},
        {"step_index": step_index, "source": "MODEL", "type": "TOOL_CALL", "status": "DONE", "content": "ran a tool"},
    ]


class _FakeAgy:
    """Stands in for `subprocess.run`, emulating the real agy agentapi CLI.

    The CLI answers the start/send call immediately and the model's reply shows
    up in the transcript afterwards, so each fake invocation appends the lines
    the corresponding poll is expected to pick up.
    """

    def __init__(self, home: Path, *, start_stdout: str | None = None) -> None:
        self._home = home
        self._start_stdout = start_stdout
        self.calls: list[dict] = []
        self.on_start: list[dict] | None = None
        self.on_send: list[dict] | None = None

    def __call__(self, cmd, **kwargs):
        cmd = list(cmd)
        self.calls.append({"cmd": cmd, "kwargs": kwargs})

        if "new-conversation" in cmd:
            if self.on_start:
                _append_transcript(self._home, self.on_start)
            stdout = self._start_stdout
            if stdout is None:
                stdout = json.dumps(
                    {
                        "response": {
                            "newConversation": {
                                "prompt": cmd[-1],
                                "conversationId": CONVERSATION_ID,
                            }
                        }
                    }
                )
            return SimpleNamespace(returncode=0, stdout=stdout + "\n", stderr="")

        if "send-message" in cmd:
            if self.on_send:
                _append_transcript(self._home, self.on_send)
            stdout = json.dumps(
                {
                    "response": {
                        "sendMessage": {
                            "recipientId": CONVERSATION_ID,
                            "content": cmd[-1],
                        }
                    }
                }
            )
            return SimpleNamespace(returncode=0, stdout=stdout + "\n", stderr="")

        raise AssertionError(f"unexpected agy invocation: {cmd}")

    def cmds_for(self, subcommand: str) -> list[list[str]]:
        return [call["cmd"] for call in self.calls if subcommand in call["cmd"]]


class _RecordingAgentAPITransport:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    def send_prompt(self, prompt_text: str, *, timeout_seconds: float) -> str:
        self.prompts.append(prompt_text)
        return self._responses[len(self.prompts) - 1]


@pytest.fixture
def agy_home(tmp_path, monkeypatch):
    """A deterministic HOME plus the env the agentapi transport requires."""

    home = tmp_path / "home"
    home.mkdir()
    # HERMES_HOME without a `home/` subdir keeps _resolve_home_dir on $HOME.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-no-profile-home"))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("ANTIGRAVITY_LS_ADDRESS", "127.0.0.1:57321")
    monkeypatch.setenv("ANTIGRAVITY_PROJECT_ID", "proj-test-1234")
    # Polling must never make the suite wait on wall-clock time.
    monkeypatch.setattr(cac.time, "sleep", lambda _seconds: None)
    return home


def _make_agentapi_client(tmp_path) -> CopilotACPClient:
    return CopilotACPClient(
        api_key="copilot-acp",
        base_url="acp://copilot",
        acp_command="agy",
        acp_args=["agentapi"],
        acp_cwd=str(tmp_path),
    )


def test_agentapi_args_select_agentapi_mode_instead_of_print_mode_error(agy_home, tmp_path, monkeypatch):
    fake = _FakeAgy(agy_home)
    fake.on_start = [_planner_response(2, "AGY_OK")]
    monkeypatch.setattr(cac.subprocess, "run", fake)

    client = _make_agentapi_client(tmp_path)
    text, _reasoning = client._run_prompt("hello", timeout_seconds=5)

    assert text == "AGY_OK"
    assert fake.cmds_for("new-conversation"), "expected agentapi mode to start a conversation"


def test_first_prompt_starts_conversation_and_returns_transcript_response(agy_home, tmp_path, monkeypatch):
    fake = _FakeAgy(agy_home)
    fake.on_start = _noise(1) + [_planner_response(2, "AGY_OK")]
    monkeypatch.setattr(cac.subprocess, "run", fake)

    client = _make_agentapi_client(tmp_path)
    text, reasoning = client._run_prompt("reply exactly once", timeout_seconds=5)

    assert text == "AGY_OK"
    assert reasoning == ""

    cmd = fake.cmds_for("new-conversation")[0]
    assert cmd[0] == "agy"
    assert cmd[1] == "agentapi"
    assert cmd[2] == "new-conversation"
    assert "reply exactly once" in cmd
    assert any(arg.startswith("--title=") for arg in cmd), f"expected a --title= argument in {cmd}"
    assert fake.calls[0]["kwargs"]["cwd"] == str(tmp_path)


def test_second_prompt_reuses_conversation_and_returns_only_newer_response(agy_home, tmp_path, monkeypatch):
    fake = _FakeAgy(agy_home)
    fake.on_start = [_planner_response(2, "AGY_OK")]
    fake.on_send = _noise(4) + [_planner_response(5, "AGY_SECOND")]
    monkeypatch.setattr(cac.subprocess, "run", fake)

    client = _make_agentapi_client(tmp_path)
    first, _ = client._run_prompt("first turn", timeout_seconds=5)
    second, _ = client._run_prompt("second turn", timeout_seconds=5)

    assert first == "AGY_OK"
    # The stale step_index=2 response is still in the transcript; the transport
    # must ignore everything at or before the pre-send baseline.
    assert second == "AGY_SECOND"

    assert len(fake.cmds_for("new-conversation")) == 1, "second turn must not start a new conversation"
    send_cmd = fake.cmds_for("send-message")[0]
    assert send_cmd[0] == "agy"
    assert send_cmd[1] == "agentapi"
    assert send_cmd[2] == "send-message"
    assert CONVERSATION_ID in send_cmd
    assert "second turn" in send_cmd


def test_agentapi_chat_completion_sends_only_delta_after_first_turn(tmp_path):
    transport = _RecordingAgentAPITransport(["FIRST_RESULT", "SECOND_RESULT"])
    client = _make_agentapi_client(tmp_path)
    client._agentapi_client = transport

    first_response = client._create_chat_completion(
        model="gpt-test",
        messages=[
            {"role": "system", "content": "system rule"},
            {"role": "user", "content": "first user turn"},
        ],
        timeout=5,
    )
    second_response = client._create_chat_completion(
        model="gpt-test",
        messages=[
            {"role": "system", "content": "system rule"},
            {"role": "user", "content": "first user turn"},
            {"role": "assistant", "content": "first assistant reply"},
            {"role": "user", "content": "second user turn"},
        ],
        timeout=5,
    )

    assert first_response.choices[0].message.content == "FIRST_RESULT"
    assert second_response.choices[0].message.content == "SECOND_RESULT"
    assert len(transport.prompts) == 2
    assert "second user turn" in transport.prompts[1]
    assert "system rule" not in transport.prompts[1]
    assert "first user turn" not in transport.prompts[1]
    assert "first assistant reply" not in transport.prompts[1]


def test_concurrent_first_agentapi_chat_completion_creates_only_one_transport(tmp_path, monkeypatch):
    init_started = threading.Event()
    second_init_started = threading.Event()
    allow_first_init_finish = threading.Event()
    creation_lock = threading.Lock()
    created_instances: list[int] = []
    prompts: list[tuple[int, str]] = []
    errors: list[BaseException] = []

    class _SlowTransport:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            with creation_lock:
                instance_id = len(created_instances) + 1
                created_instances.append(instance_id)
            self.instance_id = instance_id
            if instance_id == 1:
                init_started.set()
                assert allow_first_init_finish.wait(timeout=1), "first transport init never released"
            else:
                second_init_started.set()

        def send_prompt(self, prompt_text: str, *, timeout_seconds: float) -> str:
            del timeout_seconds
            prompts.append((self.instance_id, prompt_text))
            return f"RESULT_{self.instance_id}"

        def reset_conversation(self) -> None:
            pass

    monkeypatch.setattr(cac, "AntigravityAgentAPIClient", _SlowTransport)

    client = _make_agentapi_client(tmp_path)
    responses: list[str] = []

    def _worker() -> None:
        try:
            result = client._create_chat_completion(
                model="gpt-test",
                messages=[{"role": "user", "content": "same first turn"}],
                timeout=5,
            )
            responses.append(result.choices[0].message.content)
        except BaseException as exc:  # pragma: no cover - failure capture for assertions below
            errors.append(exc)

    first_thread = threading.Thread(target=_worker)
    second_thread = threading.Thread(target=_worker)

    first_thread.start()
    assert init_started.wait(timeout=1), "first transport initialization never started"
    second_thread.start()
    assert not second_init_started.wait(
        timeout=0.05
    ), "second concurrent first turn constructed a separate transport"
    allow_first_init_finish.set()
    first_thread.join(timeout=1)
    second_thread.join(timeout=1)

    assert not errors
    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert created_instances == [1]
    assert responses == ["RESULT_1", "RESULT_1"]
    assert len(prompts) == 2
    assert all(instance_id == 1 for instance_id, _prompt in prompts)


def test_agentapi_subprocess_receives_home_and_antigravity_env(agy_home, tmp_path, monkeypatch):
    fake = _FakeAgy(agy_home)
    fake.on_start = [_planner_response(2, "AGY_OK")]
    monkeypatch.setattr(cac.subprocess, "run", fake)

    client = _make_agentapi_client(tmp_path)
    client._run_prompt("hello", timeout_seconds=5)

    env = fake.calls[0]["kwargs"]["env"]
    assert env["HOME"] == str(agy_home)
    assert env["ANTIGRAVITY_LS_ADDRESS"] == "127.0.0.1:57321"
    assert env["ANTIGRAVITY_PROJECT_ID"] == "proj-test-1234"


def test_malformed_start_json_raises_runtime_error(agy_home, tmp_path, monkeypatch):
    fake = _FakeAgy(agy_home, start_stdout="Loading model... not-json {{{")
    monkeypatch.setattr(cac.subprocess, "run", fake)

    client = _make_agentapi_client(tmp_path)
    with pytest.raises(RuntimeError, match="(?i)(json|conversation)"):
        client._run_prompt("hello", timeout_seconds=5)


def test_timeout_waiting_for_transcript_response_raises_timeout_error(agy_home, tmp_path, monkeypatch):
    # Conversation starts fine, but the model never writes a DONE response.
    fake = _FakeAgy(agy_home)
    fake.on_start = _noise(1)
    monkeypatch.setattr(cac.subprocess, "run", fake)

    client = _make_agentapi_client(tmp_path)
    with pytest.raises(TimeoutError):
        client._run_prompt("hello", timeout_seconds=0.05)


def test_timeout_during_first_turn_forces_next_turn_to_start_fresh_conversation(
    agy_home, tmp_path, monkeypatch
):
    stale_conversation_id = "e7730000-0000-4000-8000-0000000000aa"
    fresh_conversation_id = "e7730000-0000-4000-8000-0000000000bb"

    class _TimeoutThenFreshAgy:
        def __init__(self, home: Path) -> None:
            self._home = home
            self.calls: list[dict] = []
            self._new_conversation_calls = 0

        def __call__(self, cmd, **kwargs):
            cmd = list(cmd)
            self.calls.append({"cmd": cmd, "kwargs": kwargs})

            if "new-conversation" in cmd:
                self._new_conversation_calls += 1
                if self._new_conversation_calls == 1:
                    conversation_id = stale_conversation_id
                    _append_transcript(self._home, _noise(1), conversation_id=conversation_id)
                else:
                    conversation_id = fresh_conversation_id
                    _append_transcript(
                        self._home,
                        [_planner_response(2, "FRESH_RESPONSE")],
                        conversation_id=conversation_id,
                    )
                stdout = json.dumps(
                    {
                        "response": {
                            "newConversation": {
                                "conversationId": conversation_id,
                            }
                        }
                    }
                )
                return SimpleNamespace(returncode=0, stdout=stdout + "\n", stderr="")

            if "send-message" in cmd:
                _append_transcript(
                    self._home,
                    [_planner_response(3, "STALE_RESPONSE")],
                    conversation_id=stale_conversation_id,
                )
                stdout = json.dumps(
                    {
                        "response": {
                            "sendMessage": {
                                "recipientId": stale_conversation_id,
                                "content": cmd[-1],
                            }
                        }
                    }
                )
                return SimpleNamespace(returncode=0, stdout=stdout + "\n", stderr="")

            raise AssertionError(f"unexpected agy invocation: {cmd}")

        def cmds_for(self, subcommand: str) -> list[list[str]]:
            return [call["cmd"] for call in self.calls if subcommand in call["cmd"]]

    fake = _TimeoutThenFreshAgy(agy_home)
    monkeypatch.setattr(cac.subprocess, "run", fake)

    client = _make_agentapi_client(tmp_path)
    with pytest.raises(TimeoutError):
        client._run_prompt("first turn", timeout_seconds=0.05)

    second, _ = client._run_prompt("second turn", timeout_seconds=5)

    assert second == "FRESH_RESPONSE"
    assert len(fake.cmds_for("new-conversation")) == 2
    assert not fake.cmds_for("send-message")


def test_concurrent_agentapi_turns_do_not_overlap_critical_section(
    agy_home, tmp_path, monkeypatch
):
    monkeypatch.setattr(aac.time, "sleep", lambda _seconds: None)

    env = {
        "HOME": str(agy_home),
        "ANTIGRAVITY_LS_ADDRESS": "127.0.0.1:57321",
        "ANTIGRAVITY_PROJECT_ID": "proj-test-1234",
    }
    client = aac.AntigravityAgentAPIClient(
        command="agy",
        args=["agentapi"],
        cwd=str(tmp_path),
        env_factory=lambda: dict(env),
    )
    client._conversation_id = CONVERSATION_ID

    call_lock = threading.Lock()
    call_count = {"send": 0}
    first_started = threading.Event()
    second_started = threading.Event()
    allow_first_finish = threading.Event()
    first_finished = threading.Event()
    overlap_observed = threading.Event()
    errors: list[BaseException] = []

    def _fake_run(cmd, **kwargs):
        cmd = list(cmd)
        if "send-message" not in cmd:
            raise AssertionError(f"unexpected command: {cmd}")

        with call_lock:
            call_count["send"] += 1
            send_index = call_count["send"]

        if send_index == 1:
            first_started.set()
            allow_first_finish.wait(timeout=1)
            _append_transcript(agy_home, [_planner_response(2, "FIRST")])
            first_finished.set()
        elif send_index == 2:
            second_started.set()
            if not first_finished.is_set():
                overlap_observed.set()
            _append_transcript(agy_home, [_planner_response(4, "SECOND")])
        else:
            raise AssertionError(f"unexpected send count {send_index}")

        stdout = json.dumps(
            {
                "response": {
                    "sendMessage": {
                        "recipientId": CONVERSATION_ID,
                        "content": cmd[-1],
                    }
                }
            }
        )
        return SimpleNamespace(returncode=0, stdout=stdout + "\n", stderr="")

    monkeypatch.setattr(aac.subprocess, "run", _fake_run)

    def _worker(prompt: str) -> None:
        try:
            client.send_prompt(prompt, timeout_seconds=1)
        except BaseException as exc:  # pragma: no cover - failure capture for the assertion below
            errors.append(exc)

    first_thread = threading.Thread(target=_worker, args=("first prompt",))
    second_thread = threading.Thread(target=_worker, args=("second prompt",))

    first_thread.start()
    assert first_started.wait(timeout=1), "first send-message never started"
    second_thread.start()
    assert not second_started.wait(
        timeout=0.05
    ), "second send-message entered subprocess before first turn finished"
    allow_first_finish.set()
    assert second_started.wait(timeout=1), "second send-message never started"
    first_thread.join(timeout=1)
    second_thread.join(timeout=1)

    assert not errors
    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert not overlap_observed.is_set()


def test_torn_utf8_tail_does_not_break_completed_response(tmp_path):
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_bytes(
        (json.dumps(_planner_response(2, "AGY_OK")) + "\n").encode("utf-8")
        + b'{"step_index": 3, "content": "\xe2\x82'
    )

    assert aac._final_response_after(transcript, -1) == "AGY_OK"


def test_polling_never_sleeps_longer_than_remaining_deadline(agy_home, tmp_path, monkeypatch):
    client = aac.AntigravityAgentAPIClient(
        command="agy",
        args=["agentapi"],
        cwd=str(tmp_path),
        env_factory=lambda: {},
    )

    now = [100.0]
    sleep_calls: list[float] = []

    def _fake_monotonic() -> float:
        return now[0]

    def _fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(aac.time, "monotonic", _fake_monotonic)
    monkeypatch.setattr(aac.time, "sleep", _fake_sleep)

    with pytest.raises(TimeoutError):
        client._wait_for_response(
            CONVERSATION_ID,
            after_step_index=-1,
            env={"HOME": str(agy_home)},
            deadline=100.1,
        )

    assert sleep_calls
    assert max(sleep_calls) <= 0.1


def test_agy_without_agentapi_arg_still_uses_print_mode(agy_home, tmp_path, monkeypatch):
    captured: dict = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="AGY_OK\n", stderr="")

    monkeypatch.setattr(cac.subprocess, "run", _fake_run)

    client = CopilotACPClient(
        api_key="copilot-acp",
        base_url="acp://copilot",
        acp_command="agy",
        acp_args=["--acp", "--stdio"],
        acp_cwd=str(tmp_path),
    )
    text, reasoning = client._run_prompt("reply exactly once", timeout_seconds=12)

    assert text == "AGY_OK"
    assert reasoning == ""
    assert captured["cmd"] == ["agy", "-p", "reply exactly once"]
    assert captured["kwargs"]["env"]["HOME"] == str(agy_home)
