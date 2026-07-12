from __future__ import annotations

import argparse
import asyncio
import io
import json

import pytest

from plugins.platforms.a2a import cli


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    cli.register_cli(parser)
    return parser


def _task(*, task_id="task-1", context_id="context-1", text="answer"):
    from a2a.types.a2a_pb2 import TASK_STATE_COMPLETED, Artifact, Part, Task, TaskStatus

    return Task(
        id=task_id,
        context_id=context_id,
        status=TaskStatus(state=TASK_STATE_COMPLETED),
        artifacts=[Artifact(artifact_id="artifact-1", parts=[Part(text=text)])],
    )


class FakeClient:
    instances = []
    failure = None

    def __init__(self):
        self.closed = False
        self.calls = []
        self.__class__.instances.append(self)

    async def aclose(self):
        self.closed = True

    async def fetch_card(self, peer):
        self.calls.append(("card", peer))
        if self.failure:
            raise self.failure
        from plugins.platforms.a2a.server import build_agent_card

        return build_agent_card("https://peer.example/a2a")

    async def ask(self, peer, message, *, new_context=False, context_id=None):
        self.calls.append(("ask", peer, message, new_context, context_id))
        if self.failure:
            raise self.failure
        return _task(), ["answer"]

    async def get_task(self, peer, task_id):
        self.calls.append(("get", peer, task_id))
        return _task(task_id=task_id)

    async def list_tasks(self, peer):
        self.calls.append(("list", peer))
        from a2a.types.a2a_pb2 import ListTasksResponse

        return ListTasksResponse(tasks=[_task()])

    async def cancel(self, peer, task_id):
        self.calls.append(("cancel", peer, task_id))
        return _task(task_id=task_id)


@pytest.fixture(autouse=True)
def fake_client(monkeypatch):
    FakeClient.instances = []
    FakeClient.failure = None
    monkeypatch.setattr(cli, "_load_client_class", lambda: FakeClient)


def test_ask_stdin_json_is_clean_camel_case_and_closes(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("line one\nline two\n"))
    args = _parser().parse_args(["ask", "norbert", "--stdin", "--new-context", "--json"])

    assert cli.dispatch(args) == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload["id"] == "task-1"
    assert payload["contextId"] == "context-1"
    assert FakeClient.instances[0].calls == [
        ("ask", "norbert", "line one\nline two\n", True, None)
    ]
    assert FakeClient.instances[0].closed is True


def test_ask_human_output_includes_ids_state_and_artifact(capsys):
    args = _parser().parse_args(["ask", "norbert", "hello", "--context-id", "context-old"])
    assert cli.dispatch(args) == 0
    output = capsys.readouterr().out
    assert "task id: task-1" in output
    assert "context id: context-1" in output
    assert "TASK_STATE_COMPLETED" in output
    assert output.rstrip().endswith("answer")


@pytest.mark.parametrize(
    ("argv", "method"),
    [
        (["card", "peer", "--json"], "card"),
        (["get", "peer", "t-1", "--json"], "get"),
        (["list", "peer", "--json"], "list"),
        (["cancel", "peer", "t-1", "--json"], "cancel"),
    ],
)
def test_outbound_commands_use_named_peer_and_emit_json(argv, method, capsys):
    assert cli.dispatch(_parser().parse_args(argv)) == 0
    json.loads(capsys.readouterr().out)
    assert FakeClient.instances[0].calls[0][0] == method
    assert FakeClient.instances[0].calls[0][1] == "peer"
    assert FakeClient.instances[0].closed is True


def test_url_is_rejected_before_client_request(capsys):
    args = _parser().parse_args(["card", "https://attacker.example/a2a"])
    assert cli.dispatch(args) == 2
    assert "peer must use" in capsys.readouterr().err
    assert FakeClient.instances == []


def test_ask_rejects_positional_plus_stdin_and_never_implicitly_reads(capsys):
    both = _parser().parse_args(["ask", "peer", "hello", "--stdin"])
    assert cli.dispatch(both) == 2
    assert "either MESSAGE or --stdin" in capsys.readouterr().err

    missing = _parser().parse_args(["ask", "peer"])
    assert cli.dispatch(missing) == 2
    assert "MESSAGE is required" in capsys.readouterr().err


def test_backend_error_is_sanitized_and_client_is_closed(capsys):
    FakeClient.failure = RuntimeError("secret-token https://private.example")
    assert cli.dispatch(_parser().parse_args(["card", "peer"])) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "hermes a2a: peer request failed\n"
    assert FakeClient.instances[0].closed is True


@pytest.mark.asyncio
async def test_cancellation_still_awaits_client_close():
    FakeClient.failure = asyncio.CancelledError()
    args = _parser().parse_args(["card", "peer"])
    with pytest.raises(asyncio.CancelledError):
        await cli._run_outbound(args)
    assert FakeClient.instances[0].closed is True
