import io
import json
from types import SimpleNamespace

from agent.copilot_acp_client import CopilotACPClient, _ensure_path_within_cwd


def _fake_process():
    return SimpleNamespace(stdin=io.StringIO())


def test_ensure_path_within_cwd_accepts_relative_paths(tmp_path):
    nested = tmp_path / "src" / "main.py"
    nested.parent.mkdir(parents=True)
    nested.write_text("print('ok')", encoding="utf-8")

    resolved = _ensure_path_within_cwd("src/main.py", str(tmp_path))

    assert resolved == nested.resolve()


def test_handle_server_message_reads_relative_file_path(tmp_path):
    target = tmp_path / "notes.txt"
    target.write_text("copilot context", encoding="utf-8")
    client = CopilotACPClient(acp_cwd=str(tmp_path))
    process = _fake_process()

    handled = client._handle_server_message(
        {
            "id": 1,
            "method": "fs/read_text_file",
            "params": {"path": "notes.txt"},
        },
        process=process,
        cwd=str(tmp_path),
        text_parts=None,
        reasoning_parts=None,
    )

    assert handled is True
    response = json.loads(process.stdin.getvalue())
    assert response["result"]["content"] == "copilot context"


def test_handle_server_message_writes_relative_file_path(tmp_path):
    client = CopilotACPClient(acp_cwd=str(tmp_path))
    process = _fake_process()

    handled = client._handle_server_message(
        {
            "id": 2,
            "method": "fs/write_text_file",
            "params": {"path": "out/result.txt", "content": "done"},
        },
        process=process,
        cwd=str(tmp_path),
        text_parts=None,
        reasoning_parts=None,
    )

    assert handled is True
    assert (tmp_path / "out" / "result.txt").read_text(encoding="utf-8") == "done"


def test_handle_server_message_rejects_relative_traversal(tmp_path):
    client = CopilotACPClient(acp_cwd=str(tmp_path))
    process = _fake_process()

    client._handle_server_message(
        {
            "id": 3,
            "method": "fs/write_text_file",
            "params": {"path": "../escape.txt", "content": "nope"},
        },
        process=process,
        cwd=str(tmp_path),
        text_parts=None,
        reasoning_parts=None,
    )

    response = json.loads(process.stdin.getvalue())
    assert response["error"]["code"] == -32602
    assert "outside the session cwd" in response["error"]["message"]
    assert not (tmp_path.parent / "escape.txt").exists()
