"""Terminal artifacts must be host-readable before the final reply becomes durable."""

import base64
import os
import shlex
import subprocess
from pathlib import Path

import pytest

from tests.fakes.fake_llm_provider import FakeLLMServer, Text, write_hermes_home
from tools.environments.base import BaseEnvironment


class RemoteFiles(BaseEnvironment):
    """A separate path namespace using the real bounded shell/base64 file transport."""

    _remote_home = "/home/worker"

    def __init__(self, root, remote_path="/workspace/chart.svg"):
        self.root = root
        self.remote_path = remote_path
        self.fetched = []

    def execute(self, command, cwd="", **kwargs):
        remote = self.remote_path
        source = str(self.root / "chart.svg")
        if shlex.quote(remote) not in command:
            return {"output": "", "returncode": 1}
        command = command.replace(shlex.quote(remote), shlex.quote(source))
        result = subprocess.run(["bash", "-c", command], capture_output=True, text=True, check=False)
        return {"output": result.stdout.replace(source, remote), "returncode": result.returncode}

    def fetch_file(self, remote_path, local_dest, *, max_bytes):
        self.fetched.append(remote_path)
        super().fetch_file(remote_path, local_dest, max_bytes=max_bytes)

    def cleanup(self):
        pass


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("remote_path", ["/workspace/chart.svg", "/workspace/my script.py"])
def test_remote_reply_is_downloadable_and_durable_after_terminal_file_is_removed(
    tmp_path, monkeypatch, request, remote_path,
):
    from hermes_cli import plugins, web_server
    from run_agent import AIAgent
    from hermes_state import SessionDB
    from starlette.testclient import TestClient
    from tools import terminal_tool

    terminal_root = tmp_path / "terminal"
    terminal_root.mkdir()
    payload = (b'<svg xmlns="http://www.w3.org/2000/svg"><circle r="10"/></svg>'
               if remote_path.endswith(".svg") else b'print("remote script")\n')
    source = terminal_root / "chart.svg"
    source.write_bytes(payload)
    env = RemoteFiles(terminal_root, remote_path)
    task_id = "remote-artifact-turn"
    monkeypatch.setitem(terminal_tool._active_environments, task_id, env)
    home = Path(os.environ["HERMES_HOME"])
    db = SessionDB(db_path=home / "state.db")
    request.addfinalizer(db.close)
    raw = f"Here is the file.\n\nMEDIA:{remote_path}"
    with FakeLLMServer([Text(raw, chunk_chars=5)]) as server:
        write_hermes_home(home, server.base_url, extra_config=(
            "updates:\n  check: false\nmodels_dev:\n  url: http://127.0.0.1:9/api.json\n"
            "skills:\n  creation_nudge_interval: 0\nmemory:\n  nudge_interval: 0\n"
        ))
        plugins._reset_plugin_managers_for_tests()
        try:
            agent = AIAgent(base_url=server.base_url, api_key="sk-fake-e2e", provider="custom",
                            model="fake-model", quiet_mode=True, skip_context_files=True, skip_memory=True,
                            enabled_toolsets=["todo"], platform="desktop", session_id="artifact-session", session_db=db)
            result = agent.run_conversation("Send the chart", task_id=task_id)
        finally:
            plugins._reset_plugin_managers_for_tests()

    assert "MEDIA:" in result["final_response"], repr(result["final_response"])
    delivered = result["final_response"].split("MEDIA:", 1)[1].strip().strip('"')
    assert Path(delivered).is_file(), result["final_response"]
    assert Path(delivered).read_bytes() == payload
    assert env.fetched == [remote_path]
    assert result["response_transformed"] is True
    stored = agent._session_db.get_messages(agent.session_id)
    assert any(row["role"] == "assistant" and row["content"] == result["final_response"] for row in stored)
    source.unlink()

    monkeypatch.setattr(web_server.app.state, "auth_required", False, raising=False)
    with TestClient(web_server.app) as client:
        client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
        preview = client.get("/api/fs/read-data-url", params={"path": delivered})
        download = client.get("/api/fs/download", params={"path": delivered})
    assert preview.status_code == 200, preview.text
    assert base64.b64decode(preview.json()["dataUrl"].split(",", 1)[1]) == payload
    assert download.status_code == 200, download.text
    assert download.content == payload


@pytest.mark.platforms("posix")
def test_exports_are_task_bound_without_host_fallback_and_host_artifacts_stay_unchanged(tmp_path, monkeypatch):
    from agent.terminal_artifacts import materialize_terminal_media
    from tools import terminal_tool

    home = Path(os.environ["HERMES_HOME"])
    host_artifact = home / "cache" / "generated" / "existing.svg"
    host_artifact.parent.mkdir(parents=True)
    host_artifact.write_bytes(b"host-generated")
    collision = tmp_path / "same-path.svg"
    collision.write_bytes(b"wrong host copy")
    envs = []
    for name in ("first", "second"):
        root = tmp_path / name
        root.mkdir()
        (root / "chart.svg").write_bytes(name.encode())
        env = RemoteFiles(root, str(collision))
        envs.append(env)
        monkeypatch.setitem(terminal_tool._active_environments, name, env)

    host_reply = f"MEDIA:{host_artifact}"
    assert materialize_terminal_media(host_reply, "first") == host_reply
    protected = f"```text\nMEDIA:{collision}\n```\n"
    raw = protected + f"MEDIA:{collision}\nMEDIA:{collision}"
    for name, env in zip(("first", "second"), envs):
        result = materialize_terminal_media(raw, name)
        assert result.startswith(protected)
        paths = [line.removeprefix("MEDIA:").strip('"') for line in result[len(protected):].splitlines()]
        assert paths[0] == paths[1]
        assert Path(paths[0]).read_bytes() == name.encode()
        assert env.fetched == [str(collision)]

    # A vanished terminal file never falls back to the readable host namesake.
    (envs[0].root / "chart.svg").unlink()
    failed = materialize_terminal_media(f"MEDIA:{collision}", "first")
    assert "MEDIA:" not in failed
    assert "Attachment unavailable" in failed
    assert collision.read_bytes() == b"wrong host copy"
    for invalid in ("/tmp/bad\x00.svg", "/home/worker/.ssh/id_rsa"):
        refused = materialize_terminal_media(f"MEDIA:{invalid}", "first")
        assert "MEDIA:" not in refused
        assert "Attachment unavailable" in refused


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("linked_home", [False, True], ids=["spaced-script", "symlinked-home"])
def test_managed_host_attachment_paths_remain_deliverable(tmp_path, monkeypatch, linked_home):
    from agent.terminal_artifacts import materialize_terminal_media
    from tools import terminal_tool

    home = Path(os.environ["HERMES_HOME"])
    if linked_home:
        alias = tmp_path / "home-alias"
        alias.symlink_to(home, target_is_directory=True)
        monkeypatch.setenv("HERMES_HOME", str(alias))
        home = alias
    artifact = home / "cache" / "generated" / ("picture.svg" if linked_home else "my script.py")
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"host artifact")
    env = RemoteFiles(tmp_path)
    monkeypatch.setitem(terminal_tool._active_environments, "host-artifact", env)

    raw = f"MEDIA:{artifact}"
    result = materialize_terminal_media(raw, "host-artifact")
    assert "Attachment unavailable" not in result
    delivered = result.removeprefix("MEDIA:").strip('"')
    assert Path(delivered).read_bytes() == b"host artifact"
    assert env.fetched == []
