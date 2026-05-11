from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[3] / "docker" / "daimon-sandbox" / "gh_client.py"
SPEC = importlib.util.spec_from_file_location("daimon_gh_client_test_module", MODULE_PATH)
gh_client = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(gh_client)


class FakeSocket:
    def __init__(self, *_args, **_kwargs):
        self.sent = b""
        self._chunks = [json.dumps({"ok": True, "exit_code": 0, "stdout": "ok\n", "stderr": ""}).encode(), b""]

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def sendall(self, payload):
        self.sent += payload

    def shutdown(self, _how):
        pass

    def recv(self, _size):
        return self._chunks.pop(0)


def test_request_sends_argv_and_cwd(monkeypatch, tmp_path):
    fake = FakeSocket()
    monkeypatch.setattr(gh_client.socket, "create_connection", lambda address, timeout: fake)
    monkeypatch.chdir(tmp_path)

    response = gh_client._request(["issue", "list", "-R", "NousResearch/hermes-agent"])

    request = json.loads(fake.sent.decode())
    assert request["argv"] == ["issue", "list", "-R", "NousResearch/hermes-agent"]
    assert request["cwd"] == str(tmp_path)
    assert request["timeout_sec"] == 60
    assert response["stdout"] == "ok\n"
