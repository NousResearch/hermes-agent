"""Exercise the actual viewer listener at the package's declared dependency floors."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import tomllib
import unittest
import pytest


@pytest.mark.linux_only
class ViewerFloorTests(unittest.TestCase):
    def test_declared_minimum_dependencies_start_http_and_websocket(self):
        uv = shutil.which("uv")
        if uv is None:
            self.skipTest("uv is required for the isolated minimum-dependency lane")
        root = Path(__file__).resolve().parents[1]
        dependencies = tomllib.loads((root / "pyproject.toml").read_text())["project"][
            "dependencies"
        ]
        # Test packaging metadata behavior, not a frozen expected version.
        floors = [
            dependency.replace(">=", "==").split(",")[0] for dependency in dependencies
            if dependency.split("[")[0].split(">=")[0].split("==")[0] in {"fastapi", "uvicorn", "websockets", "PyYAML", "pyyaml"}
        ]
        with tempfile.TemporaryDirectory(prefix="viewer-floor-") as directory:
            python = str(Path(directory) / "bin/python")
            subprocess.run(
                [uv, "venv", "--python", "3.11", directory],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [uv, "pip", "install", "--python", python, *floors],
                check=True,
                capture_output=True,
                text=True,
            )
            code = """
import urllib.request
import uvicorn, fastapi, websockets
from websockets.sync.client import connect
from websockets.exceptions import InvalidStatus
from realms.bridge import ViewerServer
from realms.lifecycle import identity
from pathlib import Path
import os, socket, tempfile, threading
runtime = tempfile.TemporaryDirectory(prefix="realm-floor-")
endpoint = Path(runtime.name) / "vnc"
listener = socket.socket(socket.AF_UNIX)
listener.bind(str(endpoint))
endpoint.chmod(0o600)
listener.listen()
listener.settimeout(5)
def rfb():
    with listener.accept()[0] as peer:
        peer.settimeout(5)
        peer.sendall(b"RFB 003.008\\n")
        peer.recv(1024)
worker = threading.Thread(target=rfb)
worker.start()
record = {"id": "test", "generation": "floor", "runtime_dir": runtime.name,
          "vnc_socket": str(endpoint), "processes": {"worker": identity(os.getpid())}}
server = ViewerServer(lambda rid: record if rid == "test" else None)
try:
    server.start()
    with urllib.request.urlopen(server.origin + "/realms/test/view", timeout=5) as response:
        assert response.status == 200
        assert b"Take over" in response.read()
    try:
        with connect(server.origin.replace("http:", "ws:") + "/api/realms/test/vnc",
                     origin=server.origin, subprotocols=["binary", "realm.invalid"]):
            raise AssertionError("unauthorized websocket accepted")
    except InvalidStatus as exc:
        assert exc.response.status_code == 403
    with connect(server.origin.replace("http:", "ws:") + "/api/realms/test/vnc",
                 origin=server.origin, subprotocols=["binary", "realm." + server.issue("test")]) as ws:
        assert ws.recv(timeout=5) == b"RFB 003.008\\n"
    print("HTTP 200; WebSocket authorization 403 and authenticated RFB frame; uvicorn=" + uvicorn.__version__
          + "; fastapi=" + fastapi.__version__ + "; websockets=" + websockets.__version__)
finally:
    server.stop()
    worker.join(6)
    listener.close()
    runtime.cleanup()
    assert not worker.is_alive()
"""
            result = subprocess.run(
                [python, "-c", code],
                cwd=root / "plugins/hermes-realms",
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            print(result.stdout.strip())


if __name__ == "__main__":
    unittest.main()
