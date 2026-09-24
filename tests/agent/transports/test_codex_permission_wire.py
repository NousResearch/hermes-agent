"""Permission denial through the real session and JSON-RPC stdio client."""
import subprocess
import sys
from pathlib import Path

from agent.transports.codex_app_server_session import CodexAppServerSession


def test_permission_denial_round_trip(tmp_path, monkeypatch):
    # Only replace the executable at the spawn boundary. Session dispatch,
    # JSON serialization, pipes, reader threads and reply correlation are real.
    popen = subprocess.Popen
    peer = Path(__file__).parent / "fixtures" / "codex_permissions_server.py"

    def launch_peer(argv, **kwargs):
        assert argv[:2] == ["permission-fixture", "app-server"]
        return popen([sys.executable, "-u", str(peer)], **kwargs)

    monkeypatch.setattr(subprocess, "Popen", launch_peer)
    session = CodexAppServerSession(
        cwd=str(tmp_path), codex_home=str(tmp_path), codex_bin="permission-fixture",
    )
    try:
        result = session.run_turn("go", turn_timeout=10.0)
        assert result.error is None
        assert result.final_text == "DENIED-CLEANLY"
        assert result.projected_messages == [{"role": "assistant", "content": "DENIED-CLEANLY"}]
    finally:
        session.close()
