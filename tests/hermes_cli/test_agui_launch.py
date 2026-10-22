"""AG-UI launchers are thin, non-interactive, and share argument semantics."""

import os
from pathlib import Path
import subprocess
import sys
import socket
import signal
from unittest.mock import Mock

import pytest


def test_main_parser_routes_agui_without_starting_chat(monkeypatch):
    from hermes_cli.main import _build_cli_parser
    from agui_adapter import entry

    serve = Mock()
    monkeypatch.setattr(entry, "main", serve)
    parser, _ = _build_cli_parser()
    args = parser.parse_args(
        ["agui", "--host", "127.0.0.1", "--port", "9123", "--token", "fake-token"]
    )
    args.func(args)
    serve.assert_called_once_with(host="127.0.0.1", port=9123, token="fake-token")


@pytest.mark.parametrize(
    "flag,output", [("--help", "--host"), ("--check", "Hermes AG-UI check OK")]
)
def test_module_flags_exit_without_a_tty_or_server(flag, output):
    result = subprocess.run(
        [sys.executable, "-m", "agui_adapter", flag],
        cwd=Path(__file__).resolve().parents[2],
        env=os.environ.copy(),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert output in result.stdout
    assert "Starting Hermes AG-UI adapter" not in result.stderr


def test_setup_parser_accepts_explicit_agui_section():
    from hermes_cli.main import _build_cli_parser

    parser, _ = _build_cli_parser()
    args = parser.parse_args(["setup", "agui"])
    assert args.section == "agui"


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("web_concurrency", [None, "2", "bad"])
def test_headless_module_serves_health_and_enforces_token(web_concurrency):
    import httpx
    from tests.waiting import wait_until

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    token = "r4N7g2V9q5H1c8L6p3S0u9F2d4K7m1W5"
    env = os.environ.copy()
    env.pop("WEB_CONCURRENCY", None)
    if web_concurrency is not None:
        env["WEB_CONCURRENCY"] = web_concurrency
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "agui_adapter",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--token",
            token,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        text=True,
    ) as process:
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=1) as client:

                def ready():
                    assert process.poll() is None, (
                        "server exited before becoming healthy"
                    )
                    try:
                        return client.get("/health").status_code == 200
                    except httpx.TransportError:
                        return False

                wait_until(ready, timeout=10, what="headless AG-UI health endpoint")
                assert client.post("/", json={}).status_code == 401
                assert (
                    client.post(
                        "/", json={}, headers={"X-Hermes-Session-Token": token}
                    ).status_code
                    == 400
                )
        finally:
            process.terminate()
            try:
                _, stderr = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                _, stderr = process.communicate(timeout=5)
        # uvicorn re-raises the captured signal after completing lifespan
        # shutdown. A SIGTERM exit is expected; SIGKILL or a traceback is not.
        assert process.returncode == -signal.SIGTERM, stderr
        assert "Traceback" not in stderr


def test_network_bind_without_token_fails_before_serving(monkeypatch):
    monkeypatch.delenv("HERMES_AGUI_SESSION_TOKEN", raising=False)
    result = subprocess.run(
        [sys.executable, "-m", "agui_adapter", "--host", "0.0.0.0"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
        cwd=Path(__file__).resolve().parents[2],
    )
    assert result.returncode == 1
    assert "Refusing to start" in result.stderr


@pytest.mark.parametrize("missing", ["ag_ui", "internal_adapter_bug"])
def test_dependency_errors_do_not_hide_internal_import_failures(
    monkeypatch, capsys, missing
):
    from agui_adapter import cli

    def unavailable(name):
        raise ModuleNotFoundError(f"No module named {missing!r}", name=missing)

    monkeypatch.setattr(cli.importlib, "import_module", unavailable)
    args = type("Args", (), {"check": True})()
    with pytest.raises(SystemExit if missing == "ag_ui" else ModuleNotFoundError):
        cli.run(args)
    stderr = capsys.readouterr().err
    assert ("hermes pm install --extra agui" in stderr) == (missing == "ag_ui")
