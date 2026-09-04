"""Credential-source authority for the Desktop dashboard session token."""

from __future__ import annotations

import contextlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from hermes_cli import env_loader


_MARKER = "HERMES_DESKTOP"
_TOKEN = "HERMES_DASHBOARD_SESSION_TOKEN"
_FRESH = "fresh-launch-token-00000000000000000000"
_STALE = "stale-dotenv-token-00000000000000000000"


def _write_env(path: Path, **values: str) -> None:
    path.write_text(
        "".join(f"{name}={value}\n" for name, value in values.items()),
        encoding="utf-8",
    )


def test_marked_launch_credential_survives_every_override_layer(
    tmp_path, monkeypatch
):
    """The one captured launcher credential wins across later dotenv layers."""
    first = tmp_path / "first.env"
    second = tmp_path / "second.env"
    _write_env(first, HERMES_DESKTOP="0", HERMES_DASHBOARD_SESSION_TOKEN="first")
    _write_env(
        second,
        HERMES_DESKTOP="0",
        HERMES_DASHBOARD_SESSION_TOKEN="second",
        SOURCE_AUTHORITY_PROBE="second",
    )

    monkeypatch.setattr(env_loader, "_DESKTOP_LAUNCH_CREDENTIAL", ("1", _FRESH))
    monkeypatch.setenv(_MARKER, "1")
    monkeypatch.setenv(_TOKEN, _FRESH)

    env_loader._load_dotenv_with_fallback(first, override=True)
    env_loader._load_dotenv_with_fallback(second, override=True)

    assert os.environ[_MARKER] == "1"
    assert os.environ[_TOKEN] == _FRESH
    assert os.environ["SOURCE_AUTHORITY_PROBE"] == "second"


def test_unmarked_shell_token_keeps_normal_dotenv_precedence(tmp_path, monkeypatch):
    """A plain shell export has no Desktop-launch authority."""
    dotenv = tmp_path / "profile.env"
    _write_env(dotenv, HERMES_DASHBOARD_SESSION_TOKEN=_STALE)

    monkeypatch.setattr(env_loader, "_DESKTOP_LAUNCH_CREDENTIAL", None)
    monkeypatch.delenv(_MARKER, raising=False)
    monkeypatch.setenv(_TOKEN, _FRESH)

    env_loader._load_dotenv_with_fallback(dotenv, override=True)

    assert os.environ[_TOKEN] == _STALE


def test_dotenv_cannot_mint_launcher_authority_for_a_later_layer(
    tmp_path, monkeypatch
):
    """Names loaded from dotenv are coordinates, not launch provenance."""
    first = tmp_path / "first.env"
    second = tmp_path / "second.env"
    _write_env(
        first,
        HERMES_DESKTOP="1",
        HERMES_DASHBOARD_SESSION_TOKEN="counterfeit-first-layer",
    )
    _write_env(second, HERMES_DASHBOARD_SESSION_TOKEN="second-layer-wins")

    monkeypatch.setattr(env_loader, "_DESKTOP_LAUNCH_CREDENTIAL", None)
    monkeypatch.delenv(_MARKER, raising=False)
    monkeypatch.delenv(_TOKEN, raising=False)

    env_loader._load_dotenv_with_fallback(first, override=True)
    env_loader._load_dotenv_with_fallback(second, override=True)

    assert os.environ[_MARKER] == "1"
    assert os.environ[_TOKEN] == "second-layer-wins"


def test_marker_without_launch_token_has_no_credential_authority(
    tmp_path, monkeypatch
):
    """Both launch coordinates are required; the marker alone grants nothing."""
    dotenv = tmp_path / "profile.env"
    _write_env(dotenv, HERMES_DASHBOARD_SESSION_TOKEN=_STALE)

    monkeypatch.setattr(env_loader, "_DESKTOP_LAUNCH_CREDENTIAL", None)
    monkeypatch.setenv(_MARKER, "1")
    monkeypatch.delenv(_TOKEN, raising=False)

    env_loader._load_dotenv_with_fallback(dotenv, override=True)

    assert os.environ[_TOKEN] == _STALE


def test_non_override_layer_does_not_reclaim_a_later_token_owner(
    tmp_path, monkeypatch
):
    """The launch snapshot acts only where dotenv is allowed to overwrite."""
    dotenv = tmp_path / "bootstrap.env"
    _write_env(dotenv, HERMES_DASHBOARD_SESSION_TOKEN=_STALE)

    monkeypatch.setattr(env_loader, "_DESKTOP_LAUNCH_CREDENTIAL", ("1", _FRESH))
    monkeypatch.setenv(_MARKER, "1")
    monkeypatch.setenv(_TOKEN, "later-owner-token")

    env_loader._load_dotenv_with_fallback(dotenv, override=False)

    assert os.environ[_TOKEN] == "later-owner-token"


def test_launch_credential_is_restored_when_dotenv_loading_raises(
    tmp_path, monkeypatch
):
    """A failed dotenv layer cannot leave the process on its partial credential."""
    dotenv = tmp_path / "profile.env"
    dotenv.write_text("IGNORED=1\n", encoding="utf-8")

    monkeypatch.setattr(env_loader, "_DESKTOP_LAUNCH_CREDENTIAL", ("1", _FRESH))
    monkeypatch.setenv(_MARKER, "1")
    monkeypatch.setenv(_TOKEN, _FRESH)

    def fail_after_mutation(*_args, **_kwargs):
        os.environ[_MARKER] = "0"
        os.environ[_TOKEN] = _STALE
        raise RuntimeError("dotenv failure")

    monkeypatch.setattr(env_loader, "load_dotenv", fail_after_mutation)

    with pytest.raises(RuntimeError, match="dotenv failure"):
        env_loader._load_dotenv_with_fallback(dotenv, override=True)

    assert os.environ[_MARKER] == "1"
    assert os.environ[_TOKEN] == _FRESH


@pytest.mark.parametrize(
    ("desktop_marker", "launch_token", "expected_marker", "expected_token"),
    [
        (True, _FRESH, "1", _FRESH),
        (False, _FRESH, "0", _STALE),
        (True, "", "0", _STALE),
    ],
)
def test_fresh_process_captures_only_the_real_launch_pair(
    tmp_path, desktop_marker, launch_token, expected_marker, expected_token
):
    """Exercise import-time capture plus the public loader in a fresh process."""
    case = "desktop" if desktop_marker and launch_token else "plain"
    home = tmp_path / case
    home.mkdir()
    _write_env(
        home / ".env",
        HERMES_DESKTOP="0",
        HERMES_DASHBOARD_SESSION_TOKEN=_STALE,
    )

    child_env = os.environ.copy()
    child_env["HERMES_HOME"] = str(home)
    child_env[_TOKEN] = launch_token
    if desktop_marker:
        child_env[_MARKER] = "1"
    else:
        child_env.pop(_MARKER, None)

    script = """
import json
import os
from hermes_cli import env_loader

env_loader._apply_managed_env = lambda: None
env_loader._reapply_terminal_config_bridge = lambda _home: None
env_loader.load_hermes_dotenv(
    hermes_home=os.environ["HERMES_HOME"],
    load_external_secrets=False,
)
print("SOURCE_AUTHORITY_RESULT=" + json.dumps({
    "marker": os.environ.get("HERMES_DESKTOP"),
    "token": os.environ.get("HERMES_DASHBOARD_SESSION_TOKEN"),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        env=child_env,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    result_line = next(
        line
        for line in completed.stdout.splitlines()
        if line.startswith("SOURCE_AUTHORITY_RESULT=")
    )
    result = json.loads(result_line.split("=", 1)[1])

    assert result == {"marker": expected_marker, "token": expected_token}


def _request_status(port: int, token: str) -> int:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/sessions?limit=1&offset=0",
        headers={"X-Hermes-Session-Token": token},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _log_tail(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return "<no process log>"
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


@contextlib.contextmanager
def _running_serve(home: Path):
    """Run the checkout's real headless backend and yield its bound port."""
    child_env = os.environ.copy()
    child_env.update(
        {
            "HERMES_HOME": str(home),
            "HOME": str(home.parent),
            _MARKER: "1",
            _TOKEN: _FRESH,
        }
    )
    for name in (
        "HERMES_DASHBOARD_PUBLIC_URL",
        "HERMES_DASHBOARD_BASIC_AUTH_USERNAME",
        "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD",
        "HERMES_MANAGED_DIR",
    ):
        child_env.pop(name, None)
    repo_root = Path(__file__).resolve().parents[2]
    child_env["PYTHONPATH"] = str(repo_root) + os.pathsep + child_env.get(
        "PYTHONPATH", ""
    )
    port = _free_port()
    log_path = home / "serve-subprocess.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "hermes_cli.main",
                "serve",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=repo_root,
            env=child_env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            deadline = time.monotonic() + 240
            last_error = "no response"
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    log.flush()
                    pytest.fail(
                        "hermes serve exited before accepting requests:\n"
                        + _log_tail(log_path)
                    )
                try:
                    status = _request_status(port, _FRESH)
                    if status == 200:
                        break
                    if _request_status(port, _STALE) == 200:
                        pytest.fail(
                            "hermes serve accepted the stale dotenv token instead "
                            "of the marked launch token"
                        )
                    last_error = f"HTTP {status}"
                except urllib.error.URLError as exc:
                    last_error = str(exc.reason)
                time.sleep(0.1)
            else:
                log.flush()
                pytest.fail(
                    f"hermes serve did not accept the launch token within 240s "
                    f"({last_error}):\n{_log_tail(log_path)}"
                )
            yield port
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)


def test_real_serve_accepts_launch_token_and_rejects_stale_dotenv_token(tmp_path):
    """Prove the exact Desktop credential at the live HTTP authentication seam."""
    home = tmp_path / "serve-home"
    home.mkdir()
    _write_env(
        home / ".env",
        HERMES_DESKTOP="0",
        HERMES_DASHBOARD_SESSION_TOKEN=_STALE,
        OPENAI_API_KEY="sk-test-not-used",
    )

    with _running_serve(home) as port:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=10) as response:
            token_page = response.read().decode("utf-8")
        fresh_status = _request_status(port, _FRESH)
        stale_status = _request_status(port, _STALE)

    assert _FRESH in token_page
    assert _STALE not in token_page
    assert fresh_status == 200
    assert stale_status in {401, 403}
