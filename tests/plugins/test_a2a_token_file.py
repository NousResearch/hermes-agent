from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from plugins.platforms.a2a import protocol, tools


FIXTURE_TOKEN = "synthetic-peer-credential-0123456789abcdef"
MAX_BEARER_FILE_BYTES = 4096
SECURE_OPEN_SUPPORTED = hasattr(os, "O_NOFOLLOW")
requires_secure_open = pytest.mark.skipif(
    not SECURE_OPEN_SUPPORTED,
    reason="secure token-file reads require POSIX O_NOFOLLOW",
)


def _write_token(path: Path, value: str = FIXTURE_TOKEN) -> Path:
    path.write_text(value, encoding="utf-8")
    path.chmod(0o600)
    return path


@requires_secure_open
def test_bearer_token_file_builds_authorization_without_inline_config(tmp_path: Path) -> None:
    token_file = _write_token(tmp_path / "peer.token")

    assert tools._auth_header({"type": "bearer", "token_file": str(token_file)}) == {
        "Authorization": f"Bearer {FIXTURE_TOKEN}"
    }


@requires_secure_open
@pytest.mark.parametrize("case", ["missing", "empty", "unreadable"])
def test_bearer_token_file_failures_are_closed_and_do_not_leak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    token_file = tmp_path / f"{case}.token"
    if case != "missing":
        _write_token(token_file, "" if case == "empty" else FIXTURE_TOKEN)
    if case == "unreadable":
        real_open = os.open

        def refused_open(path, flags, mode=0o777, *, dir_fd=None):
            if Path(path) == token_file:
                raise PermissionError("synthetic refusal")
            if dir_fd is None:
                return real_open(path, flags, mode)
            return real_open(path, flags, mode, dir_fd=dir_fd)

        monkeypatch.setattr(os, "open", refused_open)

    with pytest.raises(ValueError, match="token_file") as exc:
        tools._auth_header({"type": "bearer", "token_file": str(token_file)})

    message = str(exc.value)
    assert FIXTURE_TOKEN not in message
    assert "synthetic refusal" not in message


@requires_secure_open
def test_bearer_token_file_rejects_permissive_mode_without_leaking(tmp_path: Path) -> None:
    token_file = _write_token(tmp_path / "peer.token")
    token_file.chmod(0o644)

    with pytest.raises(ValueError, match="mode 0600") as exc:
        tools._auth_header({"type": "bearer", "token_file": str(token_file)})

    assert FIXTURE_TOKEN not in str(exc.value)


def test_inline_and_file_bearers_are_mutually_exclusive(tmp_path: Path) -> None:
    token_file = _write_token(tmp_path / "peer.token")

    with pytest.raises(ValueError, match="cannot combine"):
        tools._auth_header(
            {"type": "bearer", "token": "inline-synthetic", "token_file": str(token_file)}
        )


def test_bearer_token_file_fails_closed_without_nofollow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token_file = _write_token(tmp_path / "peer.token")
    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)

    with pytest.raises(ValueError, match="secure no-follow open is unavailable"):
        tools._auth_header({"type": "bearer", "token_file": str(token_file)})


@requires_secure_open
def test_bearer_token_file_does_not_block_on_fifo(tmp_path: Path) -> None:
    fifo = tmp_path / "peer.token"
    os.mkfifo(fifo, mode=0o600)
    repo_root = Path(__file__).resolve().parents[2]
    probe = (
        "from plugins.platforms.a2a import tools\n"
        f"path = {str(fifo)!r}\n"
        "try:\n"
        "    tools._auth_header({'type': 'bearer', 'token_file': path})\n"
        "except ValueError as exc:\n"
        "    assert 'regular file' in str(exc), str(exc)\n"
        "else:\n"
        "    raise AssertionError('FIFO was accepted')\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=2,
    )

    assert completed.returncode == 0, completed.stderr


@requires_secure_open
def test_bearer_token_file_read_stays_bounded_if_file_grows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token_file = _write_token(tmp_path / "peer.token", "initial-token")
    real_fstat = os.fstat
    real_read = os.read
    grew = False
    requested_sizes: list[int] = []

    def grow_after_stat(descriptor: int):
        nonlocal grew
        result = real_fstat(descriptor)
        if not grew:
            with token_file.open("ab") as handle:
                handle.write(b"x" * (MAX_BEARER_FILE_BYTES + 1))
            grew = True
        return result

    def record_read(descriptor: int, size: int) -> bytes:
        requested_sizes.append(size)
        return real_read(descriptor, size)

    monkeypatch.setattr(os, "fstat", grow_after_stat)
    monkeypatch.setattr(os, "read", record_read)

    with pytest.raises(ValueError, match="at most 4096 bytes"):
        tools._auth_header({"type": "bearer", "token_file": str(token_file)})

    assert requested_sizes == [MAX_BEARER_FILE_BYTES + 1]


@requires_secure_open
def test_bearer_token_file_sanitizes_invalid_utf8(tmp_path: Path) -> None:
    token_file = tmp_path / "peer.token"
    token_file.write_bytes(b"synthetic-invalid-utf8-" + b"\xff")
    token_file.chmod(0o600)

    with pytest.raises(ValueError) as exc:
        tools._auth_header({"type": "bearer", "token_file": str(token_file)})

    assert str(exc.value) == "bearer token_file is unreadable"
    assert "codec" not in str(exc.value)
    assert "invalid" not in str(exc.value)


@requires_secure_open
def test_bearer_token_file_accepts_short_nonempty_value(tmp_path: Path) -> None:
    token_file = _write_token(tmp_path / "peer.token", "tok")

    assert tools._auth_header({"type": "bearer", "token_file": str(token_file)}) == {
        "Authorization": "Bearer tok"
    }


@requires_secure_open
def test_native_a2a_call_sends_file_backed_header(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    token_file = _write_token(tmp_path / "peer.token")
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        tools,
        "_load_config",
        lambda: {
            "a2a_agents": {
                "receiver": {
                    "url": "https://receiver.invalid/a2a/",
                    "auth": {"type": "bearer", "token_file": str(token_file)},
                }
            }
        },
    )
    monkeypatch.setattr(tools, "_fetch_card", lambda *_args, **_kwargs: None)

    def fake_post(url: str, body: dict, headers: dict, timeout: int) -> dict:
        captured.update(url=url, headers=dict(headers), timeout=timeout)
        return protocol.jsonrpc_result(
            body["id"],
            protocol.build_task("task-fixture", "ctx-fixture", protocol.STATE_COMPLETED, "accepted"),
        )

    monkeypatch.setattr(tools, "_http_post_json", fake_post)

    result = tools.a2a_call({"agent": "receiver", "message": "synthetic request"})

    assert "accepted" in result
    assert captured["url"] == "https://receiver.invalid/a2a"
    assert captured["headers"] == {"Authorization": f"Bearer {FIXTURE_TOKEN}"}
