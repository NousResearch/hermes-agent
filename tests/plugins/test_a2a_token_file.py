from __future__ import annotations

import os
from pathlib import Path

import pytest

from plugins.platforms.a2a import protocol, tools


FIXTURE_TOKEN = "synthetic-peer-credential-0123456789abcdef"


def _write_token(path: Path, value: str = FIXTURE_TOKEN) -> Path:
    path.write_text(value, encoding="utf-8")
    path.chmod(0o600)
    return path


def test_bearer_token_file_builds_authorization_without_inline_config(tmp_path: Path) -> None:
    token_file = _write_token(tmp_path / "peer.token")

    assert tools._auth_header({"type": "bearer", "token_file": str(token_file)}) == {
        "Authorization": f"Bearer {FIXTURE_TOKEN}"
    }


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
