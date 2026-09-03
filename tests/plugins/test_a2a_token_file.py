from __future__ import annotations

from pathlib import Path
import os

import pytest

from plugins.platforms.a2a.tools import _auth_header


def test_1221_bearer_token_file_is_loaded_without_putting_value_in_config(tmp_path: Path) -> None:
    token_file = tmp_path / "sender-to-receiver.token"
    token_file.write_text("peer-secret-value-0123456789abcdef\n", encoding="utf-8")
    token_file.chmod(0o600)

    assert _auth_header({"type": "bearer", "token_file": str(token_file)}) == {
        "Authorization": "Bearer peer-secret-value-0123456789abcdef"
    }


def test_1221_bearer_token_file_fails_closed_on_missing_or_permissive_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.token"
    with pytest.raises(ValueError, match="token_file"):
        _auth_header({"type": "bearer", "token_file": str(missing)})

    token_file = tmp_path / "permissive.token"
    token_file.write_text("must-not-leak", encoding="utf-8")
    token_file.chmod(0o644)
    with pytest.raises(ValueError, match="mode 0600") as exc:
        _auth_header({"type": "bearer", "token_file": str(token_file)})
    assert "must-not-leak" not in str(exc.value)


def test_1221_inline_and_file_bearers_cannot_be_combined(tmp_path: Path) -> None:
    token_file = tmp_path / "peer.token"
    token_file.write_text("file-value", encoding="utf-8")
    token_file.chmod(0o600)
    with pytest.raises(ValueError, match="cannot combine"):
        _auth_header({"type": "bearer", "token": "inline-value", "token_file": str(token_file)})


def test_1221_bearer_token_file_rejects_short_material(tmp_path: Path) -> None:
    token_file = tmp_path / "short.token"
    token_file.write_text("short", encoding="utf-8")
    token_file.chmod(0o600)
    with pytest.raises(ValueError, match="at least 32 bytes"):
        _auth_header({"type": "bearer", "token_file": str(token_file)})


def test_1221_bearer_token_file_rejects_symlink(tmp_path: Path) -> None:
    if not hasattr(os, "O_NOFOLLOW"):
        pytest.skip("O_NOFOLLOW unavailable")
    target = tmp_path / "target.token"
    target.write_text("peer-secret-value-0123456789abcdef", encoding="utf-8")
    target.chmod(0o600)
    link = tmp_path / "linked.token"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="unreadable"):
        _auth_header({"type": "bearer", "token_file": str(link)})


def test_1221_bearer_token_file_fails_closed_without_no_follow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token_file = tmp_path / "peer.token"
    token_file.write_text("peer-secret-value-0123456789abcdef", encoding="utf-8")
    token_file.chmod(0o600)
    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)
    with pytest.raises(ValueError, match="secure no-follow"):
        _auth_header({"type": "bearer", "token_file": str(token_file)})


def test_1221_bearer_token_file_rejects_unread_trailing_content(tmp_path: Path) -> None:
    token_file = tmp_path / "oversized.token"
    token_file.write_text("x" * 4096 + "\nextra", encoding="utf-8")
    token_file.chmod(0o600)
    with pytest.raises(ValueError, match="at most 4096 bytes"):
        _auth_header({"type": "bearer", "token_file": str(token_file)})
