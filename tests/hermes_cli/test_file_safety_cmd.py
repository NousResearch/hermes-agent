"""Tests for ``hermes file-safety check`` CLI command."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.file_safety_cmd import file_safety_command


def _args(path: str, *, json_out: bool = False) -> SimpleNamespace:
    return SimpleNamespace(path=path, json=json_out)


class TestFileSafetyCheckHuman:
    def test_allowed_path_exits_zero(self, tmp_path: Path, capsys):
        with pytest.raises(SystemExit) as exc:
            file_safety_command(_args(str(tmp_path / "x.txt")))
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "OK" in out or "allowed" in out.lower()

    def test_denylist_path_exits_one(self, capsys):
        with pytest.raises(SystemExit) as exc:
            file_safety_command(_args("/etc/shadow"))
        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "denylist" in out
        assert "shadow" in out

    def test_prefix_path_exits_one(self, capsys):
        with pytest.raises(SystemExit) as exc:
            file_safety_command(_args(str(Path.home() / ".aws" / "credentials")))
        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "prefix" in out
        assert ".aws" in out

    def test_safe_root_violation_exits_one(self, tmp_path: Path, monkeypatch, capsys):
        safe_root = tmp_path / "allowed"
        safe_root.mkdir()
        monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(safe_root))

        with pytest.raises(SystemExit) as exc:
            file_safety_command(_args(str(tmp_path / "elsewhere.txt")))
        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "outside-safe-root" in out


class TestFileSafetyCheckJSON:
    def test_allowed_emits_json_with_allowed_true(self, tmp_path: Path, capsys):
        with pytest.raises(SystemExit) as exc:
            file_safety_command(_args(str(tmp_path / "ok.txt"), json_out=True))
        assert exc.value.code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["write"]["allowed"] is True
        assert payload["write"]["reason"] is None
        assert payload["read"]["allowed"] is True
        assert "resolved_path" in payload

    def test_denied_emits_json_with_reason(self, capsys):
        with pytest.raises(SystemExit) as exc:
            file_safety_command(_args("/etc/passwd", json_out=True))
        assert exc.value.code == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["write"]["allowed"] is False
        assert payload["write"]["reason"].startswith("denylist:")


class TestFileSafetyCheckReadBlock:
    """Read-block errors come from get_read_block_error (skill cache files)."""

    def test_internal_skill_cache_read_blocked(self, tmp_path: Path, monkeypatch, capsys):
        # Point HERMES_HOME at tmp_path; the cache dir is then
        # tmp_path/skills/.hub/index-cache
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cache_path = tmp_path / "skills" / ".hub" / "index-cache" / "x.json"

        with pytest.raises(SystemExit) as exc:
            file_safety_command(_args(str(cache_path), json_out=True))
        assert exc.value.code == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["read"]["allowed"] is False
        assert payload["read"]["reason"]
