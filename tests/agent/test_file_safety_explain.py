"""Tests for ``explain_write_denial`` — the diagnostic counterpart to
``is_write_denied``.

``explain_write_denial`` returns a short string identifying which rule
blocked a write (or ``None`` when the path is allowed). It powers the
``hermes file-safety check`` CLI so users can see exactly which rule
matched instead of just "denied".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent import file_safety


class TestExplainWriteDenialAllowed:
    def test_returns_none_for_normal_path(self, tmp_path: Path):
        assert file_safety.explain_write_denial(str(tmp_path / "ok.txt")) is None

    def test_returns_none_for_path_under_cwd(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert file_safety.explain_write_denial("file.txt") is None


class TestExplainWriteDenialDenylistExact:
    def test_etc_shadow_matched_by_denylist(self):
        reason = file_safety.explain_write_denial("/etc/shadow")
        assert reason is not None
        assert reason.startswith("denylist:")
        assert "shadow" in reason

    def test_etc_passwd_matched_by_denylist(self):
        reason = file_safety.explain_write_denial("/etc/passwd")
        assert reason is not None
        assert reason.startswith("denylist:")

    def test_ssh_id_rsa_matched_by_denylist(self):
        path = str(Path.home() / ".ssh" / "id_rsa")
        reason = file_safety.explain_write_denial(path)
        assert reason is not None
        assert reason.startswith("denylist:")
        assert "id_rsa" in reason


class TestExplainWriteDenialPrefix:
    def test_aws_credentials_dir_matched_by_prefix(self):
        path = str(Path.home() / ".aws" / "credentials")
        reason = file_safety.explain_write_denial(path)
        assert reason is not None
        assert reason.startswith("prefix:")
        assert ".aws" in reason

    def test_etc_sudoers_d_subpath_matched_by_prefix(self):
        reason = file_safety.explain_write_denial("/etc/sudoers.d/myrule")
        assert reason is not None
        assert reason.startswith("prefix:")
        assert "sudoers.d" in reason


class TestExplainWriteDenialSafeRoot:
    def test_outside_safe_root_blocked(self, tmp_path: Path, monkeypatch):
        safe_root = tmp_path / "allowed"
        safe_root.mkdir()
        monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(safe_root))

        outside = tmp_path / "elsewhere" / "x.txt"
        reason = file_safety.explain_write_denial(str(outside))
        assert reason is not None
        assert reason.startswith("outside-safe-root:")

    def test_inside_safe_root_allowed(self, tmp_path: Path, monkeypatch):
        safe_root = tmp_path / "allowed"
        safe_root.mkdir()
        monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(safe_root))

        inside = safe_root / "x.txt"
        assert file_safety.explain_write_denial(str(inside)) is None

    def test_safe_root_itself_allowed(self, tmp_path: Path, monkeypatch):
        safe_root = tmp_path / "allowed"
        safe_root.mkdir()
        monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(safe_root))

        assert file_safety.explain_write_denial(str(safe_root)) is None


class TestIsWriteDeniedDelegatesToExplain:
    """is_write_denied must keep its bool contract — it now wraps
    explain_write_denial. Verifies they agree on every case above."""

    @pytest.mark.parametrize(
        "path",
        [
            "/etc/shadow",
            "/etc/passwd",
            "/etc/sudoers.d/myrule",
            str(Path.home() / ".ssh" / "id_rsa"),
            str(Path.home() / ".aws" / "credentials"),
        ],
    )
    def test_denied_paths_agree(self, path: str):
        assert file_safety.is_write_denied(path) is True
        assert file_safety.explain_write_denial(path) is not None

    def test_allowed_path_agrees(self, tmp_path: Path):
        path = str(tmp_path / "ok.txt")
        assert file_safety.is_write_denied(path) is False
        assert file_safety.explain_write_denial(path) is None