"""Issue #84843: self-update must not invalidate persisted MCP OAuth tokens.

MCP OAuth tokens and RFC 7591 client registrations are persisted in
``$HERMES_HOME/mcp-tokens/`` — outside the install tree. If a self-update
deletes or replaces them, the persisted client registration is orphaned and
the next token refresh falls back to an interactive browser flow, raising
``OAuthNonInteractiveError`` in headless contexts (cron) — the exact symptom
reported in #84843.

These tests pin the two defensive layers added to the update pipeline:
  * the ZIP-update preserve set keeps a legacy in-tree ``mcp-tokens/``
    (layouts where HERMES_HOME == install root) untouched through the
    two-phase replace,
  * the post-update verifier ``_verify_mcp_tokens_preserved`` confirms the
    canonical ``$HERMES_HOME/mcp-tokens/`` is intact, and an end-to-end
    ``_update_via_zip`` run leaves every token file byte-identical.
"""

from __future__ import annotations

import inspect
import logging
import os
import shutil
import sys
import textwrap
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import main as hermes_main
from hermes_cli import update_cmd


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_TOKEN_FILES = {
    "notion.json": b'{"access_token": "tok-123", "expires_at": 9999999999}',
    ".client.json": b'{"client_id": "cid-1", "client_secret": "sec-1"}',
    ".meta.json": b'{"token_endpoint": "https://api.notion.com/v1/oauth/token"}',
}


def _seed_hermes_home_tokens(home: Path) -> dict[str, tuple[int, int]]:
    """Create ``$HERMES_HOME/mcp-tokens/`` with realistic token files.

    Returns {filename: (size, mtime_ns)} recorded before any update runs.
    """
    token_dir = home / "mcp-tokens"
    token_dir.mkdir(parents=True)
    for name, data in _TOKEN_FILES.items():
        (token_dir / name).write_bytes(data)
    return {
        name: (p.stat().st_size, p.stat().st_mtime_ns)
        for name, p in ((n, token_dir / n) for n in _TOKEN_FILES)
    }


def _build_update_zip(zip_path: Path) -> None:
    """Build a realistic update archive (top-level ``hermes-agent-main/``).

    The archive carries its own ``mcp-tokens/`` entry — if the preserve set
    were missing, the two-phase replace would overwrite a legacy in-tree
    ``mcp-tokens/`` with this content.
    """
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("hermes-agent-main/README.md", "updated readme\n")
        zf.writestr("hermes-agent-main/agent/__init__.py", "new agent\n")
        zf.writestr("hermes-agent-main/mcp-tokens/notion.json", "NEW-TOKEN")


def _seed_install(root: Path) -> Path:
    """Seed an install tree with an old in-tree ``mcp-tokens/`` (legacy layout)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "agent").mkdir(exist_ok=True)
    (root / "agent" / "__init__.py").write_text("old agent\n")
    in_tree_tokens = root / "mcp-tokens"
    in_tree_tokens.mkdir(exist_ok=True)
    (in_tree_tokens / "notion.json").write_text("OLD-TOKEN")
    return in_tree_tokens


def _fake_urlretrieve(zip_path: Path):
    def fake(url, dest):
        shutil.copyfile(zip_path, dest)
        return dest, None

    return fake


# ---------------------------------------------------------------------------
# Preserve set
# ---------------------------------------------------------------------------


def test_zip_preserve_set_includes_mcp_tokens() -> None:
    """The ZIP-update preserve set must protect a legacy in-tree mcp-tokens/."""
    assert "mcp-tokens" in update_cmd._ZIP_UPDATE_PRESERVE
    assert {"venv", "node_modules", ".git", ".env"} <= update_cmd._ZIP_UPDATE_PRESERVE


# ---------------------------------------------------------------------------
# Post-update verifier unit tests
# ---------------------------------------------------------------------------


def test_verify_reports_preserved_and_leaves_files_byte_identical(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    hermes_home = tmp_path / "home"
    baseline = _seed_hermes_home_tokens(hermes_home)
    monkeypatch.setattr(update_cmd, "get_hermes_home", lambda: str(hermes_home))

    before = update_cmd._snapshot_mcp_tokens()
    assert before == baseline

    # Nothing touched the token dir → verifier reports preserved, no raise.
    update_cmd._verify_mcp_tokens_preserved(before)

    token_dir = hermes_home / "mcp-tokens"
    for name, data in _TOKEN_FILES.items():
        assert (token_dir / name).read_bytes() == data
    assert "preserved" in capsys.readouterr().out


def test_verify_warns_when_token_dir_vanished(tmp_path: Path, monkeypatch, caplog) -> None:
    """A token dir that disappears during update must surface a warning."""
    hermes_home = tmp_path / "home"
    _seed_hermes_home_tokens(hermes_home)
    monkeypatch.setattr(update_cmd, "get_hermes_home", lambda: str(hermes_home))

    before = update_cmd._snapshot_mcp_tokens()
    assert before

    shutil.rmtree(hermes_home / "mcp-tokens")
    with caplog.at_level(logging.WARNING, logger="hermes_cli.update_cmd"):
        update_cmd._verify_mcp_tokens_preserved(before)  # must not raise

    assert any("#84843" in r.message for r in caplog.records)


def test_verify_warns_when_token_file_replaced(tmp_path: Path, monkeypatch, caplog) -> None:
    """A token file whose content changed during update must surface a warning."""
    hermes_home = tmp_path / "home"
    _seed_hermes_home_tokens(hermes_home)
    monkeypatch.setattr(update_cmd, "get_hermes_home", lambda: str(hermes_home))

    before = update_cmd._snapshot_mcp_tokens()

    # Simulate an update that stomped the token file (same name, new bytes).
    (hermes_home / "mcp-tokens" / "notion.json").write_bytes(b"REPLACED")
    with caplog.at_level(logging.WARNING, logger="hermes_cli.update_cmd"):
        update_cmd._verify_mcp_tokens_preserved(before)

    assert any("#84843" in r.message and "changed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# End-to-end: _update_via_zip
# ---------------------------------------------------------------------------


def _run_zip_update(tmp_path: Path, monkeypatch, zip_path: Path, fake_root: Path,
                    hermes_home: Path, args) -> None:
    """Run the real ``_update_via_zip`` with only I/O-heavy steps stubbed."""
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", fake_root)
    monkeypatch.setattr(update_cmd, "get_hermes_home", lambda: str(hermes_home))

    def fake_urlretrieve(url, dest):
        shutil.copyfile(zip_path, dest)
        return dest, None

    with patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve), \
         patch("subprocess.run") as fake_run, \
         patch("subprocess.check_call"), \
         patch("hermes_cli.managed_uv.update_managed_uv"), \
         patch("hermes_cli.managed_uv.ensure_uv", return_value=None), \
         patch.object(update_cmd, "_ensure_uv_for_termux", return_value=None), \
         patch.object(hermes_main, "_install_python_dependencies_with_optional_fallback"), \
         patch.object(hermes_main, "_refresh_active_memory_provider_dependencies"), \
         patch.object(hermes_main, "_build_web_ui"), \
         patch.object(hermes_main, "_record_bytecode_fingerprint"), \
         patch.object(hermes_main, "_refresh_bootstrap_cache_scripts"), \
         patch.object(hermes_main, "_kill_stale_dashboard_processes",
                      return_value={"unrecovered": False}), \
         patch.object(update_cmd, "_update_node_dependencies", return_value=[]), \
         patch.object(update_cmd, "_validate_critical_modules_import",
                      return_value=(True, None, None)), \
         patch("tools.skills_sync.sync_skills", return_value={
             "copied": [], "updated": [], "user_modified": [], "cleaned": [],
             "relocated": [],
         }), \
         patch("hermes_cli.model_catalog.seed_cache_from_checkout", return_value=False):
        fake_run.return_value = type(
            "R", (), {"returncode": 0, "stdout": "", "stderr": ""}
        )()
        try:
            hermes_main._update_via_zip(args)
        except SystemExit:
            # _update_via_zip sys.exit(1)s on hard failures; a failure here
            # will show up as a failed assertion below (files not updated).
            pass


def test_update_via_zip_preserves_mcp_tokens_end_to_end(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """A real ZIP update leaves ``$HERMES_HOME/mcp-tokens/`` byte-identical.

    Covers the reported scenario: tokens persisted at HERMES_HOME/mcp-tokens
    before ``hermes update`` must survive the self-update unchanged. Also
    covers the defensive in-tree case (legacy HERMES_HOME == install root).
    """
    zip_path = tmp_path / "update.zip"
    _build_update_zip(zip_path)

    hermes_home = tmp_path / "hermes_home"
    baseline = _seed_hermes_home_tokens(hermes_home)

    fake_root = tmp_path / "install"
    in_tree_tokens = _seed_install(fake_root)

    args = type("Args", (), {})()
    _run_zip_update(tmp_path, monkeypatch, zip_path, fake_root, hermes_home, args)

    # The update actually ran: new code landed.
    assert (fake_root / "README.md").read_text() == "updated readme\n"
    assert (fake_root / "agent" / "__init__.py").read_text() == "new agent\n"

    # Canonical token dir ($HERMES_HOME/mcp-tokens): byte-identical + mtime
    # untouched — the "self-update with token → preserved" contract.
    token_dir = hermes_home / "mcp-tokens"
    for name, data in _TOKEN_FILES.items():
        p = token_dir / name
        assert p.read_bytes() == data, f"{name} changed during update"
        size, mtime_ns = baseline[name]
        assert p.stat().st_size == size
        assert p.stat().st_mtime_ns == mtime_ns

    # Legacy in-tree mcp-tokens/ (inside PROJECT_ROOT): also preserved, so
    # the ZIP's own mcp-tokens/ member never replaced it.
    assert (in_tree_tokens / "notion.json").read_text() == "OLD-TOKEN"

    out = capsys.readouterr().out
    assert "MCP OAuth token file(s) preserved" in out


def test_update_via_zip_without_tokens_is_normal(tmp_path: Path, monkeypatch) -> None:
    """No persisted tokens → update completes normally, nothing to preserve."""
    zip_path = tmp_path / "update.zip"
    _build_update_zip(zip_path)

    hermes_home = tmp_path / "hermes_home"  # no mcp-tokens/ dir
    fake_root = tmp_path / "install"
    _seed_install(fake_root)

    args = type("Args", (), {})()
    _run_zip_update(tmp_path, monkeypatch, zip_path, fake_root, hermes_home, args)

    # Normal flow: new code landed, no token dir materialized, no error.
    assert (fake_root / "README.md").read_text() == "updated readme\n"
    assert not (hermes_home / "mcp-tokens").exists()


# ---------------------------------------------------------------------------
def test_verify_warns_when_token_file_same_size_different_mtime(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    """A token file rewritten with identical length but rotated mtime must surface a warning (#84843)."""
    hermes_home = tmp_path / "home"
    _seed_hermes_home_tokens(hermes_home)
    monkeypatch.setattr(update_cmd, "get_hermes_home", lambda: str(hermes_home))

    before = update_cmd._snapshot_mcp_tokens()
    target = hermes_home / "mcp-tokens" / "notion.json"
    old_stat = target.stat()

    # Rewrite with same byte length but advanced mtime
    target.write_bytes(b"x" * old_stat.st_size)
    os.utime(target, (old_stat.st_atime, old_stat.st_mtime + 10.0))

    with caplog.at_level(logging.WARNING, logger="hermes_cli.update_cmd"):
        update_cmd._verify_mcp_tokens_preserved(before)

    assert any("#84843" in r.message and "changed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Wiring contract
# ---------------------------------------------------------------------------


def test_cmd_update_impl_wires_mcp_token_preservation(tmp_path: Path, monkeypatch) -> None:
    """Behavioral contract: the update pipeline snapshots tokens pre-update and
    forwards the baseline to _update_via_zip (#84843)."""
    fake_baseline = {"notion.json": (100, 200)}
    snapshot_called = []
    zip_called_with = []

    monkeypatch.setattr(
        update_cmd,
        "_snapshot_mcp_tokens",
        lambda: snapshot_called.append(True) or fake_baseline,
    )
    monkeypatch.setattr(
        update_cmd,
        "_update_via_zip",
        lambda args, mcp_tokens_before=None: zip_called_with.append(mcp_tokens_before),
    )

    fake_root = tmp_path / "install"
    fake_root.mkdir()
    # No .git directory so on win32 it branches to use_zip_update
    monkeypatch.setattr(update_cmd._m(), "PROJECT_ROOT", fake_root)
    monkeypatch.setattr(update_cmd._m(), "_run_pre_update_backup", lambda args: None)
    monkeypatch.setattr(update_cmd._m(), "_pause_windows_gateways_for_update", lambda: None)
    monkeypatch.setattr(update_cmd._m(), "_resume_windows_gateways_after_update", lambda r: None)
    monkeypatch.setattr(update_cmd._m(), "_is_windows", lambda: False)
    monkeypatch.setattr(sys, "platform", "win32")

    args = type("Args", (), {"force_venv": True})()
    update_cmd._cmd_update_impl(args, gateway_mode=False)

    assert snapshot_called == [True], "pre-update snapshot must be captured"
    assert zip_called_with == [fake_baseline], "baseline must be forwarded to _update_via_zip"

