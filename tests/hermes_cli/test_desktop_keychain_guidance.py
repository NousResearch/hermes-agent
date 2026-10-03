"""Keychain guidance after a local macOS re-sign (#91115).

macOS stamps an app-created keychain item's ACL partition from the signature:
``teamid:<ID>`` for an Apple-issued Team ID, ``cdhash:<hash>`` otherwise (ad-hoc
and self-signed identities carry none). So for users who opted in to keychain
encryption, every no-Team-ID rebuild gets one keychain prompt — the updater
cannot fix that headlessly, but it can say so accurately and name the remedy
(an Apple-issued 'Apple Development' / 'Developer ID Application' identity).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import main_desktop


# --- Team ID parse -----------------------------------------------------------


def _codesign_result(stdout: str = "", stderr: str = "", returncode: int = 0):
    return subprocess.CompletedProcess(["codesign", "-dv"], returncode, stdout, stderr)


def test_team_id_present_is_parsed_from_codesign_stderr(monkeypatch, tmp_path):
    out = "Executable=...\nTeamIdentifier=ABCDE12345\nIdentifier=com.nousresearch.hermes\n"
    monkeypatch.setattr(
        main_desktop.subprocess, "run", lambda *a, **kw: _codesign_result(stderr=out))
    assert main_desktop._macos_signed_team_id("/usr/bin/codesign", tmp_path) == "ABCDE12345"


def test_team_id_not_set_means_none(monkeypatch, tmp_path):
    out = "TeamIdentifier=not set\n"
    monkeypatch.setattr(
        main_desktop.subprocess, "run", lambda *a, **kw: _codesign_result(stderr=out))
    assert main_desktop._macos_signed_team_id("/usr/bin/codesign", tmp_path) is None


def test_team_id_missing_line_means_none(monkeypatch, tmp_path):
    monkeypatch.setattr(
        main_desktop.subprocess, "run", lambda *a, **kw: _codesign_result(stderr="Identifier=x\n"))
    assert main_desktop._macos_signed_team_id("/usr/bin/codesign", tmp_path) is None


def test_codesign_failure_is_unknown_not_a_team_id(monkeypatch, tmp_path):
    """A failed codesign probe must yield None — never a false 'has Team ID' claim."""
    monkeypatch.setattr(
        main_desktop.subprocess, "run", lambda *a, **kw: _codesign_result(returncode=1, stderr="boom"))
    assert main_desktop._macos_signed_team_id("/usr/bin/codesign", tmp_path) is None


def test_codesign_exception_is_unknown_not_a_team_id(monkeypatch, tmp_path):
    def boom(*a, **kw):
        raise OSError("no codesign")
    monkeypatch.setattr(main_desktop.subprocess, "run", boom)
    assert main_desktop._macos_signed_team_id("/usr/bin/codesign", tmp_path) is None


# --- opt-in reader -----------------------------------------------------------


def _policy_dir(tmp_path: Path) -> Path:
    d = tmp_path / "user-data"
    d.mkdir()
    return d


def test_opted_in_true(tmp_path):
    d = _policy_dir(tmp_path)
    (d / main_desktop.SECRET_STORAGE_POLICY_FILE).write_text('{"on": true}', encoding="utf-8")
    assert main_desktop._desktop_secret_storage_opted_in(d) is True


def test_opted_in_false(tmp_path):
    d = _policy_dir(tmp_path)
    (d / main_desktop.SECRET_STORAGE_POLICY_FILE).write_text('{"on": false}', encoding="utf-8")
    assert main_desktop._desktop_secret_storage_opted_in(d) is False


def test_missing_policy_file_is_false(tmp_path):
    assert main_desktop._desktop_secret_storage_opted_in(_policy_dir(tmp_path)) is False


def test_corrupt_policy_file_is_false(tmp_path):
    d = _policy_dir(tmp_path)
    (d / main_desktop.SECRET_STORAGE_POLICY_FILE).write_text("{not json", encoding="utf-8")
    assert main_desktop._desktop_secret_storage_opted_in(d) is False


def test_truthy_but_not_true_on_is_false(tmp_path):
    """Mirrors the TS coercion rule: only strict ``on === true`` enables prompts."""
    d = _policy_dir(tmp_path)
    (d / main_desktop.SECRET_STORAGE_POLICY_FILE).write_text('{"on": "yes"}', encoding="utf-8")
    assert main_desktop._desktop_secret_storage_opted_in(d) is False


def test_bom_prefixed_policy_file_is_read(tmp_path):
    """Electron writes may carry a BOM; the reader must not choke into False."""
    d = _policy_dir(tmp_path)
    (d / main_desktop.SECRET_STORAGE_POLICY_FILE).write_bytes(b'\xef\xbb\xbf{"on": true}')
    assert main_desktop._desktop_secret_storage_opted_in(d) is True


# --- notice selection matrix ---------------------------------------------------


def test_opted_out_never_gets_a_keychain_notice():
    """Default users never touch safeStorage (#95015) — no prompt guidance for them."""
    assert main_desktop._macos_keychain_update_notice(
        opted_in=False, team_id=None, configured_identity=None) is None
    assert main_desktop._macos_keychain_update_notice(
        opted_in=False, team_id=None, configured_identity="Hermes Local Signing") is None
    # A Team ID is good news for everyone (logged once even when opted out).
    team_notice = main_desktop._macos_keychain_update_notice(
        opted_in=False, team_id="TEAM123", configured_identity=None)
    assert team_notice is not None and "prompt" not in team_notice


def test_team_id_logs_persistence_regardless_of_opt_in():
    for opted_in in (True, False):
        notice = main_desktop._macos_keychain_update_notice(
            opted_in=opted_in, team_id="ABCDE12345", configured_identity="whatever")
        assert notice is not None
        assert "Team ID ABCDE12345" in notice
        assert "persist across updates" in notice


def test_opted_in_no_team_id_adhoc_notice():
    notice = main_desktop._macos_keychain_update_notice(
        opted_in=True, team_id=None, configured_identity=None)
    assert notice is not None
    assert "Always Allow" in notice
    assert "never delete the item" in notice
    assert "desktop.macos_signing_identity" in notice
    assert "Apple Development" in notice and "Developer ID Application" in notice
    # Plain guidance, not a warning about a broken state.
    assert "will prompt" in notice


def test_opted_in_no_team_id_self_signed_identity_notice():
    """A configured self-signed identity preserves TCC but not the keychain item."""
    notice = main_desktop._macos_keychain_update_notice(
        opted_in=True, team_id=None, configured_identity="Hermes Local Signing")
    assert notice is not None
    assert "Hermes Local Signing" in notice
    assert "TCC grants" in notice
    assert "Always Allow" in notice
    assert "desktop.macos_signing_identity" in notice
    assert "Apple Development" in notice


def test_team_id_wins_over_self_signed_wording():
    """A Team-ID identity never gets the 'still prompts' wording."""
    notice = main_desktop._macos_keychain_update_notice(
        opted_in=True, team_id="ABCDE12345", configured_identity="Apple Development: Me (ABC)")
    assert notice is not None
    assert "Team ID ABCDE12345" in notice
    assert "Always Allow" not in notice


# --- guidance wiring (fail-safe, never raises) -------------------------------


@pytest.mark.platforms("macos")
def test_keychain_guidance_prints_adhoc_notice_for_opted_in_user(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("HERMES_DESKTOP_USER_DATA_DIR", str(tmp_path / "user-data"))
    ud = tmp_path / "user-data"
    ud.mkdir()
    (ud / main_desktop.SECRET_STORAGE_POLICY_FILE).write_text('{"on": true}', encoding="utf-8")
    monkeypatch.setattr(
        main_desktop.subprocess, "run",
        lambda *a, **kw: _codesign_result(stderr="TeamIdentifier=not set\n"))
    main_desktop._desktop_macos_keychain_guidance("/usr/bin/codesign", tmp_path, None)
    out = capsys.readouterr().out
    assert "Always Allow" in out
    assert "desktop.macos_signing_identity" in out


@pytest.mark.platforms("macos")
def test_keychain_guidance_prints_team_id_for_signed_build(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("HERMES_DESKTOP_USER_DATA_DIR", str(tmp_path / "user-data"))
    monkeypatch.setattr(
        main_desktop.subprocess, "run",
        lambda *a, **kw: _codesign_result(stderr="TeamIdentifier=ABCDE12345\n"))
    main_desktop._desktop_macos_keychain_guidance("/usr/bin/codesign", tmp_path, None)
    out = capsys.readouterr().out
    assert "Team ID ABCDE12345" in out
    assert "Always Allow" not in out


@pytest.mark.platforms("macos")
def test_keychain_guidance_silent_for_opted_out_and_adhoc(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("HERMES_DESKTOP_USER_DATA_DIR", str(tmp_path / "user-data"))  # no file
    monkeypatch.setattr(
        main_desktop.subprocess, "run",
        lambda *a, **kw: _codesign_result(stderr="TeamIdentifier=not set\n"))
    main_desktop._desktop_macos_keychain_guidance("/usr/bin/codesign", tmp_path, None)
    assert capsys.readouterr().out == ""


@pytest.mark.platforms("macos")
def test_keychain_guidance_never_raises_on_failing_probes(monkeypatch, tmp_path, capsys):
    def boom(*a, **kw):
        raise OSError("broken")
    monkeypatch.setattr(main_desktop.subprocess, "run", boom)
    monkeypatch.setattr(main_desktop, "_desktop_secret_storage_opted_in", boom)
    main_desktop._desktop_macos_keychain_guidance("/usr/bin/codesign", tmp_path, None)
    assert capsys.readouterr().out == ""
