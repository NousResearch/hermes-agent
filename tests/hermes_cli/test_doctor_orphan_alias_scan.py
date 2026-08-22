"""Tests for bounded orphan profile alias scanning in Doctor.

Doctor reports profile aliases whose target profile no longer exists. The
wrapper directory is normally ``~/.local/bin``, which also holds unrelated
binaries, so the scan must never read an arbitrary entry in full.

These tests drive the real ``run_doctor`` code path and assert on its output
rather than calling the scan helper directly, so they describe behaviour
instead of implementation.
"""

import sys
import types
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import doctor
from hermes_cli import profiles


@pytest.fixture()
def alias_scan_env(tmp_path, monkeypatch):
    """Point profile and wrapper discovery at isolated test directories."""
    wrapper_dir = tmp_path / "bin"
    profiles_root = tmp_path / "profiles"
    wrapper_dir.mkdir()
    profiles_root.mkdir()
    monkeypatch.setattr(profiles, "_get_wrapper_dir", lambda: wrapper_dir)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: profiles_root)
    return wrapper_dir, profiles_root


def _write_wrapper(wrapper_dir, alias, profile):
    (wrapper_dir / alias).write_text(
        f'#!/bin/sh\nexec hermes -p {profile} "$@"\n',
        encoding="utf-8",
    )


def _run_doctor(monkeypatch, tmp_path, profiles_root):
    """Run Doctor with external checks stubbed out, and return nothing.

    Only the profile-alias section matters here; everything else is neutered
    so the run stays offline and deterministic.
    """
    hermes_home = tmp_path / ".hermes"
    project_root = tmp_path / "project"
    hermes_home.mkdir(exist_ok=True)
    project_root.mkdir(exist_ok=True)
    (hermes_home / ".env").write_text("", encoding="utf-8")
    (hermes_home / "config.yaml").write_text("memory: {}\n", encoding="utf-8")

    live_profile = profiles_root / "live-profile"
    live_profile.mkdir(exist_ok=True)
    monkeypatch.setattr(doctor, "HERMES_HOME", hermes_home)
    monkeypatch.setattr(doctor, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(doctor, "_DHH", str(hermes_home))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(doctor.shutil, "which", lambda _command: None)
    monkeypatch.setattr(doctor, "_APIKEY_PROVIDERS_CACHE", [])

    for key in doctor._PROVIDER_ENV_HINTS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)

    fake_model_tools = types.SimpleNamespace(
        check_tool_availability=lambda *args, **kwargs: ([], []),
        TOOLSET_REQUIREMENTS={},
    )
    monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)

    from agent import bedrock_adapter
    from hermes_cli import auth

    monkeypatch.setattr(bedrock_adapter, "has_aws_credentials", lambda: False)
    monkeypatch.setattr(auth, "get_anthropic_key", lambda: "")
    monkeypatch.setattr(auth, "get_nous_auth_status_local", lambda: {})
    monkeypatch.setattr(auth, "get_codex_auth_status", lambda: {})
    monkeypatch.setattr(auth, "get_minimax_oauth_auth_status", lambda: {})
    monkeypatch.setattr(auth, "get_xai_oauth_auth_status", lambda: {})
    monkeypatch.setattr(
        profiles,
        "list_profiles",
        lambda: [
            SimpleNamespace(
                name="live-profile",
                path=live_profile,
                is_default=False,
                gateway_running=False,
                model=None,
            )
        ],
    )

    doctor.run_doctor(Namespace(fix=False))


def test_orphan_alias_is_reported(alias_scan_env, monkeypatch, tmp_path, capsys):
    """A wrapper naming a missing profile is still reported."""
    wrapper_dir, profiles_root = alias_scan_env
    _write_wrapper(wrapper_dir, "orphan-alias", "missing-profile")

    _run_doctor(monkeypatch, tmp_path, profiles_root)

    output = capsys.readouterr().out
    assert "Orphan alias: orphan-alias" in output
    assert "profile 'missing-profile' no longer exists" in output


def test_live_alias_is_not_reported(alias_scan_env, monkeypatch, tmp_path, capsys):
    """A wrapper naming an existing profile is not an orphan."""
    wrapper_dir, profiles_root = alias_scan_env
    (profiles_root / "present-profile").mkdir()
    _write_wrapper(wrapper_dir, "present-alias", "present-profile")

    _run_doctor(monkeypatch, tmp_path, profiles_root)

    assert "Orphan alias: present-alias" not in capsys.readouterr().out


def test_profile_marker_after_read_limit_is_ignored(
    alias_scan_env, monkeypatch, tmp_path, capsys
):
    """Content past the wrapper read limit must never be scanned.

    This is the regression test for the unbounded read. The marker sits
    beyond ``_WRAPPER_READ_LIMIT``, so a bounded scanner cannot see it. The
    previous implementation read the whole entry and did report it.
    """
    wrapper_dir, profiles_root = alias_scan_env
    (wrapper_dir / "oversized-entry").write_text(
        "x" * profiles._WRAPPER_READ_LIMIT + "\nhermes -p late-profile\n",
        encoding="utf-8",
    )

    _run_doctor(monkeypatch, tmp_path, profiles_root)

    assert "Orphan alias: oversized-entry" not in capsys.readouterr().out


def test_wrapper_dir_entries_are_never_read_whole(
    alias_scan_env, monkeypatch, tmp_path, capsys
):
    """Doctor must not call the unbounded whole-file read on any entry.

    ``Path.read_text()`` loads an entire file into memory. Applied to every
    entry in a directory that also holds large binaries, that is what
    exhausted memory on a small host, so the scan must not use it at all.
    """
    wrapper_dir, profiles_root = alias_scan_env
    _write_wrapper(wrapper_dir, "orphan-alias", "missing-profile")

    scanned = wrapper_dir.resolve()
    original_read_text = Path.read_text

    def _guarded_read_text(self, *args, **kwargs):
        if self.resolve().parent == scanned:
            raise AssertionError(
                f"unbounded read_text() on wrapper dir entry: {self.name}"
            )
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _guarded_read_text)

    _run_doctor(monkeypatch, tmp_path, profiles_root)

    assert "Orphan alias: orphan-alias" in capsys.readouterr().out


def test_binary_entry_is_skipped(alias_scan_env, monkeypatch, tmp_path, capsys):
    """A binary on the wrapper path is skipped without raising."""
    wrapper_dir, profiles_root = alias_scan_env
    (wrapper_dir / "binary-entry").write_bytes(
        b"\xff\xfe\x00hermes -p missing-profile\n"
    )

    _run_doctor(monkeypatch, tmp_path, profiles_root)

    assert "Orphan alias: binary-entry" not in capsys.readouterr().out


def test_every_dangling_wrapper_is_reported(
    alias_scan_env, monkeypatch, tmp_path, capsys
):
    """Each dangling wrapper gets its own warning, not one per profile.

    Adding a custom alias leaves both the profile-named wrapper and the alias
    wrapper on disk, so removing the profile strands two files. Reporting only
    one of them hides the other until the user re-runs Doctor.
    """
    wrapper_dir, profiles_root = alias_scan_env
    _write_wrapper(wrapper_dir, "work", "work")
    _write_wrapper(wrapper_dir, "w", "work")

    _run_doctor(monkeypatch, tmp_path, profiles_root)

    output = capsys.readouterr().out
    reported = [line for line in output.splitlines() if "Orphan alias:" in line]
    assert len(reported) == 2, reported
    assert "Orphan alias: w → profile 'work' no longer exists" in output
    assert "Orphan alias: work → profile 'work' no longer exists" in output


def test_mixed_case_target_reports_the_canonical_profile_id(
    alias_scan_env, monkeypatch, tmp_path, capsys
):
    """A hand-edited target is reported under the id profiles use on disk.

    Profile directories are lowercase, so the canonical id is the one the user
    would type to inspect or recreate the profile.
    """
    wrapper_dir, profiles_root = alias_scan_env
    _write_wrapper(wrapper_dir, "cased", "Ghost")

    _run_doctor(monkeypatch, tmp_path, profiles_root)

    assert (
        "Orphan alias: cased → profile 'ghost' no longer exists"
        in capsys.readouterr().out
    )


def test_case_never_decides_whether_a_profile_is_missing(
    alias_scan_env, monkeypatch, tmp_path, capsys
):
    """A mixed-case target resolving to a live profile is not an orphan."""
    wrapper_dir, profiles_root = alias_scan_env
    (profiles_root / "present-profile").mkdir()
    _write_wrapper(wrapper_dir, "cased-live", "Present-Profile")

    _run_doctor(monkeypatch, tmp_path, profiles_root)

    assert "Orphan alias: cased-live" not in capsys.readouterr().out
