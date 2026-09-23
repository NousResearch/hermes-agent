"""Tests for get_hermes_home() profile-mode fallback warning.

Regression test for https://github.com/NousResearch/hermes-agent/issues/18594.

When HERMES_HOME is unset but an active_profile file indicates a non-default
profile is active, get_hermes_home() should:
  1. STILL return ~/.hermes (raising would brick 30+ module-level callers)
  2. Emit a loud one-shot warning to stderr so operators can diagnose
     cross-profile data contamination after the fact.

The warning goes to stderr directly (not through logging) because this
function is called at module-import time from 30+ sites, often before the
logging subsystem has been configured.
"""

from pathlib import Path

import pytest


@pytest.fixture
def fresh_constants(monkeypatch, tmp_path):
    """Import hermes_constants fresh and reset the one-shot warn flag."""
    import importlib
    import hermes_constants
    importlib.reload(hermes_constants)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    return hermes_constants


class TestGetHermesHomeProfileWarning:
    def test_classic_mode_no_active_profile_no_warning(
        self, fresh_constants, tmp_path, capsys
    ):
        """Classic mode: no active_profile file → silent, returns ~/.hermes."""
        result = fresh_constants.get_hermes_home()
        assert result == tmp_path / ".hermes"
        assert "HERMES_HOME fallback" not in capsys.readouterr().err


    def test_named_profile_unset_home_warns_once(
        self, fresh_constants, tmp_path, capsys
    ):
        """active_profile=coder + HERMES_HOME unset → warn loudly, still return fallback."""
        hermes_dir = tmp_path / ".hermes"
        hermes_dir.mkdir()
        (hermes_dir / "active_profile").write_text("coder\n")

        result = fresh_constants.get_hermes_home()

        # 1. Still returns the fallback — no import-time crash
        assert result == tmp_path / ".hermes"
        # 2. Stderr got the warning exactly once
        err = capsys.readouterr().err
        assert err.count("HERMES_HOME fallback") == 1
        assert "'coder'" in err

        # 3. One-shot: second and third calls don't re-warn
        fresh_constants.get_hermes_home()
        fresh_constants.get_hermes_home()
        err2 = capsys.readouterr().err
        assert "HERMES_HOME fallback" not in err2

    def test_hermes_home_set_suppresses_warning(
        self, fresh_constants, tmp_path, capsys, monkeypatch
    ):
        """Even if active_profile is 'coder', setting HERMES_HOME suppresses warning."""
        profile_dir = tmp_path / ".hermes" / "profiles" / "coder"
        profile_dir.mkdir(parents=True)
        (tmp_path / ".hermes" / "active_profile").write_text("coder\n")
        monkeypatch.setenv("HERMES_HOME", str(profile_dir))

        result = fresh_constants.get_hermes_home()

        assert result == profile_dir
        assert "HERMES_HOME fallback" not in capsys.readouterr().err

    def test_unreadable_active_profile_no_crash(
        self, fresh_constants, tmp_path, capsys
    ):
        """active_profile that can't be decoded → fall through silently."""
        hermes_dir = tmp_path / ".hermes"
        hermes_dir.mkdir()
        # Write bytes that aren't valid utf-8
        (hermes_dir / "active_profile").write_bytes(b"\xff\xfe\x00\x00")

        result = fresh_constants.get_hermes_home()

        assert result == tmp_path / ".hermes"
        # Shouldn't crash; shouldn't warn either (can't tell what profile was intended)
        assert "HERMES_HOME fallback" not in capsys.readouterr().err


class TestBootReadersBeforeProfileOverride:
    """Readers that run before the CLI applies the sticky ``active_profile`` must not warn.

    ``hermes_bootstrap`` points ``TMPDIR`` at the scratch dir of the *process* home during
    import, and ``main._apply_profile_override`` re-homes the process a few lines later. A
    caller that already resolved its home must not send the policy back through
    ``get_hermes_home()``: for a sticky-profile user with ``HERMES_HOME`` unset in a plain
    shell that lookup falls back to the default profile and warns, on every ``hermes``
    command, while nothing lands in the wrong place. Same fix the parser's ``_cfg_path()``
    carries for the ``--no-config`` help string.
    """

    def test_scratch_export_uses_the_process_home_silently(
        self, fresh_constants, tmp_path, capsys
    ):
        """Boot scratch setup: silent, and pointed at the home the bootstrap resolved."""
        hermes_dir = tmp_path / ".hermes"
        (hermes_dir / "profiles" / "coder").mkdir(parents=True)
        (hermes_dir / "active_profile").write_text("coder\n")
        capsys.readouterr()  # drop anything the setup above printed

        env = {"PATH": "/usr/bin:/bin"}
        assert fresh_constants.apply_scratch_tmp_env(env) is True

        assert env["TMPDIR"] == str(hermes_dir / "cache" / "scratch")
        assert "HERMES_HOME fallback" not in capsys.readouterr().err

    def test_explicit_home_scratch_dir_never_reads_the_effective_home(
        self, fresh_constants, tmp_path, capsys
    ):
        """A caller that passes a home (`hermes doctor` for another profile) stays silent."""
        hermes_dir = tmp_path / ".hermes"
        profile_dir = hermes_dir / "profiles" / "coder"
        profile_dir.mkdir(parents=True)
        (hermes_dir / "active_profile").write_text("coder\n")
        capsys.readouterr()  # drop anything the setup above printed

        scratch = fresh_constants.get_scratch_dir(profile_dir)

        assert scratch == profile_dir / "cache" / "scratch"
        assert scratch.is_dir()
        assert "HERMES_HOME fallback" not in capsys.readouterr().err


class TestScratchTmpEnvGitBash:
    """Regression for https://github.com/NousResearch/hermes-agent/issues/120323.

    Git Bash on Windows pre-sets ``TMPDIR``/``TMP``/``TEMP`` to ``/tmp`` (the MSYS
    virtual path) before Hermes starts, which made ``apply_scratch_tmp_env`` return
    False and leave the scratch dir unexported — ``$TMPDIR`` in a ``terminal`` call
    pointed at the MSYS temp, the system prompt's "Scratch directory: ... TMPDIR
    points here" line was wrong, and ``HERMES_SCRATCH_DIR`` stayed unset. The fix
    recognises the MSYS pseudo temp under ``MSYSTEM`` as not a user choice; every
    other case (genuine Linux ``/tmp``, real Windows ``%TEMP%``, ``/var/folders/...``)
    is still respected.
    """

    def test_msystem_with_pseudo_tmp_exports_scratch(
        self, fresh_constants, tmp_path
    ):
        """Git Bash on Windows: MSYSTEM=MINGW64 + TMPDIR=/tmp + TMP=/tmp + TEMP=/tmp
        must re-export the scratch dir into all three and set HERMES_SCRATCH_DIR."""
        hermes_dir = tmp_path / ".hermes"
        (hermes_dir / "profiles" / "coder").mkdir(parents=True)
        (hermes_dir / "active_profile").write_text("coder\n")

        env = {
            "PATH": "/usr/bin:/bin",
            "MSYSTEM": "MINGW64",
            "TMPDIR": "/tmp",
            "TMP": "/tmp",
            "TEMP": "/tmp",
        }
        assert fresh_constants.apply_scratch_tmp_env(env) is True

        expected = str(hermes_dir / "cache" / "scratch")
        assert env["TMPDIR"] == expected
        assert env["TMP"] == expected
        assert env["TEMP"] == expected
        assert env["HERMES_SCRATCH_DIR"] == expected

    def test_linux_tmp_without_msystem_is_respected(
        self, fresh_constants, tmp_path
    ):
        """Real Linux shell: user-set TMPDIR=/tmp with no MSYSTEM stays a user
        choice. ``apply_scratch_tmp_env`` must NOT overwrite it, and the marker
        env var must remain unset so the model still reads the genuine value."""
        hermes_dir = tmp_path / ".hermes"
        (hermes_dir / "profiles" / "coder").mkdir(parents=True)
        (hermes_dir / "active_profile").write_text("coder\n")

        env = {
            "PATH": "/usr/bin:/bin",
            "TMPDIR": "/tmp",
            "TMP": "/tmp",
            "TEMP": "/tmp",
        }
        assert fresh_constants.apply_scratch_tmp_env(env) is False

        assert env["TMPDIR"] == "/tmp"
        assert env["TMP"] == "/tmp"
        assert env["TEMP"] == "/tmp"
        assert "HERMES_SCRATCH_DIR" not in env

    def test_real_windows_temp_under_msystem_is_respected(
        self, fresh_constants, tmp_path
    ):
        """Real Windows %TEMP% (C:\\Users\\<u>\\AppData\\Local\\Temp) under MSYSTEM
        is a deliberate override, not the pseudo default — leave it alone."""
        hermes_dir = tmp_path / ".hermes"
        (hermes_dir / "profiles" / "coder").mkdir(parents=True)
        (hermes_dir / "active_profile").write_text("coder\n")

        real_temp = "C:\\Users\\alice\\AppData\\Local\\Temp"
        env = {
            "PATH": "/usr/bin:/bin",
            "MSYSTEM": "MINGW64",
            "TMPDIR": real_temp,
            "TMP": real_temp,
            "TEMP": real_temp,
        }
        assert fresh_constants.apply_scratch_tmp_env(env) is False

        assert env["TMPDIR"] == real_temp
        assert env["TMP"] == real_temp
        assert env["TEMP"] == real_temp
        assert "HERMES_SCRATCH_DIR" not in env

    def test_msystem_pseudo_with_stale_marker_re_derives(
        self, fresh_constants, tmp_path
    ):
        """Even with HERMES_SCRATCH_DIR already set to an old scratch path, a
        pseudo ``/tmp`` under MSYSTEM must still trigger the re-export, because
        the marker check only short-circuits the user-value guard, not the
        pseudo-value guard."""
        hermes_dir = tmp_path / ".hermes"
        (hermes_dir / "profiles" / "coder").mkdir(parents=True)
        (hermes_dir / "active_profile").write_text("coder\n")
        stale_marker = str(hermes_dir / "profiles" / "old" / "cache" / "scratch")

        env = {
            "PATH": "/usr/bin:/bin",
            "MSYSTEM": "MINGW64",
            "TMPDIR": "/tmp",
            "TMP": "/tmp",
            "TEMP": "/tmp",
            "HERMES_SCRATCH_DIR": stale_marker,
        }
        assert fresh_constants.apply_scratch_tmp_env(env) is True

        expected = str(hermes_dir / "cache" / "scratch")
        assert env["TMPDIR"] == expected
        assert env["HERMES_SCRATCH_DIR"] == expected
        assert env["HERMES_SCRATCH_DIR"] != stale_marker

    def test_msystem_pseudo_with_trailing_slash_still_exports(
        self, fresh_constants, tmp_path
    ):
        """A MSYS-style ``/tmp/`` (trailing slash) is still the pseudo default —
        the helper normalises trailing slashes before comparing."""
        hermes_dir = tmp_path / ".hermes"
        (hermes_dir / "profiles" / "coder").mkdir(parents=True)
        (hermes_dir / "active_profile").write_text("coder\n")

        env = {
            "PATH": "/usr/bin:/bin",
            "MSYSTEM": "MINGW64",
            "TMPDIR": "/tmp/",
            "TMP": "/tmp/",
            "TEMP": "/tmp/",
        }
        assert fresh_constants.apply_scratch_tmp_env(env) is True

        expected = str(hermes_dir / "cache" / "scratch")
        assert env["TMPDIR"] == expected
        assert env["HERMES_SCRATCH_DIR"] == expected

    def test_ostype_msys_without_msystem_still_exports(self, fresh_constants, tmp_path):
        """Defensive OSTYPE signal: a standalone MSYS2 shell may set OSTYPE=msys
        but not MSYSTEM — the helper should still recognise ``/tmp`` as pseudo."""
        hermes_dir = tmp_path / ".hermes"
        (hermes_dir / "profiles" / "coder").mkdir(parents=True)
        (hermes_dir / "active_profile").write_text("coder\n")

        env = {
            "PATH": "/usr/bin:/bin",
            "OSTYPE": "msys",
            "TMPDIR": "/tmp",
        }
        assert fresh_constants.apply_scratch_tmp_env(env) is True
        assert env["TMPDIR"] == str(hermes_dir / "cache" / "scratch")

