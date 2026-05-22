"""Tests for the bare-hermes active_profile fallback warning.

When bare ``hermes`` (no -p/--profile) is invoked and ``~/.hermes/active_profile``
points to a non-default profile, ``_apply_profile_override()`` silently sets
HERMES_HOME to that profile. This trapdoor caused three profile-hijack
incidents in 24h where launchd-spawned processes ended up bound to Ruta when
the operator expected default. The fix: emit a loud one-shot stderr warning
at the moment the trapdoor fires.

The warning must:
  1. Fire ONLY on the active_profile fallback path (not -p/--profile).
  2. Skip the no-op case (resolved profile == 'default').
  3. Be one-shot per process.
  4. Skip when HERMES_HOME was already pinned in env (plists do this).
  5. Be suppressible via HERMES_QUIET_BARE_WARNING=1.
"""

from pathlib import Path

import pytest


@pytest.fixture
def fresh_main(monkeypatch, tmp_path):
    """Reset the one-shot warn flag and point HOME at a tmpdir.

    We import hermes_cli.main and reset the module-level guard so each
    test starts from a clean slate. We avoid reloading the module
    (which would re-run the top-level _apply_profile_override() call).
    """
    import hermes_cli.main as main_mod
    monkeypatch.setattr(main_mod, "_BARE_HERMES_WARNING_EMITTED", False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.delenv("HERMES_QUIET_BARE_WARNING", raising=False)
    return main_mod


def _seed_profile(tmp_path: Path, active: str) -> None:
    """Create ~/.hermes/active_profile and a matching profile dir."""
    hermes_dir = tmp_path / ".hermes"
    hermes_dir.mkdir(exist_ok=True)
    (hermes_dir / "active_profile").write_text(active + "\n")
    if active and active != "default":
        (hermes_dir / "profiles" / active).mkdir(parents=True, exist_ok=True)


class TestBareHermesWarning:
    def test_explicit_profile_flag_no_warning(
        self, fresh_main, tmp_path, monkeypatch, capsys
    ):
        """-p ruta on the command line → user knows, no warning."""
        _seed_profile(tmp_path, "ruta")
        monkeypatch.setattr("sys.argv", ["hermes", "-p", "ruta", "chat"])

        fresh_main._apply_profile_override()

        err = capsys.readouterr().err
        assert "No --profile flag" not in err

    def test_active_profile_default_no_warning(
        self, fresh_main, tmp_path, monkeypatch, capsys
    ):
        """active_profile=default → no-op, no warning."""
        _seed_profile(tmp_path, "default")
        monkeypatch.setattr("sys.argv", ["hermes"])

        fresh_main._apply_profile_override()

        err = capsys.readouterr().err
        assert "No --profile flag" not in err

    def test_active_profile_non_default_warns(
        self, fresh_main, tmp_path, monkeypatch, capsys
    ):
        """Bare hermes + active_profile=ruta → loud stderr warning."""
        _seed_profile(tmp_path, "ruta")
        monkeypatch.setattr("sys.argv", ["hermes"])

        fresh_main._apply_profile_override()

        err = capsys.readouterr().err
        assert "No --profile flag" in err
        assert "active_profile → ruta" in err
        assert "Set --profile ruta explicitly" in err
        assert "hermes profile use default" in err

    def test_hermes_home_already_set_no_warning(
        self, fresh_main, tmp_path, monkeypatch, capsys
    ):
        """HERMES_HOME explicitly set in env → caller pinned it, no warning.

        This is the launchd plist case: the .plist sets HERMES_HOME, so
        even if active_profile says something else, the caller is loud.
        We point HERMES_HOME at the hermes root (NOT a profile dir) so
        step 1.5's early-return doesn't fire and step 2 still runs.
        """
        _seed_profile(tmp_path, "ruta")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        monkeypatch.setattr("sys.argv", ["hermes"])

        fresh_main._apply_profile_override()

        err = capsys.readouterr().err
        assert "No --profile flag" not in err

    def test_warning_is_one_shot(
        self, fresh_main, tmp_path, monkeypatch, capsys
    ):
        """Two fallback firings in the same process → warning appears once."""
        _seed_profile(tmp_path, "ruta")
        monkeypatch.setattr("sys.argv", ["hermes"])

        fresh_main._apply_profile_override()
        # Clear HERMES_HOME so the second call also goes through step 2 →
        # step 3 fallback path. Without this, step 1.5 returns early.
        monkeypatch.delenv("HERMES_HOME", raising=False)
        fresh_main._apply_profile_override()

        err = capsys.readouterr().err
        assert err.count("No --profile flag") == 1

    def test_quiet_env_var_suppresses(
        self, fresh_main, tmp_path, monkeypatch, capsys
    ):
        """HERMES_QUIET_BARE_WARNING=1 → silent."""
        _seed_profile(tmp_path, "ruta")
        monkeypatch.setenv("HERMES_QUIET_BARE_WARNING", "1")
        monkeypatch.setattr("sys.argv", ["hermes"])

        fresh_main._apply_profile_override()

        err = capsys.readouterr().err
        assert "No --profile flag" not in err
