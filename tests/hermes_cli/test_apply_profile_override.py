"""Regression tests for _apply_profile_override HERMES_HOME guard (issue #22502).

When HERMES_HOME is set to the hermes root (e.g. systemd hardcodes
HERMES_HOME=/root/.hermes), _apply_profile_override must still read
active_profile and update HERMES_HOME to the profile directory.

When HERMES_HOME is already a profile directory (.../profiles/<name>),
_apply_profile_override must trust it and return without re-reading
active_profile (child-process inheritance contract).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path



def _run_apply_profile_override(
    tmp_path, monkeypatch, *, hermes_home: str | None, active_profile: str | None,
    argv: list[str] | None = None,
):
    """Run _apply_profile_override in isolation.

    Returns the value of os.environ["HERMES_HOME"] after the call,
    or None if unset.
    """
    hermes_root = tmp_path / ".hermes"
    hermes_root.mkdir(parents=True, exist_ok=True)

    if active_profile is not None:
        (hermes_root / "active_profile").write_text(active_profile)

    if active_profile and active_profile != "default":
        (hermes_root / "profiles" / active_profile).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    if hermes_home is not None:
        monkeypatch.setenv("HERMES_HOME", hermes_home)
    else:
        monkeypatch.delenv("HERMES_HOME", raising=False)

    monkeypatch.setattr(sys, "argv", argv or ["hermes", "gateway", "start"])

    from hermes_cli.main import _apply_profile_override
    _apply_profile_override()

    return os.environ.get("HERMES_HOME")


class TestApplyProfileOverrideHermesHomeGuard:
    """Regression guard for issue #22502.

    Verifies that HERMES_HOME pointing to the hermes root does NOT suppress
    the active_profile check, while HERMES_HOME already pointing to a
    profile directory IS trusted as-is.
    """

    def test_hermes_home_at_root_with_active_profile_is_redirected(
        self, tmp_path, monkeypatch
    ):
        """HERMES_HOME=/root/.hermes + active_profile=coder must redirect
        HERMES_HOME to .../profiles/coder.

        Bug scenario from #22502: systemd sets HERMES_HOME to the hermes root
        and the user switches to a profile via `hermes profile use`.
        Before the fix, the guard returned early and active_profile was ignored.
        """
        hermes_root = tmp_path / ".hermes"
        hermes_root.mkdir(parents=True, exist_ok=True)

        result = _run_apply_profile_override(
            tmp_path,
            monkeypatch,
            hermes_home=str(hermes_root),
            active_profile="coder",
        )

        assert result is not None, "HERMES_HOME must be set after profile redirect"
        assert "profiles" in result, (
            f"Expected HERMES_HOME to point into profiles/ dir, got: {result!r}"
        )
        assert result.endswith("coder"), (
            f"Expected HERMES_HOME to end with 'coder', got: {result!r}"
        )

    def test_hermes_home_already_profile_dir_is_trusted(self, tmp_path, monkeypatch):
        """HERMES_HOME=.../profiles/coder must not be overridden even when
        active_profile says something different.

        Preserves the child-process inheritance contract: a subprocess spawned
        with HERMES_HOME already set to a specific profile must stay in that
        profile.
        """
        hermes_root = tmp_path / ".hermes"
        profile_dir = hermes_root / "profiles" / "coder"
        profile_dir.mkdir(parents=True, exist_ok=True)

        (hermes_root / "active_profile").write_text("other")

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(profile_dir))
        monkeypatch.setattr(sys, "argv", ["hermes", "gateway", "start"])

        from hermes_cli.main import _apply_profile_override
        _apply_profile_override()

        assert os.environ.get("HERMES_HOME") == str(profile_dir), (
            "HERMES_HOME must remain unchanged when already pointing to a profile dir"
        )

    def test_hermes_home_unset_reads_active_profile(self, tmp_path, monkeypatch):
        """Classic case: HERMES_HOME unset + active_profile=coder must set
        HERMES_HOME to the profile directory (existing behaviour must not regress).
        """
        result = _run_apply_profile_override(
            tmp_path,
            monkeypatch,
            hermes_home=None,
            active_profile="coder",
        )

        assert result is not None
        assert "coder" in result

    def test_hermes_home_unset_default_profile_no_redirect(self, tmp_path, monkeypatch):
        """active_profile=default must not redirect HERMES_HOME."""
        hermes_root = tmp_path / ".hermes"
        hermes_root.mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setattr(sys, "argv", ["hermes", "gateway", "start"])
        (hermes_root / "active_profile").write_text("default")

        from hermes_cli.main import _apply_profile_override
        _apply_profile_override()

        assert os.environ.get("HERMES_HOME") is None


class TestApplyProfileOverrideCronJobPin:
    """Regression guard for issue #32046.

    ``hermes cron create|add|edit`` defines its own ``--profile`` argument with
    job-pin semantics (write the job to the active scheduler's jobs.json with
    ``profile=<name>``). The pre-parse hook must NOT consume those trailing
    flags as a global HERMES_HOME switch — otherwise the job lands in
    ``<name>``'s jobs.json with ``profile=null`` and is never ticked by the
    default scheduler.
    """

    def test_cron_create_profile_flag_not_consumed_as_global(
        self, tmp_path, monkeypatch
    ):
        """``hermes cron create ... --profile oracle`` must leave ``--profile
        oracle`` in argv for the cron subparser and must NOT set
        HERMES_HOME to oracle.
        """
        hermes_root = tmp_path / ".hermes"
        hermes_root.mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.delenv("HERMES_HOME", raising=False)
        argv = [
            "hermes",
            "cron",
            "create",
            "--name",
            "oracle-nightly",
            "0 3 * * *",
            "do the thing",
            "--profile",
            "oracle",
        ]
        monkeypatch.setattr(sys, "argv", list(argv))

        from hermes_cli.main import _apply_profile_override
        _apply_profile_override()

        assert os.environ.get("HERMES_HOME") is None, (
            "cron create --profile must not switch the global HERMES_HOME"
        )
        assert sys.argv == argv, (
            "cron create --profile flag must survive in argv for the subparser"
        )

    def test_cron_edit_profile_flag_not_consumed_as_global(
        self, tmp_path, monkeypatch
    ):
        """``hermes cron edit <id> --profile oracle`` likewise leaves the
        cron subparser's ``--profile`` argument alone.
        """
        hermes_root = tmp_path / ".hermes"
        hermes_root.mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.delenv("HERMES_HOME", raising=False)
        argv = ["hermes", "cron", "edit", "f0467c04d70f", "--profile", "oracle"]
        monkeypatch.setattr(sys, "argv", list(argv))

        from hermes_cli.main import _apply_profile_override
        _apply_profile_override()

        assert os.environ.get("HERMES_HOME") is None
        assert sys.argv == argv

    def test_cron_add_alias_also_protected(self, tmp_path, monkeypatch):
        """The ``add`` alias of ``cron create`` is registered in argparse —
        the pre-parse hook must treat it the same way.
        """
        hermes_root = tmp_path / ".hermes"
        hermes_root.mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.delenv("HERMES_HOME", raising=False)
        argv = [
            "hermes",
            "cron",
            "add",
            "0 3 * * *",
            "do the thing",
            "--profile",
            "oracle",
        ]
        monkeypatch.setattr(sys, "argv", list(argv))

        from hermes_cli.main import _apply_profile_override
        _apply_profile_override()

        assert os.environ.get("HERMES_HOME") is None
        assert sys.argv == argv

    def test_cron_create_profile_equals_form_not_consumed(
        self, tmp_path, monkeypatch
    ):
        """The ``--profile=oracle`` (single-token equals form) is likewise a
        cron-subparser arg, not a global override.
        """
        hermes_root = tmp_path / ".hermes"
        hermes_root.mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.delenv("HERMES_HOME", raising=False)
        argv = ["hermes", "cron", "create", "0 3 * * *", "task", "--profile=oracle"]
        monkeypatch.setattr(sys, "argv", list(argv))

        from hermes_cli.main import _apply_profile_override
        _apply_profile_override()

        assert os.environ.get("HERMES_HOME") is None
        assert sys.argv == argv

    def test_cron_create_short_p_flag_not_consumed_as_global(
        self, tmp_path, monkeypatch
    ):
        """The cron subparser does not define ``-p`` for job-pin, but a
        defensive read of the pre-parse hook ignores -p trailing the cron
        subcommand too — otherwise we would mis-route a user-typed ``-p``
        as global, breaking the subparser's later error message.
        """
        hermes_root = tmp_path / ".hermes"
        hermes_root.mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.delenv("HERMES_HOME", raising=False)
        argv = ["hermes", "cron", "create", "0 3 * * *", "task", "-p", "oracle"]
        monkeypatch.setattr(sys, "argv", list(argv))

        from hermes_cli.main import _apply_profile_override
        _apply_profile_override()

        assert os.environ.get("HERMES_HOME") is None
        assert sys.argv == argv

    def test_global_dash_p_before_cron_still_consumed(self, tmp_path, monkeypatch):
        """``hermes -p ozzy cron create ... --profile oracle`` keeps both
        semantics intact: -p (before cron) is the global switch, and
        --profile (after cron create) is the job pin.
        """
        hermes_root = tmp_path / ".hermes"
        (hermes_root / "profiles" / "ozzy").mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "hermes",
                "-p",
                "ozzy",
                "cron",
                "create",
                "0 3 * * *",
                "task",
                "--profile",
                "oracle",
            ],
        )

        from hermes_cli.main import _apply_profile_override
        _apply_profile_override()

        # Global -p ozzy was consumed → HERMES_HOME points at ozzy.
        hermes_home = os.environ.get("HERMES_HOME")
        assert hermes_home is not None and hermes_home.endswith("ozzy")
        # ``-p ozzy`` was stripped from argv but ``--profile oracle`` survives
        # for the cron subparser.
        assert "-p" not in sys.argv
        assert "ozzy" not in sys.argv
        assert sys.argv[-2:] == ["--profile", "oracle"]

    def test_cron_list_with_global_profile_still_consumed(
        self, tmp_path, monkeypatch
    ):
        """``hermes cron list --profile ozzy`` has no cron-subparser
        ``--profile`` arg on ``list``, but the pre-parse hook is gated on
        ``create|add|edit`` only — so this remains a global switch, matching
        the historical behaviour.
        """
        hermes_root = tmp_path / ".hermes"
        (hermes_root / "profiles" / "ozzy").mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setattr(
            sys, "argv", ["hermes", "cron", "list", "--profile", "ozzy"]
        )

        from hermes_cli.main import _apply_profile_override
        _apply_profile_override()

        hermes_home = os.environ.get("HERMES_HOME")
        assert hermes_home is not None and hermes_home.endswith("ozzy")
        assert "--profile" not in sys.argv
        assert "ozzy" not in sys.argv
