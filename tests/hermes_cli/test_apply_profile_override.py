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
from types import SimpleNamespace



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

    # Clear the PID-scoped run-once sentinel from issue #66907 so that
    # _apply_profile_override actually executes inside this test process
    # (previous test functions in the same pytest process may have set it).
    os.environ.pop("_HERMES_PROFILE_OVERRIDE_APPLIED", None)

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
        os.environ.pop("_HERMES_PROFILE_OVERRIDE_APPLIED", None)
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

    def test_sudo_explicit_profile_resolves_invoking_users_profile(self, tmp_path, monkeypatch):
        """sudo elias ... should resolve `-p elias` under SUDO_USER, not root."""
        root_home = tmp_path / "root"
        user_home = tmp_path / "home" / "hermes"
        profile_dir = user_home / ".hermes" / "profiles" / "elias"
        profile_dir.mkdir(parents=True, exist_ok=True)
        (root_home / ".hermes").mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(Path, "home", lambda: root_home)
        monkeypatch.setenv("SUDO_USER", "hermes")
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setattr(os, "geteuid", lambda: 0, raising=False)
        monkeypatch.setattr(sys, "argv", ["hermes", "-p", "elias", "gateway", "install", "--system"])

        import pwd

        monkeypatch.setattr(pwd, "getpwnam", lambda name: SimpleNamespace(pw_dir=str(user_home)))

        from hermes_cli.main import _apply_profile_override
        os.environ.pop("_HERMES_PROFILE_OVERRIDE_APPLIED", None)
        _apply_profile_override()

        assert os.environ.get("HERMES_HOME") == str(profile_dir)
        assert sys.argv == ["hermes", "gateway", "install", "--system"]

    def test_hermes_home_unset_default_profile_no_redirect(self, tmp_path, monkeypatch):
        """active_profile=default must not redirect HERMES_HOME."""
        hermes_root = tmp_path / ".hermes"
        hermes_root.mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setattr(sys, "argv", ["hermes", "gateway", "start"])
        (hermes_root / "active_profile").write_text("default")

        from hermes_cli.main import _apply_profile_override
        os.environ.pop("_HERMES_PROFILE_OVERRIDE_APPLIED", None)
        _apply_profile_override()

        assert os.environ.get("HERMES_HOME") is None

    def test_subcommand_profile_flag_is_not_consumed(self, tmp_path, monkeypatch):
        """Command argv flags named --profile must stay with that command.

        Docker Desktop's MCP Toolkit uses `docker mcp gateway run --profile ...`.
        When that argv is passed through `hermes mcp add --args`, the early
        profile pre-parser must not interpret the Docker profile as a Hermes
        profile.
        """
        hermes_root = tmp_path / ".hermes"
        hermes_root.mkdir(parents=True, exist_ok=True)
        argv = [
            "hermes",
            "mcp",
            "add",
            "docker-research",
            "--command",
            "docker",
            "--args",
            "mcp",
            "gateway",
            "run",
            "--profile",
            "research",
        ]

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setattr(sys, "argv", list(argv))

        from hermes_cli.main import _apply_profile_override
        os.environ.pop("_HERMES_PROFILE_OVERRIDE_APPLIED", None)
        _apply_profile_override()

        assert os.environ.get("HERMES_HOME") is None
        assert sys.argv == argv

    def test_profile_after_chat_subcommand_is_still_consumed(self, tmp_path, monkeypatch):
        """Profile flags historically work after normal Hermes subcommands."""
        result = _run_apply_profile_override(
            tmp_path,
            monkeypatch,
            hermes_home=None,
            active_profile="coder",
            argv=["hermes", "chat", "-p", "coder", "-q", "hello"],
        )

        assert result is not None
        assert result.endswith("coder")
        assert sys.argv == ["hermes", "chat", "-q", "hello"]

    def test_top_level_profile_after_value_flag_is_consumed(self, tmp_path, monkeypatch):
        """Top-level --profile still works after other top-level value flags."""
        result = _run_apply_profile_override(
            tmp_path,
            monkeypatch,
            hermes_home=None,
            active_profile="coder",
            argv=["hermes", "-m", "gpt-5", "--profile", "coder", "chat"],
        )

        assert result is not None
        assert result.endswith("coder")
        assert sys.argv == ["hermes", "-m", "gpt-5", "chat"]

    def test_top_level_profile_after_continue_flag_is_consumed(self, tmp_path, monkeypatch):
        """--continue has an optional value, so a following --profile is a flag."""
        result = _run_apply_profile_override(
            tmp_path,
            monkeypatch,
            hermes_home=None,
            active_profile="coder",
            argv=["hermes", "--continue", "--profile", "coder"],
        )

        assert result is not None
        assert result.endswith("coder")
        assert sys.argv == ["hermes", "--continue"]


class TestSupervisedChildIgnoresStickyProfile:
    """The reserved default gateway s6 slot must not follow active_profile.

    Inside the Docker s6 image the ``gateway-default`` service slot runs a
    bare ``hermes gateway run`` (no ``-p``) to mean "the root HERMES_HOME
    profile". The run-script exports ``HERMES_S6_SUPERVISED_CHILD=1``.
    Without a guard, ``_apply_profile_override`` would read the sticky
    ``active_profile`` file (set by e.g. the dashboard profile switcher) and
    redirect the reserved default gateway into that profile — producing a
    duplicate gateway for the active profile and no real default gateway.
    """

    def test_supervised_child_does_not_follow_active_profile(
        self, tmp_path, monkeypatch
    ):
        """HERMES_S6_SUPERVISED_CHILD + active_profile=briefer must NOT redirect.

        Reproduces the Docker/profile scoping bug: the supervised default
        gateway is launched as bare ``hermes gateway run`` with
        HERMES_HOME=/opt/data (the container root, whose parent is NOT
        ``profiles``), and a sticky ``active_profile`` of another profile.
        The reserved default slot must stay on the root profile.
        """
        hermes_root = tmp_path / ".hermes"
        hermes_root.mkdir(parents=True, exist_ok=True)
        (hermes_root / "active_profile").write_text("briefer")
        (hermes_root / "profiles" / "briefer").mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        # Container root HERMES_HOME: parent dir is NOT "profiles", so the
        # #22502 guard does not short-circuit — step 2 (active_profile) runs.
        monkeypatch.setenv("HERMES_HOME", str(hermes_root))
        monkeypatch.setenv("HERMES_S6_SUPERVISED_CHILD", "1")
        monkeypatch.setattr(sys, "argv", ["hermes", "gateway", "run"])

        from hermes_cli.main import _apply_profile_override
        os.environ.pop("_HERMES_PROFILE_OVERRIDE_APPLIED", None)
        _apply_profile_override()

        assert os.environ.get("HERMES_HOME") == str(hermes_root), (
            "Supervised default gateway must stay on the root profile, not be "
            f"hijacked by active_profile; got {os.environ.get('HERMES_HOME')!r}"
        )

    def test_non_supervised_run_still_follows_active_profile(
        self, tmp_path, monkeypatch
    ):
        """Without the sentinel, a normal `hermes gateway run` still honors
        active_profile — the guard is scoped strictly to supervised children."""
        result = _run_apply_profile_override(
            tmp_path,
            monkeypatch,
            hermes_home=None,
            active_profile="briefer",
            argv=["hermes", "gateway", "run"],
        )

        assert result is not None
        assert result.endswith("briefer")

    def test_supervised_named_profile_flag_still_wins(self, tmp_path, monkeypatch):
        """A supervised named-profile slot passes ``-p <name>`` explicitly;
        that must still resolve (the sentinel guard only skips the sticky
        active_profile fallback, never an explicit flag)."""
        hermes_root = tmp_path / ".hermes"
        hermes_root.mkdir(parents=True, exist_ok=True)
        (hermes_root / "active_profile").write_text("briefer")
        (hermes_root / "profiles" / "briefer").mkdir(parents=True, exist_ok=True)
        (hermes_root / "profiles" / "coder").mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setenv("HERMES_S6_SUPERVISED_CHILD", "1")
        monkeypatch.setattr(sys, "argv", ["hermes", "-p", "coder", "gateway", "run"])
        from hermes_cli.main import _apply_profile_override
        os.environ.pop("_HERMES_PROFILE_OVERRIDE_APPLIED", None)
        _apply_profile_override()

        result = os.environ.get("HERMES_HOME")
        assert result is not None
        assert result.endswith("coder")


class TestSentinelBlocksSecondPassDefaultProfileHijack:
    """Regression guard for issue #66907: PID-scoped run-once sentinel.

    Under ``python -m hermes_cli.main --profile default gateway run`` the
    module executes twice (once as ``__main__``, once as import). The first
    pass consumes ``--profile`` from ``sys.argv``; without the sentinel the
    second pass would silently re-read ``active_profile`` and rebind
    ``HERMES_HOME`` to a different profile.

    These tests exercise the sentinel's *behavior* directly, complementing
    the existing tests (which only ``pop`` the sentinel's env so the function
    keeps running).  They fail without the sentinel added in PR #67048 and
    pass with it.
    """

    def _setup_root_with_active_profile(self, tmp_path, monkeypatch, active: str):
        """Bootstrap a ``tmp_path/.hermes`` with a sticky ``active_profile`` file."""
        hermes_root = tmp_path / ".hermes"
        hermes_root.mkdir(parents=True, exist_ok=True)
        (hermes_root / "active_profile").write_text(active)
        # Create a profile subdirectory so the active profile is resolvable.
        if active != "default":
            (hermes_root / "profiles" / active).mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        return hermes_root

    def test_second_pass_does_not_rebind_when_sentinel_present(self, tmp_path, monkeypatch):
        """First pass with ``--profile default``; second pass must NOT re-read
        ``active_profile`` and rebind ``HERMES_HOME`` — the sentinel should
        make the second call a no-op.

        Without the sentinel: first pass sets HERMES_HOME to root (.hermes)
        and strips ``--profile``; second pass sees no flag, reads
        active_profile=ozzy, and rebinds HERMES_HOME to profiles/ozzy —
        the #66907 regression. With the sentinel: second pass is a no-op.
        """
        hermes_root = self._setup_root_with_active_profile(
            tmp_path, monkeypatch, active="ozzy"
        )

        # Clear sentinel + HERMES_HOME so the first-pass invocation behaves as
        # the genuine "first pass" seen under `python -m hermes_cli.main`.
        monkeypatch.delenv("_HERMES_PROFILE_OVERRIDE_APPLIED", raising=False)
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setattr(
            sys, "argv", ["hermes", "--profile", "default", "gateway", "run"]
        )

        from hermes_cli.main import _apply_profile_override
        _apply_profile_override()

        # After first pass: sentinel must be set to this PID, and HERMES_HOME
        # must point at the root home (.hermes), exactly the documented
        # behavior of `--profile default`.  This is the precondition that
        # makes the bug demonstrable: HERMES_HOME is set but NOT to a
        # profiles/<name> path, so the #22502 early-return does not apply.
        assert os.environ.get("_HERMES_PROFILE_OVERRIDE_APPLIED") == str(os.getpid())
        first_pass_home = os.environ.get("HERMES_HOME")
        assert first_pass_home == str(hermes_root), (
            f"precondition: --profile default should set HERMES_HOME to root, "
            f"got {first_pass_home!r}, expected {str(hermes_root)!r}"
        )

        # Second pass: simulate the import-time re-entry. Sentinel is set,
        # so the function must return early WITHOUT touching HERMES_HOME,
        # even though active_profile=ozzy would otherwise have rebinded it
        # to profiles/ozzy.
        monkeypatch.setattr(sys, "argv", ["hermes", "gateway", "run"])  # flag stripped
        _apply_profile_override()

        assert os.environ.get("HERMES_HOME") == str(hermes_root), (
            "second pass leaked — sentinel failed to make the call idempotent; "
            "active_profile was re-read and HERMES_HOME was rebound to a "
            "different profile. regression for issue #66907"
        )

    def test_sentinel_scoped_to_pid_not_inherited_by_child(self, tmp_path, monkeypatch):
        """A spawned child (different PID) must resolve its own profile fresh —
        the sentinel must NOT survive across fork/exec in a way that prevents
        the child from running the override at all.

        Regression value: this catches a subtle mistype where the guard is
        changed to ``if os.environ.get(...) is not None: return`` (any value
        short-circuits). That would let the parent's sentinel leak into the
        child and silently prevent profile resolution in spawned subprocesses.
        """
        self._setup_root_with_active_profile(tmp_path, monkeypatch, active="ozzy")

        # Simulate parent having already set the sentinel for its own PID.
        # The child inherits this env var but has a different pid — we model
        # that by setting the env var to a PID value != current getpid().
        parent_pid = os.getpid() + 1  # any value != current pid
        monkeypatch.setenv("_HERMES_PROFILE_OVERRIDE_APPLIED", str(parent_pid))
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setattr(
            sys, "argv", ["hermes", "--profile", "ozzy", "gateway", "run"]
        )

        from hermes_cli.main import _apply_profile_override
        _apply_profile_override()

        # Now that we've simulated "child inherited a stale sentinel from a
        # different PID", the override must have re-applied --profile ozzy and
        # set HERMES_HOME.  A guard of the form ``is not None`` would have
        # short-circuited here and left HERMES_HOME unset.
        result = os.environ.get("HERMES_HOME")
        assert result is not None, (
            "child-process resolution was prevented by inherited sentinel — "
            "sentinel must be PID-scoped (compare against os.getpid()), not "
            "process-tree-scoped (any truthy value). regression for #66907"
        )
        assert result.endswith("ozzy")

    def teardown_method(self, method):
        """Make sure we never leak the sentinel into other test classes."""
        os.environ.pop("_HERMES_PROFILE_OVERRIDE_APPLIED", None)

