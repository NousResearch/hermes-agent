"""Regression tests for _apply_profile_override HERMES_HOME guard (issue #22502).

When HERMES_HOME is set to the hermes root (e.g. systemd hardcodes
HERMES_HOME=/root/.hermes), _apply_profile_override must still read
active_profile and update HERMES_HOME to the profile directory.

When HERMES_HOME is already a profile directory (.../profiles/<name>),
_apply_profile_override must trust it and return without re-reading
active_profile (child-process inheritance contract).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


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

        _apply_profile_override()

        assert os.environ.get("HERMES_HOME") == str(profile_dir)
        assert sys.argv == ["hermes", "gateway", "install", "--system"]

    def test_unauthorized_external_noninteractive_profile_launch_is_rejected(
        self, tmp_path, monkeypatch
    ):
        """An external unattended launch cannot cross profiles without authority."""
        monkeypatch.delenv("HERMES_DISPATCH_SOURCE_PROFILE", raising=False)
        monkeypatch.delenv("HERMES_PROFILE", raising=False)

        with pytest.raises(SystemExit) as exc_info:
            _run_apply_profile_override(
                tmp_path,
                monkeypatch,
                hermes_home=None,
                active_profile="coder",
                argv=["hermes", "chat", "-p", "coder", "-q", "hello"],
            )

        assert exc_info.value.code == 77
        assert sys.argv == ["hermes", "chat", "-p", "coder", "-q", "hello"]

    def test_non_tty_interactive_form_still_fails_closed(
        self, tmp_path, monkeypatch
    ):
        """A subprocess cannot bypass authority by omitting query flags."""
        monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: False))
        monkeypatch.delenv("HERMES_DISPATCH_SOURCE_PROFILE", raising=False)
        monkeypatch.delenv("HERMES_PROFILE", raising=False)

        with pytest.raises(SystemExit) as exc_info:
            _run_apply_profile_override(
                tmp_path,
                monkeypatch,
                hermes_home=None,
                active_profile="coder",
                argv=["hermes", "chat", "-p", "coder"],
            )

        assert exc_info.value.code == 77

    def test_human_interactive_profile_after_chat_is_consumed(
        self, tmp_path, monkeypatch
    ):
        """Interactive profile selection remains intentional and consumes -p."""
        monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
        result = _run_apply_profile_override(
            tmp_path,
            monkeypatch,
            hermes_home=None,
            active_profile="coder",
            argv=["hermes", "chat", "-p", "coder"],
        )

        assert result is not None
        assert result.endswith("coder")
        assert sys.argv == ["hermes", "chat"]

    def test_direct_one_shot_noninteractive_profile_launch_records_redacted_prompt(
        self, tmp_path, monkeypatch
    ):
        """Bounded direct authority permits the launch and records no prompt."""
        sentinel = "DIRECT-PROMPT-SENTINEL"
        authority = {
            "authority_class": "direct_one_shot",
            "authority_reference": "DEC-TEST-PROFILE-001",
            "source": "external",
            "target": "coder",
            "scope": "one bounded profile integration test",
            "one_shot": True,
            "expires_at": time.time() + 300,
            "execution_id": "profile-integration-exec-001",
            "evidence": "test_apply_profile_override",
            "terminal_condition": "test call returns",
        }
        monkeypatch.delenv("HERMES_DISPATCH_SOURCE_PROFILE", raising=False)
        monkeypatch.delenv("HERMES_PROFILE", raising=False)
        monkeypatch.setenv("HERMES_EXECUTION_AUTHORITY", json.dumps(authority))

        result = _run_apply_profile_override(
            tmp_path,
            monkeypatch,
            hermes_home=None,
            active_profile="coder",
            argv=["hermes", "chat", "-p", "coder", "-q", sentinel],
        )

        ledger = tmp_path / ".hermes" / "execution-provenance.jsonl"
        row = json.loads(ledger.read_text(encoding="utf-8").strip())
        assert result is not None
        assert result.endswith("coder")
        assert sys.argv == ["hermes", "chat", "-q", sentinel]
        assert row["authority_class"] == "direct_one_shot"
        assert row["source"] == "external"
        assert row["target"] == "coder"
        assert sentinel not in row["execution_path"]
        assert sentinel not in ledger.read_text(encoding="utf-8")
        assert "[REDACTED]" in row["execution_path"]


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
        _apply_profile_override()

        result = os.environ.get("HERMES_HOME")
        assert result is not None
        assert result.endswith("coder")

