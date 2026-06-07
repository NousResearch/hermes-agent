"""Regression tests for profile propagation in gateway run (issue #41355).

When ``hermes gateway run --profile <name>`` is invoked, HERMES_HOME must be
set to the profile directory SO THAT the gateway loads config, SOUL.md,
MEMORY.md, skills, and sessions from the profile, not the default root.

The existing tests in test_apply_profile_override.py cover the sticky
active_profile path but NOT the explicit ``--profile`` flag path.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _run_apply_profile_override_with_flag(
    tmp_path, monkeypatch, *, profile_name: str, argv_prefix: list[str] | None = None,
):
    """Run _apply_profile_override with ``--profile <name>`` in sys.argv.

    Returns ``os.environ["HERMES_HOME"]`` after the call, or None if unset.

    The function is imported at call time (so monkeypatching takes effect
    before the module-level call in main.py).
    """
    hermes_root = tmp_path / ".hermes"
    profile_dir = hermes_root / "profiles" / profile_name
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "config.yaml").write_text("model:\n  default: test-model\n")

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_HOME", raising=False)

    argv = (argv_prefix or ["hermes", "gateway", "run"]) + [
        "--profile", profile_name,
    ]
    monkeypatch.setattr(sys, "argv", argv)

    from hermes_cli.main import _apply_profile_override

    _apply_profile_override()

    return os.environ.get("HERMES_HOME")


class TestGatewayRunProfilePropagation:
    """Verify that --profile flag sets HERMES_HOME for gateway run."""

    def test_gateway_run_with_profile_sets_hermes_home(
        self, tmp_path, monkeypatch,
    ):
        """``hermes gateway run --profile test-bot`` must set HERMES_HOME."""
        result = _run_apply_profile_override_with_flag(
            tmp_path, monkeypatch,
            profile_name="test-bot",
            argv_prefix=["hermes", "gateway", "run"],
        )

        assert result is not None, (
            "HERMES_HOME must be set when --profile is passed to gateway run"
        )
        assert result.endswith("test-bot"), (
            f"HERMES_HOME must end with profile name, got: {result!r}"
        )

    def test_gateway_run_with_profile_flag_after_replace(
        self, tmp_path, monkeypatch,
    ):
        """``hermes gateway run --replace --profile test-bot`` works
        (--replace between subcommand and --profile)."""
        result = _run_apply_profile_override_with_flag(
            tmp_path, monkeypatch,
            profile_name="test-bot",
            argv_prefix=["hermes", "gateway", "run", "--replace"],
        )

        assert result is not None, (
            "HERMES_HOME must be set when --profile follows --replace"
        )
        assert result.endswith("test-bot"), (
            f"HERMES_HOME must end with profile name, got: {result!r}"
        )

    def test_gateway_run_flag_stripped_from_argv(
        self, tmp_path, monkeypatch,
    ):
        """After _apply_profile_override, sys.argv must NOT contain --profile."""
        profile_name = "test-bot"
        hermes_root = tmp_path / ".hermes"
        profile_dir = hermes_root / "profiles" / profile_name
        profile_dir.mkdir(parents=True, exist_ok=True)
        (profile_dir / "config.yaml").write_text("model:\n  default: test-model\n")

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.delenv("HERMES_HOME", raising=False)

        orig_argv = ["hermes", "gateway", "run", "--profile", profile_name]
        monkeypatch.setattr(sys, "argv", list(orig_argv))

        from hermes_cli.main import _apply_profile_override

        _apply_profile_override()

        assert "--profile" not in sys.argv, (
            f"--profile flag must be stripped from sys.argv, got: {sys.argv}"
        )
        assert profile_name not in sys.argv, (
            f"profile name must be stripped from sys.argv, got: {sys.argv}"
        )
        assert "gateway" in sys.argv
        assert "run" in sys.argv

    def test_gateway_run_module_level_hermes_home_is_profile_aware(
        self, tmp_path, monkeypatch,
    ):
        """After _apply_profile_override with --profile, importing
        gateway.run must set _hermes_home to the profile dir.

        This is the critical path: gateway/run.py caches _hermes_home at
        module import time. If HERMES_HOME is not set before the import,
        _hermes_home points to the default dir and stays wrong forever.
        """
        profile_name = "test-bot"
        hermes_root = tmp_path / ".hermes"
        profile_dir = hermes_root / "profiles" / profile_name
        profile_dir.mkdir(parents=True, exist_ok=True)
        (profile_dir / "config.yaml").write_text("model:\n  default: test\n")
        (profile_dir / "sessions").mkdir(exist_ok=True)
        (profile_dir / "cron").mkdir(exist_ok=True)
        (profile_dir / "memories").mkdir(exist_ok=True)
        (profile_dir / "skills").mkdir(exist_ok=True)

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setattr(
            sys, "argv",
            ["hermes", "gateway", "run", "--profile", profile_name],
        )

        # Simulate the startup sequence exactly as main.py does it:
        # 1. _apply_profile_override sets HERMES_HOME
        from hermes_cli.main import _apply_profile_override
        _apply_profile_override()

        assert os.environ.get("HERMES_HOME", "").endswith(profile_name), (
            "HERMES_HOME must be set to profile dir before gateway import"
        )

        # 2. Import gateway.run — this reads HERMES_HOME at module level
        import importlib
        if "gateway.run" in sys.modules:
            del sys.modules["gateway.run"]

        # Install a fake hermes_cli.env_loader that does nothing
        # (load_hermes_dotenv would try to read the real .env)
        import types
        fake_env_loader = types.ModuleType("hermes_cli.env_loader")
        fake_env_loader.load_hermes_dotenv = lambda **kw: []
        fake_env_loader._APPLIED_HOMES = set()
        fake_env_loader._SECRET_SOURCES = {}
        monkeypatch.setitem(
            sys.modules, "hermes_cli.env_loader", fake_env_loader,
        )

        # Also suppress dotenv import in gateway/run.py
        import dotenv
        monkeypatch.setitem(sys.modules, "dotenv", dotenv)

        from gateway.run import _hermes_home

        assert str(_hermes_home).endswith(profile_name), (
            f"gateway.run._hermes_home must point to profile dir, "
            f"got: {_hermes_home}"
        )

    def test_apply_profile_override_sets_contextvar_override(
        self, tmp_path, monkeypatch,
    ):
        """_apply_profile_override must set both HERMES_HOME env var
        AND the ContextVar override (defense-in-depth)."""
        profile_name = "test-bot"
        hermes_root = tmp_path / ".hermes"
        profile_dir = hermes_root / "profiles" / profile_name
        profile_dir.mkdir(parents=True, exist_ok=True)
        (profile_dir / "config.yaml").write_text("model:\n  default: test\n")

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setattr(
            sys, "argv",
            ["hermes", "gateway", "run", "--profile", profile_name],
        )

        from hermes_cli.main import _apply_profile_override
        _apply_profile_override()

        # Env var must be set
        assert os.environ.get("HERMES_HOME", "").endswith(profile_name)

        # ContextVar override must also be set
        from hermes_constants import get_hermes_home_override
        override = get_hermes_home_override()
        assert override is not None, (
            "ContextVar override must be set after _apply_profile_override"
        )
        assert override.endswith(profile_name), (
            f"ContextVar override must point to profile dir, got: {override!r}"
        )
