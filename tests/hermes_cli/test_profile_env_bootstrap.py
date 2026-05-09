"""Regression tests for hermes_cli.profile_env_bootstrap (Issue #22502)."""

from __future__ import annotations

import os
from pathlib import Path


def test_canonical_home_plus_active_profile_rewrites_home(
    monkeypatch, tmp_path: Path
) -> None:
    from hermes_cli.profile_env_bootstrap import apply_profile_env_override

    home = tmp_path
    monkeypatch.setattr(Path, "home", lambda: home)

    root = home / ".hermes"
    root.mkdir(parents=True)
    prof_dir = root / "profiles" / "svc"
    prof_dir.mkdir(parents=True)

    default_home = root.resolve()
    monkeypatch.delenv("HERMES_HOME", raising=False)

    monkeypatch.setenv("HERMES_HOME", str(default_home))

    active = root / "active_profile"
    active.write_text("svc", encoding="utf-8")

    apply_profile_env_override(cli_argv=[])

    assert Path(os.environ["HERMES_HOME"]).resolve() == prof_dir.resolve()


def test_explicit_profile_home_is_not_overridden_by_active_profile(
    monkeypatch, tmp_path: Path
) -> None:
    from hermes_cli.profile_env_bootstrap import apply_profile_env_override

    home = tmp_path
    monkeypatch.setattr(Path, "home", lambda: home)

    root = home / ".hermes"
    root.mkdir(parents=True)

    svc = root / "profiles" / "svc"
    alt = root / "profiles" / "alt"
    svc.mkdir(parents=True)
    alt.mkdir(parents=True)

    (root / "active_profile").write_text("alt", encoding="utf-8")

    monkeypatch.setenv("HERMES_HOME", str(svc))

    apply_profile_env_override(cli_argv=[])

    assert Path(os.environ["HERMES_HOME"]).resolve() == svc.resolve()


def test_pytest_negative_p_alias_is_not_treated_as_profile(
    monkeypatch, tmp_path: Path,
) -> None:
    from hermes_cli.profile_env_bootstrap import apply_profile_env_override

    home = tmp_path
    monkeypatch.setattr(Path, "home", lambda: home)

    root = home / ".hermes"
    root.mkdir(parents=True)

    monkeypatch.setenv("HERMES_HOME", str(root.resolve()))

    apply_profile_env_override(cli_argv=["-p", "no:xdist", "gateway"])

    assert Path(os.environ["HERMES_HOME"]).resolve() == root.resolve()


def test_explicit_negative_p_profile_flag_still_applies(monkeypatch, tmp_path: Path) -> None:
    """``-p no`` is rejected as ambiguous; numeric-start names are allowed."""
    from hermes_cli.profile_env_bootstrap import apply_profile_env_override

    home = tmp_path
    monkeypatch.setattr(Path, "home", lambda: home)

    root = home / ".hermes"
    root.mkdir(parents=True)
    svc = root / "profiles" / "svc000"
    svc.mkdir(parents=True)

    monkeypatch.setenv("HERMES_HOME", str(root.resolve()))

    apply_profile_env_override(cli_argv=["-p", "svc000"])

    assert Path(os.environ["HERMES_HOME"]).resolve() == svc.resolve()


def test_negative_p_illegal_leader_is_ignored(monkeypatch, tmp_path: Path) -> None:
    """Invalid profile ids ignore ``-p`` as a Hermes profile flag."""
    from hermes_cli.profile_env_bootstrap import apply_profile_env_override

    home = tmp_path
    monkeypatch.setattr(Path, "home", lambda: home)

    root = home / ".hermes"
    root.mkdir(parents=True)

    monkeypatch.setenv("HERMES_HOME", str(root.resolve()))

    apply_profile_env_override(cli_argv=["-p", "_nope"])

    assert Path(os.environ["HERMES_HOME"]).resolve() == root.resolve()
