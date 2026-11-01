"""Classic CLI config loading follows the active profile scope without cross-profile fallback."""

from __future__ import annotations

import contextlib
import contextvars
import os
from pathlib import Path

import pytest

from hermes_constants import (
    mark_named_profile_deleted,
    reset_hermes_home_override,
    set_hermes_home_override,
)


def _write_config(home: Path, *, owner: str, compact: bool, cjk_fts: bool | None = None) -> None:
    home.mkdir(parents=True, exist_ok=True)
    sessions = "" if cjk_fts is None else f"sessions:\n  cjk_fts: {str(cjk_fts).lower()}\n"
    (home / "config.yaml").write_text(
        f"agent:\n  system_prompt: {owner}\ndisplay:\n  compact: {str(compact).lower()}\n{sessions}",
        encoding="utf-8",
    )


@contextlib.contextmanager
def _selected(home: Path):
    token = set_hermes_home_override(str(home))
    try:
        yield
    finally:
        reset_hermes_home_override(token)


@pytest.fixture
def cli_homes(tmp_path, monkeypatch):
    import cli

    launch = tmp_path / ".hermes"
    worker = launch / "profiles" / "worker"
    _write_config(launch, owner="launch-profile", compact=False)
    _write_config(worker, owner="worker-profile", compact=True, cjk_fts=False)
    monkeypatch.setattr(cli, "_hermes_home", launch)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)

    clear_token = set_hermes_home_override(None)
    try:
        yield cli, launch, worker
    finally:
        reset_hermes_home_override(clear_token)


def test_context_selected_profile_config_has_priority_over_launch_home(cli_homes):
    cli, _launch, worker = cli_homes

    with _selected(worker):
        cfg = cli.load_cli_config()

    assert cfg["agent"]["system_prompt"] == "worker-profile"
    assert cfg["display"]["compact"] is True


def test_selected_profile_scope_is_inherited_by_copied_context(cli_homes):
    cli, _launch, worker = cli_homes

    token = set_hermes_home_override(str(worker))
    inherited = contextvars.copy_context()
    reset_hermes_home_override(token)

    cfg = inherited.run(cli.load_cli_config)
    assert cfg["agent"]["system_prompt"] == "worker-profile"


def test_missing_selected_config_uses_defaults_not_launch_config(cli_homes):
    cli, _launch, worker = cli_homes
    (worker / "config.yaml").unlink()
    (worker / ".env").write_text("# profile identity\n", encoding="utf-8")

    with _selected(worker):
        cfg = cli.load_cli_config()

    assert cfg["agent"]["system_prompt"] != "launch-profile"
    assert cfg["display"]["compact"] is False


def test_without_profile_override_loads_launch_profile_config(cli_homes):
    cli, _launch, _worker = cli_homes

    cfg = cli.load_cli_config()

    assert cfg["agent"]["system_prompt"] == "launch-profile"
    assert cfg["display"]["compact"] is False


def test_ignore_user_config_skips_selected_profile_config(cli_homes, monkeypatch):
    cli, _launch, worker = cli_homes
    monkeypatch.setenv("HERMES_IGNORE_USER_CONFIG", "1")

    with _selected(worker):
        cfg = cli.load_cli_config()

    assert cfg["agent"]["system_prompt"] != "worker-profile"
    assert "launch-profile" not in str(cfg)


def test_managed_overlay_remains_above_selected_profile_config(cli_homes, monkeypatch):
    cli, _launch, worker = cli_homes
    from hermes_cli import managed_scope

    def apply_managed(config):
        config["display"] = {**config["display"], "compact": False}
        return config

    monkeypatch.setattr(managed_scope, "apply_managed_overlay", apply_managed)
    with _selected(worker):
        cfg = cli.load_cli_config()

    assert cfg["agent"]["system_prompt"] == "worker-profile"
    assert cfg["display"]["compact"] is False


def test_deleted_selected_profile_is_refused_before_config_read(cli_homes):
    cli, _launch, worker = cli_homes
    mark_named_profile_deleted(worker)

    with _selected(worker), pytest.raises(FileNotFoundError, match="Named profile home does not exist"):
        cli.load_cli_config()


def test_routed_profile_config_is_not_mirrored_into_launch_process_env(cli_homes, monkeypatch):
    cli, _launch, worker = cli_homes
    monkeypatch.setenv("HERMES_CJK_FTS", "launch-value")

    with _selected(worker):
        cfg = cli.load_cli_config()

    assert cfg["sessions"]["cjk_fts"] is False
    assert "launch-profile" not in str(cfg)
    assert os.environ["HERMES_CJK_FTS"] == "launch-value"


def test_routed_profile_env_refs_use_its_bound_secret_scope(cli_homes, monkeypatch):
    cli, _launch, worker = cli_homes
    (worker / "config.yaml").write_text(
        "agent:\n  system_prompt: ${PROFILE_ONLY_SECRET}\n",
        encoding="utf-8",
    )
    (worker / ".env").write_text("PROFILE_ONLY_SECRET=worker-secret\n", encoding="utf-8")
    monkeypatch.setenv("PROFILE_ONLY_SECRET", "launch-secret")

    from agent.secret_scope import (
        build_profile_secret_scope,
        reset_secret_scope,
        set_secret_scope,
    )

    home_token = set_hermes_home_override(str(worker))
    secret_token = set_secret_scope(build_profile_secret_scope(worker), profile_home=str(worker))
    try:
        cfg = cli.load_cli_config()
    finally:
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)

    assert cfg["agent"]["system_prompt"] == "worker-secret"
    assert "launch-secret" not in str(cfg)
