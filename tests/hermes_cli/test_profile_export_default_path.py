"""Regression coverage for safe automatic profile-export destinations."""

from __future__ import annotations

import importlib
from argparse import Namespace
from pathlib import Path

import pytest


@pytest.fixture()
def profiles():
    """Resolve the live profiles module at call time.

    Sibling test files reload ``hermes_cli.profiles`` / ``hermes_cli.main``;
    a top-level ``from hermes_cli.profiles import ...`` here would bind the
    pre-reload function objects and silently divorce this file's monkeypatches
    from the code under test when the whole directory runs as one sweep.
    """
    return importlib.import_module("hermes_cli.profiles")


def test_default_export_path_is_managed_and_outside_named_profiles(
    tmp_path, monkeypatch, profiles
):
    default_home = tmp_path / ".hermes"
    default_home.mkdir()
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: default_home)

    result = profiles.get_profile_export_path(
        "Research-Bot", timestamp="20260823-120000"
    )

    assert (
        result
        == default_home / "profile-exports" / "research-bot-20260823-120000.tar.gz"
    )
    assert result.parent.is_dir()
    assert not (default_home / "profiles" / "research-bot" / result.name).exists()


def test_custom_hermes_home_inside_a_checkout_uses_a_sibling_store(
    tmp_path, monkeypatch, profiles
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".git").write_text("gitdir: ../git\n", encoding="utf-8")
    monkeypatch.chdir(checkout)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: checkout)

    result = profiles.get_profile_export_path("default", timestamp="20260823-120000")

    assert not result.resolve().is_relative_to(checkout.resolve())


def test_checkout_detection_does_not_depend_on_cwd(tmp_path, monkeypatch, profiles):
    """HERMES_HOME inside a checkout must be detected even when cwd is elsewhere.

    Regression for the cwd-anchored bypass: cron/service-manager invocations
    run from outside the checkout, and the original heuristic walked
    ``Path.cwd()`` — letting the export land inside the source tree again.
    """
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".git").write_text("gitdir: ../git\n", encoding="utf-8")
    hermes_home = checkout / "data_home"
    hermes_home.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)  # cwd has no .git ancestor inside tmp_path
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: hermes_home)

    result = profiles.get_profile_export_path("default", timestamp="20260823-120000")

    assert not result.resolve().is_relative_to(checkout.resolve())


def test_cli_export_rejects_bad_profile_name_without_traceback(
    tmp_path, monkeypatch, capsys, profiles
):
    """A bad name must print a clean error — the helper raises before export."""
    main_mod = importlib.import_module("hermes_cli.main")
    default_home = tmp_path / ".hermes"
    default_home.mkdir()
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: default_home)

    with pytest.raises(SystemExit):
        main_mod.cmd_profile(
            Namespace(
                profile_action="export",
                profile_name="bad//name",
                output=None,
            )
        )

    out = capsys.readouterr().out
    assert "Error:" in out


def test_cli_export_default_does_not_write_into_the_current_checkout(
    tmp_path, monkeypatch, capsys, profiles
):
    main_mod = importlib.import_module("hermes_cli.main")
    default_home = tmp_path / ".hermes"
    default_home.mkdir()
    (default_home / "config.yaml").write_text("model: test\n", encoding="utf-8")
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.chdir(checkout)
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: default_home)
    monkeypatch.setattr(
        "hermes_constants.get_default_hermes_root", lambda: default_home
    )

    main_mod.cmd_profile(
        Namespace(
            profile_action="export",
            profile_name="default",
            output=None,
        )
    )

    exported = list((default_home / "profile-exports").glob("default-*.tar.gz"))
    assert len(exported) == 1
    assert exported[0].parent == default_home / "profile-exports"
    assert not (checkout / "default.tar.gz").exists()
    assert str(exported[0]) in capsys.readouterr().out


def test_slash_export_uses_the_same_managed_destination(
    tmp_path, monkeypatch, profiles
):
    mixin_mod = importlib.import_module("hermes_cli.cli_commands_mixin")
    default_home = tmp_path / ".hermes"
    default_home.mkdir()
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: default_home)
    monkeypatch.setattr(profiles, "get_active_profile_name", lambda: "default")
    calls = []
    monkeypatch.setattr(
        profiles,
        "export_profile",
        lambda name, output: calls.append((name, output)) or output,
    )

    mixin_mod.CLICommandsMixin()._handle_export_command("/export")

    assert len(calls) == 1
    assert calls[0][0] == "default"
    assert Path(calls[0][1]).parent == default_home / "profile-exports"
    assert not (tmp_path / "default.tar.gz").exists()


@pytest.mark.asyncio
async def test_profile_export_api_uses_the_shared_managed_destination(
    tmp_path, monkeypatch, profiles
):
    from hermes_cli.web_models import ProfileExport

    router_mod = importlib.import_module("hermes_cli.web_routers.profiles")

    managed = tmp_path / "profile-exports" / "default-20260823-120000.tar.gz"
    monkeypatch.setattr(profiles, "get_profile_export_path", lambda name: managed)
    monkeypatch.setattr(
        profiles,
        "export_profile",
        lambda name, output, extra_files=None: output,
    )

    result = await router_mod.export_profile_endpoint("default", ProfileExport())

    assert result == {"ok": True, "archive": str(managed)}


def test_cwd_in_unrelated_checkout_does_not_prove_safety(
    tmp_path, monkeypatch, profiles
):
    """cwd inside unrelated checkout A must not stand in for the safety proof
    of a HERMES_HOME inside checkout B."""
    checkout_b = tmp_path / "checkout-b"
    checkout_b.mkdir()
    (checkout_b / ".git").mkdir()
    checkout_a = tmp_path / "checkout-a"
    checkout_a.mkdir()
    (checkout_a / ".git").write_text("gitdir: ../git\n", encoding="utf-8")
    monkeypatch.chdir(checkout_a)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: checkout_b)

    result = profiles.get_profile_export_path("default", timestamp="20260823-120000")

    assert not result.resolve().is_relative_to(checkout_b.resolve())
    assert not result.resolve().is_relative_to(checkout_a.resolve())


def test_every_candidate_inside_a_checkout_fails_closed(
    tmp_path, monkeypatch, profiles
):
    """When home, sibling store, and tempdir all resolve inside checkouts the
    helper must refuse — a warning would not stop a scripted export from
    recreating the #92457 incident artifact."""
    import tempfile

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".git").mkdir()
    monkeypatch.setattr(Path, "home", lambda: checkout / "home")
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: checkout)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(checkout / "tmp"))

    with pytest.raises(ValueError, match="No safe automatic export destination"):
        profiles.get_profile_export_path("default")
