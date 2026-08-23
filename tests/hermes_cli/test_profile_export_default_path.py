"""Regression coverage for safe automatic profile-export destinations."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from hermes_cli import profiles
from hermes_cli.cli_commands_mixin import CLICommandsMixin
from hermes_cli.main import cmd_profile
from hermes_cli.profiles import get_profile_export_path


def test_default_export_path_is_managed_and_outside_named_profiles(
    tmp_path, monkeypatch
):
    default_home = tmp_path / ".hermes"
    default_home.mkdir()
    monkeypatch.setattr(
        "hermes_cli.profiles._get_default_hermes_home", lambda: default_home
    )

    result = get_profile_export_path("Research-Bot", timestamp="20260823-120000")

    assert (
        result
        == default_home / "profile-exports" / "research-bot-20260823-120000.tar.gz"
    )
    assert result.parent.is_dir()
    assert not (default_home / "profiles" / "research-bot" / result.name).exists()


def test_custom_hermes_home_inside_a_checkout_uses_a_sibling_store(
    tmp_path, monkeypatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".git").write_text("gitdir: ../git\n", encoding="utf-8")
    monkeypatch.chdir(checkout)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setattr(
        "hermes_cli.profiles._get_default_hermes_home", lambda: checkout
    )

    result = get_profile_export_path("default", timestamp="20260823-120000")

    assert not result.resolve().is_relative_to(checkout.resolve())


def test_cli_export_default_does_not_write_into_the_current_checkout(
    tmp_path, monkeypatch, capsys
):
    default_home = tmp_path / ".hermes"
    default_home.mkdir()
    (default_home / "config.yaml").write_text("model: test\n", encoding="utf-8")
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.chdir(checkout)
    monkeypatch.setattr(
        "hermes_cli.profiles._get_default_hermes_home", lambda: default_home
    )
    monkeypatch.setattr(
        "hermes_constants.get_default_hermes_root", lambda: default_home
    )

    cmd_profile(
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


def test_slash_export_uses_the_same_managed_destination(tmp_path, monkeypatch):
    default_home = tmp_path / ".hermes"
    default_home.mkdir()
    monkeypatch.setattr(
        "hermes_cli.profiles._get_default_hermes_home", lambda: default_home
    )
    monkeypatch.setattr(profiles, "get_active_profile_name", lambda: "default")
    calls = []
    monkeypatch.setattr(
        profiles,
        "export_profile",
        lambda name, output: calls.append((name, output)) or output,
    )

    CLICommandsMixin()._handle_export_command("/export")

    assert len(calls) == 1
    assert calls[0][0] == "default"
    assert Path(calls[0][1]).parent == default_home / "profile-exports"
    assert not (tmp_path / "default.tar.gz").exists()


@pytest.mark.asyncio
async def test_profile_export_api_uses_the_shared_managed_destination(
    tmp_path, monkeypatch
):
    from hermes_cli.web_models import ProfileExport
    from hermes_cli.web_routers.profiles import export_profile_endpoint

    managed = tmp_path / "profile-exports" / "default-20260823-120000.tar.gz"
    monkeypatch.setattr(profiles, "get_profile_export_path", lambda name: managed)
    monkeypatch.setattr(
        profiles,
        "export_profile",
        lambda name, output, extra_files=None: output,
    )

    result = await export_profile_endpoint("default", ProfileExport())

    assert result == {"ok": True, "archive": str(managed)}
