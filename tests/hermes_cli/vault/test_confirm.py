"""`hermes vault confirm`: manages ``vault.confirm_logins``, the sites whose saved logins need the user's
approval on every fill. Entries are normalized to origins, deduplicated, removable, and ``*`` covers every
login; ``--all-profiles`` re-runs the command once per profile."""

from __future__ import annotations

import argparse
import subprocess
from types import SimpleNamespace

import hermes_yaml as yaml
import pytest


@pytest.fixture
def hermes(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli.subcommands.vault import build_vault_parser

    parser = argparse.ArgumentParser(prog="hermes")
    build_vault_parser(parser.add_subparsers())

    def run(*argv):
        args = parser.parse_args(["vault", *argv])
        args.func(args)

    def saved():
        from hermes_cli.config import load_config

        return load_config().get("vault", {}).get("confirm_logins")

    yield run, saved, tmp_path


def test_no_args_on_a_fresh_config_says_nothing_needs_approval(hermes, capsys):
    run, saved, _ = hermes
    run("confirm")
    assert "No sites need approval" in capsys.readouterr().out
    assert not saved()


def test_add_normalizes_to_origins_and_dedupes(hermes, capsys):
    run, saved, _ = hermes
    run("confirm", "bank.example", "https://mail.example/inbox")
    run("confirm", "https://bank.example")  # same site again
    assert saved() == ["https://bank.example", "https://mail.example"]
    out = capsys.readouterr().out
    assert "https://bank.example" in out and "need your approval" in out


def test_remove_matches_bare_and_full_forms(hermes):
    run, saved, home = hermes
    (home / "config.yaml").write_text("vault:\n  confirm_logins: [bank.example, 'https://mail.example']\n")
    run("confirm", "--remove", "https://bank.example")  # stored by hand as a bare host
    assert saved() == ["https://mail.example"]
    run("confirm", "--remove", "mail.example")
    assert saved() == []


def test_star_means_every_login_and_can_be_removed(hermes, capsys):
    run, saved, _ = hermes
    run("confirm", "bank.example", "*")
    assert "Every saved login needs your approval" in capsys.readouterr().out
    run("confirm", "--remove", "*")
    assert saved() == ["https://bank.example"]


def test_scalar_string_entry_is_treated_as_a_list(hermes):
    run, saved, home = hermes
    (home / "config.yaml").write_text("vault:\n  confirm_logins: '*'\n")
    run("confirm", "bank.example")
    assert saved() == ["*", "https://bank.example"]


def test_bad_site_is_a_clean_error_and_writes_nothing(hermes, capsys):
    run, saved, _ = hermes
    run("confirm", "https://")
    assert "Error" in capsys.readouterr().out
    assert not saved()


def test_all_profiles_reruns_once_per_profile(hermes, monkeypatch):
    run, _, _ = hermes
    names = [SimpleNamespace(name=n) for n in ("default", "alpha", "beta")]
    monkeypatch.setattr("hermes_cli.profiles.list_profiles", lambda **_: names)
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/hermes")
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: calls.append((argv, kw)))

    run("confirm", "--all-profiles", "--remove", "bank.example")

    assert [argv[2] for argv, _ in calls] == ["default", "alpha", "beta"]
    for argv, kw in calls:
        assert argv[:2] == ["/usr/bin/hermes", "-p"]
        assert argv[3:] == ["vault", "confirm", "bank.example", "--remove"]
        assert kw["timeout"] and kw["check"] is False
