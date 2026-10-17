"""A channel whose release predates the gateway runtime holds the install where it is (#140729).

The channel picked that release, the user did not ask to go back: ``hermes update`` reports that
the build is newer than the release and holds there, exiting 0, instead of refusing the
downgrade on every attempt. An explicit ``--branch`` to a pre-runtime commit is still refused.
"""
import argparse
from contextlib import closing
from copy import deepcopy
import subprocess
from types import SimpleNamespace

import pytest

import hermes_state_runtime as rt
from hermes_cli import main, source_releases, update_cmd
from hermes_cli.subcommands.update import build_update_parser
from hermes_cli.update_channel import set_install_channel
from hermes_state import SessionDB

pytestmark = pytest.mark.real_release_channels

_PRE_RUNTIME = "SCHEMA_SQL = 'CREATE TABLE IF NOT EXISTS sessions (id TEXT)'\n"
_RUNTIME = "SCHEMA_SQL = '''CREATE TABLE IF NOT EXISTS session_admissions (seq INTEGER)'''\n"


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True,
                          text=True, stdin=subprocess.DEVNULL).stdout.strip()


@pytest.fixture
def source(tmp_path, monkeypatch):
    """An install on a post-runtime ``main`` commit; the release sits beside it, before the runtime."""
    home, origin, checkout = (tmp_path / name for name in ("home", "origin", "checkout"))
    home.mkdir()
    origin.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_INSTALL_ROOT", str(checkout))
    monkeypatch.delenv("HERMES_MANAGED", raising=False)
    git(origin, "init", "-b", "main")
    git(origin, "config", "user.name", "Channel Fixture")
    git(origin, "config", "user.email", "fixture@example.invalid")
    git(origin, "config", "commit.gpgsign", "false")
    (origin / "hermes_state_common.py").write_text(_PRE_RUNTIME)
    git(origin, "add", ".")
    git(origin, "commit", "-m", "before the cutover")
    git(origin, "checkout", "-q", "-b", "release")
    (origin / "notes.txt").write_text("release-only fix\n")
    git(origin, "add", ".")
    git(origin, "commit", "-m", "stable release cut beside main")
    release = git(origin, "rev-parse", "HEAD")
    git(origin, "checkout", "-q", "main")
    (origin / "hermes_state_common.py").write_text(_RUNTIME)
    git(origin, "commit", "-am", "gateway runtime")
    installed = git(origin, "rev-parse", "HEAD")
    git(tmp_path, "clone", "-q", str(origin), str(checkout))
    git(checkout, "fetch", "-q", "origin", "release")
    git(checkout, "checkout", "-q", "--detach", installed)
    monkeypatch.setattr(main, "PROJECT_ROOT", checkout)
    monkeypatch.setattr("hermes_cli.config.get_project_root", lambda: checkout)
    monkeypatch.setattr("hermes_cli.backup._sibling_profile_homes", lambda h: [])
    monkeypatch.setattr(update_cmd._commit, "preflight_refusal", lambda *a, **k: None)
    parser = argparse.ArgumentParser()
    build_update_parser(parser.add_subparsers(), cmd_update=main.cmd_update)
    opts = update_cmd._UpdateOptions(pre_update_version=None, gw_input_fn=None,
        assume_yes=True, keep_stash=False, switch_branch=False, discard_local_changes=False)
    monkeypatch.setattr(update_cmd, "_resolve_update_options", lambda *_: opts)
    monkeypatch.setattr(update_cmd, "_begin_update_receipt_and_plan", lambda *_: None)
    monkeypatch.setattr(main, "_run_pre_update_backup", lambda *_: None)
    monkeypatch.setattr(main, "_pause_windows_gateways_for_update", lambda: None)
    monkeypatch.setattr(update_cmd, "_prepare_git_command", lambda **_: (False, ["git"], False))
    completed = []
    monkeypatch.setattr(update_cmd, "_complete_source_update",
                        lambda request: completed.append(deepcopy(request)))
    return SimpleNamespace(home=home, root=checkout, release=release, installed=installed,
                           parser=parser, completed=completed)


def _ledgered(home):
    with closing(SessionDB(home / "state.db")) as db:
        epoch = rt.begin_runtime_epoch(db, instance_id="owner")
        db.create_session("s", source="cli")
        rt.admit_session_input(db, epoch=epoch, principal_id="p", session_id="s", request_id="r",
                               payload={"text": "x"})


def _channel_publishes(monkeypatch, source, name):
    record = {"schema": 1, "name": name, "repository": "NousResearch/hermes-agent", "policy": "preview",
              "state": "active", "identity": {}, "nextSequence": 2,
              "head": {"buildId": "build-fixture", "sequence": 1}}
    result = SimpleNamespace(requested=deepcopy(record), terminal=record, manifest={
        "schema": 1, "request": {"buildId": "build-fixture", "channel": name, "sequence": 1,
        "repository": "NousResearch/hermes-agent", "commit": source.release, "sourceVersion": "1.2.3",
        "version": "0.0.1", "identity": {}, "bundleEnv": {},
        "publicBase": "https://hermes-assets.nousresearch.com"}, "packages": []})
    monkeypatch.setattr(source_releases, "_resolve_channel", lambda n, repository: deepcopy(result),
                        raising=False)


def test_a_channel_release_older_than_the_runtime_holds_the_install_and_exits_cleanly(
        source, monkeypatch, capsys):
    name = "preview-before-runtime"
    set_install_channel(name, source.root)
    _channel_publishes(monkeypatch, source, name)
    _ledgered(source.home)

    update_cmd._cmd_update_impl(source.parser.parse_args(["update", "--yes"]), False)  # no SystemExit

    out = capsys.readouterr().out
    assert git(source.root, "rev-parse", "HEAD") == source.installed
    assert "Refusing" not in out and "Downgrading is unsupported" not in out
    assert f"This build is newer than the {name} release {source.release[:12]}" in out
    assert "hermes update --set-channel main" in out
    assert len(source.completed) == 1 and source.completed[0]["expected_sha"] == source.installed
    # A held install is not up to date with its channel: the closing line says it was held.
    assert source.completed[0]["completion_message"] == f"✓ Holding at this build until a newer {name} release ships."


def test_the_same_release_still_applies_while_no_store_holds_runtime_history(source, monkeypatch, capsys):
    name = "preview-before-runtime"
    set_install_channel(name, source.root)
    _channel_publishes(monkeypatch, source, name)

    update_cmd._cmd_update_impl(source.parser.parse_args(["update", "--yes"]), False)

    assert git(source.root, "rev-parse", "HEAD") == source.release
    assert "newer than the" not in capsys.readouterr().out
    assert source.completed[0]["expected_sha"] == source.release


def test_an_explicit_branch_to_a_pre_runtime_commit_is_still_refused(source, monkeypatch, capsys):
    _ledgered(source.home)

    with pytest.raises(SystemExit) as stopped:
        update_cmd._cmd_update_impl(source.parser.parse_args(["update", "--branch", "release", "--yes"]), False)

    out = capsys.readouterr().out
    assert stopped.value.code == 1 and git(source.root, "rev-parse", "HEAD") == source.installed
    assert "Downgrading is unsupported" in out and "newer than the" not in out
    assert source.completed == []
