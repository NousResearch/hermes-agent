"""A release move must not orphan commits only this checkout reaches.

``hermes update`` used to detach a checkout onto a release whose commit is not a descendant of
HEAD, silently rolling the running tree back past commits HEAD carried. ``_move_checkout_to``
returns at its merge_ref branch before any ancestry test, and the pre-flight gates never look at
divergence.

The contract after the fix:
- A release move that would orphan commits no ref protects (branch, tag, remote) is refused
  loudly before any tree write; HEAD stays put.
- An explicit ``--set-channel``/``--channel``/``--branch`` keeps the move on a warning path:
  the operator named the destination, so consent is given — but the orphaned work is named.
- Divergence where every divergent commit sits on a ref stays channel-following (the release
  hold suite pins that); only ref-unprotected work refuses.
- ``hermes update --check`` reports the held state instead of advertising the release.
"""
import argparse
import subprocess
from contextlib import closing
from copy import deepcopy
from types import SimpleNamespace

import pytest

from hermes_cli import main, source_releases, update_cmd
from hermes_cli.subcommands.update import build_update_parser
from hermes_state import SessionDB

pytestmark = pytest.mark.real_release_channels

_PRE = "SCHEMA_SQL = 'CREATE TABLE IF NOT EXISTS sessions (id TEXT)'\n"


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True,
                          text=True, stdin=subprocess.DEVNULL).stdout.strip()


@pytest.fixture
def diverged(tmp_path, monkeypatch):
    """Install detached at main's tip, then one local commit the release line never had.

    origin/main: base -> runtime (installed). release: base -> release-only commit.
    The checkout sits on detached HEAD one commit PAST installed: work that no ref protects.
    """
    home, origin, checkout = (tmp_path / name for name in ("home", "origin", "checkout"))
    home.mkdir()
    origin.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_INSTALL_ROOT", str(checkout))
    monkeypatch.delenv("HERMES_MANAGED", raising=False)
    git(origin, "init", "-b", "main")
    git(origin, "config", "user.name", "Diverged Fixture")
    git(origin, "config", "user.email", "fixture@example.invalid")
    git(origin, "config", "commit.gpgsign", "false")
    (origin / "hermes_state_common.py").write_text(_PRE)
    git(origin, "add", ".")
    git(origin, "commit", "-m", "base")
    git(origin, "checkout", "-q", "-b", "release")
    (origin / "notes.txt").write_text("release-only fix\n")
    git(origin, "add", ".")
    git(origin, "commit", "-m", "stable release cut beside main")
    release = git(origin, "rev-parse", "HEAD")
    git(origin, "checkout", "-q", "main")
    (origin / "hermes_state_common.py").write_text(
        "SCHEMA_SQL = '''CREATE TABLE IF NOT EXISTS session_admissions (seq INTEGER)'''\n")
    git(origin, "commit", "-am", "gateway runtime")
    installed = git(origin, "rev-parse", "HEAD")
    git(tmp_path, "clone", "-q", str(origin), str(checkout))
    git(checkout, "checkout", "-q", "--detach", installed)
    # The incident shape: local work on a detached checkout, pushed nowhere, protected by nothing.
    (checkout / "local_work.txt").write_text("unpushed local commit\n")
    git(checkout, "add", "local_work.txt")
    git(checkout, "commit", "-qm", "local work only this checkout has")
    local = git(checkout, "rev-parse", "HEAD")
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
    monkeypatch.setattr(update_cmd, "_pause_windows_gateways_for_update", lambda: None)
    monkeypatch.setattr(update_cmd, "_prepare_git_command", lambda **_: (False, ["git"], False))
    completed = []
    monkeypatch.setattr(update_cmd, "_complete_source_update",
                        lambda request: completed.append(deepcopy(request)))
    monkeypatch.setattr(source_releases, "_resolve_stable",
                        lambda repository, *a, **k: source_releases.SourceTarget(
                            "stable", "stable", repository, commit=release, version="0.22.2"))
    from hermes_cli.update_channel import set_install_channel
    set_install_channel("stable", checkout)
    return SimpleNamespace(home=home, root=checkout, release=release, installed=installed,
                           local=local, parser=parser, completed=completed)


def test_a_release_move_over_unprotected_local_work_is_refused(diverged, monkeypatch, capsys):
    with pytest.raises(SystemExit) as stopped:
        update_cmd._cmd_update_impl(diverged.parser.parse_args(["update", "--yes"]), False)
    out = capsys.readouterr().out
    assert stopped.value.code == 1
    assert git(diverged.root, "rev-parse", "HEAD") == diverged.local  # nothing moved
    assert "local commit" in out and "does not contain" in out
    assert "No update was applied" in out
    assert diverged.completed == []


def test_an_explicit_channel_move_warns_and_proceeds(diverged, monkeypatch, capsys):
    update_cmd._cmd_update_impl(
        diverged.parser.parse_args(["update", "--yes", "--set-channel", "stable"]), False)
    out = capsys.readouterr().out
    assert "Moving anyway" in out and "local commit" in out
    assert git(diverged.root, "rev-parse", "HEAD") == diverged.release
    # The orphaned work stays reachable: the parked-detached rescue keeps it by name.
    assert git(diverged.root, "cat-file", "-t", diverged.local) == "commit"


def test_a_diverged_release_move_is_refused_even_when_a_ref_protects_the_commits(
        diverged, monkeypatch, capsys):
    # The policy is ancestry, not loss-prevention: main carries the runtime commit and the
    # release cuts beside it, so a release move regresses the running tree even though a
    # branch keeps every commit reachable. An unchosen channel means the release is held.
    git(diverged.root, "branch", "local-work", diverged.local)
    git(diverged.root, "checkout", "-q", "--detach", diverged.installed)
    with pytest.raises(SystemExit) as stopped:
        update_cmd._cmd_update_impl(diverged.parser.parse_args(["update", "--yes"]), False)
    out = capsys.readouterr().out
    assert stopped.value.code == 1
    assert git(diverged.root, "rev-parse", "HEAD") == diverged.installed
    assert "does not contain" in out and "No update was applied" in out


def test_check_reports_held_instead_of_advertising(diverged, monkeypatch, capsys):
    update_cmd._cmd_update_check()
    out = capsys.readouterr().out
    assert "Held" in out and "does not contain" in out
    assert "Selected release available" not in out
