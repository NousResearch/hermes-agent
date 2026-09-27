"""#122691: `hermes update` emits a determinate phase progress sequence.

The update pipeline printed one bare arrow line per phase with no sense of how
much download/install work remained before the app restarted (#122691). These
tests encode the requested behavior: a progress callback fires across the
update phases (fetch -> prepare -> pull -> install) with a monotonic
``[k/N]`` sequence, and byte-level progress is reported when a transport
exposes sizes.
"""

import argparse
import subprocess
from types import SimpleNamespace

import pytest

from hermes_cli import main, update_cmd
from hermes_cli.subcommands.update import build_update_parser


def git(root, *args):
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True,
        text=True, stdin=subprocess.DEVNULL).stdout.strip()


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    """A local git install one commit behind origin/main, ready for `hermes update`."""
    home, origin, root = (tmp_path / name for name in ("home", "origin", "checkout"))
    home.mkdir()
    origin.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_INSTALL_ROOT", str(root))
    monkeypatch.delenv("HERMES_MANAGED", raising=False)
    git(origin, "init", "-b", "main")
    git(origin, "config", "user.name", "Progress Fixture")
    git(origin, "config", "user.email", "fixture@example.invalid")
    git(origin, "config", "commit.gpgsign", "false")
    (origin / "content.txt").write_text("installed", encoding="utf-8")
    git(origin, "add", "content.txt")
    git(origin, "commit", "-m", "installed")
    first = git(origin, "rev-parse", "HEAD")
    (origin / "content.txt").write_text("published", encoding="utf-8")
    git(origin, "add", "content.txt")
    git(origin, "commit", "-m", "published")
    git(tmp_path, "clone", str(origin), str(root))
    # Local main sits one commit behind origin/main so the update has real work.
    git(root, "reset", "--hard", first)
    monkeypatch.setattr(main, "PROJECT_ROOT", root)
    monkeypatch.setattr("hermes_cli.config.get_project_root", lambda: root)
    parser = argparse.ArgumentParser()
    build_update_parser(parser.add_subparsers(), cmd_update=main.cmd_update)
    opts = update_cmd._UpdateOptions(
        pre_update_version=None, gw_input_fn=None, assume_yes=True, keep_stash=False,
        switch_branch=False, discard_local_changes=False)
    monkeypatch.setattr(update_cmd, "_resolve_update_options", lambda *_: opts)
    monkeypatch.setattr(update_cmd, "_begin_update_receipt_and_plan", lambda *_: None)
    monkeypatch.setattr(main, "_run_pre_update_backup", lambda *_: None)
    monkeypatch.setattr(main, "_pause_windows_gateways_for_update", lambda: None)
    monkeypatch.setattr(update_cmd, "_prepare_git_command", lambda: (False, ["git"], False))
    # The install/restart phase runs in a child interpreter; capture the boundary instead.
    monkeypatch.setattr(update_cmd, "_complete_source_update", lambda request: None)
    monkeypatch.setattr(update_cmd, "_write_fleet_restart_pending_marker", lambda **kw: None)
    yield SimpleNamespace(home=home, root=root, parser=parser)


GIT_PHASES = ["Fetch updates", "Prepare checkout", "Pull update", "Install and restart"]


def test_update_emits_phase_progress_sequence(checkout, capsys):
    """A progress callback fires once per update phase, in plan order with `[k/N]`."""
    from hermes_cli import update_progress

    events = []
    update_progress.set_listener(events.append)
    try:
        args = checkout.parser.parse_args(["update", "--yes", "--branch", "main"])
        update_cmd._cmd_update_impl(args, False)
    finally:
        update_progress.set_listener(None)
        update_progress.end()

    phases = [event for event in events if event["kind"] == "phase"]
    assert [event["label"] for event in phases] == GIT_PHASES, \
        "progress callback must fire across fetch/prepare/pull/install phases"
    assert [event["index"] for event in phases] == [1, 2, 3, 4]
    assert {event["total"] for event in phases} == {4}

    out = capsys.readouterr().out
    assert "  [1/4] [#####---------------] Fetch updates" in out, out
    assert "  [4/4] [####################] Install and restart" in out, out
    assert "[2/4]" in out and "[3/4]" in out, out


def test_byte_progress_reports_transport_bytes(capsys):
    """Transports that expose sizes (ZIP urlretrieve) report byte sub-progress."""
    from hermes_cli import update_progress

    events = []
    update_progress.set_listener(events.append)
    try:
        update_progress.begin(("Download and swap update",))
        update_progress.step("Download and swap update")
        update_progress.byte_progress(50, 100)
        update_progress.byte_progress(100, 100)
    finally:
        update_progress.set_listener(None)
        update_progress.end()

    byte_events = [event for event in events if event["kind"] == "bytes"]
    assert [(event["read"], event["size"]) for event in byte_events] == [(50, 100), (100, 100)]
    out = capsys.readouterr().out
    assert "50% (50/100 bytes)" in out, out


def test_byte_progress_ignores_unknown_sizes():
    """A transport without Content-Length reports nothing instead of a bogus bar."""
    from hermes_cli import update_progress

    events = []
    update_progress.set_listener(events.append)
    try:
        update_progress.begin(("Download and swap update",))
        update_progress.step("Download and swap update")
        update_progress.byte_progress(4096, -1)
    finally:
        update_progress.set_listener(None)
        update_progress.end()
    assert [event for event in events if event["kind"] == "bytes"] == []
