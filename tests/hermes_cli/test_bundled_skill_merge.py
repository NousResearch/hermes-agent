"""Updater maintenance and manual resolution use the real sync implementation."""
import argparse

import pytest

from hermes_cli.subcommands.skills import build_skills_parser
from tools import skills_sync as ss
from tools import skills_sync_merge as merge


def test_update_merges_then_reports_conflict_and_cli_resolves(tmp_path, monkeypatch, capsys):
    from hermes_cli.update_cmd_maint import _print_bundled_skills_sync_report
    from hermes_cli.skills_hub import skills_command

    home = tmp_path / "home"
    source = tmp_path / "bundle" / "example"
    source.mkdir(parents=True)
    base = b"---\nname: example\n---\nfirst\nsecond\nthird\nfourth\nfifth\n"
    (source / "SKILL.md").write_bytes(base)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(ss, "_get_bundled_dir", lambda: source.parent)
    monkeypatch.setattr(ss, "_get_optional_dir", lambda: tmp_path / "absent")
    ss.sync_skills(quiet=True)
    local = home / "skills" / "example" / "SKILL.md"
    local.write_bytes(base.replace(b"first", b"mine"))
    upstream = base.replace(b"fifth", b"new fifth")
    (source / "SKILL.md").write_bytes(upstream)
    _print_bundled_skills_sync_report()
    assert local.read_bytes() == upstream.replace(b"first", b"mine")
    (source / "SKILL.md").write_bytes(upstream.replace(b"first", b"theirs"))
    _print_bundled_skills_sync_report()
    assert "hermes skills resolve example" in capsys.readouterr().out
    parser = argparse.ArgumentParser()
    build_skills_parser(parser.add_subparsers(), cmd_skills=skills_command)
    args = parser.parse_args(["skills", "resolve", "example", "--accept", "upstream"])
    assert args.func(args) is None
    assert local.read_bytes() == (source / "SKILL.md").read_bytes()
    assert not (merge.state_dir(home, "example") / "pending").exists()


def test_resolution_without_accept_is_review_only_when_noninteractive(tmp_path, monkeypatch, capsys):
    from hermes_cli.skills_hub import skills_command

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    parser = argparse.ArgumentParser()
    build_skills_parser(parser.add_subparsers(), cmd_skills=skills_command)
    args = parser.parse_args(["skills", "resolve", "example"])
    assert args.func(args) is None
    assert "No pending" in capsys.readouterr().out


@pytest.mark.parametrize("stdin_tty,stdout_tty,choice", [
    (True, True, None),
    (True, True, 0),
    (True, True, 1),
    (True, True, 2),
    (False, True, None),
    (True, False, None),
    (False, False, None),
])
@pytest.mark.parametrize("interactive", [True, False])
def test_update_time_conflict_prompt_requires_explicit_acceptance(
    tmp_path, monkeypatch, capsys, stdin_tty, stdout_tty, choice, interactive,
):
    from unittest.mock import Mock

    from hermes_cli.update_cmd_maint import _print_bundled_skills_sync_report

    home = tmp_path / "home"
    source = tmp_path / "bundle" / "example"
    source.mkdir(parents=True)
    base = b"---\nname: example\n---\nfirst\nsecond\nthird\nfourth\nfifth\n"
    (source / "SKILL.md").write_bytes(base)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(ss, "_get_bundled_dir", lambda: source.parent)
    monkeypatch.setattr(ss, "_get_optional_dir", lambda: tmp_path / "absent")
    ss.sync_skills(quiet=True)
    local = home / "skills" / "example" / "SKILL.md"
    ours = base.replace(b"first", b"mine")
    theirs = base.replace(b"first", b"theirs")
    local.write_bytes(ours)
    (source / "SKILL.md").write_bytes(theirs)
    origin = ss._read_manifest()["example"]
    monkeypatch.setattr("sys.stdin.isatty", lambda: stdin_tty)
    monkeypatch.setattr("sys.stdout.isatty", lambda: stdout_tty)
    menu = Mock(return_value=choice)
    monkeypatch.setattr("hermes_cli.curses_ui.curses_single_select", menu)

    _print_bundled_skills_sync_report(interactive=interactive)

    interactive = interactive and stdin_tty and stdout_tty
    assert menu.call_count == int(interactive)
    if interactive:
        assert menu.call_args.kwargs["default_index"] == len(menu.call_args.args[1])
        assert menu.call_args.kwargs["cancel_label"] == "Leave pending"
    accepted = interactive and choice is not None
    assert local.read_bytes() == (theirs if accepted and choice == 2 else ours)
    pending = merge.state_dir(home, "example") / "pending"
    assert pending.exists() is not accepted
    assert ss._read_manifest()["example"] == (ss.dir_hash(source) if accepted else origin)
    if not accepted:
        assert "hermes skills resolve example" in capsys.readouterr().out


@pytest.mark.parametrize("changed", ["local", "baseline"])
@pytest.mark.parametrize("accept", ["local", "merged", "upstream"])
def test_stale_review_reports_refresh_and_keep_local_preserves_latest_edits(
    tmp_path, monkeypatch, capsys, changed, accept,
):
    from hermes_cli.skills_merge import do_resolve

    home = tmp_path / "home"
    source = tmp_path / "bundle" / "example"
    source.mkdir(parents=True)
    base = b"---\nname: example\n---\nbase\n"
    (source / "SKILL.md").write_bytes(base)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(ss, "_get_bundled_dir", lambda: source.parent)
    monkeypatch.setattr(ss, "_get_optional_dir", lambda: tmp_path / "absent")
    ss.sync_skills(quiet=True)
    local = home / "skills" / "example" / "SKILL.md"
    ours = base.replace(b"base", b"mine")
    local.write_bytes(ours)
    (source / "SKILL.md").write_bytes(base.replace(b"base", b"theirs"))
    ss.sync_skills(quiet=True)
    pending = merge.state_dir(home, "example") / "pending"
    draft = b"---\nname: example\n---\nreviewed draft\n"
    (pending / "merged" / "SKILL.md").write_bytes(draft)
    latest = ours + b"latest local edit\n" if changed == "local" else ours
    local.write_bytes(latest)
    if changed == "baseline":
        manifest = ss._read_manifest()
        manifest["example"] = "changed-baseline"
        ss._write_manifest(manifest)

    assert do_resolve("example", accept=accept) is False
    assert "hermes skills resolve example --refresh" in capsys.readouterr().out
    assert local.read_bytes() == latest
    assert (pending / "merged" / "SKILL.md").read_bytes() == draft
    assert do_resolve("example", refresh=True, accept="local") is True
    assert local.read_bytes() == latest
    assert not pending.exists()
    archived = list(pending.parent.glob("superseded-*"))
    assert len(archived) == 1
    assert (archived[0] / "merged" / "SKILL.md").read_bytes() == draft
    for _ in range(2):
        ss.sync_skills(quiet=True)
        assert ss._read_manifest()["example"] == ss.dir_hash(source)
        assert not pending.exists()
        assert local.read_bytes() == latest


def test_refresh_flag_can_be_parsed_without_accepting_a_tree():
    parser = argparse.ArgumentParser()
    build_skills_parser(parser.add_subparsers(), cmd_skills=lambda args: None)
    args = parser.parse_args(["skills", "resolve", "example", "--refresh"])
    assert args.refresh is True
    assert args.accept is None
