from __future__ import annotations

import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock

import pytest

from hermes_cli import plugins_cmd as pc
from hermes_cli.plugin_supply_chain import (
    LOCK_FILENAME,
    PluginProvenance,
    read_provenance_lock,
    write_provenance_lock,
)


def _provenance(
    *, requested_ref=None, commit="1" * 40,
    source="https://example.invalid/owner/repo.git", subdir=None
):
    return PluginProvenance(source, subdir, commit, requested_ref, "2026-01-01T00:00:00+00:00")


def _installed(tmp_path: Path, monkeypatch) -> Path:
    plugins = tmp_path / "plugins"
    target = plugins / "demo"
    (target / ".git").mkdir(parents=True)
    monkeypatch.setattr(pc, "_plugins_dir", lambda: plugins)
    return target


def test_pinned_cli_refuses_without_pull_and_preserves_target(tmp_path, monkeypatch, capsys):
    target = _installed(tmp_path, monkeypatch)
    (target / "payload").write_bytes(b"old")
    pin = "a" * 40
    write_provenance_lock(target, _provenance(requested_ref=pin))
    before_lock = (target / LOCK_FILENAME).read_bytes()
    pull = Mock()
    monkeypatch.setattr(pc, "_git_pull_plugin_dir", pull)

    with pytest.raises(SystemExit) as exc:
        pc.cmd_update("demo")

    assert exc.value.code == 1
    pull.assert_not_called()
    assert (target / "payload").read_bytes() == b"old"
    assert (target / LOCK_FILENAME).read_bytes() == before_lock
    out = capsys.readouterr().out
    compact = "".join(out.split())
    assert (
        "hermespluginsinstallhttps://example.invalid/owner/repo.git"
        "--ref<40-char-sha>--force"
    ) in compact


def test_pinned_dashboard_refuses_without_pull(tmp_path, monkeypatch):
    target = _installed(tmp_path, monkeypatch)
    write_provenance_lock(target, _provenance(requested_ref="a" * 40))
    before = (target / LOCK_FILENAME).read_bytes()
    pull = Mock()
    monkeypatch.setattr(pc, "_git_pull_plugin_dir", pull)

    result = pc.dashboard_update_user_plugin("demo")

    assert result["ok"] is False
    assert "https://example.invalid/owner/repo.git" in result["error"]
    pull.assert_not_called()
    assert (target / LOCK_FILENAME).read_bytes() == before


@pytest.mark.parametrize("surface", ["cli", "dashboard"])
def test_pinned_subdir_without_git_refuses_from_lock_before_checkout_check(
    tmp_path, monkeypatch, capsys, surface
):
    target = _installed(tmp_path, monkeypatch)
    (target / ".git").rmdir()
    source = f"file://{tmp_path}/source repo"
    write_provenance_lock(
        target,
        _provenance(requested_ref="a" * 40, source=source, subdir="plugins/demo"),
    )
    pull = Mock()
    monkeypatch.setattr(pc, "_git_pull_plugin_dir", pull)

    if surface == "cli":
        with pytest.raises(SystemExit):
            pc.cmd_update("demo")
        message = capsys.readouterr().out
    else:
        result = pc.dashboard_update_user_plugin("demo")
        assert result["ok"] is False
        message = result["error"]

    assert "Pinned plugins cannot be updated in place" in message
    compact = "".join(message.split())
    assert f"'{''.join(source.split())}#plugins/demo'" in compact
    assert "--ref<40-char-sha>--force" in compact
    assert "no .git" not in message
    assert "not a git checkout" not in message
    pull.assert_not_called()


@pytest.mark.parametrize("kind", ["malformed", "symlink", "dangling"])
def test_unsafe_lock_fails_closed_before_git(tmp_path, monkeypatch, kind, capsys):
    target = _installed(tmp_path, monkeypatch)
    lock = target / LOCK_FILENAME
    if kind == "malformed":
        lock.write_text("secret-token-not-json")
    else:
        lock.symlink_to(target / ("missing" if kind == "dangling" else "payload"))
        if kind == "symlink":
            (target / "payload").write_text("{}")
    pull = Mock()
    monkeypatch.setattr(pc, "_git_pull_plugin_dir", pull)

    with pytest.raises(SystemExit):
        pc.cmd_update("demo")

    pull.assert_not_called()
    assert "secret-token" not in capsys.readouterr().out


def test_legacy_no_lock_update_still_pulls_without_creating_lock(tmp_path, monkeypatch):
    target = _installed(tmp_path, monkeypatch)
    monkeypatch.setattr(pc, "_git_pull_plugin_dir", lambda _: (True, "Already up to date."))
    monkeypatch.setattr(pc, "_copy_example_files", lambda *_: None)

    pc.cmd_update("demo")

    assert not (target / LOCK_FILENAME).exists()


def test_pull_uses_sanitized_git_environment_and_hides_stderr(tmp_path, monkeypatch):
    target = tmp_path / "demo"
    target.mkdir()
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/tmp/evil")
    monkeypatch.setenv("GIT_ASKPASS", "steal")
    run = Mock(return_value=Mock(returncode=1, stdout="", stderr="token=raw-secret"))
    monkeypatch.setattr(pc.subprocess, "run", run)
    monkeypatch.setattr(pc, "_resolve_git_executable", lambda: "/usr/bin/git")

    ok, message = pc._git_pull_plugin_dir(target)

    assert ok is False
    assert message == "Git pull failed."
    kwargs = run.call_args.kwargs
    assert run.call_args.args[0][:3] == ["/usr/bin/git", "-c", f"core.hooksPath={os.devnull}"]
    assert kwargs["env"]["GIT_CONFIG_GLOBAL"] == os.devnull
    assert "GIT_ASKPASS" not in kwargs["env"]
    assert kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0"


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        ("\x1b[31mAlready up to date. raw-secret\x1b[0m", "Already up to date."),
        ("\x1b[31mupdated raw-secret\x1b[0m", "Updated."),
    ],
)
def test_pull_success_maps_raw_output_to_fixed_message(
    tmp_path, monkeypatch, stdout, expected
):
    target = tmp_path / "demo"
    target.mkdir()
    run = Mock(return_value=Mock(returncode=0, stdout=stdout, stderr=""))
    monkeypatch.setattr(pc.subprocess, "run", run)
    monkeypatch.setattr(pc, "_resolve_git_executable", lambda: "/usr/bin/git")

    ok, message = pc._git_pull_plugin_dir(target)

    assert ok is True
    assert message == expected
    assert "raw-secret" not in message
    assert "\x1b" not in message


def test_cli_success_does_not_print_raw_git_output(tmp_path, monkeypatch, capsys):
    _installed(tmp_path, monkeypatch)
    monkeypatch.setattr(pc, "_copy_example_files", lambda *_: None)
    monkeypatch.setattr(pc, "_resolve_git_executable", lambda: "/usr/bin/git")
    monkeypatch.setattr(
        pc.subprocess,
        "run",
        Mock(return_value=Mock(returncode=0, stdout="\x1b[31mraw-secret\x1b[0m", stderr="")),
    )

    pc.cmd_update("demo")

    output = capsys.readouterr().out
    assert "raw-secret" not in output
    assert "\x1b" not in output
    assert "Plugin demo updated." in output
    assert "Updated." in output


def test_unpinned_update_refreshes_lock_to_real_remote_head(tmp_path, monkeypatch):
    remote = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    clone = tmp_path / "plugins" / "demo"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    subprocess.run(["git", "clone", str(remote), str(seed)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(seed), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(seed), "config", "user.name", "Test"], check=True)
    (seed / "plugin.yaml").write_text("name: demo\n")
    subprocess.run(["git", "-C", str(seed), "add", "."], check=True)
    subprocess.run(["git", "-C", str(seed), "commit", "-m", "one"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(seed), "push", "origin", "HEAD"], check=True, capture_output=True)
    subprocess.run(["git", "clone", str(remote), str(clone)], check=True, capture_output=True)
    old = subprocess.run(["git", "-C", str(clone), "rev-parse", "HEAD"], check=True, text=True, capture_output=True).stdout.strip()
    write_provenance_lock(clone, _provenance(commit=old, source=f"file://{remote}"))
    (seed / "plugin.yaml").write_text("name: demo\nversion: 2\n")
    subprocess.run(["git", "-C", str(seed), "commit", "-am", "two"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(seed), "push"], check=True, capture_output=True)
    expected = subprocess.run(["git", "-C", str(seed), "rev-parse", "HEAD"], check=True, text=True, capture_output=True).stdout.strip()
    monkeypatch.setattr(pc, "_plugins_dir", lambda: tmp_path / "plugins")
    monkeypatch.setattr(pc, "_copy_example_files", lambda *_: None)

    pc.cmd_update("demo")

    lock = read_provenance_lock(clone)
    assert lock is not None
    assert lock.requested_ref is None
    assert lock.resolved_commit == expected
    assert lock.source_url == f"file://{remote}"


def test_lock_refresh_failure_is_partial_for_cli_and_dashboard(tmp_path, monkeypatch, capsys):
    target = _installed(tmp_path, monkeypatch)
    write_provenance_lock(target, _provenance())
    monkeypatch.setattr(pc, "_git_pull_plugin_dir", lambda _: (True, "updated"))
    monkeypatch.setattr(pc, "_refresh_update_provenance", lambda *_: (_ for _ in ()).throw(RuntimeError("raw-secret")))
    copy = Mock()
    monkeypatch.setattr(pc, "_copy_example_files", copy)

    with pytest.raises(SystemExit) as exc:
        pc.cmd_update("demo")
    assert exc.value.code == 1
    output = capsys.readouterr().out
    assert "Plugin updated, but provenance lock refresh failed." in output
    assert "raw-secret" not in output
    copy.assert_not_called()

    result = pc.dashboard_update_user_plugin("demo")
    assert result == {"ok": False, "updated": True, "error": "Plugin updated, but provenance lock refresh failed.", "name": "demo"}
    copy.assert_not_called()


@pytest.mark.parametrize("surface", ["cli", "dashboard"])
def test_update_rechecks_pin_only_after_operation_lock_is_acquired(
    tmp_path, monkeypatch, capsys, surface
):
    target = _installed(tmp_path, monkeypatch)
    write_provenance_lock(target, _provenance(requested_ref=None))
    pull = Mock()
    monkeypatch.setattr(pc, "_git_pull_plugin_dir", pull)

    @contextmanager
    def pin_before_yield(locked_target):
        assert locked_target == target
        write_provenance_lock(target, _provenance(requested_ref="a" * 40))
        yield

    monkeypatch.setattr(pc, "_plugin_operation_lock", pin_before_yield)

    if surface == "cli":
        with pytest.raises(SystemExit):
            pc.cmd_update("demo")
        assert "Pinned plugins cannot be updated" in capsys.readouterr().out
    else:
        result = pc.dashboard_update_user_plugin("demo")
        assert result["ok"] is False
        assert "Pinned plugins cannot be updated" in result["error"]
    pull.assert_not_called()


@pytest.mark.skipif(os.name != "posix" or not hasattr(os, "link"), reason="hardlinks require POSIX")
def test_update_rejects_hardlinked_provenance_before_git(tmp_path, monkeypatch, capsys):
    target = _installed(tmp_path, monkeypatch)
    lock_path = write_provenance_lock(target, _provenance())
    os.link(lock_path, target / "second-link")
    pull = Mock()
    monkeypatch.setattr(pc, "_git_pull_plugin_dir", pull)

    with pytest.raises(SystemExit):
        pc.cmd_update("demo")

    pull.assert_not_called()
    assert "malformed or unreadable" in capsys.readouterr().out
