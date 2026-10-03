"""Repo-declared inert launchd jobs under the lifecycle guard (#129504)."""

from __future__ import annotations

import os
import plistlib
import subprocess
from pathlib import Path

import pytest

from cron import lifecycle_guard
from cron.lifecycle_guard import (
    contains_launchctl_submit_command,
    contains_gateway_lifecycle_command_or_referenced_script,
    scan_gateway_lifecycle,
)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=str(repo),
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=10,
    )


def _init_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    # Ensure commits succeed without GPG/identity prompts.
    _git(repo, "commit", "--allow-empty", "-qm", "init")


def _write_plist(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        plistlib.dump(payload, fh)


def _track(repo: Path, path: Path) -> None:
    rel = str(path.relative_to(repo))
    _git(repo, "add", "--", rel)
    _git(repo, "commit", "-qm", f"add {rel}")


def _inert_payload(label: str) -> dict:
    return {"Label": label, "ProgramArguments": ["/bin/echo", "hello"]}


def test_allowed_tracked_clean_inert_bootstrap(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    plist = repo / "com.example.inert.plist"
    _write_plist(plist, _inert_payload("com.example.inert"))
    _track(repo, plist)
    cmd = f"launchctl bootstrap gui/501 {plist}"
    assert contains_launchctl_submit_command(cmd, cwd=str(repo)) is False
    unsafe, refusal = scan_gateway_lifecycle(cmd, cwd=str(repo))
    assert unsafe is False
    assert refusal is None
    assert contains_gateway_lifecycle_command_or_referenced_script(cmd, cwd=str(repo)) is False


def test_allowed_runatload_false(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    plist = repo / "com.example.noreload.plist"
    _write_plist(
        plist,
        {"Label": "com.example.noreload", "ProgramArguments": ["/bin/echo", "hi"], "RunAtLoad": False},
    )
    _track(repo, plist)
    cmd = f"launchctl bootstrap gui/501 {plist}"
    assert contains_launchctl_submit_command(cmd, cwd=str(repo)) is False
    assert scan_gateway_lifecycle(cmd, cwd=str(repo))[0] is False


def test_allowed_relative_plist_path(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    plist = repo / "com.example.rel.plist"
    _write_plist(plist, _inert_payload("com.example.rel"))
    _track(repo, plist)
    cmd = "launchctl bootstrap gui/501 com.example.rel.plist"
    assert contains_launchctl_submit_command(cmd, cwd=str(repo)) is False
    assert scan_gateway_lifecycle(cmd, cwd=str(repo))[0] is False


def test_blocked_target_outside_workdir_tmp(tmp_path):
    repo = tmp_path / "repo"
    outside = tmp_path / "outside"
    _init_repo(repo)
    outside.mkdir(parents=True, exist_ok=True)
    plist = outside / "com.example.out.plist"
    _write_plist(plist, _inert_payload("com.example.out"))
    cmd = f"launchctl bootstrap gui/501 {plist}"
    assert contains_launchctl_submit_command(cmd, cwd=str(repo)) is True
    assert scan_gateway_lifecycle(cmd, cwd=str(repo))[0] is True


def test_blocked_target_outside_workdir_library(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    cmd = f"launchctl bootstrap gui/501 {os.path.expanduser('~/Library/LaunchAgents/com.example.out.plist')}"
    assert contains_launchctl_submit_command(cmd, cwd=str(repo)) is True
    assert scan_gateway_lifecycle(cmd, cwd=str(repo))[0] is True


def test_blocked_untracked(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    plist = repo / "com.example.new.plist"
    _write_plist(plist, _inert_payload("com.example.new"))
    cmd = f"launchctl bootstrap gui/501 {plist}"
    assert contains_launchctl_submit_command(cmd, cwd=str(repo)) is True
    assert scan_gateway_lifecycle(cmd, cwd=str(repo))[0] is True


def test_blocked_modified(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    plist = repo / "com.example.mod.plist"
    _write_plist(plist, _inert_payload("com.example.mod"))
    _track(repo, plist)
    _write_plist(
        plist, {"Label": "com.example.mod", "ProgramArguments": ["/bin/echo", "changed"]}
    )
    cmd = f"launchctl bootstrap gui/501 {plist}"
    assert contains_launchctl_submit_command(cmd, cwd=str(repo)) is True
    assert scan_gateway_lifecycle(cmd, cwd=str(repo))[0] is True


def test_blocked_keepalive_true(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    plist = repo / "com.example.keep.plist"
    _write_plist(
        plist,
        {"Label": "com.example.keep", "ProgramArguments": ["/bin/echo", "hi"], "KeepAlive": True},
    )
    _track(repo, plist)
    cmd = f"launchctl bootstrap gui/501 {plist}"
    assert contains_launchctl_submit_command(cmd, cwd=str(repo)) is True


def test_blocked_keepalive_false_still_blocked(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    plist = repo / "com.example.keepfalse.plist"
    _write_plist(
        plist,
        {
            "Label": "com.example.keepfalse",
            "ProgramArguments": ["/bin/echo", "hi"],
            "KeepAlive": False,
        },
    )
    _track(repo, plist)
    cmd = f"launchctl bootstrap gui/501 {plist}"
    assert contains_launchctl_submit_command(cmd, cwd=str(repo)) is True


def test_blocked_runatload_true(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    plist = repo / "com.example.reload.plist"
    _write_plist(
        plist,
        {
            "Label": "com.example.reload",
            "ProgramArguments": ["/bin/echo", "hi"],
            "RunAtLoad": True,
        },
    )
    _track(repo, plist)
    cmd = f"launchctl bootstrap gui/501 {plist}"
    assert contains_launchctl_submit_command(cmd, cwd=str(repo)) is True


def test_blocked_label_mismatch(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    plist = repo / "com.example.job.plist"
    _write_plist(plist, _inert_payload("com.example.other"))
    _track(repo, plist)
    cmd = f"launchctl bootstrap gui/501 {plist}"
    assert contains_launchctl_submit_command(cmd, cwd=str(repo)) is True


def test_blocked_gateway_label(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    plist = repo / "ai.hermes.gateway.plist"
    _write_plist(plist, _inert_payload("ai.hermes.gateway"))
    _track(repo, plist)
    cmd = f"launchctl bootstrap gui/501 {plist}"
    assert contains_launchctl_submit_command(cmd, cwd=str(repo)) is True


def test_blocked_gateway_in_program_arguments(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    plist = repo / "com.example.args.plist"
    _write_plist(
        plist,
        {"Label": "com.example.args", "ProgramArguments": ["/bin/echo", "hermes-gateway"]},
    )
    _track(repo, plist)
    cmd = f"launchctl bootstrap gui/501 {plist}"
    assert contains_launchctl_submit_command(cmd, cwd=str(repo)) is True


def test_blocked_xpc_service_name(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _init_repo(repo)
    plist = repo / "com.example.xpc.plist"
    _write_plist(plist, _inert_payload("com.example.xpc"))
    _track(repo, plist)
    monkeypatch.setenv("XPC_SERVICE_NAME", "com.example.xpc")
    cmd = f"launchctl bootstrap gui/501 {plist}"
    assert contains_launchctl_submit_command(cmd, cwd=str(repo)) is True


def test_blocked_referenced_script_contains_restart(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    helper = repo / "helper.sh"
    helper.write_text("#!/bin/sh\nhermes gateway restart\n", encoding="utf-8")
    plist = repo / "com.example.declared.plist"
    _write_plist(
        plist,
        {"Label": "com.example.declared", "ProgramArguments": ["/bin/sh", "helper.sh"]},
    )
    _track(repo, plist)
    cmd = f"launchctl bootstrap gui/501 {plist}"
    # Static plist shape is inert, so the direct submit check allows it ...
    assert contains_launchctl_submit_command(cmd, cwd=str(repo)) is False
    # ... but the referenced-script walk still blocks the restart inside helper.sh.
    assert scan_gateway_lifecycle(cmd, cwd=str(repo))[0] is True
    assert contains_gateway_lifecycle_command_or_referenced_script(cmd, cwd=str(repo)) is True


def test_blocked_submit_stays_blocked(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    assert (
        contains_launchctl_submit_command(
            "launchctl submit -l com.example.helper -- /usr/bin/true", cwd=str(repo)
        )
        is True
    )
    assert (
        contains_launchctl_submit_command(
            "sudo launchctl submit -l com.example.helper -- /usr/bin/true", cwd=str(repo)
        )
        is True
    )


def test_blocked_unparseable_plist(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    plist = repo / "com.example.bad.plist"
    plist.write_bytes(b"not a plist \x00\x01")
    _track(repo, plist)
    cmd = f"launchctl bootstrap gui/501 {plist}"
    assert contains_launchctl_submit_command(cmd, cwd=str(repo)) is True


def test_blocked_bootstrap_without_cwd(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    plist = repo / "com.example.nocwd.plist"
    _write_plist(plist, _inert_payload("com.example.nocwd"))
    _track(repo, plist)
    cmd = f"launchctl bootstrap gui/501 {plist}"
    assert contains_launchctl_submit_command(cmd, cwd=None) is True


def test_blocked_wrapped_bootstrap_outside_stays_blocked(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    cmd = "sudo launchctl bootstrap gui/501 /tmp/com.example.out.plist"
    assert contains_launchctl_submit_command(cmd, cwd=str(repo)) is True


def test_allowed_wrapped_bootstrap_declared(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    plist = repo / "com.example.wrap.plist"
    _write_plist(plist, _inert_payload("com.example.wrap"))
    _track(repo, plist)
    cmd = f"sudo launchctl bootstrap gui/501 {plist}"
    assert contains_launchctl_submit_command(cmd, cwd=str(repo)) is False
    assert scan_gateway_lifecycle(cmd, cwd=str(repo))[0] is False


def test_blocked_split_hermes_gateway_in_program_arguments(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    plist = repo / "com.example.split.plist"
    _write_plist(
        plist,
        {"Label": "com.example.split", "ProgramArguments": ["hermes", "gateway", "restart"]},
    )
    _track(repo, plist)
    cmd = f"launchctl bootstrap gui/501 {plist}"
    assert contains_launchctl_submit_command(cmd, cwd=str(repo)) is True


def _payload_plist(label: str) -> dict:
    return {
        "Label": label,
        "ProgramArguments": ["/bin/echo", "hermes-gateway"],
        "KeepAlive": True,
        "RunAtLoad": True,
    }


def test_blocked_multi_path_payload_first_decoy_last(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    payload = repo / "com.example.payload.plist"
    _write_plist(payload, _payload_plist("com.example.payload"))
    decoy = repo / "com.example.decoy.plist"
    _write_plist(decoy, _inert_payload("com.example.decoy"))
    _track(repo, payload)
    _track(repo, decoy)
    cmd = f"launchctl bootstrap gui/501 {payload} {decoy}"
    assert contains_launchctl_submit_command(cmd, cwd=str(repo)) is True
    assert scan_gateway_lifecycle(cmd, cwd=str(repo))[0] is True


def test_blocked_multi_path_decoy_first_payload_last(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    payload = repo / "com.example.payload.plist"
    _write_plist(payload, _payload_plist("com.example.payload"))
    decoy = repo / "com.example.decoy.plist"
    _write_plist(decoy, _inert_payload("com.example.decoy"))
    _track(repo, payload)
    _track(repo, decoy)
    cmd = f"launchctl bootstrap gui/501 {decoy} {payload}"
    assert contains_launchctl_submit_command(cmd, cwd=str(repo)) is True
    assert scan_gateway_lifecycle(cmd, cwd=str(repo))[0] is True


def test_allowed_multi_path_all_inert(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    first = repo / "com.example.first.plist"
    _write_plist(first, _inert_payload("com.example.first"))
    second = repo / "com.example.second.plist"
    _write_plist(second, _inert_payload("com.example.second"))
    _track(repo, first)
    _track(repo, second)
    cmd = f"launchctl bootstrap gui/501 {first} {second}"
    assert contains_launchctl_submit_command(cmd, cwd=str(repo)) is False
    assert scan_gateway_lifecycle(cmd, cwd=str(repo))[0] is False


def test_blocked_multi_path_first_plist_helper_restart(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    helper = repo / "helper.sh"
    helper.write_text("#!/bin/sh\nhermes gateway restart\n", encoding="utf-8")
    first = repo / "com.example.first.plist"
    _write_plist(
        first,
        {"Label": "com.example.first", "ProgramArguments": ["/bin/sh", "helper.sh"]},
    )
    second = repo / "com.example.second.plist"
    _write_plist(second, _inert_payload("com.example.second"))
    _track(repo, first)
    _track(repo, second)
    cmd = f"launchctl bootstrap gui/501 {first} {second}"
    assert contains_launchctl_submit_command(cmd, cwd=str(repo)) is False
    assert scan_gateway_lifecycle(cmd, cwd=str(repo))[0] is True
