"""The spawn log and argv must never carry the endpoint key in the clear (#137804).

The key is stable for the life of the install, so one leaked `# spawn:` line (or one `ps`
snapshot of argv) is a credential at rest. Contract: prefer `--api-key-file`, and whatever
form argv takes, the spawn log line is masked.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.local_runtime import supervisor


def _make_supervisor(tmp_path, monkeypatch, help_text):
    monkeypatch.setattr(supervisor, "runtimes_root", lambda: tmp_path / "runtime")
    executable = tmp_path / "engine" / "llama-server"

    def run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, help_text, "")

    monkeypatch.setattr(supervisor.subprocess, "run", run)
    commands = []

    def fake_spawn(cmd, **kwargs):
        commands.append(cmd)
        return SimpleNamespace(pid=123, poll=lambda: 0), None

    monkeypatch.setattr(supervisor, "spawn_server", fake_spawn)
    sup = supervisor.LlamaServerSupervisor(executable, tmp_path / "models", port=19001)
    monkeypatch.setattr(sup, "_write_state", lambda: None)
    return sup, commands


def _log_text(sup) -> str:
    return Path(sup.log_path).read_text(encoding="utf-8")


def test_spawn_prefers_api_key_file_over_argv(tmp_path, monkeypatch):
    sup, commands = _make_supervisor(tmp_path, monkeypatch,
                                     "--api-key-file FNAME\n--load-mode MODE")
    try:
        sup._spawn()
        cmd = commands[0]
        assert cmd[cmd.index("--api-key-file") + 1] == str(supervisor.api_key_path())
        assert sup.api_key not in cmd
        assert "--api-key" not in cmd
        log = _log_text(sup)
        assert sup.api_key not in log
        assert "--api-key-file" in log
    finally:
        sup.stop()


def test_spawn_masks_fallback_argv_key_in_log(tmp_path, monkeypatch):
    """Older pinned builds predate --api-key-file: argv must keep working, the log must not leak."""
    sup, commands = _make_supervisor(tmp_path, monkeypatch, "--load-mode MODE")
    try:
        sup._spawn()
        cmd = commands[0]
        assert cmd[cmd.index("--api-key") + 1] == sup.api_key
        log = _log_text(sup)
        assert sup.api_key not in log
        assert "'--api-key', '***'" in log
    finally:
        sup.stop()


def test_spawn_falls_back_when_disk_key_drifted(tmp_path, monkeypatch):
    """A key that only lives in memory (persistence failed) has no file to point at."""
    sup, commands = _make_supervisor(tmp_path, monkeypatch,
                                     "--api-key-file FNAME\n--load-mode MODE")
    supervisor.api_key_path().write_text("different-key-entirely-0123456789", encoding="utf-8")
    try:
        sup._spawn()
        cmd = commands[0]
        assert cmd[cmd.index("--api-key") + 1] == sup.api_key
        assert "--api-key-file" not in cmd
        assert sup.api_key not in _log_text(sup)
    finally:
        sup.stop()


def test_mask_argv_secrets_covers_repeats_and_trailing_flag():
    assert supervisor._mask_argv_secrets(
        ["llama-server", "--api-key", "s3cret", "--api-key", "s3cret"]
    ) == ["llama-server", "--api-key", "***", "--api-key", "***"]
    assert supervisor._mask_argv_secrets(["llama-server", "--api-key"]) == \
        ["llama-server", "--api-key"]
    untouched = ["llama-server", "--models-dir", "/home/my user"]
    assert supervisor._mask_argv_secrets(untouched) == untouched


@pytest.mark.skipif(os.name != "posix", reason="permission bits are POSIX")
def test_persisted_api_key_is_owner_only(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, "runtimes_root", lambda: tmp_path / "runtime")
    supervisor.api_key_path().unlink(missing_ok=True)
    supervisor._stable_api_key()
    assert supervisor.api_key_path().stat().st_mode & 0o777 == 0o600
