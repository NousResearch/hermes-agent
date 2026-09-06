"""Real filesystem permissions for A2A conversation and audit artifacts."""

import json
import os
import stat

import pytest

from plugins.platforms.a2a import protocol, security


def _mode(path):
    return stat.S_IMODE(path.stat().st_mode)


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits are advisory on Windows")
def test_a2a_artifacts_are_private_and_parseable(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_MANAGED", raising=False)
    old_umask = os.umask(0o022)
    try:
        protocol.persist_message("ctx-1", "user", "hello", "task-1")
        security.audit("inbound", "peer", "task-1", "summary")
    finally:
        os.umask(old_umask)

    conv_dir = tmp_path / "a2a_conversations"
    conv = conv_dir / "ctx-1.jsonl"
    audit = tmp_path / "a2a_audit.jsonl"
    assert not _mode(conv_dir) & 0o077
    assert not _mode(conv) & 0o077
    assert not _mode(audit) & 0o077
    assert json.loads(conv.read_text(encoding="utf-8").strip())["text"] == "hello"
    assert json.loads(audit.read_text(encoding="utf-8").strip())["peer"] == "peer"


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits are advisory on Windows")
def test_a2a_managed_artifacts_keep_group_access(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    os.chmod(home, 0o2770)
    probe = home / "probe"
    probe.mkdir()
    if not _mode(probe) & stat.S_ISGID:
        pytest.skip("kernel does not inherit setgid on new directories")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED", "nixos")
    old_umask = os.umask(0o007)
    try:
        protocol.persist_message("ctx-managed", "user", "hello", "task-1")
        security.audit("inbound", "peer", "task-1", "summary")
    finally:
        os.umask(old_umask)
    conv_dir = home / "a2a_conversations"
    assert _mode(conv_dir) & 0o777 == 0o770
    assert _mode(conv_dir / "ctx-managed.jsonl") == 0o660
    assert _mode(home / "a2a_audit.jsonl") == 0o660
