"""A sibling profile's credential stores get the same read/write denial as the active one's.

The guards used to cover only the active HERMES_HOME and the global root, so from the
default profile ``<root>/profiles/<other>/auth.json`` was readable (#46411) and
``<root>/profiles/<other>/mcp-tokens/`` writable (#37617) through the file tools.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import agent.file_safety as fs

READ_DENIED = (
    "auth.json", ".anthropic_oauth.json", "auth/google_oauth.json", "mcp-tokens/github.json",
    "webhook_subscriptions.json", ".env", "vault/vault.key",
)
WRITE_DENIED = (
    ".env", ".anthropic_oauth.json", "auth/google_oauth.json", "bws_cache.enc.json",
    "cache/bws_cache.enc.json", "mcp-tokens/github.json", "pairing/telegram-approved.json",
    "state.db", "sessions/session_abc.json",
)
WRITABLE_CONTROL_FILES = ("auth.json", "config.yaml", "webhook_subscriptions.json")


@pytest.fixture()
def default_profile_with_sibling(tmp_path, monkeypatch):
    """Session runs in the default profile (HERMES_HOME == root); ``other`` is a sibling."""
    root = tmp_path / "hermes_root"
    sibling = root / "profiles" / "other"
    sibling.mkdir(parents=True)
    monkeypatch.setattr(fs, "_hermes_home_path", lambda: root)
    monkeypatch.setattr(fs, "_hermes_root_path", lambda: root)
    return sibling


def _touch(base: Path, rel: str) -> Path:
    p = base / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('{"access_token": "CROSS_PROFILE_TOKEN_MARKER"}', encoding="utf-8")
    return p


def test_sibling_profile_credential_stores_are_read_and_write_denied(default_profile_with_sibling):
    from hermes_agent.tools.file_tools import read_file_tool

    sibling = default_profile_with_sibling
    for rel in READ_DENIED:
        path = _touch(sibling, rel)
        assert fs.get_read_block_error(str(path)), f"read allowed: {rel}"
        out = read_file_tool(str(path), task_id="sibling-profile-read")
        assert "CROSS_PROFILE_TOKEN_MARKER" not in out, f"read_file leaked {rel}"
    for rel in WRITE_DENIED:
        assert fs.is_write_denied(str(_touch(sibling, rel))), f"write allowed: {rel}"


def test_sibling_profile_ordinary_and_control_files_stay_usable(default_profile_with_sibling):
    sibling = default_profile_with_sibling
    notes = _touch(sibling, "notes.md")
    assert fs.get_read_block_error(str(notes)) is None
    assert fs.is_write_denied(str(notes)) is False
    for rel in WRITABLE_CONTROL_FILES:
        assert fs.is_write_denied(str(_touch(sibling, rel))) is False, f"#45947 regression: {rel}"
