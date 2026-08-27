"""Controlled-fixture hostile tests for descriptor-safe governed keys."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import pytest


MODULE = (
    Path(__file__).parents[2] / "plugins/context_engine/_context_governor/key_state.py"
)
SPEC = importlib.util.spec_from_file_location("cg_key_state_v2", MODULE)
assert SPEC and SPEC.loader
key_state = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = key_state
SPEC.loader.exec_module(key_state)


def _fixture_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Use fixed test bytes; Rust key-id execution is never invoked here."""
    values = iter((b"A" * 32, b"B" * 32))
    monkeypatch.setattr(key_state.secrets, "token_bytes", lambda _: next(values))
    names = iter(("a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8"))
    monkeypatch.setattr(key_state.secrets, "token_hex", lambda _: next(names))

    def fixture_id(_self, descriptor: int) -> str:
        os.lseek(descriptor, 0, os.SEEK_SET)
        return hashlib.sha256(os.read(descriptor, 32)).hexdigest()

    monkeypatch.setattr(key_state.ContextGovernorKeyState, "_derive_key_id", fixture_id)
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    state = key_state.ContextGovernorKeyState(home, "fixture-context-governor")
    return state, home


def test_initialize_and_rotation_publish_complete_immutable_snapshots(
    monkeypatch, tmp_path
):
    state, home = _fixture_state(monkeypatch, tmp_path)
    first = state.initialize_first_install()
    first_id = first.key_id
    first.close()
    second = state.rotate()
    assert second.key_id != first_id
    assert {key_id for key_id, _ in second.retired_key_fds} == {first_id}
    second.close()
    root = home / "context-governor/keys"
    current = key_state.json.loads((root / "current.json").read_text())
    snapshot = key_state.json.loads(
        (root / "snapshots" / current["snapshot"]).read_text()
    )
    assert snapshot["active_key_id"] != first_id
    assert snapshot["retired_key_ids"] == [first_id]
    assert (root / "by-id" / f"{first_id}.key").is_file()


def test_symlink_and_hard_link_are_rejected_before_authority_is_returned(
    monkeypatch, tmp_path
):
    state, home = _fixture_state(monkeypatch, tmp_path)
    binding = state.initialize_first_install()
    key_id = binding.key_id
    binding.close()
    root = home / "context-governor/keys"
    key = root / "by-id" / f"{key_id}.key"
    key.unlink()
    key.symlink_to(tmp_path / "outside")
    with pytest.raises(key_state.ContextGovernorKeyError, match="KeySymlinkRejected"):
        state.active_binding()

    key.unlink()
    fixture_key = tmp_path / "fixture-key"
    fixture_key.write_bytes(b"A" * 32)
    os.chmod(fixture_key, 0o600)
    os.link(fixture_key, key)
    with pytest.raises(key_state.ContextGovernorKeyError, match="KeyHardLinkRejected"):
        state.active_binding()


def test_current_metadata_tampering_fails_closed(monkeypatch, tmp_path):
    state, home = _fixture_state(monkeypatch, tmp_path)
    binding = state.initialize_first_install()
    binding.close()
    current = home / "context-governor/keys/current.json"
    current.write_text(
        '{"schema":"AresContextGovernorCurrentKeySnapshotV2","snapshot":"missing.json","snapshot_sha256":"0"}\n'
    )
    os.chmod(current, 0o600)

    with pytest.raises(key_state.ContextGovernorKeyError, match="MissingGovernedKey"):
        state.active_binding()
