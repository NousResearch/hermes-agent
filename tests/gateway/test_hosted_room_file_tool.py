"""``share_group_file``: what a Bot may hand to its Group Chat, and only during its turn."""

from __future__ import annotations

import base64
import json
import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.hosted_room_artifacts import RoomArtifactOutbox, RoomArtifactScope
from gateway.session_hosted_output import hosted_output_scope
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools.hosted_room_artifact import share_group_file

UNSAFE = "That file cannot be shared. Move it to the workspace or a Hermes media folder and try again."


def _scope(**overrides) -> RoomArtifactScope:
    value = {"room_id": "room-1", "task_id": "dtask:abc", "execution_generation": 1,
             "member_id": "member-build", "target_profile": "build", "home_install_id": "install-home",
             "target_install_id": "install-home", "authority_gateway_id": "install-home", "authority_epoch": 1}
    value.update(overrides)
    return RoomArtifactScope.from_mapping(value)


class _Turn:
    """A live turn binding over a real outbox, without the owner runtime around it."""

    def __init__(self, db: Path, scope: RoomArtifactScope):
        self.db, self.scope = db, scope
        self.active, self.used, self.owner_pid = True, False, os.getpid()

    def outbox(self):
        self.used = True
        return RoomArtifactOutbox(self.db)


def _share(home: Path, path, *, scope=None, **kwargs):
    token = set_hermes_home_override(home)
    try:
        with hosted_output_scope(_Turn(home / "state.db", scope or _scope())):
            return json.loads(share_group_file(str(path), **kwargs))
    finally:
        reset_hermes_home_override(token)


def test_outside_a_group_chat_turn_the_tool_refuses(tmp_path: Path):
    path = tmp_path / "review.md"
    path.write_text("Review this.\n", encoding="utf-8")
    assert json.loads(share_group_file(str(path))) == {
        "ok": False, "error": "File sharing is available only during a Group Chat turn."}
    turn = _Turn(tmp_path / "state.db", _scope())
    with hosted_output_scope(turn):
        pass
    assert turn.active is False  # a copied tool context expires with its turn


def test_a_shared_file_is_copied_and_reported_without_its_path(tmp_path: Path):
    path = tmp_path / "review.md"
    path.write_text("Review this.\n", encoding="utf-8")
    result = _share(tmp_path, path, name="handoff.md")
    assert result["ok"] is True and result["name"] == "handoff.md"
    assert str(path) not in json.dumps(result)
    stored, = RoomArtifactOutbox(tmp_path / "state.db").list(_scope())
    assert stored["artifact_id"] == result["artifact_id"]
    path.write_text("changed after sharing\n", encoding="utf-8")
    assert RoomArtifactOutbox(tmp_path / "state.db").read(_scope(), stored["artifact_id"])[1] == b"Review this.\n"


def test_relative_paths_and_unexpected_errors_are_refused_without_detail(tmp_path: Path, monkeypatch):
    path = tmp_path / "review.md"
    path.write_text("Review this.\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert _share(tmp_path, path.name) == {"ok": False, "error": UNSAFE}
    monkeypatch.setattr(RoomArtifactOutbox, "put_open_file", lambda *_a, **_k: (_ for _ in ()).throw(
        OSError("/Users/alice/private/review.md")))
    result = _share(tmp_path, path)
    assert result == {"ok": False, "error": "That file could not be shared. Check the file and try again."}


@pytest.mark.parametrize("layout", ["direct", "ancestor"])
def test_symbolic_links_are_never_followed(tmp_path: Path, layout):
    target = tmp_path / "target"
    target.mkdir()
    (target / "handoff.md").write_text("secret-ish\n", encoding="utf-8")
    if layout == "direct":
        link = tmp_path / "link.md"
        link.symlink_to(target / "handoff.md")
    else:
        (tmp_path / "linkdir").symlink_to(target, target_is_directory=True)
        link = tmp_path / "linkdir" / "handoff.md"
    assert _share(tmp_path, link) == {"ok": False, "error": "Symbolic links cannot be shared."}


def test_the_opened_bytes_are_shared_even_if_the_path_is_swapped(tmp_path: Path, monkeypatch):
    safe = tmp_path / "safe"
    safe.mkdir()
    source = safe / "handoff.md"
    source.write_bytes(b"SAFE BYTES\n")
    hidden = tmp_path / "hidden"
    hidden.mkdir()
    (hidden / source.name).write_bytes(b"TOP SECRET\n")
    original = RoomArtifactOutbox.put_open_file

    def swap_then_put(outbox, *args, **kwargs):
        safe.rename(tmp_path / "original-safe")
        safe.symlink_to(hidden, target_is_directory=True)
        return original(outbox, *args, **kwargs)

    monkeypatch.setattr(RoomArtifactOutbox, "put_open_file", swap_then_put)
    home = tmp_path / "home"
    home.mkdir()
    result = _share(home, source)
    assert result["ok"] is True
    assert RoomArtifactOutbox(home / "state.db").read(_scope(), result["artifact_id"])[1] == b"SAFE BYTES\n"


@pytest.mark.parametrize("storage", ["hosted-room-artifact-outbox", "hosted-room-attachments",
                                     "roomlink-attachment-spool"])
def test_private_room_storage_is_never_shared(tmp_path: Path, storage):
    home = tmp_path / ".hermes"
    private = home / storage / "private.bin"
    private.parent.mkdir(parents=True)
    private.write_bytes(b"private bytes")
    assert _share(home, private) == {"ok": False, "error": "Private Group Chat storage cannot be shared."}
    allowed = home / (storage + "-safe") / "handoff.md"  # a name prefix is not the private root
    allowed.parent.mkdir(parents=True)
    allowed.write_text("safe workspace file\n", encoding="utf-8")
    assert _share(home, allowed)["ok"] is True


def test_another_rooms_private_output_cannot_be_reshared(tmp_path: Path):
    home = tmp_path / ".hermes"
    home.mkdir()
    source = tmp_path / "private.md"
    source.write_text("room A only\n", encoding="utf-8")
    first = _share(home, source, scope=_scope(room_id="room-a", task_id="dtask:a"))
    with sqlite3.connect(home / "state.db") as conn:
        blob = conn.execute("SELECT blob_name FROM hosted_room_output_artifacts WHERE artifact_id=?",
                            (first["artifact_id"],)).fetchone()[0]
    private_blob = home / "hosted-room-artifact-outbox" / "blobs" / blob
    second = _scope(room_id="room-b", task_id="dtask:b")
    assert _share(home, private_blob, scope=second) == {
        "ok": False, "error": "Private Group Chat storage cannot be shared."}
    assert RoomArtifactOutbox(home / "state.db").list(second) == []


@pytest.mark.parametrize("name", [".env", "auth.json", "config.yaml"])
def test_another_profiles_state_is_never_shared(tmp_path: Path, monkeypatch, name):
    root = tmp_path / "hermes"
    alpha, beta = root / "profiles" / "alpha", root / "profiles" / "beta"
    alpha.mkdir(parents=True)
    beta.mkdir(parents=True)
    candidate = beta / name
    candidate.write_text("do not share\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    assert _share(alpha, candidate, scope=_scope(target_profile="alpha")) == {
        "ok": False, "error": "Files owned by another Hermes profile cannot be shared."}


@pytest.mark.parametrize("name", ["token=abc123def456ghi789.txt", "AKIAIOSFODNN7EXAMPLE.csv"])
def test_credential_shaped_names_are_refused_before_they_reach_the_turn_result(tmp_path: Path, name):
    path = tmp_path / "report.txt"
    path.write_text("report\n", encoding="utf-8")
    error = "That file name looks like a credential. Pass a different name and try again."
    assert _share(tmp_path, path, name=name) == {"ok": False, "error": error}
    credential_named = tmp_path / name
    credential_named.write_text("report\n", encoding="utf-8")
    assert _share(tmp_path, credential_named) == {"ok": False, "error": error}
    assert _share(tmp_path, credential_named, name="report.txt")["ok"] is True


def test_remote_execution_backends_are_read_through_their_file_adapter(tmp_path: Path, monkeypatch):
    from tools import file_tools, file_tools_paths

    home = tmp_path / ".hermes"
    home.mkdir()
    payload = b"remote handoff\n"
    file_ops = SimpleNamespace(
        _has_command=lambda command: command == "python3",
        _escape_shell_arg=lambda value: repr(value),
        _exec=lambda command, timeout: SimpleNamespace(exit_code=0, stdout="HERMES_ROOM_FILE_V1:" + json.dumps(
            {"ok": True, "data": base64.b64encode(payload).decode("ascii")}) + "\n"))
    monkeypatch.setattr(file_tools_paths, "_terminal_env_type_for_task", lambda task_id: "ssh")
    monkeypatch.setattr(file_tools_paths, "_resolve_path_for_task", lambda path, task_id: Path(path))
    monkeypatch.setattr(file_tools, "_get_file_ops", lambda task_id: file_ops)
    result = _share(home, "/remote/workspace/handoff.md", task_id="room-session")
    assert result["ok"] is True
    assert RoomArtifactOutbox(home / "state.db").read(_scope(), result["artifact_id"])[1] == payload
    assert _share(home, "/remote/home/.ssh/id_ed25519", task_id="room-session") == {
        "ok": False, "error": "Hermes credential and internal state files cannot be shared."}


def test_the_tool_is_offered_directly_to_group_chat_turns_only():
    from model_tools import get_tool_definitions
    from tools.tool_search import _DIRECT_SURFACE_TOOLSETS
    import toolsets

    assert "bot_room" in _DIRECT_SURFACE_TOOLSETS
    assert toolsets.TOOLSETS["bot_room"]["tools"] == ["share_group_file"]
    names = {tool["function"]["name"] for tool in get_tool_definitions(enabled_toolsets=["bot_room"], quiet_mode=True)}
    assert "share_group_file" in names
    assert "share_group_file" not in {
        tool["function"]["name"] for tool in get_tool_definitions(enabled_toolsets=["file"], quiet_mode=True)}
