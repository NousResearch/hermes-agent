"""A bot shares files in a gateway group chat: staged per attempt, published only with its reply."""
import asyncio
import base64
import json
import os
import sqlite3
import time

import pytest

from gateway import hosted_room_driver as tasks
from gateway import hosted_rooms
from gateway.hosted_room_attachments import HostedRoomAttachmentStore
from tests.gateway.fixtures.hosted_room_owner import events, owner, run_turn


def share(path, **kwargs):
    from tools import group_file_tool  # registers the tool, as discovery does
    from tools.registry import registry
    return json.loads(registry.dispatch("share_group_file", {"path": str(path), **kwargs}, task_id="default"))


def write_file(tmp_path, name="report.txt", data=b"explicit result bytes\n"):
    return _place(tmp_path / "work", name, data)


def sharing(paths, reply, *, shared=None, **kwargs):
    async def handle(event):
        for path in paths:
            result = await asyncio.to_thread(share, path, **kwargs)
            if shared is not None:
                shared.append(result)
        return reply
    return handle


def staged_rows(service):
    with sqlite3.connect(service.db_path) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute(
            "SELECT * FROM hosted_room_attachments WHERE upload_id LIKE 'out:%' ORDER BY created_at")]


def _place(directory, name, data):
    """Write ``data`` at exactly ``directory / name``."""
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def remote_sandbox(monkeypatch, sandbox, links=None):
    """Run the real remote reader on a disposable filesystem with virtual sandbox path names."""
    import shlex
    import subprocess
    import sys
    from gateway import media_fetch
    fetched = []
    for alias, target in (links or {}).items():
        if alias == target:
            continue
        link = sandbox / alias.lstrip("/")
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(sandbox / target.lstrip("/"))

    class RemoteEnv:
        _remote_home = "/home/agent"

        def execute(self, command, **kwargs):
            executable, option, script, request_json = shlex.split(command)
            assert executable == "python3"
            request = json.loads(request_json)
            path = request["path"]
            request["path"] = str(sandbox / path.lstrip("/"))
            request["policy"]["prefixes"] = [str(sandbox / p.lstrip("/")) for p in request["policy"]["prefixes"]]
            result = subprocess.run([sys.executable, option, script, json.dumps(request)],
                                    capture_output=True, text=True, check=False)
            if '"data":' in result.stdout:
                fetched.append(path)
            return {"returncode": result.returncode, "output": result.stdout + result.stderr}
    monkeypatch.setattr(media_fetch, "_active_remote_env", lambda: RemoteEnv())
    return fetched


@pytest.mark.asyncio
async def test_a_shared_file_is_published_with_the_reply_and_reaches_the_next_bot(tmp_path, monkeypatch):
    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        data = b"explicit result bytes\n"
        shared = []
        runner._handle_message = sharing([write_file(tmp_path, data=data)], "@reviewer Here is the report.",
                                         shared=shared)
        await run_turn(authority, service)
        assert shared[0]["shared"] is True and shared[0]["name"] == "report.txt"
        message, = events(service, "message.member")
        assert message["payload"]["text"] == "@reviewer Here is the report."
        attachment, = message["payload"]["attachments"]
        assert (attachment["name"], attachment["kind"], attachment["mime"], attachment["size"]) == (
            "report.txt", "file", "text/plain", len(data))
        gateway_id, epoch = service._owned_authority("room")
        saved = HostedRoomAttachmentStore(service.db_path).read_viewer(
            room_id="room", attachment_id=attachment["attachment_id"], event_id=message["event_id"],
            authority_gateway_id=gateway_id, authority_epoch=epoch)
        assert saved.data == data
        row, = staged_rows(service)
        assert (row["state"], row["event_id"], row["viewer_access"], row["expires_at"]) == (
            "committed", message["event_id"], 1, None)
        assert json.loads(row["recipient_member_ids_json"]) == ["reviewer"]
        service.prepare_room(service.bindings()[0])
        task, = tasks.list_tasks(service.db_path, room_id="room", status="queued")
        assert task["payload"]["target_member_id"] == "reviewer"
        assert task["payload"]["attachments"] == [{**attachment, "event_id": message["event_id"]}]
        assert "report.txt" in task["payload"]["prompt"]


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome, status", [("failed", "failed"), ("superseded", "settled"), ("stopped", "cancelled")])
async def test_an_unfinished_or_superseded_turn_publishes_no_files(tmp_path, monkeypatch, outcome, status):
    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        async def handle(event):
            from gateway.session_results import execution_result
            await asyncio.to_thread(share, write_file(tmp_path))
            if outcome == "superseded":  # a newer request in the same thread arrives before the reply settles
                await asyncio.to_thread(service.send, room_id="room", event_id="newer",
                                        payload={"thread_id": "thread", "text": "@writer Never mind."})
                return "Late report."
            execution_result.get()["result"] = {"final_response": "", "messages": [],
                                                "interrupted" if outcome == "stopped" else "failed": True}
            return ""
        runner._handle_message = handle
        turn = await run_turn(authority, service)
        if outcome == "stopped":
            await asyncio.to_thread(service.stop_room, "room", cancel_id="stop")
            service.prepare_room(turn.binding)
        assert tasks.get_task(service.db_path, turn.task["identity"])["status"] == status
        assert not events(service, "message.member")
        row, = staged_rows(service)
        assert (row["state"], row["event_id"], row["viewer_access"]) == ("uploaded", None, 0)
        # Publishing the outcome ended the file's hold; the store reclaims it like an abandoned upload.
        assert row["expires_at"] is not None
        HostedRoomAttachmentStore(service.db_path).prune(now=row["expires_at"])
        assert staged_rows(service) == []


@pytest.mark.asyncio
async def test_a_shared_file_waits_for_a_late_publication(tmp_path, monkeypatch):
    """Seam: the attachment store's clock, advanced past the uncommitted-upload expiry."""
    now = [time.time()]
    original = HostedRoomAttachmentStore.__init__

    def clocked(self, db_path, **kwargs):
        kwargs["clock"] = lambda: now[0]
        original(self, db_path, **kwargs)
    monkeypatch.setattr(HostedRoomAttachmentStore, "__init__", clocked)
    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        runner._handle_message = sharing([write_file(tmp_path)], "(pass)")
        turn = await run_turn(authority, service, publish=False)
        # The gateway stops before publishing and comes back a day later: the held file is kept.
        now[0] += 24 * 60 * 60
        HostedRoomAttachmentStore(service.db_path).prune()
        service.prepare_room(turn.binding)
        message, = events(service, "message.member")
        assert message["payload"]["text"] == "Shared report.txt."
        assert [a["name"] for a in message["payload"]["attachments"]] == ["report.txt"]


@pytest.mark.asyncio
async def test_a_disbanded_rooms_held_files_are_reclaimed(tmp_path, monkeypatch):
    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        runner._handle_message = sharing([write_file(tmp_path)], "Done.")
        await run_turn(authority, service, publish=False)
        gateway_id, epoch = service._owned_authority("room")
        hosted_rooms.disband_room(service.db_path, room_id="room", expected_gateway_id=gateway_id,
                                  expected_epoch=epoch)
        store = HostedRoomAttachmentStore(service.db_path)
        store.prune()
        row, = staged_rows(service)
        assert row["state"] == "disbanded" and row["expires_at"] is not None
        store.prune(now=row["expires_at"])
        assert staged_rows(service) == []


@pytest.mark.asyncio
async def test_only_the_settled_attempts_own_files_are_published(tmp_path, monkeypatch):
    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        from dataclasses import replace
        from gateway.hosted_room_outputs import current_turn, hosted_turn_scope

        async def handle(event):
            other = replace(current_turn(), execution_generation=current_turn().execution_generation + 1)
            with hosted_turn_scope(other):
                await asyncio.to_thread(share, write_file(tmp_path, "other.txt", b"other attempt\n"))
            await asyncio.to_thread(share, write_file(tmp_path, "mine.txt", b"this attempt\n"))
            return "Done."
        runner._handle_message = handle
        await run_turn(authority, service)
        message, = events(service, "message.member")
        assert [a["name"] for a in message["payload"]["attachments"]] == ["mine.txt"]
        assert sorted((row["name"], row["state"]) for row in staged_rows(service)) == [
            ("mine.txt", "committed"), ("other.txt", "uploaded")]


@pytest.mark.asyncio
async def test_sharing_refuses_outside_a_turn_bad_paths_and_too_many_files(tmp_path, monkeypatch):
    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        outside = share(write_file(tmp_path))
        assert outside["error"] == "File sharing requires a bot turn on the gateway that owns this group chat."
        results = []

        async def handle(event):
            for path in ("relative.txt", tmp_path / "missing.txt"):
                results.append(await asyncio.to_thread(share, path))
            for index in range(9):
                results.append(await asyncio.to_thread(share, write_file(tmp_path, f"f{index}.txt", b"%d\n" % index)))
            results.append(await asyncio.to_thread(share, write_file(tmp_path, "f0.txt", b"changed\n")))
            return "Done."
        runner._handle_message = handle
        await run_turn(authority, service)
        assert all("error" in result for result in results[:2])
        assert [result.get("shared") for result in results[2:10]] == [True] * 8
        assert "at most 8 files" in results[10]["error"]
        assert "already shared in this turn" in results[11]["error"]
        message, = events(service, "message.member")
        assert [a["name"] for a in message["payload"]["attachments"]] == [f"f{index}.txt" for index in range(8)]


@pytest.mark.asyncio
async def test_a_file_keeps_its_requested_name_wherever_its_bytes_are_read(tmp_path, monkeypatch):
    """Seam: remote_sandbox."""
    _place(tmp_path / "sandbox" / "workspace", "report.txt", b"remote result\n")
    remote_sandbox(monkeypatch, tmp_path / "sandbox")
    target = write_file(tmp_path, "v3.txt", b"third draft\n")
    (target.parent / "latest.txt").symlink_to(target)
    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        results = []
        # Each fetch reads the remote file into a fresh cache copy; the share is still one file.
        runner._handle_message = sharing(
            ["/workspace/report.txt", "/workspace/report.txt", target.parent / "latest.txt"], "Done.", shared=results)
        await run_turn(authority, service)
        assert [result["name"] for result in results] == ["report.txt", "report.txt", "latest.txt"]
        message, = events(service, "message.member")
        assert [a["name"] for a in message["payload"]["attachments"]] == ["report.txt", "latest.txt"]


@pytest.mark.asyncio
async def test_a_client_upload_cannot_pose_as_a_bots_shared_file(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from gateway.hosted_room_attachments import AttachmentError
    from gateway.hosted_room_outputs import current_turn, output_prefix
    from gateway.session_hosted_attachments import upload
    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        actor = SimpleNamespace(profile_id=service.authority.profile_id, capabilities={"session:submit"},
                                subject="alice")
        refused = []

        async def handle(event):
            # The room owner uploads under the running attempt's own staging id, as is and padded
            # so that only the store's normalization would turn it into that id.
            staging_id = output_prefix(current_turn().task_id, current_turn().execution_generation) + "0"
            for pad in ("", " ", "\t", "\n "):
                params = {"room_id": "room", "upload_id": pad + staging_id, "kind": "file", "name": "forged.txt",
                          "mime": "text/plain", "data_base64": base64.b64encode(b"x").decode()}
                try:
                    await asyncio.to_thread(upload, service, actor, params)
                except AttachmentError as exc:
                    refused.append(str(exc))
            return "My own reply."
        runner._handle_message = handle
        await run_turn(authority, service)
        assert len(refused) == 4 and all("reserved for files a bot shares" in reason for reason in refused)
        message, = events(service, "message.member")
        assert "attachments" not in message["payload"]
        assert staged_rows(service) == []


def test_the_tool_reaches_group_chat_member_sessions_only(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from model_tools import get_tool_definitions
    from toolsets import session_disabled_toolsets

    def offered(toolsets, platform):
        return "share_group_file" in {tool["function"]["name"] for tool in get_tool_definitions(
            enabled_toolsets=toolsets, disabled_toolsets=session_disabled_toolsets(None, platform),
            quiet_mode=True)}
    # Member sessions run as platform bot_room with the bot_room toolset folded in.
    assert offered(["terminal", "bot_room"], "bot_room")
    # Nothing else pays its schema, not even an "all" toolset selection.
    assert not offered(["terminal", "bot_room"], "desktop")
    assert not offered(["all"], "telegram")


@pytest.mark.asyncio
async def test_only_turns_that_can_share_files_are_told_how(tmp_path, monkeypatch):
    from gateway import hosted_room_discussion as discussion
    rule = "call share_group_file"
    async with owner(tmp_path, monkeypatch) as (_authority, service, _runner):
        service.send(room_id="room", event_id="ask", payload={"thread_id": "t", "text": "@writer Draft it"})
        task, = tasks.list_tasks(service.db_path, room_id="room", status="queued")
        assert rule in task["payload"]["prompt"]  # the canonical owner binds file sharing
        room = service._room("room")
        snapshot = service._policy_snapshot(room)
        # A room server that cannot bind it (the standalone server) plans without the rule.
        plain = discussion.plan_next_task(room, snapshot.events, local_profiles=service.room_profiles(room),
                                          initial_watermarks=snapshot.watermarks, freeze_input_context=True)
        assert rule not in plain.task.payload["prompt"]


def _blob(service, row):
    return HostedRoomAttachmentStore(service.db_path).blob_root / row["blob_id"]


@pytest.mark.asyncio
@pytest.mark.parametrize("damage, reply, published", [  # a deleted blob under a text reply: the next test
    ("overwritten", "Here is the report.", "Here is the report.\n\nCould not share report.txt."),
    ("deleted", "(pass)", "Could not share report.txt.")])
async def test_a_file_whose_bytes_are_gone_is_named_and_the_reply_still_published(
        tmp_path, monkeypatch, damage, reply, published):
    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        runner._handle_message = sharing([write_file(tmp_path)], reply)
        turn = await run_turn(authority, service, publish=False)
        row, = staged_rows(service)
        if damage == "deleted":
            _blob(service, row).unlink()
        else:
            _blob(service, row).write_bytes(b"X" * row["size"])
        # Publishing reads no byte, so the room is not stalled: the reply goes out and names the file.
        service.prepare_room(turn.binding)
        message, = events(service, "message.member")
        assert message["payload"]["text"] == published and "attachments" not in message["payload"]
        assert events(service, "turn.settled")
        row, = staged_rows(service)
        assert (row["state"], row["expires_at"] is not None) == ("uploaded", True)  # reclaimable


@pytest.mark.asyncio
async def test_finishing_a_publication_repeats_the_reply_already_in_the_log(tmp_path, monkeypatch):
    """Seam: hosted_rooms.append_event refuses the first turn.settled append, as when another
    event lands between the reply and its terminal event."""
    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        runner._handle_message = sharing(
            [write_file(tmp_path, "kept.txt", b"kept\n"), write_file(tmp_path, "lost.txt", b"lost\n")], "Two files.")
        turn = await run_turn(authority, service, publish=False)
        lost = next(row for row in staged_rows(service) if row["name"] == "lost.txt")
        _blob(service, lost).unlink()
        append, refused = hosted_rooms.append_event, []

        def once(db_path, **kwargs):
            if kwargs["kind"] == "turn.settled" and not refused:
                refused.append(kwargs["event_id"])
                raise hosted_rooms.EventCursorConflictError("room changed before event publication")
            return append(db_path, **kwargs)
        monkeypatch.setattr(hosted_rooms, "append_event", once)
        service.prepare_room(turn.binding)
        first, = events(service, "message.member")
        assert not events(service, "turn.settled")
        _blob(service, lost).write_bytes(b"lost\n")
        service.prepare_room(turn.binding)
        message, = events(service, "message.member")
        assert message == first and events(service, "turn.settled")
        assert message["payload"]["text"] == "Two files.\n\nCould not share lost.txt."
        assert [a["name"] for a in message["payload"]["attachments"]] == ["kept.txt"]
        assert sorted((row["name"], row["state"]) for row in staged_rows(service)) == [
            ("kept.txt", "committed"), ("lost.txt", "uploaded")]


@pytest.mark.asyncio
async def test_the_sentence_for_a_pass_that_shares_files_summons_nobody(tmp_path, monkeypatch):
    from gateway import hosted_room_discussion as discussion
    names = ["@reviewer", "extra.txt", "notes@reviewer(1).txt", "notes@all(1).txt"]
    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        runner._handle_message = sharing(
            [write_file(tmp_path, name, b"%d\n" % index) for index, name in enumerate(names)], "(pass)")
        turn = await run_turn(authority, service, text="@writer Only you")
        message, = events(service, "message.member")
        assert message["payload"]["text"] == "Shared " + ", ".join(n.replace("@", "@\u2060") for n in names) + "."
        assert [a["name"] for a in message["payload"]["attachments"]] == names
        assert not discussion.is_pass_text(message["payload"]["text"])
        service.prepare_room(turn.binding)
        assert not tasks.list_tasks(service.db_path, room_id="room", status="queued")


@pytest.mark.asyncio
async def test_secrets_and_private_hermes_files_are_never_shared(tmp_path, monkeypatch):
    work = tmp_path / "work"
    vault_key = _place(tmp_path / "vault", "vault.key", b"VAULT-KEY\n")
    secrets = [vault_key, _place(tmp_path / "vault", "vault.json.enc", b"ciphertext\n"),
               _place(work / "project", ".env", b"AWS_SECRET_ACCESS_KEY=1\n"),
               _place(work / "project", ".envrc", b"export TOKEN=1\n"),
               _place(tmp_path / "skills" / ".hub", "injected.md", b"# injected\n"),
               _place(tmp_path, ".env", b"OPENAI_API_KEY=1\n"),
               _place(tmp_path, "config.yaml", b"api_key: private\n")]
    (work / "notes.txt").symlink_to(vault_key)
    # The room's own file store, moved to another disk behind a symlink.
    (tmp_path / "elsewhere" / "store").mkdir(parents=True)
    (tmp_path / "hosted-room-attachments").symlink_to(tmp_path / "elsewhere" / "store", target_is_directory=True)
    results = []
    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        other = tmp_path / "profiles" / "reviewer"
        secrets += [_place(other / "vault", "vault.key", b"OTHER-KEY\n"),
                    _place(other, "auth.json", b"{}\n"),
                    _place(other / "work", "private.txt", b"private notes\n"),
                    _place(other / "mcp-tokens", "token.json", b"{}\n"),
                    _place(other / "skills" / ".hub", "x.md", b"# x\n")]
        store = HostedRoomAttachmentStore(service.db_path)
        item = store.put(room_id="room", upload_id="pending", kind="file", name="private.txt",
                         mime="text/plain", data=b"UNPUBLISHED\n")
        with sqlite3.connect(service.db_path) as conn:
            blob_id, = conn.execute("SELECT blob_id FROM hosted_room_attachments WHERE attachment_id=?",
                                    (item["attachment_id"],)).fetchone()
        blob = store.blob_root / blob_id
        secrets += [blob, blob.resolve(), work / "notes.txt"]

        async def handle(event):
            for path in secrets:
                results.append(await asyncio.to_thread(share, path))
            # Hard links last: until here each file has one link, so no refusal above rests on the inode check.
            os.link(vault_key, work / "key-copy.txt")
            os.link(blob, tmp_path / "hard-link.txt")
            for path in (work / "key-copy.txt", tmp_path / "hard-link.txt"):
                results.append(await asyncio.to_thread(share, path))
            results.append(await asyncio.to_thread(share, work / "project" / ".env", name="notes.txt"))
            results.append(await asyncio.to_thread(share, write_file(tmp_path, "fine.txt", b"fine\n")))
            return "Done."
        runner._handle_message = handle
        await run_turn(authority, service)
        *refused, control = results
        assert [result.get("error") for result in refused] == [
            "That file cannot be shared. Share a file from your workspace or a Hermes media folder."] * len(refused)
        assert control["shared"] is True
        message, = events(service, "message.member")
        assert [a["name"] for a in message["payload"]["attachments"]] == ["fine.txt"]


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["/workspace/.env", "/home/agent/.hermes/vault/vault.key",
    "/home/agent/.hermes/profiles/other/auth.json", "/home/agent/.hermes/profiles/other/work/notes.txt"])
@pytest.mark.parametrize("alias", [False, True])
async def test_remote_private_sources_are_refused_before_fetch(tmp_path, monkeypatch, source, alias):
    sandbox = tmp_path / "sandbox"
    _place(sandbox, source.lstrip("/"), b"private bytes")
    _place(sandbox / "workspace", "public.txt", b"public bytes")
    requested = "/workspace/report.txt" if alias else source
    fetched = remote_sandbox(monkeypatch, sandbox, links={requested: source})
    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        results = []
        runner._handle_message = sharing([requested, "/workspace/public.txt"], "Done.", shared=results)
        await run_turn(authority, service)
        assert [result.get("shared", False) for result in results] == [False, True]
        assert fetched == ["/workspace/public.txt"]
        message, = events(service, "message.member")
        assert [item["name"] for item in message["payload"]["attachments"]] == ["public.txt"]


@pytest.mark.asyncio
async def test_publication_retry_releases_files_after_terminal_was_already_logged(tmp_path, monkeypatch):
    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        runner._handle_message = sharing([write_file(tmp_path)], "Done.")
        turn = await run_turn(authority, service, publish=False)
        gateway, epoch = service._owned_authority("room")
        hosted_rooms.append_event(service.db_path, room_id="room", event_id="supersede", kind="message.user",
                                  actor={"kind": "user", "id": "desktop"}, authority_gateway_id=gateway,
                                  authority_epoch=epoch, payload={"thread_id": "thread", "text": "@writer Stop"})
        original = service._release_task_outputs
        attempts = []

        def interrupted(*args):
            attempts.append(1)
            if len(attempts) == 1:
                raise OSError("publication cleanup interrupted")
            return original(*args)

        monkeypatch.setattr(service, "_release_task_outputs", interrupted)
        with pytest.raises(OSError, match="publication cleanup interrupted"):
            service.prepare_room(turn.binding)
        service.prepare_room(turn.binding)
        row, = staged_rows(service)
        assert row["state"] == "uploaded" and row["expires_at"] is not None
        assert not events(service, "message.member")
