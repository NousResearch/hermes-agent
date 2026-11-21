"""Every output of a native slash invocation reaches only its invoker (``SlackAdapter`` reply owner).

Overlapping, queued and replayed slashes each keep their own private lane, and no egress of the
invocation (acknowledgement, retry, Block Kit, file, stream) turns its answer public.
"""

import asyncio
import contextvars
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from tests.gateway.restart_test_helpers import make_restart_runner
from tests.gateway.test_slack import SlackAdapter, _slack_mod


@pytest.fixture()
def adapter():
    """Same adapter as ``tests/gateway/test_slack.py``: a mocked client, events captured."""
    a = SlackAdapter(PlatformConfig(enabled=True, token="***"))
    a._app = MagicMock()
    a._app.client = AsyncMock()
    a._app.client.users_info = AsyncMock(return_value={"user": {
        "is_bot": False, "profile": {"display_name": "Test User"}, "real_name": "Test User"}})
    a._bot_user_id = "U_BOT"
    a._running = True
    a.handle_message = AsyncMock()
    return a


@pytest.fixture(autouse=True)
def _redirect_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.platforms.base.DOCUMENT_CACHE_DIR", tmp_path / "doc_cache")
    monkeypatch.setattr("gateway.platforms.base.VIDEO_CACHE_DIR", tmp_path / "video_cache")


class _SlashWire:
    """Slack's HTTP faked under the real slash ingress, base dispatch/drain/retry/ledger and
    send(): ``out`` records the lane each output took: ``url`` (the invocation's response_url,
    by command name), ``ephemeral`` (chat.postEphemeral) or ``public`` (postMessage / upload)."""

    def __init__(self, adapter, monkeypatch, *, url_fails=(), ephemeral_errors=()):
        self.adapter, self.out, self.replaced, self.url_posts = adapter, [], [], 0
        self.url_fails, self.ephemeral_errors = url_fails, list(ephemeral_errors)
        del adapter.handle_message  # the fixture's capture mock: run the real dispatch
        wire = self

        class _Resp:
            def __init__(self, status):
                self.status = status

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def text(self):
                return "failed"

        class _Session:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            def post(self, url, json, timeout):
                wire.url_posts += 1
                if wire.url_fails == "all" or wire.url_posts in wire.url_fails:
                    return _Resp(500)
                wire.out.append(("url", url.rsplit("/", 1)[-1], json["text"]))
                wire.replaced.append(json["replace_original"])
                return _Resp(200)

        monkeypatch.setattr(_slack_mod.aiohttp, "ClientSession", lambda **_kw: _Session())

        async def post_ephemeral(channel, user, text):
            error = wire.ephemeral_errors.pop(0) if wire.ephemeral_errors else None
            if isinstance(error, Exception):
                raise error
            if error:
                return {"ok": False, "error": error}
            wire.out.append(("ephemeral", user, text))
            return {"ok": True}

        def public(**kwargs):
            wire.out.append(("public", kwargs.get("filename") or kwargs["text"]))
            return {"ok": True, "ts": "1.2"}

        client = adapter._app.client
        client.chat_postEphemeral = AsyncMock(side_effect=post_ephemeral)
        client.chat_postMessage = AsyncMock(side_effect=public)
        client.files_upload_v2 = AsyncMock(side_effect=public)
        client.chat_update = AsyncMock(side_effect=public)

    async def slash(self, command, user="U1", channel="C1"):
        await self.adapter._handle_slash_command({
            "command": command, "text": "", "user_id": user, "channel_id": channel,
            "team_id": "T1", "response_url": f"https://hooks.slack.test/{command[1:]}"})

    def message(self, user, text):
        return MessageEvent(text=text, raw_message={"type": "message"}, source=self.adapter.build_source(
            chat_id="C1", chat_type="group", user_id=user, scope_id="T1"))

    async def settle(self):
        while pending := [t for t in self.adapter._background_tasks if not t.done()]:
            await asyncio.gather(*pending)


class TestSlashInvocationPrivateReply:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("row", ["overlap", "queued-slash", "queued-other-user", "replayed"])
    async def test_each_slash_invocation_owns_its_private_reply(self, adapter, monkeypatch, row):
        """Through the real dispatch (an inline bypass command, the base queue's drain task, the
        runner replaying an event it held), a reply reaches only the private lane of the
        invocation it answers: never public, never another invocation's response_url."""
        wire = _SlashWire(adapter, monkeypatch)
        adapter.config.extra["group_sessions_per_user"] = row != "queued-other-user"
        gate, held = asyncio.Event(), []

        async def runner(event):
            command = event.text.split()[0]
            if command == "/slow":
                await gate.wait()
                return None if row == "queued-other-user" else "reply /slow"
            if row == "replayed" and not held:
                held.append(event)  # the startup-restore gate keeps it for later
                return None
            return f"reply {command}"

        adapter._message_handler = runner
        if row == "replayed":
            await wire.slash("/later")
            await wire.settle()
            await asyncio.create_task(
                adapter.handle_message(held[0]), context=contextvars.Context())
        else:
            await wire.slash("/slow")
            if row == "overlap":
                await wire.slash("/status")  # registered: dispatched inline mid-turn
            elif row == "queued-slash":
                await wire.slash("/zz")  # unregistered: queued behind /slow
            else:
                await adapter.handle_message(wire.message("U2", "hello"))
            gate.set()
        await wire.settle()

        assert wire.out == {
            "overlap": [("url", "status", "reply /status"), ("url", "slow", "reply /slow")],
            "queued-slash": [("url", "slow", "reply /slow"), ("url", "zz", "reply /zz")],
            "queued-other-user": [("public", "reply hello")],
            "replayed": [("url", "later", "reply /later")],
        }[row]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("row", [
        "ack-then-final", "retry", "plain-fallback", "chunk-2-fails", "chunk-3-fails",
        "partial-then-retry", "ledger", "restart", "block-kit-and-file", "dm-control"])
    async def test_every_output_of_a_slash_invocation_stays_private(
        self, adapter, monkeypatch, tmp_path, row):
        """Every output of one native slash invocation in a channel stays private, through the
        real delivery path: an acknowledgement before the final, a retry or plain-text fallback
        after a failed private send, a response_url that fails mid-reply (no chunk twice), the
        delivery ledger's public replay, restart auto-resume, and Block Kit / file / channel-message-edit egress, which
        has no private form and is refused. In a DM (already private) only the first reply
        replaces the ack and the rest stays in the conversation history."""
        import functools
        import gateway.delivery_ledger as ledger
        import gateway.platforms.base as _base

        wire = _SlashWire(
            adapter, monkeypatch,
            url_fails={"retry": "all", "plain-fallback": "all", "chunk-2-fails": {2},
                       "chunk-3-fails": {3}, "partial-then-retry": {2}}.get(row, ()),
            ephemeral_errors={"retry": [ConnectionError("network unreachable")],
                              "partial-then-retry": [ConnectionError("network unreachable")],
                              "plain-fallback": ["channel_not_found"] * 9}.get(row, ()))
        monkeypatch.setattr(adapter, "_send_with_retry", functools.partial(
            SlackAdapter._send_with_retry, adapter, base_delay=0))
        monkeypatch.setattr(_base.random, "uniform", lambda lo, _hi: lo)
        monkeypatch.setattr(ledger, "ledger_enabled", lambda: True)
        monkeypatch.setattr(ledger, "record_obligation", lambda **kw: wire.out.append(
            ("ledger", kw["content"])))
        monkeypatch.setattr(ledger, "mark_attempting", lambda _oid: None)
        if row.startswith(("chunk", "partial")):
            monkeypatch.setattr(adapter, "_format_chunks", lambda content: content.split("|"))
        channel = "D1" if row == "dm-control" else "C1"
        meta = {"slack_team_id": "T1"}
        doc = tmp_path / "save.txt"
        doc.write_text("transcript")
        results = []
        gateway, _ = make_restart_runner(adapter)  # restart recovery: shutdown pre-mark + crash marker
        gateway.adapters = {Platform.SLACK: adapter}
        gateway._restart_requested = True
        gateway.session_store.mark_resume_pending = lambda sk, _why: wire.out.append(("resumable", sk))
        gateway.session_store.mark_turn_active = lambda sk: wire.out.append(("resumable", sk)) or "tok"

        async def runner(event):
            if row in ("ack-then-final", "dm-control"):
                await adapter.send(channel, "ack", metadata=meta)
            if row in ("restart", "dm-control"):  # the gateway restarts mid-turn
                sk = adapter._event_session_key(event)
                gateway._running_agents[sk] = MagicMock()
                gateway._session_state(sk).turn.event = event
                await gateway._mark_durable_active_turn(event, sk)
                await gateway._mark_running_sessions_resume_pending("restart")
                gateway._running_agents.clear()
            if row == "ledger":
                event.text = "a rewritten agent prompt"  # /plan, /learn fall through like this
            if row in ("block-kit-and-file", "dm-control"):
                results.append((await adapter.send_slash_confirm(
                    channel, "Title", "Body", "sk", "1", metadata=meta)).success)
                results.append((await adapter.send_document(channel, str(doc))).success)
            if row == "block-kit-and-file":
                adapter._status_message_ids[("C1", "", "progress")] = "9.9"  # an earlier bubble
                await adapter.send_or_update_status("C1", "progress", "status", metadata=meta)
                results.append((await adapter.send_model_picker(
                    channel, [{"slug": "p"}], "m", "p", "sk", None, metadata=meta)).success)
            return "A|B|C" if row.startswith(("chunk", "partial")) else "final"

        adapter._message_handler = runner
        await wire.slash("/cmd", channel=channel)
        await wire.settle()

        url = [("url", "cmd", text) for text in ("A", "B", "C")]
        assert wire.out == {
            "ack-then-final": [("url", "cmd", "ack"), ("url", "cmd", "final")],
            "retry": [("ephemeral", "U1", "final")],
            "plain-fallback": [],
            "chunk-2-fails": url[:1] + [("ephemeral", "U1", "B"), ("ephemeral", "U1", "C")],
            "chunk-3-fails": url[:2] + [("ephemeral", "U1", "C")],
            "partial-then-retry": url[:1],  # the retry must not resend what the user has seen
            "ledger": [("url", "cmd", "final")],
            "restart": [("url", "cmd", "final")],  # never re-run into the public lane on boot
            "block-kit-and-file": [("url", "cmd", "status"), ("url", "cmd", "final")],
            "dm-control": [("url", "cmd", "ack"), *[("resumable", "agent:main:slack:dm:T1:D1")] * 2,
                           ("public", "Title: Body"), ("public", "save.txt"), ("public", "final")],
        }[row]
        assert wire.replaced[:1] == ([True] if wire.replaced else [])
        assert not any(wire.replaced[1:])
        assert results == {"block-kit-and-file": [False, False, False],
                           "dm-control": [True, True]}.get(row, [])
