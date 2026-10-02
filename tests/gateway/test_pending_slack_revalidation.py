"""Slack pending replay requires current native authority before cached media."""

import base64
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.pending_native import PendingNativeInput
from gateway.run import GatewayRunner
from gateway.shutdown_pending import PendingQueueSnapshot
from gateway.shutdown_pending_codec import decode_pending_event


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "media,state",
    [
        (False, "current"),
        (True, "current"),
        (False, "deleted"),
        (False, "wrong-author"),
        (False, "wrong-workspace"),
        (False, "edited"),
        (True, "changed-cache"),
        (True, "ignored-channel"),
        (True, "missing-mention"),
        (True, "revoked-user"),
        (True, "changed-file"),
        (False, "api-refused"),
        (True, "malformed-file"),
        (False, "thread-current"),
        (False, "thread-edited"),
        (False, "thread-deleted"),
        (False, "thread-image"),
        (False, "thread-image-changed"),
        (False, "thread-legacy"),
    ],
)
async def test_restored_slack_input_requires_current_native_source_and_cache(
    monkeypatch, media, state
):
    pytest.importorskip("slack_bolt")
    from slack_bolt.async_app import AsyncApp
    from slack_sdk.web.async_client import AsyncWebClient
    from slack_sdk.web.async_slack_response import AsyncSlackResponse
    from plugins.platforms.slack.adapter import SlackAdapter

    monkeypatch.setenv("SLACK_ALLOW_ALL_USERS", "true")
    runner = GatewayRunner(GatewayConfig())
    adapter = SlackAdapter(
        PlatformConfig(
            enabled=True,
            token="xoxb-test",
            extra={
                "require_mention": False,
                "reply_in_thread": False,
                "reactions": False,
            },
        )
    )
    runner.adapters[Platform.SLACK] = adapter
    adapter.gateway_runner = runner
    runner._wire_adapter_handlers(adapter, message_handler=AsyncMock(return_value=None))
    client = AsyncWebClient(token="xoxb-test")
    adapter._app = AsyncApp(client=client, signing_secret="test")
    adapter._bot_user_id = "U900"
    adapter._team_clients = {"T999": client}
    adapter._team_bot_user_ids = {"T999": "U900"}
    native: dict[str, Any] = {
        "type": "message",
        "user": "U333",
        "channel": "C555",
        "team": "T999",
        "ts": "1000.000001",
        "text": "authored input",
        "client_msg_id": "test-input",
    }
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    if media:
        native["files"] = [
            {
                "id": "F777",
                "name": "image.png",
                "size": len(png),
                "mimetype": "image/png",
                "url_private_download": "https://files.slack.com/files-pri/T999-F777/image.png",
            }
        ]
    thread = state.startswith("thread-")
    root_message: dict[str, Any] = {
        "type": "message",
        "user": "U333",
        "ts": "999.000001",
        "text": "original thread quotation",
    }
    if thread:
        native["thread_ts"] = root_message["ts"]
    if state in {"thread-image", "thread-image-changed"}:
        root_message["files"] = [
            {
                "id": "F888",
                "name": "root.png",
                "size": len(png),
                "mimetype": "image/png",
                "url_private_download": "https://files.slack.com/files-pri/T999-F888/root.png",
            }
        ]
    current = deepcopy(native)

    async def api_call(api_method, **kwargs):
        payload: dict[str, Any] = {"ok": True}
        if api_method == "auth.test":
            payload.update(
                team_id="T444" if state == "wrong-workspace" else "T999", user_id="U900"
            )
        elif api_method == "users.info":
            payload["user"] = {
                "id": "U333",
                "is_bot": False,
                "deleted": False,
                "profile": {"display_name": "sender"},
            }
        elif api_method == "conversations.info":
            payload["channel"] = {
                "id": "C555",
                "name": "allowed",
                "is_im": False,
                "is_member": True,
            }
        elif api_method in {"conversations.history", "conversations.replies"}:
            if state == "api-refused":
                from slack_sdk.errors import SlackApiError

                raise SlackApiError(
                    "current channel refused", {"ok": False, "error": "not_in_channel"}
                )
            messages = (
                [deepcopy(root_message), deepcopy(current)]
                if thread
                else [deepcopy(current)]
            )
            if state == "thread-deleted" and replaying:
                messages = [deepcopy(current)]
            payload.update(
                messages=[] if state == "deleted" else messages, has_more=False
            )
        else:
            raise AssertionError(f"unexpected native API method: {api_method}")
        return AsyncSlackResponse(
            client=client,
            http_verb="GET",
            api_url="https://slack.com/api/" + api_method,
            req_args=kwargs,
            data=payload,
            headers={},
            status_code=200,
        )

    replaying = False
    monkeypatch.setattr(client, "api_call", api_call)
    monkeypatch.setattr(
        adapter, "_download_slack_file_bytes", AsyncMock(return_value=png)
    )
    prepared = await adapter._prepare_slack_message(deepcopy(native), "T999", "C555")
    assert prepared is not None
    assert adapter._canonicalize(prepared.source) is not None
    record = PendingQueueSnapshot.capture(
        runner._session_key_for_source(prepared.source), [prepared]
    ).events[0]
    restored = decode_pending_event(record, adapter=adapter)
    replaying = True
    if thread:
        adapter._session_store = runner.session_store
        runner.session_store.get_or_create_session(prepared.source)
    if state == "thread-edited":
        root_message["text"] = "current edited quotation"
    if state == "thread-image-changed":
        root_message["files"][0]["id"] = "Fchanged"
    if state == "thread-legacy" and restored._pending_native_input is not None:
        restored._pending_native_input.content.pop("thread_context", None)
    if state == "wrong-author":
        current["user"] = "U444"
    if state == "edited":
        current["text"] = "changed authored input"
    if state == "changed-cache":
        assert prepared.media_urls
        Path(prepared.media_urls[0]).write_bytes(b"changed")
    if state == "changed-file":
        current["files"][0]["id"] = "F444"
    if state == "malformed-file":
        current["files"] = [None]
    if state == "ignored-channel":
        adapter.config.extra["ignored_channels"] = ["C555"]
    if state == "missing-mention":
        adapter.config.extra["require_mention"] = True
    if state == "revoked-user":
        monkeypatch.setenv("SLACK_ALLOW_ALL_USERS", "false")
        monkeypatch.setenv("SLACK_ALLOWED_USERS", "U444")
    checks = []
    original_check = PendingNativeInput.attachments_available

    def check_available(value, paths):
        checks.append(tuple(paths))
        return original_check(value, paths)

    monkeypatch.setattr(PendingNativeInput, "attachments_available", check_available)
    verified = await adapter.revalidate_pending_event(
        restored, authorize=runner._is_user_authorized_for_source
    )
    actual = (
        None
        if verified is None
        else (
            verified.text,
            verified.media_urls,
            verified.message_type,
            verified.source.chat_id,
            verified.source.user_id,
            verified.source.scope_id,
            verified.channel_context,
        )
    )
    context = prepared.channel_context
    if state == "thread-edited" and context:
        context = context.replace(
            "original thread quotation", "current edited quotation"
        )
    success = state in {"current", "thread-current", "thread-edited", "thread-image"}
    expected = (
        None
        if not success
        else (
            prepared.text,
            prepared.media_urls,
            prepared.message_type,
            "C555",
            "U333",
            "T999",
            context,
        )
    )
    expected_checks = (
        [tuple(prepared.media_urls)]
        if (media and state in {"current", "changed-cache"}) or state == "thread-image"
        else []
    )
    assert (actual, checks) == (expected, expected_checks)


@pytest.mark.asyncio
@pytest.mark.parametrize("authorization", ["current", "revoked"])
@pytest.mark.parametrize("context_kind", ["own-file", "thread-root-file"])
async def test_slack_restoration_uses_routed_files_and_persisted_user_receipts(
    tmp_path, monkeypatch, authorization, context_kind
):
    import asyncio
    import hermes_state
    from agent.secret_scope import is_multiplex_active, set_multiplex_active
    from gateway.input_owner import gateway_input_owner
    from gateway.pending_execution import consume_pending_execution
    from gateway.profile_routing import ProfileRoute
    from gateway.run import _profile_runtime_scope
    from hermes_constants import get_hermes_home
    from utils import atomic_json_write

    pytest.importorskip("slack_bolt")
    from slack_bolt.async_app import AsyncApp
    from slack_sdk.web.async_client import AsyncWebClient
    from slack_sdk.web.async_slack_response import AsyncSlackResponse
    from plugins.platforms.slack.adapter import SlackAdapter

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    launch = tmp_path / ".hermes"
    launch.mkdir()
    (launch / ".env").write_text(
        "GATEWAY_ALLOWED_USERS=U333\nSLACK_ALLOW_ALL_USERS=false\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(
        hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH
    )
    homes = {profile: launch / "profiles" / profile for profile in ("a", "b")}
    for home in homes.values():
        home.mkdir(parents=True)
        (home / "config.yaml").write_text("{}\n")
        (home / ".env").write_text("SLACK_ALLOW_ALL_USERS=true\n")
    runner = GatewayRunner(
        GatewayConfig(
            multiplex_profiles=True,
            profile_routes=[
                ProfileRoute(
                    name=profile, platform="slack", profile=profile, chat_id=channel
                )
                for profile, channel in (("a", "C555"), ("b", "C556"))
            ],
        )
    )
    adapter = SlackAdapter(
        PlatformConfig(
            enabled=True,
            token="xoxb-test",
            extra={
                "require_mention": False,
                "reply_in_thread": False,
                "reactions": False,
            },
        )
    )
    runner.adapters[Platform.SLACK] = adapter
    runner._profile_adapters = {profile: {} for profile in homes}
    adapter.gateway_runner = runner
    adapter._mark_connected()
    client = AsyncWebClient(token="xoxb-test")
    adapter._app = AsyncApp(client=client, signing_secret="test")
    adapter._bot_user_id = "U900"
    adapter._team_clients = {"T999": client}
    adapter._team_bot_user_ids = {"T999": "U900"}
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    monkeypatch.setattr(
        adapter, "_download_slack_file_bytes", AsyncMock(return_value=png)
    )
    current = {}
    reads = []

    async def api_call(api_method, **kwargs):
        args = kwargs.get("params") or kwargs.get("json") or kwargs.get("data") or {}
        payload: dict[str, Any] = {"ok": True}
        if api_method == "auth.test":
            payload.update(team_id="T999", user_id="U900")
        elif api_method == "users.info":
            payload["user"] = {
                "id": "U333",
                "is_bot": False,
                "deleted": False,
                "profile": {"display_name": "sender"},
            }
        elif api_method == "conversations.info":
            payload["channel"] = {
                "id": args["channel"],
                "name": "allowed",
                "is_im": False,
                "is_member": True,
            }
        elif api_method in {"conversations.history", "conversations.replies"}:
            if args.get("latest") is not None:
                reads.append((args["latest"], get_hermes_home()))
                values = [deepcopy(current[args["latest"]])]
            else:
                values = [
                    deepcopy(value)
                    for value in current.values()
                    if value.get("thread_ts", value["ts"]) == args["ts"]
                ]
            payload.update(messages=values, has_more=False)
        else:
            raise AssertionError(f"unexpected native API method: {api_method}")
        return AsyncSlackResponse(
            client=client,
            http_verb="GET",
            api_url="https://slack.com/api/" + api_method,
            req_args=kwargs,
            data=payload,
            headers={},
            status_code=200,
        )

    monkeypatch.setattr(client, "api_call", api_call)
    seen = []

    async def receive(event):
        entry = runner.session_store.lookup_by_session_key(
            runner._session_key_for_source(event.source)
        )
        assert entry is not None
        seen.append((
            event.message_id,
            get_hermes_home(),
            tuple(Path(path).read_bytes() for path in event.media_urls),
            event.channel_context,
        ))
        db = runner.session_store._db_for_session_id(entry.session_id)
        db.append_message(
            entry.session_id,
            "user",
            event.text,
            display_metadata={
                "gateway_input_owner": gateway_input_owner(event, event.source)
            },
        )
        consume_pending_execution(runner, event)
        return None

    runner._wire_adapter_handlers(adapter, message_handler=receive)
    monkeypatch.setattr(runner, "_await_startup_warmup", AsyncMock())
    active = is_multiplex_active()
    set_multiplex_active(True)
    expected = []
    expected_reads = []
    try:
        for index, profile in enumerate(("a", "b", "a")):
            (launch / ".env").write_text(
                "GATEWAY_ALLOWED_USERS=U333\nSLACK_ALLOW_ALL_USERS=false\n"
            )
            ts = f"1000.{index:06d}"
            channel = "C555" if profile == "a" else "C556"
            native = {
                "type": "message",
                "user": "U333",
                "channel": channel,
                "team": "T999",
                "ts": ts,
                "text": f"authored-{index}",
                "client_msg_id": f"input-{index}",
                "files": [
                    {
                        "id": f"F77{index}",
                        "name": "image.png",
                        "size": len(png),
                        "mimetype": "image/png",
                        "url_private_download": f"https://files.slack.com/files-pri/T999-F77{index}/image.png",
                    }
                ],
            }
            root_ts = f"999.{index:06d}"
            if context_kind == "thread-root-file":
                native["thread_ts"] = root_ts
                current[root_ts] = {
                    "type": "message",
                    "user": "U333",
                    "ts": root_ts,
                    "text": f"quoted-{index}",
                    "files": [
                        {
                            "id": f"F88{index}",
                            "name": "root.png",
                            "size": len(png),
                            "mimetype": "image/png",
                            "url_private_download": f"https://files.slack.com/files-pri/T999-F88{index}/root.png",
                        }
                    ],
                }
            current[ts] = native
            with _profile_runtime_scope(homes[profile]):
                event = await adapter._prepare_slack_message(
                    deepcopy(native), "T999", channel
                )
                assert event is not None
                assert adapter._canonicalize(event.source) is not None
                entry = runner.session_store.get_or_create_session(event.source)
                snapshot = PendingQueueSnapshot.capture(entry.session_key, [event])
                path = homes[profile] / "pending_messages" / f"pending-{index}.json"
                path.parent.mkdir(exist_ok=True)
                atomic_json_write(path, snapshot.to_payload(), mode=0o600)
            if authorization == "revoked":
                (launch / ".env").write_text(
                    "GATEWAY_ALLOWED_USERS=U444\nSLACK_ALLOW_ALL_USERS=false\n"
                )
            runner._startup_restore_in_progress = True
            await runner._finish_startup_restore()
            if tasks := list(adapter._background_tasks):
                await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)
            if authorization == "current":
                expected.append((
                    ts,
                    homes[profile],
                    (png, png) if context_kind == "thread-root-file" else (png,),
                    event.channel_context,
                ))
                expected_reads.append((ts, homes[profile]))
                if context_kind == "thread-root-file":
                    expected_reads.append((root_ts, homes[profile]))
            assert (seen, reads, path.exists(), get_hermes_home()) == (
                expected,
                expected_reads,
                authorization == "revoked",
                launch,
            )
    finally:
        set_multiplex_active(active)
