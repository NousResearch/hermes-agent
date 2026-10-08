"""Parent-to-Home fallback through real SDK builders and profile config resolution."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.feishu.adapter import FeishuAdapter, _load_lark_oapi


def ok(**fields):
    return NS(success=lambda: True, data=NS(**fields))


def failed(code):
    return NS(success=lambda: False, code=code, msg="PRIVATE API detail")


def write_config(home_path, *, chat="oc_home", thread=None, app="cli_owner", platform="feishu"):
    home_path.mkdir(parents=True, exist_ok=True)
    home = None if chat is None else {"platform": platform, "chat_id": chat, "thread_id": thread}
    (home_path / "config.yaml").write_text(json.dumps({"platforms": {"feishu": {
        "enabled": True, "home_channel": home, "extra": {
            "app_id": app, "topic_delivery_fallback": "parent_then_home"}}}}))


def make_adapter():
    pytest.importorskip("lark_oapi")
    assert _load_lark_oapi()
    adapter = FeishuAdapter(PlatformConfig(extra={"app_id": "cli_owner", "topic_delivery_fallback": "parent_then_home"}))
    adapter._client = Mock()
    adapter._client.im.v1.message.list.side_effect = lambda request: ok(items=[
        NS(message_id="om_home_anchor", thread_id="omt_home")]) if request.container_id == "omt_home" else ok(items=[])
    adapter._client.im.v1.message.reply.return_value = ok(message_id="om_home_reply")
    adapter._client.im.v1.message.create.side_effect = lambda request: (
        failed(232009) if request.request_body.receive_id == "oc_origin" else ok(message_id="om_home_created"))
    adapter._client.im.v1.file.create.return_value = ok(file_key="file_uploaded")
    adapter._client.im.v1.image.create.return_value = ok(image_key="img_uploaded")
    return adapter


def close_adapter(adapter):
    if executor := getattr(adapter, "_sdk_executor", None):
        executor.shutdown(wait=True)


@pytest.fixture
def home_adapter(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("FEISHU_HOME_CHANNEL", raising=False)
    monkeypatch.delenv("FEISHU_HOME_CHANNEL_THREAD_ID", raising=False)
    monkeypatch.delenv("FEISHU_TOPIC_DELIVERY_FALLBACK", raising=False)
    write_config(tmp_path)
    adapter = make_adapter()
    yield adapter
    close_adapter(adapter)


@pytest.mark.asyncio
@pytest.mark.parametrize("home_thread", [None, "omt_home"])
@pytest.mark.parametrize("initial_anchor", [None, "om_dissolved", "om_withdrawn"])
async def test_parent_gone_routes_full_chunks_and_media_to_configured_home(home_adapter, tmp_path, home_thread, initial_anchor):
    adapter = home_adapter
    write_config(tmp_path, thread=home_thread)
    def reply(request):
        code = {"om_dissolved": 232009, "om_withdrawn": 230011}.get(request.message_id)
        return failed(code) if code else ok(message_id="om_home_reply")
    adapter._client.im.v1.message.reply.side_effect = reply
    metadata = {"thread_id": "omt_origin", "reply_to_message_id": initial_anchor}
    adapter.MAX_MESSAGE_LENGTH = 12
    result = await adapter.send("oc_origin", "First chunk. Second chunk. Third chunk.", metadata=metadata)
    assert result.success
    state = metadata["_feishu_topic_delivery"]
    assert state["destination"] == "home"
    creates = adapter._client.im.v1.message.create.call_args_list
    assert creates[0].args[0].request_body.receive_id == "oc_origin"
    if home_thread:
        assert len(creates) == 1
        home_replies = [call.args[0] for call in adapter._client.im.v1.message.reply.call_args_list
                        if call.args[0].message_id == "om_home_anchor"]
        assert len(home_replies) >= 3
        assert all(request.request_body.reply_in_thread is True for request in home_replies)
    else:
        assert len(creates) >= 4
        assert all(call.args[0].request_body.receive_id == "oc_home" for call in creates[1:])
    path = tmp_path / "private.pdf"
    path.write_bytes(b"file")
    # Streaming overflow uses the returned Home message ID as the continuation anchor.
    result = await adapter.send_document("oc_origin", str(path), caption="private caption",
                                         reply_to=result.message_id, metadata=dict(metadata))
    assert result.success
    assert sum(call.args[0].request_body.receive_id == "oc_origin"
               for call in adapter._client.im.v1.message.create.call_args_list) == 1
    if home_thread:
        request = adapter._client.im.v1.message.reply.call_args.args[0]
        assert request.message_id == "om_home_anchor" and request.request_body.reply_in_thread is True
    else:
        assert adapter._client.im.v1.message.create.call_args.args[0].request_body.receive_id == "oc_home"
    assert adapter._client.im.v1.message.list.call_count == (int(initial_anchor != "om_dissolved") + int(bool(home_thread)))


@pytest.mark.asyncio
@pytest.mark.parametrize("kwargs", [
    {"chat": None}, {"chat": "oc_origin"}, {"chat": "oc_origin", "thread": "omt_origin"},
    {"chat": "oc_origin", "thread": "omt_other"}, {"chat": "bad address"},
    {"thread": "bad thread"}, {"platform": "slack"}, {"app": "cli_other_bot"},
])
async def test_unavailable_same_or_foreign_home_stops_without_recursion(home_adapter, tmp_path, kwargs, caplog):
    write_config(tmp_path, **kwargs)
    metadata = {"thread_id": "omt_origin"}
    result = await home_adapter._send_with_retry("oc_origin", "PRIVATE CONTENT", metadata=metadata)
    assert not result.success and result.retry_suppressed
    assert "PRIVATE" not in result.error
    again = await home_adapter._send_with_retry("oc_origin", "OTHER", metadata=dict(metadata))
    assert again is result
    assert home_adapter._client.im.v1.message.create.call_count == 1
    assert "home_unavailable" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [230001, 230034, 230002, 230006, 230013, 230018, 230020, 99991663, 99991400])
async def test_parent_parameter_permission_auth_and_rate_errors_never_redirect(home_adapter, code):
    home_adapter._client.im.v1.message.create.side_effect = lambda request: failed(code)
    result = await home_adapter.send("oc_origin", "private", metadata={"thread_id": "omt_origin"})
    assert not result.success
    assert home_adapter._client.im.v1.message.create.call_count == 1
    assert home_adapter._client.im.v1.message.create.call_args.args[0].request_body.receive_id == "oc_origin"


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["topic", "parent"])
@pytest.mark.parametrize("error", [TimeoutError, ConnectionError])
async def test_uncertain_attempt_then_dissolved_never_redirects(home_adapter, monkeypatch, stage, error):
    async def no_sleep(_):
        pass
    monkeypatch.setattr("plugins.platforms.feishu.adapter_delivery.asyncio.sleep", no_sleep)
    transport = home_adapter._client.im.v1.message.reply if stage == "topic" else home_adapter._client.im.v1.message.create
    transport.side_effect = [error("private detail"), failed(232009), failed(232009)]
    metadata = {"thread_id": "omt_origin"}
    if stage == "topic":
        metadata["reply_to_message_id"] = "om_original"
    result = await home_adapter._send_with_retry("oc_origin", "private", metadata=metadata)
    assert not result.success and result.retry_suppressed
    assert metadata["_feishu_topic_delivery"]["ambiguous_send"]
    assert transport.call_count == 2
    requests = [call.args[0] for call in transport.call_args_list]
    assert len({request.request_body.uuid for request in requests}) == 1
    if stage == "topic":
        home_adapter._client.im.v1.message.create.assert_not_called()
    else:
        assert all(request.request_body.receive_id == "oc_origin" for request in requests)
    again = await home_adapter._send_with_retry("oc_origin", "private", metadata=metadata)
    assert again is result and transport.call_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [TimeoutError, ConnectionError])
async def test_parent_transport_exhaustion_is_terminal_for_outer_retries(home_adapter, monkeypatch, error):
    async def no_sleep(_):
        pass
    monkeypatch.setattr("plugins.platforms.feishu.adapter_delivery.asyncio.sleep", no_sleep)
    home_adapter._client.im.v1.message.create.side_effect = error("PRIVATE DETAILS")
    metadata = {"thread_id": "omt_origin"}
    result = await home_adapter._send_with_retry("oc_origin", "private", metadata=metadata)
    assert not result.success and result.retry_suppressed
    assert "PRIVATE" not in result.error
    creates = home_adapter._client.im.v1.message.create.call_args_list
    assert len(creates) == 3 and len({c.args[0].request_body.uuid for c in creates}) == 1
    assert all(c.args[0].request_body.receive_id == "oc_origin" for c in creates)
    assert await home_adapter._send_with_retry("oc_origin", "private", metadata=metadata) is result
    assert home_adapter._client.im.v1.message.create.call_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("home_thread", [None, "omt_home"])
@pytest.mark.parametrize("failure", [232009, 230002, "timeout", "no_anchor"])
async def test_failed_home_is_terminal_and_never_falls_back_to_home_parent(home_adapter, tmp_path, monkeypatch, home_thread, failure):
    async def no_sleep(_):
        pass
    monkeypatch.setattr("plugins.platforms.feishu.adapter_delivery.asyncio.sleep", no_sleep)
    write_config(tmp_path, thread=home_thread)
    if failure == "no_anchor":
        if not home_thread:
            return
        home_adapter._client.im.v1.message.list.side_effect = None
        home_adapter._client.im.v1.message.list.return_value = ok(items=[])
    else:
        def home_outcome(request):
            if getattr(request.request_body, "receive_id", None) == "oc_origin":
                return failed(232009)
            if failure == "timeout":
                raise TimeoutError("PRIVATE DETAILS")
            return failed(failure)
        home_adapter._client.im.v1.message.create.side_effect = home_outcome
        home_adapter._client.im.v1.message.reply.side_effect = home_outcome
    metadata = {"thread_id": "omt_origin"}
    result = await home_adapter._send_with_retry("oc_origin", "PRIVATE CONTENT", metadata=metadata)
    assert not result.success and result.retry_suppressed and result.message_id is None
    assert "PRIVATE" not in result.error
    calls = (home_adapter._client.im.v1.message.create.call_count, home_adapter._client.im.v1.message.reply.call_count)
    image = tmp_path / "private.png"
    image.write_bytes(b"png")
    again = await home_adapter.send_image_file("oc_origin", str(image), metadata=dict(metadata))
    assert again is result
    home_adapter._client.im.v1.image.create.assert_not_called()
    assert calls == (home_adapter._client.im.v1.message.create.call_count, home_adapter._client.im.v1.message.reply.call_count)
    if home_thread:
        assert home_adapter._client.im.v1.message.create.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("message_id", ["om_parent", None])
async def test_partial_parent_response_never_splits_remainder_to_home(home_adapter, message_id):
    home_adapter.MAX_MESSAGE_LENGTH = 8
    home_adapter._client.im.v1.message.create.side_effect = [ok(message_id=message_id), failed(232009)]
    result = await home_adapter.send("oc_origin", "one two three four five six", metadata={"thread_id": "omt_origin"})
    assert not result.success and result.retry_suppressed
    assert result.raw_response["partial_overflow"]
    assert home_adapter._client.im.v1.message.create.call_count == 2
    assert all(call.args[0].request_body.receive_id == "oc_origin"
               for call in home_adapter._client.im.v1.message.create.call_args_list)


@pytest.mark.asyncio
async def test_profile_a_b_a_home_env_and_yaml_never_borrow_launch_defaults(tmp_path, monkeypatch):
    import agent.secret_scope as ss
    from gateway.run import _profile_runtime_scope

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("FEISHU_HOME_CHANNEL", "oc_launch_secret")
    monkeypatch.setenv("FEISHU_HOME_CHANNEL_THREAD_ID", "omt_launch_secret")
    a, b = tmp_path / "profiles" / "a", tmp_path / "profiles" / "b"
    write_config(a, chat="oc_yaml_a", thread="omt_yaml_a")
    write_config(b, chat="oc_yaml_b")
    (a / ".env").write_text("FEISHU_HOME_CHANNEL=oc_env_a\n")
    previous = ss.is_multiplex_active()
    ss.set_multiplex_active(True)
    try:
        for path, profile, expected in [(a, "a", "oc_env_a"), (b, "b", "oc_yaml_b"), (a, "a", "oc_env_a")]:
            with _profile_runtime_scope(path, hydrate_secrets=False):
                adapter = make_adapter()
                try:
                    metadata = {"thread_id": "omt_origin", "hermes_profile": profile}
                    result = await adapter.send("oc_origin", "PRIVATE", metadata=metadata)
                    assert result.success
                    creates = adapter._client.im.v1.message.create.call_args_list
                    assert [c.args[0].request_body.receive_id for c in creates] == ["oc_origin", expected]
                    # A changed profile name with unchanged ambient scope fails closed.
                    rejected = await adapter.send("oc_origin", "PRIVATE", metadata={
                        "thread_id": "omt_origin", "hermes_profile": "other"})
                    assert not rejected.success and rejected.retry_suppressed
                finally:
                    close_adapter(adapter)
        write_config(b, chat=None)
        with _profile_runtime_scope(b, hydrate_secrets=False):
            adapter = make_adapter()
            try:
                result = await adapter.send("oc_origin", "PRIVATE", metadata={"thread_id": "omt_origin"})
                assert not result.success and adapter._client.im.v1.message.create.call_count == 1
            finally:
                close_adapter(adapter)
    finally:
        ss.set_multiplex_active(previous)


@pytest.mark.asyncio
async def test_home_topic_recovery_is_bounded_and_sticks_to_recovered_anchor(home_adapter, tmp_path):
    write_config(tmp_path, thread="omt_home")
    home_adapter._client.im.v1.message.list.side_effect = lambda req: ok(items=[
        NS(message_id="om_deleted", deleted=True), NS(message_id="om_stale"), NS(message_id="om_valid"),
        NS(message_id="om_third"), NS(message_id="om_never")]) if req.container_id == "omt_home" else ok(items=[])
    home_adapter._client.im.v1.message.reply.side_effect = lambda req: (
        failed(230011) if req.message_id == "om_stale" else ok(message_id="om_home_sent"))
    metadata = {"thread_id": "omt_origin"}
    results = await asyncio.gather(*(home_adapter.send("oc_origin", str(i), metadata=metadata) for i in range(3)))
    assert all(result.success for result in results)
    assert [c.args[0].message_id for c in home_adapter._client.im.v1.message.reply.call_args_list] == [
        "om_stale", "om_valid", "om_valid", "om_valid"]
    assert home_adapter._client.im.v1.message.create.call_count == 1
    assert home_adapter._client.im.v1.message.list.call_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("home_thread", [None, "omt_home"])
@pytest.mark.parametrize("exception", [False, True])
async def test_home_post_format_rejection_retries_plain_at_same_home(home_adapter, tmp_path, home_thread, exception):
    write_config(tmp_path, thread=home_thread)
    def outcome(request):
        if getattr(request.request_body, "receive_id", None) == "oc_origin":
            return failed(232009)
        if request.request_body.msg_type == "post":
            message = "The content format of the post type is incorrect"
            if exception:
                raise ValueError(message)
            return NS(success=lambda: False, code=230001, msg=message)
        return ok(message_id="om_home_text")
    home_adapter._client.im.v1.message.create.side_effect = outcome
    home_adapter._client.im.v1.message.reply.side_effect = outcome
    metadata = {"thread_id": "omt_origin"}
    result = await home_adapter.send("oc_origin", "**PRIVATE CONTENT**", metadata=metadata)
    assert result.success and result.message_id == "om_home_text"
    assert metadata["_feishu_topic_delivery"]["destination"] == "home"
    assert "terminal" not in metadata["_feishu_topic_delivery"]
    assert sum(c.args[0].request_body.receive_id == "oc_origin"
               for c in home_adapter._client.im.v1.message.create.call_args_list) == 1
    transport = home_adapter._client.im.v1.message.reply if home_thread else home_adapter._client.im.v1.message.create
    assert [c.args[0].request_body.msg_type for c in transport.call_args_list][-2:] == ["post", "text"]


@pytest.mark.asyncio
@pytest.mark.parametrize("home_thread", [None, "omt_home"])
async def test_real_stream_consumer_continuations_remain_at_home(home_adapter, tmp_path, home_thread):
    from gateway.stream_consumer import GatewayStreamConsumer

    write_config(tmp_path, thread=home_thread)
    consumer = GatewayStreamConsumer(home_adapter, "oc_origin", metadata={"thread_id": "omt_origin"})
    first = await consumer._send_new_chunk("first chunk", None)
    second = await consumer._send_new_chunk("second chunk", first)
    assert first and second
    creates = home_adapter._client.im.v1.message.create.call_args_list
    assert sum(c.args[0].request_body.receive_id == "oc_origin" for c in creates) == 1
    if home_thread:
        assert len(creates) == 1
        assert [c.args[0].message_id for c in home_adapter._client.im.v1.message.reply.call_args_list] == [
            "om_home_anchor", "om_home_anchor"]
    else:
        assert [c.args[0].request_body.receive_id for c in creates] == ["oc_origin", "oc_home", "oc_home"]


@pytest.mark.asyncio
async def test_missing_or_foreign_secondary_secret_scope_cannot_read_launch_home(tmp_path, monkeypatch):
    import agent.secret_scope as ss
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("FEISHU_HOME_CHANNEL", "oc_launch")
    secondary = tmp_path / "profiles" / "secondary"
    write_config(secondary, chat="oc_secondary")
    previous = ss.is_multiplex_active()
    ss.set_multiplex_active(True)
    home_token = set_hermes_home_override(secondary)
    try:
        for mapping, scope_home in [(None, None), ({}, None), ({}, str(tmp_path))]:
            token = ss.set_secret_scope(mapping, profile_home=scope_home)
            try:
                adapter = make_adapter()
                try:
                    result = await adapter.send("oc_origin", "PRIVATE", metadata={
                        "thread_id": "omt_origin", "hermes_profile": "secondary"})
                    assert not result.success and result.retry_suppressed
                    creates = adapter._client.im.v1.message.create.call_args_list
                    assert [c.args[0].request_body.receive_id for c in creates] == ["oc_origin"]
                finally:
                    close_adapter(adapter)
            finally:
                ss.reset_secret_scope(token)
    finally:
        reset_hermes_home_override(home_token)
        ss.set_multiplex_active(previous)


@pytest.mark.asyncio
async def test_unscoped_default_profile_can_use_own_home_under_multiplex(home_adapter, tmp_path, monkeypatch):
    import agent.secret_scope as ss

    monkeypatch.setenv("FEISHU_HOME_CHANNEL", "oc_default_env")
    previous = ss.is_multiplex_active()
    ss.set_multiplex_active(True)
    token = ss.set_secret_scope(None)
    try:
        result = await home_adapter.send("oc_origin", "PRIVATE", metadata={"thread_id": "omt_origin"})
        assert result.success
        assert home_adapter._client.im.v1.message.create.call_args.args[0].request_body.receive_id == "oc_default_env"
    finally:
        ss.reset_secret_scope(token)
        ss.set_multiplex_active(previous)


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["parent", "home_flat", "home_topic"])
@pytest.mark.parametrize("exception", [False, True])
async def test_uncertainty_before_format_rejection_never_sends_new_plain_payload(home_adapter, tmp_path, monkeypatch, stage, exception):
    async def no_sleep(_):
        pass
    monkeypatch.setattr("plugins.platforms.feishu.adapter_delivery.asyncio.sleep", no_sleep)
    write_config(tmp_path, thread="omt_home" if stage == "home_topic" else None)
    attempts = []
    def outcome(request):
        if stage != "parent" and getattr(request.request_body, "receive_id", None) == "oc_origin":
            return failed(232009)
        attempts.append(request)
        if len(attempts) == 1:
            raise TimeoutError("PRIVATE DETAIL")
        message = "The content format of the post type is incorrect"
        if exception:
            raise ValueError(message)
        return NS(success=lambda: False, code=230001, msg=message)
    home_adapter._client.im.v1.message.create.side_effect = outcome
    home_adapter._client.im.v1.message.reply.side_effect = outcome
    metadata = {"thread_id": "omt_origin"}
    result = await home_adapter._send_with_retry("oc_origin", "**PRIVATE CONTENT**", metadata=metadata)
    assert not result.success and result.retry_suppressed
    assert len(attempts) == 2
    assert len({request.request_body.uuid for request in attempts}) == 1
    assert all(request.request_body.msg_type == "post" for request in attempts)
    assert await home_adapter._send_with_retry("oc_origin", "**PRIVATE CONTENT**", metadata=metadata) is result
    assert len(attempts) == 2


@pytest.mark.asyncio
async def test_unstamped_secret_mapping_cannot_supply_default_home(home_adapter):
    import agent.secret_scope as ss

    previous = ss.is_multiplex_active()
    ss.set_multiplex_active(True)
    token = ss.set_secret_scope({"FEISHU_HOME_CHANNEL": "oc_foreign_unstamped"})
    try:
        result = await home_adapter.send("oc_origin", "PRIVATE", metadata={"thread_id": "omt_origin"})
        assert not result.success and result.retry_suppressed
        assert home_adapter._client.im.v1.message.create.call_count == 1
    finally:
        ss.reset_secret_scope(token)
        ss.set_multiplex_active(previous)
