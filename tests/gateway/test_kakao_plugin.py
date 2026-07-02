"""Tests for the Kakao (i Open Builder skill-server) platform adapter plugin.

Kakao's skill-server model is fundamentally request-response: there is no
push API, so this adapter's ``send()`` fulfills a per-chat "delivery slot"
(an ``asyncio.Future``) opened by an inbound webhook, instead of calling
an outbound send API like every other bundled adapter. Coverage areas:

1. shared-secret header verification (constant-time, no signature exists)
2. allowlist gating (single list -- Kakao channel chats are always 1:1)
3. SkillResponse building: simpleText chunking/truncation, 3-output cap
4. send() routing: fulfills the open delivery slot / rejects when none is
   open / swallows mid-turn busy-acks (no channel to deliver them on)
5. full webhook turn lifecycle: fast answer, slow answer with callback,
   slow answer without callback, and turn supersession
6. register() metadata + standalone_send always fails with an explanation
   (no proactive delivery is possible on this platform)
"""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_kakao = load_plugin_adapter("kakao")

verify_shared_secret = _kakao.verify_shared_secret
strip_markdown_preserving_urls = _kakao.strip_markdown_preserving_urls
split_for_kakao = _kakao.split_for_kakao
build_skill_response = _kakao.build_skill_response
build_callback_ack = _kakao.build_callback_ack
_allowed = _kakao._allowed
_is_system_bypass = _kakao._is_system_bypass
KakaoAdapter = _kakao.KakaoAdapter
register = _kakao.register
check_requirements = _kakao.check_requirements
validate_config = _kakao.validate_config
is_connected = _kakao.is_connected
_standalone_send = _kakao._standalone_send
_env_enablement = _kakao._env_enablement
KAKAO_SIMPLETEXT_MAX_CHARS = _kakao.KAKAO_SIMPLETEXT_MAX_CHARS
KAKAO_MAX_OUTPUTS = _kakao.KAKAO_MAX_OUTPUTS
RETRIEVE_QUICK_REPLY = _kakao.RETRIEVE_QUICK_REPLY


def _skill_payload(chat_id: str, utterance: str, callback_url: str | None = None) -> dict:
    payload = {
        "intent": {"id": "i1", "name": "fallback"},
        "userRequest": {
            "timezone": "Asia/Seoul",
            "block": {"id": "b1", "name": "fallback"},
            "utterance": utterance,
            "lang": "ko",
            "user": {
                "id": chat_id,
                "type": "botUserKey",
                "properties": {"botUserKey": chat_id},
            },
            "params": {},
        },
        "bot": {"id": "bot1", "name": "hermes"},
        "action": {"id": "a1", "name": "default", "params": {}, "detailParams": {}},
    }
    if callback_url:
        payload["userRequest"]["callbackUrl"] = callback_url
    return payload


def _make_adapter(monkeypatch, **extra_overrides) -> "KakaoAdapter":
    monkeypatch.delenv("KAKAO_SKILL_SECRET", raising=False)
    monkeypatch.delenv("KAKAO_ALLOW_ALL_USERS", raising=False)
    from gateway.config import PlatformConfig
    extra = {"skill_secret": "topsecret", "allow_all_users": True}
    extra.update(extra_overrides)
    cfg = PlatformConfig(enabled=True, extra=extra)
    return KakaoAdapter(cfg)


# ---------------------------------------------------------------------------
# 1. Shared-secret verification
# ---------------------------------------------------------------------------

class TestSharedSecret:

    def test_matching_secret_passes(self):
        assert verify_shared_secret("s3cret", "s3cret")

    def test_mismatched_secret_rejected(self):
        assert not verify_shared_secret("wrong", "s3cret")

    def test_missing_header_rejected(self):
        assert not verify_shared_secret("", "s3cret")
        assert not verify_shared_secret(None, "s3cret")

    def test_missing_expected_secret_rejected(self):
        # Never authorize if the adapter itself has no secret configured.
        assert not verify_shared_secret("anything", "")


# ---------------------------------------------------------------------------
# 2. Allowlist gating
# ---------------------------------------------------------------------------

class TestAllowlist:

    def test_allow_all_short_circuits(self):
        assert _allowed("anyone", allow_all=True, allowed_users=set())

    def test_user_in_allowlist_passes(self):
        assert _allowed("U1", allow_all=False, allowed_users={"U1"})

    def test_user_not_in_allowlist_rejected(self):
        assert not _allowed("U2", allow_all=False, allowed_users={"U1"})

    def test_empty_chat_id_rejected(self):
        assert not _allowed("", allow_all=False, allowed_users={"U1"})


# ---------------------------------------------------------------------------
# 3. SkillResponse building
# ---------------------------------------------------------------------------

class TestSkillResponse:

    def test_short_text_single_output(self):
        resp = build_skill_response("hello")
        assert resp["version"] == "2.0"
        outputs = resp["template"]["outputs"]
        assert len(outputs) == 1
        assert outputs[0]["simpleText"]["text"] == "hello"

    def test_markdown_stripped(self):
        resp = build_skill_response("**bold** [x](https://x.com)")
        text = resp["template"]["outputs"][0]["simpleText"]["text"]
        assert "**" not in text
        assert "https://x.com" in text

    def test_caps_at_three_outputs(self):
        long_text = "\n\n".join(["x" * 800 for _ in range(20)])
        resp = build_skill_response(long_text)
        assert len(resp["template"]["outputs"]) <= KAKAO_MAX_OUTPUTS

    def test_each_output_within_char_limit(self):
        long_text = "y" * 5000
        resp = build_skill_response(long_text)
        for out in resp["template"]["outputs"]:
            assert len(out["simpleText"]["text"]) <= KAKAO_SIMPLETEXT_MAX_CHARS

    def test_empty_text_yields_one_empty_output(self):
        resp = build_skill_response("")
        assert resp["template"]["outputs"] == [{"simpleText": {"text": ""}}]

    def test_callback_ack_omits_template(self):
        ack = build_callback_ack()
        assert ack["version"] == "2.0"
        assert ack["useCallback"] is True
        assert "template" not in ack

    def test_split_short_single_chunk(self):
        assert split_for_kakao("hi") == ["hi"]

    def test_split_long_paragraph_boundary(self):
        text = "para1\n\npara2\n\npara3"
        chunks = split_for_kakao(text, max_chars=8)
        assert all(len(c) <= 8 for c in chunks)
        assert len(chunks) >= 2


# ---------------------------------------------------------------------------
# 4. send() routing
# ---------------------------------------------------------------------------

class TestSendRouting:

    def test_system_bypass_swallowed_without_consuming_slot(self, monkeypatch):
        ad = _make_adapter(monkeypatch)

        async def _run():
            fut = asyncio.get_running_loop().create_future()
            ad._pending_turns["U1"] = _kakao._PendingTurn(future=fut, callback_url=None, created_at=0)
            result = await ad.send("U1", "⚡ Interrupting current run")
            assert result.success
            assert not fut.done()  # slot preserved for the real answer

        asyncio.run(_run())

    def test_send_fulfills_open_slot(self, monkeypatch):
        ad = _make_adapter(monkeypatch)

        async def _run():
            fut = asyncio.get_running_loop().create_future()
            ad._pending_turns["U1"] = _kakao._PendingTurn(future=fut, callback_url=None, created_at=0)
            result = await ad.send("U1", "the answer")
            assert result.success
            assert fut.done()
            assert fut.result() == "the answer"

        asyncio.run(_run())

    def test_send_without_open_slot_holds_for_next_utterance(self, monkeypatch):
        ad = _make_adapter(monkeypatch)
        result = asyncio.run(ad.send("Unknown", "hello"))
        assert result.success
        assert result.message_id is None
        assert ad._late_answers["Unknown"][1] == "hello"

    def test_send_when_slot_already_used_holds_and_joins(self, monkeypatch):
        ad = _make_adapter(monkeypatch)

        async def _run():
            fut = asyncio.get_running_loop().create_future()
            fut.set_result("already delivered")
            ad._pending_turns["U1"] = _kakao._PendingTurn(future=fut, callback_url=None, created_at=0)
            # e.g. an approval ack consumed the slot; the real answer
            # arrives afterwards -- it must be held, not dropped.
            result = await ad.send("U1", "the real answer")
            assert result.success
            assert ad._late_answers["U1"][1] == "the real answer"
            # A further follow-up joins rather than overwrites.
            await ad.send("U1", "and one more thing")
            assert ad._late_answers["U1"][1] == "the real answer\n\nand one more thing"

        asyncio.run(_run())

    def test_next_utterance_retrieves_held_answer(self, monkeypatch):
        ad = _make_adapter(monkeypatch)

        async def _run():
            ad._hold_for_next_utterance("U1", "held answer")
            resp = await ad._process_skill_request(
                {"userRequest": {"utterance": "hi", "user": {"id": "U1"}}}
            )
            assert resp == _kakao.build_skill_response("held answer")
            assert "U1" not in ad._late_answers

        asyncio.run(_run())

    def test_stale_held_answer_is_dropped(self, monkeypatch):
        ad = _make_adapter(monkeypatch)
        ad._late_answers["U1"] = (0.0, "ancient answer")  # far past freshness

        async def _run():
            resp = await ad._process_skill_request(
                {"userRequest": {"utterance": "hi", "user": {"id": "U1"}}}
            )
            # Falls through to normal turn handling, not the stale answer.
            assert resp != _kakao.build_skill_response("ancient answer")
            assert "U1" not in ad._late_answers

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# 5. Full webhook turn lifecycle
# ---------------------------------------------------------------------------

class TestProcessSkillRequest:

    def test_fast_answer_returned_inline(self, monkeypatch):
        ad = _make_adapter(monkeypatch)
        ad.sync_timeout = 2.0

        async def fast_handler(event):
            # Real hermes calls send() from a background task; here we
            # simulate an answer that's ready before the sync budget expires.
            await ad.send(event.source.chat_id, "hi there")

        ad.handle_message = fast_handler

        async def _run():
            payload = _skill_payload("U1", "hello")
            resp = await ad._process_skill_request(payload)
            assert resp["template"]["outputs"][0]["simpleText"]["text"] == "hi there"
            assert "U1" not in ad._pending_turns

        asyncio.run(_run())

    def test_slow_answer_with_callback_returns_ack_and_schedules_delivery(self, monkeypatch):
        ad = _make_adapter(monkeypatch)
        ad.sync_timeout = 0.02

        async def slow_handler(event):
            async def _later():
                await asyncio.sleep(0.1)
                await ad.send(event.source.chat_id, "late answer")
            asyncio.create_task(_later())

        ad.handle_message = slow_handler
        ad._deliver_via_callback = AsyncMock()

        async def _run():
            payload = _skill_payload("U1", "hello", callback_url="https://kakao.example/cb/1")
            resp = await ad._process_skill_request(payload)
            assert resp == build_callback_ack(ad.callback_waiting_text)
            ad._deliver_via_callback.assert_called_once()
            call_args = ad._deliver_via_callback.call_args.args
            assert call_args[0] == "U1"
            assert call_args[1] == "https://kakao.example/cb/1"

        asyncio.run(_run())

    def test_slow_answer_without_callback_returns_timeout_text(self, monkeypatch):
        ad = _make_adapter(monkeypatch)
        ad.sync_timeout = 0.02

        async def slow_handler(event):
            async def _later():
                await asyncio.sleep(0.2)
                with contextlib.suppress(Exception):
                    await ad.send(event.source.chat_id, "too late")
            asyncio.create_task(_later())

        ad.handle_message = slow_handler

        async def _run():
            payload = _skill_payload("U1", "hello")
            resp = await ad._process_skill_request(payload)
            assert resp == build_skill_response(ad.no_callback_timeout_text)
            # The slot stays open so the in-flight answer can land; the
            # stash continuation then holds it for the next utterance.
            await asyncio.sleep(0.3)
            assert ad._late_answers["U1"][1] == "too late"
            assert "U1" not in ad._pending_turns

        asyncio.run(_run())

    def test_second_request_supersedes_first(self, monkeypatch):
        ad = _make_adapter(monkeypatch)
        ad.sync_timeout = 5.0
        registered = None

        async def hang_handler(event):
            # Mirrors real hermes: handle_message spawns background work and
            # returns immediately without resolving the future itself.
            registered.set()

        ad.handle_message = hang_handler

        async def _run():
            nonlocal registered
            registered = asyncio.Event()
            payload1 = _skill_payload("U1", "first")
            task1 = asyncio.create_task(ad._process_skill_request(payload1))
            await asyncio.wait_for(registered.wait(), timeout=2)
            registered.clear()

            payload2 = _skill_payload("U1", "second")
            task2 = asyncio.create_task(ad._process_skill_request(payload2))
            await asyncio.wait_for(registered.wait(), timeout=2)

            result1 = await asyncio.wait_for(task1, timeout=2)
            assert result1 == build_skill_response(ad.superseded_text)

            task2.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task2

        asyncio.run(_run())

    def test_unknown_user_returns_bad_request_text(self, monkeypatch):
        ad = _make_adapter(monkeypatch)

        async def _run():
            payload = _skill_payload("", "hello")
            resp = await ad._process_skill_request(payload)
            assert resp == build_skill_response(ad.bad_request_text)

        asyncio.run(_run())

    def test_unauthorized_user_rejected(self, monkeypatch):
        ad = _make_adapter(monkeypatch, allow_all_users=False, allowed_users=["U1"])

        async def _run():
            payload = _skill_payload("Uintruder", "hello")
            resp = await ad._process_skill_request(payload)
            assert resp == build_skill_response(ad.unauthorized_text)
            assert "Uintruder" not in ad._pending_turns

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# 6. Register() metadata + plugin entry points
# ---------------------------------------------------------------------------

class TestRegister:

    class _FakeCtx:
        def __init__(self):
            self.kwargs = None

        def register_platform(self, **kw):
            self.kwargs = kw

    def test_register_calls_register_platform(self):
        ctx = self._FakeCtx()
        register(ctx)
        assert ctx.kwargs is not None
        assert ctx.kwargs["name"] == "kakao"
        assert ctx.kwargs["label"] == "Kakao"

    def test_register_advertises_required_env(self):
        ctx = self._FakeCtx()
        register(ctx)
        assert ctx.kwargs["required_env"] == ["KAKAO_SKILL_SECRET"]

    def test_register_wires_allowlist_envs(self):
        ctx = self._FakeCtx()
        register(ctx)
        assert ctx.kwargs["allowed_users_env"] == "KAKAO_ALLOWED_USERS"
        assert ctx.kwargs["allow_all_env"] == "KAKAO_ALLOW_ALL_USERS"

    def test_register_provides_standalone_sender(self):
        ctx = self._FakeCtx()
        register(ctx)
        assert callable(ctx.kwargs["standalone_sender_fn"])

    def test_register_has_no_cron_home_channel(self):
        # No proactive delivery exists on this platform -- cron
        # deliver=kakao targets are intentionally unsupported.
        ctx = self._FakeCtx()
        register(ctx)
        assert ctx.kwargs.get("cron_deliver_env_var", "") == ""

    def test_register_factory_yields_kakao_adapter(self, monkeypatch):
        monkeypatch.delenv("KAKAO_SKILL_SECRET", raising=False)
        ctx = self._FakeCtx()
        register(ctx)
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(enabled=True, extra={"skill_secret": "s"})
        ad = ctx.kwargs["adapter_factory"](cfg)
        assert isinstance(ad, KakaoAdapter)

    def test_max_message_length_below_simpletext_limit(self):
        ctx = self._FakeCtx()
        register(ctx)
        assert ctx.kwargs["max_message_length"] <= KAKAO_SIMPLETEXT_MAX_CHARS

    def test_supports_async_delivery_is_false(self):
        # No channel exists to push a background-completion notification
        # after a turn's one delivery slot has been used.
        assert KakaoAdapter.supports_async_delivery is False


class TestStandaloneSend:

    def test_always_returns_error(self, monkeypatch):
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(enabled=True, extra={"skill_secret": "s"})
        result = asyncio.run(_standalone_send(cfg, "U1", "hi"))
        assert "error" in result
        assert "request" in result["error"].lower() or "proactive" in result["error"].lower()


class TestEnvEnablement:

    def test_returns_none_without_secret(self, monkeypatch):
        monkeypatch.delenv("KAKAO_SKILL_SECRET", raising=False)
        assert _env_enablement() is None

    def test_returns_dict_with_secret(self, monkeypatch):
        monkeypatch.setenv("KAKAO_SKILL_SECRET", "s")
        for k in ("KAKAO_PORT", "KAKAO_HOST", "KAKAO_PUBLIC_URL", "KAKAO_SECRET_HEADER", "KAKAO_BOT_ID"):
            monkeypatch.delenv(k, raising=False)
        assert _env_enablement() == {}

    def test_seeds_port_and_public_url(self, monkeypatch):
        monkeypatch.setenv("KAKAO_SKILL_SECRET", "s")
        monkeypatch.setenv("KAKAO_PORT", "9000")
        monkeypatch.setenv("KAKAO_PUBLIC_URL", "https://tunnel.example.com")
        result = _env_enablement()
        assert result["port"] == 9000
        assert result["public_url"] == "https://tunnel.example.com"


class TestCheckRequirementsAndValidate:

    def test_rejects_without_secret(self, monkeypatch):
        monkeypatch.delenv("KAKAO_SKILL_SECRET", raising=False)
        assert not check_requirements()

    def test_accepts_with_secret_and_aiohttp(self, monkeypatch):
        monkeypatch.setenv("KAKAO_SKILL_SECRET", "s")
        assert check_requirements()  # aiohttp is a project dependency

    def test_validate_config_from_extra(self):
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(enabled=True, extra={"skill_secret": "s"})
        assert validate_config(cfg)

    def test_validate_config_rejects_empty(self, monkeypatch):
        monkeypatch.delenv("KAKAO_SKILL_SECRET", raising=False)
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(enabled=True, extra={})
        assert not validate_config(cfg)

    def test_is_connected_matches_validate_config(self, monkeypatch):
        monkeypatch.delenv("KAKAO_SKILL_SECRET", raising=False)
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(enabled=True, extra={"skill_secret": "s"})
        assert is_connected(cfg) == validate_config(cfg)


class TestAdapterInit:

    def test_init_from_config_extra(self, monkeypatch):
        for k in ("KAKAO_SKILL_SECRET", "KAKAO_PORT", "KAKAO_SECRET_HEADER"):
            monkeypatch.delenv(k, raising=False)
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(
            enabled=True,
            extra={
                "skill_secret": "tok",
                "secret_header": "X-Custom",
                "port": 7777,
                "public_url": "https://x.example.com",
                "allowed_users": ["U1", "U2"],
            },
        )
        ad = KakaoAdapter(cfg)
        assert ad.skill_secret == "tok"
        assert ad.secret_header == "X-Custom"
        assert ad.webhook_port == 7777
        assert ad.public_base_url == "https://x.example.com"
        assert ad.allowed_users == {"U1", "U2"}

    def test_env_overrides_extra(self, monkeypatch):
        monkeypatch.setenv("KAKAO_SKILL_SECRET", "env-secret")
        monkeypatch.setenv("KAKAO_PORT", "1234")
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(
            enabled=True,
            extra={"skill_secret": "extra-secret", "port": 5555},
        )
        ad = KakaoAdapter(cfg)
        assert ad.skill_secret == "env-secret"
        assert ad.webhook_port == 1234

    def test_default_secret_header(self, monkeypatch):
        monkeypatch.delenv("KAKAO_SECRET_HEADER", raising=False)
        monkeypatch.setenv("KAKAO_SKILL_SECRET", "s")
        from gateway.config import PlatformConfig
        ad = KakaoAdapter(PlatformConfig(enabled=True))
        assert ad.secret_header == _kakao.DEFAULT_SECRET_HEADER

    def test_csv_allowlist_parsed(self, monkeypatch):
        monkeypatch.setenv("KAKAO_SKILL_SECRET", "s")
        monkeypatch.setenv("KAKAO_ALLOWED_USERS", "U1, U2,U3")
        from gateway.config import PlatformConfig
        ad = KakaoAdapter(PlatformConfig(enabled=True))
        assert ad.allowed_users == {"U1", "U2", "U3"}

    def test_get_chat_info_is_always_dm(self, monkeypatch):
        monkeypatch.setenv("KAKAO_SKILL_SECRET", "s")
        from gateway.config import PlatformConfig
        ad = KakaoAdapter(PlatformConfig(enabled=True))
        info = asyncio.run(ad.get_chat_info("U123"))
        assert info["type"] == "dm"

    def test_format_message_strips_markdown(self, monkeypatch):
        ad = _make_adapter(monkeypatch)
        out = ad.format_message("**bold** [link](https://x.com)")
        assert "**" not in out
        assert "https://x.com" in out


class TestConnectRequiresSecret:

    def test_connect_fails_without_secret(self, monkeypatch):
        monkeypatch.delenv("KAKAO_SKILL_SECRET", raising=False)
        from gateway.config import PlatformConfig
        ad = KakaoAdapter(PlatformConfig(enabled=True, extra={}))
        ok = asyncio.run(ad.connect())
        assert ok is False
        assert ad._fatal_error_code == "config_missing"


class TestSystemBypass:

    def test_bypass_prefixes_recognized(self):
        assert _is_system_bypass("⚡ Interrupting current run")
        assert _is_system_bypass("⏳ Queued — agent is busy")
        assert _is_system_bypass("⏩ Steered toward new task")
        assert _is_system_bypass("💾 Background review complete")
        assert not _is_system_bypass("Hello world")
        assert not _is_system_bypass("")


# ---------------------------------------------------------------------------
# 7. Callback delivery machinery (_deliver_via_callback / _post_callback)
# ---------------------------------------------------------------------------

class TestCallbackDelivery:

    def _armed(self, monkeypatch, post_result):
        ad = _make_adapter(monkeypatch)
        ad.callback_timeout = 0.05
        posts = []

        async def fake_post(chat_id, url, content, ack_at=0.0, quick_replies=None):
            posts.append({"content": content, "quick_replies": quick_replies, "url": url})
            return post_result

        ad._post_callback = fake_post
        return ad, posts

    def test_answer_in_window_posts_and_clears_slot(self, monkeypatch):
        ad, posts = self._armed(monkeypatch, 200)

        async def _run():
            fut = asyncio.get_running_loop().create_future()
            ad._pending_turns["U1"] = _kakao._PendingTurn(future=fut, callback_url="u", created_at=0)
            task = asyncio.create_task(ad._deliver_via_callback("U1", "https://bot-api.kakao.com/cb", fut))
            await asyncio.sleep(0)
            fut.set_result("the answer")
            await task
            assert posts[0]["content"] == "the answer"
            assert posts[0]["quick_replies"] is None
            assert "U1" not in ad._pending_turns
            assert "U1" not in ad._late_answers  # 200 → not re-stashed

        asyncio.run(_run())

    def test_window_expires_sends_notice_and_stashes(self, monkeypatch):
        ad, posts = self._armed(monkeypatch, 200)

        async def _run():
            fut = asyncio.get_running_loop().create_future()
            ad._pending_turns["U1"] = _kakao._PendingTurn(future=fut, callback_url="u", created_at=0)
            await ad._deliver_via_callback("U1", "https://bot-api.kakao.com/cb", fut)
            # notice went out with the retrieval button
            assert posts[0]["content"] == ad.late_answer_notice_text
            assert posts[0]["quick_replies"] == [RETRIEVE_QUICK_REPLY]
            # answer completes late → stash continuation holds it
            fut.set_result("late answer")
            await asyncio.sleep(0.05)
            assert ad._late_answers["U1"][1] == "late answer"

        asyncio.run(_run())

    def test_rejected_token_restashes_answer(self, monkeypatch):
        ad, posts = self._armed(monkeypatch, 400)

        async def _run():
            fut = asyncio.get_running_loop().create_future()
            ad._pending_turns["U1"] = _kakao._PendingTurn(future=fut, callback_url="u", created_at=0)
            task = asyncio.create_task(ad._deliver_via_callback("U1", "https://bot-api.kakao.com/cb", fut))
            await asyncio.sleep(0)
            fut.set_result("the answer")
            await task
            # 400 → not lost, held for next utterance
            assert ad._late_answers["U1"][1] == "the answer"

        asyncio.run(_run())


class TestPostCallbackSSRFGuard:

    def test_non_kakao_host_refused(self, monkeypatch):
        ad = _make_adapter(monkeypatch)

        async def _run():
            status = await ad._post_callback("U1", "https://evil.example.com/cb", "x")
            assert status is None  # refused before any network call

        asyncio.run(_run())

    def test_http_scheme_refused(self, monkeypatch):
        ad = _make_adapter(monkeypatch)

        async def _run():
            status = await ad._post_callback("U1", "http://bot-api.kakao.com/cb", "x")
            assert status is None

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# 8. Turn-slot identity: supersession must not evict the newer turn
# ---------------------------------------------------------------------------

class TestTurnSlotIdentity:

    def test_stale_cleanup_does_not_evict_newer_turn(self, monkeypatch):
        ad = _make_adapter(monkeypatch)

        async def _run():
            old = asyncio.get_running_loop().create_future()
            new = asyncio.get_running_loop().create_future()
            new_turn = _kakao._PendingTurn(future=new, callback_url=None, created_at=1)
            ad._pending_turns["U1"] = new_turn
            # old turn's cleanup runs with the OLD future — must be a no-op
            ad._pop_turn_if_current("U1", old)
            assert ad._pending_turns.get("U1") is new_turn

        asyncio.run(_run())

    def test_interrupt_resolves_open_slot(self, monkeypatch):
        ad = _make_adapter(monkeypatch)

        async def _run():
            fut = asyncio.get_running_loop().create_future()
            ad._pending_turns["U1"] = _kakao._PendingTurn(future=fut, callback_url=None, created_at=0)
            await ad.interrupt_session_activity("", "U1")
            assert fut.done() and fut.result() == ad.interrupted_text
            assert "U1" not in ad._pending_turns

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# 9. Malformed payloads and markdown stripping
# ---------------------------------------------------------------------------

class TestMalformedPayloads:

    def test_empty_payload_returns_bad_request(self, monkeypatch):
        ad = _make_adapter(monkeypatch)
        resp = asyncio.run(ad._process_skill_request({}))
        assert resp == build_skill_response(ad.bad_request_text)

    def test_null_user_request_returns_bad_request(self, monkeypatch):
        ad = _make_adapter(monkeypatch)
        resp = asyncio.run(ad._process_skill_request({"userRequest": None}))
        assert resp == build_skill_response(ad.bad_request_text)

    def test_non_string_user_id_coerced(self, monkeypatch):
        ad = _make_adapter(monkeypatch)
        # integer id must not raise; it is coerced to str and handled
        resp = asyncio.run(
            ad._process_skill_request({"userRequest": {"utterance": "hi", "user": {"id": 12345}}})
        )
        assert resp["version"] == "2.0"


class TestMarkdownStripping:

    def test_bold_and_italic_stripped(self):
        assert "**" not in strip_markdown_preserving_urls("**bold** and *italic*")

    def test_inline_code_unfenced(self):
        assert "`" not in strip_markdown_preserving_urls("run `hermes gateway`")

    def test_heading_prefix_stripped(self):
        out = strip_markdown_preserving_urls("# Heading")
        assert not out.lstrip().startswith("#")

    def test_link_becomes_label_and_url(self):
        out = strip_markdown_preserving_urls("[docs](https://x.com)")
        assert "https://x.com" in out
        assert "](" not in out

    def test_quick_reply_present_only_when_passed(self):
        with_qr = build_skill_response("x", quick_replies=[RETRIEVE_QUICK_REPLY])
        assert with_qr["template"]["quickReplies"] == [RETRIEVE_QUICK_REPLY]
        assert "quickReplies" not in build_skill_response("x")["template"]
