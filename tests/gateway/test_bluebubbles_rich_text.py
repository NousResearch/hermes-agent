"""Tests for BlueBubbles native iMessage text styles, send effects and unsend."""
import httpx
import pytest

from gateway.config import PlatformConfig
from gateway.platforms.helpers import strip_markdown

from plugins.platforms.bluebubbles.rich_text import (
    IMESSAGE_EFFECTS, extract_effect, mark_emphasis, styled_chunk,
)

CHAT = "iMessage;-;user@example.com"


def _styled(text):
    plain, runs, _ = styled_chunk(mark_emphasis(text))
    return plain, runs


def _adapter(monkeypatch, *, private_api=True, helper=True, macos=15):
    monkeypatch.setenv("BLUEBUBBLES_SERVER_URL", "http://localhost:1234")
    monkeypatch.setenv("BLUEBUBBLES_PASSWORD", "secret")
    from plugins.platforms.bluebubbles.adapter import BlueBubblesAdapter

    adapter = BlueBubblesAdapter(PlatformConfig(
        enabled=True, extra={"server_url": "http://localhost:1234", "password": "secret"}))
    adapter._private_api_enabled, adapter._helper_connected, adapter._macos_major = private_api, helper, macos
    adapter.client = object()
    posts = []

    async def fake_post(path, payload):
        posts.append((path, payload))
        return {"status": 200, "data": {"guid": f"guid-{len(posts)}"}}

    monkeypatch.setattr(adapter, "_api_post", fake_post)
    return adapter, posts


def _http_error(status):
    request = httpx.Request("POST", "http://localhost:1234/api/v1/message/text")
    return httpx.HTTPStatusError("refused", request=request, response=httpx.Response(status, request=request))


class TestEffectMarker:
    def test_known_effect_is_removed_and_resolved(self):
        assert extract_effect("Happy birthday! [[effect:confetti]]") == (
            "Happy birthday!", "com.apple.messages.effect.CKConfettiEffect")

    def test_marker_mid_sentence_keeps_one_space(self):
        assert extract_effect("Big news [[effect:slam]] we shipped")[0] == "Big news we shipped"

    def test_names_ignore_case_spaces_and_underscores(self):
        assert extract_effect("[[effect: Invisible Ink]] the answer is 42")[1] == IMESSAGE_EFFECTS["invisibleink"]
        assert extract_effect("[[Effect:shooting_star]] wow")[1] == IMESSAGE_EFFECTS["shootingstar"]

    def test_unknown_effect_is_removed_without_an_id(self):
        assert extract_effect("[[effect:glitter]] hi") == ("hi", None)

    def test_first_known_marker_wins_and_all_are_removed(self):
        text, effect = extract_effect("[[effect:nope]] a [[effect:heart]] b [[effect:lasers]]")
        assert (text, effect) == ("a b", IMESSAGE_EFFECTS["heart"])

    def test_text_without_marker_is_untouched(self):
        assert extract_effect("see [[as_document]] and [link](x)") == ("see [[as_document]] and [link](x)", None)


class TestTextStyleRuns:
    def test_plain_text_matches_strip_markdown(self):
        for text in ("**Heads up**: the deploy is *done*", "## Title\nbody with `code`",
                     "[docs](https://example.com) and __init__", "FEATURE_FLAG_NAME stays", "nothing styled"):
            assert _styled(text)[0] == strip_markdown(text, keep_link_targets=True)

    def test_bold_and_italic_runs(self):
        assert _styled("**Heads up**: the build is *green*") == ("Heads up: the build is green", [
            {"start": 0, "length": 8, "styles": ["bold"]},
            {"start": 23, "length": 5, "styles": ["italic"]}])

    def test_offsets_count_utf16_units(self):
        plain, runs = _styled("café 👋 **now**")
        assert plain == "café 👋 now"
        assert runs == [{"start": 8, "length": 3, "styles": ["bold"]}]

    def test_nested_bold_italic_is_one_run(self):
        assert _styled("***both*** here")[1] == [{"start": 0, "length": 4, "styles": ["bold", "italic"]}]

    def test_strikethrough_markers_become_a_style(self):
        assert _styled("price ~~$20~~ $15") == ("price $20 $15", [
            {"start": 6, "length": 3, "styles": ["strikethrough"]}])

    def test_underscores_in_identifiers_are_not_styled(self):
        assert _styled("set MAX_RETRY_COUNT in config_file.json")[1] == []

    def test_style_left_open_carries_into_the_next_bubble(self):
        first = mark_emphasis("**one\n\ntwo**")
        head, tail = first.split("\n\n")
        plain1, runs1, active = styled_chunk(head)
        plain2, runs2, _ = styled_chunk(tail, active)
        assert (plain1, runs1) == ("one", [{"start": 0, "length": 3, "styles": ["bold"]}])
        assert (plain2, runs2) == ("two", [{"start": 0, "length": 3, "styles": ["bold"]}])

    def test_input_sentinels_are_discarded(self):
        assert _styled("abc") == ("abc", [])


class TestStyledSends:
    @pytest.mark.asyncio
    async def test_styles_ride_as_text_formatting_on_macos_15(self, monkeypatch):
        adapter, posts = _adapter(monkeypatch)
        result = await adapter.send(CHAT, "Meet at **6pm**")
        assert result.success and result.message_id == "guid-1"
        assert posts == [("/api/v1/message/text", {
            "chatGuid": CHAT, "tempGuid": posts[0][1]["tempGuid"], "message": "Meet at 6pm",
            "method": "private-api", "textFormatting": [{"start": 8, "length": 3, "styles": ["bold"]}]})]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("private_api, helper, macos", [(False, False, 15), (True, False, 15), (True, True, 14)])
    async def test_without_helper_or_on_old_macos_the_send_is_unchanged(self, monkeypatch, private_api, helper, macos):
        adapter, posts = _adapter(monkeypatch, private_api=private_api, helper=helper, macos=macos)
        await adapter.send(CHAT, "Meet at **6pm**\n\nsee ~~you~~")
        assert [p["message"] for _, p in posts] == ["Meet at 6pm", "see ~~you~~"]
        assert not any("textFormatting" in p or "method" in p for _, p in posts)

    @pytest.mark.asyncio
    async def test_unstyled_reply_takes_the_plain_path(self, monkeypatch):
        adapter, posts = _adapter(monkeypatch)
        await adapter.send(CHAT, "just words")
        assert "method" not in posts[0][1] and "textFormatting" not in posts[0][1]

    @pytest.mark.asyncio
    async def test_each_paragraph_bubble_gets_its_own_offsets(self, monkeypatch):
        adapter, posts = _adapter(monkeypatch)
        await adapter.send(CHAT, "First *one*\n\nthen **two**")
        assert [(p["message"], p.get("textFormatting")) for _, p in posts] == [
            ("First one", [{"start": 6, "length": 3, "styles": ["italic"]}]),
            ("then two", [{"start": 5, "length": 3, "styles": ["bold"]}])]

    @pytest.mark.asyncio
    async def test_rejected_styles_are_resent_as_plain_text(self, monkeypatch):
        adapter, _ = _adapter(monkeypatch)
        calls = []

        async def refuse_rich(path, payload):
            calls.append(payload)
            if "textFormatting" in payload:
                raise _http_error(400)
            return {"data": {"guid": "plain-guid"}}

        monkeypatch.setattr(adapter, "_api_post", refuse_rich)
        result = await adapter.send(CHAT, "Meet at **6pm**")
        assert result.success and result.message_id == "plain-guid"
        assert "textFormatting" not in calls[1] and "method" not in calls[1]
        assert calls[1]["message"] == "Meet at 6pm"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("error", [_http_error(500), httpx.ReadTimeout("slow")])
    async def test_server_errors_are_not_resent(self, monkeypatch, error):
        adapter, _ = _adapter(monkeypatch)
        calls = []

        async def fail(path, payload):
            calls.append(payload)
            raise error

        monkeypatch.setattr(adapter, "_api_post", fail)
        result = await adapter.send(CHAT, "Meet at **6pm**")
        assert not result.success and len(calls) == 1
        assert "secret" not in (result.error or "")


class TestEffectSends:
    @pytest.mark.asyncio
    async def test_effect_rides_on_the_first_bubble_only(self, monkeypatch):
        adapter, posts = _adapter(monkeypatch)
        await adapter.send(CHAT, "Happy birthday! [[effect:balloons]]\n\nHave a great one")
        assert posts[0][1]["effectId"] == IMESSAGE_EFFECTS["balloons"]
        assert posts[0][1]["method"] == "private-api" and posts[0][1]["message"] == "Happy birthday!"
        assert "effectId" not in posts[1][1]

    @pytest.mark.asyncio
    async def test_effect_needs_the_private_api(self, monkeypatch):
        adapter, posts = _adapter(monkeypatch, private_api=False, helper=False)
        await adapter.send(CHAT, "Happy birthday! [[effect:balloons]]")
        assert posts[0][1]["message"] == "Happy birthday!"
        assert "effectId" not in posts[0][1] and "method" not in posts[0][1]

    @pytest.mark.asyncio
    async def test_effect_works_below_macos_15_without_styles(self, monkeypatch):
        adapter, posts = _adapter(monkeypatch, macos=14)
        await adapter.send(CHAT, "**Shipped** [[effect:fireworks]]")
        assert posts[0][1]["effectId"] == IMESSAGE_EFFECTS["fireworks"]
        assert posts[0][1]["message"] == "Shipped" and "textFormatting" not in posts[0][1]

    @pytest.mark.asyncio
    async def test_marker_alone_is_not_sent_as_an_empty_bubble(self, monkeypatch):
        adapter, posts = _adapter(monkeypatch)
        result = await adapter.send(CHAT, "[[effect:confetti]]")
        assert not result.success and posts == []

    @pytest.mark.asyncio
    async def test_reply_threading_and_effect_combine(self, monkeypatch):
        adapter, posts = _adapter(monkeypatch)
        await adapter.send(CHAT, "Yes! [[effect:heart]]", reply_to="parent-guid")
        assert posts[0][1]["selectedMessageGuid"] == "parent-guid"
        assert posts[0][1]["effectId"] == IMESSAGE_EFFECTS["heart"]


    @pytest.mark.asyncio
    async def test_invisible_ink_is_never_sent_as_plain_text_without_the_helper(self, monkeypatch):
        adapter, posts = _adapter(monkeypatch, private_api=False, helper=False)
        result = await adapter.send(CHAT, "[[effect:invisibleink]] The butler did it")
        assert not result.success and "hidden text was not sent" in result.error
        assert posts == []

    @pytest.mark.asyncio
    async def test_rejected_invisible_ink_is_not_resent_plain(self, monkeypatch):
        adapter, _ = _adapter(monkeypatch)
        calls = []

        async def refuse(path, payload):
            calls.append(payload)
            raise _http_error(400)

        monkeypatch.setattr(adapter, "_api_post", refuse)
        result = await adapter.send(CHAT, "[[effect:invisibleink]] The butler did it")
        assert not result.success and len(calls) == 1


class TestUnsend:
    @pytest.mark.asyncio
    async def test_unsend_posts_to_the_message_route(self, monkeypatch):
        adapter, posts = _adapter(monkeypatch, macos=13)
        assert await adapter.delete_message(CHAT, "ABC-123") is True
        assert posts == [("/api/v1/message/ABC-123/unsend", {"partIndex": 0})]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("private_api, helper, macos, message_id", [
        (False, False, 15, "ABC"), (True, False, 15, "ABC"), (True, True, 12, "ABC"), (True, True, 15, "ok"),
        (True, True, 15, "")])
    async def test_unsend_is_refused_when_it_cannot_work(self, monkeypatch, private_api, helper, macos, message_id):
        adapter, posts = _adapter(monkeypatch, private_api=private_api, helper=helper, macos=macos)
        assert await adapter.delete_message(CHAT, message_id) is False
        assert posts == []

    @pytest.mark.asyncio
    async def test_unsend_outside_apples_window_returns_false(self, monkeypatch):
        adapter, _ = _adapter(monkeypatch)

        async def too_late(path, payload):
            raise _http_error(500)

        monkeypatch.setattr(adapter, "_api_post", too_late)
        assert await adapter.delete_message(CHAT, "ABC") is False


class TestServerVersionAndHint:
    @pytest.mark.parametrize("raw, major", [("15.6.1", 15), ("26.1", 26), ("13", 13), ("", 0), (None, 0), ("x", 0)])
    def test_macos_major(self, raw, major):
        from plugins.platforms.bluebubbles.adapter import _macos_major
        assert _macos_major(raw) == major

    @pytest.mark.asyncio
    async def test_connect_reads_the_macos_version(self, monkeypatch):
        adapter, _ = _adapter(monkeypatch, private_api=False, helper=False, macos=0)

        async def fake_get(path):
            return {"data": {"private_api": True, "helper_connected": True, "os_version": "15.3"}}

        monkeypatch.setattr(adapter, "_api_get", fake_get)

        async def no_listener(*args, **kwargs):
            return None

        monkeypatch.setattr("gateway.platforms.shared_ingress.bind_listener", no_listener)
        monkeypatch.setattr(adapter, "_register_webhook", lambda: no_listener())
        assert await adapter.connect() is True
        assert adapter._macos_major == 15 and adapter._text_styles_live
        await adapter._close_client()

    def test_hint_names_every_effect_the_adapter_accepts(self):
        from plugins.platforms.bluebubbles.adapter import _PLATFORM_HINT
        assert all(name in _PLATFORM_HINT for name in IMESSAGE_EFFECTS)
        assert "[[effect:" in _PLATFORM_HINT and "clarify" in _PLATFORM_HINT
