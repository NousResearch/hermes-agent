"""Rich-only triggers follow group policy through real offline PTB dispatch."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from plugins.platforms.telegram import telegram_rich_inbound as rich
from tests.plugins.test_telegram_update_admission import connected, update
from tests.plugins.platforms.telegram.test_rich_inbound_dispatch import drain, rich_update


@pytest.mark.asyncio
async def test_rich_explicit_mention_dispatches_in_require_mention_group(monkeypatch):
    async with connected(monkeypatch, extra={"allowed_chats": ["42"], "require_mention": True}) as (adapter, app, delivered):
        incoming = rich_update(app.bot, group=True, blocks=[
            {"type": "paragraph", "text": "@offline_bot hello"},
        ])
        before = incoming.to_dict()
        await app.process_update(incoming)
        await app.process_update(incoming)
        await drain(adapter)
        assert len(delivered) == 1
        assert delivered[0].text == "hello"
        assert delivered[0].raw_message is incoming.message
        assert delivered[0].allow_gateway_control is False
        assert incoming.to_dict() == before


@pytest.mark.asyncio
@pytest.mark.parametrize("blocks", [
    [{"type": "paragraph", "text": {"type": "url", "text": "link", "url": "https://example.test/?q=@offline_bot"}}],
    [{"type": "paragraph", "text": {"type": "code", "text": "@offline_bot"}}],
    [{"type": "pre", "text": "@offline_bot"}],
])
async def test_rich_code_and_link_destinations_do_not_trigger(monkeypatch, blocks):
    async with connected(monkeypatch, extra={"allowed_chats": ["42"], "require_mention": True,
                                           "mention_patterns": ["offline_bot"]}) as (adapter, app, delivered):
        await app.process_update(rich_update(app.bot, group=True, blocks=blocks))
        await drain(adapter)
        assert delivered == []


@pytest.mark.asyncio
@pytest.mark.parametrize("blocks", [
    [{"type": "paragraph", "text": {"type": "code", "text": "example"}}],
    [{"type": "paragraph", "text": {"type": "url", "text": "", "url": "https://example.test"}}],
    [{"type": "future", "text": "ignored"}],
])
async def test_no_visible_trigger_text_does_not_match_wildcard_pattern(monkeypatch, blocks):
    async with connected(monkeypatch, extra={"require_mention": True, "mention_patterns": ["."]}) as (adapter, app, delivered):
        await app.process_update(rich_update(app.bot, group=True, blocks=blocks))
        await drain(adapter)
        assert delivered == []


@pytest.mark.asyncio
@pytest.mark.parametrize("options,chat", [({}, 42), ({"guest_mode": True}, 43),
                                        ({"bots_require_mention": True, "mention_patterns": ["offline"]}, 42)])
async def test_link_label_is_not_an_explicit_mention(monkeypatch, options, chat):
    async with connected(monkeypatch, extra={"require_mention": True, "allowed_chats": ["42"], **options}) as (adapter, app, delivered):
        incoming = rich_update(app.bot, group=True, chat=chat,
                               fields={"from": {"id": 88, "is_bot": True, "first_name": "Sender"}},
                               blocks=[{"type": "paragraph", "text": {
                                   "type": "url", "text": "@offline_bot", "url": "https://example.test",
                               }}])
        await app.process_update(incoming)
        await drain(adapter)
        assert delivered == []


@pytest.mark.asyncio
@pytest.mark.parametrize("length", [31, 32, 33])
@pytest.mark.parametrize("suffix", ["", "_suffix"], ids=["whole", "longer-handle"])
@pytest.mark.parametrize("own", [False, True], ids=["exclusive-other", "own"])
async def test_rich_mention_length_boundary_through_dispatch(monkeypatch, length, suffix, own):
    handle = "a" * (length - 3) + "bot"
    valid = length <= 32 and not suffix
    text = "😀 wake " + ("@other_bot " if own else "") + f"@{handle}{suffix} hello"
    extra = {"require_mention": True, "exclusive_bot_mentions": True, "mention_patterns": ["wake"]}
    async with connected(monkeypatch, extra=extra) as (adapter, app, delivered):
        if own:
            monkeypatch.setattr(adapter, "_current_bot_username", lambda: handle)
        incoming = rich_update(app.bot, group=True, blocks=[{"type": "paragraph", "text": text}])
        before = incoming.to_dict()
        await app.process_update(incoming)
        await drain(adapter)
        assert bool(delivered) is (valid if own else not valid)
        assert incoming.to_dict() == before
        if delivered:
            assert len(delivered) == 1
            assert delivered[0].raw_message is incoming.message
            assert not delivered[0].allow_gateway_control
        view = rich.RichMessageView(incoming.message)
        encoded = view.text.encode("utf-16-le")
        handles = [encoded[e.offset * 2:(e.offset + e.length) * 2].decode("utf-16-le")
                   for e in view.entities if e.type == "mention"]
        assert handles == (["@other_bot"] if own else []) + ([f"@{handle}"] if valid else [])


@pytest.mark.asyncio
@pytest.mark.parametrize("separator", ["\n", "\r", r"\n", r"\r"], ids=["LF", "CR", "literal-LF", "literal-CR"])
@pytest.mark.parametrize("own", [False, True], ids=["exclusive-other", "own"])
async def test_cell_mentions_use_real_line_boundaries(monkeypatch, separator, own):
    literal = len(separator) == 2
    text = separator + "@offline_bot hello" if own else "wake" + separator + "@other_bot hello"
    extra = {"require_mention": True, "exclusive_bot_mentions": True, "mention_patterns": ["wake"]}
    async with connected(monkeypatch, extra=extra) as (adapter, app, delivered):
        projection = Mock(wraps=rich._project_rich_message)
        monkeypatch.setattr(rich, "_project_rich_message", projection)
        incoming = rich_update(app.bot, group=True, blocks=[{"type": "table", "cells": [[{"text": text}]]}])
        before = incoming.to_dict()
        await app.process_update(incoming)
        await app.process_update(incoming)
        await drain(adapter)
        assert bool(delivered) is (own != literal)
        projection.assert_called_once()
        assert incoming.to_dict() == before
        if delivered:
            escaped = text.replace("\\", "\\\\").replace("\r", r"\r").replace("\n", r"\n")
            if own:
                escaped = escaped.replace("@offline_bot ", "")  # Existing adapter cleanup.
            assert len(delivered) == 1 and delivered[0].text == f"| {escaped} |"
            assert delivered[0].raw_message is incoming.message
            assert not delivered[0].allow_gateway_control


@pytest.mark.asyncio
@pytest.mark.parametrize("separator", ["\n", "\r", r"\n", r"\r"], ids=["LF", "CR", "literal-LF", "literal-CR"])
@pytest.mark.parametrize("link_label", [False, True])
async def test_cell_wake_pattern_uses_semantic_word_boundary(monkeypatch, separator, link_label):
    text = "before" + separator + "wake please"
    inline = {"type": "url", "text": text, "url": "https://example.test/@offline_bot"} if link_label else text
    async with connected(monkeypatch, extra={"require_mention": True, "mention_patterns": [r"\bwake\b"]}) as (adapter, app, delivered):
        await app.process_update(rich_update(app.bot, group=True, blocks=[{"type": "table", "cells": [[{"text": inline}]]}]))
        await drain(adapter)
        assert bool(delivered) is (len(separator) == 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("own", [False, True])
async def test_cell_mention_offsets_after_escapes_astrals_and_excluded_nodes(monkeypatch, own):
    prefix = "😀" + r"\n\r|" + "\n@human wake "
    tail = "\r@other_bot 😀" + ("\n@offline_bot hello" if own else "")
    inline = [prefix, {"type": "url", "text": "🧭\r@offline_bot", "url": "https://example.test/@offline_bot"},
              {"type": "code", "text": "🧪\n@offline_bot"}, {"type": "future"}, tail]
    extra = {"require_mention": True, "exclusive_bot_mentions": True, "mention_patterns": ["wake"]}
    async with connected(monkeypatch, extra=extra) as (adapter, app, delivered):
        incoming = rich_update(app.bot, group=True, blocks=[{"type": "table", "cells": [[{"text": inline}]]}])
        view = rich.RichMessageView(incoming.message)
        assert prefix in view.text and tail in view.text
        encoded = view.text.encode("utf-16-le")
        handles = [encoded[e.offset * 2:(e.offset + e.length) * 2].decode("utf-16-le") for e in view.entities]
        assert handles == ["@human", "@other_bot"] + (["@offline_bot"] if own else [])
        await app.process_update(incoming)
        await drain(adapter)
        assert bool(delivered) is own


@pytest.mark.asyncio
@pytest.mark.parametrize("separator", ["\n", "\r"], ids=["LF", "CR"])
@pytest.mark.parametrize("cut", ["handle", "escape", "beyond", "early"])
async def test_cell_escapes_keep_display_budget_and_safe_trigger_truncation(monkeypatch, separator, cut):
    prefix = "😀" + r"\n\r|" + separator
    escaped_prefix = "😀" + r"\\n\\r\|" + {"\n": r"\n", "\r": r"\r"}[separator]
    if cut in {"handle", "escape"}:
        text = prefix + "@offline_bot" + ("_suffix" if cut == "handle" else separator) + "x" * 200
        retained = "| " + escaped_prefix + "@offline_bot" + ("\\" if cut == "escape" else "")
        limit = len(retained) + len(rich.TRUNCATED) + 1
    else:
        text = (separator + "@offline_bot hello " if cut == "early" else "") + prefix * 30 + "@offline_bot hello"
        limit = 100
    monkeypatch.setattr(rich, "MAX_OUTPUT_CHARS", limit)
    async with connected(monkeypatch, extra={"require_mention": True}) as (adapter, app, delivered):
        incoming = rich_update(app.bot, group=True, blocks=[{"type": "table", "cells": [[{"text": text}]]}])
        view = rich.RichMessageView(incoming.message)
        assert rich.TRUNCATED in view.display_text
        assert len(view.text) <= len(view.display_text) <= limit
        if cut in {"handle", "escape"}:
            assert view.display_text == retained + "\n" + rich.TRUNCATED
        await app.process_update(incoming)
        await drain(adapter)
        assert bool(delivered) is (cut == "early")
        if delivered:
            assert not delivered[0].allow_gateway_control


def observation_store(adapter):
    store = SimpleNamespace(get_or_create_session=Mock(return_value=SimpleNamespace(session_id="offline")),
                            append_to_transcript=Mock(), close_all_db_handles=Mock())
    adapter._session_store = store
    return store


@pytest.mark.asyncio
@pytest.mark.parametrize("text,options,chat,is_bot,dispatch,observe", [
    ("wake please", {}, 42, False, True, False),
    ("chatter", {}, 42, False, False, True),
    ("@offline_bot hello", {"guest_mode": True}, 43, False, True, False),
    ("@offline_bot hello", {"guest_mode": False}, 43, False, False, False),
    ("wake please", {"guest_mode": True}, 43, False, False, False),
    ("/stop@offline_bot", {"guest_mode": True}, 43, False, False, False),
    ("/stop", {"guest_mode": True}, 43, False, False, False),
    ("wake @other_bot", {"exclusive_bot_mentions": True}, 42, False, False, False),
    ("wake @other_bot", {"exclusive_bot_mentions": False}, 42, False, True, False),
    ("@other_bot @offline_bot hello", {"exclusive_bot_mentions": True}, 42, False, True, False),
    ("wake please", {"bots_require_mention": True}, 42, True, False, True),
    ("@offline_bot hello", {"bots_require_mention": True}, 42, True, True, False),
    ("chatter", {"require_mention": False}, 42, False, True, False),
    ("chatter", {"free_response_chats": ["42"]}, 42, False, True, False),
])
async def test_rich_policy_matches_dispatch_and_observation(monkeypatch, text, options, chat, is_bot, dispatch, observe):
    extra = {"allowed_chats": ["42"], "group_allowed_chats": ["42"], "require_mention": True,
             "observe_unmentioned_group_messages": True, "mention_patterns": ["wake"], **options}
    async with connected(monkeypatch, extra=extra) as (adapter, app, delivered):
        store = observation_store(adapter)
        projection = Mock(wraps=rich._project_rich_message)
        monkeypatch.setattr(rich, "_project_rich_message", projection)
        fields = {"from": {"id": 88, "is_bot": is_bot, "first_name": "Sender"}}
        incoming = rich_update(app.bot, group=True, chat=chat, fields=fields,
                               blocks=[{"type": "paragraph", "text": text}])
        before = incoming.to_dict()
        await app.process_update(incoming)
        await app.process_update(incoming)
        await drain(adapter)
        assert bool(delivered) is dispatch
        assert bool(store.append_to_transcript.call_count) is observe
        assert projection.call_count == (0 if chat == 43 and not options.get("guest_mode") else 1)
        assert incoming.to_dict() == before
        if dispatch:
            assert len(delivered) == 1 and delivered[0].allow_gateway_control is False
            if "@other_bot" in text and "@offline_bot" in text:
                assert "@offline_bot" in delivered[0].text
        if observe:
            store.append_to_transcript.assert_called_once()
            assert text in store.append_to_transcript.call_args.args[1]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["user", "allowed_topic", "ignored_topic", "own"])
async def test_rich_guest_cannot_bypass_content_independent_vetoes(monkeypatch, mode):
    extra = {"allowed_chats": ["42"], "guest_mode": True, "require_mention": True,
             "group_allow_from": ["88", "111"], "observe_unmentioned_group_messages": True}
    if mode == "allowed_topic":
        extra["allowed_topics"] = ["7"]
    if mode == "ignored_topic":
        extra["ignored_threads"] = [8]
    async with connected(monkeypatch, extra=extra) as (adapter, app, delivered):
        projection = Mock(side_effect=AssertionError("Hard veto precedes projection"))
        monkeypatch.setattr(rich, "_project_rich_message", projection)
        store = observation_store(adapter)
        incoming = rich_update(app.bot, group=True, chat=43, user={"own": 111, "user": 99}.get(mode, 88),
                               fields={"message_thread_id": 8, "is_topic_message": True},
                               blocks=[{"type": "paragraph", "text": "@offline_bot hello"}])
        await app.process_update(incoming)
        assert not delivered and not adapter._pending_text_batches
        projection.assert_not_called()
        store.append_to_transcript.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("excluded", [
    {"type": "code", "text": ""},
    {"type": "code", "text": "x"},
    {"type": "url", "text": "", "url": "https://example.test"},
    {"type": "future", "text": ""},
])
async def test_excluded_inline_nodes_never_join_handle_fragments(monkeypatch, excluded):
    async with connected(monkeypatch, extra={"require_mention": True}) as (adapter, app, delivered):
        await app.process_update(rich_update(app.bot, group=True, blocks=[
            {"type": "paragraph", "text": ["@offline", excluded, "_bot hello"]},
        ]))
        await drain(adapter)
        assert delivered == []


@pytest.mark.asyncio
@pytest.mark.parametrize("blocks,expected", [
    ([{"type": "paragraph", "text": ["@off", {"type": "bold", "text": "line_bot"}, " hello"]}], True),
    ([{"type": "paragraph", "text": "@offline"}, {"type": "paragraph", "text": "_bot"}], False),
    ([{"type": "table", "cells": [[{"text": "@offline"}, {"text": "_bot"}]]}], False),
    ([{"type": "paragraph", "text": {"type": "url", "text": "wake please", "url": "https://example.test"}}], True),
    ([{"type": "paragraph", "text": "😀 @offline_bot hello @human"}], True),
])
async def test_visible_inline_text_and_structural_boundaries(monkeypatch, blocks, expected):
    async with connected(monkeypatch, extra={"require_mention": True, "mention_patterns": ["wake"]}) as (adapter, app, delivered):
        await app.process_update(rich_update(app.bot, group=True, blocks=blocks))
        await drain(adapter)
        assert bool(delivered) is expected
        if expected and "@human" in delivered[0].text:
            assert "@offline_bot" in delivered[0].text


@pytest.mark.asyncio
@pytest.mark.parametrize("carrier", ["text", "caption"])
@pytest.mark.parametrize("ordinary,trigger", [("ordinary", False), ("@offline_bot ordinary", True)])
async def test_normal_carrier_wins_for_trigger_and_control(monkeypatch, carrier, ordinary, trigger):
    async with connected(monkeypatch, extra={"require_mention": True}) as (adapter, app, delivered):
        projection = Mock(side_effect=AssertionError("Normal carrier wins"))
        monkeypatch.setattr(rich, "_project_rich_message", projection)
        incoming = rich_update(app.bot, group=True, fields={carrier: ordinary},
                               blocks=[{"type": "paragraph", "text": "@offline_bot rich"}])
        await app.process_update(incoming)
        await drain(adapter)
        assert bool(delivered) is trigger
        projection.assert_not_called()
        if trigger:
            assert delivered[0].text == "ordinary" and delivered[0].allow_gateway_control


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [OSError, asyncio.CancelledError])
async def test_projection_preparation_failure_can_retry(monkeypatch, failure):
    async with connected(monkeypatch, extra={"require_mention": True}) as (adapter, app, delivered):
        incoming = rich_update(app.bot, group=True, blocks=[{"type": "paragraph", "text": "@offline_bot hello"}])
        with monkeypatch.context() as broken:
            broken.setattr(rich, "_project_rich_message", Mock(side_effect=failure("projection fault")))
            if failure is asyncio.CancelledError:
                with pytest.raises(asyncio.CancelledError):
                    await app.process_update(incoming)
            else:
                await app.process_update(incoming)
        assert not adapter._seen_update_ids and not adapter._inflight_update_ids
        await app.process_update(incoming)
        await drain(adapter)
        assert len(delivered) == 1


def test_typed_payload_and_raw_payload_share_trigger_projection():
    typed = SimpleNamespace(blocks=[SimpleNamespace(type="paragraph", text=["😀 ", SimpleNamespace(type="bold", text="@offline_bot")])])
    raw = {"blocks": [{"type": "paragraph", "text": ["😀 ", {"type": "bold", "text": "@offline_bot"}]}]}
    typed_message = SimpleNamespace(rich_message=typed, api_kwargs={"rich_message": {"blocks": []}})
    raw_message = SimpleNamespace(api_kwargs={"rich_message": raw})
    a, b = rich.RichMessageView(typed_message), rich.RichMessageView(raw_message)
    assert a.text == b.text == "😀 @offline_bot"
    assert a.display_text == b.display_text
    assert a.entities[0].offset == b.entities[0].offset == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["output", "nodes", "blocks", "depth", "cells"])
async def test_trigger_projection_respects_existing_limits(monkeypatch, mode):
    limits = {"output": ("MAX_OUTPUT_CHARS", 80), "nodes": ("MAX_NODES", 8), "blocks": ("MAX_BLOCKS", 1),
              "depth": ("MAX_DEPTH", 5), "cells": ("MAX_CELLS", 1)}
    monkeypatch.setattr(rich, *limits[mode])
    blocks = [{"type": "paragraph", "text": "neutral"}] * 5 + [{"type": "paragraph", "text": "@offline_bot"}]
    if mode == "output":
        # The truncated display ends exactly at the bot name, but the original is a longer handle.
        prefix = "x" * (rich.MAX_OUTPUT_CHARS - len(rich.TRUNCATED) - 2 - len("@offline_bot")) + " "
        blocks = [{"type": "paragraph", "text": prefix + "@offline_bot_suffix" + "x" * 100}]
    elif mode == "depth":
        blocks = [{"type": "paragraph", "text": [[[[["@offline_bot"]]]]]}]
    elif mode == "cells":
        blocks = [{"type": "table", "cells": [[{"text": "neutral"}, {"text": "@offline_bot"}]]}]
    async with connected(monkeypatch, extra={"require_mention": True}) as (adapter, app, delivered):
        incoming = rich_update(app.bot, group=True, blocks=blocks)
        view = rich.RichMessageView(incoming.message)
        assert rich.TRUNCATED in view.display_text
        assert len(view.text) <= rich.MAX_OUTPUT_CHARS
        await app.process_update(incoming)
        await drain(adapter)
        assert delivered == []


@pytest.mark.asyncio
async def test_group_rich_cleanup_and_batching_cannot_upgrade_control(monkeypatch):
    async with connected(monkeypatch, extra={"require_mention": True}) as (adapter, app, delivered):
        for uid in (10, 11):
            await app.process_update(rich_update(app.bot, uid, group=True, blocks=[
                {"type": "paragraph", "text": "@offline_bot /stop"},
            ]))
        await app.process_update(update(app.bot, 12, group=True, text="@offline_bot ordinary"))
        assert len(adapter._pending_text_batches) == 2
        await drain(adapter)
        projected = next(event for event in delivered if not event.allow_gateway_control)
        assert projected.text == "/stop\n/stop" and not projected.is_command()
        assert projected.get_command() is None
        plain = next(event for event in delivered if event.allow_gateway_control)
        assert plain.text == "ordinary"


@pytest.mark.asyncio
@pytest.mark.parametrize("is_bot", [False, True])
async def test_rich_reply_keeps_human_trigger_and_bot_loop_breaker(monkeypatch, is_bot):
    async with connected(monkeypatch, extra={"require_mention": True, "bots_require_mention": True}) as (adapter, app, delivered):
        replied = update(app.bot, user=111).message.to_dict()
        replied["from"]["is_bot"] = True
        incoming = rich_update(app.bot, group=True, fields={
            "reply_to_message": replied, "from": {"id": 88, "is_bot": is_bot, "first_name": "Sender"},
        }, blocks=[{"type": "paragraph", "text": "chatter"}])
        await app.process_update(incoming)
        await drain(adapter)
        assert bool(delivered) is not is_bot


@pytest.mark.asyncio
async def test_truncated_rich_keeps_complete_early_trigger(monkeypatch):
    async with connected(monkeypatch, extra={"require_mention": True}) as (adapter, app, delivered):
        incoming = rich_update(app.bot, group=True, blocks=[
            {"type": "paragraph", "text": "@offline_bot hello " + "x" * rich.MAX_OUTPUT_CHARS},
        ])
        await app.process_update(incoming)
        await drain(adapter)
        assert len(delivered) == 1 and rich.TRUNCATED in delivered[0].text
        assert len(delivered[0].text) <= rich.MAX_OUTPUT_CHARS
