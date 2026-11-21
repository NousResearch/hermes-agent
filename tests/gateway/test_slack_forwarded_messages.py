"""Slack forwarded ("shared") messages reach the agent.

Slack delivers a forward as an ``attachments[]`` entry flagged ``is_share`` (verified on 13 real
exported forwards: every one also carried ``is_msg_unfurl``, plus ``message_blocks``, ``text``,
``fallback``, ``author_*``, ``channel_id``, ``from_url``, ``ts``; ``channel_name`` only when the
source is visible; ``files`` when the original had files). A pasted permalink unfurl carries
``is_msg_unfurl`` WITHOUT ``is_share``. The adapter skipped every ``is_msg_unfurl`` attachment, so
a forward vanished: the agent got only the user's typed request, with no hint anything was there.

Fixtures mirror that payload; names and ids are fictional.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.event import MessageType
from plugins.platforms.slack.adapter import SlackAdapter

PERMALINK = "https://acme.slack.com/archives/G0PRIV/p1793830020123456"
CI_BODY = "No - Bugbot and CI completion are not confirmed."
CI_BULLET = "The PR review history shows the approval and local test evidence"


def _rich(*elements):
    return {"type": "rich_text", "elements": list(elements)}


def _section(*elements):
    return {"type": "rich_text_section", "elements": list(elements)}


def _text(value):
    return {"type": "text", "text": value}


def _share(**over):
    """A forward of a message from a private conversation (the reported case): the bot is not a
    member, so ``channel_name`` is absent, but the content rides ``message_blocks``/``text``."""
    blocks = [_rich(
        _section(_text(CI_BODY + "\n")),
        {"type": "rich_text_list", "style": "bullet", "elements": [
            _section(_text("The approval says required CI and Bugbot results remain unverified.")),
            _section(_text(CI_BULLET + "; it does not show a Bugbot completion.")),
        ]},
    )]
    att = {
        "ts": "1793830020.123456",
        "author_id": "U0SCOUT",
        "channel_id": "G0PRIV",
        "channel_team": "T_TEAM",
        "is_msg_unfurl": True,
        "is_share": True,
        "message_blocks": [{"team": "T_TEAM", "channel": "G0PRIV", "ts": "1793830020.123456",
                            "message": {"blocks": blocks}}],
        "color": "D0D0D0",
        "from_url": PERMALINK,
        "fallback": f"[November 4th, 2026 6:07 PM] scout: {CI_BODY}",
        "text": f"{CI_BODY}\n- {CI_BULLET}",
        "author_name": "Scout",
        "author_subname": "Scout",
        "mrkdwn_in": ["text"],
        "footer": "Slack Conversation",
    }
    att.update(over)
    return att


def _event(text="<@U_BOT> look into this", attachments=None, **over):
    ev = {
        "type": "message",
        "text": text,
        "user": "U_USER",
        "channel": "C_CHAN",
        "channel_type": "channel",
        "ts": "1793900000.000100",
        "team": "T_TEAM",
        "attachments": [_share()] if attachments is None else attachments,
    }
    ev.update(over)
    return ev


@pytest.fixture()
def adapter():
    a = SlackAdapter(PlatformConfig(enabled=True, token="***"))
    a._app = MagicMock()
    a._app.client = AsyncMock()
    a._app.client.users_info = AsyncMock(return_value={
        "user": {"is_bot": False, "profile": {"display_name": "Test User"}, "real_name": "Test User"}})
    a._app.client.conversations_replies = AsyncMock(return_value={"ok": True, "messages": []})
    a._bot_user_id = "U_BOT"
    a._running = True
    a.handle_message = AsyncMock()
    return a


@pytest.fixture(autouse=True)
def _redirect_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.platforms.base.DOCUMENT_CACHE_DIR", tmp_path / "doc_cache")
    monkeypatch.setattr("gateway.platforms.base.VIDEO_CACHE_DIR", tmp_path / "video_cache")
    monkeypatch.setattr("gateway.platforms.base.IMAGE_CACHE_DIR", tmp_path / "img_cache", raising=False)


async def _delivered(adapter, event):
    await adapter._handle_slack_message(event)
    adapter.handle_message.assert_awaited_once()
    return adapter.handle_message.await_args.args[0]


# -- live inbound: every forward shape -------------------------------------------------------


@pytest.mark.asyncio
async def test_private_conversation_forward_reaches_agent_with_header(adapter):
    """The reported case: a request plus a forward from a private conversation."""
    msg = await _delivered(adapter, _event())
    assert "look into this" in msg.text
    assert CI_BODY in msg.text
    assert CI_BULLET in msg.text
    assert "Forwarded Slack message from Scout" in msg.text
    assert PERMALINK in msg.text
    # The user's request comes first; the forward is attached after it.
    assert msg.text.index("look into this") < msg.text.index(CI_BODY)


@pytest.mark.asyncio
async def test_channel_source_names_the_channel(adapter):
    msg = await _delivered(adapter, _event(attachments=[_share(channel_id="C0ENG", channel_name="eng-ci")]))
    assert "#eng-ci" in msg.text
    assert CI_BODY in msg.text


@pytest.mark.asyncio
async def test_dm_forward_with_no_added_text(adapter):
    """A bare forward into a DM: the forward is the whole message and must not be dropped."""
    msg = await _delivered(adapter, _event(text="", channel="D_DM", channel_type="im"))
    assert CI_BODY in msg.text
    assert "Forwarded Slack message" in msg.text


@pytest.mark.asyncio
async def test_bot_authored_source_with_section_blocks(adapter):
    """A forwarded app/bot post keeps its body in ``section`` mrkdwn blocks, not rich_text."""
    section_blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": "*Deploy failed* on `main`"}}]
    att = _share(bot_id="B0SCOUT", text="", fallback="",
                 message_blocks=[{"message": {"blocks": section_blocks}}])
    msg = await _delivered(adapter, _event(attachments=[att]))
    assert "*Deploy failed* on `main`" in msg.text


@pytest.mark.asyncio
async def test_text_used_when_message_blocks_absent(adapter):
    att = _share(message_blocks=None)
    msg = await _delivered(adapter, _event(attachments=[att]))
    assert CI_BULLET in msg.text


@pytest.mark.asyncio
async def test_fallback_used_when_text_and_blocks_absent(adapter):
    att = _share(message_blocks=None, text="")
    msg = await _delivered(adapter, _event(attachments=[att]))
    assert CI_BODY in msg.text


@pytest.mark.asyncio
async def test_unreadable_forward_gets_explicit_marker(adapter):
    """No text, blocks, fallback or files: say a forward is there but unreadable, never vanish."""
    att = {"is_share": True, "is_msg_unfurl": True, "author_name": "Scout", "author_id": "U0SCOUT",
           "channel_id": "G0PRIV", "from_url": PERMALINK, "ts": "1793830020.123456"}
    msg = await _delivered(adapter, _event(attachments=[att]))
    assert "Forwarded Slack message from Scout is present" in msg.text
    assert "not available" in msg.text
    assert PERMALINK in msg.text


@pytest.mark.asyncio
async def test_long_forward_is_bounded_with_elision_marker(adapter):
    """'Show more' content arrives whole; a huge one is capped with the shared elision marker."""
    long_body = "x" * 30_000
    att = _share(message_blocks=[{"message": {"blocks": [_rich(_section(_text(long_body)))]}}])
    msg = await _delivered(adapter, _event(attachments=[att]))
    assert "x" * 1000 in msg.text
    assert len(msg.text) < 10_000
    assert "omitted" in msg.text


@pytest.mark.asyncio
async def test_forward_into_a_thread(adapter):
    """A forward posted as a thread reply (Slack marks thread forwards ``is_reply_unfurl``)."""
    adapter._app.client.conversations_replies = AsyncMock(return_value={"ok": True, "messages": [
        {"ts": "1793899999.000001", "user": "U_OTHER", "text": "earlier root message"}]})
    att = _share(is_reply_unfurl=True, footer="Thread in Slack Conversation")
    msg = await _delivered(adapter, _event(thread_ts="1793899999.000001", attachments=[att]))
    assert CI_BODY in msg.text


@pytest.mark.asyncio
async def test_forwarded_file_is_downloaded_like_a_direct_attachment(adapter):
    pdf = b"%PDF-1.4 forwarded"
    att = _share(files=[{"id": "F0FWD", "name": "ci-report.pdf", "mimetype": "application/pdf",
                         "url_private_download": "https://files.slack.com/files-pri/T_TEAM-F0FWD/ci-report.pdf",
                         "size": len(pdf)}])
    with patch.object(adapter, "_download_slack_file_bytes", new_callable=AsyncMock) as dl:
        dl.return_value = pdf
        msg = await _delivered(adapter, _event(attachments=[att]))
    assert msg.message_type == MessageType.DOCUMENT
    assert len(msg.media_urls) == 1 and os.path.exists(msg.media_urls[0])
    assert "ci-report.pdf" in msg.text  # the forward lists the file it carried


@pytest.mark.asyncio
async def test_forwarded_file_already_on_the_event_is_not_downloaded_twice(adapter):
    pdf = b"%PDF-1.4 forwarded"
    f = {"id": "F0FWD", "name": "ci-report.pdf", "mimetype": "application/pdf",
         "url_private_download": "https://files.slack.com/files-pri/T_TEAM-F0FWD/ci-report.pdf",
         "size": len(pdf)}
    with patch.object(adapter, "_download_slack_file_bytes", new_callable=AsyncMock) as dl:
        dl.return_value = pdf
        msg = await _delivered(adapter, _event(attachments=[_share(files=[f])], files=[dict(f)]))
    assert dl.await_count == 1
    assert len(msg.media_urls) == 1


@pytest.mark.asyncio
async def test_forwarded_slack_connect_stub_file_resolved_via_files_info(adapter):
    pdf = b"%PDF-1.4 forwarded"
    full = {"id": "F0STUB", "name": "spec.pdf", "mimetype": "application/pdf",
            "url_private_download": "https://files.slack.com/files-pri/T_TEAM-F0STUB/spec.pdf", "size": len(pdf)}
    adapter._app.client.files_info = AsyncMock(return_value={"ok": True, "file": full})
    att = _share(files=[{"id": "F0STUB", "file_access": "check_file_info"}])
    with patch.object(adapter, "_download_slack_file_bytes", new_callable=AsyncMock) as dl:
        dl.return_value = pdf
        msg = await _delivered(adapter, _event(attachments=[att]))
    adapter._app.client.files_info.assert_awaited_once()
    assert len(msg.media_urls) == 1


@pytest.mark.asyncio
async def test_file_only_forward_is_not_reported_unavailable(adapter):
    att = {"is_share": True, "is_msg_unfurl": True, "author_name": "Scout", "from_url": PERMALINK,
           "files": [{"id": "F0IMG", "name": "graph.png", "mimetype": "image/png"}]}
    msg = await _delivered(adapter, _event(attachments=[att]))
    assert "graph.png" in msg.text
    assert "not available" not in msg.text


@pytest.mark.asyncio
async def test_two_forwards_keep_their_order(adapter):
    a1 = _share(author_name="Ana", text="first forwarded", message_blocks=None)
    a2 = _share(author_name="Ben", text="second forwarded", message_blocks=None)
    msg = await _delivered(adapter, _event(attachments=[a1, a2]))
    assert msg.text.index("first forwarded") < msg.text.index("second forwarded")


# -- guards that must stay ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_permalink_unfurl_without_is_share_is_still_skipped(adapter):
    """Echo prevention (6087e04043): a pasted permalink's unfurl repeats a message; not content."""
    unfurl = {"is_msg_unfurl": True, "author_id": "U_BOT", "text": "our own earlier reply",
              "fallback": "our own earlier reply", "from_url": PERMALINK}
    msg = await _delivered(adapter, _event(text=f"<@U_BOT> see <{PERMALINK}>", attachments=[unfurl]))
    assert "our own earlier reply" not in msg.text
    assert "Forwarded Slack message" not in msg.text


@pytest.mark.asyncio
async def test_pasted_permalink_to_someone_elses_message_reaches_agent(adapter):
    """Pasting a permalink (no Forward action) to another person's message: Slack's unfurl is the
    agent's only copy when the bot is not in the source conversation; our own message stays an echo."""
    other = {"is_msg_unfurl": True, "is_thread_root_unfurl": True, "author_id": "U0SCOUT",
             "author_name": "Scout", "text": CI_BODY, "from_url": PERMALINK}
    own = {"is_msg_unfurl": True, "author_id": "U_BOT", "text": "our own earlier reply"}
    msg = await _delivered(adapter, _event(text=f"<@U_BOT> is this valid? <{PERMALINK}>",
                                           attachments=[other, own]))
    assert "[Linked Slack message from Scout]" in msg.text
    assert CI_BODY in msg.text
    assert "our own earlier reply" not in msg.text
    rendered = adapter._render_message_text({"text": f"<{PERMALINK}>", "attachments": [other, own]},
                                            bot_uid="U_BOT")
    assert CI_BODY in rendered and "our own earlier reply" not in rendered


@pytest.mark.asyncio
async def test_forwarded_bot_mention_does_not_summon_the_bot(adapter):
    """A mention inside the forwarded content is someone else's words, not an address."""
    att = _share(text="<@U_BOT> please rerun", message_blocks=None)
    await adapter._handle_slack_message(_event(text="fyi", attachments=[att]))
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_forward_already_quoted_in_the_message_is_not_repeated(adapter):
    quoted = "the deploy failed twice on the staging box"
    att = _share(text=quoted, message_blocks=None)
    msg = await _delivered(adapter, _event(
        text="<@U_BOT> look", blocks=[_rich(_section(_text("look")),
                                            {"type": "rich_text_quote", "elements": [_text(quoted)]})],
        attachments=[att]))
    assert msg.text.count(quoted) == 1


# -- thread-context hydration and transcript ---------------------------------------------------


def test_hydration_renders_a_forward_in_history(adapter):
    rendered = adapter._render_message_text({"text": "<@U_BOT> <@U_A> <@U_B>", "attachments": [_share()]})
    assert CI_BODY in rendered
    assert "Forwarded Slack message from Scout" in rendered


def test_hydration_still_skips_a_bare_permalink_unfurl(adapter):
    rendered = adapter._render_message_text({"text": f"<{PERMALINK}>", "attachments": [
        {"is_msg_unfurl": True, "text": "the linked message body"}]})
    assert "the linked message body" not in rendered


def test_hydration_lists_forwarded_files(adapter):
    rendered = adapter._render_message_text({"text": "", "attachments": [
        _share(files=[{"id": "F1", "name": "chart.png", "mimetype": "image/png"}])]})
    assert "[image: chart.png]" in rendered


@pytest.mark.asyncio
async def test_thread_parent_forward_reaches_a_later_reply(adapter):
    """The reported thread: the root holds the mentions and the forward; a later reply in that
    thread must see the forward in its cold-start context, not just three @-mentions."""
    adapter._app.client.conversations_replies = AsyncMock(return_value={"ok": True, "messages": [
        {"ts": "1793899999.000001", "user": "U_ROOT", "text": "<@U_BOT> <@U_A> <@U_B>",
         "attachments": [_share()]},
        {"ts": "1793899999.000002", "user": "U_BOT", "bot_id": "B_SELF", "text": "What do you need from us?"},
    ]})
    with patch.object(adapter, "_resolve_user_name", new=AsyncMock(side_effect=lambda uid, **_: uid)):
        content = await adapter._fetch_thread_context(
            channel_id="C_CHAN", thread_ts="1793899999.000001", current_ts="1793899999.000009")
    assert CI_BODY in content
    assert "[thread parent]" in content


@pytest.mark.asyncio
async def test_thread_root_forwarded_image_is_recovered(adapter):
    adapter._app.client.conversations_replies = AsyncMock(return_value={"ok": True, "messages": [
        {"ts": "1793899999.000001", "user": "U_ROOT", "text": "<@U_BOT> look",
         "attachments": [_share(files=[{"id": "F0IMG", "name": "graph.png", "mimetype": "image/png",
                                        "url_private_download": "https://files.slack.com/files-pri/T_TEAM-F0IMG/graph.png"}])]},
    ]})
    adapter._download_slack_file = AsyncMock(return_value="/tmp/hermes-cached.png")
    await adapter._fetch_thread_context(
        channel_id="C_CHAN", thread_ts="1793899999.000001", current_ts="1793899999.000009")
    urls, _types = await adapter._collect_thread_root_images(
        channel_id="C_CHAN", thread_ts="1793899999.000001", team_id="")
    assert urls == ["/tmp/hermes-cached.png"]


# -- review follow-ups: shared budget, block order, quoted source, echo across workspaces --------


def test_history_shares_one_budget_across_forwards(adapter):
    """Thread history gets the same single body budget as the live path, not one per forward."""
    long = [_share(author_name=n, ts=f"179383002{i}.0", from_url=f"{PERMALINK}{i}", text="y" * 6000,
                   message_blocks=None) for i, n in enumerate(("Ana", "Ben", "Cy"))]
    rendered = adapter._render_message_text({"text": "", "attachments": long}, bot_uid="U_BOT")
    assert rendered.count("Forwarded Slack message from") == 3
    assert len(rendered) < 10_000


@pytest.mark.asyncio
async def test_forwarded_app_blocks_keep_their_order(adapter):
    blocks = [{"type": "header", "text": {"type": "plain_text", "text": "Quarterly numbers"}},
              _rich(_section(_text("rich body line"))),
              {"type": "section", "text": {"type": "mrkdwn", "text": "closing explanation"}}]
    att = _share(text="", fallback="", message_blocks=[{"message": {"blocks": blocks}}])
    msg = await _delivered(adapter, _event(attachments=[att]))
    assert msg.text.index("Quarterly numbers") < msg.text.index("rich body line") < msg.text.index(
        "closing explanation")


@pytest.mark.asyncio
async def test_quoted_forward_keeps_its_source_and_files(adapter):
    quoted = "the deploy failed twice on the staging box"
    att = _share(text=quoted, message_blocks=None, files=[{"id": "F9", "name": "log.txt", "mimetype": "text/plain"}])
    with patch.object(adapter, "_download_slack_file_bytes", new_callable=AsyncMock, return_value=b"x"):
        msg = await _delivered(adapter, _event(text=f"<@U_BOT> {quoted}", attachments=[att]))
    assert msg.text.count(quoted) == 1
    assert "Forwarded Slack message from Scout" in msg.text and PERMALINK in msg.text
    assert "log.txt" in msg.text


@pytest.mark.asyncio
async def test_own_message_from_another_workspace_is_still_an_echo(adapter):
    """Multi-workspace: each team has its own bot user; a permalink to our message in team B
    pasted into team A must not come back as someone else's linked message."""
    adapter._team_bot_user_ids = {"T_TEAM": "U_BOT", "T_OTHER": "U_BOT_B"}
    own_b = {"is_msg_unfurl": True, "author_id": "U_BOT_B", "text": "our reply in the other team",
             "from_url": PERMALINK}
    msg = await _delivered(adapter, _event(text=f"<@U_BOT> see <{PERMALINK}>", attachments=[own_b]))
    assert "our reply in the other team" not in msg.text


@pytest.mark.asyncio
async def test_same_message_as_forward_and_unfurl_renders_once(adapter):
    unfurl = {"is_msg_unfurl": True, "author_id": "U0SCOUT", "author_name": "Scout", "text": CI_BODY,
              "from_url": PERMALINK + "?thread_ts=1793830020.123456&cid=G0PRIV",
              "channel_id": "G0PRIV", "ts": "1793830020.123456"}
    msg = await _delivered(adapter, _event(text=f"<@U_BOT> <{PERMALINK}>", attachments=[unfurl, _share()]))
    assert msg.text.count(CI_BODY) == 1
    assert "Forwarded Slack message from Scout" in msg.text
    assert "Linked Slack message" not in msg.text
