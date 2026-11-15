"""A Telegram message whose content arrives only as ``rich_message`` blocks (empty ``text``) reaches the
agent as plain text, through the REAL gateway and adapter against the Bot API stand-in. PTB keeps the
field in ``Message.api_kwargs``, so only a real SDK parse shows whether the adapter reads it."""

from __future__ import annotations

import pytest

from tests.e2e.core.platforms import _contract as C
from tests.e2e.core.platforms._drv_telegram import TelegramDriver
from tests.e2e.core.platforms._suite import rig_fixtures

pytestmark = [
    pytest.mark.spawns_gateway_lookalike,
    pytest.mark.platforms("posix"),  # POSIX process-group gateway harness
]

rig, rig_stream = rig_fixtures(TelegramDriver)


def test_rich_message_blocks_reach_the_agent_as_text(rig) -> None:
    token, aid = "richin", "A-richin"
    rig.director.script(token, C.answer(aid, "got it"))
    uid = int(rig.drv.user_id)
    chat = {"id": uid, "type": "private", "first_name": "Tester", "username": f"user{uid}"}
    rig.drv.standin._inbound(chat, uid, rich_message={"blocks": [
        {"type": "paragraph", "text": ["Can't you update it? ", {"type": "bold", "text": f"[in:{token}]"}]},
        {"type": "paragraph", "text": ""},
        {"type": "list", "items": [
            {"label": "•", "blocks": [{"type": "paragraph", "text": "Kapazität: 1TB"}]},
            {"label": "•", "blocks": [{"type": "paragraph", "text": "Modell: HD103UJ"}]}]}]})
    C.wait_reply(rig, str(uid), aid, "the rich_message DM to be answered")
    assert C.turns(rig, token) == 1
    assert rig.gw.user_rows(f"[in:{token}]") == [
        f"Can't you update it? [in:{token}]\n\n• Kapazität: 1TB\n• Modell: HD103UJ"]


def test_rich_message_photo_block_reaches_the_model_as_an_image(rig) -> None:
    """A photo with a caption can arrive as ``rich_message`` blocks only (a ``photo`` block, no top-level
    ``photo``): the image reaches the model, not just the caption (#155902)."""
    token, aid = "richphoto", "A-richphoto"
    rig.director.script(token, C.answer(aid, "saw it"))
    uid = int(rig.drv.user_id)
    fid = "rich-photo-1"
    rig.drv.standin.files[fid] = C._image("PNG")
    rig.drv.standin.file_paths[fid] = "photos/file_1.png"
    chat = {"id": uid, "type": "private", "first_name": "Tester", "username": f"user{uid}"}
    rig.drv.standin._inbound(chat, uid, rich_message={"blocks": [
        {"type": "photo", "photo": [{"file_id": fid, "file_unique_id": f"u-{fid}", "width": 32, "height": 32}],
         "caption": {"text": "the receipt"}},
        {"type": "paragraph", "text": f"what does this say? [in:{token}]"}]})
    C.wait_reply(rig, str(uid), aid, "the rich photo DM to be answered")
    assert C._model_saw_image(rig.director.requests[token][0]), rig.gw.grep(r"photo|image|cache|rich")
    assert rig.gw.user_rows(f"[in:{token}]")[0].startswith(f"the receipt\nwhat does this say? [in:{token}]")


def test_rich_message_group_mention_reaches_the_agent(rig) -> None:
    """In a require_mention group, a rich-only message that @mentions the bot is answered, not dropped."""
    token, aid = "richgrp", "A-richgrp"
    rig.director.script(token, C.answer(aid, "on it"))
    uid, gid = int(rig.drv.user_id), int(rig.drv.group_chat_id)
    handle = "@hermes_standin_bot"
    chat = {"id": gid, "type": "supergroup", "title": "Standin Group"}
    rig.drv.standin._inbound(chat, uid, rich_message={"blocks": [
        {"type": "paragraph", "text": [handle, f" please look [in:{token}]"]},
        {"type": "paragraph", "text": "second paragraph"}]})
    C.wait_reply(rig, str(gid), aid, "the rich_message group mention to be answered")
    assert C.turns(rig, token) == 1
