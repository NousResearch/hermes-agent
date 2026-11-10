"""A native ``vision_analyze`` image is re-sent on turns rebuilt from the session DB (#142288).

The live turn sends the pixels inside the tool result, and the native embed is meant to ride every
later request (``tools/AGENTS.md``). The session DB stores the tool row as text with a
``[screenshot]`` line, and the gateway rebuilds each turn's history from it, so the next request
showed the model a placeholder and no longer matched what the previous request sent: the prompt
cache misses from that row on, and a model with preserved thinking rejects the replayed thinking
("The block is bound to a different conversation").
"""

import asyncio
import base64
import json
import os
import time
from pathlib import Path

from agent.provider_media import MEDIA_CACHE_MAX_AGE_HOURS
from agent.session_persistence import _durable_content
from agent.turn_context import build_api_messages
from gateway.session_ingress_media import sweep_settled_media
from hermes_constants import get_hermes_home
from hermes_state import SessionDB
from tests.agent.test_image_replay import _PNG, _Agent, _config
from tests.tools.test_vision_native_fast_path import _TINY_PNG
from tools.vision_tools import _vision_analyze_native


def test_rebuilt_turn_resends_the_vision_analyze_image_byte_identical(tmp_path):
    _config()
    image = tmp_path / "label.png"
    image.write_bytes(_TINY_PNG)
    envelope = asyncio.run(_vision_analyze_native(str(image), "What is on this label?"))
    live = envelope["content"]  # what a vision model's tool message carries (_tool_result_content_for_active_model)

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="s", source="telegram")
    call = {"id": "call_1", "type": "function",
            "function": {"name": "vision_analyze", "arguments": json.dumps({"image_url": str(image)})}}
    db.append_message("s", role="user", content="What is on this label?", timestamp=1000.0)
    db.append_message("s", role="assistant", content="", tool_calls=[call], timestamp=1001.0)
    db.append_message("s", role="tool", content=_durable_content(live), tool_call_id="call_1",
                      tool_name="vision_analyze", timestamp=1002.0)
    db.append_message("s", role="assistant", content="It says 250 ml.", timestamp=1003.0)
    history = db.get_messages_as_conversation("s") + [{"role": "user", "content": "and the expiry date?"}]

    request, _ = build_api_messages(
        _Agent(), history, current_turn_user_idx=len(history) - 1,
        ext_prefetch_cache="", plugin_user_context="", moa_config=None, active_system_prompt="")

    # Same bytes as the request that first sent it: the prompt-cache prefix holds.
    assert next(m for m in request if m["role"] == "tool")["content"] == live


def _kept(data):
    from agent.image_replay import retain_tool_image

    _, hint = retain_tool_image("data:image/png;base64," + base64.b64encode(data).decode())
    return Path(hint.removeprefix("[Image attached at: ").removesuffix("]"))


def test_kept_tool_images_leave_at_the_media_cache_age():
    old, fresh = _kept(_PNG), _kept(_PNG + b"\x00")
    aged = time.time() - MEDIA_CACHE_MAX_AGE_HOURS * 3600 - 60
    os.utime(old, (aged, aged))
    db = SessionDB(db_path=get_hermes_home() / "state.db")  # the sweep only empties its own profile's store

    assert sweep_settled_media(db) == 1
    assert not old.exists() and fresh.exists()
