"""QQ C2C ``stream_messages`` state.

QQ private chats update one bubble through ``POST /v2/users/{openid}/stream_messages``.
The first frame omits ``stream_msg_id``; later frames reuse the returned ``id``,
keep one ``msg_seq``, and increment ``index``. ``input_mode=replace`` sends the
full text every time, and that text must extend the previous frame. ``input_state``
10 closes the generating indicator.

This module owns that protocol state. The adapter owns the HTTP call.
"""

from __future__ import annotations

import time
from typing import Any, Optional

INPUT_STATE_GENERATING = 1
INPUT_STATE_DONE = 10
MAX_STREAM_CONTENT_LEN = 4096
STREAM_SESSION_TTL_SECONDS = 600.0


class C2CStream:
    """One in-flight C2C stream, keyed by the adapter on ``(chat_id, turn_id)``."""

    def __init__(self, openid: str, passive_msg_id: str, msg_seq: int) -> None:
        self.openid = openid
        self.passive_msg_id = passive_msg_id
        self.msg_seq = msg_seq
        self.stream_msg_id: Optional[str] = None
        self.next_index = 0
        self.last_sent_content = ""
        self.finalized = False
        self.created_at = time.monotonic()

    @property
    def opened(self) -> bool:
        return self.stream_msg_id is not None

    def expired(self, now: Optional[float] = None) -> bool:
        return (now if now is not None else time.monotonic()) - self.created_at > STREAM_SESSION_TTL_SECONDS

    def outgoing(self, content: str) -> tuple[str, bool]:
        """Full text safe for ``input_mode=replace``.

        A frame that does not extend the last accepted text is replaced by that
        text so QQ does not reject the stream. The bool is True when that happens.
        """
        text = (content or "")[:MAX_STREAM_CONTENT_LEN]
        last = self.last_sent_content
        if last and not text.startswith(last):
            return last, True
        return text, False

    def request_body(self, content_raw: str, *, done: bool) -> dict[str, Any]:
        body: dict[str, Any] = {
            "input_mode": "replace",
            "input_state": INPUT_STATE_DONE if done else INPUT_STATE_GENERATING,
            "content_type": "markdown",
            "content_raw": content_raw[:MAX_STREAM_CONTENT_LEN],
            "event_id": self.passive_msg_id,
            "msg_id": self.passive_msg_id,
            "msg_seq": self.msg_seq,
            "index": self.next_index,
        }
        if self.stream_msg_id:
            body["stream_msg_id"] = self.stream_msg_id
        return body

    def accept(self, response: Any, sent: str, *, done: bool) -> None:
        """Record a successful frame. The first response must carry ``id``."""
        if not self.stream_msg_id:
            stream_id = response.get("id") if isinstance(response, dict) else None
            if not stream_id:
                raise RuntimeError(f"QQ stream response missing id: {response!r}")
            self.stream_msg_id = str(stream_id)
        self.last_sent_content = sent[:MAX_STREAM_CONTENT_LEN]
        self.next_index += 1
        self.finalized = done
