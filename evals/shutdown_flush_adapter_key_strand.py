"""Shutdown flush spooled under the adapter's session key strands on the next boot (#136189).

``_pending_messages`` slots are keyed by the adapter's own isolation flags
(``PlatformConfig.extra``), while ``SessionStore`` routes by the gateway-wide
``group_sessions_per_user`` / ``thread_sessions_per_user``. ``recover_pending_to_db`` resolves a
payload's key through the store alone (routing map, then the durable row under the exact key), so
a spool file written under the adapter key matches neither and is preserved forever.

This probe drives the real entry points only — ``adapter.cancel_background_tasks()`` then
``recover_pending_to_db`` — so it runs unchanged against a fixed and an unfixed checkout:

    python evals/shutdown_flush_adapter_key_strand.py [ROOT]

Prints the two derivations, the key actually spooled, the replay count and a ``VERDICT=`` line
(``STRANDED`` before the fix, ``RECOVERED`` after).
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HOME = Path(tempfile.mkdtemp(prefix="flush-strand-"))
os.environ["HERMES_HOME"] = str(HOME)
os.environ["HERMES_NIX_BUILD"] = "1"

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource, SessionStore
from gateway.shutdown_flush import recover_pending_to_db


class _Adapter(BasePlatformAdapter):
    """The minimum a real adapter needs to reach the shutdown flush."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)

    async def connect(self, *, is_reconnect: bool = False):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="1")

    async def send_typing(self, chat_id, metadata=None):
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


class _Row:
    """The durable row the resolver's exact-key peer finder would return."""

    def __init__(self, session_id: str, session_key: str):
        self.session_id = session_id
        self.session_key = session_key
        self.appended: list[dict] = []

    def find_latest_gateway_session_for_peer(self, *, source, session_key=None, **kwargs):
        # The real finder runs only the ``s.session_key = ?`` branch when no chat tuple is given:
        # a row matches one exact key, never "any key".
        if session_key != self.session_key:
            return None
        return {"id": self.session_id, "started_at": 0.0}

    def append_message(self, **kwargs):
        self.appended.append(kwargs)


def _spooled_keys() -> list[str]:
    directory = HOME / "pending_messages"
    return sorted(json.loads(path.read_text())["session_key"]
                  for path in directory.glob("*.json"))


def main() -> int:
    store = SessionStore(sessions_dir=HOME / "sessions", config=GatewayConfig())

    adapter = _Adapter()
    adapter.set_session_store(store)
    # One platform overrides per-user thread isolation; the store keeps the gateway-wide default.
    adapter.config.extra["thread_sessions_per_user"] = True

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="900", chat_type="group",
                           thread_id="900", user_id="u1")
    event = MessageEvent(text="queued while busy", message_type=MessageType.TEXT,
                         source=source, message_id="queued-1")
    slot_key = adapter._event_session_key(event)
    store_key = store._generate_session_key(source)
    db = _Row("sid-900", store_key)
    store._db = db  # every key resolves against this handle (no profile I/O)
    adapter._pending_messages[slot_key] = event

    asyncio.run(adapter.cancel_background_tasks())
    spooled = _spooled_keys()
    recovered = recover_pending_to_db(db, session_resolver=store.resolve_session_id_for_key)
    remaining = len(_spooled_keys())

    print(f"slot_key={slot_key}")
    print(f"store_key={store_key}")
    print(f"spooled_keys={spooled}")
    print(f"recovered={recovered}")
    print(f"replayed_content={[call.get('content') for call in db.appended]}")
    print(f"spool_remaining={remaining}")
    verdict = "RECOVERED" if recovered == 1 and not remaining else "STRANDED"
    print(f"VERDICT={verdict}")
    return 0 if verdict == "RECOVERED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
