"""Gateway-B subprocess driver for the cross-instance claim check (t_24b1959a).

Runs a COMPLETE second gateway stack (runner + mattermost adapter + session
authority) in its own process, own HERMES_HOME, own state.db, over a SHARED
relay routing root. Drives one Mattermost websocket event and reports what it
admitted, posted and logged as a single JSON line on stdout.

Usage: python -m tests.gateway._pipe_b_driver <job.json>
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

CHANNEL = "8grzznya3trpbfxjwf5goe778h"
SLOW_TOKEN = " SLOW"


def answer_for(text: str) -> str:
    import hashlib

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "ANSWER::" + hashlib.sha256((lines[-1] if lines else text).encode()).hexdigest()[:10]


def patch_model_transport() -> None:
    from openai.resources.chat.completions import AsyncCompletions, Completions

    def _content_to_text(c) -> str:
        if c is None:
            return ""
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            parts = []
            for p in c:
                if isinstance(p, dict):
                    parts.append(str(p.get("text") or p.get("content") or ""))
                elif isinstance(p, str):
                    parts.append(p)
                else:
                    parts.append(str(getattr(p, "text", None) or getattr(p, "content", None) or p))
            return "\n".join(t for t in parts if t)
        if isinstance(c, dict):
            return str(c.get("text") or c.get("content") or "")
        return str(c)

    def _last_user_content(kwargs):
        messages = kwargs.get("messages")
        if not messages and isinstance(kwargs.get("extra_body"), dict):
            messages = kwargs["extra_body"].get("messages") or messages
        last = ""
        for m in reversed(list(messages or [])):
            role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
            if role != "user":
                continue
            content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
            last = _content_to_text(content)
            break
        return last


    async def async_create(self, **kwargs):
        if SLOW_TOKEN.strip() in (_last_user_content(kwargs) or ""):
            await asyncio.sleep(3.2)  # health: allow HX013 -- the slow model IS the behavior under test: it keeps gateway B's turn in flight while A stands down
        return _completion(answer_for(_last_user_content(kwargs) or ""),
                           kwargs.get("model", ""))

    def sync_create(self, **kwargs):
        return _completion(answer_for(_last_user_content(kwargs) or ""),
                          kwargs.get("model", ""))

    AsyncCompletions.create = async_create          # type: ignore[method-assign]
    Completions.create = sync_create                # type: ignore[method-assign]


def _completion(answer: str, model: str):
    from openai.types.chat import ChatCompletion

    return ChatCompletion.model_validate({
        "id": "chatcmpl-stub", "object": "chat.completion", "created": 1,
        "model": model or "stub-glm-5.2",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": answer},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
    })


class Journal(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.lines.append(f"[{record.levelname}] {record.name}: {record.getMessage()}")
        except Exception:
            pass


async def run(job: dict) -> dict:
    home = Path(job["home"])
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "model:\n  default: stub-glm-5.2\n  provider: custom\n"
        "memory:\n  memory_enabled: false\n", encoding="utf-8")
    os.environ["HERMES_HOME"] = str(home)
    os.environ["GATEWAY_RELAY_INSTANCE_ID"] = job["instance_id"]
    os.environ["HERMES_AGENT_NOTIFY_INTERVAL"] = "1"
    os.environ["CUSTOM_API_KEY"] = "stub-key"
    os.environ["CUSTOM_BASE_URL"] = "http://127.0.0.1:9/v1"

    journal = Journal()
    logging.getLogger().addHandler(journal)
    logging.getLogger().setLevel(logging.DEBUG)
    from gateway.config import ClaimLockConfig, GatewayConfig, Platform, PlatformConfig
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.platforms.base import SendResult
    from plugins.platforms.mattermost.adapter import MattermostAdapter

    sends: list[dict] = []
    edits: list[dict] = []
    counter = {"n": 0}

    runner = GatewayRunner(GatewayConfig(group_sessions_per_user=False,
        claim_lock=ClaimLockConfig(mode="enforce")))
    adapter = MattermostAdapter(PlatformConfig(
        enabled=True, token="pipe-b-token", typing_indicator=False,
        extra={"reply_mode": "thread"}))

    async def send(chat_id, content, reply_to=None, metadata=None):
        counter["n"] += 1
        mid = f"B-{counter['n']:04d}"
        sends.append({"content": str(content), "reply_to": reply_to, "message_id": mid})
        return SendResult(success=True, message_id=mid)

    async def edit_message(chat_id, message_id, content, *, finalize: bool = False):
        edits.append({"content": str(content), "message_id": message_id})
        return SendResult(success=True, message_id=message_id)

    adapter.send = send                     # type: ignore[method-assign]
    adapter.edit_message = edit_message      # type: ignore[method-assign]
    adapter.gateway_runner = runner
    runner.adapters = {Platform.MATTERMOST: adapter}

    await initialize_session_authority(
        runner, profile_id="default", instance_id=job["instance_id"])
    runner.session_store._routing_home = Path(job["relay_root"])
    runner._wire_adapter_handlers(adapter, busy_text_mode="queue")

    # bootstrap (plugin loader) reconfigures root logging; re-attach capture
    # so the stand-down lines from the claim gate are actually recorded.
    logging.getLogger().addHandler(journal)
    logging.getLogger().setLevel(logging.DEBUG)
    for _name in ("gateway", "gateway.session", "gateway.session_ingress",
                  "plugins.platforms.mattermost", "hermes_state_runtime"):
        _lg = logging.getLogger(_name)
        _lg.setLevel(logging.DEBUG)
        _lg.addHandler(journal)
    await adapter._handle_ws_event(job["event"])
    # The inbound spawns a background task which may lazily boot the whole
    # plugin/model stack; wait for it to reach a terminal state (stand-down
    # log line, or full quiet) instead of a fixed 0.25s.
    async def _settled() -> bool:
        marker_words = ("claim", "admiss", "session", "respond", "queue", "parked")
        recent = [ln for ln in journal.lines[-160:]]
        return any(any(w in ln.lower() for w in marker_words) for ln in recent
                   if "Scanning" not in ln and "Parsed manifest" not in ln)

    _t0 = time.monotonic()
    _quiet = 0.0
    while time.monotonic() - _t0 < 40:
        await asyncio.sleep(0.5)
        if _settled():
            await asyncio.sleep(1.0)  # health: allow HX013 -- absence window: elapsed time after settle proves gateway B posted nothing (zero sends IS the assertion)
            break
    await asyncio.sleep(1.5)  # health: allow HX013 -- absence window: admitted-turn drain before state.db inspection; elapsed time with no posts IS the assertion

    state_db = Path(home) / "state.db"
    sessions = []
    if state_db.exists():
        conn = sqlite3.connect(str(state_db))
        conn.row_factory = sqlite3.Row
        try:
            for r in conn.execute(
                    "SELECT session_key FROM sessions WHERE session_key LIKE ?",
                    (f"%{CHANNEL}%",)):
                sessions.append(r["session_key"])
        except sqlite3.OperationalError:
            pass
        finally:
            conn.close()

    relay_posts = list(Path(job["relay_root"]).glob("**/channel_claims/*"))
    return {
        "sends": sends,
        "edits": edits,
        "session_keys": sessions,
        "relay_claim_files": [str(p) for p in relay_posts],
        "log": journal.lines,
    }


def main() -> None:
    job = json.loads(Path(sys.argv[1]).read_text())
    patch_model_transport()
    out = asyncio.run(run(job))
    sys.stdout.write(json.dumps(out))
    sys.stdout.write("\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
