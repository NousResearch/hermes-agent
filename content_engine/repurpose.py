"""Repurpose inbox: ingest reference media/links Sahil drops in a Discord channel.

IP guardrail: references are INSPIRATION only. The pipeline never republishes a
source file; it regenerates an original asset in Sahil's brand voice (see the
content-repurpose agent cron, which writes the copy and calls gen-image). This
module only ingests: it polls the inbox channel, downloads new attachments, and
returns the items (with Sahil's caption as the creative intent) for the agent.
"""
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

DISCORD_API = "https://discord.com/api/v10"
INBOX_CHANNEL = os.getenv("REPURPOSE_INBOX_CHANNEL", "1510403518169878689")
# State lives in the content engine's own data dir, not KENSEI's ~/.hermes state.
STATE_PATH = Path(__file__).resolve().parent / "data" / "repurpose-inbox-state.json"
INBOX_DIR = Path(__file__).resolve().parent / "inbox"

_IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".gif")
_VID_EXT = (".mp4", ".mov", ".webm", ".m4v")
_APPROVE_EMOJI = {"✅", "white_check_mark"}


def _safe_inbox_dest(msg_id: str, raw_name: str) -> Optional[Path]:
    """Return a write path pinned inside INBOX_DIR, or None if the name is unsafe.

    Discord attachment filenames are attacker-controllable, so strip them to a
    basename of safe characters and verify the resolved path stays under the
    inbox root before any download.
    """
    safe_msg = re.sub(r"[^0-9]", "", str(msg_id))
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(raw_name or "file"))
    if not safe_msg or not safe_name or safe_name.startswith("."):
        return None
    root = INBOX_DIR.resolve()
    dest = (root / safe_msg / safe_name).resolve()
    if not dest.is_relative_to(root):
        return None
    return dest


def _has_approve_reaction(message: Dict) -> bool:
    """True if the message carries a ✅ reaction (Sahil approving a suggestion)."""
    for r in message.get("reactions", []) or []:
        name = (r.get("emoji", {}) or {}).get("name", "")
        if name in _APPROVE_EMOJI and r.get("count", 0) > 0:
            return True
    return False


def _token() -> str:
    return os.getenv("DISCORD_BOT_TOKEN", "").strip()


def _load_state() -> Dict:
    try:
        return json.loads(STATE_PATH.read_text())
    except Exception:  # noqa: BLE001
        return {}


def _save_state(state: Dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state))


def _download(url: str, dest: Path) -> bool:
    try:
        r = requests.get(url, timeout=60)
        if r.status_code == 200:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(r.content)
            return True
    except Exception as exc:  # noqa: BLE001
        print(f"[repurpose] download failed {url}: {exc}")
    return False


def fetch_inbox(channel_id: str = INBOX_CHANNEL) -> List[Dict]:
    """Poll the inbox for messages newer than last seen. Returns new items.

    Each item: {msg_id, caption, images: [local paths], videos: [local paths],
    links: [urls]}. Bot messages (suggestions, our own posts) are skipped.
    """
    if not _token():
        print("[repurpose] DISCORD_BOT_TOKEN not set.")
        return []

    state = _load_state()
    last_seen = state.get(channel_id)
    params = {"limit": 30}
    if last_seen:
        params["after"] = last_seen

    try:
        r = requests.get(
            f"{DISCORD_API}/channels/{channel_id}/messages",
            headers={"Authorization": f"Bot {_token()}"}, params=params, timeout=30,
        )
        if r.status_code != 200:
            print(f"[repurpose] poll failed {r.status_code}: {r.text[:160]}")
            return []
        messages = r.json()
    except Exception as exc:  # noqa: BLE001
        print(f"[repurpose] poll error: {exc}")
        return []

    # Discord returns newest first; process oldest first so state advances cleanly.
    messages = sorted(messages, key=lambda m: int(m["id"]))
    items: List[Dict] = []
    newest = last_seen

    for m in messages:
        newest = m["id"]
        # Skip our own bot posts unless Sahil approved a suggestion with a ✅.
        if m.get("author", {}).get("bot") and not _has_approve_reaction(m):
            continue
        caption = (m.get("content") or "").strip()
        images, videos = [], []
        for a in m.get("attachments", []):
            url = a.get("url", "")
            name = (a.get("filename") or "").lower()
            dest = _safe_inbox_dest(m["id"], a.get("filename") or "file")
            if dest is None:
                continue
            if name.endswith(_IMG_EXT) and _download(url, dest):
                images.append(str(dest))
            elif name.endswith(_VID_EXT) and _download(url, dest):
                videos.append(str(dest))
        links = [w for w in caption.split() if w.startswith("http")]
        if caption or images or videos or links:
            items.append({
                "msg_id": m["id"], "caption": caption,
                "images": images, "videos": videos, "links": links,
            })

    if newest:
        state[channel_id] = newest
        _save_state(state)
    return items
