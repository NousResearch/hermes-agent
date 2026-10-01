"""Semantic icon selection for auto-titled Telegram forum topics.

The Bot API restricts topic icons to a fixed custom-emoji set exposed by
``getForumTopicIconStickers``; an arbitrary custom emoji identifier (or an
object emoji like a laptop) is rejected with ``STICKERSET_INVALID``-class
errors. So a semantic icon is picked in three steps:

1. fetch the allowed sticker set once and index it by its ``emoji`` face
   (the index survives across topics in the process);
2. match the session title against a keyword→emoji table (first title word
   that has a mapping wins — the title, not a preference order, decides
   which concept the user's conversation is about);
3. resolve the emoji face to an allowed ``custom_emoji_id``; if the face is
   missing from the fetched set the icon is skipped and the rename proceeds
   name-only (a mismatched or missing icon must never block the rename).

No LLM call: the title already carries the semantics (it is the model title
from ``auxiliary.title_generation``); this module only projects it onto the
fixed icon vocabulary Telegram allows.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple

# keyword → emoji face. Only faces the forum-icon set is known to carry are
# listed; anything else would silently never match. The lookup is by whole
# word, so "home" does not fire inside "chrome".
_TOPIC_ICON_KEYWORDS: Dict[str, str] = {
    # home & household
    "home": "🏠", "house": "🏠", "flat": "🏠", "apartment": "🏠", "homelab": "🏠",
    "kitchen": "🍳", "cooking": "🍳", "recipe": "🍳", "bake": "🍳", "baking": "🍳",
    "garden": "🌻", "plants": "🌻", "plant": "🌻",
    # work & money
    "work": "💼", "job": "💼", "office": "💼", "meeting": "💼", "career": "💼", "cv": "💼", "resume": "💼",
    "money": "💰", "finance": "💰", "financial": "💰", "budget": "💰", "invest": "💰", "investing": "💰",
    "investment": "💰", "stocks": "💰", "crypto": "💰", "salary": "💰", "invoice": "💰", "tax": "💰",
    "taxes": "💰", "bank": "🏦", "banking": "🏦",
    # engineering (Hermes' home turf — rich coverage)
    "code": "💻", "coding": "💻", "bug": "🐛", "debug": "🐛", "debugging": "🐛", "fix": "🔧",
    "fixing": "🔧", "refactor": "🔧", "build": "🔧", "builds": "🔧", "ci": "🔧", "deploy": "🚀",
    "deployment": "🚀", "release": "🚀", "ship": "🚀", "devops": "🚀", "server": "🖥",
    "servers": "🖥", "database": "🗃", "postgres": "🗃", "mysql": "🗃", "sql": "🗃", "sqlite": "🗃",
    "migration": "🗃", "migrations": "🗃", "schema": "🗃", "docker": "🐳", "container": "🐳",
    "containers": "🐳", "kubernetes": "🐳", "k8s": "🐳", "api": "⚙️", "test": "✅",
    "tests": "✅", "testing": "✅", "pytest": "✅", "unittest": "✅", "tdd": "✅",
    "flaky": "✅", "pipeline": "⚙️", "config": "⚙️", "performance": "⚡️", "perf": "⚡️",
    "optimize": "⚡️", "optimization": "⚡️", "latency": "⚡️", "speed": "⚡️", "timeout": "⚠️",
    "regression": "⚠️", "incident": "⚠️", "outage": "⚠️", "error": "⚠️", "errors": "⚠️",
    "failure": "⚠️", "crash": "⚠️", "warn": "⚠️", "failing": "⚠️", "broken": "⚠️",
    # media
    "photo": "📷", "photos": "📷", "picture": "📷", "pictures": "📷", "image": "🖼",
    "images": "🖼", "album": "🖼", "gallery": "🖼", "camera": "📷", "video": "🎬",
    "movie": "🎬", "movies": "🎬", "film": "🎬", "youtube": "🎬", "song": "🎵",
    "music": "🎵", "playlist": "🎵", "audio": "🎧", "podcast": "🎧", "voice": "🎧",
    # travel
    "travel": "✈️", "trip": "✈️", "flight": "✈️", "flights": "✈️", "vacation": "🌴",
    "holiday": "🌴", "beach": "🌴", "hotel": "🏨", "booking": "🏨", "visa": "🛂",
    "passport": "🛂", "map": "🗺", "route": "🗺", "directions": "🗺", "train": "🚄",
    "car": "🚗", "drive": "🚗", "driving": "🚗",
    # health & sport
    "health": "💊", "medicine": "💊", "doctor": "💊", "symptom": "💊", "pill": "💊",
    "workout": "🏋️", "gym": "🏋️", "run": "🏃", "running": "🏃", "training": "🏃",
    "sport": "⚽️", "football": "⚽️", "soccer": "⚽️", "steps": "🏃", "sleep": "😴",
    "diet": "🥗", "calories": "🥗", "nutrition": "🥗", "weight": "⚖️", "keto": "🥗",
    # knowledge & planning
    "read": "📖", "reading": "📖", "book": "📖", "books": "📖", "article": "📰",
    "news": "📰", "research": "🔬", "study": "📚", "learn": "📚", "learning": "📚",
    "course": "📚", "language": "📚", "english": "📚", "spanish": "📚", "german": "📚",
    "note": "📝", "notes": "📝", "todo": "📝", "plan": "📝", "planning": "📝",
    "schedule": "📅", "calendar": "📅", "reminder": "📅", "deadline": "📅",
    "meeting-notes": "📝",
    # chat & social
    "chat": "💬", "convo": "💬", "conversation": "💬", "message": "💬", "intro": "👋",
    "hi": "👋", "hello": "👋", "greeting": "👋", "hey": "👋", "question": "❓",
    "help": "❓", "idea": "💡", "brainstorm": "💡", "advice": "💡", "advice?": "💡",
    # fun
    "game": "🎮", "games": "🎮", "gaming": "🎮", "puzzle": "🎮", "quiz": "🎮",
    "meme": "😂", "joke": "😂", "funny": "😂", "party": "🎉", "birthday": "🎉",
    "celebrate": "🎉", "gift": "🎁", "christmas": "🎄", "holiday-card": "🎄",
    # people & pets
    "family": "👨‍👩‍👧", "baby": "👶", "kids": "👶", "child": "👶", "children": "👶",
    "friend": "👤", "friends": "👤", "contact": "👤", "people": "👥", "team": "👥",
    "cat": "🐱", "cats": "🐱", "kitten": "🐱", "dog": "🐶", "dogs": "🐶",
    "puppy": "🐶", "pet": "🐾", "pets": "🐾", "hamster": "🐹", "bird": "🐦",
    "fish": "🐟", "aquarium": "🐟",
    # misc
    "shopping": "🛍", "shop": "🛍", "order": "📦", "package": "📦", "delivery": "📦",
    "shipping": "🛳", "trade": "🔄", "buy": "🛒", "sell": "🏷", "price": "🏷",
    "prices": "🏷", "discount": "🏷", "sale": "🏷", "weather": "⛅️", "forecast": "⛅️",
    "rain": "🌧", "snow": "❄️", "temperature": "🌡", "energy": "💡", "electricity": "⚡️",
    "power": "⚡️", "battery": "🔋", "time": "⏰", "clock": "⏰", "timer": "⏰",
    "location": "📍", "geocode": "📍", "address": "📍", "coords": "📍",
    "translate": "🔤", "translation": "🔤", "spelling": "🔤", "grammar": "🔤",
    "ai": "🤖", "agent": "🤖", "llm": "🤖", "model": "🤖", "prompt": "🤖",
    "hermes": "🤖", "automation": "🤖", "script": "📜", "cron": "⏰", "backup": "🗄",
    "archive": "🗄", "storage": "🗄", "file": "📄", "files": "📄", "folder": "📂",
    "pdf": "📄", "doc": "📄", "docx": "📄", "excel": "📊", "sheet": "📊",
    "spreadsheet": "📊", "csv": "📊", "chart": "📊", "charts": "📊", "graph": "📈",
    "stats": "📈", "statistics": "📈", "metrics": "📈", "dashboard": "📈",
    "report": "📊", "log": "📋", "logs": "📋", "debugging-notes": "📋",
    "secret": "🔒", "password": "🔒", "security": "🔒", "auth": "🔑", "login": "🔑",
    "key": "🔑", "token": "🔑", "certificate": "🔑", "wifi": "📶", "network": "📡",
    "networking": "📡", "ssh": "🔑", "vpn": "🔒", "proxy": "📡", "dns": "📡",
    "ip": "📡", "ip-address": "📡", "url": "🔗", "link": "🔗", "website": "🌐",
    "web": "🌐", "scrape": "🌐", "scraping": "🌐", "browser": "🌐", "email": "📧",
    "mail": "📧", "inbox": "📧", "spam": "📧", "sms": "📱", "phone": "📱",
    "android": "📱", "iphone": "📱", "ios": "📱", "desktop": "🖥", "laptop": "💻",
    "pc": "🖥", "mac": "🖥", "windows": "🖥", "linux": "🐧", "ubuntu": "🐧",
    "shell": "⌨️", "bash": "⌨️", "zsh": "⌨️", "command": "⌨️", "commands": "⌨️",
    "regex": "🔎", "search": "🔎", "searching": "🔎", "filter": "🔎", "grep": "🔎",
    "summary": "📝", "summarize": "📝", "digest": "📰", "weekly": "📅", "monthly": "📅",
    "yearly": "📅", "review": "👀", "checklist": "📝", "steps-to-reproduce": "📋",
}

# Fetch budget: one getForumTopicIconStickers call per process; failures retry
# no earlier than this many seconds (a permanently failing call must not add
# latency to every rename).
_ICON_SET_RETRY_SECONDS = 3600.0


def suggest_topic_icon_emoji(title: str) -> Optional[str]:
    """Emoji face (e.g. ``"🏠"``) for *title*, or None when nothing matches.

    The first title word carrying a mapping wins: the title's word order is
    the user's emphasis, so a keyword preference order would let a trailing
    "fix" override a leading "Home lights" and re-icon the topic away from
    what the conversation is actually about.
    """
    words = str(title or "").lower().split()
    for word in words:
        face = _TOPIC_ICON_KEYWORDS.get(word)
        if face is not None:
            return face
    return None


class ForumTopicIconIndex:
    """Index of the allowed forum-topic custom-emoji set, fetched lazily.

    Callers pass the ``bot`` object (anything exposing
    ``get_forum_topic_icon_stickers``); the class never imports PTB itself,
    which keeps it importable under the test mock and outside the adapter.
    """

    def __init__(self, fetch_timeout: float = 10.0):
        self._fetch_timeout = fetch_timeout
        self._by_face: Optional[Dict[str, str]] = None  # None = not fetched yet
        self._last_attempt: float = 0.0

    def resolve(self, face: Optional[str]) -> Optional[str]:
        """Allowed ``custom_emoji_id`` for *face* from the cached set, or None.

        Synchronous pure lookup on the last fetched index — callers wanting a
        fresh fetch should call :meth:`ensure_fetched` first.
        """
        if not face or not self._by_face:
            return None
        return self._by_face.get(face)

    async def ensure_fetched(self, bot: Any) -> None:
        """Fetch the allowed set once per process (retry-budgeted).

        Failure is latched for an hour, not forever: a transient network error
        at rename time should not disable icons for the process lifetime, but
        must also not add a network round-trip to every subsequent rename.
        """
        if self._by_face is not None or bot is None:
            return
        now = time.monotonic()
        if now - self._last_attempt < _ICON_SET_RETRY_SECONDS:
            return
        self._last_attempt = now
        try:
            stickers = await asyncio.wait_for(
                bot.get_forum_topic_icon_stickers(), timeout=self._fetch_timeout
            )
        except Exception:
            # Keep _by_face None; the retry budget above gates re-attempts.
            return
        by_face: Dict[str, str] = {}
        for sticker in stickers or ():
            face = getattr(sticker, "emoji", None)
            emoji_id = getattr(sticker, "custom_emoji_id", None)
            if face and emoji_id:
                by_face.setdefault(str(face), str(emoji_id))
        self._by_face = by_face


async def select_topic_icon(title: str, bot: Any, index: ForumTopicIconIndex) -> Optional[str]:
    """Allowed ``custom_emoji_id`` for *title*, or None (no eligible icon).

    One ``getForumTopicIconStickers`` call per process (cached by *index*);
    no LLM call — the model title already carries the semantics.
    """
    if bot is None:
        return None
    face = suggest_topic_icon_emoji(title)
    if not face:
        return None
    await index.ensure_fetched(bot)
    return index.resolve(face)
