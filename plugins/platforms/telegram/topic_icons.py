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
from ``auxiliary.title_generation``, in whatever language the user writes);
this module only projects it onto the fixed icon vocabulary Telegram allows.
The keyword table itself covers common English and Russian vocabulary —
Russian is matched tolerantly to inflections (``дома``/``доме``/``квартире``/
``жилья`` all resolve to their base concept) via light suffix stripping, and
tokenization ignores punctuation. Adding another language is a table
extension, not a code change.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any, Dict, List, Optional, Tuple

# keyword → emoji face. Only faces the forum-icon set is known to carry are
# listed; anything else would silently never match. The lookup is by whole
# word, so "home" does not fire inside "chrome". Russian keys are written in
# lowercase with "е" (not "ё"); lookup normalizes the same way, and Russian
# inflections are covered by the stem index built below.
_TOPIC_ICON_KEYWORDS: Dict[str, str] = {
    # home & household
    "home": "🏠", "house": "🏠", "flat": "🏠", "apartment": "🏠", "homelab": "🏠",
    "кухня": "🍳", "готовка": "🍳", "рецепт": "🍳", "рецепты": "🍳", "выпечка": "🍳",
    "kitchen": "🍳", "cooking": "🍳", "recipe": "🍳", "bake": "🍳", "baking": "🍳",
    "сад": "🌻", "огород": "🌻", "растение": "🌻", "цветы": "🌻", "рассада": "🌻",
    "garden": "🌻", "plants": "🌻", "plant": "🌻",
    # work & money
    "работа": "💼", "офис": "💼", "встреча": "💼", "карьера": "💼",
    "собеседование": "💼", "резюме": "💼", "вакансия": "💼",
    "деньги": "💰", "финансы": "💰", "финансовый": "💰", "бюджет": "💰",
    "инвестиции": "💰", "инвестиция": "💰", "крипта": "💰", "криптовалюта": "💰",
    "зарплата": "💰", "вклад": "💰", "ипотека": "💰", "налог": "💰",
    "налоги": "💰", "банк": "🏦", "банковский": "🏦",
    "work": "💼", "job": "💼", "office": "💼", "meeting": "💼", "career": "💼",
    "cv": "💼", "resume": "💼", "money": "💰", "finance": "💰", "financial": "💰",
    "budget": "💰", "invest": "💰", "investing": "💰", "investment": "💰",
    "stocks": "💰", "crypto": "💰", "salary": "💰", "invoice": "💰", "tax": "💰",
    "taxes": "💰", "bank": "🏦", "banking": "🏦",
    # engineering (Hermes' home turf — rich coverage)
    "код": "💻", "кода": "💻", "разработка": "💻", "программирование": "💻",
    "баг": "🐛", "баги": "🐛", "отладка": "🐛", "дебаг": "🐛",
    "фикс": "🔧", "ремонт": "🔧", "починка": "🔧",
    "code": "💻", "coding": "💻", "bug": "🐛", "debug": "🐛", "debugging": "🐛",
    "fix": "🔧", "fixing": "🔧", "refactor": "🔧", "build": "🔧", "builds": "🔧",
    "ci": "🔧", "deploy": "🚀", "deployment": "🚀", "release": "🚀", "ship": "🚀",
    "devops": "🚀", "сервер": "🖥", "сервера": "🖥", "компьютер": "🖥", "пк": "🖥",
    "server": "🖥", "servers": "🖥", "база": "🗃", "базы": "🗃", "миграция": "🗃",
    "миграции": "🗃", "database": "🗃", "postgres": "🗃", "mysql": "🗃", "sql": "🗃",
    "sqlite": "🗃", "migration": "🗃", "migrations": "🗃", "schema": "🗃",
    "docker": "🐳", "контейнер": "🐳", "контейнеры": "🐳", "докер": "🐳",
    "container": "🐳", "containers": "🐳", "kubernetes": "🐳", "k8s": "🐳",
    "api": "⚙️", "конфиг": "⚙️", "настройки": "⚙️", "настройка": "⚙️",
    "test": "✅", "тест": "✅", "тесты": "✅", "проверка": "✅", "pytest": "✅",
    "unittest": "✅", "tdd": "✅", "flaky": "✅", "pipeline": "⚙️",
    "performance": "⚡️", "perf": "⚡️", "скорость": "⚡️",
    "производительность": "⚡️", "электричество": "⚡️", "optimize": "⚡️",
    "optimization": "⚡️", "latency": "⚡️", "speed": "⚡️", "timeout": "⚠️",
    "ошибка": "⚠️", "ошибки": "⚠️", "сбой": "⚠️", "проблема": "⚠️",
    "проблемы": "⚠️", "инцидент": "⚠️", "падение": "⚠️", "regression": "⚠️",
    "incident": "⚠️", "outage": "⚠️", "error": "⚠️", "errors": "⚠️", "crash": "⚠️",
    "warn": "⚠️", "failing": "⚠️", "broken": "⚠️",
    # media
    "фото": "📷", "фотография": "📷", "фотографии": "📷", "фотограф": "📷",
    "снимок": "📷", "селфи": "📷", "камера": "📷", "photo": "📷", "photos": "📷",
    "picture": "📷", "pictures": "📷", "image": "🖼", "images": "🖼", "album": "🖼",
    "gallery": "🖼", "camera": "📷", "видео": "🎬", "ролик": "🎬", "фильм": "🎬",
    "кино": "🎬", "мультик": "🎬", "сериал": "🎬", "video": "🎬", "movie": "🎬",
    "movies": "🎬", "film": "🎬", "youtube": "🎬", "музыка": "🎵", "песня": "🎵",
    "трек": "🎵", "плейлист": "🎵", "song": "🎵", "music": "🎵", "playlist": "🎵",
    "аудио": "🎧", "подкаст": "🎧", "звук": "🎧", "audio": "🎧", "podcast": "🎧",
    "voice": "🎧",
    # travel
    "поездка": "✈️", "путешествие": "✈️", "перелет": "✈️", "рейс": "✈️",
    "билет": "✈️", "самолет": "✈️", "travel": "✈️", "trip": "✈️", "flight": "✈️",
    "flights": "✈️", "отпуск": "🌴", "каникулы": "🌴", "пляж": "🌴", "море": "🌴",
    "отдых": "🌴", "vacation": "🌴", "holiday": "🌴", "beach": "🌴",
    "отель": "🏨", "гостиница": "🏨", "бронь": "🏨", "бронирование": "🏨",
    "hotel": "🏨", "booking": "🏨", "паспорт": "🛂", "виза": "🛂", "passport": "🛂",
    "visa": "🛂", "карта": "🗺", "маршрут": "🗺", "навигация": "🗺", "map": "🗺",
    "route": "🗺", "directions": "🗺", "поезд": "🚄", "train": "🚄",
    "машина": "🚗", "авто": "🚗", "автомобиль": "🚗", "такси": "🚗",
    "car": "🚗", "drive": "🚗", "driving": "🚗",
    # health & sport
    "здоровье": "💊", "лекарство": "💊", "врач": "💊", "таблетки": "💊",
    "health": "💊", "medicine": "💊", "doctor": "💊", "symptom": "💊", "pill": "💊",
    "спортзал": "🏋️", "качалка": "🏋️", "зал": "🏋️", "бег": "🏃", "пробежка": "🏃",
    "тренировка": "🏃", "workout": "🏋️", "gym": "🏋️", "run": "🏃", "running": "🏃",
    "training": "🏃", "спорт": "⚽️", "футбол": "⚽️", "sport": "⚽️",
    "football": "⚽️", "soccer": "⚽️", "steps": "🏃", "сон": "😴", "sleep": "😴",
    "диета": "🥗", "питание": "🥗", "калории": "🥗", "вес": "⚖️", "diet": "🥗",
    "calories": "🥗", "nutrition": "🥗", "weight": "⚖️", "keto": "🥗",
    # knowledge & planning
    "книга": "📖", "чтение": "📖", "новости": "📰", "статья": "📰",
    "read": "📖", "reading": "📖", "book": "📖", "books": "📖", "article": "📰",
    "news": "📰", "research": "🔬", "учеба": "📚", "обучение": "📚", "урок": "📚",
    "курс": "📚", "язык": "📚", "study": "📚", "learn": "📚", "learning": "📚",
    "course": "📚", "language": "📚", "english": "📚", "spanish": "📚",
    "german": "📚", "заметки": "📝", "план": "📝", "список": "📝", "запись": "📝",
    "note": "📝", "notes": "📝", "todo": "📝", "planning": "📝",
    "календарь": "📅", "расписание": "📅", "напоминание": "📅", "дедлайн": "📅",
    "schedule": "📅", "calendar": "📅", "reminder": "📅", "deadline": "📅",
    "meeting-notes": "📝", "обзор": "👀", "ревью": "👀", "review": "👀",
    # chat & social
    "чат": "💬", "разговор": "💬", "беседа": "💬", "переписка": "💬",
    "привет": "👋", "здравствуйте": "👋", "знакомство": "👋", "chat": "💬",
    "convo": "💬", "conversation": "💬", "message": "💬", "intro": "👋",
    "hi": "👋", "hello": "👋", "greeting": "👋", "hey": "👋", "вопрос": "❓",
    "вопросы": "❓", "помощь": "❓", "помогите": "❓", "question": "❓",
    "help": "❓", "идея": "💡", "идеи": "💡", "энергия": "💡", "idea": "💡",
    "brainstorm": "💡", "advice": "💡", "advice?": "💡",
    # fun
    "игра": "🎮", "игры": "🎮", "мем": "😂", "мемы": "😂", "шутка": "😂",
    "анекдот": "😂", "праздник": "🎉", "подарок": "🎁", "подарки": "🎁",
    "game": "🎮", "games": "🎮", "gaming": "🎮", "puzzle": "🎮", "quiz": "🎮",
    "meme": "😂", "joke": "😂", "funny": "😂", "party": "🎉", "birthday": "🎉",
    "celebrate": "🎉", "gift": "🎁", "christmas": "🎄", "holiday-card": "🎄",
    # people & pets
    "семья": "👨‍👩‍👧", "ребенок": "👶", "дети": "👶", "family": "👨‍👩‍👧",
    "baby": "👶", "kids": "👶", "child": "👶", "children": "👶", "друг": "👤",
    "друзья": "👤", "знакомый": "👤", "люди": "👥", "команда": "👥",
    "friend": "👤", "friends": "👤", "contact": "👤", "people": "👥",
    "team": "👥", "кот": "🐱", "кошка": "🐱", "котенок": "🐱", "котята": "🐱",
    "пес": "🐶", "собака": "🐶", "щенок": "🐶", "щенки": "🐶", "питомец": "🐾",
    "животное": "🐾", "хомяк": "🐹", "птица": "🐦", "рыба": "🐟", "аквариум": "🐟",
    "cat": "🐱", "cats": "🐱", "kitten": "🐱", "dog": "🐶", "dogs": "🐶",
    "puppy": "🐶", "pet": "🐾", "pets": "🐾", "hamster": "🐹", "bird": "🐦",
    "fish": "🐟", "aquarium": "🐟",
    # misc
    "покупки": "🛍", "магазин": "🛍", "шопинг": "🛍", "заказ": "📦",
    "посылка": "📦", "доставка": "📦", "обмен": "🔄", "купить": "🛒",
    "продажа": "🏷", "продать": "🏷", "продам": "🏷", "цена": "🏷", "цены": "🏷",
    "стоимость": "🏷", "скидка": "🏷", "распродажа": "🏷", "shopping": "🛍",
    "shop": "🛍", "order": "📦", "package": "📦", "delivery": "📦",
    "shipping": "🛳", "trade": "🔄", "buy": "🛒", "sell": "🏷", "price": "🏷",
    "prices": "🏷", "discount": "🏷", "sale": "🏷", "погода": "⛅️",
    "прогноз": "⛅️", "дождь": "🌧", "снег": "❄️", "зима": "❄️",
    "температура": "🌡", "weather": "⛅️", "forecast": "⛅️", "rain": "🌧",
    "snow": "❄️", "батарея": "🔋", "заряд": "🔋", "зарядка": "🔋",
    "temperature": "🌡", "energy": "💡", "electricity": "⚡️", "power": "⚡️",
    "battery": "🔋", "время": "⏰", "таймер": "⏰", "будильник": "⏰",
    "time": "⏰", "clock": "⏰", "timer": "⏰", "адрес": "📍", "место": "📍",
    "локация": "📍", "location": "📍", "geocode": "📍", "coords": "📍",
    "перевод": "🔤", "translate": "🔤", "translation": "🔤", "spelling": "🔤",
    "grammar": "🔤", "агент": "🤖", "нейросеть": "🤖", "модель": "🤖", "бот": "🤖",
    "ai": "🤖", "agent": "🤖", "llm": "🤖", "model": "🤖", "prompt": "🤖",
    "hermes": "🤖", "automation": "🤖", "скрипт": "📜", "cron": "⏰",
    "бэкап": "🗄", "архив": "🗄", "хранилище": "🗄", "backup": "🗄",
    "файл": "📄", "файлы": "📄", "документ": "📄", "документы": "📄",
    "storage": "🗄", "file": "📄", "files": "📄", "folder": "📂", "pdf": "📄",
    "doc": "📄", "docx": "📄", "таблица": "📊", "отчет": "📊", "график": "📈",
    "статистика": "📈", "excel": "📊", "sheet": "📊", "spreadsheet": "📊",
    "csv": "📊", "chart": "📊", "charts": "📊", "graph": "📈", "stats": "📈",
    "statistics": "📈", "metrics": "📈", "dashboard": "📈", "report": "📊",
    "лог": "📋", "логи": "📋", "log": "📋", "logs": "📋",
    "пароль": "🔒", "безопасность": "🔒", "secret": "🔒", "password": "🔒",
    "security": "🔒", "логин": "🔑", "ключ": "🔑", "auth": "🔑", "login": "🔑",
    "key": "🔑", "token": "🔑", "certificate": "🔑", "вайфай": "📶", "сеть": "📡",
    "сети": "📡", "прокси": "📡", "wifi": "📶", "network": "📡",
    "networking": "📡", "ssh": "🔑", "vpn": "🔒", "proxy": "📡", "dns": "📡",
    "ip": "📡", "ip-address": "📡", "ссылка": "🔗", "сайт": "🌐", "браузер": "🌐",
    "url": "🔗", "link": "🔗", "website": "🌐", "web": "🌐", "scrape": "🌐",
    "scraping": "🌐", "browser": "🌐", "почта": "📧", "письмо": "📧",
    "письма": "📧", "email": "📧", "mail": "📧", "inbox": "📧", "spam": "📧",
    "телефон": "📱", "смс": "📱", "sms": "📱", "phone": "📱", "android": "📱",
    "iphone": "📱", "ios": "📱", "desktop": "🖥", "laptop": "💻", "pc": "🖥",
    "mac": "🖥", "windows": "🖥", "linux": "🐧", "ubuntu": "🐧",
    "поиск": "🔎", "shell": "⌨️", "bash": "⌨️", "zsh": "⌨️", "command": "⌨️",
    "commands": "⌨️", "regex": "🔎", "search": "🔎", "searching": "🔎",
    "filter": "🔎", "grep": "🔎", "summary": "📝", "summarize": "📝",
    "digest": "📰", "weekly": "📅", "monthly": "📅", "yearly": "📅",
    "checklist": "📝", "steps-to-reproduce": "📋", "debugging-notes": "📋",
    # home (Russian; kept last only for readability: the leading-word rule
    # decides which concept wins, never this declaration order)
    "дом": "🏠", "квартира": "🏠", "жилье": "🏠", "дача": "🏠",
    "коттедж": "🏠", "апартаменты": "🏠", "домашний": "🏠",
}

# Russian inflection suffixes for the stem index: applied once, longest
# first, never shortening a word below three letters. This is deliberately
# NOT a general morphological analyzer — it only needs to converge title
# word forms (дома/доме/дому/жилья/продажи) onto the table's base forms.
_RU_SUFFIXES = (
    "иями", "ями", "ами", "иях",
    "ому", "ему", "ого", "его", "ыми", "ими", "ете", "ите",
    "ах", "ях", "ов", "ев", "ей", "ой", "ый", "ий", "ая", "яя", "ое", "ее",
    "ые", "ие", "ет", "ют", "ут", "ат", "ят", "ла", "ть",
    "ом", "ем", "ам", "ям", "ую", "юю",
    "у", "ю", "а", "я", "ы", "и", "е", "о",
)
_RU_SUFFIXES_SORTED = tuple(sorted(set(_RU_SUFFIXES), key=len, reverse=True))

_TOKEN_RE = re.compile(r"\w+")
_HAS_CYR_RE = re.compile("[а-яё]")


def _normalize_word(word: str) -> str:
    return word.lower().replace("ё", "е")


def _ru_stem(word: str) -> str:
    for suffix in _RU_SUFFIXES_SORTED:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


# Stem index for Russian keywords: title word → stem must equal keyword →
# stem, so inflected title forms resolve without enumerating every case.
_RU_STEM_INDEX: Dict[str, str] = {}
for _keyword, _face in _TOPIC_ICON_KEYWORDS.items():
    if _HAS_CYR_RE.search(_keyword):
        _RU_STEM_INDEX.setdefault(_ru_stem(_normalize_word(_keyword)), _face)

# Fetch budget: one getForumTopicIconStickers call per process; failures retry
# no earlier than this many seconds (a permanently failing call must not add
# latency to every rename).
_ICON_SET_RETRY_SECONDS = 3600.0


def suggest_topic_icon_emoji(title: str) -> Optional[str]:
    """Emoji face (e.g. ``"🏠"``) for *title*, or None when nothing matches.

    The first title word carrying a mapping wins: the title's word order is
    the user's emphasis, so a keyword preference order would let a trailing
    "fix" override a leading "Home lights" and re-icon the topic away from
    what the conversation is actually about. Words are matched in any
    language the table covers; Russian word forms are matched via the stem
    index (``дома`` → ``дом`` → 🏠), and punctuation is not part of a word.
    """
    for token in _TOKEN_RE.findall(str(title or "")):
        word = _normalize_word(token)
        face = _TOPIC_ICON_KEYWORDS.get(word)
        if face is not None:
            return face
        if _HAS_CYR_RE.search(word):
            face = _RU_STEM_INDEX.get(_ru_stem(word))
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
