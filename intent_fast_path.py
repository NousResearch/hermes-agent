"""Pre-LLM intent fast-path for the Hermes gateway.

Why: The normal request flow runs a two-level orchestrator/worker LLM agent
loop, which makes even trivial deterministic questions (e.g. "weather") take
21-63s.  For a small set of well-defined intents we can answer directly from a
cheap HTTP API in ~300-500ms, bypassing the agent entirely.  Latency is the
whole point — if anything is ambiguous we MUST defer to the agent rather than
risk a wrong/partial answer.

What: A tiny intent registry plus an async ``_intent_fast_path(text)`` dispatch
function.  Each intent is a ``(matcher, handler)`` pair.  The matcher is a fast
synchronous predicate; the handler is an async function that returns the reply
string on a clean hit or ``None`` to fall through to the agent.  The first
handler that returns a non-None string wins.  Currently ships one intent:
weather (Open-Meteo).

Test: Import this module standalone (no heavy gateway deps), call
``_intent_fast_path("weather")`` and assert a string; call it with
"weather affects my mood" and assert ``None``.  Handler-level behaviour is
covered by monkeypatching ``httpx.AsyncClient`` (see tests/test_intent_fast_path.py).

Design note — STRICT FALL-THROUGH: every handler returns ``None`` on ANY doubt
(no match, empty geocoding, timeout, HTTP error, parse error, missing fields).
A ``None`` return means "I am not confident, let the agent handle it."  We never
return an empty or partial string.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Awaitable, Callable, List, Optional, Tuple
from urllib.parse import quote

logger = logging.getLogger(__name__)

# Intent registry.  Each entry: (matcher, handler).
#   matcher: Callable[[str], bool]      — cheap sync predicate
#   handler: Callable[[str], Awaitable[Optional[str]]] — async, None == fall through
_INTENT_HANDLERS: List[
    Tuple[Callable[[str], bool], Callable[[str], Awaitable[Optional[str]]]]
] = []


def register_intent(
    matcher: Callable[[str], bool],
    handler: Callable[[str], Awaitable[Optional[str]]],
) -> None:
    """Register a fast-path intent.

    Why: Adding a future intent should be a one-liner — define a matcher + an
    async handler, call ``register_intent(...)`` at import time, done.
    What: Appends the ``(matcher, handler)`` pair to the dispatch list.
    Test: Call ``register_intent(lambda t: True, h)`` then assert the pair is the
    last element of ``_INTENT_HANDLERS``.
    """
    _INTENT_HANDLERS.append((matcher, handler))


async def _intent_fast_path(text: str) -> Optional[str]:
    """Dispatch ``text`` through registered intents; return a reply or None.

    Why: Single entry point the gateway calls before building any agent.  Must
    be bullet-proof — a buggy matcher/handler must never raise into the gateway,
    it must just defer to the agent.
    What: Iterates intents in registration order; the first handler returning a
    non-None string wins.  Any exception in a matcher or handler is swallowed and
    treated as "no match" (fall through).
    Test: Register an intent whose handler raises, then assert
    ``_intent_fast_path("x")`` returns None instead of propagating.
    """
    for matcher, handler in _INTENT_HANDLERS:
        try:
            if matcher(text):
                result = await handler(text)
                if result is not None:
                    return result
        except Exception:  # noqa: BLE001 — fall-through is the safe default
            # Never let a fast-path failure break the request; defer to agent.
            pass
    return None


# ──────────────────────────────────────────────────────────────────────────
# Weather intent
# ──────────────────────────────────────────────────────────────────────────

# Woodstock, IL — the default "home" location when no place is named.  Hardcoded
# so the common "weather" / "is it raining" case skips the geocoding round-trip.
_DEFAULT_LAT = 42.3147
_DEFAULT_LON = -88.4487
_DEFAULT_NAME = "Woodstock, IL"

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Per-call HTTP budget: 1s to connect, 1.5s to read the body.  The old flat
# ``timeout=2.0`` applied to BOTH the geocode and the forecast leg, so a named
# location could burn ~4s — breaking the sub-second promise.  We additionally
# cap the whole geocode+forecast (named-location) branch with a hard ceiling
# below.
_HTTP_TIMEOUT_CONNECT = 1.0
_HTTP_TIMEOUT_READ = 1.5
# Hard wall-clock ceiling for the two-call named-location branch.
_NAMED_BRANCH_CEILING_S = 3.5

# Matcher.  Anchored to end-of-string so it fires on a *question about* the
# weather, not on conversational sentences that merely contain the word
# "weather" ("weather affects my mood", "a story about weather in...").
#
# To avoid false positives like "weather affects my mood", a trailing location
# is ONLY accepted when it is introduced by an explicit connector
# (in/for/at/near/around) OR the message is just the keyword (+ trivial
# fillers).  A bare "weather <free text>" without a connector is treated as
# location-less ONLY if the trailing text is empty/filler; otherwise it does
# not match (it's prose, defer to the agent).
#
# Accepted shapes (end-anchored):
#   1. Keyword alone / with fillers:      "weather", "weather today", "forecast now"
#   2. Optional question lead-in + keyword + connector + location:
#        "what's the weather in Denver", "weather for Woodstock, IL 60098"
#   3. "is it <condition>" with optional connector-introduced location:
#        "is it raining", "is it snowing in chicago"
_WEATHER_KEYWORD = r"(?:weather|forecast|temperature|temp)"
# Question lead-ins we tolerate before the keyword.
_LEADIN = (
    r"(?:what(?:'|’)?s?\s+(?:is\s+)?the\s+|what\s+is\s+the\s+|"
    r"how(?:'|’)?s?\s+the\s+|tell\s+me\s+the\s+|"
    r"give\s+me\s+the\s+|show\s+me\s+the\s+|current\s+|today(?:'|’)?s?\s+)?"
)
# Trivial fillers allowed directly after the keyword with no location.  NOTE:
# "like" is handled separately by ``_LIKE`` (below) because it is the one filler
# that can idiomatically bridge to a connector ("what's the weather LIKE in X");
# the others (report/today/…) must NEVER bridge to a connector.
# Time qualifiers (tonight, this week, …) are also accepted as fillers — they
# narrow the *when* of the forecast, not the *where*.
_FILLER = (
    r"(?:\s+(?:"
    r"report|today|now|outside|right\s+now|please"
    r"|tonight|tomorrow"
    r"|this\s+week|this\s+weekend|this\s+morning|this\s+afternoon|this\s+evening"
    r"|next\s+week|next\s+weekend|this\s+month"
    r"))*"
)

# The idiomatic "what's the weather LIKE in Denver" bridge — the only word
# permitted between the keyword and a connector.  Optional and singular.
_LIKE = r"(?:\s+like)?"

# Connector tokens that introduce a trailing location.
_CONNECTOR = r"(?:in|for|at|near|around)"

# Shape 1+2 — restructured into two NON-optional, end-anchored alternations so a
# NOUN filler (report/budget/…) can NEVER be followed by a connector+location
# (that combination is what let "weather report for the Q3 sales" leak through):
#   (a) keyword + optional filler + EOL        e.g. "weather", "weather report"
#   (b) keyword + [optional "like"] + connector + location + EOL
#                                              e.g. "weather in Denver",
#                                              "what's the weather like in Denver"
# Branch (a) is tried first; if text remains it backtracks to (b), where the only
# token allowed between the keyword and the connector is the idiomatic "like".
# The connector-captured location is independently sanity-checked by
# ``_connector_location_ok`` before we ever treat it as a real place — that guard
# is the backstop that rejects "the Q3 sales", "my code", etc.
_WEATHER_RE = re.compile(
    r"^\s*"
    + _LEADIN
    + _WEATHER_KEYWORD
    + r"(?:"
    + _FILLER  # (a) keyword + optional filler ...
    + r"|"
    + _LIKE
    + r"\s+"
    + _CONNECTOR
    + r"\s+(?P<loc1>[^?]+?)"  # (b) keyword [like] connector loc
    + r")"
    + r"\s*\??\s*$",  # ... + EOL (shared by both branches)
    re.IGNORECASE,
)
_IS_IT_RE = re.compile(
    r"^\s*is\s+it\s+"
    r"(?:rain|snow|sunny|cloud|fog|storm|hot|cold|warm|cool|wind)\w*"
    r"(?:\s+(?:out|outside|today|now|right\s+now))?"
    r"(?:\s+(?:in|at|near|around)\s+(?P<loc2>[^?]+?))?"
    r"\s*\??\s*$",
    re.IGNORECASE,
)
# Shape: "weather <place>" WITHOUT a connector — only when every trailing token
# is place-like (a word starting with a letter, or a US state abbr), capped at
# 4 tokens.  This catches "weather woodstock il" / "temperature austin tx"
# while rejecting "weather affects my mood" (has the stopword-y verb "affects").
_NO_CONNECTOR_RE = re.compile(
    r"^\s*" + _WEATHER_KEYWORD + r"\s+(?P<loc3>[A-Za-z][A-Za-z .'-]*?)" + r"\s*\??\s*$",
    re.IGNORECASE,
)
# Words that, if present in a trailing "location" phrase (either the
# no-connector shape OR a connector-introduced one), mark it as prose / a
# non-place noun rather than a real place name.  When any of these appears we
# REJECT the candidate and fall through to the agent — a false weather answer is
# far worse than a missed fast-path.
_NON_PLACE_WORDS = {
    # grammar / pronouns / articles
    "affects",
    "affect",
    "is",
    "was",
    "were",
    "are",
    "my",
    "your",
    "his",
    "her",
    "their",
    "our",
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "of",
    "about",
    "when",
    "we",
    "i",
    "you",
    "they",
    "it",
    "this",
    "that",
    "next",
    "permitting",
    # time / vibe words that aren't places
    "mood",
    "today",
    "tomorrow",
    "yesterday",
    "here",
    "there",
    "nice",
    "bad",
    "good",
    "like",
    "love",
    "hate",
    "now",
    "outside",
    "feels",
    "feel",
    "looks",
    "look",
    "story",
    "world",
    "worlds",
    "fantasy",
    # business / office / non-place nouns that geocoders wrongly resolve
    "meeting",
    "lab",
    "code",
    "budget",
    "sales",
    "quarter",
    "report",
    "market",
    "stock",
    "project",
    "team",
    "call",
    "email",
    "deadline",
    "sprint",
    "standup",
    "review",
    "roadmap",
    "backlog",
    "ticket",
    "issue",
    "demo",
    "launch",
    "release",
    "metrics",
    "revenue",
    "kpi",
    "okr",
    # abstract location nouns — not geocodable; fall through to agent for clarification
    "work",
    "school",
    "office",
    "class",
    "gym",
    "church",
    "home",
}

# Time expressions that may appear after a connector (e.g. "forecast for this
# week") — they are NOT location names, but they also do NOT make the query
# non-weather.  When a connector group captures one of these we treat the whole
# query as a weather request for the default home location (no geocoding).
_TIME_QUALIFIERS: frozenset = frozenset({
    "today",
    "tonight",
    "tomorrow",
    "now",
    "right now",
    "this week",
    "this weekend",
    "this morning",
    "this afternoon",
    "this evening",
    "next week",
    "next weekend",
    "this month",
    "the week",
    "the weekend",
})

# State abbreviations / tokens we strip from a parsed location so geocoding
# (which chokes on "City, ST ZIP") gets a clean city name.
_US_STATE_ABBRS = {
    "al",
    "ak",
    "az",
    "ar",
    "ca",
    "co",
    "ct",
    "de",
    "fl",
    "ga",
    "hi",
    "id",
    "il",
    "in",
    "ia",
    "ks",
    "ky",
    "la",
    "me",
    "md",
    "ma",
    "mi",
    "mn",
    "ms",
    "mo",
    "mt",
    "ne",
    "nv",
    "nh",
    "nj",
    "nm",
    "ny",
    "nc",
    "nd",
    "oh",
    "ok",
    "or",
    "pa",
    "ri",
    "sc",
    "sd",
    "tn",
    "tx",
    "ut",
    "vt",
    "va",
    "wa",
    "wv",
    "wi",
    "wy",
    "dc",
}

_ZIP_RE = re.compile(r"\b\d{5}(?:-\d{4})?\b")

# WMO weather interpretation codes → short human text.
_WMO_CODES = {
    0: "Clear",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Rime fog",
    51: "Light drizzle",
    53: "Drizzle",
    55: "Heavy drizzle",
    56: "Freezing drizzle",
    57: "Freezing drizzle",
    61: "Light rain",
    63: "Rain",
    65: "Heavy rain",
    66: "Freezing rain",
    67: "Freezing rain",
    71: "Light snow",
    73: "Snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Light showers",
    81: "Showers",
    82: "Heavy showers",
    85: "Snow showers",
    86: "Snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm w/ hail",
    99: "Thunderstorm w/ hail",
}


def _wmo_text(code: object) -> str:
    """Map a WMO weathercode to short text, defaulting gracefully.

    Why: Open-Meteo returns integer codes; users want words.
    What: Looks up ``_WMO_CODES``; unknown/None codes yield "Unknown".
    Test: Assert ``_wmo_text(0) == "Clear"``, ``_wmo_text(61) == "Light rain"``,
    ``_wmo_text(None) == "Unknown"``.
    """
    try:
        return _WMO_CODES.get(int(code), "Unknown")  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "Unknown"


def _is_place_like(phrase: str) -> bool:
    """Heuristic: does ``phrase`` look like a (no-connector) place name?

    Why: "weather woodstock il" should match; "weather affects my mood" should
    not.  Without a connector we need a guard against prose being parsed as a
    location.
    What: True iff the phrase is ≤60 chars and 1-4 tokens, every token starts
    with a letter, and no token is a known non-place stopword/verb.
    Test: True for "woodstock il", "new york", "austin tx"; False for
    "affects my mood", "was nice yesterday", and a single 70-char token.
    """
    phrase = phrase.strip()
    # Guard against an arbitrarily long single token sneaking through.
    if len(phrase) > 60:
        return False
    tokens = [t for t in re.split(r"\s+", phrase) if t]
    if not tokens or len(tokens) > 4:
        return False
    for tok in tokens:
        low = tok.lower().strip(".,'-")
        if not low or not low[0].isalpha():
            return False
        if low in _NON_PLACE_WORDS:
            return False
    return True


def _connector_location_ok(phrase: Optional[str]) -> bool:
    """Sanity-check a connector-introduced location (loc1/loc2) before trusting it.

    Why: The connector path (``in|for|at|near|around <X>``) would otherwise accept
    ANY trailing noun as a place, so non-weather prose got geocoded into a bogus
    weather answer ("forecast for the meeting" -> Nenagh, Ireland; "forecast in the
    lab" -> Indiana).  A false answer is far worse than a missed match, so when in
    doubt we REJECT.
    What: Returns False (reject) when the candidate (a) is empty/letterless, (b)
    starts with "the ", (c) is longer than 60 chars, (d) has more than 5 tokens,
    or (e) contains any ``_NON_PLACE_WORDS`` stopword.  Unlike ``_is_place_like``
    it tolerates digit tokens (ZIP codes) so "Woodstock, IL 60098" still passes.
    Test: True for "Denver", "New York City", "Woodstock, IL 60098"; False for
    "the meeting", "the lab", "budget for next quarter", "my code",
    "the stock market".
    """
    if not phrase:
        return False
    candidate = phrase.strip()
    if len(candidate) < 2 or len(candidate) > 60:
        return False
    if not re.search(r"[A-Za-z]", candidate):
        return False
    # Reject "the <noun>" — almost always prose, never how a place is named here.
    if re.match(r"(?i)^the\s+", candidate):
        return False
    tokens = [t for t in re.split(r"\s+", candidate) if t]
    if len(tokens) > 5:
        return False
    for tok in tokens:
        low = tok.lower().strip(".,'-")
        if low in _NON_PLACE_WORDS:
            return False
    return True


def _weather_matcher(text: str) -> bool:
    """Return True iff ``text`` is a weather question.

    Why: Gate the handler so we never even attempt a fast-path on unrelated text.
    What: Matches keyword-alone / connector-introduced-location / "is it
    <condition>" shapes — but the connector-captured location must pass
    ``_connector_location_ok`` (reject "the meeting", "my code", …) — UNLESS the
    captured string is a time qualifier ("this week", "tonight", …), in which case
    we still match (the query is weather for the default home location).
    Plus a guarded no-connector "weather <place>" shape.
    Test: True for "weather", "what's the weather in Denver", "is it raining",
    "weather woodstock il", "what's the forecast for this week", "forecast today",
    "weather tonight"; False for "weather affects my mood",
    "forecast for the meeting", "forecast in the lab",
    "weather report for the Q3 sales", "/weather".
    """
    m = _WEATHER_RE.match(text)
    if m:
        loc1 = m.group("loc1")
        if loc1 is None:
            return True  # keyword-only / filler — clean weather question
        if loc1.strip().lower() in _TIME_QUALIFIERS:
            return True  # time qualifier — still weather, use default location
        if _connector_location_ok(loc1):
            return True  # looks like a real place
        return False
    m2 = _IS_IT_RE.match(text)
    if m2:
        loc2 = m2.group("loc2")
        if loc2 is None:
            return True
        if loc2.strip().lower() in _TIME_QUALIFIERS:
            return True  # time qualifier — still weather, use default location
        if _connector_location_ok(loc2):
            return True
        return False
    m3 = _NO_CONNECTOR_RE.match(text)
    if m3:
        loc3 = m3.group("loc3")
        # Time qualifiers without a connector ("weather this weekend") should
        # match as a default-location weather request.
        if loc3.strip().lower() in _TIME_QUALIFIERS or _is_place_like(loc3):
            return True
    return False


def _extract_location(text: str) -> Optional[str]:
    """Pull a raw location string out of a weather question, or None.

    Why: "weather" alone means the default home location; "weather in Denver"
    means Denver.  Distinguishing the two drives whether we geocode.
    What: Reads the location from whichever shape matched (connector group, the
    is-it group, or the guarded no-connector group), applies the SAME
    ``_connector_location_ok`` sanity check used by the matcher to the connector
    captures, strips trailing punctuation, and rejects noise (empty, <2 chars,
    no letters).  A connector location that fails the guard yields None so the
    request defers to the agent rather than geocoding prose.
    Test: "weather" -> None; "weather woodstock il" -> "woodstock il";
    "what's the weather in Denver" -> "Denver"; "weather 12" -> None;
    "forecast for the meeting" -> None.
    """
    loc: Optional[str] = None
    m = _WEATHER_RE.match(text)
    if m:
        cand = m.group("loc1")
        if cand:
            if cand.strip().lower() in _TIME_QUALIFIERS:
                return None  # time qualifier — treat as default location request
            if _connector_location_ok(cand):
                loc = cand
    if not loc:
        m2 = _IS_IT_RE.match(text)
        if m2:
            cand2 = m2.group("loc2")
            if cand2:
                if cand2.strip().lower() in _TIME_QUALIFIERS:
                    return None  # time qualifier — treat as default location request
                if _connector_location_ok(cand2):
                    loc = cand2
    if not loc:
        m3 = _NO_CONNECTOR_RE.match(text)
        if m3:
            loc3 = m3.group("loc3")
            # Time qualifier without connector — treat as default location (no geocode).
            if loc3.strip().lower() not in _TIME_QUALIFIERS and _is_place_like(loc3):
                loc = loc3
    if loc is None:
        return None
    loc = loc.strip().strip(".,!?;:").strip()
    if len(loc) < 2 or not re.search(r"[A-Za-z]", loc):
        return None
    return loc


def _clean_location_for_geocode(loc: str) -> str:
    """Reduce a messy location to a bare city name for Open-Meteo geocoding.

    Why: Open-Meteo's geocoder fails on "City, ST ZIP" — it wants just the city.
    What: Takes the token before the first comma, strips ZIP codes and trailing
    2-letter US state abbreviations, collapses whitespace.
    Test: "Woodstock, IL 60098" -> "Woodstock"; "Denver, CO" -> "Denver";
    "New York" -> "New York".
    """
    # City is whatever precedes the first comma (if any).
    city = loc.split(",")[0]
    # Drop ZIP codes anywhere in the remaining string.
    city = _ZIP_RE.sub(" ", city)
    # Drop a trailing standalone state abbreviation ("Woodstock IL" -> "Woodstock").
    tokens = [t for t in re.split(r"\s+", city.strip()) if t]
    while tokens and tokens[-1].lower().strip(".") in _US_STATE_ABBRS:
        tokens.pop()
    cleaned = " ".join(tokens).strip()
    return cleaned or loc.split(",")[0].strip()


def _build_display_name(result: dict) -> str:
    """Compose a human display name from a geocoding result.

    Why: "Denver, Colorado, United States" reads better than raw "Denver".
    What: Joins name + admin1 + country, skipping blanks/dupes.
    Test: ``{"name":"Denver","admin1":"Colorado","country":"United States"}``
    -> "Denver, Colorado, United States".
    """
    parts: List[str] = []
    for key in ("name", "admin1", "country"):
        val = result.get(key)
        if isinstance(val, str) and val.strip() and val not in parts:
            parts.append(val.strip())
    return ", ".join(parts) if parts else str(result.get("name") or "")


async def _geocode(client: "object", city: str) -> Optional[Tuple[float, float, str]]:
    """Resolve a city name to (lat, lon, display_name) via Open-Meteo, or None.

    Why: Named locations need coordinates before we can fetch a forecast.
    What: Calls the geocoding API for the cleaned city; returns the top hit's
    lat/lon/display name.  Returns None on empty/missing results, HTTP error, or
    parse error — caller treats None as fall-through.
    Test: Monkeypatch the client to return ``{"results": []}`` -> None; to return
    one result -> (lat, lon, name).
    """
    url = f"{_GEOCODE_URL}?name={quote(city)}&count=1&language=en&format=json"
    try:
        resp = await client.get(url)  # type: ignore[attr-defined]
        if resp.status_code >= 400:
            return None
        data = resp.json()
    except Exception:  # noqa: BLE001
        return None
    results = data.get("results") if isinstance(data, dict) else None
    if not results:
        return None
    top = results[0]
    try:
        lat = float(top["latitude"])
        lon = float(top["longitude"])
    except (KeyError, TypeError, ValueError):
        return None
    return lat, lon, _build_display_name(top)


def _format_day_label(index: int) -> str:
    """Label a forecast day by offset.

    Why: "Today / Tomorrow / +2d" reads better than bare dates.
    What: 0 -> "Today", 1 -> "Tomorrow", else "Day N".
    Test: Assert _format_day_label(0)=="Today", (1)=="Tomorrow".
    """
    if index == 0:
        return "Today"
    if index == 1:
        return "Tomorrow"
    return f"Day {index + 1}"


def _render_weather(display_name: str, data: dict) -> Optional[str]:
    """Render the Open-Meteo forecast payload into a terse Telegram-safe reply.

    Why: Centralizes the "is this payload complete enough to answer?" check and
    the formatting, so the handler stays small.
    What: Requires ``current.temperature_2m``; emits current temp/condition/wind
    plus up to a 3-day forecast (label: condition, low–high °F).  Returns None if
    the required current temperature is missing (strict fall-through).
    Test: Pass a canned payload with current+daily -> assert the string contains
    the temp, condition word, and three day labels; pass a payload missing
    ``current.temperature_2m`` -> assert None.
    """
    current = data.get("current") if isinstance(data, dict) else None
    if not isinstance(current, dict):
        return None
    temp = current.get("temperature_2m")
    if temp is None:
        # Strict fall-through: without a current temperature we have no answer.
        return None

    cond = _wmo_text(current.get("weathercode"))
    wind = current.get("windspeed_10m")

    lines: List[str] = []
    header = f"*Weather — {display_name}*"
    lines.append(header)

    try:
        temp_str = f"{round(float(temp))}°F"
    except (TypeError, ValueError):
        return None
    now_line = f"Now: {temp_str}, {cond}"
    if wind is not None:
        try:
            now_line += f", wind {round(float(wind))} mph"
        except (TypeError, ValueError):
            pass
    lines.append(now_line)

    daily = data.get("daily") if isinstance(data, dict) else None
    if isinstance(daily, dict):
        codes = daily.get("weathercode") or []
        highs = daily.get("temperature_2m_max") or []
        lows = daily.get("temperature_2m_min") or []
        n = min(len(codes), len(highs), len(lows), 3)
        if n:
            lines.append("")
            for i in range(n):
                label = _format_day_label(i)
                day_cond = _wmo_text(codes[i])
                try:
                    lo = round(float(lows[i]))
                    hi = round(float(highs[i]))
                    rng = f"{lo}–{hi}°F"
                except (TypeError, ValueError):
                    rng = ""
                line = f"{label}: {day_cond}"
                if rng:
                    line += f", {rng}"
                lines.append(line)

    return "\n".join(lines)


async def _fetch_forecast(client: "object", lat: float, lon: float) -> Optional[dict]:
    """Fetch the Open-Meteo 3-day forecast JSON for coordinates, or None.

    Why: Both the default and named-location paths need the same forecast call;
    factoring it keeps the timeout/error handling in one place.
    What: Builds the forecast URL, GETs it, and returns the parsed JSON dict.
    Returns None on HTTP 4xx/5xx, transport/timeout error, or non-dict JSON
    (strict fall-through).
    Test: Monkeypatch the client to return status 500 -> None; to return a dict
    payload -> that dict.
    """
    forecast_url = (
        f"{_FORECAST_URL}?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,weathercode,windspeed_10m"
        "&daily=weathercode,temperature_2m_max,temperature_2m_min"
        "&temperature_unit=fahrenheit&windspeed_unit=mph"
        "&forecast_days=3&timezone=auto"
    )
    try:
        resp = await client.get(forecast_url)  # type: ignore[attr-defined]
        if resp.status_code >= 400:
            return None
        data = resp.json()
    except Exception:  # noqa: BLE001 — timeout, transport, JSON: all fall through
        return None
    return data if isinstance(data, dict) else None


async def _named_location_weather(client: "object", location: str) -> Optional[str]:
    """Geocode ``location`` then fetch+render its forecast, or None.

    Why: The named path makes TWO HTTP calls (geocode + forecast); isolating it
    lets the caller wrap the pair in a single hard wall-clock ceiling so the
    sub-second promise isn't broken by two stacked per-call timeouts.
    What: Cleans the location, geocodes it, and renders the forecast.  Returns
    None on empty/failed geocoding or any forecast failure.
    Test: Monkeypatch geocode -> empty results -> None; geocode+forecast OK ->
    rendered string.
    """
    city = _clean_location_for_geocode(location)
    geo = await _geocode(client, city)
    if geo is None:
        # Unknown place — defer to the agent, which may know better.
        return None
    lat, lon, name = geo
    data = await _fetch_forecast(client, lat, lon)
    if data is None:
        return None
    return _render_weather(name, data)


async def _weather_handler(text: str) -> Optional[str]:
    """Answer a weather question directly from Open-Meteo, or None to fall through.

    Why: This is the latency win — a deterministic HTTP answer instead of a
    multi-second agent loop.  Latency is the whole point, so HTTP is bounded per
    call (connect 1s / read 1.5s) AND the two-call named-location branch is wrapped
    in a hard 3.5s wall-clock ceiling; the default Woodstock path is a single call.
    What: Resolves the location (default Woodstock if none named, else geocode),
    fetches a 3-day forecast, and renders a terse reply.  Returns None on ANY
    failure: no/empty geocoding, httpx timeout, HTTP 4xx/5xx, JSON/parse error,
    missing ``current.temperature_2m``, or the named-branch ceiling being hit.
    Test: Monkeypatch httpx to (a) timeout -> None, (b) HTTP 500 -> None,
    (c) empty geocoding -> None, (d) canned success -> terse °F string with a
    3-day forecast, (e) a slow geocode -> None within the hard ceiling.
    """
    import httpx  # local import keeps the module importable without httpx for matcher-only tests

    location = _extract_location(text)
    timeout = httpx.Timeout(
        connect=_HTTP_TIMEOUT_CONNECT,
        read=_HTTP_TIMEOUT_READ,
        write=_HTTP_TIMEOUT_READ,
        pool=_HTTP_TIMEOUT_CONNECT,
    )

    async with httpx.AsyncClient(timeout=timeout) as client:
        if location is None:
            # Default home: single forecast call, no geocoding, no extra ceiling.
            data = await _fetch_forecast(client, _DEFAULT_LAT, _DEFAULT_LON)
            if data is None:
                return None
            return _render_weather(_DEFAULT_NAME, data)

        # Named location: two stacked HTTP calls — enforce a hard total ceiling so
        # we never blow the sub-second budget into multi-second territory.
        try:
            return await asyncio.wait_for(
                _named_location_weather(client, location),
                timeout=_NAMED_BRANCH_CEILING_S,
            )
        except Exception:  # noqa: BLE001 — TimeoutError/transport: fall through to agent
            # asyncio.TimeoutError (the ceiling) is an Exception subclass; a
            # bare CancelledError (BaseException) intentionally still propagates.
            return None


# Register the weather intent at module load so importers get it for free.
register_intent(_weather_matcher, _weather_handler)


# Time/date intent — answers "what time is it?" / "what day is it?" in <1ms.
try:
    import intent_time  # noqa: F401
except ImportError:
    pass
