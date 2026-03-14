"""Radiooooo API client.

Python port of the Acephale Radio TypeScript client (which itself ports the
radio5 Ruby gem).  95% of functionality requires no authentication.

API base: https://radiooooo.com
Asset CDN: https://asset.radiooooo.com
"""

import logging
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# -- Types -----------------------------------------------------------------

Decade = int  # 1900, 1910, ..., 2020
Mood = str  # "slow", "fast", "weird"

ALL_DECADES: List[Decade] = [
    1900, 1910, 1920, 1930, 1940, 1950,
    1960, 1970, 1980, 1990, 2000, 2010, 2020,
]

ALL_MOODS: List[Mood] = ["slow", "fast", "weird"]

# Mood weighting: weird-heavy by default
DEFAULT_MOOD_WEIGHTS: Dict[Mood, float] = {
    "weird": 0.55,
    "slow": 0.30,
    "fast": 0.15,
}

# Acephale-style decade weighting: bell curve around the golden era
DEFAULT_DECADE_WEIGHTS: Dict[Decade, float] = {
    1900: 0.02, 1910: 0.03, 1920: 0.05, 1930: 0.08, 1940: 0.12,
    1950: 0.15, 1960: 0.20, 1970: 0.25, 1980: 0.20, 1990: 0.15,
    2000: 0.12, 2010: 0.08, 2020: 0.05,
}

# Country weighting: boost certain countries during discovery.
# Countries not listed get a base weight of 1.0.
# High weights needed because there are 50-80 countries per decade.
DEFAULT_COUNTRY_WEIGHTS: Dict[str, float] = {
    "GBR": 12.0,  # UK
    "JPN": 12.0,  # Japan
    "FRA": 12.0,  # France
    "USA": 12.0,  # USA
}

BASE_URL = "https://radiooooo.com"


@dataclass
class Track:
    """A single track from the Radiooooo API."""
    id: str
    uuid: str = ""
    title: str = "Unknown"
    artist: str = "Unknown"
    album: str = ""
    year: str = ""
    country: str = ""
    decade: int = 0
    mood: str = ""
    label: str = ""
    length: int = 0
    audio_url: str = ""
    audio_url_ogg: str = ""
    cover_url: str = ""

    @property
    def display(self) -> str:
        parts = [self.artist, self.title]
        if self.decade:
            parts.append(f"{self.decade}s")
        if self.country:
            parts.append(self.country)
        return " - ".join(p for p in parts if p and p != "Unknown")


@dataclass
class CountryMoods:
    country: str
    moods: List[Mood] = field(default_factory=list)


@dataclass
class Island:
    id: str
    name: str = ""
    description: str = ""


# -- Helpers ---------------------------------------------------------------

def _strip_time_limit(url: str) -> str:
    return re.sub(r"#t=\d*,\d+", "", url)


def _parse_track(data: dict) -> Track:
    links = data.get("links") or {}
    cover = data.get("image") or data.get("cover") or ""
    if isinstance(cover, dict):
        cover = cover.get("full", "")

    return Track(
        id=str(data.get("_id") or data.get("id") or ""),
        uuid=str(data.get("uuid") or ""),
        title=str(data.get("title") or "Unknown"),
        artist=str(data.get("artist") or "Unknown"),
        album=str(data.get("album") or ""),
        year=str(data.get("year") or ""),
        country=str(data.get("country") or ""),
        decade=int(data.get("decade") or 0),
        mood=str(data.get("mood") or "slow").lower(),
        label=str(data.get("label") or ""),
        length=int(data.get("length") or 0),
        audio_url=_strip_time_limit(str(links.get("mpeg") or links.get("mp3") or "")),
        audio_url_ogg=_strip_time_limit(str(links.get("ogg") or "")),
        cover_url=str(cover),
    )


# -- Client ----------------------------------------------------------------

class RadioooooClient:
    """Async client for the Radiooooo API."""

    def __init__(self, timeout: float = 15.0):
        self._client = httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=timeout,
            headers={"Accept": "application/json"},
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def random_track(
        self,
        decades: Optional[List[Decade]] = None,
        country: Optional[str] = None,
        moods: Optional[List[Mood]] = None,
    ) -> Optional[Track]:
        """Fetch a random track matching the given criteria.

        Args:
            decades: List of decades to draw from (e.g. [1960, 1970]).
            country: ISO country code (e.g. "FRA", "JPN").
            moods: List of moods ("slow", "fast", "weird").
        """
        body = {
            "mode": "explore",
            "isocodes": [country] if country else [],
            "decades": decades or [1970],
            "moods": [m.upper() for m in (moods or ALL_MOODS)],
        }
        resp = await self._client.post("/play", json=body)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return _parse_track(resp.json())

    async def get_track(self, track_id: str) -> Optional[Track]:
        """Fetch a specific track by ID."""
        resp = await self._client.get(f"/track/play/{track_id}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return _parse_track(resp.json())

    async def island_track(
        self,
        island_id: str,
        moods: Optional[List[Mood]] = None,
    ) -> Optional[Track]:
        """Fetch a random track from a themed island/playlist."""
        body = {
            "mode": "islands",
            "island": island_id,
            "moods": [m.upper() for m in (moods or ALL_MOODS)],
        }
        resp = await self._client.post("/play", json=body)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return _parse_track(resp.json())

    async def countries_for_decade(self, decade: Decade) -> List[CountryMoods]:
        """Return countries with content for a given decade, grouped by mood."""
        resp = await self._client.get("/country/mood", params={"decade": decade})
        resp.raise_for_status()
        data = resp.json()

        # API returns {"Slow": [...], "Fast": [...], "Weird": [...]}
        by_mood: Dict[str, List[str]] = {}
        for key, countries in data.items():
            by_mood[key.lower()] = countries

        country_map: Dict[str, List[str]] = {}
        for mood in ALL_MOODS:
            for c in by_mood.get(mood, []):
                country_map.setdefault(c, []).append(mood)

        return [CountryMoods(country=c, moods=m) for c, m in country_map.items()]

    async def get_islands(self) -> List[Island]:
        """Return all themed islands/playlists."""
        resp = await self._client.get("/islands")
        resp.raise_for_status()
        return [
            Island(
                id=str(item.get("_id") or item.get("id") or ""),
                name=str(item.get("name") or ""),
                description=str(item.get("description") or ""),
            )
            for item in resp.json()
        ]

    # -- High-level helpers ------------------------------------------------

    async def dig(
        self,
        decades: Optional[List[Decade]] = None,
        moods: Optional[List[Mood]] = None,
        country: Optional[str] = None,
        weighted: bool = True,
        mood_weights: Optional[Dict[str, float]] = None,
        country_weights: Optional[Dict[str, float]] = None,
        decade_weights: Optional[Dict[int, float]] = None,
    ) -> Optional[Track]:
        """Crate-dig: pick a decade (weighted), discover a country, fetch a track.

        This mirrors Acephale Radio's Crate Digger logic.  All weight dicts
        are configurable -- pass overrides or they fall back to the module
        defaults (weird-heavy moods, bell-curve decades, GBR/JPN/FRA/USA
        boosted countries).
        """
        _dw = decade_weights or DEFAULT_DECADE_WEIGHTS
        _mw = mood_weights or DEFAULT_MOOD_WEIGHTS
        _cw = country_weights or DEFAULT_COUNTRY_WEIGHTS

        # Pick decade
        if decades and len(decades) == 1:
            decade = decades[0]
        elif decades:
            if weighted:
                weights = [_dw.get(d, 0.1) for d in decades]
                decade = random.choices(decades, weights=weights, k=1)[0]
            else:
                decade = random.choice(decades)
        else:
            pool = list(_dw.keys())
            if weighted:
                weights = list(_dw.values())
                decade = random.choices(pool, weights=weights, k=1)[0]
            else:
                decade = random.choice(pool)

        # Pick mood -- weighted toward weird by default.
        # The API picks randomly from whatever moods we send, so we must
        # send only one mood at a time and fall back if it yields nothing.
        if moods and len(moods) == 1:
            mood_order = moods
        elif moods:
            weights = [_mw.get(m, 0.2) for m in moods]
            picked = random.choices(moods, weights=weights, k=1)[0]
            mood_order = [picked] + [m for m in moods if m != picked]
        else:
            pool = list(_mw.keys())
            weights = list(_mw.values())
            picked = random.choices(pool, weights=weights, k=1)[0]
            mood_order = [picked] + [m for m in pool if m != picked]

        # Discover country if not specified -- weighted toward boosted countries
        if not country:
            try:
                available = await self.countries_for_decade(decade)
                # Filter to countries that have the primary mood
                matching = [
                    cm for cm in available
                    if mood_order[0] in cm.moods
                ]
                if not matching:
                    # Broaden: any mood in our list
                    matching = [
                        cm for cm in available
                        if any(m in cm.moods for m in mood_order)
                    ]
                if matching:
                    weights = [_cw.get(cm.country, 1.0) for cm in matching]
                    cm = random.choices(matching, weights=weights, k=1)[0]
                    country = cm.country
            except Exception:
                logger.debug("Country discovery failed for decade %d, trying without", decade)

        # Try each mood in order until we get a track with an audio URL.
        # Some tracks come back without URLs -- retry up to 3 times per mood.
        track = None
        for mood in mood_order:
            for _attempt in range(3):
                candidate = await self.random_track(
                    decades=[decade],
                    country=country,
                    moods=[mood],
                )
                if candidate and (candidate.audio_url or candidate.audio_url_ogg):
                    track = candidate
                    break
                elif candidate:
                    logger.debug("Track %s has no audio URL, retrying", candidate.id)
            if track:
                break
        if track:
            logger.info("Dug: %s (%s, %ds, %s)", track.display, track.mood, track.decade, track.country)
        return track
