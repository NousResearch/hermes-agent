"""SomaFM client.

~40 curated underground/indie radio channels.  Zero auth.
API: https://api.somafm.com/channels.json
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import httpx

logger = logging.getLogger(__name__)

API_URL = "https://api.somafm.com/channels.json"

# Channels that fit the pirate radio aesthetic
FEATURED_CHANNELS = [
    "defcon", "darkzone", "sf1033", "vaporwaves", "dronezone",
    "deepspaceone", "secretagent", "lush", "cliqhop", "groovesalad",
    "thistle", "dubstep", "metal", "suburbsofgoa", "thetrip",
]


@dataclass
class SomaChannel:
    id: str
    title: str = ""
    description: str = ""
    genre: str = ""
    dj: str = ""
    listeners: int = 0
    stream_url: str = ""
    image_url: str = ""

    @property
    def display(self) -> str:
        return f"{self.title} [{self.genre}]" if self.genre else self.title


async def get_channels(timeout: float = 10.0) -> List[SomaChannel]:
    """Fetch all SomaFM channels."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(API_URL)
        resp.raise_for_status()

    channels = []
    for ch in resp.json().get("channels", []):
        # Pick the highest quality MP3 stream
        playlists = ch.get("playlists", [])
        stream_url = ""
        for pl in playlists:
            if pl.get("format") == "mp3" and pl.get("quality") == "highest":
                stream_url = pl.get("url", "")
                break
        if not stream_url:
            for pl in playlists:
                if pl.get("format") == "mp3":
                    stream_url = pl.get("url", "")
                    break

        channels.append(SomaChannel(
            id=ch.get("id", ""),
            title=ch.get("title", ""),
            description=ch.get("description", ""),
            genre=ch.get("genre", ""),
            dj=ch.get("dj", ""),
            listeners=int(ch.get("listeners", 0)),
            stream_url=stream_url,
            image_url=ch.get("xlimage") or ch.get("image") or "",
        ))

    return channels


async def get_channel(channel_id: str, timeout: float = 10.0) -> Optional[SomaChannel]:
    """Fetch a single channel by ID."""
    channels = await get_channels(timeout=timeout)
    for ch in channels:
        if ch.id == channel_id:
            return ch
    return None


async def get_featured(timeout: float = 10.0) -> List[SomaChannel]:
    """Return the curated set of pirate-radio-adjacent channels."""
    channels = await get_channels(timeout=timeout)
    featured_set = set(FEATURED_CHANNELS)
    return [ch for ch in channels if ch.id in featured_set]
