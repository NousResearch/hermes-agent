"""Radio Browser API client.

Open directory of 45,000+ radio stations.  JSON API, no auth.
API: https://all.api.radio-browser.info
"""

import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# Multiple mirrors available; pick one at random for load distribution
API_HOSTS = [
    "https://de1.api.radio-browser.info",
    "https://nl1.api.radio-browser.info",
    "https://at1.api.radio-browser.info",
]


@dataclass
class Station:
    uuid: str
    name: str = ""
    url: str = ""
    url_resolved: str = ""
    country: str = ""
    country_code: str = ""
    tags: str = ""
    codec: str = ""
    bitrate: int = 0
    votes: int = 0
    click_count: int = 0
    favicon: str = ""
    homepage: str = ""

    @property
    def stream_url(self) -> str:
        return self.url_resolved or self.url

    @property
    def display(self) -> str:
        parts = [self.name]
        if self.country:
            parts.append(f"[{self.country}]")
        if self.tags:
            # Show first 3 tags
            tag_list = [t.strip() for t in self.tags.split(",")][:3]
            parts.append(", ".join(tag_list))
        return " ".join(parts)


def _parse_station(data: dict) -> Station:
    return Station(
        uuid=data.get("stationuuid", ""),
        name=data.get("name", ""),
        url=data.get("url", ""),
        url_resolved=data.get("url_resolved", ""),
        country=data.get("country", ""),
        country_code=data.get("countrycode", ""),
        tags=data.get("tags", ""),
        codec=data.get("codec", ""),
        bitrate=int(data.get("bitrate", 0)),
        votes=int(data.get("votes", 0)),
        click_count=int(data.get("clickcount", 0)),
        favicon=data.get("favicon", ""),
        homepage=data.get("homepage", ""),
    )


class RadioBrowserClient:
    """Async client for the Radio Browser API."""

    def __init__(self, timeout: float = 15.0):
        self._base_url = random.choice(API_HOSTS)
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout,
            headers={
                "Accept": "application/json",
                "User-Agent": "HermesRadio/1.0",
            },
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def search(
        self,
        name: Optional[str] = None,
        tag: Optional[str] = None,
        country: Optional[str] = None,
        country_code: Optional[str] = None,
        codec: Optional[str] = None,
        limit: int = 25,
        order: str = "clickcount",
        reverse: bool = True,
        hide_broken: bool = True,
    ) -> List[Station]:
        """Search for stations by various criteria."""
        params: Dict[str, Any] = {
            "limit": limit,
            "order": order,
            "reverse": str(reverse).lower(),
            "hidebroken": str(hide_broken).lower(),
        }
        if name:
            params["name"] = name
        if tag:
            params["tag"] = tag
        if country:
            params["country"] = country
        if country_code:
            params["countrycode"] = country_code
        if codec:
            params["codec"] = codec

        resp = await self._client.get("/json/stations/search", params=params)
        resp.raise_for_status()
        return [_parse_station(s) for s in resp.json()]

    async def top_clicked(self, limit: int = 25) -> List[Station]:
        resp = await self._client.get(f"/json/stations/topclick/{limit}")
        resp.raise_for_status()
        return [_parse_station(s) for s in resp.json()]

    async def top_voted(self, limit: int = 25) -> List[Station]:
        resp = await self._client.get(f"/json/stations/topvote/{limit}")
        resp.raise_for_status()
        return [_parse_station(s) for s in resp.json()]

    async def by_tag(self, tag: str, limit: int = 25) -> List[Station]:
        """Search stations by tag (genre)."""
        return await self.search(tag=tag, limit=limit)

    async def resolve_url(self, uuid: str) -> Optional[str]:
        """Resolve a station UUID to its direct stream URL."""
        resp = await self._client.get(f"/json/url/{uuid}")
        resp.raise_for_status()
        data = resp.json()
        return data.get("url") if data.get("ok") else None

    async def tags(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Return popular tags with station counts."""
        resp = await self._client.get("/json/tags", params={
            "limit": limit, "order": "stationcount", "reverse": "true",
        })
        resp.raise_for_status()
        return resp.json()
