"""Radio Garden API client.

Live radio stations worldwide via geographic discovery.
API: https://radio.garden/api  (unofficial, no auth)

The key trick: GET /ara/content/listen/{channelId}/channel.mp3 returns a
302 redirect to the actual Icecast/SHOUTcast stream.  Region restrictions
are client-side only.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://radio.garden/api"


@dataclass
class Place:
    id: str
    title: str = ""
    country: str = ""
    geo: tuple = (0.0, 0.0)  # (lon, lat) -- reversed from typical convention
    station_count: int = 0

    @property
    def display(self) -> str:
        return f"{self.title}, {self.country}" if self.country else self.title


@dataclass
class Station:
    id: str
    title: str = ""
    url: str = ""
    place: str = ""
    country: str = ""
    website: str = ""

    @property
    def stream_url(self) -> str:
        """Return the redirect URL that resolves to the actual stream."""
        return f"https://radio.garden/api/ara/content/listen/{self.id}/channel.mp3"

    @property
    def display(self) -> str:
        parts = [self.title]
        if self.place:
            parts.append(f"[{self.place}]")
        if self.country:
            parts.append(self.country)
        return " ".join(parts)


class RadioGardenClient:
    """Async client for the Radio Garden API."""

    def __init__(self, timeout: float = 15.0):
        self._client = httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=timeout,
            headers={"Accept": "application/json"},
            follow_redirects=False,  # We want to capture the 302
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def search(self, query: str) -> Dict[str, List[Any]]:
        """Search for places, countries, and stations.

        Returns a dict with keys 'places', 'stations', 'countries' (lists).
        """
        resp = await self._client.get("/search", params={"q": query})
        resp.raise_for_status()
        data = resp.json()

        hits = data.get("hits", data.get("data", {}))

        results: Dict[str, List[Any]] = {
            "places": [],
            "stations": [],
        }

        # Parse hits -- structure: hits.hits[]._source.{type, page.{url, title, ...}}
        hit_list = []
        if isinstance(hits, dict):
            hit_list = hits.get("hits", [])
        elif isinstance(hits, list):
            hit_list = hits

        for hit in hit_list:
            source = hit.get("_source", hit)
            page = source.get("page", {})
            hit_type = source.get("type", page.get("type", ""))
            url = page.get("url", "")

            if hit_type == "place":
                # Extract place ID from URL like /visit/tokyo/eR8K4rBb
                place_id = url.rsplit("/", 1)[-1] if "/" in url else source.get("code", "")
                results["places"].append(Place(
                    id=place_id,
                    title=page.get("title", ""),
                    country=page.get("subtitle", ""),
                    station_count=page.get("count", 0),
                ))
            elif hit_type == "channel":
                # Extract channel ID from URL like /listen/station-name/OjFb4M9Q
                channel_id = url.rsplit("/", 1)[-1] if "/" in url else ""
                place_data = page.get("place", {})
                country_data = page.get("country", {})
                results["stations"].append(Station(
                    id=channel_id,
                    title=page.get("title", ""),
                    url=url,
                    place=place_data.get("title", page.get("subtitle", "")),
                    country=country_data.get("title", "") if isinstance(country_data, dict) else str(country_data),
                ))

        return results

    async def get_place(self, place_id: str) -> List[Station]:
        """Get all stations at a place."""
        resp = await self._client.get(f"/ara/content/page/{place_id}")
        resp.raise_for_status()
        data = resp.json()

        content = data.get("data", data)
        stations = []

        if isinstance(content, dict):
            place_title = content.get("title", "")
            place_country = content.get("subtitle", "")
            for section in content.get("content", []):
                for item in section.get("items", []):
                    page = item.get("page", item)
                    url = page.get("url", "")
                    # Extract channel ID from URL like /listen/station-name/OjFb4M9Q
                    channel_id = url.rsplit("/", 1)[-1] if "/" in url else ""
                    if channel_id:
                        country_data = page.get("country", {})
                        stations.append(Station(
                            id=channel_id,
                            title=page.get("title", ""),
                            url=url,
                            place=place_title,
                            country=country_data.get("title", "") if isinstance(country_data, dict) else place_country,
                        ))

        return stations

    async def get_station(self, channel_id: str) -> Optional[Station]:
        """Get details for a single station."""
        resp = await self._client.get(f"/ara/content/channel/{channel_id}")
        resp.raise_for_status()
        data = resp.json().get("data", resp.json())

        if not data:
            return None

        place_data = data.get("place", {})
        return Station(
            id=channel_id,
            title=data.get("title", ""),
            url=data.get("url", ""),
            place=place_data.get("title", ""),
            country=data.get("country", place_data.get("country", "")),
            website=data.get("website", ""),
        )

    async def resolve_stream(self, channel_id: str) -> Optional[str]:
        """Resolve a channel ID to its actual stream URL by following the 302."""
        url = f"/ara/content/listen/{channel_id}/channel.mp3"
        try:
            resp = await self._client.get(url)
            if resp.status_code in (301, 302):
                return resp.headers.get("location")
            # Some stations return 200 with direct stream
            if resp.status_code == 200:
                return f"https://radio.garden/api{url}"
        except Exception:
            pass
        # mpv can follow the redirect itself, so return the garden URL
        return f"https://radio.garden/api{url}"

    async def explore(self, query: str, limit: int = 10) -> List[Station]:
        """Search for a city and return stations there.

        Convenience method: searches for places, picks the top result,
        and returns its stations.
        """
        results = await self.search(query)

        # If we got stations directly, return those
        if results["stations"]:
            return results["stations"][:limit]

        # Otherwise, get stations from the top place
        if results["places"]:
            place = results["places"][0]
            logger.info("Exploring %s (%s)", place.title, place.country)
            stations = await self.get_place(place.id)
            return stations[:limit]

        return []
