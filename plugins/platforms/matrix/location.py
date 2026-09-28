"""RFC 5870 coordinates and text for inbound Matrix location messages."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass


_GEO_URI = re.compile(
    r"geo:(?P<latitude>-?[0-9]{1,2}(?:\.[0-9]+)?),"
    r"(?P<longitude>-?[0-9]{1,3}(?:\.[0-9]+)?)"
    r"(?:,(?P<altitude>-?[0-9]+(?:\.[0-9]+)?))?"
    r"(?P<parameters>(?:;[a-z0-9-]+(?:=(?:[a-z0-9\-_.!~*'()\[\]:&+$]|%[0-9a-f]{2})+)?)*)",
    re.IGNORECASE | re.ASCII,
)
_UNCERTAINTY = re.compile(r"[0-9]+(?:\.[0-9]+)?")


@dataclass(frozen=True)
class GeoPoint:
    latitude: float
    longitude: float
    altitude: float | None = None
    uncertainty: float | None = None

    @classmethod
    def from_uri(cls, uri: object) -> GeoPoint | None:
        if not isinstance(uri, str):
            return None
        match = _GEO_URI.fullmatch(uri)
        if match is None:
            return None

        latitude = float(match["latitude"])
        longitude = float(match["longitude"])
        altitude = float(match["altitude"]) if match["altitude"] is not None else None
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            return None
        if altitude is not None and not math.isfinite(altitude):
            return None

        crs_seen = extensions_seen = False
        uncertainty = None
        for parameter in match["parameters"].split(";")[1:]:
            key, _, value = parameter.partition("=")
            key = key.lower()
            if key == "crs":
                if (
                    crs_seen
                    or uncertainty is not None
                    or extensions_seen
                    or value.lower() != "wgs84"
                ):
                    return None
                crs_seen = True
            elif key == "u":
                if (
                    uncertainty is not None
                    or extensions_seen
                    or _UNCERTAINTY.fullmatch(value) is None
                ):
                    return None
                uncertainty = float(value)
                if not math.isfinite(uncertainty):
                    return None
            else:
                extensions_seen = True

        return cls(latitude, longitude, altitude, uncertainty)

    def as_text(self) -> str:
        text = f"📍 Location: {self.latitude}, {self.longitude}"
        if self.altitude is not None:
            text += f"; altitude: {self.altitude} m"
        if self.uncertainty is not None:
            text += f"; uncertainty: {self.uncertainty} m"
        return text


def format_location_content(content: Mapping[str, object]) -> str | None:
    location = content.get("org.matrix.msc3488.location")
    if not isinstance(location, dict):
        location = {}
    point = GeoPoint.from_uri(location.get("uri", content.get("geo_uri")))
    if point is None:
        return None

    description = location.get("description")
    text = point.as_text()
    if isinstance(description, str) and description.strip():
        text += f" ({description.strip()})"
        return text

    body = content.get("body")
    if isinstance(body, str) and body.strip() not in ("", "Location", "Posizione"):
        text += f" ({body.strip()})"
    return text
