"""Whoop API endpoint registry.

Maps endpoint names to API paths, required scopes, and query parameter specs.
Add new endpoints here to extend data collection without modifying client code.

Note: Whoop API v2 paths use the /developer/v2/ prefix, not /v2/ directly.
Base URL: https://api.prod.whoop.com/developer/v2/
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Endpoint:
    name: str
    path: str
    scopes: list[str]
    requires_pagination: bool
    description: str


ENDPOINTS: dict[str, Endpoint] = {
    "cycle": Endpoint(
        name="cycle",
        path="/developer/v2/cycle",
        scopes=["read:cycles"],
        requires_pagination=True,
        description="Strain, heart rate, kilojoules per cycle",
    ),
    "recovery": Endpoint(
        name="recovery",
        path="/developer/v2/recovery",
        scopes=["read:recovery"],
        requires_pagination=True,
        description="Recovery %, HRV, resting heart rate",
    ),
    "sleep": Endpoint(
        name="sleep",
        path="/developer/v2/activity/sleep",
        scopes=["read:sleep"],
        requires_pagination=True,
        description="Sleep stages, efficiency, sleep debt",
    ),
    "workout": Endpoint(
        name="workout",
        path="/developer/v2/activity/workout",
        scopes=["read:workout"],
        requires_pagination=True,
        description="Strain by activity type, duration",
    ),
    "body": Endpoint(
        name="body",
        path="/developer/v2/user/measurement/body",
        scopes=["read:body_measurement"],
        requires_pagination=False,
        description="Weight in kg, height in meters",
    ),
    "profile": Endpoint(
        name="profile",
        path="/developer/v2/user/profile/basic",
        scopes=["read:profile"],
        requires_pagination=False,
        description="User ID, name, email",
    ),
}


def get_endpoint(name: str) -> Endpoint:
    """Look up an endpoint by name. Raises KeyError if not found."""
    return ENDPOINTS[name]


def all_endpoints() -> list[Endpoint]:
    """Return all registered endpoints."""
    return list(ENDPOINTS.values())