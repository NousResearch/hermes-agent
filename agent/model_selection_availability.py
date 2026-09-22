"""Pure, destination- and profile-bound health evidence for route selection."""
from agent.model_selection_types import RoutingBlocked

UNAVAILABLE = frozenset({"missing_auth", "denied_model", "quota", "outage"})


def availability_snapshot(requirements: dict, routes: list, evidence: dict, now: int) -> dict:
    if not isinstance(evidence, dict):
        raise RoutingBlocked("schema_invalid", "availability must be an object")
    snapshot = {}
    for route in routes:
        observed = evidence.get(route["route_id"])
        snapshot[route["route_id"]] = health_for_route(
            route, requirements.get("target_profile"), observed, now,
        )
    return snapshot


def health_for_route(route: dict, target_profile, observation, now: int) -> dict:
    unknown = {"status": "unknown", "age_seconds": None}
    if observation is None:
        return unknown
    if not isinstance(observation, dict):
        raise RoutingBlocked("schema_invalid", "health observation must be an object")
    status = observation.get("status")
    stamp = observation.get("observed_at")
    retry_after = observation.get("retry_after", 0)
    if status not in UNAVAILABLE | {"healthy", "unknown"}:
        raise RoutingBlocked("schema_invalid", "unknown health status")
    if type(stamp) is not int or type(retry_after) is not int:
        raise RoutingBlocked("schema_invalid", "health timestamps must be integers")
    if not target_profile or observation.get("target_profile") != target_profile:
        return unknown
    if (observation.get("route_revision") != route["route_revision"]
            or observation.get("endpoint") != route["endpoint"] or stamp > now):
        return unknown
    age = now - stamp
    deadline = stamp + (300 if status == "healthy" else 60)
    if status in UNAVAILABLE:
        deadline = max(deadline, retry_after)
    if now >= deadline or status == "unknown":
        return unknown
    return {"status": status, "age_seconds": age, "observed_at": stamp,
            "retry_after": deadline}
