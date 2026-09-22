"""Receipted route health; never infer successful access from a catalogue."""
import json
import time
from contextlib import contextmanager

from agent.model_selection_store import _connect, append_outcome, get_receipt
from agent.model_selection_types import RoutingBlocked
from hermes_cli.sqlite_util import transaction


def load_availability(home, policy_id: str, target_profile: str) -> dict:
    with transaction(_connect(home)) as conn:
        rows = conn.execute(
            "SELECT r.decision_json, o.payload_json FROM routing_outcomes o "
            "JOIN routing_receipts r ON r.id=o.receipt_id "
            "WHERE r.policy_id=? AND o.kind='routing_health' ORDER BY o.id DESC",
            (policy_id,),
        ).fetchall()
    result = {}
    for row in rows:
        payload = json.loads(row["payload_json"])
        if payload.get("target_profile") == target_profile:
            route = json.loads(row["decision_json"])["selected"]
            result.setdefault(route["route_id"], payload)
    return result


@contextmanager
def observe_request(home, receipt_id):
    if not receipt_id:
        yield
        return
    from agent.error_classifier import classify_api_error

    receipt = get_receipt(home, receipt_id)
    if receipt is None:
        raise RoutingBlocked("stale_or_revoked_decision", "health receipt is missing")
    route = receipt["selected"]
    payload = dict(target_profile=receipt["requirements"].get("target_profile"),
                   route_revision=route["route_revision"], endpoint=route["endpoint"])
    try:
        yield
    except Exception as exc:
        reason = classify_api_error(exc, provider=route["provider"], model=route["model"]).reason.value
        status = {"auth": "missing_auth", "auth_permanent": "missing_auth",
                  "model_not_found": "denied_model", "billing": "quota", "rate_limit": "quota",
                  "upstream_rate_limit": "quota", "overloaded": "outage",
                  "server_error": "outage", "timeout": "outage"}.get(reason, "unknown")
        payload.update(status=status, observed_at=int(time.time()), retry_after=0)
        headers = getattr(getattr(exc, "response", None), "headers", {}) or {}
        retry = headers.get("retry-after")
        if isinstance(retry, str) and retry.isdecimal():
            payload["retry_after"] = payload["observed_at"] + int(retry)
        append_outcome(home, receipt_id, "routing_health", payload)
        raise
    else:
        payload.update(status="healthy", observed_at=int(time.time()), retry_after=0)
        append_outcome(home, receipt_id, "routing_health", payload)
