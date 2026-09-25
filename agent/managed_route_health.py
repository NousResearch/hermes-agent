"""Receipted route health; never infer successful access from a catalogue."""
import json
import time
import uuid
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


def record_response_identity(observation, response):
    """Keep provider evidence separate from adapters' requested-model defaults."""
    reported = getattr(response, "_hermes_reported_model", getattr(response, "model", None))
    if isinstance(reported, str):
        observation["reported_model"] = reported
    return response


@contextmanager
def observe_request(home, receipt_id):
    if not receipt_id:
        yield {}
        return
    from agent.error_classifier import classify_api_error

    receipt = get_receipt(home, receipt_id)
    if receipt is None:
        raise RoutingBlocked("stale_or_revoked_decision", "health receipt is missing")
    route = receipt["selected"]
    payload: dict = dict(target_profile=receipt["requirements"].get("target_profile"),
                   route_revision=route["route_revision"], endpoint=route["endpoint"],
                   request_id=uuid.uuid4().hex)
    # Persist before contact: a crash or failed terminal append leaves a pending request.
    append_outcome(home, receipt_id, "routing_request_started", {"request_id": payload["request_id"]})
    try:
        yield payload
    except Exception as exc:
        reason = classify_api_error(exc, provider=route["provider"], model=route["model"]).reason.value
        status = {"auth": "missing_auth", "auth_permanent": "missing_auth",
                  "model_not_found": "denied_model", "billing": "quota", "rate_limit": "quota",
                  "upstream_rate_limit": "quota", "overloaded": "outage",
                  "server_error": "outage", "timeout": "outage"}.get(reason, "unknown")
        replay_safe = not payload.get("output_observed") and reason in {
            "auth", "auth_permanent", "model_not_found", "billing", "rate_limit",
            "upstream_rate_limit", "overloaded", "server_error",
        }
        payload.update(
            status=status, observed_at=int(time.time()), retry_after=0,
            replay_safe=replay_safe,
        )
        headers = getattr(getattr(exc, "response", None), "headers", {}) or {}
        retry = headers.get("retry-after")
        if isinstance(retry, str) and retry.isdecimal():
            payload["retry_after"] = payload["observed_at"] + int(retry)
        append_outcome(home, receipt_id, "routing_health", payload)
        raise
    else:
        # A successful model response can already have driven tool/external
        # effects before a worker later crashes, so replaying the whole worker
        # attempt is not automatically safe.
        payload.update(
            status="healthy", observed_at=int(time.time()), retry_after=0,
            replay_safe=False,
        )
        reported = payload.setdefault("reported_model", None)
        payload["identity_status"] = (
            "missing" if not reported else "matching" if reported == route["model"] else "changed"
        )
        append_outcome(home, receipt_id, "routing_health", payload)


def observe_stream(home, receipt_id, stream):
    """Yield a lazy provider stream while attributing its terminal outcome.

    Returning a stream object only proves that request construction succeeded;
    transport errors commonly surface later from ``next()``.  Keep the health
    observation open until the consumer exhausts the stream so a partial stream
    is never recorded as healthy before its actual outcome is known.
    """
    with observe_request(home, receipt_id) as observation:
        for chunk in stream:
            observation["output_observed"] = True
            record_response_identity(observation, chunk)
            yield chunk
