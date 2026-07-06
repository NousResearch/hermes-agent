"""Webhook parsing and prompt construction for Linear Agent Sessions."""

from __future__ import annotations

import hashlib
import hmac
import json
import base64
import binascii
import re
import time
from dataclasses import dataclass
from typing import Any, Mapping


SUPPORTED_ACTIONS = frozenset({"created", "prompted", "update"})


class LinearWebhookError(ValueError):
    """Raised for invalid Linear webhook payloads."""


@dataclass(frozen=True)
class LinearWebhookContext:
    action: str
    delivery_id: str
    agent_session_id: str
    workspace_id: str = ""
    issue_id: str = ""
    issue_identifier: str = ""
    issue_title: str = ""
    comment_id: str = ""
    actor_user_id: str = ""
    actor_user_name: str = ""
    team_id: str = ""
    prompt_context: str = ""
    guidance: str = ""
    user_request: str = ""
    signal: str = ""
    # The comment the agent was mentioned in when the session was spawned from
    # a DIFFERENT thread (Linear re-anchors the session to a copied root, so
    # this points back at the original human thread). Empty for fresh-root
    # mentions. Its own thread is a normal thread, so replies there render.
    source_comment_id: str = ""
    # Delegate as serialized in the webhook payload itself (issue-update
    # webhooks carry the issue model). Only trustworthy when
    # issue_delegate_known is True — an absent field means "not serialized",
    # not "no delegate".
    issue_delegate_id: str = ""
    issue_delegate_known: bool = False

    @property
    def thread_id(self) -> str:
        return self.issue_id or self.comment_id

    @property
    def chat_name(self) -> str:
        if self.issue_identifier and self.issue_title:
            return f"{self.issue_identifier}: {self.issue_title}"
        return self.issue_identifier or self.issue_title or self.agent_session_id

    def metadata(self) -> dict[str, str]:
        return {
            "platform": "linear_agent",
            "linear_agent_session_id": self.agent_session_id,
            "linear_workspace_id": self.workspace_id,
            "linear_issue_id": self.issue_id,
            "linear_comment_id": self.comment_id,
            "linear_actor_user_id": self.actor_user_id,
            "linear_team_id": self.team_id,
            "linear_signal": self.signal,
            "linear_source_comment_id": self.source_comment_id,
        }


def _get_path(data: Any, *path: str) -> Any:
    cur = data
    for key in path:
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(key)
    return cur


def _string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _guidance_text(value: Any) -> str:
    """Guidance arrives as a list of ``{body, origin}`` rules; render bodies."""
    if isinstance(value, (list, tuple)):
        bodies = (
            _string(item.get("body")) if isinstance(item, Mapping) else _string(item)
            for item in value
        )
        return "\n\n".join(body for body in bodies if body)
    if isinstance(value, Mapping):
        return _string(value.get("body"))
    return _string(value)


# Header aliases used both for lookup and for safe-to-log diagnostics.
SIGNATURE_HEADERS = (
    "Linear-Signature",
    "Linear-Webhook-Signature",
    "X-Linear-Signature",
    "X-Linear-Webhook-Signature",
    "X-Webhook-Signature",
    "Webhook-Signature",
    "Svix-Signature",
    "X-Hub-Signature-256",
    "X-Signature",
)
TIMESTAMP_HEADERS = (
    "Linear-Timestamp",
    "X-Linear-Timestamp",
    "Webhook-Timestamp",
    "X-Webhook-Timestamp",
    "Svix-Timestamp",
)
WEBHOOK_ID_HEADERS = (
    "Linear-Delivery",  # Linear's actual delivery-id header
    "Webhook-Id",
    "X-Webhook-Id",
    "Svix-Id",
    "Linear-Delivery-Id",
    "X-Linear-Delivery",
    "X-Request-ID",
)


def _header(headers: Mapping[str, str], *names: str) -> str:
    lowered = {str(k).lower(): str(v) for k, v in headers.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value:
            return value.strip()
    return ""


def parse_json_body(raw_body: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - normalize webhook boundary errors
        raise LinearWebhookError("Invalid JSON payload") from exc
    if not isinstance(payload, dict):
        raise LinearWebhookError("Webhook payload must be a JSON object")
    return payload


def verify_linear_signature(
    headers: Mapping[str, str],
    body: bytes,
    secret: str,
    *,
    tolerance_seconds: int = 60,
) -> bool:
    """Validate a Linear-style HMAC-SHA256 webhook signature.

    Fails closed: with no secret there is nothing to verify against, so the
    signature is treated as invalid. Callers that intentionally accept
    unsigned webhooks must gate on their own explicit opt-in flag instead of
    passing an empty secret (see LinearAgentAdapter.handle_webhook).
    """
    if not secret:
        return False

    signature = _header(headers, *SIGNATURE_HEADERS)
    if not signature:
        return False

    timestamp = _header(headers, *TIMESTAMP_HEADERS)
    if timestamp:
        try:
            ts = int(timestamp)
        except ValueError:
            return False
        # Linear sends Linear-Timestamp/webhookTimestamp as Unix milliseconds
        # (for example 1676056940508), while Standard Webhooks/Svix style
        # timestamps are Unix seconds.  Normalize only for freshness checks;
        # keep the original header string for providers that include it in the
        # signed payload.
        ts_seconds = ts / 1000 if ts > 10_000_000_000 else ts
        if abs(time.time() - ts_seconds) > tolerance_seconds:
            return False

    webhook_id = _header(headers, *WEBHOOK_ID_HEADERS)

    keys = [secret.encode("utf-8")]
    if secret.startswith("whsec_"):
        try:
            keys.insert(0, base64.b64decode(secret.removeprefix("whsec_"), validate=True))
        except (binascii.Error, ValueError):
            return False

    def _digests(payload: bytes) -> set[str]:
        values: set[str] = set()
        for key in keys:
            digest_bytes = hmac.new(key, payload, hashlib.sha256).digest()
            digest_hex = digest_bytes.hex()
            digest_b64 = base64.b64encode(digest_bytes).decode("ascii")
            values.update(
                {
                    digest_hex,
                    f"sha256={digest_hex}",
                    f"sha256:{digest_hex}",
                    f"sha256,{digest_hex}",
                    digest_b64,
                    f"v1,{digest_b64}",
                    f"v1={digest_b64}",
                    f"v1={digest_hex}",
                }
            )
        return values

    signed_payloads = [body]
    if timestamp:
        signed_payloads.append(timestamp.encode("utf-8") + b"." + body)
    if webhook_id and timestamp:
        signed_payloads.append(
            webhook_id.encode("utf-8") + b"." + timestamp.encode("utf-8") + b"." + body
        )

    candidates: set[str] = set()
    for payload in signed_payloads:
        candidates.update(_digests(payload))

    # Compare the raw header plus each delimiter-separated piece, deduped —
    # multi-signature headers use spaces, commas, or semicolons.
    supplied = signature.strip()
    pieces = (part.strip() for part in re.split(r"[;,\s]+", supplied))
    parts = list(dict.fromkeys([supplied, *(p for p in pieces if p)]))
    return any(
        hmac.compare_digest(part, candidate)
        for part in parts
        for candidate in candidates
    )


def is_stale_body_timestamp(
    payload: Mapping[str, Any],
    headers: Mapping[str, str],
    *,
    tolerance_seconds: int = 60,
) -> bool:
    """Replay guard for deliveries whose timestamp rides only in the body.

    Header freshness is enforced during signature verification; without a
    timestamp header, the signed ``webhookTimestamp`` body field is the only
    thing stopping a captured delivery from replaying after the dedup TTL.
    """
    if _header(headers, *TIMESTAMP_HEADERS):
        return False
    ts = payload.get("webhookTimestamp")
    if not ts:
        return False
    try:
        ts_value = float(ts)
    except (TypeError, ValueError):
        return False
    ts_seconds = ts_value / 1000 if ts_value > 10_000_000_000 else ts_value
    return abs(time.time() - ts_seconds) > tolerance_seconds


def describe_signature_headers(headers: Mapping[str, str]) -> dict[str, Any]:
    """Return safe-to-log metadata about signature-related headers."""
    interesting = {
        name.lower()
        for name in (*SIGNATURE_HEADERS, *TIMESTAMP_HEADERS, *WEBHOOK_ID_HEADERS)
    }
    details: dict[str, Any] = {}
    for key, value in headers.items():
        lowered = str(key).lower()
        if lowered not in interesting:
            continue
        raw = str(value or "")
        if "signature" in lowered:
            details[lowered] = {
                "length": len(raw),
                "prefix": raw[:12],
                "has_comma": "," in raw,
                "has_equals": "=" in raw,
                "has_colon": ":" in raw,
            }
        else:
            details[lowered] = "<present>"
    return details


def extract_delivery_id(payload: Mapping[str, Any], headers: Mapping[str, str]) -> str:
    header_value = _header(headers, *WEBHOOK_ID_HEADERS)
    if header_value:
        return header_value

    for path in (
        ("webhookEvent", "id"),
        ("event", "id"),
        ("id",),
        ("agentActivity", "id"),
    ):
        value = _string(_get_path(payload, *path))
        if value:
            return value

    # Data-webhook envelope: webhookId identifies the webhook CONFIG (same on
    # every delivery), so it only deduplicates paired with the per-delivery
    # timestamp.
    webhook_id = _string(payload.get("webhookId"))
    webhook_ts = _string(payload.get("webhookTimestamp"))
    if webhook_id and webhook_ts:
        return f"{webhook_id}:{webhook_ts}"

    action = normalize_action(payload)
    session_id = _string(_get_path(payload, "agentSession", "id"))
    activity_id = _string(_get_path(payload, "agentActivity", "id"))
    entity_id = _string(_get_path(payload, "data", "id"))
    entity_updated = _string(_get_path(payload, "data", "updatedAt"))
    return ":".join(
        part
        for part in (action, session_id, activity_id, entity_id, entity_updated)
        if part
    )


def normalize_action(payload: Mapping[str, Any]) -> str:
    return _string(
        payload.get("action")
        or payload.get("type")
        or _get_path(payload, "event", "action")
    ).lower()


def extract_context(
    payload: Mapping[str, Any],
    headers: Mapping[str, str],
) -> LinearWebhookContext:
    action = normalize_action(payload)
    if not action:
        raise LinearWebhookError("Missing webhook action")

    # Standard Linear webhook (Issue update, etc.)
    if action == "update" and payload.get("type") == "Issue":
        data = payload.get("data") or {}
        actor = payload.get("actor") or {}
        issue_id = _string(data.get("id"))
        identifier = _string(data.get("identifier"))
        title = _string(data.get("title"))
        team = data.get("team") if isinstance(data.get("team"), dict) else {}
        if not team:
            team = payload.get("team") or {}

        # For delegation, we do not have an agentSession. Use a synthetic
        # "update:<issue_id>" session id: the prefix lets the adapter route
        # replies as issue comments (a raw delivery UUID would be
        # indistinguishable from a real agent session id), and keying by
        # issue groups every update of the same issue into one session
        # instead of spawning a fresh session per webhook delivery.
        delivery_id = extract_delivery_id(payload, headers)
        session_id = f"update:{issue_id or delivery_id}"

        # Surface the payload's own delegate so the auto-start path can skip
        # its verification fetch on a definitive negative (delegated to
        # someone else / nobody) without ever trusting a positive unverified.
        delegate_known = "delegateId" in data or "delegate" in data
        delegate_id = _string(data.get("delegateId") or _get_path(data, "delegate", "id"))

        return LinearWebhookContext(
            action=action,
            delivery_id=delivery_id,
            agent_session_id=session_id,
            workspace_id=_string(
                payload.get("workspaceId")
                or payload.get("organizationId")
                or _get_path(payload, "workspace", "id")
            ),
            issue_id=issue_id,
            issue_identifier=identifier,
            issue_title=title,
            comment_id="",
            actor_user_id=_string(actor.get("id") or payload.get("actorUserId")),
            actor_user_name=_string(actor.get("name")),
            team_id=_string(team.get("id") or payload.get("teamId")),
            prompt_context="",
            guidance="",
            user_request="",
            issue_delegate_id=delegate_id,
            issue_delegate_known=delegate_known,
        )

    # Existing Agent Session path
    session = payload.get("agentSession") if isinstance(payload.get("agentSession"), dict) else {}
    issue = session.get("issue") if isinstance(session.get("issue"), dict) else {}
    if not issue and isinstance(payload.get("issue"), dict):
        issue = payload["issue"]
    comment = session.get("comment") if isinstance(session.get("comment"), dict) else {}
    if not comment and isinstance(payload.get("comment"), dict):
        comment = payload["comment"]
    team = issue.get("team") if isinstance(issue.get("team"), dict) else {}
    if not team and isinstance(payload.get("team"), dict):
        team = payload["team"]
    actor = payload.get("actor") if isinstance(payload.get("actor"), dict) else {}
    agent_activity = (
        payload.get("agentActivity")
        if isinstance(payload.get("agentActivity"), dict)
        else {}
    )
    activity_actor = (
        agent_activity.get("actor")
        if isinstance(agent_activity.get("actor"), dict)
        else {}
    )

    agent_session_id = _string(session.get("id") or payload.get("agentSessionId"))
    if not agent_session_id:
        raise LinearWebhookError("Missing agentSession.id")

    # Agent activities serialize their text as `content: {type, body}`
    # (AgentActivityPromptContent in Linear's schema); flat `body` and plain
    # string `content` variants are kept as fallbacks against shape drift.
    activity_content = agent_activity.get("content")
    user_request = _string(
        agent_activity.get("body")
        or _get_path(agent_activity, "content", "body")
        or (activity_content if isinstance(activity_content, str) else "")
        or payload.get("body")
    )
    if action == "created" and not user_request:
        user_request = _string(
            comment.get("body")
            or comment.get("content")
            or payload.get("commentBody")
            or payload.get("prompt")
        )

    # Human→agent signals (e.g. "stop") ride on the prompting agent activity;
    # normalize and fall back to a top-level payload.signal for defensiveness
    # against payload-shape drift.
    signal = _string(agent_activity.get("signal") or payload.get("signal")).lower()

    return LinearWebhookContext(
        action=action,
        delivery_id=extract_delivery_id(payload, headers),
        agent_session_id=agent_session_id,
        workspace_id=_string(
            session.get("workspaceId")
            or payload.get("workspaceId")
            or payload.get("organizationId")
            or _get_path(payload, "workspace", "id")
        ),
        issue_id=_string(issue.get("id")),
        issue_identifier=_string(issue.get("identifier")),
        issue_title=_string(issue.get("title")),
        comment_id=_string(comment.get("id")),
        actor_user_id=_string(
            actor.get("id")
            or activity_actor.get("id")
            or payload.get("actorUserId")
            or _get_path(payload, "user", "id")
        ),
        actor_user_name=_string(
            actor.get("name")
            or activity_actor.get("name")
            or _get_path(payload, "user", "name")
        ),
        team_id=_string(team.get("id") or payload.get("teamId")),
        prompt_context=_string(payload.get("promptContext") or session.get("promptContext")),
        guidance=_guidance_text(payload.get("guidance") or session.get("guidance")),
        user_request=user_request,
        signal=signal,
        source_comment_id=_string(
            payload.get("sourceCommentId")
            or _get_path(session, "sourceComment", "id")
        ),
    )


def is_authorized(
    context: LinearWebhookContext,
    *,
    allowed_users: list[str] | tuple[str, ...] | set[str] | None = None,
    allowed_teams: list[str] | tuple[str, ...] | set[str] | None = None,
    allow_all_users: bool = False,
) -> bool:
    """Adapter-layer authorization for an inbound webhook — FAIL CLOSED.

    A sender is authorized only via ``allow_all_users`` or an explicit
    ``allowed_users`` match; ``allowed_teams`` (when set) narrows further.
    With no allowlist configured at all the answer is **deny**: the gateway
    env layer enforces the same fail-closed policy at dispatch, but this
    check runs FIRST, and pre-dispatch side effects (created-ack thought,
    auto-start, the stop-signal interrupt + confirmation) must never fire
    for a sender the gateway would reject.
    """
    users = {str(item).strip() for item in (allowed_users or []) if str(item).strip()}
    teams = {str(item).strip() for item in (allowed_teams or []) if str(item).strip()}

    if not allow_all_users and context.actor_user_id not in users:
        return False
    if teams and context.team_id not in teams:
        return False
    return True


def _render_previous_comments(value: Any, *, budget: int = 6000) -> str:
    """Render previous comments as compact `author: body` lines.

    Raw JSON dumps carry noisy ids and a mid-object truncation risk; unknown
    shapes still fall back to (whole-item-truncated) JSON.
    """
    if not isinstance(value, (list, tuple)):
        return json.dumps(value)[:budget]
    lines: list[str] = []
    used = 0
    for item in value:
        if isinstance(item, Mapping):
            author = _string(
                _get_path(item, "user", "name")
                or _get_path(item, "user", "displayName")
                or item.get("userId")
            ) or "unknown"
            body = _string(item.get("body"))
            rendered = f"- {author}: {body}"
        else:
            rendered = f"- {_string(item)}"
        if used + len(rendered) > budget:
            lines.append(f"- … ({len(value) - len(lines)} more comments truncated)")
            break
        lines.append(rendered)
        used += len(rendered)
    return "\n".join(lines)


def build_created_prompt(context: LinearWebhookContext, payload: Mapping[str, Any]) -> str:
    previous_comments = payload.get("previousComments")
    previous_comments_text = ""
    if previous_comments:
        previous_comments_text = _render_previous_comments(previous_comments)

    lines = [
        "You were invoked from Linear.",
        "",
        "## Linear Agent Session",
        f"- Session ID: {context.agent_session_id}",
        f"- Workspace ID: {context.workspace_id or '(not provided)'}",
        f"- Issue ID: {context.issue_id or '(not provided)'}",
        f"- Issue: {context.issue_identifier or '(not provided)'} {context.issue_title}".rstrip(),
        f"- Comment ID: {context.comment_id or '(not provided)'}",
        f"- Initiating user ID: {context.actor_user_id or '(not provided)'}",
        f"- Team ID: {context.team_id or '(not provided)'}",
        "",
        "## Linear Prompt Context",
        "",
        context.prompt_context or "(not provided)",
    ]
    if context.guidance:
        lines.extend(["", "## Linear Guidance", "", context.guidance])
    if previous_comments_text:
        lines.extend(["", "## Previous Comments", "", previous_comments_text])
    lines.extend(
        [
            "",
            "## User Request",
            "",
            context.user_request or "(not provided)",
            "",
            "## Instructions",
            "",
            "Respond as the configured assistant. If more information is needed, ask a concise clarification. If work is complete, provide a final response suitable for Linear.",
        ]
    )
    return "\n".join(lines)


def build_prompted_message(context: LinearWebhookContext) -> str:
    return context.user_request or "(No follow-up text was provided by Linear.)"


def build_update_prompt(context: LinearWebhookContext) -> str:
    """Prompt used when a standard Issue update webhook arrives (for example, assignment)."""
    lines = [
        "Linear issue update received.",
        "",
        "## Linear Issue",
        f"- Issue: {context.issue_identifier or '(not provided)'} {context.issue_title}".rstrip(),
        f"- Issue ID: {context.issue_id or '(not provided)'}",
        f"- Team ID: {context.team_id or '(not provided)'}",
        "",
        "You have been assigned to this issue or the issue was updated.",
        "Review the issue and take appropriate next steps.",
        "",
        "## Instructions",
        "Respond as the configured assistant. If more information is needed, ask a concise clarification. If work is complete, provide a final response suitable for Linear.",
    ]
    return "\n".join(lines)
