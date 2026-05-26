"""slack-bot-bridge — bot-to-bot ingress for private-network Hermes.

The plugin watches a single Slack channel that operators have nominated as
the "bridge" channel. When an external bot posts a tagged JSON envelope into
that channel and mentions Hermes, the plugin rewrites the message into a
canonical prompt so the regular Hermes dispatch path picks it up. Hermes
never has to be reachable from the public internet — Slack is the relay.

Wire format (inside the Slack ``event.text``)::

    <@U_HERMES_BOT> hermes-bridge
    ```json
    {
      "request_id": "abc-123",
      "actor": "github-actions",
      "prompt": "Triage failures in PR #4242",
      "metadata": {"pr": 4242},
      "signature": "<optional hex-encoded HMAC-SHA256>"
    }
    ```

Only ``request_id`` and ``prompt`` are required. ``signature`` is verified
against ``HERMES_SLACK_BRIDGE_HMAC_SECRET`` when both are present. The
plugin never accepts free-form bot text — the marker token and the JSON
envelope are mandatory.

Configuration (env vars; all optional unless noted):

* ``HERMES_SLACK_BRIDGE_CHANNEL`` (required)
    Slack channel ID (``C0123…``) of the dedicated bridge channel. Messages
    in any other channel are ignored. Operators must also restrict the
    Slack adapter itself to this channel (``platforms.slack.extra.allowed_channels``).
* ``HERMES_SLACK_BRIDGE_ALLOWED_BOT_IDS``
    Comma-separated allowlist of Slack ``bot_id`` values (``B0…``). Empty
    means "accept any bot in the bridge channel" — fine for a single-tenant
    workspace, but you almost always want at least one entry.
* ``HERMES_SLACK_BRIDGE_ALLOWED_APP_IDS``
    Comma-separated allowlist of Slack ``app_id`` values (``A0…``). Slack
    includes ``app_id`` on bot events sourced from an installed app; this
    is the most stable identifier when ``bot_id`` rotates.
* ``HERMES_SLACK_BRIDGE_ALLOWED_TEAMS``
    Comma-separated allowlist of Slack workspace ``team_id`` values.
* ``HERMES_SLACK_BRIDGE_HMAC_SECRET``
    When set, every envelope must carry a ``signature`` field (hex
    HMAC-SHA256 over ``request_id|actor|prompt|canonical_metadata``).
    Envelopes with a missing or mismatched signature are rejected.
* ``HERMES_SLACK_BRIDGE_DEDUP_TTL_SECONDS``
    Override the dedup window (default 86400 seconds = 24 h). Messages with
    a ``request_id`` already in the cache during the window are dropped.

Auth caveat (do not patch core):
    The plugin returns ``{"action": "rewrite", "text": ...}`` and leaves the
    rest of the gateway path untouched, including authorization. Slack's
    ``MessageEvent.source.user_id`` is set to the *sending* bot's Slack
    user id, so the operator must include that id (typically ``U0…``) in
    ``SLACK_ALLOWED_USERS`` for the rewritten message to dispatch. See the
    plugin README for the exact env-var snippet.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MARKER = "hermes-bridge"
"""Required token in the Slack text. The plugin will only consider events
whose text contains this exact (case-insensitive) token alongside a
``<@U…>`` Slack mention — see :func:`_extract_envelope_text`."""

_DEFAULT_DEDUP_TTL_SECONDS = 86_400
"""Default request_id retention window (24 hours)."""

_DEDUP_MAX_ENTRIES = 10_000
"""Hard cap on the dedup store to keep it bounded even if TTL is huge."""

_DEDUP_FILE_NAME = "dedup.json"

# ``request_id`` and ``actor`` must be short opaque strings — anything else
# is almost certainly a misuse of the envelope. Validated to keep the
# canonical signing input bounded.
_ID_SAFE = re.compile(r"^[A-Za-z0-9_\-\.:]{1,128}$")

# Slack mention smoke check. We just want to know "was this addressed to
# someone" before treating the marker as a bridge request — the actual
# Hermes-specific @-mention gate is enforced upstream by the Slack adapter
# (strict_mention=true plus the channel allowlist). Kept deliberately
# permissive so test/synthetic IDs (which may contain ``_``) also match.
_MENTION_RE = re.compile(r"<@[A-Z0-9_]+>", re.IGNORECASE)

# A fenced ```json …``` block is the canonical envelope shape, but we also
# tolerate an inline ``{ … }`` JSON object after the marker so the plugin
# stays usable from clients that strip code-fence formatting (Slackbot
# scheduled posts, mobile composers, etc.).
_FENCED_JSON_RE = re.compile(
    r"```(?:json)?\s*(?P<body>.*?)\s*```",
    re.DOTALL | re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_set(name: str) -> set[str]:
    raw = _env(name)
    if not raw:
        return set()
    return {part.strip() for part in raw.split(",") if part.strip()}


def _dedup_ttl_seconds() -> int:
    raw = _env("HERMES_SLACK_BRIDGE_DEDUP_TTL_SECONDS")
    if not raw:
        return _DEFAULT_DEDUP_TTL_SECONDS
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "slack-bot-bridge: invalid HERMES_SLACK_BRIDGE_DEDUP_TTL_SECONDS=%r, using default",
            raw,
        )
        return _DEFAULT_DEDUP_TTL_SECONDS
    return max(0, value)


def _hermes_home_path() -> Path:
    """Resolve HERMES_HOME the same way the rest of Hermes does.

    Imported lazily so test isolation that monkeypatches ``HERMES_HOME``
    after module load is honored.
    """
    try:
        from hermes_constants import get_hermes_home

        return get_hermes_home()
    except Exception:
        # Fall back to the documented default if hermes_constants somehow
        # isn't importable (broken venv, etc.) — the plugin shouldn't take
        # the gateway down with it.
        return Path(os.path.expanduser("~/.hermes"))


def _dedup_path() -> Path:
    return _hermes_home_path() / "slack-bot-bridge" / _DEDUP_FILE_NAME


# ---------------------------------------------------------------------------
# Dedup store — file-backed for cross-process survival, lock-guarded
# ---------------------------------------------------------------------------


_DEDUP_LOCK = threading.Lock()
_DEDUP_MEMORY_FALLBACK: Dict[str, float] = {}


def _read_dedup_state(path: Path) -> Dict[str, float]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        logger.warning("slack-bot-bridge: cannot read dedup store %s: %s", path, exc)
        return {}
    try:
        loaded = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        logger.warning("slack-bot-bridge: dedup store at %s is corrupt: %s", path, exc)
        return {}
    if not isinstance(loaded, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in loaded.items():
        if isinstance(k, str) and isinstance(v, (int, float)):
            out[k] = float(v)
    return out


def _write_dedup_state(path: Path, state: Dict[str, float]) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state), encoding="utf-8")
        tmp.replace(path)
        return True
    except OSError as exc:
        logger.warning("slack-bot-bridge: cannot write dedup store %s: %s", path, exc)
        return False


def _prune_dedup_state(state: Dict[str, float], ttl: int, now: float) -> Dict[str, float]:
    if ttl <= 0:
        return state
    cutoff = now - ttl
    pruned = {k: v for k, v in state.items() if v >= cutoff}
    if len(pruned) > _DEDUP_MAX_ENTRIES:
        # Keep the most recent _DEDUP_MAX_ENTRIES // 2 entries — bound the
        # file size when an operator picks a very large TTL or a single bot
        # floods the channel.
        keep = sorted(pruned.items(), key=lambda kv: kv[1], reverse=True)[: _DEDUP_MAX_ENTRIES // 2]
        pruned = dict(keep)
    return pruned


def _claim_request_id(request_id: str, *, now: Optional[float] = None) -> bool:
    """Return True if *request_id* was freshly accepted; False if duplicate.

    The check-and-insert is atomic with respect to other threads in the same
    process via :data:`_DEDUP_LOCK`. Across processes a lost-update race is
    possible if two gateways come up at once on the same HERMES_HOME, but
    Hermes is a single-instance daemon — that race is documented rather
    than locked against.
    """
    now = time.time() if now is None else now
    ttl = _dedup_ttl_seconds()
    path = _dedup_path()
    with _DEDUP_LOCK:
        state = _prune_dedup_state(_read_dedup_state(path), ttl, now)
        if request_id in state:
            return False
        state[request_id] = now
        if _write_dedup_state(path, state):
            return True

        # If the file store is unavailable, fail closed for duplicates within
        # this process rather than accepting every Slack retry. This keeps the
        # gateway usable on transient filesystem errors while preserving the
        # core idempotency guarantee for the running daemon.
        fallback = _prune_dedup_state(_DEDUP_MEMORY_FALLBACK, ttl, now)
        if request_id in fallback:
            _DEDUP_MEMORY_FALLBACK.clear()
            _DEDUP_MEMORY_FALLBACK.update(fallback)
            return False
        fallback[request_id] = now
        _DEDUP_MEMORY_FALLBACK.clear()
        _DEDUP_MEMORY_FALLBACK.update(fallback)
    return True


# ---------------------------------------------------------------------------
# Envelope parsing
# ---------------------------------------------------------------------------


def _looks_like_bridge_attempt(text: str) -> bool:
    """True when the Slack text has a mention plus the bridge marker."""
    return bool(
        isinstance(text, str)
        and text
        and _MENTION_RE.search(text)
        and MARKER.lower() in text.lower()
    )


def _extract_envelope_text(text: str) -> Optional[str]:
    """Return the JSON body following the bridge marker, or ``None``.

    The marker must follow a Slack ``<@…>`` mention so accidental matches
    in casual conversation are impossible. ``text`` is the raw Slack
    payload before adapter post-processing (we receive it via
    ``event.raw_message["text"]``), so user/bot mentions are still present
    as ``<@…>`` tokens.
    """
    if not isinstance(text, str) or not text:
        return None

    # Must include at least one Slack mention. The Slack adapter already
    # required Hermes specifically to be mentioned (strict_mention=true),
    # so we don't need to re-verify the bot uid here — guarding on *any*
    # mention is just a cheap safety net against random bot chatter that
    # happens to contain the marker string.
    if not _MENTION_RE.search(text):
        return None

    # Locate the marker; lowercase the haystack so we accept "Hermes-Bridge",
    # "HERMES-BRIDGE", etc. We still slice with the original-cased indices
    # so JSON content (which IS case-sensitive) is untouched.
    haystack = text.lower()
    idx = haystack.find(MARKER.lower())
    if idx < 0:
        return None

    after = text[idx + len(MARKER):]
    # Prefer the fenced ```json ... ``` block — that's the canonical wire
    # format and round-trips through every Slack client. Fall back to the
    # first balanced ``{...}`` chunk if no fence is present.
    fenced = _FENCED_JSON_RE.search(after)
    if fenced:
        return fenced.group("body")
    return _first_balanced_json_object(after)


def _first_balanced_json_object(text: str) -> Optional[str]:
    """Return the substring of *text* containing the first balanced ``{...}``.

    Tolerates extra whitespace/commentary around the JSON but does not
    attempt to parse multiple objects — only the first.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _parse_envelope(body: str) -> Optional[Dict[str, Any]]:
    try:
        loaded = json.loads(body)
    except json.JSONDecodeError as exc:
        logger.info("slack-bot-bridge: envelope is not valid JSON: %s", exc)
        return None
    if not isinstance(loaded, dict):
        logger.info("slack-bot-bridge: envelope top-level must be an object")
        return None
    return loaded


def _validate_envelope_fields(envelope: Dict[str, Any]) -> Optional[Tuple[str, str, str, Dict[str, Any]]]:
    """Return ``(request_id, actor, prompt, metadata)`` or ``None``.

    Rejects oversize or malformed values up front so downstream logic and
    dedup keys stay bounded. ``actor`` defaults to ``""`` when omitted; the
    canonical signing input uses the empty string for that field.
    """
    request_id = envelope.get("request_id")
    prompt = envelope.get("prompt")
    actor = envelope.get("actor", "")
    metadata = envelope.get("metadata", {})

    if not isinstance(request_id, str) or not _ID_SAFE.match(request_id):
        logger.info("slack-bot-bridge: envelope missing/invalid request_id")
        return None
    if not isinstance(prompt, str) or not prompt.strip():
        logger.info("slack-bot-bridge: envelope missing/empty prompt")
        return None
    if len(prompt) > 32_000:
        # Slack already truncates at ~40k; this keeps the prompt within a
        # sensible LLM-side budget without us having to know the model.
        logger.info("slack-bot-bridge: envelope prompt too large (%d chars)", len(prompt))
        return None
    if not isinstance(actor, str) or (actor and not _ID_SAFE.match(actor)):
        logger.info("slack-bot-bridge: envelope actor is invalid")
        return None
    if not isinstance(metadata, dict):
        logger.info("slack-bot-bridge: envelope metadata must be an object")
        return None

    return request_id, actor, prompt, metadata


# ---------------------------------------------------------------------------
# HMAC validation (optional)
# ---------------------------------------------------------------------------


def _canonical_metadata(metadata: Dict[str, Any]) -> str:
    """Return the stable JSON form covered by the envelope HMAC."""
    return json.dumps(metadata or {}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _canonical_signing_input(request_id: str, actor: str, prompt: str, metadata: Dict[str, Any]) -> bytes:
    # Fixed delimiter `|` — actor/request_id are validated against _ID_SAFE
    # so neither can contain it. Metadata is canonical JSON so clients and
    # Hermes sign the same bytes. We sign metadata too because CI/deploy
    # workflows often place repo/run identifiers there.
    return f"{request_id}|{actor}|{prompt}|{_canonical_metadata(metadata)}".encode("utf-8")


def _verify_signature(secret: str, envelope: Dict[str, Any], request_id: str, actor: str, prompt: str, metadata: Dict[str, Any]) -> bool:
    provided = envelope.get("signature", "")
    if not isinstance(provided, str) or not provided:
        return False
    try:
        expected = hmac.new(
            secret.encode("utf-8"),
            _canonical_signing_input(request_id, actor, prompt, metadata),
            hashlib.sha256,
        ).hexdigest()
    except Exception as exc:  # pragma: no cover - hmac never raises on bytes/str
        logger.warning("slack-bot-bridge: hmac computation failed: %s", exc)
        return False
    return hmac.compare_digest(expected.lower(), provided.strip().lower())


# ---------------------------------------------------------------------------
# Allowlist gating
# ---------------------------------------------------------------------------


def _slack_channel_allowed(channel_id: str) -> bool:
    target = _env("HERMES_SLACK_BRIDGE_CHANNEL")
    if not target:
        return False  # Plugin is inert until the operator names a channel.
    return channel_id == target


def _identifiers_allowed(raw_message: Dict[str, Any]) -> bool:
    bot_allowed = _env_set("HERMES_SLACK_BRIDGE_ALLOWED_BOT_IDS")
    app_allowed = _env_set("HERMES_SLACK_BRIDGE_ALLOWED_APP_IDS")
    team_allowed = _env_set("HERMES_SLACK_BRIDGE_ALLOWED_TEAMS")

    if not (bot_allowed or app_allowed or team_allowed):
        # Operators that haven't pinned identifiers fall back to channel-only
        # gating. The bridge channel itself is single-purpose, so this is a
        # reasonable default for quick bring-up, but production should set
        # at least one app_id or bot_id allowlist.
        return True

    bot_id = str(raw_message.get("bot_id") or "")
    app_id = str(raw_message.get("app_id") or raw_message.get("api_app_id") or "")
    team_id = str(raw_message.get("team") or raw_message.get("team_id") or "")

    if team_allowed and (not team_id or team_id not in team_allowed):
        return False

    # app_id and bot_id are alternative identities for the same sender. If
    # either list is configured, one of those sender identities must match.
    sender_id_lists_configured = bool(bot_allowed or app_allowed)
    if sender_id_lists_configured:
        return bool(
            (bot_allowed and bot_id and bot_id in bot_allowed)
            or (app_allowed and app_id and app_id in app_allowed)
        )

    # Only the team list was configured, and it matched above.
    return True


# ---------------------------------------------------------------------------
# Canonical prompt rewrite
# ---------------------------------------------------------------------------


def _canonical_prompt(*, prompt: str, actor: str, request_id: str, metadata: Dict[str, Any]) -> str:
    """Format the rewritten prompt sent into Hermes dispatch.

    Kept terse on purpose — the LLM gets a small structured header it can
    ignore and the original prompt text immediately after. Metadata is
    JSON-encoded so downstream tools can re-parse it without ambiguity.
    """
    header_lines = [
        "[slack-bot-bridge]",
        f"request_id: {request_id}",
    ]
    if actor:
        header_lines.append(f"actor: {actor}")
    if metadata:
        # Sort keys for stable output (eases test assertions and grep).
        header_lines.append(
            "metadata: " + json.dumps(metadata, sort_keys=True, ensure_ascii=False)
        )
    header = "\n".join(header_lines)
    return f"{header}\n\n{prompt.strip()}"


# ---------------------------------------------------------------------------
# Hook entry point
# ---------------------------------------------------------------------------


def _is_slack_event(event: Any) -> bool:
    source = getattr(event, "source", None)
    platform = getattr(source, "platform", None)
    platform_value = getattr(platform, "value", platform)
    return isinstance(platform_value, str) and platform_value.lower() == "slack"


def on_pre_gateway_dispatch(*, event: Any = None, **_: Any) -> Optional[Dict[str, Any]]:
    """Inspect a Slack event; rewrite into a canonical prompt if it's a bridge envelope.

    Returns ``None`` for everything that isn't a bridge envelope — the
    gateway treats that as "let normal dispatch handle it". On success
    returns ``{"action": "rewrite", "text": <canonical-prompt>}``. The
    function is intentionally exception-tolerant; any unexpected failure
    is logged and we return ``None`` so a broken bridge never blocks
    normal Slack traffic.
    """
    try:
        return _on_pre_gateway_dispatch_inner(event)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("slack-bot-bridge: handler raised, ignoring: %s", exc)
        return None


def _on_pre_gateway_dispatch_inner(event: Any) -> Optional[Dict[str, Any]]:
    if event is None or not _is_slack_event(event):
        return None

    raw = getattr(event, "raw_message", None)
    if not isinstance(raw, dict):
        return None

    # The Slack adapter only stamps ``raw_message`` from real message_event
    # callbacks; slash-command callbacks build their own dict with a
    # ``command`` key. We don't speak the slash-command shape here.
    if "command" in raw and "text" in raw and "channel_id" in raw:
        return None

    source = getattr(event, "source", None)
    chat_id = getattr(source, "chat_id", "") or ""
    if not _slack_channel_allowed(chat_id):
        return None

    # Use the raw Slack text (mentions intact). The adapter strips the
    # mention before forwarding, but raw_message preserves the original.
    body_text = raw.get("text")
    body_text = body_text if isinstance(body_text, str) else ""
    bridge_attempt = _looks_like_bridge_attempt(body_text)
    if not bridge_attempt:
        return None

    # The plugin is bot-to-bot ingress only. If a human happens to type
    # "hermes-bridge ..." in the bridge channel, fail closed for the bridge
    # attempt instead of letting the raw envelope fall through to normal
    # dispatch.
    bot_id = raw.get("bot_id") or ""
    subtype = raw.get("subtype") or ""
    if not bot_id and subtype != "bot_message":
        return {"action": "skip", "reason": "bridge-attempt-not-from-bot"}

    if not _identifiers_allowed(raw):
        logger.info(
            "slack-bot-bridge: rejecting message in bridge channel from non-allowlisted bot "
            "(bot_id=%s app_id=%s team=%s)",
            bot_id,
            raw.get("app_id") or raw.get("api_app_id") or "",
            raw.get("team") or raw.get("team_id") or "",
        )
        return {"action": "skip", "reason": "sender-not-allowlisted"}

    envelope_body = _extract_envelope_text(body_text)
    if not envelope_body:
        return {"action": "skip", "reason": "missing-json-envelope"}

    envelope = _parse_envelope(envelope_body)
    if envelope is None:
        return {"action": "skip", "reason": "invalid-json-envelope"}

    fields = _validate_envelope_fields(envelope)
    if fields is None:
        return {"action": "skip", "reason": "invalid-envelope-fields"}
    request_id, actor, prompt, metadata = fields

    secret = _env("HERMES_SLACK_BRIDGE_HMAC_SECRET")
    if secret:
        if not _verify_signature(secret, envelope, request_id, actor, prompt, metadata):
            logger.warning(
                "slack-bot-bridge: rejecting envelope with bad/missing signature (request_id=%s)",
                request_id,
            )
            return {"action": "skip", "reason": "bad-signature"}

    if not _claim_request_id(request_id):
        logger.info("slack-bot-bridge: dropping duplicate request_id=%s", request_id)
        # Returning skip drops the message without a reply; the relay
        # already received an HTTP 200 from Slack so the upstream caller
        # won't retry needlessly.
        return {"action": "skip", "reason": "duplicate-request-id"}

    rewritten = _canonical_prompt(
        prompt=prompt,
        actor=actor,
        request_id=request_id,
        metadata=metadata,
    )
    logger.info(
        "slack-bot-bridge: accepted envelope request_id=%s actor=%s (prompt %d chars)",
        request_id,
        actor or "<none>",
        len(prompt),
    )
    return {"action": "rewrite", "text": rewritten}


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    ctx.register_hook("pre_gateway_dispatch", on_pre_gateway_dispatch)
