"""Copilot auto-mode router: unlocks the 10% billing discount.

Implements the three-call dance that GitHub Copilot Chat uses internally
to route ``model: auto`` selections through a server-side ML router and
have token costs charged at a 0.9× multiplier:

    1. ``POST /models/session``         → ``session_token`` (ES256 JWT, 1h TTL)
    2. ``POST /models/session/intent``  → router decision (chosen_model + scores)
    3. ``POST /chat/completions``       → request with ``Copilot-Session-Token``
       (or ``/responses`` / ``/v1/messages`` depending on chosen_model family)

Step 2 is optional (*only step 1 + step 3 are required for the discount*).
The session_token from step 1 already encodes the ``discounted_costs`` map;
the server applies the 0.9× multiplier whenever the header is present.

Empirical verification (2026-06-02, ``probe_automode_full.py``)::

    WITH    Copilot-Session-Token   input=225e9 output=1.35e12 nano_aiu=9_675_000
    WITHOUT                         input=250e9 output=1.50e12 nano_aiu=10_750_000
    ratio = 0.90 across all token-cost lines.

Endpoint routing (CAPI fixed binding by model family):
    gpt-5.* / *codex     → /responses
    claude-*             → /v1/messages
    everything else      → /chat/completions

Env overrides
-------------
- ``HERMES_COPILOT_AUTO_MODE``        ``1`` (default) or ``0`` to disable.
- ``HERMES_COPILOT_AUTO_HINTS``       comma-separated model hints
                                      (default ``auto``).
- ``HERMES_COPILOT_AUTO_INTENT``      ``1`` (default) to call the intent
                                      router for ``chosen_model``;
                                      ``0`` to use ``selected_model`` from
                                      the session response (still 10% off).
- ``HERMES_COPILOT_API_VERSION``      pin ``X-GitHub-Api-Version`` (else
                                      resolved by ``hermes_cli.copilot_auth``).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

CAPI_BASE_URL = "https://api.githubcopilot.com"
CAPI_AUTO_MODEL_URL = f"{CAPI_BASE_URL}/models/session"
CAPI_MODEL_ROUTER_URL = f"{CAPI_BASE_URL}/models/session/intent"
CAPI_CHAT_URL = f"{CAPI_BASE_URL}/chat/completions"
CAPI_RESPONSES_URL = f"{CAPI_BASE_URL}/responses"
CAPI_MESSAGES_URL = f"{CAPI_BASE_URL}/v1/messages"

# Refresh JWT this many seconds before its ``exp`` claim to avoid mid-flight
# expiry. Matches the 5-minute window observed in extension.js (``oUe`` /
# ``dne`` cache logic).
_SESSION_REFRESH_MARGIN_SECONDS = 300


# ─── Public dataclasses ────────────────────────────────────────────────────


@dataclass
class AutoModeSession:
    """Session-token bundle returned by ``POST /models/session``.

    The ``session_token`` is an ES256 JWT whose payload mirrors the
    top-level fields here (plus ``sub``, ``iat``, ``exp``).
    """

    session_token: str
    available_models: list[str]
    selected_model: str
    expires_at: float  # epoch seconds (from JWT ``exp``)
    discounted_costs: dict[str, float] = field(default_factory=dict)

    def is_stale(self, *, now: Optional[float] = None) -> bool:
        now = now if now is not None else time.time()
        return now >= self.expires_at - _SESSION_REFRESH_MARGIN_SECONDS


@dataclass
class RouterDecision:
    """Response from ``POST /models/session/intent``."""

    chosen_model: str
    candidate_models: list[str]
    predicted_label: str = ""
    confidence: float = 0.0
    scores: dict[str, float] = field(default_factory=dict)
    hydra_scores: dict[str, float] = field(default_factory=dict)
    reasoning_bucket: str = ""
    routing_method: str = ""
    fallback: bool = False
    fallback_reason: str = ""
    sticky_override: bool = False
    latency_ms: int = 0
    raw: dict = field(default_factory=dict)


@dataclass
class AutoRouteResult:
    """Resolved auto-routing outcome for a single Copilot turn."""

    session: Optional[AutoModeSession]
    decision: Optional[RouterDecision]
    chosen_model: str
    session_token: str = ""
    fallback_reason: str = ""


# ─── Endpoint mapping ──────────────────────────────────────────────────────


def endpoint_for_model(model: str) -> str:
    """Return the CAPI URL the chosen model must be POSTed to.

    The CAPI rejects cross-family requests (e.g. ``model: gpt-5.3-codex`` on
    ``/chat/completions`` returns ``unsupported_api_for_model``).
    """
    m = model.lower()
    if "codex" in m or m.startswith("gpt-5.") or m.startswith("o1") or m.startswith("o3"):
        return CAPI_RESPONSES_URL
    if "claude" in m:
        return CAPI_MESSAGES_URL
    return CAPI_CHAT_URL


# ─── JWT helpers ───────────────────────────────────────────────────────────


def _b64url_decode(segment: str) -> bytes:
    pad = "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(segment + pad)


def _decode_jwt_payload(token: str) -> dict:
    """Decode the payload of an unverified JWT.

    The session JWT is ES256-signed by the CAPI; we don't have the public key
    and don't need it (the server validates on receipt). We only read the
    payload to learn ``exp`` and ``discounted_costs`` for client-side caching.
    """
    try:
        _, payload, _ = token.split(".", 2)
        return json.loads(_b64url_decode(payload))
    except Exception as exc:  # pragma: no cover - malformed tokens
        logger.debug("auto-mode: JWT decode failed: %s", exc)
        return {}


# ─── Auto-mode router ──────────────────────────────────────────────────────


def is_enabled() -> bool:
    """Master switch. Defaults to on."""
    val = os.getenv("HERMES_COPILOT_AUTO_MODE", "1").strip().lower()
    return val not in ("0", "false", "no", "off")


def _model_hints() -> list[str]:
    raw = os.getenv("HERMES_COPILOT_AUTO_HINTS", "auto").strip()
    return [h.strip() for h in raw.split(",") if h.strip()] or ["auto"]


def _intent_enabled() -> bool:
    val = os.getenv("HERMES_COPILOT_AUTO_INTENT", "1").strip().lower()
    return val not in ("0", "false", "no", "off")


def _render_prompt_from_messages(messages: list[dict]) -> str:
    """Render the current turn into a plain-text routing prompt."""
    sections: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "context").strip().lower()
        content = message.get("content")
        rendered = _render_message_content(content)
        if not rendered:
            continue
        label = {
            "system": "System",
            "user": "User",
            "assistant": "Assistant",
            "tool": "Tool",
        }.get(role, "Context")
        sections.append(f"{label}:\n{rendered}")
    return "\n\n".join(section.strip() for section in sections if section and section.strip())


def _render_message_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return str(content.get("text") or "").strip()
        if isinstance(content.get("content"), str):
            return str(content.get("content") or "").strip()
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    parts.append(text)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts).strip()
    return str(content).strip()


class AutoRouter:
    """Caches auto-mode sessions and applies the discount header.

    Session JWTs are reused across requests within their TTL, matching the
    extension's ``oUe._autoModelCache`` keyed by conversation-id. Pass a
    stable ``conversation_id`` to share a session across turns; pass
    ``None`` for a one-shot bank.

    Thread-safe: a single lock guards the cache map.
    """

    def __init__(self) -> None:
        self._cache: dict[str, AutoModeSession] = {}
        self._lock = threading.Lock()

    # -- session lifecycle ------------------------------------------------

    def get_session(
        self,
        bearer_token: str,
        *,
        conversation_id: Optional[str] = None,
        force_refresh: bool = False,
        timeout: float = 10.0,
    ) -> Optional[AutoModeSession]:
        """Return a cached or freshly-issued auto-mode session.

        Returns ``None`` if auto-mode is disabled or the issuance call fails;
        callers should fall through to the normal (un-discounted) request path.
        """
        if not is_enabled():
            return None
        if not bearer_token:
            return None

        key = conversation_id or "__default__"
        now = time.time()

        with self._lock:
            cached = self._cache.get(key)
            if cached and not force_refresh and not cached.is_stale(now=now):
                return cached

        session = self._issue_session(bearer_token, timeout=timeout)
        if session is None:
            if cached is not None:
                logger.warning(
                    "auto-mode: /models/session refresh failed for conversation=%s; using cached session",
                    key,
                )
                return cached
            return None

        with self._lock:
            self._cache[key] = session
        return session

    def invalidate(self, conversation_id: Optional[str] = None) -> None:
        key = conversation_id or "__default__"
        with self._lock:
            self._cache.pop(key, None)

    # -- HTTP calls -------------------------------------------------------

    def _issue_session(
        self,
        bearer_token: str,
        *,
        timeout: float,
    ) -> Optional[AutoModeSession]:
        body = json.dumps({"auto_mode": {"model_hints": _model_hints()}}).encode()
        headers = self._base_headers(bearer_token)
        req = urllib.request.Request(
            CAPI_AUTO_MODEL_URL, data=body, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode()
            except Exception:
                detail = ""
            logger.warning(
                "auto-mode: /models/session failed http=%s detail=%s",
                exc.code,
                detail[:200],
            )
            return None
        except Exception as exc:
            logger.warning("auto-mode: /models/session error: %s", exc)
            return None

        token = payload.get("session_token") or ""
        if not token:
            logger.warning("auto-mode: /models/session returned no session_token")
            return None

        decoded = _decode_jwt_payload(token)
        exp = float(decoded.get("exp") or (time.time() + 3600))

        session = AutoModeSession(
            session_token=token,
            available_models=list(
                payload.get("available_models")
                or decoded.get("available_models")
                or []
            ),
            selected_model=str(
                payload.get("selected_model")
                or decoded.get("selected_model")
                or ""
            ),
            expires_at=exp,
            discounted_costs=dict(
                payload.get("discounted_costs")
                or decoded.get("discounted_costs")
                or {}
            ),
        )
        logger.debug(
            "auto-mode: session issued, models=%s selected=%s discounted=%s exp_in=%.0fs",
            session.available_models,
            session.selected_model,
            session.discounted_costs,
            session.expires_at - time.time(),
        )
        return session

    def route(
        self,
        bearer_token: str,
        prompt: str,
        session: AutoModeSession,
        *,
        timeout: float = 10.0,
    ) -> Optional[RouterDecision]:
        """Call ``/models/session/intent`` to pick a model for ``prompt``.

        Both ``prompt`` and ``available_models`` are required by the server.
        On failure, returns a fail-open decision that preserves the
        session-selected model instead of aborting the request.
        """
        fallback_model = (
            str(session.selected_model or "").strip()
            or (session.available_models[0] if session.available_models else "")
        )
        if not _intent_enabled():
            return RouterDecision(
                chosen_model=fallback_model,
                candidate_models=list(session.available_models),
                routing_method="session_selected",
                fallback=True,
                fallback_reason="intent_disabled",
            )
        if not session.available_models:
            return RouterDecision(
                chosen_model=fallback_model,
                candidate_models=[],
                routing_method="session_selected",
                fallback=True,
                fallback_reason="no_available_models",
            )

        body = json.dumps(
            {"prompt": prompt, "available_models": session.available_models}
        ).encode()
        headers = self._base_headers(bearer_token)
        headers["Copilot-Session-Token"] = session.session_token
        req = urllib.request.Request(
            CAPI_MODEL_ROUTER_URL, data=body, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode()
            except Exception:
                detail = ""
            logger.warning(
                "auto-mode: /models/session/intent failed http=%s detail=%s",
                exc.code,
                detail[:200],
            )
            return RouterDecision(
                chosen_model=fallback_model,
                candidate_models=list(session.available_models),
                routing_method="session_selected",
                fallback=True,
                fallback_reason=f"http_{exc.code}",
            )
        except Exception as exc:
            logger.warning("auto-mode: /models/session/intent error: %s", exc)
            return RouterDecision(
                chosen_model=fallback_model,
                candidate_models=list(session.available_models),
                routing_method="session_selected",
                fallback=True,
                fallback_reason=str(exc)[:200],
            )

        chosen = str(payload.get("chosen_model") or session.selected_model or "")
        decision = RouterDecision(
            chosen_model=chosen,
            candidate_models=list(payload.get("candidate_models") or session.available_models or []),
            predicted_label=str(payload.get("predicted_label") or ""),
            confidence=float(payload.get("confidence") or 0.0),
            scores=dict(payload.get("scores") or {}),
            hydra_scores=dict(payload.get("hydra_scores") or {}),
            reasoning_bucket=str(payload.get("reasoning_bucket") or ""),
            routing_method=str(payload.get("routing_method") or ""),
            fallback=bool(payload.get("fallback") or False),
            fallback_reason=str(payload.get("fallback_reason") or ""),
            sticky_override=bool(payload.get("sticky_override") or False),
            latency_ms=int(payload.get("latency_ms") or 0),
            raw=payload,
        )
        if not decision.chosen_model:
            decision.chosen_model = fallback_model
            decision.fallback = True
            decision.fallback_reason = decision.fallback_reason or "empty_router_choice"
            decision.routing_method = decision.routing_method or "session_selected"
        logger.debug(
            "auto-mode: router chose %s (label=%s conf=%.2f bucket=%s fallback=%s)",
            decision.chosen_model,
            decision.predicted_label,
            decision.confidence,
            decision.reasoning_bucket,
            decision.fallback,
        )
        return decision

    def resolve(
        self,
        bearer_token: str,
        prompt: str,
        *,
        conversation_id: Optional[str] = None,
        timeout: float = 10.0,
        fallback_model: str = "",
    ) -> AutoRouteResult:
        """Resolve an auto-mode turn to a concrete model.

        This keeps the session token attached even when the intent route
        fails, and it falls open to the server/session-selected model rather
        than aborting the request.
        """
        session = self.get_session(
            bearer_token,
            conversation_id=conversation_id,
            timeout=timeout,
        )
        if session is None:
            fallback = str(fallback_model or "").strip()
            return AutoRouteResult(
                session=None,
                decision=None,
                chosen_model=fallback,
                session_token="",
                fallback_reason="session_unavailable",
            )

        decision = self.route(
            bearer_token,
            prompt,
            session,
            timeout=timeout,
        )
        chosen = str(decision.chosen_model if decision else "").strip()
        if not chosen:
            chosen = str(session.selected_model or "").strip()
        if not chosen and session.available_models:
            chosen = session.available_models[0]
        if not chosen:
            chosen = str(fallback_model or "").strip()

        if decision is not None:
            if not decision.chosen_model:
                decision.chosen_model = chosen
            if decision.fallback_reason and not decision.routing_method:
                decision.routing_method = "session_selected"

        return AutoRouteResult(
            session=session,
            decision=decision,
            chosen_model=chosen,
            session_token=session.session_token,
            fallback_reason=(decision.fallback_reason if decision else "") or "",
        )

    # -- Header helpers ---------------------------------------------------

    @staticmethod
    def attach_session_header(
        headers: dict[str, str], session: AutoModeSession
    ) -> dict[str, str]:
        """Return ``headers`` with ``Copilot-Session-Token`` set.

        Mutates and returns the same dict for chaining.
        """
        headers["Copilot-Session-Token"] = session.session_token
        return headers

    def _base_headers(self, bearer_token: str) -> dict[str, str]:
        # Lazy import to avoid a hard dep on hermes_cli at module import time
        # (e.g. for unit tests that stub HTTP calls).
        try:
            from hermes_cli.copilot_auth import (
                copilot_request_headers,
                get_copilot_api_token,
            )

            base = copilot_request_headers(is_agent_turn=True, model="auto")
            api_token = get_copilot_api_token(bearer_token)
        except Exception as exc:  # pragma: no cover
            logger.debug("auto-mode: hermes_cli.copilot_auth unavailable: %s", exc)
            # copilot_auth is the single source of truth; this fallback only
            # fires if it is unavailable. Mirror its Copilot CLI identity shape
            # (no Editor-* VS Code headers) so the identity stays consistent.
            base = {
                "User-Agent": "copilot/1.0.63",
                "Copilot-Integration-Id": "copilot-developer-cli",
                "Runtime-Client-Version": "1.0.63",
                "Openai-Intent": "conversation-edits",
                "X-GitHub-Api-Version": "2026-06-01",
                "x-initiator": "agent",
            }
            api_token = bearer_token

        base.setdefault("Accept", "application/json")
        base["Content-Type"] = "application/json"
        base["Authorization"] = f"Bearer {api_token}"
        return base


# ─── Module-level singleton ───────────────────────────────────────────────


_default_router: Optional[AutoRouter] = None
_default_lock = threading.Lock()


def default_router() -> AutoRouter:
    """Process-wide singleton, lazily constructed."""
    global _default_router
    if _default_router is None:
        with _default_lock:
            if _default_router is None:
                _default_router = AutoRouter()
    return _default_router
