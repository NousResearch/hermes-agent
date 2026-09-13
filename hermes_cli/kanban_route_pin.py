"""Dispatch-time route-pin guard.

A card stamped with ``model_override``/``provider_override`` is pinned to a
backend by the SDLC route-binding machinery. When the backend cannot serve that
model id the *worker* is what discovers it: the spawn succeeds, the first API
call returns ``HTTP 400 No provider available for model '<X>'`` in ~0.01s, and
the card burns a run under an error that reads like a provider outage instead
of a bad pin.

This module owns the pre-spawn verdict the dispatcher uses to refuse that spawn
and diagnose the card instead:

* :func:`check_model_pin` — cached, cheap probe per distinct pin string.
* :func:`refusal_message` — the actionable operator-facing text.
* :func:`probe_model_pin` — the transport (one POST, ``max_tokens=16``).

Only a hard 400 carrying :data:`BAD_PIN_MARKER` is a bad pin. Everything else —
200 with empty content (a reasoning-budget artefact, not a dead route), 401/403,
404, 429, 5xx, timeouts, unresolvable provider config — is NOT evidence of a bad
pin and must never refuse a spawn, or a credential/route hiccup would stall the
board.

``GET /v1/models`` is deliberately NOT the authority: a served model can be
absent from that list (``ds : deepseek-v4-flash-vision-exp`` answered 200 in ~1s
while missing from ``/v1/models`` on 2026-09-12), so list membership would both
miss real bad pins and refuse good ones.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional

#: The router's own refusal for a model id no route claims. The only signal that
#: is allowed to refuse a spawn.
BAD_PIN_MARKER = "No provider available for model"

#: Floor for the reasoning models on this fleet: a smaller budget can return 200
#: with empty content, which must not be mistaken for a dead route.
PROBE_MAX_TOKENS = 16

DEFAULT_PROBE_TIMEOUT_S = 10.0

#: Cached verdict TTL, keyed by (provider, model, base_url). The fleet uses a
#: handful of distinct pins, so a dispatch tick stays cheap.
PROBE_CACHE_TTL_S = 300.0
#: Shorter TTL for indecisive verdicts (unreachable route, auth rejection): a
#: rotated credential or a recovered route should be picked up within a tick or
#: two, not after five minutes.
UNKNOWN_CACHE_TTL_S = 30.0

_MAX_BODY_CHARS = 500

# status -> served | bad_pin | unknown
SERVED = "served"
BAD_PIN = "bad_pin"
UNKNOWN = "unknown"

_PROBE_CACHE: dict[tuple[str, str, str], tuple[float, "PinProbeResult"]] = {}


@dataclass(frozen=True)
class PinProbeResult:
    """Outcome of one pin probe (or a cache hit)."""

    status: str
    http_status: Optional[int] = None
    detail: str = ""
    error_body: str = ""

    @property
    def bad_pin(self) -> bool:
        return self.status == BAD_PIN

    @property
    def served(self) -> bool:
        return self.status == SERVED

    def to_payload(self) -> dict:
        """Structured form for the card's ``model_pin_rejected`` event."""
        return {
            "verdict": self.status,
            "http_status": self.http_status,
            "detail": self.detail,
            "router_error": self.error_body,
        }


def classify_pin_response(http_status: Optional[int], body_text: str = "") -> PinProbeResult:
    """Map one probe response onto served / bad_pin / unknown.

    Pure: the whole classification policy lives here so it can be tested as data.
    """
    body = (body_text or "")[:_MAX_BODY_CHARS]
    if http_status is None:
        return PinProbeResult(UNKNOWN, None, "route unreachable", body)
    if http_status == 400 and BAD_PIN_MARKER in body:
        return PinProbeResult(BAD_PIN, http_status, BAD_PIN_MARKER, body)
    if 200 <= http_status < 300:
        # An empty completion is a reasoning-budget artefact, not a bad pin.
        return PinProbeResult(SERVED, http_status, "route accepted the model", "")
    if http_status in (401, 403):
        return PinProbeResult(
            UNKNOWN, http_status, "route rejected the probe credentials (auth)", body)
    if http_status == 404:
        return PinProbeResult(UNKNOWN, http_status, "no chat-completions endpoint at base_url", body)
    return PinProbeResult(UNKNOWN, http_status, f"route answered HTTP {http_status}", body)


def _chat_completions_url(base_url: str) -> str:
    root = str(base_url or "").strip()
    if root.startswith(":"):
        root = "http://127.0.0.1" + root
    elif root and "://" not in root:
        root = "http://" + root
    root = root.rstrip("/")
    if root.endswith("/chat/completions"):
        return root
    if root.endswith("/v1"):
        return root + "/chat/completions"
    return root + "/v1/chat/completions"


def probe_model_pin(
    base_url: str,
    model: str,
    *,
    api_key: str = "",
    timeout: float = DEFAULT_PROBE_TIMEOUT_S,
) -> PinProbeResult:
    """One tiny chat completion against *base_url* to see if the route serves *model*.

    Never raises: a transport failure is an ``unknown`` verdict, not a bad pin.
    """
    url = _chat_completions_url(base_url)
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
        "max_tokens": PROBE_MAX_TOKENS,
        "temperature": 0,
        "stream": False,
    }).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read(_MAX_BODY_CHARS).decode("utf-8", "replace")
            return classify_pin_response(int(getattr(response, "status", 200) or 200), body)
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read(_MAX_BODY_CHARS).decode("utf-8", "replace")
        except Exception:
            body = ""
        return classify_pin_response(int(getattr(exc, "code", 0) or 0), body)
    except Exception as exc:  # URLError, TimeoutError, OSError, ValueError, socket...
        return PinProbeResult(UNKNOWN, None, f"route unreachable ({type(exc).__name__})")


def _load_config_at(hermes_home: Optional[str]) -> Optional[dict]:
    """``load_config()`` under *hermes_home* (``None`` = the launching home)."""
    from hermes_cli.config import load_config

    token = None
    if hermes_home:
        from hermes_constants import set_hermes_home_override

        token = set_hermes_home_override(hermes_home)
    try:
        return load_config()
    except Exception:
        return None
    finally:
        if token is not None:
            try:
                from hermes_constants import reset_hermes_home_override

                reset_hermes_home_override(token)
            except Exception:
                pass


def _endpoint_from_config(config: dict, provider: str) -> tuple[str, str]:
    """``(base_url, api_key)`` for *provider* in one loaded config, else ``("", "")``."""
    from hermes_cli.config import get_env_value_prefer_dotenv

    wanted = str(provider).strip().lower()
    entries: list[dict] = []
    providers_cfg = config.get("providers")
    if isinstance(providers_cfg, dict):
        entry = providers_cfg.get(provider) or providers_cfg.get(wanted)
        if isinstance(entry, dict):
            entries.append(entry)
    custom = config.get("custom_providers")
    if isinstance(custom, list):
        for entry in custom:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or entry.get("provider_key") or "").strip().lower()
            if name == wanted:
                entries.append(entry)

    for entry in entries:
        base_url = str(entry.get("base_url") or entry.get("api") or entry.get("url") or "").strip()
        if not base_url:
            continue
        api_key = str(entry.get("api_key") or "").strip()
        if not api_key:
            key_env = str(entry.get("key_env") or entry.get("api_key_env") or "").strip()
            if key_env:
                api_key = (get_env_value_prefer_dotenv(key_env) or os.environ.get(key_env, "")).strip()
        return base_url, api_key
    return "", ""


def resolve_provider_endpoint(provider: str, *, hermes_home: Optional[str] = None) -> tuple[str, str]:
    """``(base_url, api_key)`` for *provider*, preferring the worker's own profile.

    ``hermes_home`` is the *worker's* home (the assignee profile): a worker is
    spawned with its profile's ``HERMES_HOME``, so that is the config whose
    provider entry decides where the model resolves. A profile that does not
    redeclare ``providers`` falls back to the launching home — the config that
    stamped the pin in the first place. Returns ``("", "")`` when the provider is
    unknown or carries no base_url; the caller treats that as "cannot check",
    never as a bad pin.
    """
    if not provider:
        return "", ""
    homes: list[Optional[str]] = [hermes_home] if hermes_home else []
    if None not in homes:
        homes.append(None)
    for home in homes:
        config = _load_config_at(home)
        if not isinstance(config, dict):
            continue
        base_url, api_key = _endpoint_from_config(config, provider)
        if base_url:
            return base_url, api_key
    return "", ""


def reset_pin_cache() -> None:
    """Drop every cached verdict (tests, and credential rotations)."""
    _PROBE_CACHE.clear()


def check_model_pin(
    model: Optional[str],
    provider: Optional[str],
    *,
    hermes_home: Optional[str] = None,
    probe: Callable[..., PinProbeResult] = probe_model_pin,
    timeout: float = DEFAULT_PROBE_TIMEOUT_S,
    now: Optional[float] = None,
) -> PinProbeResult:
    """Cached verdict for the (model, provider) pin. Never raises, never refuses blind.

    A pin is only reported as :data:`BAD_PIN` when a probe actually reached the
    route and it answered with the marker. Every other path — no provider, no
    base_url, unreachable route, auth rejection — is ``unknown`` and the
    dispatcher spawns the worker unchanged.
    """
    model = str(model or "").strip()
    provider = str(provider or "").strip()
    if not model:
        return PinProbeResult(SERVED, None, "no model pin")
    if not provider:
        return PinProbeResult(UNKNOWN, None, "no provider_override to check the model against")

    base_url, api_key = resolve_provider_endpoint(provider, hermes_home=hermes_home)
    if not base_url:
        return PinProbeResult(
            UNKNOWN, None, f"provider '{provider}' has no base_url in the profile config")

    key = (provider.lower(), model, base_url)
    ts = time.time() if now is None else now
    cached = _PROBE_CACHE.get(key)
    if cached is not None and cached[0] > ts:
        return cached[1]

    try:
        result = probe(base_url, model, api_key=api_key, timeout=timeout)
    except Exception as exc:  # a probe must never break a dispatch tick
        result = PinProbeResult(UNKNOWN, None, f"probe failed ({type(exc).__name__})")
    ttl = UNKNOWN_CACHE_TTL_S if not result.bad_pin else PROBE_CACHE_TTL_S
    _PROBE_CACHE[key] = (ts + ttl, result)
    return result


def refusal_message(model: str, provider: str, result: PinProbeResult) -> str:
    """The operator-facing refusal, including the route's own 400 body as evidence."""
    message = (
        f"model '{model}' is not served by route '{provider}' (bad pin) — "
        "fix the card's model_override or clear it to fall back to the "
        "profile/contract binding"
    )
    evidence = (result.error_body or "").strip()
    if evidence:
        message = f"{message}\nroute said: {evidence[:300]}"
    return message
