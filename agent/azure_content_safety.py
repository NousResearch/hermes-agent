"""Azure AI Content Safety client (text:analyze + text:shieldPrompt).

Stateless, stdlib-only HTTP client for Azure AI Content Safety.  The
runtime calls into this module from the Azure Foundry guardrails block
to pre-screen prompts before forwarding them to the upstream model.

Why stdlib instead of ``azure-ai-contentsafety``?
    * Hermes core has no hard dependency on the Azure SDK; the
      ``[azure]`` extra is *opt-in* for users who want managed-identity
      auth or pinning.  Importing the SDK at module-import time would
      defeat the opt-in.
    * The Content Safety REST surface is two POST endpoints with a tiny
      JSON shape — ``urllib`` is the simpler answer.

Public API:
    * :func:`analyze_text` — POST ``contentsafety/text:analyze``
    * :func:`shield_prompt` — POST ``contentsafety/text:shieldPrompt``
    * :class:`ContentSafetyError` — base error type
    * :class:`ContentSafetyBlocked` — raised when severity ≥ threshold
      so the agent's existing :class:`agent.error_classifier.FailoverReason.content_filter`
      pattern matchers fire (the string repr contains
      ``"content safety blocked"``).

Conventions:
    * Single retry on HTTP 429 honouring ``Retry-After`` (capped at 30s).
    * The API key is *never* surfaced in exception messages — the
      :func:`_redact` helper strips it from any error payload before
      raising.  Callers can rely on logging the exception verbatim.
    * All network calls accept a ``timeout`` (default 10s).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

#: Default Content Safety REST API version.  The 2024-09-01 GA version
#: covers both ``text:analyze`` and ``text:shieldPrompt``.
DEFAULT_API_VERSION = "2024-09-01"

#: Default categories returned by ``text:analyze`` when none are
#: requested.  Mirrors Azure's portal default.
DEFAULT_CATEGORIES: tuple[str, ...] = ("Hate", "SelfHarm", "Sexual", "Violence")

#: Maximum time we'll honour from a 429 ``Retry-After`` header.  Beyond
#: this the client gives up and raises so the caller can fall back to
#: an alternate provider rather than blocking the agent loop.
MAX_RETRY_AFTER_SECONDS = 30

#: Single retry on transient 429 — Azure's CS service is bursty but
#: rarely sustained; one retry catches the common case without
#: doubling tail latency on real outages.
MAX_RETRIES = 1


class ContentSafetyError(RuntimeError):
    """Base class for all Content Safety client errors."""


class ContentSafetyBlocked(ContentSafetyError):
    """Raised when analysed text exceeds the configured severity threshold.

    The string representation deliberately contains
    ``"content safety blocked"`` so the agent's error classifier maps
    it to :class:`agent.error_classifier.FailoverReason.content_filter`.
    """

    def __init__(self, category: str, severity: int, threshold: int):
        self.category = category
        self.severity = severity
        self.threshold = threshold
        super().__init__(
            f"content safety blocked: category={category} "
            f"severity={severity} threshold={threshold}"
        )


@dataclass(frozen=True)
class _AnalyzeRequest:
    text: str
    categories: tuple[str, ...]
    output_type: str = "FourSeverityLevels"


def _redact(key: str, message: str) -> str:
    """Strip the API key from any string before it enters logs/exceptions."""
    if not key:
        return message
    return message.replace(key, "***REDACTED***")


def has_violation(
    result: Dict[str, Any],
    block_categories: Sequence[str],
    threshold: int,
) -> bool:
    """Return True if any analysed category in ``block_categories`` is at
    or above ``threshold``.  Pure helper — no side effects."""
    blocked = {c.lower() for c in block_categories}
    for entry in result.get("categoriesAnalysis", []) or []:
        if not isinstance(entry, dict):
            continue
        cat = str(entry.get("category") or "").lower()
        if cat not in blocked:
            continue
        sev = entry.get("severity")
        if isinstance(sev, int) and sev >= threshold:
            return True
    return False


def _build_url(endpoint: str, route: str, api_version: str) -> str:
    base = endpoint.rstrip("/")
    return f"{base}/contentsafety/{route}?api-version={api_version}"


def _post_json(
    url: str,
    payload: Dict[str, Any],
    *,
    key: str,
    timeout: float,
) -> Dict[str, Any]:
    """POST ``payload`` to ``url`` with a single 429 retry.  Raises
    :class:`ContentSafetyError` (with redacted message) on failure."""

    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "application/json",
    }

    last_error: Optional[BaseException] = None
    for attempt in range(MAX_RETRIES + 1):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                if not raw:
                    return {}
                try:
                    return json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ContentSafetyError(
                        f"Content Safety returned non-JSON body: {exc}"
                    ) from None
        except urllib.error.HTTPError as exc:
            status = exc.code
            try:
                err_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                err_body = ""
            err_body = _redact(key, err_body)
            if status == 429 and attempt < MAX_RETRIES:
                retry_after = _parse_retry_after(exc.headers.get("Retry-After"))
                if retry_after is not None:
                    time.sleep(min(retry_after, MAX_RETRY_AFTER_SECONDS))
                last_error = exc
                continue
            raise ContentSafetyError(
                f"Content Safety HTTP {status}: {err_body}"
            ) from None
        except urllib.error.URLError as exc:
            msg = _redact(key, str(exc.reason or exc))
            raise ContentSafetyError(f"Content Safety transport error: {msg}") from None
        except TimeoutError as exc:
            raise ContentSafetyError(
                f"Content Safety request timed out after {timeout}s"
            ) from None

    # Exhausted retries on 429.
    raise ContentSafetyError(
        f"Content Safety rate limited after {MAX_RETRIES + 1} attempt(s): {last_error}"
    )


def _parse_retry_after(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    value = value.strip()
    try:
        return float(value)
    except (TypeError, ValueError):
        # HTTP-date form is uncommon for Azure; ignore.
        return None


def analyze_text(
    text: str,
    *,
    endpoint: str,
    key: str,
    categories: Optional[Sequence[str]] = None,
    severity_threshold: Optional[int] = None,
    api_version: str = DEFAULT_API_VERSION,
    timeout: float = 10.0,
) -> Dict[str, Any]:
    """Analyse ``text`` against Azure AI Content Safety categories.

    When ``severity_threshold`` is provided and any returned category
    severity is ``>= threshold``, raises :class:`ContentSafetyBlocked`.
    Otherwise returns the parsed JSON response.
    """
    if not endpoint:
        raise ValueError("Content Safety endpoint is required.")
    if not key:
        raise ValueError("Content Safety key is required.")
    if not text:
        return {"categoriesAnalysis": []}

    cats = tuple(categories) if categories else DEFAULT_CATEGORIES
    payload: Dict[str, Any] = {
        "text": text,
        "categories": list(cats),
        "outputType": "FourSeverityLevels",
    }
    url = _build_url(endpoint, "text:analyze", api_version)
    result = _post_json(url, payload, key=key, timeout=timeout)

    if severity_threshold is not None:
        for entry in result.get("categoriesAnalysis", []) or []:
            if not isinstance(entry, dict):
                continue
            severity = entry.get("severity")
            category = str(entry.get("category") or "")
            if isinstance(severity, int) and severity >= severity_threshold:
                raise ContentSafetyBlocked(
                    category=category,
                    severity=severity,
                    threshold=severity_threshold,
                )
    return result


def shield_prompt(
    *,
    user_prompt: Optional[str],
    documents: Optional[Iterable[str]],
    endpoint: str,
    key: str,
    api_version: str = DEFAULT_API_VERSION,
    timeout: float = 10.0,
) -> Dict[str, Any]:
    """Call ``text:shieldPrompt`` to detect prompt-injection attacks.

    If the response indicates an attack on the user prompt or any
    document, raises :class:`ContentSafetyBlocked` with category
    ``"PromptShield"``.  Otherwise returns the parsed response.
    """
    if not endpoint:
        raise ValueError("Content Safety endpoint is required.")
    if not key:
        raise ValueError("Content Safety key is required.")

    payload: Dict[str, Any] = {}
    if user_prompt:
        payload["userPrompt"] = user_prompt
    docs = [d for d in (documents or []) if d]
    if docs:
        payload["documents"] = docs
    if not payload:
        return {}

    url = _build_url(endpoint, "text:shieldPrompt", api_version)
    result = _post_json(url, payload, key=key, timeout=timeout)

    user_attack = bool(
        (result.get("userPromptAnalysis") or {}).get("attackDetected")
    )
    docs_attack = any(
        bool(d.get("attackDetected"))
        for d in (result.get("documentsAnalysis") or [])
        if isinstance(d, dict)
    )
    if user_attack or docs_attack:
        raise ContentSafetyBlocked(
            category="PromptShield",
            severity=4,  # Shield is binary — surface as max severity.
            threshold=1,
        )
    return result


def _try_import_sdk() -> Any:
    """Lazy import of the optional ``azure-ai-contentsafety`` SDK.

    Currently unused at the call site (the stdlib path covers both
    REST endpoints), but exposed so future managed-identity support
    can opt into the SDK without changing the public API.  Returns
    ``None`` when the SDK is not installed.
    """
    try:  # pragma: no cover — exercised only when SDK is installed
        import azure.ai.contentsafety as _sdk  # type: ignore[import-not-found]
        return _sdk
    except ImportError:
        return None


__all__ = [
    "DEFAULT_API_VERSION",
    "DEFAULT_CATEGORIES",
    "ContentSafetyError",
    "ContentSafetyBlocked",
    "analyze_text",
    "shield_prompt",
]
