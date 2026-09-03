"""Structured error-surface descriptors for UI clients (Desktop/TUI).

Maps the internal failure taxonomy (``agent.error_classifier.FailoverReason``
values carried in turn results as ``failure_reason``, or raw exceptions from
the turn dispatcher) onto a small, stable wire descriptor:

    {"layer": <ui layer>, "code": <specific code>, "retryable": <bool>}

The *layer* names which part of the stack failed, so clients can say
"Provider error" / "Gateway error" instead of toasting an opaque string and
leaving the user to guess whether the model, the gateway, or the app froze:

    provider   — the model/provider API rejected or failed the call
    endpoint   — a user-configured custom/local endpoint failed (transport)
    streaming  — the provider's SSE/stream connection dropped mid-turn
    auth       — authentication/authorization failed
    billing    — credits/quota wall (clients usually have a richer
                 billing_block descriptor; this is the fallback signal)
    gateway    — the local gateway/agent runtime itself errored
    runtime    — agent initialization / local environment failure
    disk       — local disk full / persistence failure

This module is intentionally dependency-light and NEVER raises: surfacing
diagnostics must not be able to break the error path it describes. Clients
treat the descriptor as advisory — an absent or partial descriptor falls
back to today's string-sniffing behavior (older backends keep working).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

# UI layers (wire values — stable contract with desktop/TUI clients).
LAYER_PROVIDER = "provider"
LAYER_ENDPOINT = "endpoint"
LAYER_STREAMING = "streaming"
LAYER_AUTH = "auth"
LAYER_BILLING = "billing"
LAYER_GATEWAY = "gateway"
LAYER_RUNTIME = "runtime"
LAYER_DISK = "disk"

# failure_reason (FailoverReason.value) → UI layer. Reasons not listed fall
# back to LAYER_PROVIDER: every FailoverReason is produced by classifying a
# provider API call, so "the provider call failed" is the honest default.
_REASON_TO_LAYER = {
    "auth": LAYER_AUTH,
    "auth_permanent": LAYER_AUTH,
    "billing": LAYER_BILLING,
    "billing_unverified": LAYER_BILLING,
}

# Transport-ish reasons: the failure is between us and the base_url, not a
# verdict the provider returned. On a custom/local endpoint these point at
# the user's endpoint config, so they surface as LAYER_ENDPOINT there.
_TRANSPORT_REASONS = {
    "timeout",
    "ssl_cert_verification",
}

# Reasons that are deterministic for the request — a bare "Retry" repeats the
# same failure, so clients shouldn't lead with it. Fallback only: results
# from current backends carry the classifier's own verdict in
# ``failure_retryable`` and never consult this set. Kept in sync with
# ``classify_api_error``'s retryable=False verdicts.
_NON_RETRYABLE_REASONS = {
    "auth",
    "auth_permanent",
    "billing",
    "billing_unverified",
    "content_policy_blocked",
    "provider_policy_blocked",
    "model_not_found",
    "format_error",
    "ssl_cert_verification",
}

# Providers whose base_url is user-supplied rather than a known vendor —
# transport failures against these are endpoint-config problems.
_CUSTOM_ENDPOINT_PROVIDERS = {
    "custom",
    "local",
    "llama.cpp",
    "llamacpp",
    "ollama",
    "lmstudio",
    "vllm",
}

# Message fragments that mark a mid-stream connection drop. Deliberately
# narrow: these strings come from our own retry-exhaustion summaries and the
# OpenAI SDK's stream-abort errors. Each fragment must be unambiguous as a
# RAW SUBSTRING — a short generic token here silently matches ordinary
# provider prose and misroutes the layer (see _SSE_TOKEN_RE below).
_STREAM_DROP_FRAGMENTS = (
    "stream connection",
    "peer closed connection",
    "incomplete chunked read",
    "connection broken",
    "stream ended prematurely",
    "mid-stream",
)

# "sse" must match as a standalone token only: as a bare substring it hits
# ordinary provider prose ("processed", "surpassed", "dismissed") and
# misroutes provider-layer failures to the streaming layer.
# Input is pre-lowered by _looks_like_stream_drop(), so re.IGNORECASE is
# intentionally omitted — keep the lowering and this pattern in sync.
_SSE_TOKEN_RE = re.compile(r"\bsse\b")

# Exception modules that indicate the failure came from an API/transport call
# (vs. a bug in our own dispatcher code, which is a gateway-layer failure).
# Covers every SDK family our provider adapters raise from: OpenAI-compatible
# (openai/httpx/httpcore), Anthropic, Bedrock (botocore/boto3), Google
# (google.*/grpc), plus raw transports (requests/aiohttp/ssl/socket/urllib).
_API_EXC_MODULE_PREFIXES = (
    "openai",
    "httpx",
    "httpcore",
    "anthropic",
    "botocore",
    "boto3",
    "google",
    "grpc",
    "requests",
    "aiohttp",
    "ssl",
    "socket",
    "urllib",
)


def _is_custom_endpoint(provider: Optional[str]) -> bool:
    p = (provider or "").strip().lower()
    return p in _CUSTOM_ENDPOINT_PROVIDERS or p.startswith("custom:")


def _looks_like_stream_drop(message: str) -> bool:
    msg = message.lower()
    if any(fragment in msg for fragment in _STREAM_DROP_FRAGMENTS):
        return True
    return bool(_SSE_TOKEN_RE.search(msg))


def _surface(
    layer: str,
    code: str,
    retryable: bool,
    provider: str = "",
    model: str = "",
) -> dict:
    out = {"layer": layer, "code": code, "retryable": bool(retryable)}
    # The failing session's identity, captured at classification time so
    # clients report the model/provider that actually failed — not whatever
    # the foreground composer points at when a button is clicked later.
    if provider:
        out["provider"] = provider
    if model:
        out["model"] = model
    return out


def build_error_surface_from_result(
    result: Any, provider: str = "", model: str = ""
) -> Optional[dict]:
    """Descriptor for a returned-error turn result (``failed=True`` dicts).

    Reads the ``failure_reason`` the conversation loop already stamps
    (a ``FailoverReason.value``) plus the error text, and maps them onto a
    UI layer. Returns None when the result carries no failure signal.
    """
    try:
        if not isinstance(result, dict):
            return None
        error_text = str(result.get("error") or "")
        reason = str(result.get("failure_reason") or "").strip()
        if not error_text and not reason:
            return None

        # Disk-full wins outright: the fix (free space) is unrelated to the
        # provider stack, and hermes_state owns the pattern list.
        try:
            from hermes_state import is_disk_full_error

            if error_text and is_disk_full_error(error_text):
                return _surface(LAYER_DISK, "disk_full", False, provider, model)
        except Exception:  # pragma: no cover - defensive import guard
            pass

        if result.get("billing_block") or reason in ("billing", "billing_unverified"):
            return _surface(LAYER_BILLING, reason or "billing", False, provider, model)

        if not reason:
            # Failed result without a classified reason (legacy paths).
            if _looks_like_stream_drop(error_text):
                return _surface(LAYER_STREAMING, "stream_drop", True, provider, model)
            return _surface(LAYER_PROVIDER, "unknown", True, provider, model)

        layer = _REASON_TO_LAYER.get(reason)
        if layer is None:
            if reason in _TRANSPORT_REASONS and _is_custom_endpoint(provider):
                layer = LAYER_ENDPOINT
            elif _looks_like_stream_drop(error_text):
                layer = LAYER_STREAMING
            else:
                layer = LAYER_PROVIDER
        # Prefer the classifier's own retry verdict when the result carries it
        # (conversation_loop stamps ``failure_retryable`` next to
        # ``failure_reason``); the reason-set fallback covers older results.
        retryable = result.get("failure_retryable")
        if not isinstance(retryable, bool):
            retryable = reason not in _NON_RETRYABLE_REASONS
        return _surface(layer, reason, retryable, provider, model)
    except Exception:  # pragma: no cover — never break the error path
        logger.debug("error_surface: result classification failed", exc_info=True)
        return None


def build_error_surface_from_exception(
    exc: BaseException, provider: str = "", model: str = ""
) -> Optional[dict]:
    """Descriptor for an exception that escaped the turn dispatcher.

    API/transport exceptions are classified through the real
    ``classify_api_error`` pipeline (same taxonomy as the retry loop);
    anything else is a gateway-layer failure — a bug or environment problem
    in our own dispatcher, not a provider verdict.
    """
    try:
        message = str(exc) or type(exc).__name__

        try:
            from hermes_state import is_disk_full_error

            if is_disk_full_error(exc):
                return _surface(LAYER_DISK, "disk_full", False, provider, model)
        except Exception:  # pragma: no cover - defensive import guard
            pass

        exc_module = type(exc).__module__ or ""
        api_like = exc_module.split(".")[0] in _API_EXC_MODULE_PREFIXES or hasattr(
            exc, "status_code"
        )

        if not api_like or not isinstance(exc, Exception):
            return _surface(LAYER_GATEWAY, type(exc).__name__, True, provider, model)

        from agent.error_classifier import classify_api_error

        classified = classify_api_error(exc, provider=provider, model=model)
        reason = classified.reason.value

        synthetic = {
            "error": classified.message or message,
            "failure_reason": reason,
        }
        surface = build_error_surface_from_result(
            synthetic, provider=provider, model=model
        )
        if surface is not None:
            surface["retryable"] = bool(classified.retryable)
        return surface
    except Exception:  # pragma: no cover — never break the error path
        logger.debug("error_surface: exception classification failed", exc_info=True)
        return None
