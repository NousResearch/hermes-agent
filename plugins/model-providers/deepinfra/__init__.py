"""DeepInfra provider profile (chat surface; image-gen/TTS/STT are wired via
their own plugin subsystems)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from agent.reasoning_effort import OPENAI_COMPAT_WIRE_EFFORTS, clamp_effort, requested_effort
from providers import register_provider
from providers.base import ProviderProfile

if TYPE_CHECKING:
    from agent.account_usage import AccountUsageSnapshot

# /usage: the payment checklist hangs off the account API, not off the chat route. The profile's
# base_url carries the inference route suffix (``/v1/openai``, or ``/v1`` under the
# Anthropic-compatible mode), so that suffix is dropped and any other path prefix is kept, which
# keeps a path-routed proxy on the right host. The payload also carries billing PII (holder name,
# postal address, email flag) alongside the figures read below — none of those keys are touched.
_BALANCE_ORIGIN = "https://api.deepinfra.com"
_BALANCE_TIMEOUT_S = 8.0  # under the shared hook bound, PLUGIN_USAGE_HOOK_DEADLINE_S (10 s)
_DEPLETED_LINE = "Status: access depleted — top up to restore"  # core's line, same wording here
_INFERENCE_ROUTE_SUFFIXES = ("/v1/openai", "/v1")


def _json_number(value: Any) -> float | None:
    """Coerce a JSON number or numeric string to a finite float; providers disagree on which they send.

    ``bool`` is rejected explicitly (it is an ``int`` subclass, and ``True`` is not a balance).
    ``OverflowError`` is caught with the conversion errors: a JSON integer big enough to overflow
    a float is malformed input, not a reason to traceback through ``/usage``.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _api_origin(base_url: Any) -> str | None:
    """Origin to fetch the payment checklist from, or ``None`` when it cannot be determined.

    ``base_url`` is the profile's *inference* route, so the inference suffix is stripped and any
    other path prefix is preserved. A configured-but-unusable value returns ``None`` instead of
    guessing DeepInfra's own host: the credential in hand may belong to whatever that value points
    at, and sending it to production would be a leak. Only an absent ``base_url`` falls back.
    """
    raw = str(base_url or "").strip()
    if not raw:
        return _BALANCE_ORIGIN
    parts = urlsplit(raw)
    if not parts.netloc:
        return None
    path = parts.path.rstrip("/")
    for suffix in _INFERENCE_ROUTE_SUFFIXES:
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    scheme = parts.scheme if parts.scheme in {"http", "https"} else "https"
    return f"{scheme}://{parts.netloc}{path}"


class _DeepInfraProfile(ProviderProfile):
    """DeepInfra profile with live vision-default discovery, so shared vision
    resolution in ``agent/auxiliary_client.py`` stays provider-agnostic."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, **context: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Map Hermes reasoning controls to DeepInfra's top-level ``reasoning_effort``.

        DeepInfra applies a per-model default when the field is absent (DeepSeek-V4.x off,
        GLM/Qwen-Thinking on), so ``none`` is the only working off switch and an unset effort
        is omitted rather than guessed. The core ``_supports_reasoning_extra_body`` allowlist
        does not know this host, so the transport always passes ``supports_reasoning=False``
        here — gating on it would make the method a permanent no-op (#111872).
        """
        if isinstance(reasoning_config, dict) and reasoning_config.get("enabled") is False:
            return {}, {"reasoning_effort": "none"}
        effort = requested_effort(reasoning_config)
        clamped = clamp_effort(effort, OPENAI_COMPAT_WIRE_EFFORTS)
        return ({}, {"reasoning_effort": clamped}) if clamped in OPENAI_COMPAT_WIRE_EFFORTS else ({}, {})

    def default_vision_model(self):  # type: ignore[override]
        """First vision-capable *chat* model from the live catalog, or None. Key-gated so a box
        without DEEPINFRA_API_KEY never pays the round-trip; requires the ``chat`` surface tag so
        an image-gen model carrying a ``vision`` tag can't be picked as a chat vision backend."""
        from agent.secret_scope import get_secret

        if not (get_secret("DEEPINFRA_API_KEY") or "").strip():
            return None
        try:
            from hermes_cli.models import _fetch_deepinfra_models_by_tag
            items = _fetch_deepinfra_models_by_tag("chat")
        except Exception:
            return None
        for item in items or []:
            metadata = item.get("metadata") or {}
            tags = metadata.get("tags") if isinstance(metadata, dict) else None
            if isinstance(tags, list) and "vision" in tags and item.get("id"):
                return item["id"]
        return None


    def fetch_account_usage(
        self, *, base_url: str | None = None, api_key: str | None = None
    ) -> AccountUsageSnapshot | None:
        """Prepaid credit held at DeepInfra, for ``/usage`` (the profile hook, not a core entry).

        DeepInfra keeps customer credit as a *negative* Stripe balance, so the sign is flipped
        once, here. Fail-soft on purpose: no credential, an origin that cannot be determined, a
        transport error, a non-200, or a payload without ``stripe_balance`` returns ``None`` and
        leaves ``/usage`` unchanged — a billing endpoint's hiccup must not traceback on every
        refresh. The one exception is a rejected credential (401/403), which is the user's to fix
        and so is reported through ``unavailable_reason`` rather than hidden. Only numeric billing
        fields are read; the payload's holder name, address and email flag are never touched.
        """
        from datetime import datetime, timezone

        import httpx

        from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow
        from hermes_cli.runtime_provider import resolve_runtime_provider

        runtime = resolve_runtime_provider(
            requested=self.name, explicit_base_url=base_url, explicit_api_key=api_key)
        token = str(runtime.get("api_key", "") or "").strip()
        if not token:
            return None
        origin = _api_origin(runtime.get("base_url") or base_url or self.base_url)
        if origin is None:
            return None
        try:
            with httpx.Client(timeout=_BALANCE_TIMEOUT_S) as client:
                response = client.get(
                    f"{origin}/payment/checklist?compute_owed=true",
                    headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
                )
            status = response.status_code
            payload = response.json() if status == 200 else None
        except Exception:
            return None
        if status in {401, 403}:
            return AccountUsageSnapshot(
                provider=self.name, source="payment_checklist", fetched_at=datetime.now(timezone.utc),
                unavailable_reason=(
                    f"DeepInfra rejected the credential (HTTP {status}) — check DEEPINFRA_API_KEY."
                ),
            )
        if status != 200 or not isinstance(payload, dict):
            return None  # a JSON array or scalar body is not a checklist; treat it as no data
        stripe_balance = _json_number(payload.get("stripe_balance"))
        if stripe_balance is None:
            return None
        # A positive Stripe balance means no credit is left (and possibly an amount owed); /usage
        # reports the credit floor, since the endpoint does not document the debt's units.
        balance = max(0.0, -stripe_balance)  # credit is held as a negative Stripe balance
        details = [f"Credits balance: ${balance:,.2f}"]
        recent = _json_number(payload.get("recent"))
        if recent is not None and recent > 0:
            details.append(f"Recent spend: ${recent:,.2f}")
        windows = []
        limit = _json_number(payload.get("limit"))
        if limit is not None and limit > 0:
            windows.append(AccountUsageWindow(
                label="Spending limit",
                used_percent=min(100.0, max(0.0, (1 - balance / limit) * 100)),
                detail=f"${balance:,.2f} of ${limit:,.2f} remaining",
            ))
        if payload.get("suspended"):
            details.append("Status: suspended — billing action required")
        elif balance <= 0:
            details.append(_DEPLETED_LINE)
        return AccountUsageSnapshot(
            provider=self.name, source="payment_checklist", fetched_at=datetime.now(timezone.utc),
            windows=tuple(windows), details=tuple(details),
        )


deepinfra = _DeepInfraProfile(
    name="deepinfra", aliases=("deep-infra", "deepinfra-ai"), display_name="DeepInfra",
    description="DeepInfra — 100+ open models, pay-per-use", signup_url="https://deepinfra.com/dash/api_keys",
    env_vars=("DEEPINFRA_API_KEY", "DEEPINFRA_BASE_URL"), base_url="https://api.deepinfra.com/v1/openai",
    auth_type="api_key",
    default_max_tokens=None,  # DeepInfra applies its documented per-model limit
    # The only hardcoded DeepInfra model: aux resolution is synchronous, so it
    # can't wait on a catalog round-trip. Everything else is discovered live.
    default_aux_model="deepseek-ai/DeepSeek-V4-Flash",
    # Empty on purpose: the live catalog is the source of truth; an empty picker
    # beats silently routing to a retired model.
    fallback_models=(),
)

register_provider(deepinfra)
