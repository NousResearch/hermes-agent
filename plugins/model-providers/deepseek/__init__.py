"""DeepSeek provider profile.

V4 defaults to thinking ON when ``extra_body.thinking`` is unset, and then
requires ``reasoning_content`` to be echoed back on later turns (HTTP 400 after
the first tool call otherwise). This profile sets ``thinking`` explicitly and
maps effort onto DeepSeek's ``reasoning_effort``; V3 models are left untouched.
Retired ``deepseek-chat``/``deepseek-reasoner`` IDs are remapped in
``hermes_cli.model_normalize`` before reaching here.
"""

import math
import re
from typing import Any

from agent.reasoning_effort import DEEPSEEK_V4_EFFORTS, DEEPSEEK_V4_OVERRIDES, thinking_toggle_extras
from providers import register_provider
from providers.base import ProviderProfile


def _amount(value: Any) -> float | None:
    """A balance field (sent as a decimal string) as a finite float, or None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


# Version-less canonical ids for thinking-capable DeepSeek models. The 2026-09 Flash
# refresh dropped the ``v<N>`` marker from the public id: ``GET /v1/models`` reports
# ``deepseek-flash`` and the API accepts it directly, so the generation check in
# ``build_api_kwargs_extras`` cannot recognise it.
_THINKING_CAPABLE_IDS: frozenset[str] = frozenset({"deepseek-flash"})


class DeepSeekProfile(ProviderProfile):
    """DeepSeek — extra_body.thinking + top-level reasoning_effort."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, model: str | None = None, **context
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        m = (model or "").strip().lower()
        # v4+ only; v3 excluded. Version-less canonicals (``deepseek-flash``) carry the
        # same thinking-mode contract but no ``v<N>`` prefix, so consult the id set too —
        # missing them makes Hermes omit ``thinking``, so the server defaults to on and
        # the user's thinking toggle / effort setting is silently ignored.
        versioned_v4_plus = m.startswith("deepseek-v") and not m.startswith("deepseek-v3")
        if not versioned_v4_plus and m not in _THINKING_CAPABLE_IDS:
            return {}, {}
        # Always set thinking explicitly (default enabled, matching the API default)
        # to avoid the reasoning_content echo trap on subsequent turns.
        extras, top_level = thinking_toggle_extras(
            reasoning_config, DEEPSEEK_V4_EFFORTS, DEEPSEEK_V4_OVERRIDES, always_emit_toggle=True
        )
        # ``{effort: "none"|"false"|"disabled"}`` is also an explicit disable:
        # UI/session surfaces report Off from the effort field even when
        # ``enabled`` is missing or still True (#107238).
        rc = reasoning_config
        effort = (rc.get("effort") or "").strip().lower() if isinstance(rc, dict) else ""
        if rc is not None and effort in {"none", "false", "disabled"}:
            return {"thinking": {"type": "disabled"}}, {}
        return extras, top_level

    def fetch_account_usage(self, *, base_url: str | None = None, api_key: str | None = None):
        """Account balance for /usage via ``GET /user/balance`` (api-docs.deepseek.com/api/get-user-balance).

        The endpoint sits beside ``/v1`` on the host this slot is configured for, so the key only
        ever goes to that host. One entry per currency, amounts as decimal strings; nothing readable
        → None rather than an empty gauge."""
        from datetime import UTC, datetime

        import httpx

        from agent.account_usage import AccountBalance, AccountUsageSnapshot
        from hermes_cli.runtime_provider import resolve_runtime_provider

        runtime = resolve_runtime_provider(requested=self.name, explicit_base_url=base_url, explicit_api_key=api_key)
        token = str(runtime.get("api_key", "") or "").strip()
        if not token:
            return None
        root = str(runtime.get("base_url", "") or "").rstrip("/").removesuffix("/v1")
        # Under the shared 10 s plugin-hook deadline (PLUGIN_USAGE_HOOK_DEADLINE_S).
        with httpx.Client(timeout=8.0) as client:
            response = client.get(f"{root}/user/balance",
                                  headers={"Authorization": f"Bearer {token}", "Accept": "application/json"})
            response.raise_for_status()
        payload = response.json() or {}
        infos = payload.get("balance_infos")
        balances = []
        for info in infos if isinstance(infos, list) else ():
            currency = str(info.get("currency") or "") if isinstance(info, dict) else ""
            total = _amount(info.get("total_balance")) if currency else None
            if total is not None and re.fullmatch(r"[A-Z]{3}", currency):
                balances.append(AccountBalance(label="Balance", amount=total, currency=currency))
        if not balances:
            return None
        depleted = ("Status: balance too low for API calls — top up to restore",)
        return AccountUsageSnapshot(provider=self.name, source="balance_api", fetched_at=datetime.now(UTC),
                                    title="Account balance", balances=tuple(balances),
                                    details=depleted if payload.get("is_available") is False else (), raw=payload)


deepseek = DeepSeekProfile(
    name="deepseek", aliases=("deepseek-chat", "deep-seek"), env_vars=("DEEPSEEK_API_KEY",), display_name="DeepSeek",
    description="DeepSeek — native DeepSeek API", signup_url="https://platform.deepseek.com/",
    fallback_models=("deepseek-v4-pro", "deepseek-flash"), base_url="https://api.deepseek.com/v1",
    default_aux_model="deepseek-flash",
    # Native API implements only ``json_object`` (https://api-docs.deepseek.com/guides/json_mode);
    # ``json_schema`` is a guaranteed HTTP 400 "This response_format type is unavailable now".
    unsupported_response_formats=("json_schema",),
)

register_provider(deepseek)
