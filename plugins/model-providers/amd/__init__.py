"""AMD Radeon open-model service (developer.amd.com.cn).

OpenAI-compatible Chat Completions endpoint that fronts several upstream families
(DeepSeek / Qwen / GLM / MiMo / MiniCPM / Gemma). ``GET /v1/models`` returns an
OpenRouter-shaped catalogue; assistant messages may carry ``reasoning_content`` and
responses include a ``cost`` block.
"""

import httpx

from providers import register_provider
from providers.base import ProviderProfile
from datetime import UTC

# Same budget the dashboard's balance route gives every probe.
_BALANCE_TIMEOUT = 10.0


def _beijing(iso_instant: str) -> str:
    """``2026-11-05T09:36:12.414Z`` -> ``2026-11-05 17:36 GMT+8`` (the service resets on
    Beijing wall-clock time; the API reports a bare UTC instant with no timezone field)."""
    from datetime import datetime, timedelta, timezone

    text = (iso_instant or "").strip()
    if not text:
        return ""
    try:
        moment = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        return moment.astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M GMT+8")
    except ValueError:
        return text


def balance_probe(api_key: str) -> dict:
    """``GET /radeon/api/v1/usage`` → today's remaining quota.

    The ``daily_cost_*_usd`` fields are NOT money: they are the account's points, the
    ``_usd`` suffix is legacy naming and the service bills nothing — render them as pts.
    """
    response = httpx.get(
        "https://developer.amd.com.cn/radeon/api/v1/usage",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=_BALANCE_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()

    used = float(data.get("daily_cost_used_usd") or 0.0)
    limit = float(data.get("daily_cost_limit_usd") or 0.0)
    remaining = float(data.get("daily_cost_remaining_usd") or 0.0)
    today = data.get("today") if isinstance(data.get("today"), dict) else {}
    return {
        "primary": {"amount": remaining, "unit": "pts", "label": "Remaining today"},
        "breakdown": {"used": used, "limit": limit},
        "quota": {
            "used": used,
            "limit": limit,
            "remaining": remaining,
            "reset_at": _beijing(str(data.get("daily_reset_at") or "")),
        },
        "extra": {
            "rpm_limit": data.get("rpm_limit"),
            "requests_today": today.get("requests"),
            "errors_today": today.get("errors"),
            "tokens_today": today.get("total_tokens"),
        },
    }


amd = ProviderProfile(
    name="amd",
    aliases=("amd-radeon", "radeon"),
    display_name="AMD Radeon",
    description="AMD Radeon open models — DeepSeek / Qwen / GLM / MiMo direct API",
    signup_url="https://developer.amd.com.cn/",
    env_vars=("AMD_API_KEY",),
    base_url="https://developer.amd.com.cn/radeon/api/v1",
    # Only models advertising tools=true are listed — Hermes needs tool calls.
    fallback_models=(
        "DeepSeek-V4.1-Flash",
        "DeepSeek-V4-Flash",
        "Qwen3.8-Flash-Next",
        "Qwen3.8-27B",
        "GLM-5.3-Flash",
        "MiMo-V2.6-Flash",
        "Gemma-4-31B-IT",
        "MiniCPM-V-4.7-35B-A3B",
        "MiniCPM-V-4.7-1B",
        "MiniCPM5-2B",
    ),
    default_aux_model="Qwen3.8-27B",
    supports_vision=True,
    # Shared pool: /v1/models congestion (HTTP 429 "concurrency limit") is the normal
    # failure here and clears within seconds, so retry the same model for ~45 s
    # (15 tries × 3 s) before handing the turn to the fallback chain.
    rate_limit_retry_attempts=15,
    rate_limit_backoff_seconds=3.0,
    balance_probe=balance_probe,
)

register_provider(amd)
