"""API connectivity probes for ``hermes doctor`` (split out of ``doctor.py``).

Every probe is a pure function: one HTTP/SDK call returning a ``ProbeResult`` with the row(s) to
print and issue strings to append. No printing inside workers — the caller prints in submission order.
"""

from __future__ import annotations

import concurrent.futures
import functools
import os
import sys
from typing import NamedTuple

from hermes_cli.colors import Colors, color
from hermes_cli.models import _HERMES_USER_AGENT
from hermes_constants import OPENROUTER_MODELS_URL
from utils import base_url_host_matches

_APIKEY_PROVIDERS_CACHE: list | None = None


class ProbeResult(NamedTuple):
    label: str
    lines: list  # [(glyph, label, detail)]
    issues: list


_GLYPH = {"ok": ("✓", Colors.GREEN), "warn": ("⚠", Colors.YELLOW), "fail": ("✗", Colors.RED)}


def _row(name: str, status: str, detail: str = "", issues: list | None = None, label: str | None = None) -> ProbeResult:
    glyph, col = _GLYPH[status]
    return ProbeResult(name, [(color(glyph, col), name if label is None else label, color(detail, Colors.DIM) if detail else "")],
                       list(issues or []))


def _skip(name: str) -> ProbeResult:
    return ProbeResult(name, [], [])


def _has_healthy_oauth_fallback_for_apikey_provider(provider_label: str) -> bool:
    """True when a failed direct API-key probe is non-blocking because the same provider family's OAuth
    runtime path is already healthy: the failed row is still shown, but not promoted into the summary."""
    getter = {"minimax": "get_minimax_oauth_auth_status", "xai": "get_xai_oauth_auth_status"}.get((provider_label or "").strip().lower())
    if not getter:
        return False
    try:
        from hermes_cli import auth
        return bool((getattr(auth, getter)() or {}).get("logged_in"))
    except Exception:
        return False


def _build_apikey_providers_list() -> list:
    """Build the API-key provider health-check list once and cache it.

    Tuple format: (name, env_vars, default_url, base_env, supports_models_endpoint). Base list augmented
    with any ProviderProfile with auth_type="api_key" not already present — adding
    plugins/model-providers/<name>/ is sufficient to get into doctor.
    """
    _static = [
        ("Z.AI / GLM",      ("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"), "https://api.z.ai/api/paas/v4/models", "GLM_BASE_URL", True),
        ("Kimi / Moonshot",  ("KIMI_API_KEY",),                              "https://api.moonshot.ai/v1/models",   "KIMI_BASE_URL", True),
        ("StepFun Step Plan", ("STEPFUN_API_KEY",),                          "https://api.stepfun.ai/step_plan/v1/models", "STEPFUN_BASE_URL", True),
        ("Kimi / Moonshot (China)", ("KIMI_CN_API_KEY",),                    "https://api.moonshot.cn/v1/models",   None, True),
        ("Arcee AI",         ("ARCEEAI_API_KEY",),                           "https://api.arcee.ai/api/v1/models",  "ARCEE_BASE_URL", True),
        ("GMI Cloud",        ("GMI_API_KEY",),                               "https://api.gmi-serving.com/v1/models", "GMI_BASE_URL", True),
        ("DeepSeek",         ("DEEPSEEK_API_KEY",),                          "https://api.deepseek.com/v1/models",  "DEEPSEEK_BASE_URL", True),
        ("Hugging Face",     ("HF_TOKEN",),                                  "https://router.huggingface.co/v1/models", "HF_BASE_URL", True),
        ("NVIDIA NIM",       ("NVIDIA_API_KEY",),                            "https://integrate.api.nvidia.com/v1/models", "NVIDIA_BASE_URL", True),
        ("Alibaba/DashScope", ("DASHSCOPE_API_KEY",),                        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/models", "DASHSCOPE_BASE_URL", True),
        # MiniMax global: /v1 endpoint supports /models.
        ("MiniMax",          ("MINIMAX_API_KEY",),                           "https://api.minimax.io/v1/models",    "MINIMAX_BASE_URL", True),
        # MiniMax CN: /v1 endpoint does NOT support /models (returns 404).
        ("MiniMax (China)",  ("MINIMAX_CN_API_KEY",),                        "https://api.minimaxi.com/v1/models",  "MINIMAX_CN_BASE_URL", False),
        ("Vercel AI Gateway", ("AI_GATEWAY_API_KEY",),                       "https://ai-gateway.vercel.sh/v1/models", "AI_GATEWAY_BASE_URL", True),
        ("Kilo Code",        ("KILOCODE_API_KEY",),                          "https://api.kilo.ai/api/gateway/models", "KILOCODE_BASE_URL", True),
        ("OpenCode Zen",     ("OPENCODE_ZEN_API_KEY",),                      "https://opencode.ai/zen/v1/models",  "OPENCODE_ZEN_BASE_URL", True),
        # OpenCode Go has no shared /models endpoint; skip the health check.
        ("OpenCode Go",      ("OPENCODE_GO_API_KEY",),                       None,                                  "OPENCODE_GO_BASE_URL", False),
    ]
    _known_names = {t[0] for t in _static}
    # Providers with a dedicated health check (custom headers/auth): skip their pluggable profiles so
    # the generic Bearer loop doesn't run a duplicate, broken check (Anthropic needs x-api-key).
    _dedicated_canonical = {"anthropic", "openrouter", "bedrock"}
    # Canonical profile names of the static rows, so profiles without a display_name don't duplicate.
    _known_canonical = {
        "zai", "kimi-coding", "stepfun", "kimi-coding-cn", "arcee", "gmi", "deepseek", "huggingface", "nvidia",
        "alibaba", "minimax", "minimax-cn", "ai-gateway", "kilocode", "opencode-zen", "opencode-go",
    } | _dedicated_canonical
    try:
        from providers import list_providers
        from providers.base import ProviderProfile as _PP
        try:
            from hermes_cli.providers import normalize_provider as _normalize_provider
        except Exception:  # pragma: no cover - normalization is best-effort
            def _normalize_provider(_name: str) -> str:
                return (_name or "").strip().lower()
        for _pp in list_providers():
            if not isinstance(_pp, _PP) or _pp.auth_type != "api_key" or not _pp.env_vars:
                continue
            _label = _pp.display_name or _pp.name
            if _label in _known_names or _pp.name in _known_canonical:
                continue
            if {_normalize_provider(a) for a in (_pp.name, *(_pp.aliases or ()))} & _dedicated_canonical:
                continue
            # Key vars vs base-URL vars: the first found value goes out as Authorization: Bearer, never a URL.
            _is_url = lambda v: v.endswith("_BASE_URL") or v.endswith("_URL")  # noqa: E731
            _key_vars = tuple(v for v in _pp.env_vars if not _is_url(v))
            if not _key_vars:
                continue
            _base_var = next((v for v in _pp.env_vars if _is_url(v)), None)
            _models_url = (_pp.models_url or (_pp.base_url.rstrip("/") + "/models")) if _pp.base_url else None
            _static.append((_label, _key_vars, _models_url, _base_var, getattr(_pp, "supports_health_check", True)))
    except Exception:
        pass
    return _static


# HTTP status -> (detail, issue) for the OpenRouter probe; anything else is a generic HTTP failure.
_OPENROUTER_STATUS = {
    401: ("(invalid API key)", "Check OPENROUTER_API_KEY in .env"),
    402: ("(out of credits — payment required)",
          "OpenRouter account has insufficient credits. "
          "Fix: run 'hermes config set model.provider <provider>' "
          "to switch providers, or fund your OpenRouter account "
          "at https://openrouter.ai/settings/credits"),
    429: ("(rate limited)", "OpenRouter rate limit hit — consider switching to a different provider or waiting"),
}


def _probe_openrouter() -> ProbeResult:
    name = "OpenRouter API"
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        return _row(name, "warn", "(not configured)")
    try:
        import httpx
        r = httpx.get(OPENROUTER_MODELS_URL, headers={"Authorization": f"Bearer {key}"}, timeout=10)
    except Exception as e:
        return _row(name, "fail", f"({e})", ["Check network connectivity"])
    if r.status_code == 200:
        return _row(name, "ok")
    detail, issue = _OPENROUTER_STATUS.get(r.status_code, (f"(HTTP {r.status_code})", None))
    return _row(name, "fail", detail, [issue] if issue else None)


def _probe_anthropic() -> ProbeResult:
    name = "Anthropic API"
    from hermes_cli.auth import get_anthropic_key
    key = get_anthropic_key()
    if not key:
        return _skip(name)
    try:
        import httpx
        from agent.anthropic_adapter import _COMMON_BETAS, _OAUTH_ONLY_BETAS, _CONTEXT_1M_BETA
        from agent.anthropic_credentials import _is_oauth_token
        is_oauth = _is_oauth_token(key)
        headers = {"anthropic-version": "2023-06-01", **({"Authorization": f"Bearer {key}", "anthropic-beta": ",".join(_COMMON_BETAS + _OAUTH_ONLY_BETAS)}
                                                         if is_oauth else {"x-api-key": key})}
        url = "https://api.anthropic.com/v1/models"
        r = httpx.get(url, headers=headers, timeout=10)
        # OAuth subscriptions without 1M context reject with 400 "long context beta is not yet available";
        # retry once with that beta stripped so doctor doesn't falsely report Anthropic as unreachable.
        if is_oauth and r.status_code == 400 and "long context beta" in r.text.lower() and "not yet available" in r.text.lower():
            headers["anthropic-beta"] = ",".join([b for b in _COMMON_BETAS if b != _CONTEXT_1M_BETA] + list(_OAUTH_ONLY_BETAS))
            r = httpx.get(url, headers=headers, timeout=10)
    except Exception as e:
        return _row(name, "warn", f"({e})")
    return _row(name, *{200: ("ok",), 401: ("fail", "(invalid API key)")}.get(r.status_code, ("warn", "(couldn't verify)")))


def _probe_apikey_provider(pname, env_vars, default_url, base_env, supports_health_check) -> ProbeResult:
    key = next((k for k in (os.getenv(ev, "") for ev in env_vars) if k), "")
    if not key:
        return _skip(pname)
    label = pname.ljust(20)
    if not supports_health_check:
        return _row(pname, "ok", "(key configured)", label=label)
    try:
        import httpx
        base, url, headers = _apikey_request(key, base_env, default_url)
        r = httpx.get(url, headers=headers, timeout=10)
        if pname == "Alibaba/DashScope" and not base and r.status_code == 401:
            r = httpx.get("https://dashscope.aliyuncs.com/compatible-mode/v1/models", headers=headers, timeout=10)
    except Exception as e:
        return _row(pname, "warn", f"({e})", label=label)
    if r.status_code == 401:
        return _row(pname, "fail", "(invalid API key)", [f"Check {env_vars[0]} in .env"], label=label)
    return _row(pname, "ok", label=label) if r.status_code == 200 else _row(pname, "warn", f"(HTTP {r.status_code})", label=label)


def _apikey_request(key: str, base_env, default_url) -> tuple:
    """(effective base, models URL, headers) for a generic Bearer-auth probe, with the per-vendor rewrites."""
    base = os.getenv(base_env, "") if base_env else ""
    # Kimi Code keys (sk-kimi-) → api.kimi.com/coding/v1 (OpenAI-compat surface exposing /models).
    if not base and key.startswith("sk-kimi-"):
        base = "https://api.kimi.com/coding/v1"
    # Anthropic-compat endpoints (/anthropic, api.kimi.com/coding sans /v1) lack /models — use the OpenAI-compat /v1 surface.
    if base and base.rstrip("/").endswith("/anthropic"):
        from agent.auxiliary_client import _to_openai_base_url
        base = _to_openai_base_url(base)
    if base_url_host_matches(base, "api.kimi.com") and base.rstrip("/").endswith("/coding"):
        base = base.rstrip("/") + "/v1"
    url = (base.rstrip("/") + "/models") if base else default_url
    headers = {"Authorization": f"Bearer {key}", "User-Agent": _HERMES_USER_AGENT}
    if base_url_host_matches(base, "api.kimi.com"):
        headers["User-Agent"] = "claude-code/0.1.0"
    # Google's Generative Language API rejects ``Authorization: Bearer <api-key>`` with 401
    # ACCESS_TOKEN_TYPE_UNSUPPORTED (reserved for OAuth 2 tokens); plain keys use ``x-goog-api-key``.
    if url and base_url_host_matches(url, "generativelanguage.googleapis.com"):
        headers.pop("Authorization", None)
        headers["x-goog-api-key"] = key
    return base, url, headers


def _probe_bedrock() -> ProbeResult:
    name = "AWS Bedrock"
    try:
        from agent.bedrock_adapter import has_aws_credentials, resolve_aws_auth_env_var, resolve_bedrock_region
    except ImportError:
        return _skip(name)
    if not has_aws_credentials():
        return _skip(name)
    auth_var, region, label = resolve_aws_auth_env_var(), resolve_bedrock_region(), name.ljust(20)
    try:
        import boto3
        from botocore.config import Config as _BotoConfig
        # Trim retries so a transient failure doesn't pad the doctor run by 30+ seconds.
        client = boto3.client("bedrock", region_name=region, config=_BotoConfig(connect_timeout=5, read_timeout=10, retries={"max_attempts": 1}))
        n = len(client.list_foundation_models().get("modelSummaries", []))
        return _row(name, "ok", f"({auth_var}, {region}, {n} models)", label=label)
    except ImportError:
        pip = f"{sys.executable} -m pip install boto3"
        return _row(name, "warn", f"(boto3 not installed — {pip})", [f"Install boto3 for Bedrock: {pip}"], label=label)
    except Exception as e:
        err_name = type(e).__name__
        return _row(name, "warn", f"({err_name}: {e})", [f"AWS Bedrock: {err_name} — check IAM permissions for bedrock:ListFoundationModels"], label=label)


def _probe_azure_entra() -> ProbeResult:
    """Probe Azure Foundry Entra ID auth, parallel to ``_probe_bedrock``.

    Skipped unless the active config has ``model.provider: azure-foundry`` AND ``model.auth_mode: entra_id``
    — we don't probe the token-service / CLI chain for plain API-key Azure. Bounded by a 10s timeout.
    """
    name = "Azure Foundry (Entra ID)"
    label = name.ljust(28)
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        model_cfg = cfg.get("model") if isinstance(cfg, dict) else {}
        if not isinstance(model_cfg, dict):
            return _skip(name)
        if [str(model_cfg.get(k) or "").strip().lower() for k in ("provider", "auth_mode")] != ["azure-foundry", "entra_id"]:
            return _skip(name)
    except Exception:
        return _skip(name)
    try:
        from agent.azure_identity_adapter import (
            EntraIdentityConfig, SCOPE_AI_AZURE_DEFAULT, describe_active_credential, has_azure_identity_installed,
        )
    except Exception as exc:
        return _row(name, "warn", f"(adapter import failed: {exc})", [f"Azure Foundry adapter import failed: {exc}"], label=label)
    if not has_azure_identity_installed():
        return _row(name, "warn", "(azure-identity not installed)", [f"Install azure-identity: {sys.executable} -m pip install azure-identity"], label=label)
    entra_cfg = model_cfg.get("entra") or {}
    scope = (str(entra_cfg.get("scope") or "").strip() if isinstance(entra_cfg, dict) else "") or SCOPE_AI_AZURE_DEFAULT
    info = describe_active_credential(config=EntraIdentityConfig(scope=scope), timeout_seconds=10.0)
    if info.get("ok"):
        tag = ", ".join(info.get("env_sources") or []) or "default credential chain"
        return _row(name, "ok", f"({tag}, scope={scope})", label=label)
    err = info.get("error") or "credential chain exhausted"
    hint = info.get("hint") or "Run `az login`, set AZURE_TENANT_ID/AZURE_CLIENT_ID/AZURE_CLIENT_SECRET, or attach a managed identity to this VM."
    return _row(name, "warn", f"({err})", [f"Azure Foundry Entra: {err}. {hint}"], label=label)


def build_probes() -> list:
    """(label, callable) pairs in display order."""
    global _APIKEY_PROVIDERS_CACHE
    if _APIKEY_PROVIDERS_CACHE is None:
        _APIKEY_PROVIDERS_CACHE = _build_apikey_providers_list()
    return [
        ("OpenRouter API", _probe_openrouter), ("Anthropic API", _probe_anthropic),
        # functools.partial binds each row's args so every callable keeps its own provider.
        *((row[0], functools.partial(_probe_apikey_provider, *row)) for row in _APIKEY_PROVIDERS_CACHE),
        ("AWS Bedrock", _probe_bedrock), ("Azure Foundry (Entra ID)", _probe_azure_entra),
    ]


def run_probes(probes: list) -> list:
    """Run every probe in a thread pool; results in submission order.

    Probes are independent HTTP calls (series cost ~5s wall, 2s of it boto3's IMDS lookup);
    parallel collapses the section to roughly the slowest single probe without changing the output.
    """
    # Disable boto3's EC2 instance-metadata probe (169.254.169.254, multi-second timeout off-EC2). Set on the
    # parent thread before submitting so it never races a worker; has_aws_credentials() already gates on real creds.
    _imds_prev = os.environ.get("AWS_EC2_METADATA_DISABLED")
    os.environ["AWS_EC2_METADATA_DISABLED"] = "true"
    try:
        # 8 workers is plenty — each probe is one HTTP call plus a TLS handshake.
        with concurrent.futures.ThreadPoolExecutor(max_workers=8, thread_name_prefix="doctor-probe") as ex:
            return [f.result() for f in [ex.submit(fn) for _, fn in probes]]
    finally:
        if _imds_prev is None:
            os.environ.pop("AWS_EC2_METADATA_DISABLED", None)
        else:
            os.environ["AWS_EC2_METADATA_DISABLED"] = _imds_prev
