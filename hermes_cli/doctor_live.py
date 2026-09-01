"""``hermes doctor --live`` — opt-in bounded real-call tool-backend probes.

Opt-in only: these probes make real (cheap, metadata/read-only) network calls and may spend a
trivial amount of quota. They run ONLY when the user passes ``hermes doctor --live``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, List, Optional

from hermes_cli.browser_runtime import chromium_executable
from hermes_cli.doctor import _section, check_info
from hermes_cli.doctor_report import check_fail, check_ok, check_warn
from utils import normalize_proxy_env_vars

DEFAULT_PROBE_TIMEOUT = 10.0

# Metadata-only endpoints (none spend generation credits): name -> (url, env var, auth scheme).
_KEYED_PROBES = {
    "Firecrawl": ("https://api.firecrawl.dev/v2/team/credit-usage", "FIRECRAWL_API_KEY", "Bearer"),
    "FAL": ("https://fal.ai/api/models?page=1", "FAL_KEY", "Key"),
}
# TTS/STT providers that never touch the network (nothing to probe).
_LOCAL_AUDIO_PROVIDERS = {"", "local", "edge", "neutts", "kittentts", "piper"}
_AUDIO_PROBES = {
    "openai": ("https://api.openai.com/v1/models", "OPENAI_API_KEY", "Bearer"),
    "groq": ("https://api.groq.com/openai/v1/models", "GROQ_API_KEY", "Bearer"),
    "elevenlabs": ("https://api.elevenlabs.io/v1/voices", "ELEVENLABS_API_KEY", "xi"),
}


@dataclass
class ProbeResult:
    """Outcome of one backend probe."""

    name: str
    status: str  # "pass" | "warn" | "fail" | "skip"
    detail: str = ""


# ── Small seams (monkeypatchable in tests, and single points of control) ──

def _http_get(url: str, headers: Optional[dict] = None, timeout: Optional[float] = None):
    """Single HTTP GET seam for all metadata probes."""
    import httpx
    # Sanitize at the seam (idempotent, no-op on a clean env): a bracketed-IPv6 NO_PROXY entry
    # ([::1], Clash Verge/mihomo) makes this bare trust_env client raise InvalidURL at
    # construction (#118159), which _run_one's catch-all would dress up as backend downtime.
    normalize_proxy_env_vars()
    return httpx.get(url, headers=headers or {}, timeout=timeout)


def _browser_available() -> bool:
    """Is there a Chromium for browser_exec's local launcher?"""
    return chromium_executable() is not None


def _launch_browser_probe(timeout: float) -> tuple:
    """Launch Chromium through browser_exec's own launcher, wait for its CDP endpoint, stop it.
    Returns (ok, detail); the probe owns the whole lifecycle and always cleans up."""
    from tools.browser_launcher import launch_chromium, stop_browser

    key = "doctor-live-probe"
    try:
        browser, error = launch_chromium(key, ephemeral=True, timeout=timeout)
    finally:
        stop_browser(key)
    if browser is None:
        return (False, error or "launch failed")
    return (True, "launched + CDP ready + stopped")


def _probe_mcp_server(name: str, config: dict, timeout: float):
    """initialize + tools/list against one configured MCP server."""
    from hermes_cli.mcp_config import _probe_single_server
    return _probe_single_server(name, config, connect_timeout=timeout)


# ── Per-backend probes. Each returns a ProbeResult; _run_one's catch-all handles crashes. ──

def _classify_http(name: str, resp, key_hint: str) -> ProbeResult:
    code = getattr(resp, "status_code", None)
    if code is not None and 200 <= code < 300:
        return ProbeResult(name, "pass", f"(HTTP {code})")
    return ProbeResult(name, "fail", f"(HTTP {code} — check {key_hint})" if code in (401, 403) else f"(HTTP {code})")


def _keyed_probe(name: str, url: str, env_var: str, scheme: str, timeout: float) -> ProbeResult:
    """Metadata GET authenticated by one env var (never a generation call)."""
    key = os.getenv(env_var, "").strip()
    if not key:
        return ProbeResult(name, "skip", "(not configured)")
    resp = _http_get(url, headers={"Authorization": f"{scheme} {key}"}, timeout=timeout)
    return _classify_http(name, resp, env_var)


def _probe_browser(timeout: float) -> ProbeResult:
    if not _browser_available():
        return ProbeResult("Browser", "skip", "(not configured)")
    ok, detail = _launch_browser_probe(timeout)
    return ProbeResult("Browser", "pass" if ok else "fail", f"({detail})")


def _probe_audio(kind: str, config: dict, timeout: float) -> ProbeResult:
    """Shared TTS/STT metadata probe (voices/models list GET only)."""
    name = kind.upper()
    provider = (((config.get(kind) or {}).get("provider")) or "").strip().lower()
    if provider in _LOCAL_AUDIO_PROVIDERS:
        return ProbeResult(name, "skip", f"(provider '{provider or 'local'}' — no remote backend to probe)")
    if provider not in _AUDIO_PROBES:
        return ProbeResult(name, "skip", f"(provider '{provider}' — no live probe implemented)")
    url, env_var, scheme = _AUDIO_PROBES[provider]
    key = os.getenv(env_var, "").strip()
    if not key:
        return ProbeResult(name, "warn", f"(provider '{provider}' configured but {env_var} is not set)")
    headers = {"xi-api-key": key} if scheme == "xi" else {"Authorization": f"Bearer {key}"}
    result = _classify_http(name, _http_get(url, headers=headers, timeout=timeout), env_var)
    result.detail = f"({provider}) {result.detail}"
    return result


# ── Orchestration ──

_REPORTERS = {"pass": check_ok, "warn": check_warn, "fail": check_fail}


def _report(result: ProbeResult, issues: list[str]) -> None:
    reporter = _REPORTERS.get(result.status)
    if reporter is None:  # skip
        check_info(f"{result.name} {result.detail} — skipped")
        return
    reporter(result.name, result.detail)
    if result.status == "fail":
        issues.append(f"Live probe failed: {result.name} {result.detail}")


def _run_one(name: str, fn: Callable[[], ProbeResult], issues: list[str]) -> ProbeResult:
    """Run one probe with a catch-all so a crash never kills doctor."""
    try:
        result = fn()
    except TimeoutError as exc:
        result = ProbeResult(name, "fail", f"(timed out: {exc})")
    except Exception as exc:
        msg = str(exc) or exc.__class__.__name__
        result = ProbeResult(name, "fail", f"(timed out: {msg})" if "time" in msg.lower() else f"({msg})")
    _report(result, issues)
    return result


def run_live_checks(issues: list[str]) -> list[ProbeResult]:
    """Run one bounded, read-only probe per configured tool backend — sequential by design (predictable output
    ordering). Appends a remediation line to ``issues`` per failed probe; skipped backends never append."""
    from hermes_cli.config import load_config_readonly
    config = load_config_readonly()
    try:
        timeout = float((config.get("doctor") or {}).get("live_probe_timeout", DEFAULT_PROBE_TIMEOUT))
    except (TypeError, ValueError):
        timeout = DEFAULT_PROBE_TIMEOUT
    timeout = max(1.0, timeout)
    _section("Live Backend Probes (opt-in, real calls)")
    results: list[ProbeResult] = [
        _run_one(name, lambda n=name, spec=spec: _keyed_probe(n, *spec, timeout), issues)
        for name, spec in _KEYED_PROBES.items()
    ]
    results.append(_run_one("Browser", lambda: _probe_browser(timeout), issues))
    servers = config.get("mcp_servers") or {}
    if isinstance(servers, dict) and servers:
        for name in sorted(servers):
            def _probe(n=name, e=servers[name]) -> ProbeResult:
                if not isinstance(e, dict):
                    return ProbeResult(f"MCP: {n}", "skip", "(malformed config entry)")
                return ProbeResult(f"MCP: {n}", "pass", f"({len(_probe_mcp_server(n, e, timeout))} tool(s))")
            results.append(_run_one(f"MCP: {name}", _probe, issues))
    else:
        results.append(ProbeResult("MCP", "skip", "(no servers configured)"))
        _report(results[-1], issues)

    for probe_result in _probe_configured_models(config, timeout):
        results.append(probe_result)
        _report(probe_result, issues)

    for kind in ("tts", "stt"):
        results.append(_run_one(kind.upper(), lambda k=kind: _probe_audio(k, config, timeout), issues))
    return results


# ---------------------------------------------------------------------------
# Configured-model existence (primary + fallback chain)
# ---------------------------------------------------------------------------
# doctor's static block validates that ``model.provider`` NAMES a real provider
# and that slug style suits it, plus a hardcoded retired-model list. It never
# asks the provider whether the configured model is actually served, and it
# never looks at ``fallback_providers`` at all.
#
# That combination hides a specific, silent failure: a fallback entry whose
# provider is valid and whose model simply is not there any more. The primary
# path masks it until the moment the fallback is needed, and then every call
# returns a hard 400 (observed in the wild: ``lmstudio / hermes-4-14b`` against
# an LM Studio that had been re-provisioned with different models). A dead
# fallback is worse than a dead primary, because it is invisible right up to
# the moment it is load-bearing.


def _served_model_ids(base_url: str, api_key: Optional[str],
                      timeout: float) -> Optional[set]:
    """Model ids an OpenAI-compatible endpoint reports, or None if unverifiable.

    None means "could not establish the truth" (unreachable, auth-gated, not an
    OpenAI-compatible surface, unparseable body). It never means "empty", so a
    caller can never mistake a failed probe for proof of absence.
    """
    root = (base_url or "").strip().rstrip("/")
    if not root:
        return None
    if not root.endswith("/v1"):
        root = f"{root}/v1"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        resp = _http_get(f"{root}/models", headers=headers, timeout=timeout)
    except Exception:
        return None
    if getattr(resp, "status_code", 0) != 200:
        return None
    try:
        payload = resp.json()
    except Exception:
        return None
    entries = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return None
    ids = {
        str(e.get("id")).strip()
        for e in entries
        if isinstance(e, dict) and e.get("id")
    }
    return ids or None


def _configured_model_routes(config: dict) -> list:
    """(label, provider, model, base_url) for the primary and every fallback.

    The fallback chain comes from ``get_fallback_chain`` rather than reading
    ``fallback_providers`` directly, so legacy ``fallback_model`` entries and
    de-duplication behave exactly as they do at call time.
    """
    routes = []
    model_section = config.get("model")
    if isinstance(model_section, dict):
        primary = str(model_section.get("default") or "").strip()
        if primary:
            routes.append((
                "primary",
                str(model_section.get("provider") or "").strip(),
                primary,
                str(model_section.get("base_url") or "").strip() or None,
            ))
    try:
        from hermes_cli.fallback_config import get_fallback_chain

        chain = get_fallback_chain(config)
    except Exception:
        chain = []
    for i, entry in enumerate(chain):
        if not isinstance(entry, dict):
            continue
        model = str(entry.get("model") or "").strip()
        provider = str(entry.get("provider") or "").strip()
        if not model or not provider:
            continue
        routes.append((
            f"fallback[{i}]",
            provider,
            model,
            str(entry.get("base_url") or "").strip() or None,
        ))
    return routes


def _pool_base_url(provider: str) -> Optional[str]:
    """Endpoint the credential pool would use for ``provider``, if any.

    A fallback entry usually carries only ``provider`` and ``model``; the
    endpoint lives with the credential. Without this, the entries most likely
    to rot (local providers whose model set changes) could not be checked.
    """
    try:
        from agent.credential_pool import load_pool

        pool = load_pool()
    except Exception:
        return None
    try:
        creds = pool.get(provider) if isinstance(pool, dict) else None
        if creds is None and hasattr(pool, "for_provider"):
            creds = pool.for_provider(provider)
        for cred in creds or []:
            url = getattr(cred, "runtime_base_url", None)
            url = url() if callable(url) else url
            if url:
                return str(url)
            if isinstance(cred, dict) and cred.get("base_url"):
                return str(cred["base_url"])
    except Exception:
        return None
    return None


def _probe_configured_models(config: dict, timeout: float) -> list:
    """One ProbeResult per configured route. Absence is only ever a `fail`."""
    routes = _configured_model_routes(config)
    if not routes:
        return [ProbeResult("Models", "skip", "(no model configured)")]

    results = []
    for label, provider, model, base_url in routes:
        name = f"Model {label}: {provider or '?'}/{model}"
        endpoint = base_url or _pool_base_url(provider)
        if not endpoint:
            results.append(ProbeResult(
                name, "skip", "(no endpoint resolvable for this provider)"))
            continue
        served = _served_model_ids(endpoint, None, timeout)
        if served is None:
            results.append(ProbeResult(
                name, "warn",
                "(could not read /v1/models — unreachable, auth-gated, or "
                "not an OpenAI-compatible surface)"))
            continue
        # A provider-prefixed slug is configured as "provider/model" but served
        # under the bare id, so compare both spellings before calling it absent.
        wanted = {model, model.split("/", 1)[-1]}
        lowered = {m.lower() for m in served}
        if any(w in served or w.lower() in lowered for w in wanted):
            results.append(ProbeResult(name, "pass", f"({len(served)} served)"))
        else:
            sample = ", ".join(sorted(served)[:3])
            results.append(ProbeResult(
                name, "fail",
                f"not served by {provider} — available: {sample}"
                + (" …" if len(served) > 3 else "")))
    return results


def maybe_run_live_checks(args, issues: list[str]):
    """Called from ``run_doctor`` after the static checks; no-op (None) unless ``--live`` was passed.
    A crash anywhere in the live subsystem must never break doctor."""
    if not getattr(args, "live", False):
        return None
    try:
        return run_live_checks(issues)
    except Exception as exc:  # catch-all: doctor must survive
        check_warn("Live backend probes crashed", f"({exc})")
        return None
