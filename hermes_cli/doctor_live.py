"""``hermes doctor --live`` — opt-in bounded real-call tool-backend probes.

Opt-in only: these probes make real (cheap, metadata/read-only) network calls and may spend a
trivial amount of quota. They run ONLY when the user passes ``hermes doctor --live``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, List, Optional

from hermes_cli.doctor import _section, check_info
from hermes_cli.doctor_report import check_fail, check_ok, check_warn

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

def _load_config() -> dict:
    try:
        from hermes_cli.config import load_config
        return load_config() or {}
    except Exception:
        return {}


def _http_get(url: str, headers: Optional[dict] = None, timeout: Optional[float] = None):
    """Single HTTP GET seam for all metadata probes."""
    import httpx
    return httpx.get(url, headers=headers or {}, timeout=timeout)


def _browser_available() -> bool:
    """Is the local browser automation backend (agent-browser) installed?"""
    import shutil
    if shutil.which("agent-browser"):
        return True
    try:
        from hermes_cli.doctor import HERMES_HOME, PROJECT_ROOT
        if (PROJECT_ROOT / "node_modules" / "agent-browser").exists():
            return True
        for candidate in (HERMES_HOME / "node" / "bin", HERMES_HOME / "node", HERMES_HOME / "node_modules" / ".bin"):
            if shutil.which("agent-browser", path=str(candidate)):
                return True
    except Exception:
        pass
    # agent-browser resolves lazily via npx on the default install, invisible to the PATH/node_modules
    # probes above. Mirror the rung hermes_cli.doctor uses so this probe can't diverge from it, including
    # the Termux carve-out (bare npx is too fragile to advertise as ready there).
    try:
        from tools.browser_tool_install import _find_agent_browser, _is_npx_agent_browser_sentinel, _requires_real_termux_browser_install
        browser_cmd = _find_agent_browser(validate=False)
    except Exception:
        return False
    return _is_npx_agent_browser_sentinel(browser_cmd) and not _requires_real_termux_browser_install(browser_cmd)


def _launch_browser_probe(timeout: float) -> tuple:
    """Launch a browser, open about:blank, close. Returns (ok, detail). Uses Playwright directly (what
    agent-browser drives underneath) so the probe owns the full lifecycle and always cleans up."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return (False, "playwright not installed")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, timeout=timeout * 1000)
        try:
            browser.new_page().goto("about:blank", timeout=timeout * 1000)
        finally:
            browser.close()
    return (True, "launched + about:blank + closed")


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


def _report(result: ProbeResult, issues: List[str]) -> None:
    reporter = _REPORTERS.get(result.status)
    if reporter is None:  # skip
        check_info(f"{result.name} {result.detail} — skipped")
        return
    reporter(result.name, result.detail)
    if result.status == "fail":
        issues.append(f"Live probe failed: {result.name} {result.detail}")


def _run_one(name: str, fn: Callable[[], ProbeResult], issues: List[str]) -> ProbeResult:
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


def run_live_checks(issues: List[str]) -> List[ProbeResult]:
    """Run one bounded, read-only probe per configured tool backend — sequential by design (predictable output
    ordering). Appends a remediation line to ``issues`` per failed probe; skipped backends never append."""
    config = _load_config()
    try:
        timeout = float((config.get("doctor") or {}).get("live_probe_timeout", DEFAULT_PROBE_TIMEOUT))
    except (TypeError, ValueError):
        timeout = DEFAULT_PROBE_TIMEOUT
    timeout = max(1.0, timeout)
    _section("Live Backend Probes (opt-in, real calls)")
    results: List[ProbeResult] = [
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
    """Model ids an OpenAI-compatible endpoint reports, or None if unverifiable."""
    return _fetch_served_models(base_url, api_key, timeout)[0]


def _fetch_served_models(base_url: str, api_key: Optional[str],
                         timeout: float) -> tuple:
    """(model ids, reason) — ids are None when the truth could not be established.

    None means "could not establish the truth" (unreachable, auth-gated, not an
    OpenAI-compatible surface, unparseable body). It never means "empty", so a
    caller can never mistake a failed probe for proof of absence.
    """
    root = (base_url or "").strip().rstrip("/")
    if not root:
        return None, "no endpoint"
    if not root.endswith("/v1"):
        root = f"{root}/v1"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        resp = _http_get(f"{root}/models", headers=headers, timeout=timeout)
    except Exception as exc:
        return None, f"unreachable ({type(exc).__name__})"
    status = getattr(resp, "status_code", 0)
    if status != 200:
        return None, f"HTTP {status}"
    try:
        payload = resp.json()
    except Exception:
        return None, "unparseable body"
    entries = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return None, "no model list in the response"
    ids = {
        str(e.get("id")).strip()
        for e in entries
        if isinstance(e, dict) and e.get("id")
    }
    if not ids:
        return None, "empty model list"
    return ids, "ok"


def _configured_model_routes(config: dict) -> list:
    """(label, provider, model, base_url, api_key) for the primary and every fallback.

    ``api_key`` is the route's own credential when the entry carries one
    (inline ``api_key`` or ``key_env``, resolved exactly as the runtime does
    via ``resolve_entry_api_key``); None otherwise, so the probe falls back to
    the provider's pooled credential.

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
                None,
            ))
    try:
        from hermes_cli.fallback_config import get_fallback_chain, resolve_entry_api_key

        chain = get_fallback_chain(config)
    except Exception:
        chain = []
        resolve_entry_api_key = None
    for i, entry in enumerate(chain):
        if not isinstance(entry, dict):
            continue
        model = str(entry.get("model") or "").strip()
        provider = str(entry.get("provider") or "").strip()
        if not model or not provider:
            continue
        route_key = None
        if resolve_entry_api_key is not None:
            try:
                route_key = resolve_entry_api_key(entry)
            except Exception:
                route_key = None
        routes.append((
            f"fallback[{i}]",
            provider,
            model,
            str(entry.get("base_url") or "").strip() or None,
            route_key,
        ))
    return routes


def _pool_credential(provider: str) -> tuple:
    """(base_url, api_key, due_for_refresh) of the pooled credential for ``provider``.

    ``due_for_refresh`` is the pool's OWN verdict (``_entry_needs_refresh``):
    an OAuth access token the runtime would refresh before its next call. The
    probe must not send such a token (the provider would reject it and the
    result would read as a bad credential) and must not refresh it either —
    for xAI / Codex / Nous the refresh token is single-use, so a refresh from
    ``doctor`` could rotate the grant out from under the running gateway.

    A fallback entry usually carries only ``provider`` and ``model``; the
    endpoint AND the key live with the pooled credential (``load_pool`` takes
    the provider — calling it bare raises, which an earlier version swallowed,
    so nothing pooled was ever resolved). ``peek()`` is the runtime's own
    choice: the current credential, else the first available one. For OAuth
    providers ``access_token`` is the bearer the runtime sends, so ``/v1/models``
    is read with the same credential inference uses.
    """
    try:
        from agent.credential_pool import load_pool
        pool = load_pool(provider)
        cred = pool.peek()
    except Exception:
        return None, None, False
    if cred is None:
        return None, None, False
    url = getattr(cred, "runtime_base_url", None)
    url = url() if callable(url) else url
    if not url:
        url = getattr(cred, "base_url", None)
    key = getattr(cred, "access_token", None)
    try:
        due_for_refresh = bool(pool._entry_needs_refresh(cred))
    except Exception:
        due_for_refresh = False
    return (str(url) if url else None), (str(key) if key else None), due_for_refresh


def _registry_api_key(provider: str) -> Optional[str]:
    """Env-var credential for an API-key provider, when nothing is pooled.

    Follows ``PROVIDER_REGISTRY[provider].api_key_env_vars`` in priority order
    through ``agent.secret_scope.get_secret`` (profile-scoped under a
    multiplexed gateway, plain ``os.environ`` otherwise).
    """
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY
        from agent.secret_scope import get_secret
    except Exception:
        return None
    pconfig = PROVIDER_REGISTRY.get(provider)
    for var in getattr(pconfig, "api_key_env_vars", ()) or ():
        try:
            value = (get_secret(var) or "").strip()
        except Exception:
            value = ""
        if value:
            return value
    return None


def _probe_configured_models(config: dict, timeout: float) -> list:
    """One ProbeResult per configured route. Absence is only ever a `fail`."""
    routes = _configured_model_routes(config)
    if not routes:
        return [ProbeResult("Models", "skip", "(no model configured)")]

    results = []
    for label, provider, model, base_url, route_key in routes:
        name = f"Model {label}: {provider or '?'}/{model}"
        pool_url, pool_key, pool_due_for_refresh = _pool_credential(provider)
        endpoint = base_url or pool_url
        if not endpoint:
            results.append(ProbeResult(
                name, "skip", "(no endpoint resolvable for this provider)"))
            continue
        # Same order the runtime resolves a route's credential: the entry's
        # own key, then the pooled credential, then the provider's env vars.
        usable_pool_key = None if pool_due_for_refresh else pool_key
        api_key = route_key or usable_pool_key or _registry_api_key(provider)
        if api_key is None and pool_due_for_refresh:
            results.append(ProbeResult(
                name, "warn",
                f"(pooled {provider} credential is due for refresh; the runtime "
                "refreshes it on first use — not probed, a refresh from doctor "
                "could rotate a single-use grant)"))
            continue
        served, reason = _fetch_served_models(endpoint, api_key, timeout)
        if served is None:
            credential = "with the resolved credential" if api_key else "no credential resolved"
            results.append(ProbeResult(
                name, "warn",
                f"(could not read /v1/models: {reason}, {credential})"))
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


def maybe_run_live_checks(args, issues: List[str]):
    """Called from ``run_doctor`` after the static checks; no-op (None) unless ``--live`` was passed.
    A crash anywhere in the live subsystem must never break doctor."""
    if not getattr(args, "live", False):
        return None
    try:
        return run_live_checks(issues)
    except Exception as exc:  # catch-all: doctor must survive
        check_warn("Live backend probes crashed", f"({exc})")
        return None


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

ELEVENLABS_VOICES_URL = "https://api.elevenlabs.io/v1/voices"

FAL_MODELS_URL = "https://fal.ai/api/models?page=1"

FIRECRAWL_HEALTH_URL = "https://api.firecrawl.dev/v2/team/credit-usage"

GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"

OPENAI_MODELS_URL = "https://api.openai.com/v1/models"
# ---- END PLUGIN-COMPAT ----
