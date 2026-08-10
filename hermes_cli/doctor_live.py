"""``hermes doctor --live`` — opt-in bounded real-call tool-backend probes.

Design invariants:

- **Opt-in only.** These probes make real (cheap, metadata/read-only) network
  calls and may spend a trivial amount of quota. They run ONLY when the user
  passes ``hermes doctor --live``.
- **Bounded.** One probe per configured backend, sequential, each with a
  ~10s timeout (configurable via ``doctor.live_probe_timeout`` in
  config.yaml).
- **Read-only.** Metadata GETs only — no generation, no scrapes that spend
  credits, no state mutation anywhere.
- **Failure-isolated.** A probe crashing must never crash the doctor run;
  every probe is wrapped in a catch-all.
- **Configured-only.** Backends without credentials / config are skipped with
  a note, never failed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, List, Optional

from hermes_cli.doctor import (
    _section,
    check_fail,
    check_info,
    check_ok,
    check_warn,
)

DEFAULT_PROBE_TIMEOUT = 10.0

# Metadata-only endpoints. None of these spend generation credits.
FIRECRAWL_HEALTH_URL = "https://api.firecrawl.dev/v2/team/credit-usage"
FAL_MODELS_URL = "https://fal.ai/api/models?page=1"
OPENAI_MODELS_URL = "https://api.openai.com/v1/models"
GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"
ELEVENLABS_VOICES_URL = "https://api.elevenlabs.io/v1/voices"

# TTS/STT providers that never touch the network (nothing to probe).
_LOCAL_AUDIO_PROVIDERS = {"", "local", "edge", "neutts", "kittentts", "piper"}


@dataclass
class ProbeResult:
    """Outcome of one backend probe."""

    name: str
    status: str  # "pass" | "warn" | "fail" | "skip"
    detail: str = ""


# ---------------------------------------------------------------------------
# Small seams (monkeypatchable in tests, and single points of control).
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    try:
        from hermes_cli.config import load_config

        return load_config() or {}
    except Exception:
        return {}


def _http_get(url: str, headers: Optional[dict] = None,
              timeout: Optional[float] = None):
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
        for candidate in (HERMES_HOME / "node" / "bin",
                          HERMES_HOME / "node",
                          HERMES_HOME / "node_modules" / ".bin"):
            if shutil.which("agent-browser", path=str(candidate)):
                return True
    except Exception:
        pass
    return False


def _launch_browser_probe(timeout: float) -> tuple:
    """Launch a browser, open about:blank, close. Returns (ok, detail).

    Uses Playwright directly (what agent-browser drives underneath) so the
    probe owns the full lifecycle and always cleans up.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return (False, "playwright not installed")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True,
                                    timeout=timeout * 1000)
        try:
            page = browser.new_page()
            page.goto("about:blank", timeout=timeout * 1000)
        finally:
            browser.close()
    return (True, "launched + about:blank + closed")


def _probe_mcp_server(name: str, config: dict, timeout: float):
    """initialize + tools/list against one configured MCP server.

    Reuses the exact machinery behind ``hermes mcp test``.
    """
    from hermes_cli.mcp_config import _probe_single_server

    return _probe_single_server(name, config, connect_timeout=timeout)


# ---------------------------------------------------------------------------
# Per-backend probes. Each returns a ProbeResult and never raises upward
# beyond what run_live_checks' catch-all handles.
# ---------------------------------------------------------------------------

def _classify_http(name: str, resp, key_hint: str) -> ProbeResult:
    code = getattr(resp, "status_code", None)
    if code is not None and 200 <= code < 300:
        return ProbeResult(name, "pass", f"(HTTP {code})")
    if code in (401, 403):
        return ProbeResult(name, "fail",
                           f"(HTTP {code} — check {key_hint})")
    return ProbeResult(name, "fail", f"(HTTP {code})")


def _probe_firecrawl(timeout: float) -> ProbeResult:
    key = os.getenv("FIRECRAWL_API_KEY", "").strip()
    if not key:
        return ProbeResult("Firecrawl", "skip", "(not configured)")
    resp = _http_get(FIRECRAWL_HEALTH_URL,
                     headers={"Authorization": f"Bearer {key}"},
                     timeout=timeout)
    return _classify_http("Firecrawl", resp, "FIRECRAWL_API_KEY")


def _probe_fal(timeout: float) -> ProbeResult:
    key = os.getenv("FAL_KEY", "").strip()
    if not key:
        return ProbeResult("FAL", "skip", "(not configured)")
    # Metadata GET only — never a generation call.
    resp = _http_get(FAL_MODELS_URL,
                     headers={"Authorization": f"Key {key}"},
                     timeout=timeout)
    return _classify_http("FAL", resp, "FAL_KEY")


def _probe_browser(timeout: float) -> ProbeResult:
    if not _browser_available():
        return ProbeResult("Browser", "skip", "(not configured)")
    ok, detail = _launch_browser_probe(timeout)
    return ProbeResult("Browser", "pass" if ok else "fail", f"({detail})")


def _audio_provider_probe(kind: str, provider: str,
                          timeout: float) -> ProbeResult:
    """Shared TTS/STT metadata probe (voices/models list GET only)."""
    name = kind.upper()
    provider = (provider or "").strip().lower()
    if provider in _LOCAL_AUDIO_PROVIDERS:
        return ProbeResult(name, "skip",
                           f"(provider '{provider or 'local'}' — no remote "
                           "backend to probe)")

    probes = {
        "openai": (OPENAI_MODELS_URL, "OPENAI_API_KEY", "Bearer"),
        "groq": (GROQ_MODELS_URL, "GROQ_API_KEY", "Bearer"),
        "elevenlabs": (ELEVENLABS_VOICES_URL, "ELEVENLABS_API_KEY", "xi"),
    }
    entry = probes.get(provider)
    if entry is None:
        return ProbeResult(name, "skip",
                           f"(provider '{provider}' — no live probe "
                           "implemented)")
    url, env_var, scheme = entry
    key = os.getenv(env_var, "").strip()
    if not key:
        return ProbeResult(name, "warn",
                           f"(provider '{provider}' configured but "
                           f"{env_var} is not set)")
    if scheme == "xi":
        headers = {"xi-api-key": key}
    else:
        headers = {"Authorization": f"Bearer {key}"}
    resp = _http_get(url, headers=headers, timeout=timeout)
    result = _classify_http(name, resp, env_var)
    result.detail = f"({provider}) {result.detail}"
    return result


def _probe_tts(config: dict, timeout: float) -> ProbeResult:
    provider = ((config.get("tts") or {}).get("provider")) or ""
    return _audio_provider_probe("tts", provider, timeout)


def _probe_stt(config: dict, timeout: float) -> ProbeResult:
    provider = ((config.get("stt") or {}).get("provider")) or ""
    return _audio_provider_probe("stt", provider, timeout)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _report(result: ProbeResult, issues: List[str]) -> None:
    if result.status == "pass":
        check_ok(result.name, result.detail)
    elif result.status == "warn":
        check_warn(result.name, result.detail)
    elif result.status == "fail":
        check_fail(result.name, result.detail)
        issues.append(f"Live probe failed: {result.name} {result.detail}")
    else:  # skip
        check_info(f"{result.name} {result.detail} — skipped")


def _run_one(name: str, fn: Callable[[], ProbeResult],
             issues: List[str]) -> ProbeResult:
    """Run one probe with a catch-all so a crash never kills doctor."""
    try:
        result = fn()
    except TimeoutError as exc:
        result = ProbeResult(name, "fail", f"(timed out: {exc})")
    except Exception as exc:
        msg = str(exc) or exc.__class__.__name__
        if "time" in msg.lower():
            result = ProbeResult(name, "fail", f"(timed out: {msg})")
        else:
            result = ProbeResult(name, "fail", f"({msg})")
    _report(result, issues)
    return result


def run_live_checks(issues: List[str]) -> List[ProbeResult]:
    """Run one bounded, read-only probe per configured tool backend.

    Sequential by design (bounded, predictable output ordering). Appends a
    remediation line to ``issues`` for each failed probe. Skipped backends
    never fail and never append issues.
    """
    config = _load_config()
    try:
        timeout = float(
            (config.get("doctor") or {}).get("live_probe_timeout",
                                             DEFAULT_PROBE_TIMEOUT))
    except (TypeError, ValueError):
        timeout = DEFAULT_PROBE_TIMEOUT
    timeout = max(1.0, timeout)

    _section("Live Backend Probes (opt-in, real calls)")
    results: List[ProbeResult] = []

    results.append(_run_one(
        "Firecrawl", lambda: _probe_firecrawl(timeout), issues))
    results.append(_run_one(
        "FAL", lambda: _probe_fal(timeout), issues))
    results.append(_run_one(
        "Browser", lambda: _probe_browser(timeout), issues))

    servers = config.get("mcp_servers") or {}
    if isinstance(servers, dict) and servers:
        for name in sorted(servers):
            entry = servers[name]
            label = f"MCP: {name}"

            def _probe(n=name, e=entry) -> ProbeResult:
                if not isinstance(e, dict):
                    return ProbeResult(f"MCP: {n}", "skip",
                                       "(malformed config entry)")
                tools = _probe_mcp_server(n, e, timeout)
                return ProbeResult(f"MCP: {n}", "pass",
                                   f"({len(tools)} tool(s))")

            results.append(_run_one(label, _probe, issues))
    else:
        results.append(ProbeResult("MCP", "skip", "(no servers configured)"))
        _report(results[-1], issues)

    results.append(_run_one(
        "TTS", lambda: _probe_tts(config, timeout), issues))
    results.append(_run_one(
        "STT", lambda: _probe_stt(config, timeout), issues))

    return results


def maybe_run_live_checks(args, issues: List[str]):
    """Entry point called from ``run_doctor`` after the static checks.

    No-ops (returns None) unless the user explicitly passed ``--live``.
    A crash anywhere in the live subsystem must never break doctor.
    """
    if not getattr(args, "live", False):
        return None
    try:
        return run_live_checks(issues)
    except Exception as exc:  # catch-all: doctor must survive
        check_warn("Live backend probes crashed", f"({exc})")
        return None
