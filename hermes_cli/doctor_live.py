"""``hermes doctor --live`` — opt-in bounded real-call tool-backend probes.

Opt-in only: these probes make real (cheap, metadata/read-only) network calls and may spend a
trivial amount of quota. They run ONLY when the user passes ``hermes doctor --live``.
"""

from __future__ import annotations

import json
import os
import uuid
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

def _http_get(url: str, headers: Optional[dict] = None, timeout: Optional[float] = None):
    """Single HTTP GET seam for all metadata probes."""
    import httpx
    return httpx.get(url, headers=headers or {}, timeout=timeout)


def _browser_available() -> bool:
    """Is the local browser automation backend (agent-browser) installed?"""
    try:
        from tools.browser_tool_install import _find_agent_browser
        return bool(_find_agent_browser(validate=False))
    except Exception:
        return False


def _launch_browser_probe(timeout: float) -> tuple:
    """Open and close an isolated tab through the built-in agent-browser backend."""
    from tools.browser_tool_lifecycle import _cleanup_single_browser_session
    from tools.browser_tool_session import _run_browser_command

    session = f"doctor-{uuid.uuid4().hex}"
    try:
        result = _run_browser_command(session, "open", ["about:blank"], timeout=timeout)
        return (bool(result.get("success")),
                "agent-browser opened about:blank" if result.get("success")
                else str(result.get("error") or "agent-browser navigation failed"))
    finally:
        _cleanup_single_browser_session(session)


def _launch_browser_use_probe(timeout: float) -> tuple:
    """Exercise the same Browser Use CLI route that the agent's browser_exec tool uses."""
    from tools.browser_use_cli import browser_exec

    session = f"doctor-{uuid.uuid4().hex}"
    code = ("# Validate local browser automation\n"
            "opened = False\n"
            "try:\n"
            "    new_tab('about:blank')\n"
            "    opened = True\n"
            "    print('HERMES_DOCTOR_READY=' + str(js('document.readyState')))\n"
            "finally:\n"
            "    if opened:\n"
            "        close_tab()\n")
    result = json.loads(browser_exec(code, session=session, task_id=session, timeout_s=int(timeout)))
    ok = result.get("success") and "HERMES_DOCTOR_READY=complete" in result.get("output", "")
    detail = ("browser_exec opened about:blank and read DOM" if ok else
              str(result.get("error") or result.get("stderr") or result.get("output") or
                  "browser_exec did not confirm DOM access")[:250])
    return (bool(ok), detail)


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
    from tools.browser_use_cli import is_browser_use_cli_mode

    if is_browser_use_cli_mode():
        ok, detail = _launch_browser_use_probe(timeout)
        return ProbeResult("Browser", "pass" if ok else "fail", f"({detail})")
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
    from hermes_cli.config import load_config_readonly
    config = load_config_readonly()
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
    for kind in ("tts", "stt"):
        results.append(_run_one(kind.upper(), lambda k=kind: _probe_audio(k, config, timeout), issues))
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
