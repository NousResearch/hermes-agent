"""Read-only capability probe for llama.cpp endpoints.

Server kinds:
- "llama-swap":   GET <root>/running answers 200 with {"running": [...]}
                  (llama-swap-only endpoint)
- "llama-server": /running answers 404 (bare llama-server has no such route)
- "unknown":      endpoint unreachable or unrecognized

Template capabilities come from /props: accepted reasoning_effort values,
values the template remaps, its default, and the enable_thinking toggle.

SAFETY: on llama-swap every model-dispatched route STARTS a non-resident
model - /props?model= included. fetch_props therefore only queries /props
for ids present in /running; a non-resident id yields None with no HTTP
request.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

logger = logging.getLogger("providers.llamacpp")

DEFAULT_TIMEOUT = 3.0
_CACHE_TTL = 60.0
_cache: dict[tuple[str, str], tuple[float, "ProbeResult"]] = {}


@dataclass(frozen=True)
class ServerInfo:
    kind: str  # "llama-swap" | "llama-server" | "unknown"
    running: tuple[str, ...]  # resident model ids (always () for bare/unknown)


@dataclass(frozen=True)
class TemplateCaps:
    has_reasoning_effort: bool
    accepted_efforts: tuple[str, ...]  # literals the template's check allows
    remapped_efforts: dict[str, str]  # input -> value the template rewrites to
    default_effort: str | None
    supports_thinking_toggle: bool  # template consults enable_thinking
    tolerated_efforts: tuple[str, ...]  # accepted + remapped inputs


@dataclass(frozen=True)
class ProbeResult:
    server: ServerInfo
    props: dict | None
    caps: TemplateCaps | None


def _http_get_json(url: str, timeout: float) -> tuple[int, Any]:
    """GET url -> (status, parsed JSON or None). Raises on transport errors."""
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            body = resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, None
    try:
        return status, json.loads(body)
    except Exception:
        return status, None


def _server_root(base_url: str | None) -> str:
    root = (base_url or "").strip().rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")].rstrip("/")
    return root


def detect_server(base_url: str, timeout: float = DEFAULT_TIMEOUT) -> ServerInfo:
    root = _server_root(base_url)
    if not root:
        return ServerInfo(kind="unknown", running=())
    try:
        status, payload = _http_get_json(f"{root}/running", timeout)
    except Exception as exc:
        logger.debug("llamacpp probe: %s/running unreachable: %s", root, exc)
        return ServerInfo(kind="unknown", running=())
    if status == 200 and isinstance(payload, dict) and "running" in payload:
        running = tuple(
            str(entry.get("model"))
            for entry in payload.get("running") or []
            if isinstance(entry, dict) and entry.get("model")
        )
        return ServerInfo(kind="llama-swap", running=running)
    if status == 404:
        return ServerInfo(kind="llama-server", running=())
    return ServerInfo(kind="unknown", running=())


def fetch_props(
    base_url: str,
    model: str | None = None,
    *,
    server: ServerInfo | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict | None:
    root = _server_root(base_url)
    if server is None:
        server = detect_server(base_url, timeout=timeout)
    if server.kind == "llama-server":
        url = f"{root}/props"
    elif server.kind == "llama-swap":
        if not model or model not in server.running:
            logger.info(
                "llamacpp probe: skipping /props for non-resident model %r "
                "(llama-swap would start it)",
                model,
            )
            return None
        url = f"{root}/props?model={quote(str(model))}"
    else:
        return None
    try:
        status, payload = _http_get_json(url, timeout)
    except Exception as exc:
        logger.debug("llamacpp probe: GET %s failed: %s", url, exc)
        return None
    if status != 200 or not isinstance(payload, dict):
        logger.debug("llamacpp probe: GET %s -> %s", url, status)
        return None
    return payload


_EFFORT_VAR = r"\w*reasoning_effort\w*"


def parse_template_caps(chat_template: str | None) -> TemplateCaps:
    tpl = chat_template or ""
    has = "reasoning_effort" in tpl
    accepted: list[str] = []
    remapped: dict[str, str] = {}
    default = None
    if has:
        m = re.search(
            r"reasoning_effort\s*\|\s*default\(\s*['\"]([A-Za-z0-9_-]+)['\"]", tpl
        )
        if m:
            default = m.group(1)
        for m in re.finditer(
            rf"({_EFFORT_VAR})\s+(?:not\s+)?in\s*\(([^)]*)\)", tpl
        ):
            for lit in re.findall(r"['\"]([A-Za-z0-9_-]+)['\"]", m.group(2)):
                if lit not in accepted:
                    accepted.append(lit)
        for m in re.finditer(
            rf"if\s+({_EFFORT_VAR})\s*==\s*['\"]([A-Za-z0-9_-]+)['\"]\s*-?%\}}"
            rf"\s*\{{%-?\s*set\s+\1\s*=\s*['\"]([A-Za-z0-9_-]+)['\"]",
            tpl,
        ):
            if m.group(2) not in accepted:
                remapped[m.group(2)] = m.group(3)
    tolerated = tuple(accepted) + tuple(k for k in remapped if k not in accepted)
    return TemplateCaps(
        has_reasoning_effort=has,
        accepted_efforts=tuple(accepted),
        remapped_efforts=remapped,
        default_effort=default,
        supports_thinking_toggle="enable_thinking" in tpl,
        tolerated_efforts=tolerated,
    )


def probe_model(
    base_url: str,
    model: str | None,
    timeout: float = DEFAULT_TIMEOUT,
    use_cache: bool = True,
) -> ProbeResult:
    key = (_server_root(base_url), str(model or ""))
    now = time.monotonic()
    if use_cache:
        hit = _cache.get(key)
        if hit is not None and now - hit[0] < _CACHE_TTL:
            return hit[1]
    server = detect_server(base_url, timeout=timeout)
    props = None
    if server.kind != "unknown":
        props = fetch_props(base_url, model, server=server, timeout=timeout)
    caps = None
    if isinstance(props, dict):
        caps = parse_template_caps(props.get("chat_template", ""))
    if caps is not None:
        efforts = ",".join(caps.tolerated_efforts) or "none"
        remap = "".join(
            f" ({src}->{dst})" for src, dst in caps.remapped_efforts.items()
        )
        logger.info(
            "llamacpp probe: server=%s model=%s efforts=%s%s thinking-toggle=%s",
            server.kind,
            model,
            efforts,
            remap,
            "yes" if caps.supports_thinking_toggle else "no",
        )
    else:
        logger.info(
            "llamacpp probe: server=%s model=%s (no props available)",
            server.kind,
            model,
        )
    result = ProbeResult(server=server, props=props, caps=caps)
    if use_cache:
        _cache[key] = (now, result)
    return result
