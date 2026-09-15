"""Read-only MCP discovery and explicit, profile-scoped connection.

Discovery reads only documented configuration locations and probes only loopback
ports already reported as listening by psutil. Candidate configurations never
leave this module or enter the short-lived resolver cache.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import hashlib
import ipaddress
import json
import logging
import os
import re
import stat
import sys
import threading
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qsl, urlsplit

from hermes_cli.mcp_discovery_codex import normalize_codex_mcp
from hermes_cli.mcp_security import validate_mcp_server_entry

logger = logging.getLogger(__name__)

_CONFIG_MAX_BYTES = 2 * 1024 * 1024
_HTTP_RESPONSE_MAX_BYTES = 256 * 1024
_HTTP_PROBE_TIMEOUT = 1.0
_HTTP_CLEANUP_TIMEOUT = 0.2
_HTTP_DISCOVERY_DEADLINE = 4.0
_HTTP_PROBE_WORKERS = 8
_CACHE_TTL_SECONDS = 5 * 60
_CACHE_MAX_ENTRIES = 256
_ENV_REF_RE = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}")
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_SUPPORTED_PROTOCOL_VERSIONS = frozenset(("2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"))
_INITIALIZE_PROTOCOL_VERSION = "2025-03-26"


class DiscoveryNotFound(Exception):
    pass


class DiscoveryConflict(Exception):
    pass


@dataclass(frozen=True)
class _SourceRecord:
    kind: str
    path: Optional[str]
    server_key: Optional[str]
    scope: Optional[str]
    endpoint: Optional[str]
    fingerprint: str
    name: str
    expires_at: float


_CACHE_LOCK = threading.Lock()
_CANDIDATE_CACHE: dict[str, _SourceRecord] = {}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")


def _fingerprint(source_identity: str, config: dict) -> str:
    digest = hashlib.sha256()
    digest.update(source_identity.encode("utf-8"))
    digest.update(b"\0")
    digest.update(_canonical_bytes(config))
    return digest.hexdigest()


def _only_env_refs(value: str) -> bool:
    return bool(value) and not _ENV_REF_RE.sub("", value).strip(" \t;,:/")


def _is_sensitive_key(key: object) -> bool:
    from agent.redact import _key_has_secret_keyword

    text = str(key or "")
    return text.casefold() not in {"auth", "authentication"} and _key_has_secret_keyword(text)


def _url_has_inline_credentials(url: str) -> bool:
    from agent.redact import _SENSITIVE_QUERY_PARAMS

    try:
        parsed = urlsplit(url)
    except ValueError:
        return False
    if parsed.username or parsed.password:
        return True
    return any(
        key.casefold() in _SENSITIVE_QUERY_PARAMS and not _only_env_refs(value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
    )


def _args_have_inline_credentials(args: object) -> bool:
    if not isinstance(args, list):
        return False
    expect_secret = False
    for item in args:
        text = str(item)
        if expect_secret:
            if not _only_env_refs(text):
                return True
            expect_secret = False
            continue
        if text.startswith("-"):
            flag = text.lstrip("-").split("=", 1)
            if _is_sensitive_key(flag[0]):
                if len(flag) == 1:
                    expect_secret = True
                elif not _only_env_refs(flag[1]):
                    return True
    return False


def _has_inline_credentials(config: dict) -> bool:
    """Detect literals that cannot be copied without duplicating credentials."""
    from agent.redact import redact_sensitive_text

    serialized = _canonical_bytes(config).decode("utf-8")
    if redact_sensitive_text(serialized, force=True, redact_url_credentials=True) != serialized:
        return True

    headers = config.get("headers")
    if isinstance(headers, dict):
        from agent.redact import _SECRET_HEADER_NAMES
        from hermes_cli.mcp_config import redact_mcp_header_display

        for key, value in headers.items():
            key_text = str(key)
            credential_header = re.fullmatch(
                rf"(?:(?:Proxy-)?Authorization|{_SECRET_HEADER_NAMES})", str(key), re.IGNORECASE
            ) or key_text.casefold() in {"cookie", "set-cookie"} or _is_sensitive_key(key_text)
            if credential_header and redact_mcp_header_display(key_text, value) == "***":
                return True

    env = config.get("env")
    if isinstance(env, dict):
        for key, value in env.items():
            if _is_sensitive_key(key) and not _only_env_refs(str(value)):
                return True

    url = config.get("url")
    if isinstance(url, str) and _url_has_inline_credentials(url):
        return True
    if _args_have_inline_credentials(config.get("args")):
        return True

    def _nested(value: object, parent: Optional[str] = None) -> bool:
        if isinstance(value, dict):
            for key, child in value.items():
                key_text = str(key)
                if parent not in {"env", "headers"} and _is_sensitive_key(key_text):
                    if not isinstance(child, str) or not _only_env_refs(child):
                        return True
                if _nested(child, key_text):
                    return True
        elif isinstance(value, list):
            return any(_nested(child, parent) for child in value)
        return False

    return _nested(config)


def _credential_safe_config(config: dict) -> dict:
    """Keep non-secret shape for stable IDs while replacing credential values."""
    from agent.redact import _SECRET_HEADER_NAMES, redact_sensitive_text

    def _scrub(value: object, parent: Optional[str] = None) -> object:
        if isinstance(value, dict):
            result: dict = {}
            for key, child in value.items():
                key_text = str(key)
                credential_header = parent == "headers" and (
                    re.fullmatch(
                        rf"(?:(?:Proxy-)?Authorization|{_SECRET_HEADER_NAMES})", key_text, re.IGNORECASE
                    )
                    or key_text.casefold() in {"cookie", "set-cookie"}
                    or _is_sensitive_key(key_text)
                )
                if credential_header or (parent == "env" and _is_sensitive_key(key_text)):
                    result[key] = "<inline-credential>"
                elif parent not in {"env", "headers"} and _is_sensitive_key(key_text):
                    result[key] = "<inline-credential>"
                elif key_text == "url" and isinstance(child, str) and _url_has_inline_credentials(child):
                    parsed = urlsplit(child)
                    result[key] = f"{parsed.scheme}://<credentialed-endpoint>{parsed.path}"
                elif key_text == "args" and _args_have_inline_credentials(child):
                    result[key] = "<args-with-inline-credential>"
                else:
                    result[key] = _scrub(child, key_text)
            return result
        if isinstance(value, list):
            return [_scrub(child, parent) for child in value]
        if isinstance(value, str):
            return redact_sensitive_text(value, force=True, redact_url_credentials=True)
        return value

    scrubbed = _scrub(config)
    return scrubbed if isinstance(scrubbed, dict) else {"credentialed": True}


def _read_mapping(
    path: Path, *, yaml_file: bool, warnings: list[str], toml_file: bool = False,
) -> Optional[dict]:
    label = str(path)
    try:
        if path.is_symlink():
            warnings.append(f"Skipped symlinked MCP config: {label}")
            return None
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        if os.name == "posix":
            flags |= os.O_NONBLOCK
        fd = os.open(path, flags)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            warnings.append(f"Skipped non-regular MCP config: {label}")
            return None
        with os.fdopen(fd, "rb") as stream:
            if os.fstat(stream.fileno()).st_size > _CONFIG_MAX_BYTES:
                warnings.append(f"Skipped oversized MCP config: {label}")
                return None
            raw = stream.read(_CONFIG_MAX_BYTES + 1)
        if len(raw) > _CONFIG_MAX_BYTES:
            warnings.append(f"Skipped oversized MCP config: {label}")
            return None
        text = raw.decode("utf-8")
        if toml_file:
            value = tomllib.loads(text)
        elif yaml_file:
            import yaml

            try:
                value = yaml.safe_load(text) or {}
            except yaml.YAMLError:
                warnings.append(f"Could not read MCP config {label}: YAMLError")
                return None
        else:
            value = json.loads(text)
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        warnings.append(f"Could not read MCP config {label}: {type(exc).__name__}")
        return None
    if not isinstance(value, dict):
        warnings.append(f"Ignored non-object MCP config: {label}")
        return None
    return value


def _client_config_paths() -> list[tuple[str, Path]]:
    home = Path.home()
    paths = [
        ("Claude Code", home / ".claude.json"),
        ("Cursor", home / ".cursor" / "mcp.json"),
        ("Codex", Path(os.environ.get("CODEX_HOME") or home / ".codex").expanduser() / "config.toml"),
    ]
    if sys.platform == "darwin":
        desktop = home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    elif os.name == "nt":
        appdata = Path(os.environ.get("APPDATA", home / "AppData" / "Roaming"))
        desktop = appdata / "Claude" / "claude_desktop_config.json"
    else:
        desktop = Path(os.environ.get("XDG_CONFIG_HOME", home / ".config")) / "Claude" / "claude_desktop_config.json"
    paths.append(("Claude Desktop", desktop))
    return paths


def _raw_sources(warnings: list[str]) -> list[tuple[str, str, Path, str, dict, Optional[str]]]:
    """Return source records without resolving env refs or reading credential stores."""
    from hermes_cli.profiles import _get_default_hermes_home, _iter_named_profile_dirs

    sources: list[tuple[str, str, Path, str, dict, Optional[str]]] = []
    profile_paths = [("default", _get_default_hermes_home())]
    try:
        profile_paths.extend((entry.name, entry) for entry in _iter_named_profile_dirs())
    except OSError as exc:
        warnings.append(f"Hermes profile enumeration was partial: {type(exc).__name__}")
    for profile_name, home in profile_paths:
        path = home / "config.yaml"
        root = _read_mapping(path, yaml_file=True, warnings=warnings)
        if root is None:
            continue
        servers = root.get("mcp_servers")
        if servers is None:
            continue
        if not isinstance(servers, dict):
            warnings.append(f"Ignored malformed mcp_servers in Hermes profile '{profile_name}'")
            continue
        label = f"Hermes profile '{profile_name}' ({path})"
        for name, config in servers.items():
            if isinstance(name, str) and name.strip() and isinstance(config, dict):
                sources.append(("hermes", label, path, name, copy.deepcopy(config), None))
            else:
                warnings.append(f"Ignored malformed MCP entry in {label}")

    for client_name, path in _client_config_paths():
        is_codex = client_name == "Codex"
        root = _read_mapping(path, yaml_file=False, toml_file=is_codex, warnings=warnings)
        if root is None:
            continue
        server_key = "mcp_servers" if is_codex else "mcpServers"
        servers = root.get(server_key)
        label = f"{client_name} ({path})"
        if servers is not None and not isinstance(servers, dict):
            warnings.append(f"Ignored malformed {server_key} in {label}")
        elif isinstance(servers, dict):
            for name, config in servers.items():
                if isinstance(name, str) and name.strip() and isinstance(config, dict):
                    kind = "codex" if is_codex else "client"
                    sources.append((kind, label, path, name, copy.deepcopy(config), None))
                else:
                    warnings.append(f"Ignored malformed MCP entry in {label}")
        if client_name == "Claude Code":
            projects = root.get("projects")
            if isinstance(projects, dict):
                for project, project_cfg in projects.items():
                    project_servers = project_cfg.get("mcpServers") if isinstance(project_cfg, dict) else None
                    if not isinstance(project_servers, dict):
                        continue
                    project_label = f"Claude Code project {project!r} ({path})"
                    for name, config in project_servers.items():
                        if isinstance(name, str) and name.strip() and isinstance(config, dict):
                            sources.append(
                                ("client", project_label, path, name, copy.deepcopy(config), str(project))
                            )
                        else:
                            warnings.append(f"Ignored malformed MCP entry in {project_label}")
    return sources


def _transport(config: dict) -> Optional[str]:
    has_command = isinstance(config.get("command"), str) and bool(config["command"].strip())
    has_url = isinstance(config.get("url"), str) and bool(config["url"].strip())
    if has_command == has_url:
        return None
    return "stdio" if has_command else "http"


def _safe_endpoint_display(url: str) -> str:
    from hermes_cli.mcp_config import redact_mcp_probe_text

    return redact_mcp_probe_text(url)


def _cache(record_id: str, record: _SourceRecord) -> None:
    now = time.monotonic()
    with _CACHE_LOCK:
        stale = [key for key, value in _CANDIDATE_CACHE.items() if value.expires_at <= now]
        for key in stale:
            _CANDIDATE_CACHE.pop(key, None)
        if len(_CANDIDATE_CACHE) >= _CACHE_MAX_ENTRIES:
            oldest = min(_CANDIDATE_CACHE, key=lambda key: _CANDIDATE_CACHE[key].expires_at)
            _CANDIDATE_CACHE.pop(oldest, None)
        _CANDIDATE_CACHE[record_id] = record


def _candidate_from_raw(
    kind: str,
    label: str,
    path: Path,
    name: str,
    config: dict,
    scope: Optional[str],
    target_names: set[str],
) -> dict:
    identity = f"{kind}:{path}:{scope or ''}:{name}"
    raw_config = config
    if kind == "codex":
        try:
            config = normalize_codex_mcp(raw_config)
        except ValueError as exc:
            return {
                "id": _fingerprint(identity, {"unsupported": True}),
                "name": name,
                "source": label,
                "transport": "http" if "url" in raw_config else "stdio",
                "summary": f"MCP configuration from {label}",
                "connectable": False,
                "reason": str(exc),
            }
    inline_credentials = _has_inline_credentials(config)
    fingerprint_config = raw_config if not inline_credentials else _credential_safe_config(config)
    candidate_id = _fingerprint(identity, fingerprint_config)
    transport = _transport(config)
    issues = validate_mcp_server_entry(name, config)
    reason: Optional[str] = None
    if inline_credentials:
        reason = "Inline credentials cannot be copied; replace them with ${ENV_VAR} references in the source config."
    elif transport is None:
        reason = "Configuration must define exactly one MCP transport (command or url)."
    elif issues:
        reason = "Configuration was rejected by MCP security validation."
    elif name in target_names:
        reason = f"Target profile already has an MCP server named '{name}'."

    if transport == "stdio" and not inline_credentials:
        command = str(config.get("command") or "")
        summary = f"stdio command {command!r} from {label}"
    elif transport == "http" and not inline_credentials:
        summary = f"HTTP endpoint {_safe_endpoint_display(str(config.get('url')))} from {label}"
    else:
        summary = f"MCP configuration from {label}"

    candidate = {
        "id": candidate_id,
        "name": name,
        "source": label,
        "transport": transport or ("http" if "url" in config else "stdio"),
        "summary": summary,
        "connectable": reason is None,
    }
    if reason is not None:
        candidate["reason"] = reason
    else:
        _cache(
            candidate_id,
            _SourceRecord(
                kind=kind,
                path=str(path),
                server_key=name,
                scope=scope,
                endpoint=None,
                fingerprint=_fingerprint(identity, raw_config),
                name=name,
                expires_at=time.monotonic() + _CACHE_TTL_SECONDS,
            ),
        )
    return candidate


def _listener_endpoints(warnings: list[str]) -> list[str]:
    try:
        import psutil

        connections = psutil.net_connections(kind="inet")
    except Exception as exc:
        warnings.append(f"Could not enumerate local listening ports: {type(exc).__name__}")
        return []

    hosts_and_ports: set[tuple[str, int]] = set()
    malformed = 0
    for connection in connections:
        try:
            if connection.status != psutil.CONN_LISTEN or not connection.laddr:
                continue
            host = str(connection.laddr.ip if hasattr(connection.laddr, "ip") else connection.laddr[0])
            port = int(connection.laddr.port if hasattr(connection.laddr, "port") else connection.laddr[1])
            address = ipaddress.ip_address(host.split("%", 1)[0])
            if address.is_unspecified:
                hosts_and_ports.add(("::1" if address.version == 6 else "127.0.0.1", port))
            elif address.is_loopback:
                hosts_and_ports.add((str(address), port))
        except (AttributeError, TypeError, ValueError):
            malformed += 1
    if malformed:
        warnings.append(f"Local listening-port scan was partial: ignored {malformed} malformed entr{'y' if malformed == 1 else 'ies'}." )

    endpoints: list[str] = []
    for host, port in sorted(hosts_and_ports, key=lambda item: (item[1], item[0])):
        display_host = f"[{host}]" if ":" in host else host
        endpoints.extend(f"http://{display_host}:{port}{path}" for path in ("/mcp", "/sse"))
    return endpoints


def _parse_initialize_response(body: bytes, content_type: str) -> Optional[dict]:
    if len(body) > _HTTP_RESPONSE_MAX_BYTES:
        return None
    payloads: list[str]
    text = body.decode("utf-8", errors="strict")
    if "text/event-stream" in content_type.casefold() or text.lstrip().startswith(("data:", "event:")):
        payloads = []
        data_lines: list[str] = []
        for line in text.splitlines():
            if not line:
                if data_lines:
                    payloads.append("\n".join(data_lines))
                    data_lines = []
                continue
            if line.startswith("data:"):
                data = line[5:]
                data_lines.append(data[1:] if data.startswith(" ") else data)
        if data_lines:
            payloads.append("\n".join(data_lines))
    else:
        payloads = [text]
    for payload in payloads:
        try:
            message = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or message.get("id") != 1:
            continue
        result = message.get("result")
        if not isinstance(result, dict):
            continue
        server_info = result.get("serverInfo")
        if (
            isinstance(result.get("protocolVersion"), str)
            and result["protocolVersion"] in _SUPPORTED_PROTOCOL_VERSIONS
            and isinstance(server_info, dict)
            and isinstance(server_info.get("name"), str)
            and bool(server_info["name"].strip())
            and isinstance(server_info.get("version"), str)
            and bool(server_info["version"].strip())
            and isinstance(result.get("capabilities"), dict)
        ):
            return result
    return None


async def _read_initialize_response(response) -> Optional[dict]:
    content_type = response.headers.get("Content-Type", "")
    is_sse = "text/event-stream" in content_type.casefold()
    body = bytearray()
    async for chunk in response.aiter_raw():
        body.extend(chunk)
        if len(body) > _HTTP_RESPONSE_MAX_BYTES:
            return None
        if is_sse:
            result = _parse_initialize_response(bytes(body), content_type)
            if result is not None:
                return result
    return _parse_initialize_response(bytes(body), content_type)


def _literal_loopback_url(endpoint: str) -> bool:
    try:
        parsed = urlsplit(endpoint)
        address = ipaddress.ip_address(parsed.hostname or "")
        return (
            parsed.scheme == "http"
            and address.is_loopback
            and parsed.port is not None
            and parsed.username is None
            and parsed.password is None
            and not parsed.fragment
        )
    except ValueError:
        return False


async def _probe_http(endpoint: str, deadline: Optional[float]) -> Optional[dict]:
    import httpx

    started = time.monotonic()
    overall_deadline = min(deadline or float("inf"), started + _HTTP_PROBE_TIMEOUT)
    response_deadline = overall_deadline - min(
        _HTTP_CLEANUP_TIMEOUT, max(0.0, (overall_deadline - started) / 4)
    )
    session_id = ""
    protocol_version = _INITIALIZE_PROTOCOL_VERSION
    payload = {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {
            "protocolVersion": _INITIALIZE_PROTOCOL_VERSION, "capabilities": {},
            "clientInfo": {"name": "hermes-discovery", "version": "1"},
        },
    }
    # Cancellation covers connect + response headers + body, not merely an idle
    # socket timeout. Never inherit a proxy or follow redirects off loopback.
    async with httpx.AsyncClient(trust_env=False, follow_redirects=False) as client:
        async def initialize():
            nonlocal session_id, protocol_version
            async with client.stream(
                "POST", endpoint, json=payload,
                headers={"Accept": "application/json, text/event-stream"},
            ) as response:
                response.raise_for_status()
                session_id = response.headers.get("Mcp-Session-Id", "")
                result = await _read_initialize_response(response)
                if result is not None:
                    protocol_version = result["protocolVersion"]
                return result

        try:
            return await asyncio.wait_for(initialize(), max(0.0, response_deadline - time.monotonic()))
        except (httpx.HTTPError, OSError, asyncio.TimeoutError, UnicodeError, ValueError):
            return None
        finally:
            remaining = min(_HTTP_CLEANUP_TIMEOUT, overall_deadline - time.monotonic())
            if remaining > 0 and session_id and len(session_id) <= 1024 and not any(
                char in session_id for char in "\r\n"
            ):
                try:
                    await asyncio.wait_for(client.delete(endpoint, headers={
                        "Mcp-Session-Id": session_id, "MCP-Protocol-Version": protocol_version,
                    }), remaining)
                except (httpx.HTTPError, OSError, asyncio.TimeoutError, ValueError):
                    pass


def _probe_endpoint(endpoint: str, deadline: Optional[float] = None) -> Optional[dict]:
    if not _literal_loopback_url(endpoint):
        return None
    return asyncio.run(_probe_http(endpoint, deadline))


def _safe_server_name(server_info: dict, endpoint: str) -> str:
    raw = str(server_info.get("name") or "").strip()
    if raw:
        cleaned = _SAFE_NAME_RE.sub("-", raw).strip("-.")[:64]
        if cleaned:
            return cleaned
    return f"local-mcp-{urlsplit(endpoint).port}"


def _http_candidates(target_names: set[str], warnings: list[str]) -> list[dict]:
    endpoints = _listener_endpoints(warnings)
    if not endpoints:
        return []
    deadline = time.monotonic() + _HTTP_DISCOVERY_DEADLINE
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=_HTTP_PROBE_WORKERS, thread_name_prefix="mcp-discovery")
    futures = {executor.submit(_probe_endpoint, endpoint, deadline): endpoint for endpoint in endpoints}
    results: list[tuple[str, dict]] = []
    unfinished: set[concurrent.futures.Future] = set()
    try:
        done, unfinished = concurrent.futures.wait(futures, timeout=_HTTP_DISCOVERY_DEADLINE)
        for future in done:
            try:
                result = future.result()
            except Exception:
                logger.debug("Unexpected MCP endpoint probe failure", exc_info=True)
                continue
            if result is not None:
                results.append((futures[future], result))
    finally:
        for future in unfinished:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
    if unfinished:
        warnings.append(f"Local MCP probe deadline reached; {len(unfinished)} endpoint(s) were not checked completely.")

    # One successful conventional path is enough for a listener; prefer /mcp.
    selected: dict[tuple[str, int], tuple[str, dict]] = {}
    for endpoint, result in sorted(results, key=lambda item: (not item[0].endswith("/mcp"), item[0])):
        parsed = urlsplit(endpoint)
        selected.setdefault((parsed.hostname or "", parsed.port or 0), (endpoint, result))

    candidates: list[dict] = []
    for endpoint, result in selected.values():
        name = _safe_server_name(result.get("serverInfo") or {}, endpoint)
        config = {"url": endpoint}
        identity = f"http-probe:{endpoint}"
        candidate_id = _fingerprint(identity, config)
        reason = f"Target profile already has an MCP server named '{name}'." if name in target_names else None
        candidate = {
            "id": candidate_id,
            "name": name,
            "source": f"Local HTTP endpoint {endpoint}",
            "transport": "http",
            "summary": f"Verified MCP initialize endpoint at {endpoint}",
            "connectable": reason is None,
        }
        if reason:
            candidate["reason"] = reason
        else:
            _cache(
                candidate_id,
                _SourceRecord(
                    kind="http",
                    path=None,
                    server_key=None,
                    scope=None,
                    endpoint=endpoint,
                    fingerprint=_fingerprint(identity, config),
                    name=name,
                    expires_at=time.monotonic() + _CACHE_TTL_SECONDS,
                ),
            )
        candidates.append(candidate)
    return candidates


def _target_names(profile: Optional[str]) -> set[str]:
    from hermes_cli.profiles import get_profile_dir, profile_exists

    requested = (profile or "").strip()
    if requested and requested not in {"current"}:
        if not profile_exists(requested):
            raise DiscoveryNotFound(f"Profile '{requested}' does not exist")
        path = get_profile_dir(requested) / "config.yaml"
    else:
        from hermes_constants import get_hermes_home

        path = get_hermes_home() / "config.yaml"
    root = _read_mapping(path, yaml_file=True, warnings=[])
    servers = root.get("mcp_servers") if root else None
    return set(servers) if isinstance(servers, dict) else set()


def discover_candidates(profile: Optional[str]) -> dict:
    warnings: list[str] = []
    target_names = _target_names(profile)
    candidates = [
        _candidate_from_raw(kind, label, path, name, config, scope, target_names)
        for kind, label, path, name, config, scope in _raw_sources(warnings)
    ]
    candidates.extend(_http_candidates(target_names, warnings))
    candidates.sort(key=lambda candidate: (candidate["name"].casefold(), candidate["source"], candidate["id"]))
    return {"candidates": candidates, "warnings": warnings}


def _reread_record(record: _SourceRecord) -> dict:
    config: Any
    if record.kind == "http":
        if not record.endpoint or _probe_endpoint(record.endpoint) is None:
            raise DiscoveryConflict("Discovered HTTP endpoint no longer completes an MCP initialize handshake")
        config = {"url": record.endpoint}
        identity = f"http-probe:{record.endpoint}"
    else:
        if not record.path or not record.server_key:
            raise DiscoveryConflict("Discovery source metadata is incomplete")
        path = Path(record.path)
        root = _read_mapping(
            path, yaml_file=(record.kind == "hermes"), toml_file=(record.kind == "codex"), warnings=[],
        )
        key = "mcp_servers" if record.kind in {"hermes", "codex"} else "mcpServers"
        if record.scope is not None:
            projects = root.get("projects") if root else None
            root = projects.get(record.scope) if isinstance(projects, dict) else None
        servers = root.get(key) if root else None
        config = servers.get(record.server_key) if isinstance(servers, dict) else None
        if not isinstance(config, dict):
            raise DiscoveryConflict("Discovery source changed; refresh the candidate list")
        config = copy.deepcopy(config)
        identity = f"{record.kind}:{path}:{record.scope or ''}:{record.server_key}"
    raw_config = config
    if record.kind == "codex":
        try:
            config = normalize_codex_mcp(raw_config)
        except ValueError as exc:
            raise DiscoveryConflict(str(exc)) from None
    if _has_inline_credentials(config):
        raise DiscoveryConflict("Discovery source now contains inline credentials; refresh the candidate list")
    if _fingerprint(identity, raw_config) != record.fingerprint:
        raise DiscoveryConflict("Discovery source changed; refresh the candidate list")
    if _transport(config) is None or validate_mcp_server_entry(record.name, config):
        raise DiscoveryConflict("Discovery source is no longer a valid MCP configuration")
    if config.get("enabled") is False:
        config["enabled"] = True
    return config


def connect_candidate(candidate_id: str) -> str:
    candidate_id = str(candidate_id or "").strip()
    if not re.fullmatch(r"[0-9a-f]{64}", candidate_id):
        raise DiscoveryNotFound("Discovery candidate not found or expired")
    now = time.monotonic()
    with _CACHE_LOCK:
        record = _CANDIDATE_CACHE.get(candidate_id)
        if record is None or record.expires_at <= now:
            _CANDIDATE_CACHE.pop(candidate_id, None)
            raise DiscoveryNotFound("Discovery candidate not found or expired")
    config = _reread_record(record)

    from hermes_cli.mcp_config import _get_mcp_servers, _save_mcp_server

    if record.name in _get_mcp_servers():
        raise DiscoveryConflict(f"Target profile already has an MCP server named '{record.name}'")
    if not _save_mcp_server(record.name, config):
        raise DiscoveryConflict("Discovery candidate was rejected by MCP security validation")
    with _CACHE_LOCK:
        _CANDIDATE_CACHE.pop(candidate_id, None)
    return record.name
