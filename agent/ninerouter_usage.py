from __future__ import annotations

import hashlib
import ipaddress
import math
import os
import re
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote, urlsplit, urlunsplit

import httpx

from agent.account_usage import AccountUsageRoute, AccountUsageSnapshot
from agent.redact import redact_sensitive_text

NINEROUTER_PROVIDER = "9router"
NINEROUTER_SOURCE = "9router_management_api"


class _ManagementAuthRequired(Exception):
    pass


class _ManagementResponseInvalid(Exception):
    pass


class _ManagementRequestFailed(Exception):
    def __init__(self, status_code: int):
        super().__init__(str(status_code))
        self.status_code = status_code


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


_DISPLAY_ASSIGNMENT_RE = re.compile(
    r"(?<![A-Za-z0-9_-])[A-Za-z][A-Za-z0-9_.-]{0,64}\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s,};\]]+)",
)
_DISPLAY_CREDENTIAL_FIELD_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?:api[_-]?key|access[_-]?token|auth(?:orization)?|password|passwd|secret|token|cookie|credential|private[_-]?key)\s*[:=]",
)
_DISPLAY_BEARER_RE = re.compile(r"(?i)(?<![A-Za-z0-9])(?:bearer|basic)\s+[^\s,};\]]+")
_DISPLAY_URL_USERINFO_RE = re.compile(
    r"(?i)\b[a-z][a-z0-9+.-]{1,31}://[^\s/:@]+:[^\s/@]+@"
)
_DISPLAY_LONG_HEX_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:0x)?[0-9A-Fa-f]{24,}(?![A-Za-z0-9])"
)
_DISPLAY_LONG_OPAQUE_RE = re.compile(
    r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{32,}(?![A-Za-z0-9_-])"
)
_DISPLAY_BASE64_BLOB_RE = re.compile(
    r"(?<![A-Za-z0-9+/=_-])[A-Za-z0-9+/]{24,}={0,2}(?![A-Za-z0-9+/=_-])"
)


def _looks_like_credential_material(value: str) -> bool:
    if (
        _DISPLAY_ASSIGNMENT_RE.search(value)
        or _DISPLAY_CREDENTIAL_FIELD_RE.search(value)
        or _DISPLAY_BEARER_RE.search(value)
        or _DISPLAY_URL_USERINFO_RE.search(value)
    ):
        return True

    # Reuse Hermes' strict redaction boundary for known vendor prefixes,
    # JWTs, auth headers, and other credential forms.  ``force`` keeps this
    # display-data boundary fail-closed even if log redaction is disabled.
    try:
        if redact_sensitive_text(
            value,
            force=True,
            file_read=True,
            redact_url_credentials=True,
        ) != value:
            return True
    except Exception:
        # A sanitizer failure must not turn untrusted provider data into a
        # displayable label.
        return True

    # Provider labels are not a credential transport.  Reject long, opaque
    # blobs even when they have no recognizable vendor prefix or keyword.
    return bool(
        _DISPLAY_LONG_HEX_RE.search(value)
        or _DISPLAY_LONG_OPAQUE_RE.search(value)
        or _DISPLAY_BASE64_BLOB_RE.search(value)
    )


def _clean_display_label(value: Any, limit: int = 120) -> Optional[str]:
    if not isinstance(value, str):
        return None
    if any(ord(c) < 32 or (127 <= ord(c) <= 159) for c in value):
        return None
    text = " ".join(value.split())
    if not text or _looks_like_credential_material(text):
        return None
    return text[:limit]


def _safe_label(value: Any, fallback: str) -> str:
    label = _clean_display_label(value)
    if label is not None:
        return label
    if isinstance(fallback, str) and not fallback.strip():
        return ""
    return _clean_display_label(fallback) or "Unknown"


def _safe_scope(value: Optional[str]) -> Optional[str]:
    return _clean_display_label(value, limit=128)


def sanitize_scope(value: Optional[str], fallback: str = "profile:current") -> str:
    """Return a safe scope label for API/UI serialization."""
    return _safe_scope(value) or _safe_scope(fallback) or "profile:current"


def _normalise_url(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return None
    parsed = urlsplit(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        return None
    try:
        parsed.port
    except ValueError:
        return None
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", "")).rstrip("/")


def _management_base_url(
    base_url: Optional[str],
    explicit_management_url: Optional[str],
) -> Optional[str]:
    explicit = _normalise_url(explicit_management_url)
    if explicit:
        return explicit

    normalized = _normalise_url(base_url)
    if not normalized:
        return None
    parsed = urlsplit(normalized)
    host = (parsed.hostname or "").lower()
    if host != "localhost":
        try:
            if not ipaddress.ip_address(host).is_loopback:
                return None
        except ValueError:
            return None

    path = parsed.path.rstrip("/")
    if path == "/v1":
        path = ""
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", "")).rstrip("/")


def _numeric_loopback_base_url(value: Optional[str]) -> Optional[str]:
    """Canonicalize a loopback URL before sending a local credential.

    Hostnames are not used for credential-bearing requests: ``localhost`` is
    resolved and every returned address must be loopback before the request is
    rewritten to a numeric loopback peer.  This is paired with ``trust_env``
    disabled on the HTTP client so an environment proxy cannot receive the
    management credential.
    """
    normalized = _normalise_url(value)
    if not normalized:
        return None
    parsed = urlsplit(normalized)
    host = (parsed.hostname or "").lower()
    selected: Optional[str] = None
    try:
        port = parsed.port
        if host != "localhost":
            address = ipaddress.ip_address(host)
            if not address.is_loopback:
                return None
            selected = str(address)
        else:
            infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
            addresses: list[str] = []
            for info in infos:
                sockaddr = info[4] if len(info) > 4 else ()
                raw_address = sockaddr[0] if sockaddr else ""
                try:
                    address = ipaddress.ip_address(str(raw_address).split("%", 1)[0])
                except ValueError:
                    return None
                if not address.is_loopback:
                    return None
                addresses.append(str(address))
            if not addresses:
                return None
            # Prefer the runtime's usual IPv4 listener when it was actually
            # returned; otherwise connect only to a returned loopback address.
            selected = "127.0.0.1" if "127.0.0.1" in addresses else addresses[0]
    except (OSError, ValueError):
        return None

    netloc = f"[{selected}]" if selected and ":" in selected else selected
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path.rstrip("/"), "", "")).rstrip("/")


def _is_loopback_url(value: Optional[str]) -> bool:
    return _numeric_loopback_base_url(value) is not None


def _ninerouter_data_dir(
    configured: Optional[str] = None,
    *,
    use_process_env: bool = True,
) -> Path:
    if configured is None and use_process_env:
        configured = os.environ.get("HERMES_9ROUTER_DATA_DIR")
    if configured and "\x00" not in configured and len(configured) <= 4096:
        return Path(configured)
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        return Path(appdata) / "9router" if appdata else Path.home() / "AppData" / "Roaming" / "9router"
    return Path.home() / ".9router"


def resolve_ninerouter_cli_token(
    base_url: Optional[str],
    management_base_url: Optional[str] = None,
    *,
    data_dir: Optional[str] = None,
    use_process_env: bool = True,
    allow_default_data_dir: bool = True,
) -> Optional[str]:
    """Resolve 9Router's local CLI auth token without using inference auth.

    9Router's management middleware intentionally supports a derived
    ``x-9r-cli-token`` for local CLI clients.  Only derive it for a loopback
    management endpoint, and keep the source values and resulting token in
    memory; callers must not log, serialize, or expose the result.
    """
    management_url = _management_base_url(base_url, management_base_url)
    if not management_url:
        return None
    if not _is_loopback_url(management_url):
        return None
    if not allow_default_data_dir:
        if (
            not isinstance(data_dir, str)
            or not data_dir.strip()
            or len(data_dir) > 4096
            or any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in data_dir)
        ):
            return None
        try:
            data_dir_path = Path(data_dir)
        except (OSError, TypeError, ValueError):
            return None
    else:
        data_dir_path = _ninerouter_data_dir(data_dir, use_process_env=use_process_env)
    try:
        machine_id = (data_dir_path / "machine-id").read_text(encoding="utf-8").strip()
        cli_secret = (data_dir_path / "auth" / "cli-secret").read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    if not machine_id or not cli_secret or len(machine_id) > 4096 or len(cli_secret) > 4096:
        return None
    if "\x00" in machine_id or "\x00" in cli_secret:
        return None
    return hashlib.sha256((machine_id + "9r-cli-auth" + cli_secret).encode("utf-8")).hexdigest()[:16]


def _headers(auth_cookie: Optional[str], cli_token: Optional[str] = None) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if isinstance(auth_cookie, str) and auth_cookie.strip():
        # The value is intentionally opaque.  It is never logged or returned.
        cookie = auth_cookie.strip()
        if "\r" not in cookie and "\n" not in cookie and len(cookie) <= 4096:
            headers["Cookie"] = cookie
    if isinstance(cli_token, str) and cli_token.strip():
        # This is a separate local management credential, never the inference
        # API key.  Keep it out of response objects and diagnostic messages.
        token = cli_token.strip()
        if "\r" not in token and "\n" not in token and len(token) <= 128:
            headers["x-9r-cli-token"] = token
    return headers


def _get_json(client: httpx.Client, url: str, headers: dict[str, str]) -> dict[str, Any]:
    try:
        response = client.get(url, headers=headers)
    except Exception as exc:
        raise _ManagementRequestFailed(0) from exc
    status = int(getattr(response, "status_code", 0) or 0)
    if status in {401, 403} or 300 <= status < 400:
        raise _ManagementAuthRequired
    if status < 200 or status >= 300:
        raise _ManagementRequestFailed(status)
    try:
        payload = response.json()
    except Exception as exc:
        raise _ManagementResponseInvalid from exc
    if not isinstance(payload, dict):
        raise _ManagementResponseInvalid
    return payload


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _remaining_percent(limit: Optional[float], remaining: Optional[float]) -> Optional[float]:
    """Derive a percentage only within one complete route contract."""
    if limit is None or remaining is None or limit <= 0 or remaining < 0 or remaining > limit:
        return None
    return round((remaining / limit) * 100.0, 1)


def _account_label(raw_connection: dict[str, Any], fallback: str) -> str:
    """Return a display label without forwarding credential-like fields."""
    for key in ("name", "email", "account", "label"):
        label = _clean_display_label(raw_connection.get(key), limit=160)
        if label is not None:
            return label
    return _safe_label(fallback, "account")


def _disambiguate_account_label(
    raw_connection: dict[str, Any],
    provider: str,
    provider_count: int,
    used_labels: set[str],
    label_occurrences: dict[str, int],
) -> str:
    """Return a display label unique within the provider, without raw IDs or credentials."""
    raw_label: Optional[str] = None
    for key in ("name", "email", "account", "label"):
        label = _clean_display_label(raw_connection.get(key), limit=160)
        if label is not None:
            raw_label = label
            break

    if raw_label is not None:
        occ = label_occurrences.get(raw_label, 0) + 1
        label_occurrences[raw_label] = occ
        if occ == 1 and raw_label not in used_labels:
            disambiguated = raw_label
        else:
            count = max(occ, 2)
            candidate = f"{raw_label} #{count}"
            while candidate in used_labels:
                count += 1
                candidate = f"{raw_label} #{count}"
            disambiguated = candidate
    else:
        candidate = f"{provider} #{provider_count}"
        n = provider_count
        while candidate in used_labels:
            n += 1
            candidate = f"{provider} #{n}"
        disambiguated = candidate

    used_labels.add(disambiguated)
    return disambiguated


def _reset_at(value: Any) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            timestamp = float(value)
            if not math.isfinite(timestamp):
                return None
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
        except (OSError, OverflowError, TypeError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        # A naive timestamp has no billing-window timezone semantics.  Do not
        # silently turn it into UTC and claim a reset time was reported.
        return parsed if parsed.tzinfo else None
    return None


def _route(
    route_name: str,
    provider: Optional[str],
    *,
    account: Optional[str] = None,
    usage: Optional[float] = None,
    limit: Optional[float] = None,
    remaining: Optional[float] = None,
    remaining_percent: Optional[float] = None,
    unit: Optional[str] = None,
    reset_at: Optional[datetime] = None,
    status: str = "unknown",
    detail: Optional[str] = None,
) -> AccountUsageRoute:
    return AccountUsageRoute(
        route=_safe_label(route_name, "route"),
        provider=_safe_label(provider, "Unknown provider"),
        account=(_safe_label(account, "") or None) if account is not None else None,
        usage=usage,
        limit=limit,
        remaining=remaining,
        remaining_percent=remaining_percent,
        unit=(_safe_label(unit, "") or None) if unit is not None else None,
        reset_at=reset_at,
        status=status,
        source=NINEROUTER_SOURCE,
        detail=detail,
    )


def _parse_quota_routes(
    connection_label: str,
    provider: str,
    payload: dict[str, Any],
    account: Optional[str] = None,
) -> tuple[AccountUsageRoute, ...]:
    quotas = payload.get("quotas")
    if not isinstance(quotas, dict) or not quotas:
        return (
            _route(
                "quota",
                provider,
                account=account,
                detail="Quota fields were not returned for this route.",
            ),
        )

    routes: list[AccountUsageRoute] = []
    for raw_name, raw_quota in quotas.items():
        quota_name = _safe_label(raw_name, "quota")
        route_name = quota_name
        if not isinstance(raw_quota, dict):
            routes.append(
                _route(
                    route_name,
                    provider,
                    account=account,
                    detail="Quota entry had an unsupported shape.",
                )
            )
            continue

        usage = _number(raw_quota.get("used"))
        limit = _number(raw_quota.get("total"))
        remaining_value = raw_quota.get("remaining")
        remaining = _number(remaining_value)
        remaining_present = "remaining" in raw_quota
        remaining_invalid = remaining_present and remaining_value is not None and remaining is None
        unit_value = raw_quota.get("unit")
        unit = _safe_label(unit_value, "") or None
        unit_present = "unit" in raw_quota
        unit_invalid = unit_present and unit_value is not None and unit is None

        has_reset_camel = "resetAt" in raw_quota
        has_reset_snake = "reset_at" in raw_quota
        reset_invalid = False
        reset_conflict = False
        reset_at: Optional[datetime] = None

        if has_reset_camel and has_reset_snake:
            val_camel = raw_quota["resetAt"]
            val_snake = raw_quota["reset_at"]
            empty_camel = val_camel is None or val_camel == ""
            empty_snake = val_snake is None or val_snake == ""
            if empty_camel and empty_snake:
                reset_at = None
            elif empty_camel != empty_snake:
                reset_conflict = True
            else:
                r1 = _reset_at(val_camel)
                r2 = _reset_at(val_snake)
                if r1 is None or r2 is None:
                    reset_invalid = True
                elif r1 != r2:
                    reset_conflict = True
                else:
                    reset_at = r1
        elif has_reset_camel:
            val = raw_quota["resetAt"]
            if val is not None and val != "":
                reset_at = _reset_at(val)
                if reset_at is None:
                    reset_invalid = True
        elif has_reset_snake:
            val = raw_quota["reset_at"]
            if val is not None and val != "":
                reset_at = _reset_at(val)
                if reset_at is None:
                    reset_invalid = True

        if usage is None or limit is None:
            routes.append(
                _route(
                    route_name,
                    provider,
                    account=account,
                    detail="Quota fields were incomplete or used an unsupported type.",
                )
            )
            continue

        if remaining_invalid:
            routes.append(
                _route(
                    route_name,
                    provider,
                    account=account,
                    usage=usage,
                    limit=limit,
                    unit=unit,
                    reset_at=reset_at,
                    detail="Provider returned an invalid remaining value.",
                )
            )
            continue

        if unit_invalid:
            routes.append(
                _route(
                    route_name,
                    provider,
                    account=account,
                    usage=usage,
                    limit=limit,
                    remaining=remaining,
                    reset_at=reset_at,
                    detail="Provider returned an invalid quota unit.",
                )
            )
            continue

        if (
            usage < 0
            or limit < 0
            or (usage > limit and raw_quota.get("unlimited") is not True)
            or (remaining is not None and (remaining < 0 or remaining > limit))
        ):
            routes.append(
                _route(
                    route_name,
                    provider,
                    account=account,
                    usage=usage,
                    limit=limit,
                    remaining=remaining,
                    unit=unit,
                    detail="Quota values were outside a valid range.",
                )
            )
            continue

        if reset_invalid or reset_conflict:
            routes.append(
                _route(
                    route_name,
                    provider,
                    account=account,
                    usage=usage,
                    limit=limit,
                    remaining=remaining,
                    unit=unit,
                    detail=(
                        "Provider returned conflicting quota reset times."
                        if reset_conflict
                        else "Quota reset time was not parseable."
                    ),
                )
            )
            continue

        has_pct_camel = "remainingPercentage" in raw_quota
        has_pct_snake = "remaining_percent" in raw_quota
        reported_percent: Optional[float] = None
        percent_invalid = False
        percent_conflict = False

        if has_pct_camel and has_pct_snake:
            val_camel = raw_quota["remainingPercentage"]
            val_snake = raw_quota["remaining_percent"]
            p1 = _number(val_camel)
            p2 = _number(val_snake)
            if p1 is None or not (0 <= p1 <= 100) or p2 is None or not (0 <= p2 <= 100):
                percent_invalid = True
            elif not math.isclose(p1, p2, rel_tol=0.0, abs_tol=1e-5):
                percent_conflict = True
            else:
                reported_percent = p1
        elif has_pct_camel or has_pct_snake:
            val = raw_quota["remainingPercentage"] if has_pct_camel else raw_quota["remaining_percent"]
            p = _number(val)
            if p is None or not (0 <= p <= 100):
                percent_invalid = True
            else:
                reported_percent = p

        if percent_invalid or percent_conflict:
            routes.append(
                _route(
                    route_name,
                    provider,
                    account=account,
                    usage=usage,
                    limit=limit,
                    remaining=remaining,
                    unit=unit,
                    reset_at=reset_at,
                    detail=(
                        "Provider returned conflicting remaining percentage aliases."
                        if percent_conflict
                        else "Provider returned an invalid remaining percentage."
                    ),
                )
            )
            continue

        inconsistent_values = False
        if reported_percent is not None and limit == 0:
            # A zero denominator has no defined percentage semantics.  Do not
            # accept an upstream percentage unless a provider-specific contract
            # explicitly supplies one (none does for this adapter).
            inconsistent_values = True
        elif remaining is not None and not math.isclose(
            usage + remaining,
            limit,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            inconsistent_values = True
        elif reported_percent is not None and remaining is not None:
            expected_percent = _remaining_percent(limit, remaining)
            inconsistent_values = expected_percent is None or not math.isclose(
                reported_percent,
                expected_percent,
                rel_tol=0.0,
                abs_tol=0.6,
            )
        elif reported_percent is not None and limit > 0:
            expected_percent = ((limit - usage) / limit) * 100.0
            inconsistent_values = not math.isclose(
                reported_percent,
                expected_percent,
                rel_tol=0.0,
                abs_tol=0.6,
            )

        if inconsistent_values:
            routes.append(
                _route(
                    route_name,
                    provider,
                    account=account,
                    usage=usage,
                    limit=limit,
                    remaining=remaining,
                    remaining_percent=None,
                    unit=unit,
                    reset_at=reset_at,
                    detail="Provider returned an internally inconsistent or zero-limit quota value.",
                )
            )
            continue

        remaining_percent = reported_percent if reported_percent is not None else _remaining_percent(limit, remaining)
        if remaining_percent is None:
            routes.append(
                _route(
                    route_name,
                    provider,
                    account=account,
                    usage=usage,
                    limit=limit,
                    remaining=remaining,
                    unit=unit,
                    reset_at=reset_at,
                    detail="Provider did not report a valid remaining value or percentage.",
                )
            )
            continue

        routes.append(
            _route(
                route_name,
                provider,
                account=account,
                usage=usage,
                limit=limit,
                remaining=remaining,
                remaining_percent=remaining_percent,
                unit=unit,
                reset_at=reset_at,
                status="reported",
                detail="Provider did not specify a quota unit." if not unit else None,
            )
        )
    return tuple(routes)


def _unavailable(
    reason: str,
    scope: Optional[str],
    *,
    routes: tuple[AccountUsageRoute, ...] = (),
    partial: bool = False,
) -> AccountUsageSnapshot:
    return AccountUsageSnapshot(
        provider=NINEROUTER_PROVIDER,
        source=NINEROUTER_SOURCE,
        fetched_at=_utc_now(),
        title="9Router usage & quota",
        routes=routes,
        unavailable_reason=reason,
        scope=_safe_scope(scope),
        partial=partial,
    )


def fetch_ninerouter_account_usage(
    *,
    base_url: Optional[str],
    management_base_url: Optional[str] = None,
    auth_cookie: Optional[str] = None,
    cli_token: Optional[str] = None,
    inference_api_key: Optional[str] = None,
    scope: Optional[str] = None,
) -> AccountUsageSnapshot:
    """Fetch 9Router's independently reported connection quotas.

    ``base_url`` is the OpenAI-compatible inference endpoint.  Its API key is
    deliberately accepted only for call-site clarity and is never sent to the
    management API: 9Router authenticates these routes with its own management
    session or its local CLI token.  A separate, explicitly scoped cookie or
    CLI token may be supplied when the caller has one; neither is inferred from
    a browser or another Hermes profile.
    """

    # Keep the argument explicit so callers cannot accidentally omit the
    # inference/management credential boundary while refactoring this code.
    del inference_api_key
    scope = _safe_scope(scope)
    base = _management_base_url(base_url, management_base_url)
    if not base:
        return _unavailable(
            "9Router management endpoint is not configured for this provider endpoint.",
            scope,
        )
    numeric_base = _numeric_loopback_base_url(base)
    if not numeric_base:
        return _unavailable(
            "9Router management requests are restricted to a loopback management endpoint.",
            scope,
        )
    base = numeric_base

    headers = _headers(auth_cookie, cli_token)
    routes: list[AccountUsageRoute] = []
    partial = False
    try:
        with httpx.Client(timeout=10.0, follow_redirects=False, trust_env=False) as client:
            connections_payload = _get_json(client, f"{base}/api/providers", headers)
            connections = connections_payload.get("connections")
            if not isinstance(connections, list):
                return _unavailable(
                    "9Router management response did not contain a connection list.",
                    scope,
                )
            if not connections:
                return _unavailable("9Router returned no provider connections.", scope)

            provider_counts: dict[str, int] = {}
            used_account_labels: dict[str, set[str]] = {}
            account_label_occurrences: dict[str, dict[str, int]] = {}
            invalid_connection_count = 0
            for raw_connection in connections:
                if not isinstance(raw_connection, dict):
                    partial = True
                    invalid_connection_count += 1
                    routes.append(
                        _route(
                            "Unknown route",
                            None,
                            account=f"Unknown connection #{invalid_connection_count}",
                            status="unavailable",
                            detail="9Router returned an invalid connection record.",
                        )
                    )
                    continue
                provider = _safe_label(raw_connection.get("provider"), "Unknown provider")
                provider_counts[provider] = provider_counts.get(provider, 0) + 1
                connection_label = f"{provider} #{provider_counts[provider]}"
                provider_used = used_account_labels.setdefault(provider, set())
                provider_occs = account_label_occurrences.setdefault(provider, {})
                account_label = _disambiguate_account_label(
                    raw_connection,
                    provider,
                    provider_counts[provider],
                    provider_used,
                    provider_occs,
                )
                connection_id = raw_connection.get("id")
                if not isinstance(connection_id, str) or not connection_id.strip() or len(connection_id) > 512:
                    partial = True
                    routes.append(
                        _route(
                            connection_label,
                            provider,
                            account=account_label,
                            status="unavailable",
                            detail="9Router did not return a valid connection identifier.",
                        )
                    )
                    continue
                # Quote the identifier as a single path segment and never
                # accept a slash from the management response.
                usage_url = f"{base}/api/usage/{quote(connection_id.strip(), safe='')}"
                try:
                    usage_payload = _get_json(client, usage_url, headers)
                except _ManagementAuthRequired:
                    partial = True
                    routes.append(
                        _route(
                            connection_label,
                            provider,
                            account=account_label,
                            status="unavailable",
                            detail="9Router management authentication is required for this route.",
                        )
                    )
                    continue
                except _ManagementRequestFailed as exc:
                    partial = True
                    status_detail = (
                        "9Router route quota endpoint was unavailable."
                        if exc.status_code == 0
                        else f"9Router route quota endpoint returned HTTP {exc.status_code}."
                    )
                    routes.append(
                        _route(
                            connection_label,
                            provider,
                            account=account_label,
                            status="unavailable",
                            detail=status_detail,
                        )
                    )
                    continue
                except _ManagementResponseInvalid:
                    partial = True
                    routes.append(
                        _route(
                            connection_label,
                            provider,
                            account=account_label,
                            status="unknown",
                            detail="9Router returned an invalid quota response.",
                        )
                    )
                    continue

                parsed_routes = _parse_quota_routes(connection_label, provider, usage_payload, account_label)
                routes.extend(parsed_routes)
                if any(route.status != "reported" for route in parsed_routes):
                    partial = True
    except _ManagementAuthRequired:
        return _unavailable(
            "9Router management authentication is required; "
            "the OpenAI-compatible inference credential was not sent.",
            scope,
        )
    except _ManagementRequestFailed as exc:
        status_detail = (
            "9Router management endpoint was unavailable."
            if exc.status_code == 0
            else f"9Router management endpoint returned HTTP {exc.status_code}."
        )
        return _unavailable(status_detail, scope)
    except _ManagementResponseInvalid:
        return _unavailable("9Router management response was invalid.", scope)
    except Exception:
        # Do not propagate provider response text into the dashboard; upstream
        # errors can contain account identifiers or credential-like material.
        return _unavailable("9Router management request failed.", scope)

    route_tuple = tuple(routes)
    if not route_tuple:
        return _unavailable("9Router returned no usable route records.", scope)
    reported = any(route.status == "reported" for route in route_tuple)
    return AccountUsageSnapshot(
        provider=NINEROUTER_PROVIDER,
        source=NINEROUTER_SOURCE,
        fetched_at=_utc_now(),
        title="9Router usage & quota",
        routes=route_tuple,
        unavailable_reason=None if reported else (
            "9Router returned route data but no quota row contained a complete "
            "usage/limit/remaining contract."
        ),
        scope=scope,
        partial=partial or not reported,
    )
