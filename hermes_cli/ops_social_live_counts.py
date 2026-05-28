"""One-shot read-only social platform counting probes for Jenny Ops Center.

This module is deliberately narrow:
- reads existing credentials only
- redacts all credential evidence
- calls only read/list style endpoints
- writes only the local Ops Center social snapshot when explicitly requested
- never posts, schedules, deletes, changes privacy, repairs tokens, or creates cron jobs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from hermes_constants import get_env_path, get_hermes_home
from hermes_cli.ops_social_status import read_social_platform_status, write_manual_social_platform_status

HTTP_TIMEOUT_SECONDS = 12
SOURCE_NAME = "live-read-only-probe"

CredentialMap = Dict[str, Sequence[str]]

PLATFORM_CREDENTIALS: CredentialMap = {
    "youtube": ("YOUTUBE_API_KEY", "YOUTUBE_CHANNEL_ID", "YOUTUBE_ACCESS_TOKEN"),
    "facebook": ("META_ACCESS_TOKEN", "META_PAGE_ACCESS_TOKEN", "META_PAGE_ID", "FACEBOOK_PAGE_ACCESS_TOKEN", "FACEBOOK_PAGE_ID"),
    "instagram": ("INSTAGRAM_ACCESS_TOKEN", "INSTAGRAM_BUSINESS_ACCOUNT_ID", "META_IG_USER_ID", "META_PAGE_ACCESS_TOKEN", "META_ACCESS_TOKEN"),
    "tiktok": ("TIKTOK_ACCESS_TOKEN",),
}

PLATFORM_LABELS = {
    "youtube": "YouTube",
    "facebook": "Facebook",
    "instagram": "Instagram",
    "tiktok": "TikTok",
}

YOUTUBE_TOKEN_FILES = ("youtube_token_signalroom.json", "youtube_token.json")

HttpGet = Callable[[str, Mapping[str, str], int], Dict[str, Any]]
HttpPost = Callable[[str, Mapping[str, str], Mapping[str, str], int], Dict[str, Any]]


def _load_env_file_values(path: Optional[Path] = None) -> Dict[str, str]:
    env_path = path or get_env_path()
    values: Dict[str, str] = {}
    if not env_path.exists():
        return values
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return values
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _env_value(key: str, env_file_values: Mapping[str, str]) -> str:
    return os.environ.get(key) or env_file_values.get(key, "")


def _first_env_value(keys: Sequence[str], env_file_values: Mapping[str, str]) -> str:
    for key in keys:
        value = _env_value(key, env_file_values)
        if value:
            return value
    return ""


def _load_json_file(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _youtube_token_file() -> Optional[Path]:
    explicit = os.environ.get("YOUTUBE_TOKEN_FILE", "").strip()
    candidates = [Path(explicit).expanduser()] if explicit else []
    candidates.extend(get_hermes_home() / name for name in YOUTUBE_TOKEN_FILES)
    for candidate in candidates:
        if candidate.exists():
            data = _load_json_file(candidate)
            if data.get("token"):
                return candidate
    return None


def _youtube_token_file_metadata() -> Dict[str, Any]:
    path = _youtube_token_file()
    if path is None:
        return {"present": False, "source": "missing", "shape": "missing"}
    data = _load_json_file(path)
    return {
        "present": True,
        "source": "token_file",
        "shape": "present" if len(str(data.get("token") or "")) >= 8 else "present_short",
        "path_name": path.name,
        "channel_id_present": bool(data.get("channel_id")),
        "expiry_present": bool(data.get("expiry")),
        "scopes_count": len(data.get("scopes") or []),
    }


def _credential_shape(value: str) -> str:
    if not value:
        return "missing"
    stripped = value.strip()
    if len(stripped) < 8:
        return "present_short"
    if " " in stripped or "\n" in stripped:
        return "present_multivalue"
    return "present"


def credential_inventory(platforms: Iterable[str] = PLATFORM_CREDENTIALS, *, env_file: Optional[Path] = None) -> Dict[str, Any]:
    """Return redacted credential presence metadata only."""

    env_file_values = _load_env_file_values(env_file)
    checked_at = datetime.now(timezone.utc).isoformat()
    result: Dict[str, Any] = {
        "ok": True,
        "mode": "redacted_presence_only",
        "checked_at": checked_at,
        "platforms": [],
        "message": "Credential inventory is redacted. Values are never returned.",
    }
    for platform in _normalize_platforms(platforms):
        keys = PLATFORM_CREDENTIALS[platform]
        credentials = []
        for key in keys:
            file_present = key in env_file_values and bool(env_file_values[key])
            env_present = bool(os.environ.get(key))
            value = _env_value(key, env_file_values)
            credentials.append(
                {
                    "key": key,
                    "present": bool(value),
                    "source": "process_env" if env_present else "env_file" if file_present else "missing",
                    "shape": _credential_shape(value),
                }
            )
        if platform == "youtube":
            credentials.append({"key": "YOUTUBE_TOKEN_FILE", **_youtube_token_file_metadata()})
        result["platforms"].append(
            {
                "platform": PLATFORM_LABELS[platform],
                "key": platform,
                "credentials": credentials,
                "can_attempt_probe": _can_attempt_platform(platform, env_file_values),
            }
        )
    return result


def _normalize_platforms(platforms: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    for raw in platforms:
        item = str(raw or "").strip().lower()
        if item == "all":
            return list(PLATFORM_CREDENTIALS)
        if item in PLATFORM_CREDENTIALS and item not in normalized:
            normalized.append(item)
    return normalized or list(PLATFORM_CREDENTIALS)


def _can_attempt_platform(platform: str, env_file_values: Mapping[str, str]) -> bool:
    if platform == "youtube":
        return bool(_youtube_token_file()) or bool(_env_value("YOUTUBE_ACCESS_TOKEN", env_file_values)) or bool(
            _env_value("YOUTUBE_API_KEY", env_file_values) and _env_value("YOUTUBE_CHANNEL_ID", env_file_values)
        )
    if platform == "facebook":
        return bool(
            _first_env_value(("FACEBOOK_PAGE_ACCESS_TOKEN", "META_PAGE_ACCESS_TOKEN", "META_ACCESS_TOKEN"), env_file_values)
            and _first_env_value(("FACEBOOK_PAGE_ID", "META_PAGE_ID"), env_file_values)
        )
    if platform == "instagram":
        return bool(
            _first_env_value(("INSTAGRAM_ACCESS_TOKEN", "META_PAGE_ACCESS_TOKEN", "META_ACCESS_TOKEN"), env_file_values)
            and _first_env_value(("INSTAGRAM_BUSINESS_ACCOUNT_ID", "META_IG_USER_ID"), env_file_values)
        )
    if platform == "tiktok":
        return bool(_env_value("TIKTOK_ACCESS_TOKEN", env_file_values))
    return False


def _http_get_json(url: str, headers: Mapping[str, str], timeout: int = HTTP_TIMEOUT_SECONDS) -> Dict[str, Any]:
    request = urllib.request.Request(url, headers=dict(headers), method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310 - explicit approved read-only user API probe
            raw = response.read().decode("utf-8", errors="replace")
            return {"ok": True, "status_code": response.status, "json": json.loads(raw)}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        return {"ok": False, "status_code": exc.code, "error": _sanitize_error(body or str(exc))}
    except Exception as exc:
        return {"ok": False, "status_code": None, "error": _sanitize_error(str(exc))}


def _http_post_json(url: str, data: Mapping[str, str], headers: Mapping[str, str], timeout: int = HTTP_TIMEOUT_SECONDS) -> Dict[str, Any]:
    encoded = urllib.parse.urlencode(data).encode("utf-8")
    request = urllib.request.Request(url, data=encoded, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310 - OAuth token endpoint for existing user credential refresh
            raw = response.read().decode("utf-8", errors="replace")
            return {"ok": True, "status_code": response.status, "json": json.loads(raw)}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        return {"ok": False, "status_code": exc.code, "error": _sanitize_error(body or str(exc))}
    except Exception as exc:
        return {"ok": False, "status_code": None, "error": _sanitize_error(str(exc))}


def _token_file_sensitive_values() -> List[str]:
    path = _youtube_token_file()
    if path is None:
        return []
    data = _load_json_file(path)
    values: List[str] = []
    for key, value in data.items():
        if isinstance(value, str) and ("token" in key.lower() or "secret" in key.lower() or "client" in key.lower()):
            values.append(value)
    return values


def _sanitize_error(text: str) -> str:
    cleaned = str(text or "").replace("\n", " ").replace("\r", " ")
    sensitive_values = _token_file_sensitive_values()
    for key in {key for keys in PLATFORM_CREDENTIALS.values() for key in keys}:
        value = os.environ.get(key)
        if value:
            sensitive_values.append(value)
    for value in sensitive_values:
        if value:
            cleaned = cleaned.replace(value, "[redacted]")
    return cleaned[:500]


def _response_error(response: Mapping[str, Any]) -> str:
    return _sanitize_error(str(response.get("error") or "API unavailable"))


def _row(platform: str, *, published: Any = "Needs sync", scheduled: Any = "Needs sync", issues: str, readiness: str, status: str, source: str = SOURCE_NAME) -> Dict[str, Any]:
    return {
        "platform": PLATFORM_LABELS.get(platform, platform.title()),
        "published": published,
        "scheduled": scheduled,
        "issues_private": issues,
        "readiness": readiness,
        "source": source,
        "status": status,
        "last_checked_at": datetime.now(timezone.utc).isoformat(),
    }


def _token_is_expired(token_data: Mapping[str, Any]) -> bool:
    expiry = str(token_data.get("expiry") or "").strip()
    if not expiry:
        return False
    try:
        expires_at = datetime.fromisoformat(expiry.replace("Z", "+00:00"))
    except ValueError:
        return False
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) >= expires_at - timedelta(seconds=60)


def _refresh_youtube_access_token(token_data: Mapping[str, Any], http_post: HttpPost) -> Dict[str, Any]:
    refresh_token = str(token_data.get("refresh_token") or "")
    client_id = str(token_data.get("client_id") or "")
    client_secret = str(token_data.get("client_secret") or "")
    token_uri = str(token_data.get("token_uri") or "https://oauth2.googleapis.com/token")
    if not (refresh_token and client_id and client_secret):
        return {"ok": False, "error": "Token file cannot refresh because OAuth client fields are missing."}
    return http_post(
        token_uri,
        {
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        },
        {"Content-Type": "application/x-www-form-urlencoded"},
        HTTP_TIMEOUT_SECONDS,
    )


def _probe_youtube(env_values: Mapping[str, str], http_get: HttpGet, http_post: HttpPost = _http_post_json) -> Dict[str, Any]:
    token_path = _youtube_token_file()
    token_data = _load_json_file(token_path) if token_path else {}
    access_token = _env_value("YOUTUBE_ACCESS_TOKEN", env_values) or str(token_data.get("token") or "")
    api_key = _env_value("YOUTUBE_API_KEY", env_values)
    channel_id = _env_value("YOUTUBE_CHANNEL_ID", env_values) or str(token_data.get("channel_id") or "")
    if access_token and token_data and _token_is_expired(token_data):
        refresh_response = _refresh_youtube_access_token(token_data, http_post)
        if refresh_response.get("ok"):
            access_token = str(refresh_response.get("json", {}).get("access_token") or access_token)
        else:
            return _row("youtube", issues=f"Read-only YouTube token refresh failed: {refresh_response.get('status_code') or 'error'}", readiness=_response_error(refresh_response), status="needs_review")
    if access_token:
        params = urllib.parse.urlencode({"part": "statistics,snippet", "mine": "true"})
        response = http_get(f"https://www.googleapis.com/youtube/v3/channels?{params}", {"Authorization": f"Bearer {access_token}"}, HTTP_TIMEOUT_SECONDS)
    elif api_key and channel_id:
        params = urllib.parse.urlencode({"part": "statistics", "id": channel_id, "key": api_key})
        response = http_get(f"https://www.googleapis.com/youtube/v3/channels?{params}", {}, HTTP_TIMEOUT_SECONDS)
    else:
        return _row("youtube", issues="Missing existing YouTube credential/channel configuration.", readiness="Read-only live count probe not attempted.", status="not_connected")
    if not response.get("ok"):
        return _row("youtube", issues=f"Read-only YouTube probe failed: {response.get('status_code') or 'error'}", readiness=_response_error(response), status="needs_review")
    items = response.get("json", {}).get("items") or []
    stats = (items[0].get("statistics") if items else {}) or {}
    return _row("youtube", published=stats.get("videoCount", "Needs sync"), scheduled="Needs sync", issues="Live read-only channel statistics retrieved; scheduled count not available from this probe.", readiness="Existing credential supported read-only YouTube statistics.", status="ok")


def _probe_facebook(env_values: Mapping[str, str], http_get: HttpGet) -> Dict[str, Any]:
    token = _first_env_value(("FACEBOOK_PAGE_ACCESS_TOKEN", "META_PAGE_ACCESS_TOKEN", "META_ACCESS_TOKEN"), env_values)
    page_id = _first_env_value(("FACEBOOK_PAGE_ID", "META_PAGE_ID"), env_values)
    if not (token and page_id):
        return _row("facebook", issues="Missing existing Facebook page token/page id configuration.", readiness="Read-only live count probe not attempted.", status="not_connected")
    fields = "fan_count,followers_count,instagram_business_account{id,username,media_count,followers_count}"
    params = urllib.parse.urlencode({"fields": fields, "access_token": token})
    response = http_get(f"https://graph.facebook.com/v19.0/{urllib.parse.quote(page_id)}?{params}", {}, HTTP_TIMEOUT_SECONDS)
    if not response.get("ok"):
        return _row("facebook", issues=f"Read-only Facebook probe failed: {response.get('status_code') or 'error'}", readiness=_response_error(response), status="needs_review")
    data = response.get("json", {})
    return _row("facebook", published=data.get("followers_count") or data.get("fan_count") or "Needs sync", scheduled="Needs sync", issues="Live read-only page counts retrieved; scheduled queue count not available from this probe.", readiness="Existing credential supported read-only Facebook page metadata.", status="ok")


def _probe_instagram(env_values: Mapping[str, str], http_get: HttpGet) -> Dict[str, Any]:
    token = _first_env_value(("INSTAGRAM_ACCESS_TOKEN", "META_PAGE_ACCESS_TOKEN", "META_ACCESS_TOKEN"), env_values)
    account_id = _first_env_value(("INSTAGRAM_BUSINESS_ACCOUNT_ID", "META_IG_USER_ID"), env_values)
    if not (token and account_id):
        return _row("instagram", issues="Missing existing Instagram business token/account id configuration.", readiness="Read-only live count probe not attempted.", status="not_connected")
    params = urllib.parse.urlencode({"fields": "media_count,followers_count,username", "access_token": token})
    response = http_get(f"https://graph.facebook.com/v19.0/{urllib.parse.quote(account_id)}?{params}", {}, HTTP_TIMEOUT_SECONDS)
    if not response.get("ok"):
        return _row("instagram", issues=f"Read-only Instagram probe failed: {response.get('status_code') or 'error'}", readiness=_response_error(response), status="needs_review")
    data = response.get("json", {})
    return _row("instagram", published=data.get("media_count", "Needs sync"), scheduled="Needs sync", issues="Live read-only media count retrieved; scheduler readiness still separate.", readiness="Existing credential supported read-only Instagram business metadata.", status="ok")


def _probe_tiktok(env_values: Mapping[str, str], http_get: HttpGet) -> Dict[str, Any]:
    token = _env_value("TIKTOK_ACCESS_TOKEN", env_values)
    if not token:
        return _row("tiktok", published="0", scheduled="0", issues="Missing existing TikTok access token configuration.", readiness="Read-only live count probe not attempted.", status="not_connected")
    fields = "open_id,display_name,is_verified,follower_count,likes_count,video_count"
    response = http_get(f"https://open.tiktokapis.com/v2/user/info/?fields={urllib.parse.quote(fields)}", {"Authorization": f"Bearer {token}"}, HTTP_TIMEOUT_SECONDS)
    if not response.get("ok"):
        return _row("tiktok", issues=f"Read-only TikTok probe failed: {response.get('status_code') or 'error'}", readiness=_response_error(response), status="needs_review")
    user = ((response.get("json", {}).get("data") or {}).get("user") or {})
    return _row("tiktok", published=user.get("video_count", "Needs sync"), scheduled="0", issues="Live read-only TikTok user count retrieved; scheduling/posting remains gated.", readiness="Existing credential supported read-only TikTok user info.", status="ok")


PROBERS: Dict[str, Callable[[Mapping[str, str], HttpGet], Dict[str, Any]]] = {
    "youtube": _probe_youtube,
    "facebook": _probe_facebook,
    "instagram": _probe_instagram,
    "tiktok": _probe_tiktok,
}


def probe_social_counts(
    platforms: Iterable[str] = ("all",),
    *,
    env_file: Optional[Path] = None,
    http_get: HttpGet = _http_get_json,
    http_post: HttpPost = _http_post_json,
    write_snapshot: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    env_values = _load_env_file_values(env_file)
    selected = _normalize_platforms(platforms)
    inventory = credential_inventory(selected, env_file=env_file)
    rows: List[Dict[str, Any]] = []
    for platform in selected:
        if dry_run:
            can_attempt = next((item.get("can_attempt_probe") for item in inventory["platforms"] if item.get("key") == platform), False)
            rows.append(_row(platform, issues="Dry run only; no platform API call performed.", readiness="Existing credentials look sufficient for a probe." if can_attempt else "Existing credentials missing for probe.", status="needs_sync" if can_attempt else "not_connected"))
            continue
        if platform == "youtube":
            rows.append(_probe_youtube(env_values, http_get, http_post))
        else:
            rows.append(PROBERS[platform](env_values, http_get))

    result: Dict[str, Any] = {
        "ok": True,
        "mode": "live_read_only_probe",
        "dry_run": dry_run,
        "write_snapshot": write_snapshot,
        "source": SOURCE_NAME,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "credential_inventory": inventory,
        "platforms": rows,
        "snapshot": None,
        "forbidden_actions": ["post", "upload", "schedule", "delete", "privacy_change", "token_repair", "cron_create"],
    }
    if write_snapshot and not dry_run:
        existing_rows = read_social_platform_status().get("platforms", [])
        by_platform = {
            str(item.get("platform", "")).lower(): item
            for item in existing_rows
            if isinstance(item, dict) and item.get("platform")
        }
        for row in rows:
            by_platform[str(row.get("platform", "")).lower()] = row
        result["snapshot"] = write_manual_social_platform_status(
            {
                "source": SOURCE_NAME,
                "platforms": list(by_platform.values()),
            }
        )
    return result


def _print_json(data: Mapping[str, Any]) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


def cli_main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run redacted, read-only social platform count probes for Jenny Ops Center.")
    sub = parser.add_subparsers(dest="command", required=True)
    cred = sub.add_parser("credentials", help="Show redacted credential presence only")
    cred.add_argument("--platform", action="append", help="Platform to inspect: youtube/facebook/instagram/tiktok/all")
    cred.add_argument("--env-file", type=Path, help="Read credentials from this env file instead of the active Hermes .env")
    probe = sub.add_parser("probe", help="Run one-shot read-only platform count probe")
    probe.add_argument("--platform", action="append", help="Platform to probe: youtube/facebook/instagram/tiktok/all")
    probe.add_argument("--env-file", type=Path, help="Read credentials from this env file instead of the active Hermes .env")
    probe.add_argument("--write-snapshot", action="store_true", help="Write successful probe rows to local Ops Center snapshot/history")
    probe.add_argument("--dry-run", action="store_true", help="Do not call platform APIs; report what would be attempted")
    args = parser.parse_args(argv)

    if args.command == "credentials":
        _print_json(credential_inventory(args.platform or ["all"], env_file=args.env_file))
        return 0
    if args.command == "probe":
        _print_json(probe_social_counts(args.platform or ["all"], env_file=args.env_file, write_snapshot=args.write_snapshot, dry_run=args.dry_run))
        return 0
    parser.error("unknown command")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main(sys.argv[1:]))
