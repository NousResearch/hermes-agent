"""Show OpenAI Codex subscription remaining limits.

This script normalizes the same quota data shown by the Codex app's
"Rate limits remaining" panel: a short 5h window and a weekly/7d window.

Sources, in order:
1. Codex App Server JSON-RPC: account/rateLimits/read
2. ChatGPT backend fallback: https://chatgpt.com/backend-api/wham/usage

It never prints raw tokens or auth JSON.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import select
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from hermes_constants import get_hermes_home

CHATGPT_USAGE_URL = "https://chatgpt.com/backend-api/wham/usage"


@dataclass
class AuthMaterial:
    access_token: str | None = None
    account_id: str | None = None
    source: str | None = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def iso_from_unix(value: Any) -> str | None:
    if value is None:
        return None
    try:
        ts = float(value)
    except (TypeError, ValueError):
        return None
    # Some callers may accidentally provide milliseconds.
    if ts > 10_000_000_000:
        ts = ts / 1000.0
    return datetime.fromtimestamp(ts, timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def clamp_percent(value: float | None) -> float | None:
    if value is None:
        return None
    return round(max(0.0, min(100.0, value)), 2)


def label_for_window(minutes: float | None, fallback: str) -> str:
    if minutes is None:
        return fallback
    if abs(minutes - 300) < 1:
        return "5h"
    if abs(minutes - 10080) < 1:
        return "Weekly"
    if minutes < 60:
        return f"{int(minutes)}m"
    if minutes % 1440 == 0:
        days = int(minutes / 1440)
        return f"{days}d"
    if minutes % 60 == 0:
        hours = int(minutes / 60)
        return f"{hours}h"
    return fallback


def normalize_window(
    raw: dict[str, Any] | None,
    *,
    style: str,
    fallback_label: str,
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {
            "label": fallback_label,
            "remaining_percent": None,
            "used_percent": None,
            "reset_at": None,
            "window_minutes": None,
        }

    if style == "app_server":
        used = as_float(raw.get("usedPercent"))
        reset = raw.get("resetsAt")
        minutes = as_float(raw.get("windowDurationMins"))
    else:
        used = as_float(raw.get("used_percent"))
        reset = raw.get("reset_at")
        seconds = as_float(raw.get("limit_window_seconds"))
        minutes = seconds / 60.0 if seconds is not None else None

    remaining = clamp_percent(100.0 - used) if used is not None else None
    used = clamp_percent(used)
    return {
        "label": label_for_window(minutes, fallback_label),
        "remaining_percent": remaining,
        "used_percent": used,
        "reset_at": iso_from_unix(reset),
        "window_minutes": int(minutes) if minutes is not None and minutes.is_integer() else minutes,
    }


def normalize_app_server(payload: dict[str, Any]) -> dict[str, Any]:
    buckets: list[dict[str, Any]] = []

    by_id = payload.get("rateLimitsByLimitId")
    if isinstance(by_id, dict):
        # Preserve the base codex bucket first when present, then stable-sort extras.
        keys = list(by_id.keys())
        keys.sort(key=lambda k: (0 if k == "codex" else 1, str(k)))
        for key in keys:
            item = by_id.get(key)
            if isinstance(item, dict):
                buckets.append(item)

    primary_bucket = payload.get("rateLimits")
    if isinstance(primary_bucket, dict):
        limit_id = primary_bucket.get("limitId")
        if not buckets or all(bucket.get("limitId") != limit_id for bucket in buckets):
            buckets.insert(0, primary_bucket)

    if not buckets:
        raise RuntimeError("App Server response did not contain rateLimits data")

    out_limits = []
    for bucket in buckets:
        limit_id = bucket.get("limitId")
        raw_name = bucket.get("limitName") or limit_id or "Rate limits remaining"
        name = "Rate limits remaining" if limit_id in (None, "codex") else str(raw_name)
        five_h = normalize_window(bucket.get("primary"), style="app_server", fallback_label="5h")
        week = normalize_window(bucket.get("secondary"), style="app_server", fallback_label="Weekly")
        if five_h["remaining_percent"] is None and week["remaining_percent"] is None:
            continue
        out_limits.append(
            {
                "name": name,
                "limit_id": limit_id,
                "plan_type": bucket.get("planType") or payload.get("planType"),
                "five_h": five_h,
                "week": week,
            }
        )

    if not out_limits:
        raise RuntimeError("App Server response contained no usable rate limit windows")

    return {
        "source": {"provider": "app_server", "captured_at": utc_now_iso()},
        "rate_limits": out_limits,
    }


def normalize_wham(payload: dict[str, Any]) -> dict[str, Any]:
    rate_limit = payload.get("rate_limit")
    if not isinstance(rate_limit, dict):
        raise RuntimeError("wham/usage response did not contain rate_limit")
    five_h = normalize_window(rate_limit.get("primary_window"), style="wham", fallback_label="5h")
    week = normalize_window(rate_limit.get("secondary_window"), style="wham", fallback_label="Weekly")
    if five_h["remaining_percent"] is None and week["remaining_percent"] is None:
        raise RuntimeError("wham/usage response contained no usable rate limit windows")
    return {
        "source": {"provider": "codex_wham", "captured_at": utc_now_iso()},
        "rate_limits": [
            {
                "name": "Rate limits remaining",
                "limit_id": "codex",
                "plan_type": payload.get("plan_type"),
                "five_h": five_h,
                "week": week,
            }
        ],
    }


def codex_app_server_command(raw_command: str | None) -> list[str]:
    command = shlex.split(raw_command or "codex") or ["codex"]
    if command[0] == "codex":
        resolved = shutil.which("codex")
        macos_bundled = "/Applications/Codex.app/Contents/Resources/codex"
        if resolved:
            command[0] = resolved
        elif Path(macos_bundled).exists():
            command[0] = macos_bundled
    return command + ["app-server", "--listen", "stdio://"]


def send_jsonl(proc: subprocess.Popen[str], payload: dict[str, Any]) -> None:
    if proc.stdin is None:
        raise RuntimeError("Codex app-server stdin is unavailable")
    proc.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
    proc.stdin.flush()


def fetch_app_server(timeout: int, command: str | None = None) -> dict[str, Any]:
    cmd = codex_app_server_command(command)
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Codex app-server command was not found: {cmd[0]}") from exc

    try:
        send_jsonl(
            proc,
            {
                "method": "initialize",
                "id": 0,
                "params": {
                    "clientInfo": {
                        "name": "codex-subscription-limits",
                        "title": "Codex Subscription Limits",
                        "version": "1.0.0",
                    }
                },
            },
        )
        send_jsonl(proc, {"method": "initialized", "params": {}})
        send_jsonl(proc, {"method": "account/rateLimits/read", "id": 1, "params": None})

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                stderr = proc.stderr.read() if proc.stderr else ""
                raise RuntimeError(f"Codex app-server exited early with code {proc.returncode}: {stderr.strip()}")

            remaining = max(deadline - time.monotonic(), 0.0)
            ready, _, _ = select.select([proc.stdout], [], [], min(0.25, remaining))
            if not ready:
                continue
            line = proc.stdout.readline() if proc.stdout else ""
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(message, dict) or message.get("id") != 1:
                continue
            if isinstance(message.get("error"), dict):
                err = message["error"]
                raise RuntimeError(err.get("message") or json.dumps(err, ensure_ascii=False))
            result = message.get("result")
            if not isinstance(result, dict):
                raise RuntimeError("Codex app-server rate limit response had no result object")
            return result
        raise RuntimeError("Timed out waiting for Codex app-server rate limit response")
    finally:
        if proc.poll() is None:
            proc.kill()
        try:
            proc.communicate(timeout=2)
        except Exception:
            pass


def iter_dict_values(obj: Any) -> Iterable[Any]:
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from iter_dict_values(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from iter_dict_values(value)


def load_json_file(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def chatgpt_account_id_from_jwt(access_token: str | None) -> str | None:
    """Extract the ChatGPT account id claim without logging or returning the token."""
    if not isinstance(access_token, str) or not access_token.strip():
        return None
    try:
        parts = access_token.split(".")
        if len(parts) < 2:
            return None
        payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload_b64))
        account_id = claims.get("https://api.openai.com/auth", {}).get("chatgpt_account_id")
        return account_id if isinstance(account_id, str) and account_id.strip() else None
    except Exception:
        return None


def default_auth_paths() -> list[Path]:
    codex_home = os.getenv("CODEX_HOME", "").strip()
    codex_auth = Path(codex_home).expanduser() / "auth.json" if codex_home else Path.home() / ".codex" / "auth.json"
    return [codex_auth, get_hermes_home() / "auth.json"]


def discover_auth(paths: list[Path]) -> AuthMaterial:
    for path in paths:
        payload = load_json_file(path.expanduser())
        if payload is None:
            continue
        token = None
        account_id = None
        for item in iter_dict_values(payload):
            if token is None:
                for key in ("access_token", "accessToken", "access"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        token = value.strip()
                        break
            if account_id is None:
                for key in ("account_id", "accountId", "chatgpt_account_id", "chatgptAccountId"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        account_id = value.strip()
                        break
            if token and account_id:
                break
        if token:
            return AuthMaterial(
                access_token=token,
                account_id=account_id or chatgpt_account_id_from_jwt(token),
                source=str(path.expanduser()),
            )
    return AuthMaterial()


def resolve_hermes_codex_auth(*, force_refresh: bool = False) -> AuthMaterial:
    """Resolve a fresh Codex token from Hermes auth, refreshing if needed.

    Prefer the multi-credential pool because normal OpenAI-Codex inference uses
    it too.  Falling back to the singleton provider store keeps compatibility
    with older installs that have not migrated to credential pools yet.
    """
    try:
        from agent.credential_pool import load_pool

        pool = load_pool("openai-codex")
        entry = pool.select()
        if force_refresh and entry is not None:
            refreshed = pool.try_refresh_current()
            if refreshed is not None:
                entry = refreshed
        if entry is not None:
            token = str(entry.access_token or "").strip()
            if token:
                label = str(entry.label or entry.id or "pool").strip()
                return AuthMaterial(
                    access_token=token,
                    account_id=chatgpt_account_id_from_jwt(token),
                    source=f"credential-pool:{label}",
                )
    except Exception:
        pass

    try:
        from hermes_cli.auth import resolve_codex_runtime_credentials

        creds = resolve_codex_runtime_credentials(force_refresh=force_refresh)
    except Exception:
        return AuthMaterial()
    token = creds.get("api_key")
    if not isinstance(token, str) or not token.strip():
        return AuthMaterial()
    return AuthMaterial(
        access_token=token.strip(),
        account_id=chatgpt_account_id_from_jwt(token),
        source=str(creds.get("source") or "hermes-auth-store"),
    )


def fetch_wham(timeout: int, usage_url: str, auth_paths: list[Path] | None = None) -> dict[str, Any]:
    paths = auth_paths if auth_paths is not None else default_auth_paths()
    auth = resolve_hermes_codex_auth() or AuthMaterial()
    if not auth.access_token:
        auth = discover_auth(paths)
    if not auth.access_token:
        searched = ", ".join(str(p.expanduser()) for p in paths)
        raise RuntimeError(f"No Codex OAuth access token found. Searched: {searched}")

    headers = {
        "Authorization": f"Bearer {auth.access_token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "codex_cli_rs/0.0.0 (Hermes Agent)",
        "originator": "codex_cli_rs",
    }
    if auth.account_id:
        headers["ChatGPT-Account-ID"] = auth.account_id

    request = urllib.request.Request(usage_url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            refreshed_auth = resolve_hermes_codex_auth(force_refresh=True)
            if refreshed_auth.access_token:
                retry_headers = dict(headers)
                retry_headers["Authorization"] = f"Bearer {refreshed_auth.access_token}"
                retry_headers.pop("ChatGPT-Account-ID", None)
                if refreshed_auth.account_id:
                    retry_headers["ChatGPT-Account-ID"] = refreshed_auth.account_id
                retry_request = urllib.request.Request(usage_url, headers=retry_headers, method="GET")
                try:
                    with urllib.request.urlopen(retry_request, timeout=timeout) as response:
                        payload = json.loads(response.read().decode("utf-8"))
                except urllib.error.HTTPError as retry_exc:
                    if retry_exc.code in (401, 403):
                        raise RuntimeError(
                            f"wham/usage rejected the refreshed token with HTTP {retry_exc.code}; refresh Codex login"
                        ) from retry_exc
                    raise RuntimeError(f"wham/usage request failed with HTTP {retry_exc.code}") from retry_exc
            else:
                raise RuntimeError(f"wham/usage rejected the token with HTTP {exc.code}; refresh Codex login") from exc
        else:
            raise RuntimeError(f"wham/usage request failed with HTTP {exc.code}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("wham/usage response was not a JSON object")
    return payload


def fixture_payload(name: str) -> tuple[str, dict[str, Any]]:
    if name == "app-server":
        return "app_server", {
            "rateLimits": {
                "limitId": "codex",
                "limitName": "Codex",
                "planType": "pro",
                "primary": {"usedPercent": 25, "resetsAt": 1770000000, "windowDurationMins": 300},
                "secondary": {"usedPercent": 5, "resetsAt": 1770500000, "windowDurationMins": 10080},
            },
            "rateLimitsByLimitId": {
                "codex": {
                    "limitId": "codex",
                    "limitName": "Codex",
                    "primary": {"usedPercent": 25, "resetsAt": 1770000000, "windowDurationMins": 300},
                    "secondary": {"usedPercent": 5, "resetsAt": 1770500000, "windowDurationMins": 10080},
                },
                "codex_model": {
                    "limitId": "codex_model",
                    "limitName": "GPT-Codex model bucket",
                    "primary": {"usedPercent": 11, "resetsAt": 1770100000, "windowDurationMins": 300},
                    "secondary": {"usedPercent": 6, "resetsAt": 1770600000, "windowDurationMins": 10080},
                },
            },
        }
    if name == "wham":
        return "wham", {
            "plan_type": "pro",
            "rate_limit": {
                "primary_window": {"used_percent": 24, "limit_window_seconds": 18000, "reset_at": 1770000000},
                "secondary_window": {"used_percent": 7, "limit_window_seconds": 604800, "reset_at": 1770500000},
            },
        }
    raise ValueError(f"Unknown fixture: {name}")


def normalize(provider: str, payload: dict[str, Any]) -> dict[str, Any]:
    if provider == "app_server":
        return normalize_app_server(payload)
    if provider == "wham":
        return normalize_wham(payload)
    raise ValueError(provider)


def human_reset(value: str | None) -> str:
    if not value:
        return "unknown"
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone()
        return dt.strftime("%b %d %H:%M")
    except Exception:
        return value


def format_pretty(state: dict[str, Any]) -> str:
    provider = state.get("source", {}).get("provider", "unknown")
    lines = [f"Codex limits remaining ({provider})"]
    for bucket in state.get("rate_limits", []):
        five = bucket.get("five_h", {})
        week = bucket.get("week", {})
        five_pct = five.get("remaining_percent")
        week_pct = week.get("remaining_percent")
        five_text = "5h n/a" if five_pct is None else f"5h {five_pct:g}% reset {human_reset(five.get('reset_at'))}"
        week_text = "Weekly n/a" if week_pct is None else f"Weekly {week_pct:g}% reset {human_reset(week.get('reset_at'))}"
        lines.append(f"{bucket.get('name', 'Rate limits remaining')}: {five_text} · {week_text}")
    return "\n".join(lines)


def get_codex_limits(
    *,
    provider: str = "auto",
    timeout: int = 20,
    codex_command: str | None = None,
    usage_url: str = CHATGPT_USAGE_URL,
    auth_paths: list[Path] | None = None,
) -> dict[str, Any]:
    """Fetch and normalize Codex subscription rate-limit data.

    ``provider`` accepts ``auto``, ``app-server``, or ``wham``. The returned
    structure contains only normalized percentages and reset timestamps; raw
    tokens, auth JSON, and account identifiers are never included.
    """
    if provider in ("auto", "app-server"):
        try:
            return normalize_app_server(fetch_app_server(timeout, codex_command))
        except Exception:
            if provider == "app-server":
                raise
            return normalize_wham(fetch_wham(timeout, usage_url, auth_paths))
    if provider == "wham":
        return normalize_wham(fetch_wham(timeout, usage_url, auth_paths))
    raise ValueError(f"Unknown provider: {provider}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show Codex subscription 5h/weekly remaining limits")
    parser.add_argument("--provider", choices=["auto", "app-server", "wham"], default="auto")
    parser.add_argument("--fixture", choices=["app-server", "wham"], help="Use built-in fixture instead of live network/process calls")
    parser.add_argument("--json", action="store_true", help="Print normalized JSON")
    parser.add_argument("--pretty", action="store_true", help="Print concise human-readable output")
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--codex-command", help="Codex executable/command prefix, default: codex")
    parser.add_argument("--usage-url", default=os.getenv("CODEX_WHAM_USAGE_URL", CHATGPT_USAGE_URL))
    parser.add_argument(
        "--auth-file",
        action="append",
        default=[],
        help="Auth JSON file to search for OAuth token; can be passed multiple times",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if not args.json and not args.pretty:
        args.pretty = True

    try:
        if args.fixture:
            provider, payload = fixture_payload(args.fixture)
            state = normalize(provider, payload)
        else:
            auth_paths = [Path(p) for p in args.auth_file] if args.auth_file else None
            state = get_codex_limits(
                provider=args.provider,
                timeout=args.timeout,
                codex_command=args.codex_command,
                usage_url=args.usage_url,
                auth_paths=auth_paths,
            )

        if args.json:
            print(json.dumps(state, indent=2, ensure_ascii=False, sort_keys=True))
        if args.pretty:
            print(format_pretty(state))
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
