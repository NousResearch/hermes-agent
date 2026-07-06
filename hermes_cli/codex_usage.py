"""OpenAI Codex quota usage helpers for Hermes.

This module powers both the `hermes codex-usage` CLI command and local watchdog
scripts. It reads Hermes' `openai-codex` credential pool and calls the Codex
quota endpoint once per credential. Runtime OAuth tokens are never printed.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from hermes_cli.config import get_hermes_home

USAGE_URL = "https://chatgpt.com/backend-api/wham/usage"
DEFAULT_WATERMARK = get_hermes_home() / "state" / "codex_usage_alerts.json"


def local_dt(epoch: Any) -> Optional[datetime]:
    if isinstance(epoch, str):
        try:
            return datetime.fromisoformat(epoch).astimezone()
        except ValueError:
            return None
    if not isinstance(epoch, (int, float)):
        return None
    return datetime.fromtimestamp(epoch).astimezone()


def human_delta(target: Optional[datetime], now: Optional[datetime] = None) -> str:
    if target is None:
        return "unknown"
    now = now or datetime.now().astimezone()
    seconds = int((target - now).total_seconds())
    if seconds <= 0:
        return "already reset"
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes or not parts:
        parts.append(f"{minutes}m")
    return " ".join(parts)


def risk(used_percent: Any) -> Dict[str, str]:
    try:
        value = float(used_percent)
    except (TypeError, ValueError):
        return {"level": "unknown", "icon": "⚪", "label": "unknown"}
    if value >= 95:
        return {"level": "critical", "icon": "🔴", "label": "거의 소진"}
    if value >= 80:
        return {"level": "warning", "icon": "🟠", "label": "주의"}
    return {"level": "ok", "icon": "🟢", "label": "정상"}


def summarize_window(window: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(window, dict):
        return None
    reset_dt = local_dt(window.get("reset_at"))
    used = window.get("used_percent")
    return {
        "used_percent": used,
        "limit_window_seconds": window.get("limit_window_seconds"),
        "reset_at": reset_dt.isoformat(timespec="seconds") if reset_dt else window.get("reset_at"),
        "remaining": human_delta(reset_dt),
        "risk": risk(used),
    }


# Backward-compatible alias used by the first local helper script.
window_summary = summarize_window


def fetch_usage(access_token: str, account_id: Optional[str] = None, timeout: int = 30) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "Hermes Codex usage check",
        "Accept": "application/json",
    }
    if account_id:
        headers["ChatGPT-Account-Id"] = account_id
    req = urllib.request.Request(USAGE_URL, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def collect() -> Dict[str, Any]:
    from agent.credential_pool import load_pool

    pool = load_pool("openai-codex")
    # clear_expired=True + refresh=True lets Hermes refresh OAuth tokens if needed.
    entries = pool._available_entries(clear_expired=True, refresh=True)  # intentional internal API
    now = datetime.now().astimezone()
    rows: list[dict[str, Any]] = []
    for entry in entries:
        row: Dict[str, Any] = {
            "label": entry.label,
            "priority": entry.priority,
            "source": entry.source,
            "last_status": entry.last_status or "ok",
        }
        account_id = (
            entry.extra.get("account_id")
            or entry.extra.get("chatgpt_account_id")
            or entry.extra.get("accountId")
        )
        try:
            data = fetch_usage(entry.runtime_api_key, account_id=account_id)
            rate_limit = data.get("rate_limit") or {}
            credits = data.get("credits") or {}
            row.update(
                {
                    "ok": True,
                    "plan_type": data.get("plan_type"),
                    "primary_window": summarize_window(rate_limit.get("primary_window")),
                    "secondary_window": summarize_window(rate_limit.get("secondary_window")),
                    "credits": {
                        key: credits.get(key)
                        for key in ("has_credits", "unlimited", "balance")
                        if key in credits
                    },
                }
            )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:500]
            row.update({"ok": False, "http": exc.code, "error": body})
        except Exception as exc:  # noqa: BLE001 - CLI report should survive per-account failures
            row.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        rows.append(row)
    payload = {"checked_at": now.isoformat(timespec="seconds"), "provider": "openai-codex", "accounts": rows}
    payload["recommendation"] = compute_recommendation(rows)
    return payload


def _percent(row: Dict[str, Any], window_name: str) -> float:
    try:
        value = (row.get(window_name) or {}).get("used_percent")
        if value is None:
            return 999.0
        return float(value)
    except (TypeError, ValueError):
        return 999.0


def compute_recommendation(accounts: List[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    ok_accounts = [row for row in accounts if row.get("ok")]
    if not ok_accounts:
        return None
    best = min(ok_accounts, key=lambda row: (_percent(row, "secondary_window"), _percent(row, "primary_window")))
    sec = best.get("secondary_window") or {}
    pri = best.get("primary_window") or {}
    reason = f"7d {sec.get('used_percent', '?')}%, 5h {pri.get('used_percent', '?')}%"
    return {"label": str(best.get("label")), "reason": reason}


# Backward-compatible alias.
recommend = compute_recommendation


def iter_windows(row: Dict[str, Any]) -> Iterable[Tuple[str, str, Dict[str, Any]]]:
    yield "5h", "primary_window", row.get("primary_window") or {}
    yield "7d", "secondary_window", row.get("secondary_window") or {}


def render_text(payload: Dict[str, Any]) -> str:
    lines = [f"Codex usage ({payload['checked_at']})"]
    accounts = payload.get("accounts") or []
    if not accounts:
        lines.append("No available openai-codex credentials found.")
        return "\n".join(lines)
    recommendation = payload.get("recommendation") or {}
    if recommendation:
        lines.append(f"추천: {recommendation.get('label')} ({recommendation.get('reason')})")
    for row in accounts:
        lines.append("")
        lines.append(f"[{row.get('label')}] plan={row.get('plan_type', 'unknown')} status={row.get('last_status', 'unknown')}")
        if not row.get("ok"):
            lines.append(f"  ERROR: {row.get('http', '')} {row.get('error', '')}".rstrip())
            continue
        for display, _key, window in iter_windows(row):
            r = window.get("risk") or risk(window.get("used_percent"))
            lines.append(
                f"  {display}: {r.get('icon')} "
                f"{window.get('used_percent', '?')}% used ({r.get('label')}), "
                f"reset {window.get('reset_at', '?')} ({window.get('remaining', '?')})"
            )
        credits = row.get("credits") or {}
        if credits:
            lines.append(
                "  credits: "
                f"has={credits.get('has_credits')}, "
                f"unlimited={credits.get('unlimited')}, "
                f"balance={credits.get('balance')}"
            )
    return "\n".join(lines)


def short_reset(reset_at: Any) -> str:
    if not isinstance(reset_at, str):
        return "?"
    try:
        dt = datetime.fromisoformat(reset_at)
        return dt.strftime("%m/%d %H:%M")
    except ValueError:
        return reset_at


def short_checked_at(checked_at: Any) -> str:
    if not isinstance(checked_at, str):
        return "?"
    try:
        return datetime.fromisoformat(checked_at).strftime("%m/%d %H:%M")
    except ValueError:
        return checked_at


def usage_bar(used_percent: Any, width: int = 10) -> str:
    """Return a compact text progress bar for Telegram/CLI scanning."""
    try:
        value = max(0.0, min(100.0, float(used_percent)))
    except (TypeError, ValueError):
        return "?" * max(1, width)
    filled = int(value / 100 * width)
    if value > 0 and filled == 0:
        filled = 1
    if value >= 95:
        filled = width
    return "█" * filled + "░" * (width - filled)


def _fmt_percent(value: Any) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return " ?%"
    if f.is_integer():
        return f"{int(f):>2}%"
    return f"{f:>4.1f}%"


def _worst_risk(accounts: list[dict[str, Any]]) -> dict[str, str]:
    order = {"critical": 3, "warning": 2, "ok": 1, "unknown": 0}
    worst = {"level": "unknown", "icon": "⚪", "label": "unknown"}
    for row in accounts:
        if not row.get("ok"):
            return {"level": "error", "icon": "❌", "label": "조회 실패"}
        for _display, _key, window in iter_windows(row):
            r = window.get("risk") or risk(window.get("used_percent"))
            if order.get(r.get("level", "unknown"), 0) > order.get(worst.get("level", "unknown"), 0):
                worst = r
    return worst


def render_compact(payload: Dict[str, Any]) -> str:
    accounts = payload.get("accounts") or []
    worst = _worst_risk(accounts)
    lines = [f"🧭 Codex 사용량 · {short_checked_at(payload.get('checked_at'))} · {worst.get('icon')} {worst.get('label')}"]
    recommendation = payload.get("recommendation") or {}
    if recommendation:
        lines.append(f"✅ 추천 {recommendation.get('label')} — {recommendation.get('reason')}")
    if not accounts:
        lines.append("계정 없음")
        return "\n".join(lines)
    for row in accounts:
        label = row.get("label")
        plan = row.get("plan_type") or "unknown"
        if not row.get("ok"):
            lines.append(f"\n❌ {label} · {plan}")
            lines.append(f"└ ERROR {row.get('http', '')} {row.get('error', '')}".rstrip())
            continue
        lines.append(f"\n• {label} · {plan}")
        for display, _key, window in iter_windows(row):
            r = window.get("risk") or risk(window.get("used_percent"))
            used = window.get("used_percent")
            lines.append(
                f"  {display:>2} {_fmt_percent(used)} {r.get('icon')} "
                f"[{usage_bar(used)}] reset {short_reset(window.get('reset_at'))} · {window.get('remaining', '?')}"
            )
    return "\n".join(lines)


def load_seen(path: Path) -> set[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return {str(item) for item in data}
    except FileNotFoundError:
        return set()
    except Exception:
        return set()
    return set()


def save_seen(path: Path, seen: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(seen), ensure_ascii=False, indent=2), encoding="utf-8")


def apply_alert_policy(
    payload: Dict[str, Any],
    *,
    threshold: Optional[float] = None,
    primary_threshold: Optional[float] = None,
    secondary_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Return alert events from a collected usage payload.

    `threshold` is the legacy single threshold. `primary_threshold` controls the
    5h window and `secondary_threshold` controls the 7d window. Missing specific
    thresholds fall back to `threshold`, then 90%.
    """
    fallback = 90.0 if threshold is None else float(threshold)
    primary = float(primary_threshold) if primary_threshold is not None else fallback
    secondary = float(secondary_threshold) if secondary_threshold is not None else fallback
    thresholds = {"primary_window": primary, "secondary_window": secondary}

    events: list[dict[str, Any]] = []
    for row in payload.get("accounts") or []:
        if not row.get("ok"):
            events.append({"key": f"error:{row.get('label')}:{row.get('http')}:{row.get('error')}", "row": row, "error": True})
            continue
        for display, key, window in iter_windows(row):
            try:
                raw_used = window.get("used_percent")
                if raw_used is None:
                    continue
                used = float(raw_used)
            except (TypeError, ValueError):
                continue
            if used >= thresholds[key]:
                reset_key = str(window.get("reset_at"))[:16]
                events.append(
                    {
                        "key": f"quota:{row.get('label')}:{key}:{reset_key}",
                        "label": row.get("label"),
                        "window": display,
                        "window_key": key,
                        "threshold": thresholds[key],
                        "used": used if used % 1 else int(used),
                        "reset_at": window.get("reset_at"),
                        "remaining": window.get("remaining"),
                        "risk": window.get("risk") or risk(used),
                    }
                )
    return events


def alert_events(payload: Dict[str, Any], threshold: float) -> List[Dict[str, Any]]:
    return apply_alert_policy(payload, threshold=threshold)


def render_alert(
    payload: Dict[str, Any],
    threshold: Optional[float] = None,
    watermark: Optional[Path] = None,
    *,
    primary_threshold: Optional[float] = None,
    secondary_threshold: Optional[float] = None,
) -> str:
    events = apply_alert_policy(
        payload,
        threshold=threshold,
        primary_threshold=primary_threshold,
        secondary_threshold=secondary_threshold,
    )
    if watermark:
        seen = load_seen(watermark)
        fresh = [event for event in events if event["key"] not in seen]
        if fresh:
            seen.update(event["key"] for event in fresh)
            save_seen(watermark, seen)
        events = fresh
    if not events:
        return ""
    lines = [f"⚠️ Codex usage alert ({payload['checked_at']})"]
    for event in events:
        if event.get("error"):
            row = event["row"]
            lines.append(f"{row.get('label')}: ERROR {row.get('http', '')} {row.get('error', '')}".rstrip())
            continue
        r = event.get("risk") or {}
        lines.append(
            f"{r.get('icon', '⚠️')} {event.get('label')} {event.get('window')} "
            f"{event.get('used'):g}% >= {event.get('threshold'):g}% — "
            f"reset {event.get('reset_at')} ({event.get('remaining')})"
        )
    recommendation = payload.get("recommendation") or {}
    if recommendation:
        lines.append(f"추천: {recommendation.get('label')} — {recommendation.get('reason')}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show Hermes OpenAI Codex quota usage by credential")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    parser.add_argument("--compact", action="store_true", help="print Telegram-friendly compact output (default)")
    parser.add_argument("--verbose", action="store_true", help="print detailed per-account text output")
    parser.add_argument("--alert-threshold", type=float, help="legacy: print only accounts/windows at or above this percent")
    parser.add_argument("--primary-threshold", type=float, help="5h window alert threshold percent")
    parser.add_argument("--secondary-threshold", type=float, help="7d window alert threshold percent")
    parser.add_argument("--quiet-ok", action="store_true", help="with alert thresholds, print nothing when below threshold")
    parser.add_argument(
        "--watermark",
        type=Path,
        nargs="?",
        const=DEFAULT_WATERMARK,
        help="with alert thresholds, suppress duplicate alerts for the same reset window",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = collect()
    alert_mode = any(
        value is not None
        for value in (args.alert_threshold, args.primary_threshold, args.secondary_threshold)
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif alert_mode:
        output = render_alert(
            payload,
            args.alert_threshold,
            args.watermark,
            primary_threshold=args.primary_threshold,
            secondary_threshold=args.secondary_threshold,
        )
        if output:
            print(output)
        elif not args.quiet_ok:
            primary = args.primary_threshold if args.primary_threshold is not None else args.alert_threshold or 90
            secondary = args.secondary_threshold if args.secondary_threshold is not None else args.alert_threshold or 90
            print(f"Codex usage OK: 5h below {primary:g}%, 7d below {secondary:g}%")
    elif args.verbose:
        print(render_text(payload))
    else:
        print(render_compact(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
