"""Generate an Acta situation-room dashboard for latest cron outputs."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import yaml
except Exception:  # pragma: no cover - yaml is available in Hermes runtime
    yaml = None  # type: ignore[assignment]

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional in tests
    load_dotenv = None  # type: ignore[assignment]

from hermes_constants import get_hermes_home

from cron.html_artifacts import CSP, REPORT_END, REPORT_START, HtmlReportMetadata, render_html_report, render_report_body
from cron.html_publish import HtmlArtifactPublishError, publish_html_artifact

ACTA_DASHBOARD_CSP = f"{CSP}; script-src 'unsafe-inline'"
DEFAULT_HIDDEN_JOBS = ("e9b0a041ced3", "P Morning Audio Briefing")


@dataclass(frozen=True)
class CronSituationItem:
    job_id: str
    name: str
    schedule: str
    deliver: str
    enabled: bool
    latest_md: Path | None
    latest_html: Path | None
    latest_time: datetime | None
    status: str
    excerpt: str
    artifact_url: str | None = None
    telegram_url: str | None = None


def _safe_text(value: object) -> str:
    return html.escape(str(value or ""))


def _read_key(item: CronSituationItem) -> str:
    latest = item.latest_time.isoformat() if item.latest_time else "never"
    return f"{item.job_id}:{latest}"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists() or yaml is None:
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _jobs_from_file(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    jobs = data.get("jobs", data) if isinstance(data, dict) else data
    if not isinstance(jobs, list):
        return []
    return [job for job in jobs if isinstance(job, dict)]


def _schedule_display(job: Mapping[str, Any]) -> str:
    schedule = job.get("schedule")
    if isinstance(schedule, Mapping):
        return str(schedule.get("display") or schedule.get("expr") or "")
    return str(schedule or "")


def _telegram_web_url(chat_id: object, thread_id: object | None = None) -> str | None:
    """Return a Telegram deep link for private supergroup/forum targets.

    Hermes cron delivery targets for Telegram forum topics are stored as
    ``telegram:<chat_id>:<thread_id>``. Telegram's web/app deep link for
    private supergroups uses the internal ID with the leading ``-100`` removed:
    ``https://t.me/c/<internal_id>/<topic_or_message_id>``.
    """
    chat = str(chat_id or "").strip()
    thread = str(thread_id or "").strip()
    if not chat:
        return None
    if chat.startswith("-100") and thread:
        internal_id = chat[4:]
        if internal_id.isdigit() and thread.isdigit():
            return f"https://t.me/c/{internal_id}/{thread}"
    if chat.startswith("@") and len(chat) > 1:
        username = chat[1:]
        return f"https://t.me/{username}/{thread}" if thread else f"https://t.me/{username}"
    return None


def _telegram_url_from_job(job: Mapping[str, Any]) -> str | None:
    """Resolve the follow-up Telegram thread URL from cron delivery metadata."""
    deliver = str(job.get("deliver") or "").strip()
    platform = ""
    chat_id: object | None = None
    thread_id: object | None = None

    if deliver == "origin":
        origin = job.get("origin")
        if isinstance(origin, Mapping):
            platform = str(origin.get("platform") or "")
            chat_id = origin.get("chat_id")
            thread_id = origin.get("thread_id")
    elif deliver == "telegram":
        # Bare home-channel delivery may be a DM or an env-configured channel;
        # without an explicit chat/topic we cannot build a stable thread link.
        return None
    elif deliver.startswith("telegram:"):
        parts = deliver.split(":", 2)
        platform = parts[0]
        if len(parts) >= 2:
            chat_id = parts[1]
        if len(parts) == 3:
            thread_id = parts[2]

    if platform.lower() != "telegram":
        return None
    return _telegram_web_url(chat_id, thread_id)


def _latest_file(output_dir: Path, suffix: str, run_date: date | None = None) -> Path | None:
    if not output_dir.exists():
        return None
    pattern = f"{run_date.isoformat()}*{suffix}" if run_date else f"*{suffix}"
    files = list(output_dir.glob(pattern))
    return max(files, key=lambda p: p.stat().st_mtime, default=None)


def available_run_dates(hermes_home: Path | None = None, limit: int = 30) -> list[date]:
    home = hermes_home or get_hermes_home()
    dates: set[date] = set()
    output_root = home / "cron" / "output"
    if not output_root.exists():
        return []
    for md_path in output_root.glob("*/*.md"):
        if md_path.parent.name == "acta-situation-room":
            continue
        match = re.match(r"(\d{4}-\d{2}-\d{2})", md_path.name)
        if not match:
            continue
        try:
            dates.add(date.fromisoformat(match.group(1)))
        except ValueError:
            continue
    return sorted(dates, reverse=True)[:limit]


def _mtime(path: Path | None) -> datetime | None:
    if path is None:
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _run_time_from_filename(path: Path | None) -> datetime | None:
    if path is None:
        return None
    match = re.match(r"(\d{4}-\d{2}-\d{2})(?:_(\d{2})-(\d{2})-(\d{2}))?", path.name)
    if not match:
        return None
    day, hour, minute, second = match.groups()
    try:
        if hour is None:
            return datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
        return datetime.fromisoformat(f"{day}T{hour}:{minute}:{second}+00:00")
    except ValueError:
        return None


def _latest_run_time(*paths: Path | None) -> datetime | None:
    candidates = [_run_time_from_filename(path) or _mtime(path) for path in paths if path is not None]
    return max([value for value in candidates if value is not None], default=None)


def _strip_embedded_html_report(markdown: str) -> str:
    """Remove scheduler HTML artifact payloads from Markdown display text."""
    if REPORT_START not in markdown:
        return markdown.strip()
    before, _, after_start = markdown.partition(REPORT_START)
    _, found_end, after_end = after_start.partition(REPORT_END)
    if found_end:
        markdown = before + after_end
    else:
        markdown = before
    return markdown.strip()


def _extract_response(markdown: str) -> str:
    match = re.search(r"(?:^|\n)## Response\s*\n", markdown)
    if not match:
        return _strip_embedded_html_report(markdown)
    response = markdown[match.end():]
    next_heading = re.search(r"\n## [A-Z][^\n]*\n", response)
    if next_heading:
        response = response[: next_heading.start()]
    return _strip_embedded_html_report(response)


def _plain_excerpt(markdown: str, max_chars: int = 320) -> str:
    text = re.sub(r"```.*?```", " ", markdown, flags=re.S)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[*_`>#-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rsplit(" ", 1)[0] + "…"


def collect_situation_items(hermes_home: Path | None = None, run_date: date | None = None) -> list[CronSituationItem]:
    home = hermes_home or get_hermes_home()
    jobs = _jobs_from_file(home / "cron" / "jobs.json")
    output_root = home / "cron" / "output"
    items: list[CronSituationItem] = []
    for job in jobs:
        job_id = str(job.get("id") or "").strip()
        if not job_id:
            continue
        output_dir = output_root / job_id
        latest_md = _latest_file(output_dir, ".md", run_date=run_date)
        latest_html = _latest_file(output_dir, ".html", run_date=run_date)
        latest_time = _latest_run_time(latest_md, latest_html)
        response = ""
        status = "silent"
        if latest_md:
            try:
                response = _extract_response(latest_md.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                response = ""
        if response and response.strip() != "[SILENT]":
            status = "fresh" if latest_time else "ok"
        elif latest_md or latest_html:
            status = "silent"
        else:
            status = "missing"
        if run_date is not None and latest_md is None and latest_html is None:
            continue
        items.append(
            CronSituationItem(
                job_id=job_id,
                name=str(job.get("name") or job_id),
                schedule=_schedule_display(job),
                deliver=str(job.get("deliver") or ""),
                enabled=bool(job.get("enabled", True)),
                latest_md=latest_md,
                latest_html=latest_html,
                latest_time=latest_time,
                status=status,
                excerpt=_plain_excerpt(response or "No output yet."),
                telegram_url=_telegram_url_from_job(job),
            )
        )
    return sorted(items, key=lambda item: (item.enabled, item.latest_time or datetime.min.replace(tzinfo=timezone.utc)), reverse=True)


def _age_label(dt: datetime | None, now: datetime) -> str:
    if dt is None:
        return "never"
    seconds = max(0, int((now - dt).total_seconds()))
    if seconds < 90:
        return "just now"
    minutes = seconds // 60
    if minutes < 90:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 48:
        return f"{hours}h ago"
    days = hours // 24
    return f"{days}d ago"


def _status_class(item: CronSituationItem) -> str:
    if not item.enabled:
        return "paused"
    return item.status


def _telegram_link_html(item: CronSituationItem, label: str = "THREAD") -> str:
    if not item.telegram_url:
        return ""
    return (
        f' · <a class="thread-link" href="{html.escape(item.telegram_url, quote=True)}" '
        f'target="_blank" rel="noopener">{html.escape(label)}</a>'
    )


def _is_system_item(item: CronSituationItem) -> bool:
    lowered = item.name.lower()
    return "situation room refresh" in lowered or (item.deliver or "").lower() == "local"


def _config_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [str(item) for item in value if str(item).strip()]
    return []


def _item_matches_selector(item: CronSituationItem, selector: str) -> bool:
    needle = selector.strip().casefold()
    if not needle:
        return False
    return needle == item.job_id.casefold() or needle in item.name.casefold()


def _selector_rank(item: CronSituationItem, selectors: Sequence[str]) -> int | None:
    for index, selector in enumerate(selectors):
        if _item_matches_selector(item, selector):
            return index
    return None


def acta_dashboard_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    cron_cfg = config.get("cron") if isinstance(config, Mapping) else {}
    if isinstance(cron_cfg, Mapping):
        acta_cfg = cron_cfg.get("acta_dashboard") or cron_cfg.get("acta")
        if isinstance(acta_cfg, Mapping):
            return acta_cfg
    direct = config.get("acta_dashboard") if isinstance(config, Mapping) else None
    return direct if isinstance(direct, Mapping) else {}


def apply_feed_preferences(
    items: Sequence[CronSituationItem], preferences: Mapping[str, Any] | None = None
) -> list[CronSituationItem]:
    prefs = preferences or {}
    pinned = _config_list(prefs.get("pinned") or prefs.get("pinned_jobs"))
    order = _config_list(prefs.get("order") or prefs.get("feed_order"))
    demoted = _config_list(prefs.get("demoted") or prefs.get("demoted_jobs"))
    hidden = [*DEFAULT_HIDDEN_JOBS, *_config_list(prefs.get("hidden") or prefs.get("hidden_jobs"))]
    show_silent = bool(prefs.get("show_silent", True))
    show_system = bool(prefs.get("show_system", True))

    visible_items = []
    for item in items:
        if hidden and any(_item_matches_selector(item, selector) for selector in hidden):
            continue
        if not show_silent and item.status == "silent":
            continue
        if not show_system and _is_system_item(item):
            continue
        visible_items.append(item)

    def sort_key(item: CronSituationItem) -> tuple[int, int, int, int, int, float]:
        pinned_rank = _selector_rank(item, pinned)
        order_rank = _selector_rank(item, order)
        demoted_rank = _selector_rank(item, demoted)
        bucket = 0
        rank = 0
        if pinned_rank is not None:
            bucket, rank = 0, pinned_rank
        elif order_rank is not None:
            bucket, rank = 1, order_rank
        elif demoted_rank is not None:
            bucket, rank = 5, demoted_rank
        else:
            bucket, rank = 2, 0
        return (
            bucket,
            rank,
            1 if _is_system_item(item) else 0,
            1 if item.status == "silent" else 0,
            1 if item.status == "missing" else 0,
            -(item.latest_time.timestamp() if item.latest_time else 0),
        )

    return sorted(visible_items, key=sort_key)


def _render_jobs_rows(items: Sequence[CronSituationItem], now: datetime) -> list[str]:
    rows: list[str] = []
    for index, item in enumerate([item for item in items if item.enabled], start=1):
        status_class = _status_class(item)
        latest = item.latest_time.isoformat() if item.latest_time else "No run yet"
        age = _age_label(item.latest_time, now)
        schedule = item.schedule or "manual"
        rows.append(
            f"""
<div class="job-row {status_class}">
  <div class="job-rank">{index:02d}</div>
  <div class="job-main"><b>{_safe_text(item.name)}</b><span>{_safe_text(item.job_id)} · {_safe_text(item.deliver or "local")}{_telegram_link_html(item)}</span></div>
  <div class="job-schedule"><em>SCHEDULE</em>{_safe_text(schedule)}</div>
  <div class="job-last"><em>LAST RUN</em><time>{_safe_text(latest)}</time><small>{_safe_text(age)}</small></div>
</div>"""
        )
    return rows


def render_dashboard(
    items: Sequence[CronSituationItem],
    generated_at: datetime | None = None,
    selected_date: date | None = None,
    archive_dates: Sequence[date] = (),
    feed_preferences: Mapping[str, Any] | None = None,
) -> str:
    now = generated_at or datetime.now(timezone.utc)
    day_label = selected_date.isoformat() if selected_date else "latest"
    ordered_items = apply_feed_preferences(items, feed_preferences)
    total = len(ordered_items)
    active = sum(1 for item in ordered_items if item.enabled)
    visible = sum(1 for item in ordered_items if item.status == "fresh")
    silent = sum(1 for item in ordered_items if item.status == "silent")
    missing = sum(1 for item in ordered_items if item.status == "missing")

    def _priority_label(index: int, item: CronSituationItem) -> str:
        if _is_system_item(item):
            return "SYS"
        if item.status == "missing":
            return "MISS"
        if item.status == "silent":
            return "SIL"
        if index == 0:
            return "P1"
        if index == 1:
            return "P2"
        return f"P{min(index + 1, 9)}"

    lead_item = next((item for item in ordered_items if item.status == "fresh" and not _is_system_item(item)), ordered_items[0] if ordered_items else None)
    feed_items = [item for item in ordered_items if item is not lead_item]

    rows: list[str] = []
    read_order: list[str] = []
    audit_rows: list[str] = []
    for index, item in enumerate(feed_items):
        status_class = _status_class(item)
        status_label = "paused" if not item.enabled else item.status
        latest = item.latest_time.isoformat() if item.latest_time else "No run yet"
        age = _age_label(item.latest_time, now)
        priority = _priority_label(index + (1 if lead_item is not None else 0), item)
        href = html.escape(item.artifact_url, quote=True) if item.artifact_url else ""
        open_label = "OPEN" if item.artifact_url else "NO PAGE"
        ask_link = (
            f'<a class="ask-label" href="{html.escape(item.telegram_url, quote=True)}" target="_blank" rel="noopener">ASK</a>'
            if item.telegram_url
            else ""
        )
        read_key = html.escape(_read_key(item), quote=True)
        open_attr = f' data-open-url="{href}"' if href else ' aria-disabled="true"'
        rows.append(
            f"""
<section class="brief-row readable unread {status_class}" data-read-key="{read_key}"{open_attr}>
  <div class="swipe-action" aria-hidden="true">MARK READ</div>
  <div class="swipe-content">
    <div class="priority"><span class="read-dot"></span>{_safe_text(priority)}</div>
    <div class="brief-copy">
      <div class="row-kicker"><span class="read-state">UNREAD</span> · {_safe_text(status_label)} · {_safe_text(age)} · {_safe_text(item.schedule or "manual")}</div>
      <h2>{_safe_text(item.name)}</h2>
      <p>{_safe_text(item.excerpt)}</p>
      <div class="source-line">{_safe_text(item.job_id)} · {_safe_text(item.deliver or "local")} · {_safe_text(latest)}</div>
    </div>
    <span class="card-actions"><span class="open-label">{open_label}</span>{ask_link}</span>
  </div>
</section>"""
        )
        if len(read_order) < 4:
            read_order.append(
                f"""
<div class="order-row {status_class}">
  <strong>{index + 1:02d}</strong>
  <div><b>{_safe_text(item.name)}</b><p>{_safe_text(status_label)} · {_safe_text(age)}</p></div>
  <span>{_safe_text(priority)}</span>
</div>"""
            )
        if item.latest_time and len(audit_rows) < 4:
            audit_rows.append(
                f"""
<div class="audit-row">
  <time>{_safe_text(item.latest_time.strftime('%H:%M'))}</time>
  <div><b>{_safe_text(item.name)}</b><span>{_safe_text(status_label)} · {_safe_text(item.job_id)}</span></div>
</div>"""
            )

    jobs_rows = _render_jobs_rows(ordered_items, now)

    lead_title = lead_item.name if lead_item else "No briefing output yet"
    lead_excerpt = lead_item.excerpt if lead_item else "Acta is waiting for the next generated briefing packet."
    lead_href = html.escape(lead_item.artifact_url, quote=True) if lead_item and lead_item.artifact_url else ""
    lead_href_attr = f' data-open-url="{lead_href}"' if lead_href else ' aria-disabled="true"'
    lead_ask_link = (
        f'<a class="ask-label" href="{html.escape(lead_item.telegram_url, quote=True)}" target="_blank" rel="noopener">ASK TELEGRAM</a>'
        if lead_item and lead_item.telegram_url
        else ""
    )
    lead_read_key = html.escape(_read_key(lead_item), quote=True) if lead_item else ""

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, viewport-fit=cover, user-scalable=no">
<meta http-equiv="Content-Security-Policy" content="{html.escape(ACTA_DASHBOARD_CSP, quote=False)}">
<title>Acta Situation Room</title>
<meta name="description" content="Your cron command center, redesigned as a Bloomberg-black reading surface.">
<style>
:root {{ color-scheme: dark; --black:#000; --panel:#050505; --panel2:#0b0b0b; --line:#252525; --line-soft:#171717; --text:#fff; --body:#e8e8e8; --muted:#a5a5a5; --faint:#737373; --accent:#f5a400; --green:#57a773; --amber:#f5a400; --red:#d05a4e; --ui:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif; --read:'Iowan Old Style','Charter','Source Serif Pro',Georgia,serif; --mono:'SFMono-Regular','Roboto Mono','IBM Plex Mono',Consolas,monospace; }}
* {{ box-sizing:border-box; }}
html {{ width:100%; min-width:320px; overflow-x:hidden; background:#000; }}
body {{ margin:0; width:100%; min-width:320px; overflow-x:hidden; background:var(--black); color:var(--body); font:14px/1.45 var(--ui); -webkit-font-smoothing:antialiased; text-rendering:optimizeLegibility; -webkit-text-size-adjust:100%; touch-action:pan-y; }}
a {{ color:inherit; }}
.shell {{ min-height:100vh; display:grid; grid-template-columns:210px minmax(0,1fr); background:#000; }}
.rail {{ border-right:1px solid var(--line); background:#030303; display:flex; flex-direction:column; }}
.brand {{ height:54px; border-bottom:1px solid var(--line); display:flex; align-items:center; gap:10px; padding:0 14px; }}
.logo {{ width:25px; height:25px; border:1px solid #555; display:grid; place-items:center; font:700 14px var(--read); color:#fff; background:#080808; }}
.brand b {{ font-size:12px; letter-spacing:.18em; color:#fff; }}
.brand small {{ display:block; font:10px var(--mono); color:var(--muted); letter-spacing:.08em; }}
.nav-side {{ padding:14px 8px; }}
.nav-side h4 {{ margin:16px 8px 7px; color:var(--faint); font:700 10px var(--mono); letter-spacing:.12em; text-transform:uppercase; }}
.nav-side a {{ display:flex; align-items:center; gap:8px; color:#c8c8c8; padding:8px; border-radius:1px; text-decoration:none; }}
.nav-side a.active {{ background:#111; color:#fff; border-left:2px solid var(--accent); padding-left:6px; }}
.nav-side span {{ margin-left:auto; color:var(--faint); font:10px var(--mono); }}
.railfoot {{ margin-top:auto; border-top:1px solid var(--line); padding:11px 14px; color:var(--muted); font:11px var(--mono); }}
.live {{ display:inline-block; width:7px; height:7px; border-radius:50%; background:var(--green); margin-right:7px; }}
.main {{ min-width:0; }}
.top {{ height:54px; border-bottom:1px solid var(--line); display:flex; align-items:center; gap:12px; padding:0 16px; background:#020202; position:sticky; top:0; z-index:2; }}
.ticker {{ color:#fff; font:700 12px var(--mono); letter-spacing:.08em; }}
.ticker em {{ font-style:normal; color:var(--accent); }}
.search {{ height:30px; flex:1; max-width:520px; border:1px solid var(--line); background:#070707; color:var(--faint); display:flex; align-items:center; padding:0 10px; font:12px var(--mono); }}
.topstats {{ display:flex; gap:13px; margin-left:auto; }}
.topstats div {{ font:11px var(--mono); color:var(--muted); }}
.topstats b {{ color:var(--text); font-weight:600; }}
.content {{ padding:16px; display:grid; grid-template-columns:minmax(0,1fr) 340px; gap:16px; }}
.lead {{ display:block; border-bottom:1px solid var(--line); padding-bottom:15px; margin-bottom:0; text-decoration:none; color:inherit; cursor:pointer; position:relative; }}
.lead[aria-disabled='true'] {{ cursor:default; }}
.lead:hover h1 {{ color:var(--accent); }}
.label {{ font:700 11px var(--mono); letter-spacing:.12em; color:var(--accent); text-transform:uppercase; }}
h1 {{ font:700 clamp(34px,4.8vw,52px)/1.02 var(--read); letter-spacing:-.04em; margin:8px 0 10px; color:#fff; max-width:980px; }}
.lead p {{ font:18px/1.45 var(--read); color:var(--body); max-width:940px; margin:0; }}
.meta {{ display:flex; flex-wrap:wrap; gap:14px; margin-top:12px; color:var(--muted); font:11px var(--mono); text-transform:uppercase; }}
.meta b {{ color:#fff; font-weight:600; }}
.read-dot {{ width:8px; height:8px; border-radius:50%; background:var(--accent); display:inline-block; margin-right:7px; box-shadow:0 0 0 2px rgba(245,164,0,.14); }}
.readable.read {{ opacity:.68; }}
.readable.read .read-dot {{ background:transparent; box-shadow:inset 0 0 0 1px var(--faint); }}
.readable.read h1, .readable.read h2 {{ color:#c8c8c8; }}
.feed {{ border-top:1px solid var(--line); }}
.brief-row {{ display:block; border-bottom:1px solid var(--line-soft); text-decoration:none; color:inherit; cursor:pointer; position:relative; overflow:hidden; background:#000; }}
.swipe-content {{ display:grid; grid-template-columns:82px minmax(0,1fr) 76px; gap:14px; padding:16px 0; position:relative; z-index:1; background:#000; transition:transform .22s cubic-bezier(.2,.8,.2,1), opacity .18s ease, background .18s ease; will-change:transform; }}
.swipe-action {{ position:absolute; inset:0 auto 0 0; width:118px; display:flex; align-items:center; padding-left:16px; color:#000; background:var(--accent); font:800 10px var(--mono); letter-spacing:.08em; z-index:0; opacity:0; transition:opacity .12s ease; }}
.brief-row.swiping .swipe-content {{ transition:none; }}
.brief-row.swipe-peek .swipe-content {{ transform:translateX(92px); }}
.brief-row.swipe-peek .swipe-action, .brief-row.swiping .swipe-action {{ opacity:1; }}
.brief-row:hover .swipe-content {{ background:#050505; outline:1px solid var(--line); outline-offset:0; padding-left:10px; padding-right:10px; margin-left:-10px; margin-right:-10px; }}
.brief-row[aria-disabled='true'] {{ cursor:default; }}
.priority {{ font:700 11px var(--mono); color:var(--accent); }}
.silent .priority {{ color:var(--red); }}
.missing .priority {{ color:var(--red); }}
.paused {{ opacity:.6; }}
.row-kicker {{ color:var(--muted); font:11px var(--mono); text-transform:uppercase; letter-spacing:.08em; margin-bottom:5px; }}
h2 {{ font:700 23px/1.1 var(--read); margin:0 0 6px; color:#fff; letter-spacing:-.015em; }}
.brief-copy p {{ font:16px/1.45 var(--read); color:var(--body); margin:0; max-width:860px; }}
.source-line {{ font:11px var(--mono); color:var(--muted); margin-top:8px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.open-label {{ justify-self:end; align-self:start; border:1px solid var(--line); color:#fff; padding:6px 8px; font:11px var(--mono); background:#080808; }}
.card-actions {{ justify-self:end; align-self:start; display:flex; gap:7px; align-items:center; }}
.ask-label {{ border:1px solid var(--accent); color:#000; background:var(--accent); text-decoration:none; padding:6px 8px; font:800 11px var(--mono); }}
.brief-row:hover .open-label {{ border-color:var(--accent); color:var(--accent); }}
.jobs-panel {{ margin-top:22px; border-top:1px solid var(--line); scroll-margin-top:112px; }}
.jobs-head {{ display:flex; align-items:flex-end; gap:12px; padding:16px 0 8px; border-bottom:1px solid var(--line-soft); }}
.jobs-head h2 {{ margin:0; font:800 13px var(--mono); letter-spacing:.12em; text-transform:uppercase; color:#fff; }}
.jobs-head span {{ margin-left:auto; color:var(--muted); font:11px var(--mono); text-transform:uppercase; }}
.job-row {{ display:grid; grid-template-columns:42px minmax(0,1.2fr) minmax(120px,.7fr) minmax(180px,.9fr); gap:12px; align-items:center; padding:13px 0; border-bottom:1px solid var(--line-soft); }}
.job-rank {{ color:var(--accent); font:800 11px var(--mono); }}
.job-main b {{ display:block; color:#fff; font:700 15px/1.2 var(--ui); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.job-main span, .job-schedule, .job-last {{ color:var(--muted); font:11px var(--mono); min-width:0; }}
.thread-link {{ color:var(--accent); text-decoration:none; border-bottom:1px solid rgba(245,164,0,.55); font-weight:800; }}
.job-schedule, .job-last {{ display:grid; gap:3px; }}
.job-schedule em, .job-last em {{ color:var(--faint); font-style:normal; font-size:9px; letter-spacing:.1em; }}
.job-last time {{ color:#fff; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.job-last small {{ color:var(--accent); font:10px var(--mono); text-transform:uppercase; }}
.side {{ display:grid; gap:16px; align-content:start; }}
.card {{ background:var(--panel); border:1px solid var(--line); }}
.card-head {{ height:34px; border-bottom:1px solid var(--line); display:flex; align-items:center; padding:0 10px; font:700 11px var(--mono); letter-spacing:.1em; text-transform:uppercase; color:#fff; }}
.card-head span {{ margin-left:auto; color:var(--muted); font-weight:400; }}
.readorder, .audit, .assist {{ padding:10px 12px; }}
.order-row {{ display:grid; grid-template-columns:32px 1fr auto; gap:9px; padding:10px 0; border-bottom:1px solid var(--line-soft); }}
.order-row:last-child, .audit-row:last-child {{ border-bottom:0; }}
.order-row strong {{ color:#fff; font:700 12px var(--mono); }}
.order-row b {{ color:#fff; font-size:13px; }}
.order-row p {{ margin:2px 0 0; color:var(--muted); font-size:12px; }}
.order-row span {{ color:var(--accent); border:1px solid var(--line); padding:2px 5px; height:19px; font:10px var(--mono); }}
.audit-row {{ display:grid; grid-template-columns:50px 1fr; gap:9px; padding:9px 0; border-bottom:1px solid var(--line-soft); }}
.audit-row time {{ color:var(--faint); font:11px var(--mono); }}
.audit-row b {{ display:block; color:#fff; font-size:12px; }}
.audit-row span {{ display:block; color:var(--muted); font-size:12px; }}
.prompt {{ font:14px/1.45 var(--read); color:var(--body); border-left:2px solid var(--accent); padding-left:10px; }}
.chiprow {{ display:flex; flex-wrap:wrap; gap:7px; margin-top:12px; }}
.chip {{ border:1px solid var(--line); color:var(--muted); font:10px var(--mono); padding:4px 6px; background:#080808; }}
.date-nav {{ display:flex; gap:8px; flex-wrap:wrap; padding:0 16px 16px; border-bottom:1px solid var(--line); }}
.nav-link {{ color:var(--muted); text-decoration:none; border:1px solid var(--line); padding:6px 9px; font:11px var(--mono); background:#050505; }}
.nav-link.primary {{ color:#fff; border-color:var(--accent); }}
footer {{ color:var(--faint); margin:24px 16px 36px; font:12px var(--mono); text-align:center; }}
.mobilebar {{ display:none; }}
.pull-refresh {{ display:none; position:fixed; left:50%; top:calc(8px + env(safe-area-inset-top, 0px)); transform:translate(-50%,-130%); min-width:150px; padding:9px 12px; border:1px solid var(--line); background:rgba(2,2,2,.96); color:var(--accent); font:800 10px var(--mono); letter-spacing:.12em; text-align:center; z-index:5; opacity:0; transition:transform .18s ease, opacity .18s ease; box-shadow:0 12px 32px rgba(0,0,0,.55); }}
.pull-refresh.ready {{ color:#000; background:var(--accent); border-color:var(--accent); }}
.pull-refresh.visible {{ opacity:1; transform:translate(-50%,0); }}
@media (max-width:980px) {{ .pull-refresh {{ display:block; }} .shell {{ display:block; min-width:0; width:100%; }} .rail {{ display:none; }} .main {{ width:100%; min-width:0; }} .top {{ height:50px; padding:0 max(14px, env(safe-area-inset-left, 0px)) 0 max(14px, env(safe-area-inset-left, 0px)); }} .date-nav {{ position:static; background:#000; padding:8px 14px; gap:8px; }} .nav-link {{ min-height:38px; display:inline-flex; align-items:center; padding:0 12px; }} .content {{ display:block; padding:12px 14px calc(132px + env(safe-area-inset-bottom, 0px)); }} .side {{ display:none; }} .topstats {{ display:none; }} .lead {{ padding-bottom:16px; margin-bottom:0; touch-action:pan-y; }} .lead p {{ display:-webkit-box; -webkit-line-clamp:5; -webkit-box-orient:vertical; overflow:hidden; }} .meta {{ gap:9px; }} .feed {{ border-top:0; }} .brief-row {{ min-height:96px; touch-action:pan-y; }} .swipe-content {{ grid-template-columns:42px minmax(0,1fr); gap:8px; min-height:96px; padding:15px 0; touch-action:pan-y; }} .brief-row:hover .swipe-content {{ background:#000; outline:0; padding-left:0; padding-right:0; margin-left:0; margin-right:0; }} .priority {{ font-size:10px; }} .row-kicker {{ font-size:10px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }} .brief-copy p {{ display:-webkit-box; -webkit-line-clamp:4; -webkit-box-orient:vertical; overflow:hidden; }} .source-line {{ display:none; }} .open-label {{ display:none; }} .card-actions {{ grid-column:2; justify-self:start; }} .ask-label {{ padding:5px 7px; font-size:10px; }} .jobs-panel {{ margin-top:18px; scroll-margin-top:100px; }} .jobs-head {{ padding-top:14px; }} .job-row {{ grid-template-columns:34px minmax(0,1fr); gap:8px 10px; padding:13px 0; }} .job-schedule, .job-last {{ grid-column:2; }} .job-main b {{ font-size:14px; }} .search {{ max-width:none; }} .mobilebar {{ display:grid; position:fixed; left:max(10px, env(safe-area-inset-left, 0px)); right:max(10px, env(safe-area-inset-right, 0px)); bottom:calc(14px + env(safe-area-inset-bottom, 0px)); min-height:62px; background:rgba(2,2,2,.96); backdrop-filter:blur(14px); border:1px solid var(--line); grid-template-columns:repeat(4,1fr); z-index:3; box-shadow:0 -10px 24px rgba(0,0,0,.72); opacity:0; transform:translateY(calc(100% + 24px)); pointer-events:none; transition:opacity .18s ease, transform .22s cubic-bezier(.2,.8,.2,1); }} .mobilebar.visible {{ opacity:1; transform:translateY(0); pointer-events:auto; }} .mobilebar a {{ display:grid; place-items:center; min-height:62px; color:#ddd; text-decoration:none; font:11px var(--mono); touch-action:manipulation; -webkit-tap-highlight-color:rgba(245,164,0,.18); }} .mobilebar a:first-child {{ color:var(--accent); }} }}
@media (max-width:620px) {{ .top {{ gap:8px; }} .ticker {{ font-size:11px; }} .search {{ display:none; }} h1 {{ font-size:clamp(31px,9vw,42px); max-width:100%; }} .lead p {{ font-size:16px; line-height:1.42; }} h2 {{ font-size:20px; }} .brief-copy p {{ font-size:15px; line-height:1.4; }} }}
</style>
</head>
<body>
<div class="pull-refresh" aria-live="polite">PULL TO REFRESH</div>
<div class="shell">
  <aside class="rail">
    <div class="brand"><div class="logo">A</div><div><b>ACTA</b><small>SITUATION ROOM</small></div></div>
    <nav class="nav-side">
      <h4>Today</h4>
      <a class="active" href="/">Briefing Packet <span>{total}</span></a>
      <a href="/">Operator News <span>{visible}</span></a>
      <a href="/archive">Archive <span>{len(archive_dates)}</span></a>
      <a href="/jobs">Jobs <span>{len(jobs_rows)}</span></a>
      <h4>System</h4>
      <a href="/">Source Runs</a>
      <a href="/">Delivery Routes</a>
      <a href="/">Audit Trail</a>
    </nav>
    <div class="railfoot"><span class="live"></span>LIVE {html.escape(now.strftime('%H:%M UTC'))}<br>DAY {html.escape(day_label)}</div>
  </aside>
  <main class="main">
    <header class="top">
      <div class="ticker"><em>ACTA</em> / DAILY PACKET</div>
      <div class="search">Search briefings, sources, jobs, archive…</div>
      <div class="topstats"><div>VISIBLE <b>{visible}</b></div><div>SILENT <b>{silent}</b></div><div>MISSING <b>{missing}</b></div></div>
    </header>
    <nav class="date-nav"><a class="nav-link primary" href="/">Today</a><a class="nav-link" href="/jobs">Jobs</a><a class="nav-link" href="/archive">Archive</a></nav>
    <section class="content">
      <div>
        <article class="lead readable unread" data-read-key="{lead_read_key}"{lead_href_attr}>
          <div class="label"><span class="read-dot"></span><span class="read-state">UNREAD</span> · Read First</div>
          <h1>{_safe_text(lead_title)}</h1>
          <p>{_safe_text(lead_excerpt)}</p>
          <div class="meta"><span>{html.escape(day_label)}</span><span><b>Active:</b> {active}</span><span><b>Status:</b> {visible} fresh / {silent} silent / {missing} missing</span><span>Tap to open</span>{lead_ask_link}</div>
        </article>
        <section class="feed">
          {''.join(rows)}
        </section>
      </div>
      <aside class="side">
        <section class="card"><div class="card-head">Read Order <span>PRIORITIZED</span></div><div class="readorder">{''.join(read_order) or '<p class="prompt">No outputs yet.</p>'}</div></section>
        <section class="card"><div class="card-head">Audit Trail <span>PROVENANCE</span></div><div class="audit">{''.join(audit_rows) or '<p class="prompt">No runs yet.</p>'}</div></section>
        <section class="card"><div class="card-head">Operator Assist <span>NEXT PASS</span></div><div class="assist"><div class="prompt">Open the top briefing packet first. Use silent or missing rows as source gaps, not as promoted content. Keep raw prompts and local paths out of the human surface.</div><div class="chiprow"><span class="chip">TEXT FIRST</span><span class="chip">NO RAW PROMPTS</span><span class="chip">CLICK ANY ROW</span></div></div></section>
      </aside>
    </section>
    <footer>Generated {html.escape(now.isoformat())}. Signed Acta links expire automatically.</footer>
  </main>
</div>
<nav class="mobilebar"><a href="/">TODAY</a><a href="/jobs">JOBS</a><a href="/archive">ARCHIVE</a><a href="/">READ</a></nav>
<script>
(function(){{
  var KEY='acta:read:v1';
  var COOKIE='acta_read_v1';
  function readFromCookie(){{
    var parts=(document.cookie||'').split('; ');
    for(var i=0;i<parts.length;i++){{
      if(parts[i].indexOf(COOKIE+'=')===0){{
        try{{ return JSON.parse(decodeURIComponent(parts[i].slice(COOKIE.length+1)))||{{}}; }}catch(e){{ return {{}}; }}
      }}
    }}
    return {{}};
  }}
  function writeToCookie(value){{
    try{{ document.cookie=COOKIE+'='+encodeURIComponent(JSON.stringify(value))+'; Max-Age=31536000; Path=/; SameSite=Lax; Secure'; }}catch(e){{}}
  }}
  var state=readFromCookie();
  try{{ state=JSON.parse(localStorage.getItem(KEY)||JSON.stringify(state)||'{{}}')||state||{{}}; }}catch(e){{ state=state||{{}}; }}
  function save(){{ try{{ localStorage.setItem(KEY, JSON.stringify(state)); }}catch(e){{}} writeToCookie(state); }}
  var pull=document.querySelector('.pull-refresh');
  var mobileBar=document.querySelector('.mobilebar');
  var topNav=document.querySelector('.date-nav');
  function setMobileBarVisible(visible){{ if(mobileBar) mobileBar.classList.toggle('visible', !!visible); }}
  if(mobileBar && topNav){{
    if('IntersectionObserver' in window){{
      new IntersectionObserver(function(entries){{
        var entry=entries[0];
        setMobileBarVisible(!entry.isIntersecting && window.matchMedia('(max-width: 980px)').matches);
      }}, {{root:null, threshold:0.01}}).observe(topNav);
    }} else {{
      var navBottom=topNav.offsetTop+topNav.offsetHeight;
      function updateMobileBar(){{ setMobileBarVisible(window.matchMedia('(max-width: 980px)').matches && window.scrollY>navBottom); }}
      window.addEventListener('scroll', updateMobileBar, {{passive:true}});
      window.addEventListener('resize', updateMobileBar, {{passive:true}});
      updateMobileBar();
    }}
  }}
  var psx=0, psy=0, pdy=0, pulling=false, ptrReady=false;
  function ptrStart(ev){{
    if(window.scrollY>2) return;
    var t=(ev.touches&&ev.touches[0]) || ev;
    psx=t.clientX||0; psy=t.clientY||0; pdy=0; pulling=true; ptrReady=false;
  }}
  function ptrMove(ev){{
    if(!pulling || !pull) return;
    var t=(ev.touches&&ev.touches[0]) || ev;
    var dx=Math.abs((t.clientX||0)-psx); pdy=(t.clientY||0)-psy;
    if(pdy>12 && pdy>dx*1.25 && window.scrollY<=2){{
      var y=Math.min(72, Math.max(0, pdy*.55));
      pull.classList.add('visible');
      ptrReady=y>=54;
      pull.classList.toggle('ready', ptrReady);
      pull.textContent=ptrReady?'RELEASE TO REFRESH':'PULL TO REFRESH';
      pull.style.transform='translate(-50%,'+y+'px)';
      if(ev.cancelable) ev.preventDefault();
    }}
  }}
  function ptrEnd(){{
    if(!pulling) return;
    pulling=false;
    if(ptrReady){{
      if(pull){{ pull.textContent='REFRESHING'; pull.classList.add('visible','ready'); pull.style.transform='translate(-50%,54px)'; }}
      location.reload();
      return;
    }}
    if(pull){{ pull.classList.remove('visible','ready'); pull.style.transform=''; pull.textContent='PULL TO REFRESH'; }}
  }}
  document.addEventListener('touchstart', ptrStart, {{passive:true}});
  document.addEventListener('touchmove', ptrMove, {{passive:false}});
  document.addEventListener('touchend', ptrEnd, {{passive:true}});
  document.addEventListener('touchcancel', ptrEnd, {{passive:true}});
  function apply(el){{
    var k=el.dataset.readKey || '';
    var isRead=!!state[k];
    el.classList.toggle('read', isRead);
    el.classList.toggle('unread', !isRead);
    var label=el.querySelector('.read-state');
    if(label) label.textContent=isRead?'READ':'UNREAD';
    var action=el.querySelector('.swipe-action');
    if(action) action.textContent=isRead?'MARK UNREAD':'MARK READ';
  }}
  function toggle(el){{
    var k=el.dataset.readKey || '';
    if(!k) return;
    state[k]=!state[k];
    save();
    apply(el);
    el.classList.add('swipe-peek');
    setTimeout(function(){{ el.classList.remove('swipe-peek'); }}, 260);
  }}
  document.querySelectorAll('.readable').forEach(function(el){{
    apply(el);
    var content=el.querySelector('.swipe-content') || el;
    var sx=0, sy=0, dx=0, swiping=false, didSwipe=false, active=false;
    function point(ev){{
      var t=(ev.touches&&ev.touches[0]) || ev;
      return {{x:t.clientX||0, y:t.clientY||0}};
    }}
    function start(ev){{
      var p=point(ev); sx=p.x; sy=p.y; dx=0; swiping=false; didSwipe=false; active=true;
    }}
    function move(ev){{
      if(!active) return;
      var p=point(ev); dx=p.x-sx;
      var dy=Math.abs(p.y-sy);
      if(dx>10 && Math.abs(dx)>dy*1.15){{
        swiping=true; didSwipe=true; el.classList.add('swiping','swipe-peek');
        content.style.transform='translateX('+Math.min(dx,96)+'px)';
        if(ev.cancelable) ev.preventDefault();
      }}
    }}
    function end(ev){{
      if(!active) return;
      active=false;
      if(swiping){{
        if(ev.cancelable) ev.preventDefault();
        if(dx>58) toggle(el);
      }}
      el.classList.remove('swiping');
      content.style.transform='';
      if(didSwipe){{
        setTimeout(function(){{ el.classList.remove('swipe-peek'); }}, 220);
      }} else {{
        el.classList.remove('swipe-peek');
      }}
    }}
    el.addEventListener('click', function(ev){{
      if(didSwipe){{ ev.preventDefault(); ev.stopPropagation(); didSwipe=false; return; }}
      if(ev.target && ev.target.closest && ev.target.closest('a')) return;
      var k=el.dataset.readKey || '';
      if(k){{ state[k]=true; save(); apply(el); }}
      var openUrl=el.dataset.openUrl || '';
      if(openUrl) window.location.href=openUrl;
    }}, true);
    el.addEventListener('touchstart', start, {{passive:true}});
    el.addEventListener('touchmove', move, {{passive:false}});
    el.addEventListener('touchend', end, {{passive:false}});
    el.addEventListener('pointerdown', function(ev){{ if(ev.pointerType==='touch'||ev.pointerType==='pen') start(ev); }});
    el.addEventListener('pointermove', function(ev){{ if(ev.pointerType==='touch'||ev.pointerType==='pen') move(ev); }});
    el.addEventListener('pointerup', function(ev){{ if(ev.pointerType==='touch'||ev.pointerType==='pen') end(ev); }});
    el.addEventListener('pointercancel', end);
  }});
}})();
</script>
</body>
</html>
"""


def render_jobs_page(
    items: Sequence[CronSituationItem],
    generated_at: datetime | None = None,
    feed_preferences: Mapping[str, Any] | None = None,
) -> str:
    now = generated_at or datetime.now(timezone.utc)
    ordered_items = apply_feed_preferences(items, feed_preferences)
    jobs_rows = _render_jobs_rows(ordered_items, now)
    fresh = sum(1 for item in ordered_items if item.enabled and item.status == "fresh")
    silent = sum(1 for item in ordered_items if item.enabled and item.status == "silent")
    missing = sum(1 for item in ordered_items if item.enabled and item.status == "missing")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, viewport-fit=cover, user-scalable=no">
<meta http-equiv="Content-Security-Policy" content="{html.escape(CSP, quote=False)}">
<title>Acta Jobs</title>
<style>
:root {{ color-scheme: dark; --bg:#000; --panel:#050505; --line:#252525; --line-soft:#171717; --text:#fff; --body:#e8e8e8; --muted:#a5a5a5; --faint:#737373; --accent:#f5a400; --green:#57a773; --red:#d05a4e; --ui:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif; --read:'Iowan Old Style','Charter','Source Serif Pro',Georgia,serif; --mono:'SFMono-Regular','Roboto Mono','IBM Plex Mono',Consolas,monospace; }}
* {{ box-sizing:border-box; }}
html {{ width:100%; min-width:320px; overflow-x:hidden; background:#000; }}
body {{ margin:0; width:100%; min-width:320px; overflow-x:hidden; background:var(--bg); color:var(--body); font:14px/1.45 var(--ui); -webkit-font-smoothing:antialiased; text-rendering:optimizeLegibility; -webkit-text-size-adjust:100%; }}
a {{ color:inherit; }}
.top {{ height:54px; border-bottom:1px solid var(--line); display:flex; align-items:center; gap:12px; padding:0 16px; background:#020202; position:sticky; top:0; z-index:2; }}
.ticker {{ color:#fff; font:700 12px var(--mono); letter-spacing:.08em; text-decoration:none; }}
.ticker em {{ font-style:normal; color:var(--accent); }}
.nav {{ margin-left:auto; display:flex; gap:8px; }}
.nav a {{ color:var(--body); text-decoration:none; border:1px solid var(--line); background:#050505; padding:7px 10px; font:11px var(--mono); text-transform:uppercase; letter-spacing:.08em; }}
.nav a.active, .nav a:hover {{ color:#000; background:var(--accent); border-color:var(--accent); }}
main {{ max-width:1180px; margin:0 auto; padding:26px 16px 84px; }}
.kicker {{ margin:0; color:var(--accent); font:800 11px var(--mono); text-transform:uppercase; letter-spacing:.14em; }}
h1 {{ margin:8px 0 10px; color:var(--text); font:700 clamp(42px,7vw,78px)/.95 var(--read); letter-spacing:-.06em; }}
.lede {{ max-width:760px; margin:0 0 22px; color:var(--body); font:18px/1.5 var(--read); }}
.stats {{ display:flex; flex-wrap:wrap; gap:8px; margin:18px 0 18px; }}
.stat {{ border:1px solid var(--line); background:#050505; padding:9px 11px; color:var(--muted); font:11px var(--mono); text-transform:uppercase; }}
.stat b {{ color:#fff; font-size:14px; margin-left:6px; }}
.jobs-panel {{ border-top:1px solid var(--line); }}
.jobs-head {{ display:flex; align-items:flex-end; gap:12px; padding:16px 0 8px; border-bottom:1px solid var(--line-soft); }}
.jobs-head h2 {{ margin:0; font:800 13px var(--mono); letter-spacing:.12em; text-transform:uppercase; color:#fff; }}
.jobs-head span {{ margin-left:auto; color:var(--muted); font:11px var(--mono); text-transform:uppercase; }}
.job-row {{ display:grid; grid-template-columns:42px minmax(0,1.2fr) minmax(120px,.7fr) minmax(180px,.9fr); gap:12px; align-items:center; padding:13px 0; border-bottom:1px solid var(--line-soft); }}
.job-rank {{ color:var(--accent); font:800 11px var(--mono); }}
.job-main b {{ display:block; color:#fff; font:700 15px/1.2 var(--ui); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.job-main span, .job-schedule, .job-last {{ color:var(--muted); font:11px var(--mono); min-width:0; }}
.thread-link {{ color:var(--accent); text-decoration:none; border-bottom:1px solid rgba(245,164,0,.55); font-weight:800; }}
.job-schedule, .job-last {{ display:grid; gap:3px; }}
.job-schedule em, .job-last em {{ color:var(--faint); font-style:normal; font-size:9px; letter-spacing:.1em; }}
.job-last time {{ color:#fff; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.job-last small {{ color:var(--accent); font:10px var(--mono); text-transform:uppercase; }}
.silent .job-rank, .missing .job-rank {{ color:var(--red); }}
.prompt {{ font:14px/1.45 var(--read); color:var(--body); border-left:2px solid var(--accent); padding-left:10px; }}
footer {{ color:var(--faint); margin-top:24px; font:12px var(--mono); text-align:center; }}
@media (max-width:760px) {{ .top {{ height:50px; padding:0 14px; }} .nav a {{ min-height:36px; display:inline-flex; align-items:center; }} main {{ padding:18px 14px 88px; }} h1 {{ font-size:clamp(36px,12vw,52px); }} .lede {{ font-size:16px; }} .job-row {{ grid-template-columns:34px minmax(0,1fr); gap:8px 10px; padding:14px 0; }} .job-schedule, .job-last {{ grid-column:2; }} .job-main b {{ font-size:14px; }} }}
</style>
</head>
<body>
<header class="top"><a class="ticker" href="/"><em>ACTA</em> / JOBS</a><nav class="nav"><a href="/">Today</a><a class="active" href="/jobs">Jobs</a><a href="/archive">Archive</a></nav></header>
<main>
  <p class="kicker">Acta Situation Room · Operations</p>
  <h1>Active Cron Jobs</h1>
  <p class="lede">Operational visibility for active relevant Hermes jobs: schedule, delivery route, latest run timestamp, and freshness status.</p>
  <section class="stats"><div class="stat">Relevant <b>{len(jobs_rows)}</b></div><div class="stat">Fresh <b>{fresh}</b></div><div class="stat">Silent <b>{silent}</b></div><div class="stat">Missing <b>{missing}</b></div></section>
  <section class="jobs-panel">
    <div class="jobs-head"><h2>Active Cron Jobs</h2><span>{len(jobs_rows)} relevant</span></div>
    {''.join(jobs_rows) or '<p class="prompt">No active relevant jobs.</p>'}
  </section>
  <footer>Generated {html.escape(now.isoformat())}.</footer>
</main>
</body>
</html>
"""


def render_archive_index(archive_dates: Sequence[date], generated_at: datetime | None = None) -> str:
    now = generated_at or datetime.now(timezone.utc)
    cards = "".join(
        f'<a class="archive-card" href="/archive/{d.isoformat()}"><span>Acta Day</span><strong>{d.isoformat()}</strong></a>'
        for d in archive_dates
    ) or '<p class="lede">No archived cron outputs yet.</p>'
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, viewport-fit=cover, user-scalable=no">
<meta http-equiv="Content-Security-Policy" content="{html.escape(CSP, quote=False)}">
<title>Acta Archive</title>
<style>
:root {{ color-scheme: dark; --bg:#000; --panel:#050505; --line:#252525; --text:#fff; --body:#e8e8e8; --muted:#a5a5a5; --accent:#f5a400; --ui:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif; --read:'Iowan Old Style','Charter','Source Serif Pro',Georgia,serif; --mono:'SFMono-Regular','Roboto Mono','IBM Plex Mono',Consolas,monospace; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:var(--ui); background:#000; color:var(--body); -webkit-font-smoothing:antialiased; text-rendering:optimizeLegibility; }}
.top {{ height:54px; border-bottom:1px solid var(--line); display:flex; align-items:center; gap:12px; padding:0 16px; background:#020202; position:sticky; top:0; z-index:2; }}
.ticker {{ color:#fff; font:700 12px var(--mono); letter-spacing:.08em; text-decoration:none; }}
.ticker em {{ font-style:normal; color:var(--accent); }}
.nav {{ margin-left:auto; display:flex; gap:8px; }}
.nav a {{ color:var(--body); text-decoration:none; border:1px solid var(--line); background:#050505; padding:7px 10px; font:11px var(--mono); text-transform:uppercase; letter-spacing:.08em; }}
.nav a.active, .nav a:hover {{ color:#000; background:var(--accent); border-color:var(--accent); }}
main {{ max-width:1040px; margin:0 auto; padding:28px 20px 72px; }}
.kicker {{ color:var(--accent); font:700 11px/1 var(--mono); text-transform:uppercase; letter-spacing:.16em; }}
h1 {{ margin:8px 0 12px; color:var(--text); font:700 clamp(42px,7vw,76px)/.95 var(--read); letter-spacing:-.06em; }}
.lede {{ color:var(--body); font:18px/1.55 var(--read); }}
.quick-nav {{ display:flex; gap:8px; margin:24px 0 28px; }}
.quick-nav a {{ color:var(--body); text-decoration:none; border:1px solid var(--line); background:#050505; padding:8px 12px; font:12px var(--mono); }}
.quick-nav a.active {{ color:#000; background:var(--accent); border-color:var(--accent); }}
.grid {{ display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:14px; }}
.archive-card {{ text-decoration:none; border:1px solid var(--line); padding:20px; background:var(--panel); }}
.archive-card:hover {{ border-color:var(--accent); }}
.archive-card span {{ display:block; color:var(--muted); font:700 11px/1 var(--mono); text-transform:uppercase; letter-spacing:.12em; margin-bottom:10px; }}
.archive-card strong {{ color:var(--text); font:700 25px var(--read); letter-spacing:-.03em; }}
footer {{ color:var(--muted); margin-top:24px; font:12px var(--mono); text-align:center; }}
@media (max-width:760px) {{ .top {{ height:50px; padding:0 14px; }} .nav a {{ min-height:36px; display:inline-flex; align-items:center; }} main {{ padding:22px 14px 88px; }} .grid {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<header class="top"><a class="ticker" href="/"><em>ACTA</em> / ARCHIVE</a><nav class="nav"><a href="/">Today</a><a href="/jobs">Jobs</a><a class="active" href="/archive">Archive</a></nav></header>
<main>
<p class="kicker">Acta · Archive</p>
<h1>Previous days.</h1>
<p class="lede">Browse prior Situation Room snapshots by day.</p>
<nav class="quick-nav"><a href="/">Today</a><a href="/jobs">Jobs</a><a class="active" href="/archive">Archive</a></nav>
<section class="grid">{cards}</section>
<footer>Generated {html.escape(now.isoformat())}.</footer>
</main>
</body>
</html>
"""


def render_acta_detail_report(
    body: str,
    metadata: HtmlReportMetadata | Mapping[str, str],
    telegram_url: str | None = None,
) -> str:
    """Render a standalone Acta detail page using the dashboard visual system."""
    if isinstance(metadata, Mapping):
        meta = HtmlReportMetadata(**{k: str(v) for k, v in metadata.items() if k in HtmlReportMetadata.__annotations__})
    else:
        meta = metadata
    title = str(meta.job_name or f"Cron report {meta.job_id}")
    job_id = str(meta.job_id or "")
    run_time = str(meta.run_time or datetime.now(timezone.utc).isoformat())
    source_filename = str(meta.source_filename or "")
    rendered = render_report_body(body)
    footer_bits = [
        f"<span><b>JOB</b> {html.escape(job_id)}</span>",
        f"<span><b>RUN</b> {html.escape(run_time)}</span>",
    ]
    if source_filename:
        footer_bits.append(f"<span><b>SOURCE</b> {html.escape(Path(source_filename).name)}</span>")
    followup_link = ""
    if telegram_url:
        followup_link = (
            f'<a class="followup" href="{html.escape(telegram_url, quote=True)}" '
            'target="_blank" rel="noopener">Ask follow-up in Telegram</a>'
        )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, viewport-fit=cover, user-scalable=no">
<meta http-equiv="Content-Security-Policy" content="{html.escape(CSP, quote=False)}">
<title>{html.escape(title)} · Acta Situation Room</title>
<style>
:root {{ color-scheme:dark; --black:#000; --panel:#050505; --panel2:#0b0b0b; --line:#252525; --line-soft:#171717; --text:#fff; --body:#e8e8e8; --muted:#a5a5a5; --faint:#737373; --accent:#f5a400; --green:#57a773; --red:#d05a4e; --ui:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif; --read:'Iowan Old Style','Charter','Source Serif Pro',Georgia,serif; --mono:'SFMono-Regular','Roboto Mono','IBM Plex Mono',Consolas,monospace; }}
* {{ box-sizing:border-box; }}
html {{ width:100%; min-width:320px; overflow-x:hidden; background:#000; }}
body {{ margin:0; width:100%; min-width:320px; overflow-x:hidden; background:var(--black); color:var(--body); font:16px/1.58 var(--read); -webkit-font-smoothing:antialiased; text-rendering:optimizeLegibility; -webkit-text-size-adjust:100%; }}
a {{ color:inherit; }}
.top {{ height:54px; border-bottom:1px solid var(--line); display:flex; align-items:center; gap:12px; padding:0 16px; background:#020202; position:sticky; top:0; z-index:2; font-family:var(--ui); }}
.ticker {{ color:#fff; font:700 12px var(--mono); letter-spacing:.08em; text-decoration:none; }}
.ticker em {{ font-style:normal; color:var(--accent); }}
.actions {{ margin-left:auto; display:flex; gap:8px; align-items:center; }}
.followup {{ color:#000; text-decoration:none; border:1px solid var(--accent); background:var(--accent); padding:7px 10px; font:800 11px var(--mono); text-transform:uppercase; letter-spacing:.08em; }}
.back {{ color:var(--muted); text-decoration:none; border:1px solid var(--line); background:#050505; padding:7px 10px; font:11px var(--mono); text-transform:uppercase; letter-spacing:.08em; }}
.back:hover {{ color:#000; background:var(--accent); border-color:var(--accent); }}
main {{ max-width:980px; margin:0 auto; padding:18px 16px 72px; }}
.kicker {{ margin:0; color:var(--accent); font:800 11px var(--mono); text-transform:uppercase; letter-spacing:.14em; }}
h1.report-title {{ margin:8px 0 12px; max-width:900px; color:var(--text); font:700 clamp(36px,6.5vw,68px)/.96 var(--read); letter-spacing:-.055em; }}
.meta {{ display:flex; flex-wrap:wrap; gap:9px 14px; margin:0 0 20px; color:var(--muted); font:11px var(--mono); text-transform:uppercase; }}
.meta b {{ color:#fff; font-weight:700; }}
article {{ border-top:1px solid var(--line); background:#000; padding-top:18px; }}
.report-section {{ margin:18px 0; padding:18px 0; border-top:1px solid var(--line-soft); }}
.section-title {{ display:flex; gap:10px; align-items:flex-start; margin:0 0 10px; color:#fff; font:700 25px/1.08 var(--read); letter-spacing:-.02em; }}
.section-title:before {{ content:""; flex:0 0 7px; width:7px; height:22px; margin-top:3px; background:var(--accent); box-shadow:0 0 18px rgba(245,164,0,.22); }}
h3 {{ margin:18px 0 8px; color:#fff; font:700 18px/1.2 var(--ui); }}
p {{ margin:.78em 0; color:var(--body); }}
ul, ol {{ margin:.75em 0; padding-left:0; list-style:none; }}
li {{ position:relative; margin:.62em 0; padding-left:22px; }}
li:before {{ content:""; position:absolute; left:4px; top:.72em; width:6px; height:6px; border-radius:50%; background:var(--accent); }}
ol {{ counter-reset:item; }}
ol li {{ counter-increment:item; padding-left:34px; }}
ol li:before {{ content:counter(item); top:.05em; left:0; width:22px; height:22px; display:grid; place-items:center; color:#000; background:var(--accent); font:800 11px/1 var(--mono); }}
strong {{ color:#fff; font-weight:700; }}
em {{ color:#f0d7a0; font-style:normal; }}
article a {{ color:#fff; text-decoration:none; border-bottom:1px solid rgba(245,164,0,.6); }}
pre {{ overflow-x:auto; background:#050505; border:1px solid var(--line); padding:16px; font-size:13px; }}
code {{ font-family:var(--mono); color:#f0d7a0; }}
p code, li code {{ background:#080808; border:1px solid var(--line); padding:1px 5px; }}
footer {{ margin-top:24px; border-top:1px solid var(--line); padding-top:14px; color:var(--muted); font:11px var(--mono); display:flex; flex-wrap:wrap; gap:10px 16px; text-transform:uppercase; }}
footer b {{ color:#fff; }}
@media (max-width:700px) {{ .top {{ height:50px; padding:0 14px; }} .actions {{ gap:6px; }} .followup {{ max-width:120px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; padding:7px 9px; }} .back {{ padding:7px 9px; }} main {{ padding:14px 14px 92px; }} h1.report-title {{ font-size:clamp(32px,9vw,46px); }} body {{ font-size:15.5px; line-height:1.54; }} .section-title {{ font-size:22px; }} }}
</style>
</head>
<body>
<header class="top"><a class="ticker" href="/"><em>ACTA</em> / REPORT</a><div class="actions">{followup_link}<a class="back" href="/">Back to Acta</a></div></header>
<main>
  <p class="kicker">Acta Situation Room · Detail</p>
  <h1 class="report-title">{html.escape(title)}</h1>
  <div class="meta">{''.join(footer_bits)}</div>
  <article>{rendered}</article>
</main>
</body>
</html>
"""


def _detail_body(item: CronSituationItem) -> str:
    if not item.latest_md:
        return f"# {item.name}\n\nNo Markdown output exists yet."
    text = item.latest_md.read_text(encoding="utf-8", errors="replace")
    response = _extract_response(text)
    if not response or response.strip() == "[SILENT]":
        response = "No visible response was produced for this run."
    return response


def attach_artifact_urls(
    items: Sequence[CronSituationItem],
    publish_settings: Mapping[str, Any],
    output_dir: Path,
) -> list[CronSituationItem]:
    output_dir.mkdir(parents=True, exist_ok=True)
    linked: list[CronSituationItem] = []
    for item in items:
        url: str | None = None
        source_html = item.latest_html
        temp_html: Path | None = None
        if item.latest_md is not None:
            # Re-render from Markdown even when an older HTML artifact exists so
            # historical content receives the current Acta detail UI.
            temp_html = output_dir / f"{item.job_id}-{item.latest_md.stem}.html"
            detail_html = render_acta_detail_report(
                _detail_body(item),
                HtmlReportMetadata(
                    job_id=item.job_id,
                    job_name=item.name,
                    run_time=item.latest_time.isoformat() if item.latest_time else "",
                    source_filename=item.latest_md.name,
                ),
                telegram_url=item.telegram_url,
            )
            temp_html.write_text(detail_html, encoding="utf-8")
            source_html = temp_html
        if source_html is not None:
            try:
                url = publish_html_artifact(source_html, {"id": item.job_id}, publish_settings)
            except HtmlArtifactPublishError:
                url = None
        linked.append(
            CronSituationItem(
                job_id=item.job_id,
                name=item.name,
                schedule=item.schedule,
                deliver=item.deliver,
                enabled=item.enabled,
                latest_md=item.latest_md,
                latest_html=item.latest_html,
                latest_time=item.latest_time,
                status=item.status,
                excerpt=item.excerpt,
                artifact_url=url,
                telegram_url=item.telegram_url,
            )
        )
    return linked


def build_dashboard(
    hermes_home: Path | None = None,
    publish: bool = False,
) -> tuple[Path, str | None]:
    home = hermes_home or get_hermes_home()
    if load_dotenv is not None:
        load_dotenv(home / ".env", override=True)
    config = _load_yaml(home / "config.yaml")
    html_cfg = ((config.get("cron") or {}).get("html_artifacts") or {}) if isinstance(config, dict) else {}
    publish_settings = dict(html_cfg.get("publish") or {})
    publish_settings["enabled"] = bool(publish_settings.get("enabled", False))
    output_dir = home / "cron" / "output" / "acta-situation-room"
    dates = available_run_dates(home)
    selected_date = dates[0] if dates else None
    items = collect_situation_items(home, run_date=selected_date)
    if publish:
        publish_settings["enabled"] = True
        items = attach_artifact_urls(items, publish_settings, output_dir / "details")
    generated_at = datetime.now(timezone.utc)
    dashboard_html = render_dashboard(
        items,
        generated_at=generated_at,
        selected_date=selected_date,
        archive_dates=dates,
        feed_preferences=acta_dashboard_config(config),
    )
    dashboard_path = output_dir / f"{generated_at.strftime('%Y-%m-%d_%H-%M-%S')}.html"
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    dashboard_path.write_text(dashboard_html, encoding="utf-8")
    dashboard_url: str | None = None
    if publish:
        # Stable homepage: https://acta.imperatr.com/
        dashboard_url = publish_html_artifact(
            dashboard_path,
            {"id": "acta-situation-room"},
            {**publish_settings, "object_key": "public/index.html"},
        )
        archive_index_path = output_dir / "archive.html"
        archive_index_path.write_text(render_archive_index(dates, generated_at=generated_at), encoding="utf-8")
        publish_html_artifact(
            archive_index_path,
            {"id": "acta-situation-room"},
            {**publish_settings, "object_key": "public/archive/index.html"},
        )
        jobs_path = output_dir / "jobs.html"
        jobs_path.write_text(
            render_jobs_page(
                items,
                generated_at=generated_at,
                feed_preferences=acta_dashboard_config(config),
            ),
            encoding="utf-8",
        )
        publish_html_artifact(
            jobs_path,
            {"id": "acta-situation-room"},
            {**publish_settings, "object_key": "public/jobs/index.html"},
        )
        for run_day in dates:
            day_items = collect_situation_items(home, run_date=run_day)
            day_items = attach_artifact_urls(day_items, publish_settings, output_dir / "details")
            day_path = output_dir / "archive" / f"{run_day.isoformat()}.html"
            day_path.parent.mkdir(parents=True, exist_ok=True)
            day_path.write_text(
                render_dashboard(
                    day_items,
                    generated_at=generated_at,
                    selected_date=run_day,
                    archive_dates=dates,
                    feed_preferences=acta_dashboard_config(config),
                ),
                encoding="utf-8",
            )
            publish_html_artifact(
                day_path,
                {"id": "acta-situation-room"},
                {**publish_settings, "object_key": f"public/archive/{run_day.isoformat()}.html"},
            )
    return dashboard_path, dashboard_url


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Acta cron situation-room dashboard")
    parser.add_argument("--publish", action="store_true", help="Upload dashboard/detail pages to Acta and print signed URL")
    args = parser.parse_args(argv)
    path, url = build_dashboard(publish=args.publish)
    print(path)
    if url:
        print(url)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
