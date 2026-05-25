"""Generate an Acta situation-room dashboard for latest cron outputs."""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qs, urlparse

try:
    import yaml
except Exception:  # pragma: no cover - yaml is available in Hermes runtime
    yaml = None  # type: ignore[assignment]

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional in tests
    load_dotenv = None  # type: ignore[assignment]

from hermes_constants import get_hermes_home

from cron.acta_catalog import default_catalog_path, default_outputs_dir, import_acta_outputs, load_catalog
from cron.html_artifacts import CSP, REPORT_END, REPORT_START, HtmlReportMetadata, render_html_report, render_report_body
from cron.html_publish import HtmlArtifactPublishError, publish_html_artifact

DEFAULT_HIDDEN_JOBS = ("e9b0a041ced3", "P Morning Audio Briefing")


def _is_safe_http_or_root_url(value: str | None, *, host_suffix: str | None = None) -> bool:
    """Return whether a rendered Acta href is safe to expose as a clickable URL."""
    if not value:
        return False
    if "\\" in value or any(ord(char) < 32 or ord(char) == 127 for char in value):
        return False
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"}:
        if host_suffix is not None and parsed.scheme != "https":
            return False
        return host_suffix is None or parsed.hostname == host_suffix or (parsed.hostname or "").endswith(f".{host_suffix}")
    return not parsed.scheme and not parsed.netloc and value.startswith("/") and not value.startswith("//")


_SIGNED_ACTA_ARTIFACT_SEGMENT_RE = r"(?=[A-Za-z0-9._-]*[A-Za-z0-9])[A-Za-z0-9._-]+"
_SIGNED_ACTA_ARTIFACT_PATH_RE = re.compile(
    rf"^/r/{_SIGNED_ACTA_ARTIFACT_SEGMENT_RE}/{_SIGNED_ACTA_ARTIFACT_SEGMENT_RE}\.html$"
)


def _has_signed_artifact_query(value: str) -> bool:
    query = parse_qs(urlparse(value).query)
    return bool(query.get("exp")) and bool(query.get("sig"))


def _is_safe_signed_acta_artifact_url(value: str | None) -> bool:
    """Return whether an Acta artifact URL is safe and signed for clickable output rows."""
    if not _is_safe_http_or_root_url(value, host_suffix="acta.imperatr.com"):
        return False
    assert value is not None
    parsed = urlparse(value)
    if parsed.username is not None or parsed.password is not None:
        return False
    if parsed.scheme == "https" and parsed.hostname != "acta.imperatr.com":
        return False
    if not _SIGNED_ACTA_ARTIFACT_PATH_RE.fullmatch(parsed.path):
        return False
    return _has_signed_artifact_query(value)


def _is_safe_telegram_url(value: str | None) -> bool:
    """Return whether a Telegram URL is an absolute HTTPS t.me link."""
    if not _is_safe_http_or_root_url(value, host_suffix="t.me"):
        return False
    assert value is not None
    parsed = urlparse(value)
    return parsed.scheme == "https" and parsed.hostname == "t.me" and parsed.username is None and parsed.password is None


def _inline_script_csp(script: str) -> str:
    digest = base64.b64encode(hashlib.sha256(script.encode("utf-8")).digest()).decode("ascii")
    return f"{CSP}; script-src 'sha256-{digest}'"


def _inline_style_and_script_csp(style: str, script: str) -> str:
    style_digest = base64.b64encode(hashlib.sha256(style.encode("utf-8")).digest()).decode("ascii")
    csp = _inline_script_csp(script)
    style_source = "style-src 'unsafe-inline'"
    if style_source not in csp:
        raise RuntimeError("base CSP style-src directive missing unsafe-inline placeholder")
    return csp.replace(style_source, f"style-src 'sha256-{style_digest}'")


def _outputs_read_state_script() -> str:
    return """
(function(){
  var KEY='acta:read:v1';
  var COOKIE='acta_read_v1';
  function readFromCookie(){
    var parts=(document.cookie||'').split('; ');
    for(var i=0;i<parts.length;i++){
      if(parts[i].indexOf(COOKIE+'=')===0){
        try{ return JSON.parse(decodeURIComponent(parts[i].slice(COOKIE.length+1)))||{}; }catch(e){ return {}; }
      }
    }
    return {};
  }
  function writeToCookie(value){
    try{ document.cookie=COOKIE+'='+encodeURIComponent(JSON.stringify(value))+'; Max-Age=31536000; Path=/; SameSite=Lax; Secure'; }catch(e){}
  }
  var state=readFromCookie();
  try{ state=JSON.parse(localStorage.getItem(KEY)||JSON.stringify(state)||'{}')||state||{}; }catch(e){ state=state||{}; }
  function save(){ try{ localStorage.setItem(KEY, JSON.stringify(state)); }catch(e){} writeToCookie(state); }
  function apply(el){
    var k=el.dataset.readKey || '';
    var isRead=Object.prototype.hasOwnProperty.call(state,k) ? !!state[k] : el.dataset.readInitial==='true';
    el.classList.toggle('read', isRead);
    el.classList.toggle('unread', !isRead);
    var label=el.querySelector('.read-state');
    if(label) label.textContent=isRead?'READ':'UNREAD';
    var button=el.querySelector('.read-toggle');
    if(button){
      button.textContent=isRead?'Mark unread':'Mark read';
      button.setAttribute('aria-label', (isRead?'Mark output unread: ':'Mark output read: ')+(el.dataset.readTitle||''));
    }
  }
  function updateUnreadCount(){
    var unread=0;
    document.querySelectorAll('.output-row.readable').forEach(function(row){ if(!row.classList.contains('read')) unread++; });
    document.querySelectorAll('[data-unread-count]').forEach(function(el){ el.textContent=String(unread); el.dataset.unreadCount=String(unread); });
  }
  function setRead(el, value){
    var k=el.dataset.readKey || '';
    if(!k) return;
    state[k]=!!value;
    save();
    apply(el);
    updateUnreadCount();
  }
  document.querySelectorAll('.output-row.readable').forEach(function(el){
    apply(el);
    el.querySelectorAll('.read-toggle').forEach(function(button){
      button.addEventListener('click', function(ev){
        ev.preventDefault();
        ev.stopPropagation();
        setRead(el, el.classList.contains('read') ? false : true);
      });
    });
    el.querySelectorAll('.output-open-overlay').forEach(function(anchor){
      anchor.addEventListener('click', function(){
        setRead(el, true);
      });
    });
  });
  updateUnreadCount();
})();
""".strip()


def _dashboard_inline_script() -> str:
    return """
(function(){
  var KEY='acta:read:v1';
  var COOKIE='acta_read_v1';
  function readFromCookie(){
    var parts=(document.cookie||'').split('; ');
    for(var i=0;i<parts.length;i++){
      if(parts[i].indexOf(COOKIE+'=')===0){
        try{ return JSON.parse(decodeURIComponent(parts[i].slice(COOKIE.length+1)))||{}; }catch(e){ return {}; }
      }
    }
    return {};
  }
  function writeToCookie(value){
    try{ document.cookie=COOKIE+'='+encodeURIComponent(JSON.stringify(value))+'; Max-Age=31536000; Path=/; SameSite=Lax; Secure'; }catch(e){}
  }
  var state=readFromCookie();
  try{ state=JSON.parse(localStorage.getItem(KEY)||JSON.stringify(state)||'{}')||state||{}; }catch(e){ state=state||{}; }
  function save(){ try{ localStorage.setItem(KEY, JSON.stringify(state)); }catch(e){} writeToCookie(state); }
  var pull=document.querySelector('.pull-refresh');
  var mobileBar=document.querySelector('.mobilebar');
  var topNav=document.querySelector('.date-nav');
  function setMobileBarVisible(visible){ if(mobileBar) mobileBar.classList.toggle('visible', !!visible); }
  if(mobileBar && topNav){
    if('IntersectionObserver' in window){
      new IntersectionObserver(function(entries){
        var entry=entries[0];
        setMobileBarVisible(!entry.isIntersecting && window.matchMedia('(max-width: 980px)').matches);
      }, {root:null, threshold:0.01}).observe(topNav);
    } else {
      var navBottom=topNav.offsetTop+topNav.offsetHeight;
      function updateMobileBar(){ setMobileBarVisible(window.matchMedia('(max-width: 980px)').matches && window.scrollY>navBottom); }
      window.addEventListener('scroll', updateMobileBar, {passive:true});
      window.addEventListener('resize', updateMobileBar, {passive:true});
      updateMobileBar();
    }
  }
  var psx=0, psy=0, pdy=0, pulling=false, ptrReady=false;
  function ptrStart(ev){
    if(window.scrollY>2) return;
    var t=(ev.touches&&ev.touches[0]) || ev;
    psx=t.clientX||0; psy=t.clientY||0; pdy=0; pulling=true; ptrReady=false;
  }
  function ptrMove(ev){
    if(!pulling || !pull) return;
    var t=(ev.touches&&ev.touches[0]) || ev;
    var dx=Math.abs((t.clientX||0)-psx); pdy=(t.clientY||0)-psy;
    if(pdy>12 && pdy>dx*1.25 && window.scrollY<=2){
      var y=Math.min(72, Math.max(0, pdy*.55));
      pull.classList.add('visible');
      ptrReady=y>=54;
      pull.classList.toggle('ready', ptrReady);
      pull.textContent=ptrReady?'RELEASE TO REFRESH':'PULL TO REFRESH';
      if(ev.cancelable) ev.preventDefault();
    }
  }
  function ptrEnd(){
    if(!pulling) return;
    pulling=false;
    if(ptrReady){
      if(pull){ pull.textContent='REFRESHING'; pull.classList.add('visible','ready'); }
      location.reload();
      return;
    }
    if(pull){ pull.classList.remove('visible','ready'); pull.textContent='PULL TO REFRESH'; }
  }
  document.addEventListener('touchstart', ptrStart, {passive:true});
  document.addEventListener('touchmove', ptrMove, {passive:false});
  document.addEventListener('touchend', ptrEnd, {passive:true});
  document.addEventListener('touchcancel', ptrEnd, {passive:true});
  function apply(el){
    var k=el.dataset.readKey || '';
    var isRead=!!state[k];
    el.classList.toggle('read', isRead);
    el.classList.toggle('unread', !isRead);
    var label=el.querySelector('.read-state');
    if(label) label.textContent=isRead?'READ':'UNREAD';
    var action=el.querySelector('.swipe-action');
    if(action) action.textContent=isRead?'MARK UNREAD':'MARK READ';
  }
  function toggle(el){
    var k=el.dataset.readKey || '';
    if(!k) return;
    state[k]=!state[k];
    save();
    apply(el);
    el.classList.add('swipe-peek');
    setTimeout(function(){ el.classList.remove('swipe-peek'); }, 260);
  }
  document.querySelectorAll('.readable').forEach(function(el){
    apply(el);
    var sx=0, sy=0, dx=0, swiping=false, didSwipe=false, active=false;
    function openReadable(){
      var k=el.dataset.readKey || '';
      if(k){ state[k]=true; save(); apply(el); }
      var openUrl=el.dataset.openUrl || '';
      if(openUrl) window.location.href=openUrl;
    }
    function point(ev){
      var t=(ev.touches&&ev.touches[0]) || ev;
      return {x:t.clientX||0, y:t.clientY||0};
    }
    function start(ev){
      var p=point(ev); sx=p.x; sy=p.y; dx=0; swiping=false; didSwipe=false; active=true;
    }
    function move(ev){
      if(!active) return;
      var p=point(ev); dx=p.x-sx;
      var dy=Math.abs(p.y-sy);
      if(dx>10 && Math.abs(dx)>dy*1.15){
        swiping=true; didSwipe=true; el.classList.add('swiping','swipe-peek');
        if(ev.cancelable) ev.preventDefault();
      }
    }
    function end(ev){
      if(!active) return;
      active=false;
      if(swiping){
        if(ev.cancelable) ev.preventDefault();
        if(dx>58) toggle(el);
      }
      el.classList.remove('swiping');
      if(didSwipe){
        setTimeout(function(){ el.classList.remove('swipe-peek'); }, 220);
      } else {
        el.classList.remove('swipe-peek');
      }
    }
    el.addEventListener('click', function(ev){
      if(didSwipe){ ev.preventDefault(); ev.stopPropagation(); didSwipe=false; return; }
      if(ev.target && ev.target.closest && ev.target.closest('a')) return;
      openReadable();
    }, true);
    el.querySelectorAll('.row-open-overlay').forEach(function(anchor){
      anchor.addEventListener('click', function(){
        var k=el.dataset.readKey || '';
        if(k){ state[k]=true; save(); apply(el); }
      });
    });
    el.addEventListener('touchstart', start, {passive:true});
    el.addEventListener('touchmove', move, {passive:false});
    el.addEventListener('touchend', end, {passive:false});
    el.addEventListener('pointerdown', function(ev){ if(ev.pointerType==='touch'||ev.pointerType==='pen') start(ev); });
    el.addEventListener('pointermove', function(ev){ if(ev.pointerType==='touch'||ev.pointerType==='pen') move(ev); });
    el.addEventListener('pointerup', function(ev){ if(ev.pointerType==='touch'||ev.pointerType==='pen') end(ev); });
    el.addEventListener('pointercancel', end);
  });
})();
""".strip()


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


@dataclass(frozen=True)
class ActaOutputItem:
    id: str
    title: str
    href: str
    summary: str
    tags: tuple[str, ...]
    source_name: str
    created_at: str
    updated_at: str
    pinned: bool = False
    read: bool = False
    archived: bool = False


@dataclass(frozen=True)
class ActaRunItem:
    job_id: str
    name: str
    schedule: str
    deliver: str
    enabled: bool
    run_id: str
    run_time: datetime | None
    status: str
    excerpt: str
    source_name: str
    has_markdown: bool
    has_html: bool
    telegram_url: str | None = None
    artifact_url: str | None = None
    md_path: Path | None = None
    html_path: Path | None = None


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
    """Return a safe Telegram web link for explicit cron delivery targets.

    Hermes cron delivery targets for Telegram forum topics are stored as
    ``telegram:<chat_id>:<thread_id>``. Telegram's web/app deep link for
    private supergroups uses the internal ID with the leading ``-100`` removed:
    ``https://t.me/c/<internal_id>/<topic_or_message_id>``. Public ``@username``
    targets are also supported, but only when the username and optional numeric
    thread/message ID pass a fail-closed allowlist.
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
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{4,31}", username):
            if thread:
                if thread.isdigit():
                    return f"https://t.me/{username}/{thread}"
            else:
                return f"https://t.me/{username}"
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


def _extract_response_if_present(markdown: str) -> str | None:
    match = re.search(r"(?:^|\n)## Response\s*\n", markdown)
    if not match:
        return None
    response = markdown[match.end():]
    next_heading = re.search(r"\n## [^\n]+\n", response)
    if next_heading:
        response = response[: next_heading.start()]
    return _strip_embedded_html_report(response)


def _extract_response(markdown: str) -> str:
    response = _extract_response_if_present(markdown)
    if response is None:
        return _strip_embedded_html_report(markdown)
    return response


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
                response = _extract_response_if_present(latest_md.read_text(encoding="utf-8", errors="replace")) or ""
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
                excerpt=_plain_excerpt(
                    response or ("No visible response was produced for this run." if latest_md or latest_html else "No output yet.")
                ),
                telegram_url=_telegram_url_from_job(job),
            )
        )
    return sorted(items, key=lambda item: (item.enabled, item.latest_time or datetime.min.replace(tzinfo=timezone.utc)), reverse=True)


def _parse_iso_datetime(value: object) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _is_safe_catalog_href(value: str | None) -> bool:
    return bool(value and re.fullmatch(r"/outputs/[a-z0-9][a-z0-9-]*/?", value))


def collect_catalog_outputs(hermes_home: Path | None = None) -> list[ActaOutputItem]:
    """Load persistent Acta outputs, importing static artifacts once when needed."""
    home = hermes_home or get_hermes_home()
    catalog_path = default_catalog_path(home)
    try:
        catalog = import_acta_outputs(hermes_home=home, save=True)
    except OSError:
        catalog = load_catalog(catalog_path)

    outputs: list[ActaOutputItem] = []
    for entry in catalog.get("outputs", []):
        if not isinstance(entry, Mapping) or bool(entry.get("archived", False)):
            continue
        entry_id = str(entry.get("id") or "output")
        href = str(entry.get("href") or "")
        if not _is_safe_catalog_href(href):
            href = ""
        raw_source_ref = entry.get("source_ref")
        source_ref: Mapping[str, Any] = raw_source_ref if isinstance(raw_source_ref, Mapping) else {}
        outputs.append(
            ActaOutputItem(
                id=entry_id,
                title=str(entry.get("title") or entry.get("id") or "Output"),
                href=href,
                summary=str(entry.get("summary") or ""),
                tags=tuple(str(tag) for tag in (entry.get("tags") or []) if str(tag).strip()),
                source_name=str(source_ref.get("name") or source_ref.get("label") or "acta-output"),
                created_at=str(entry.get("created_at") or ""),
                updated_at=str(entry.get("updated_at") or ""),
                pinned=bool(entry.get("pinned", False)),
                read=bool(entry.get("read", False)),
                archived=bool(entry.get("archived", False)),
            )
        )
    return sorted(outputs, key=lambda item: (not item.pinned, -(_parse_iso_datetime(item.updated_at) or datetime.min.replace(tzinfo=timezone.utc)).timestamp(), item.title.casefold()))


def collect_run_history(hermes_home: Path | None = None, limit: int = 200) -> list[ActaRunItem]:
    """Scan cron output history across all run files, excluding Acta's own dashboard job."""
    home = hermes_home or get_hermes_home()
    jobs_path = home / "cron" / "jobs.json"
    jobs = {str(job.get("id") or ""): job for job in (_jobs_from_file(jobs_path) if jobs_path.exists() else []) if str(job.get("id") or "")}
    output_root = home / "cron" / "output"
    if not output_root.exists():
        return []
    grouped: dict[tuple[str, str], dict[str, Path]] = {}
    output_root_resolved = output_root.resolve()
    for path in output_root.glob("*/*"):
        if path.parent.name == "acta-situation-room" or path.suffix.lower() not in {".md", ".html"} or path.is_symlink() or not path.is_file():
            continue
        try:
            path.resolve().relative_to(output_root_resolved)
        except (OSError, ValueError):
            continue
        grouped.setdefault((path.parent.name, path.stem), {})[path.suffix.lower()] = path

    runs: list[ActaRunItem] = []
    for (job_id, run_id), paths in grouped.items():
        md_path = paths.get(".md")
        html_path = paths.get(".html")
        job = jobs.get(job_id, {})
        response = ""
        if md_path is not None:
            try:
                response = _extract_response_if_present(md_path.read_text(encoding="utf-8", errors="replace")) or ""
            except OSError:
                response = ""
        status = "fresh" if response and response.strip() != "[SILENT]" else "silent"
        run_time = _latest_run_time(md_path, html_path)
        source_path = md_path or html_path
        runs.append(
            ActaRunItem(
                job_id=job_id,
                name=str(job.get("name") or job_id),
                schedule=_schedule_display(job),
                deliver=str(job.get("deliver") or ""),
                enabled=bool(job.get("enabled", True)),
                run_id=run_id,
                run_time=run_time,
                status=status,
                excerpt=_plain_excerpt(response or "No visible response was produced for this run."),
                source_name=source_path.name if source_path else run_id,
                has_markdown=md_path is not None,
                has_html=html_path is not None,
                telegram_url=_telegram_url_from_job(job),
                md_path=md_path,
                html_path=html_path,
            )
        )
    return sorted(runs, key=lambda item: item.run_time or datetime.min.replace(tzinfo=timezone.utc), reverse=True)[:limit]


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


def _confidence_bucket(enabled: bool, status: str, timestamp: datetime | None, now: datetime) -> str:
    """Return a compact confidence bucket derived from source state.

    Acta exposes confidence as an explicit operator signal, but the dashboard
    should not invent numeric precision. Buckets are derived only from actual
    freshness/readiness state: enabled fresh/recent sources are HIGH; enabled
    but older visible sources are MED; silent, missing, or paused sources are
    LOW/GAP.
    """
    if not enabled or status in {"silent", "missing"}:
        return "CONF LOW/GAP"
    if timestamp is None:
        return "CONF LOW/GAP"
    age_seconds = max(0, int((now - timestamp).total_seconds()))
    if status == "fresh" and age_seconds <= 24 * 60 * 60:
        return "CONF HIGH"
    return "CONF MED"


def _confidence_label(item: CronSituationItem, now: datetime) -> str:
    return _confidence_bucket(item.enabled, item.status, item.latest_time, now)


def _status_class(item: CronSituationItem) -> str:
    if not item.enabled:
        return "paused"
    return item.status


def _telegram_link_html(item: CronSituationItem, label: str = "THREAD") -> str:
    if not _is_safe_telegram_url(item.telegram_url):
        return ""
    assert item.telegram_url is not None
    return (
        f' · <a class="thread-link" href="{html.escape(item.telegram_url, quote=True)}" '
        f'target="_blank" rel="noopener">{html.escape(label)}</a>'
    )


def _is_system_item(item: CronSituationItem) -> bool:
    lowered = item.name.lower()
    return "situation room refresh" in lowered or (item.deliver or "").lower() == "local"


_DEV_FEED_KEYWORDS = (
    "startup sprint",
    "sprint ceo",
    "self-healing",
    "self-healing sentinel",
    "repair loop",
    "qa smoke",
    "smoke run",
    "user-testing sweep",
    "user testing sweep",
    "user testing",
    "security audit",
    "security scan",
    "app security",
    "vesta import",
    "vesta startup",
    "acta startup",
    "minerva startup",
    "praetor startup",
)


def _feed_lane(item: CronSituationItem) -> str:
    """Classify Acta source rows into operator-readable feed lanes.

    High-frequency autonomous app-development jobs are useful, but when they
    share the same stream as daily life/news automations they bury the user's
    actual daily briefing surface. Keep system/local jobs separate, isolate
    SDLC/dev loops, and let everything else remain in the daily-life feed.
    """
    if _is_system_item(item):
        return "system"
    haystack = f"{item.job_id} {item.name}".casefold()
    if any(keyword in haystack for keyword in _DEV_FEED_KEYWORDS):
        return "dev"
    return "daily"


def _feed_lane_label(lane: str) -> str:
    return {
        "daily": "Daily life feed",
        "dev": "Development sprint cycles",
        "system": "System/local jobs",
    }.get(lane, "Other feed")


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
        status_label = item.status
        confidence = _confidence_label(item, now)
        latest = item.latest_time.isoformat() if item.latest_time else "No run yet"
        age = _age_label(item.latest_time, now)
        schedule = item.schedule or "manual"
        source_path = item.latest_md or item.latest_html
        source_label = source_path.name if source_path else "no-source"
        artifact_url = item.artifact_url if item.enabled and _is_safe_signed_acta_artifact_url(item.artifact_url) else None
        href = html.escape(artifact_url, quote=True) if artifact_url else ""
        open_attr = f' data-open-url="{href}"' if href else ' aria-disabled="true"'
        openable_class = " openable" if href else " no-page"
        open_overlay = (
            f'<a class="row-open-overlay job-open-overlay" href="{href}" aria-label="Open signed Acta artifact: {html.escape(item.name, quote=True)}"></a>'
            if href
            else ""
        )
        open_state = (
            '<span class="job-open-state signed">OPEN/SIGNED</span>'
            if href
            else '<span class="job-open-state muted">NO PAGE</span>'
        )
        rows.append(
            f"""
<div class="job-row {status_class}{openable_class}"{open_attr}>
  {open_overlay}
  <div class="job-rank">{index:02d}</div>
  <div class="job-main"><b>{_safe_text(item.name)}</b><span>{_safe_text(item.job_id)} · SOURCE {_safe_text(source_label)} · {_safe_text(item.deliver or "local")}{_telegram_link_html(item)}</span></div>
  <div class="job-schedule"><em>SCHEDULE</em>{_safe_text(schedule)}</div>
  <div class="job-last"><em>LAST RUN</em><time>{_safe_text(latest)}</time><small><span class="confidence-chip">{_safe_text(confidence)}</span><span>{_safe_text(status_label)}</span><span>{_safe_text(age)}</span>{open_state}</small></div>
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

    def _row_signal(item: CronSituationItem) -> str:
        if _is_system_item(item):
            return "SYS"
        if item.status == "missing":
            return "GAP"
        if item.status == "silent":
            return "QUIET"
        if not item.enabled:
            return "PAUSED"
        return "OUT"

    daily_items = [item for item in ordered_items if _feed_lane(item) == "daily"]
    dev_items = [item for item in ordered_items if _feed_lane(item) == "dev"]
    system_items = [item for item in ordered_items if _feed_lane(item) == "system"]
    lead_item = next(
        (item for item in daily_items if item.status == "fresh"),
        daily_items[0] if daily_items else (ordered_items[0] if ordered_items else None),
    )
    grouped_feed_items: dict[str, list[CronSituationItem]] = {
        "daily": [item for item in daily_items if item is not lead_item],
        "dev": [item for item in dev_items if item is not lead_item],
        "system": [item for item in system_items if item is not lead_item],
    }

    rows_by_lane: dict[str, list[str]] = {"daily": [], "dev": [], "system": []}
    read_order: list[str] = []
    audit_rows: list[str] = []

    def append_feed_row(item: CronSituationItem, index: int, lane: str) -> None:
        status_class = _status_class(item)
        status_label = "paused" if not item.enabled else item.status
        latest = item.latest_time.isoformat() if item.latest_time else "No run yet"
        age = _age_label(item.latest_time, now)
        confidence = _confidence_label(item, now)
        row_signal = _row_signal(item)
        artifact_url = item.artifact_url if item.enabled and _is_safe_signed_acta_artifact_url(item.artifact_url) else None
        href = html.escape(artifact_url, quote=True) if artifact_url else ""
        open_label = "OPEN" if href else "NO PAGE"
        telegram_url = item.telegram_url if _is_safe_telegram_url(item.telegram_url) else None
        ask_link = (
            f'<a class="ask-label" href="{html.escape(telegram_url, quote=True)}" target="_blank" rel="noopener">ASK</a>'
            if telegram_url
            else ""
        )
        read_key = html.escape(_read_key(item), quote=True) if href else ""
        open_attr = f' data-open-url="{href}" data-read-key="{read_key}"' if href else ' aria-disabled="true"'
        readable_class = " readable unread" if href else ""
        open_overlay = (
            f'<a class="row-open-overlay" href="{href}" aria-label="Open briefing: {html.escape(item.name, quote=True)}"></a>'
            if href
            else ""
        )
        swipe_action = '<div class="swipe-action" aria-hidden="true">MARK READ</div>' if href else ""
        read_state = '<span class="read-state">UNREAD</span>' if href else ""
        row_read_dot = '<span class="read-dot"></span>' if href else ""
        rows_by_lane[lane].append(
            f"""
<section class="brief-row{readable_class} {status_class}" data-feed-lane="{_safe_text(lane)}"{open_attr}>
  {open_overlay}
  {swipe_action}
  <div class="swipe-content">
    <div class="row-signal">{row_read_dot}<span>{_safe_text(row_signal)}</span></div>
    <div class="brief-copy">
      <h2>{_safe_text(item.name)}</h2>
      <p>{_safe_text(item.excerpt)}</p>
      <div class="row-kicker"><span class="lane-chip">{_safe_text(_feed_lane_label(lane))}</span>{read_state}<span class="confidence-chip">{_safe_text(confidence)}</span><span>{_safe_text(status_label)}</span><span>{_safe_text(age)}</span><span>{_safe_text(item.schedule or "manual")}</span></div>
      <div class="source-line">{_safe_text(item.job_id)} · {_safe_text(item.deliver or "local")} · {_safe_text(latest)}</div>
    </div>
    <span class="card-actions"><span class="open-label">{open_label}</span>{ask_link}</span>
  </div>
</section>"""
        )
        if len(read_order) < 4 and lane == "daily":
            read_order.append(
                f"""
<div class="order-row {status_class}">
  <strong>{index + 1:02d}</strong>
  <div><b>{_safe_text(item.name)}</b><p>{_safe_text(status_label)} · {_safe_text(age)}</p></div>
  <span>{_safe_text(row_signal)}</span>
</div>"""
            )

    for lane, lane_items in grouped_feed_items.items():
        for index, item in enumerate(lane_items):
            append_feed_row(item, index, lane)

    for item in ordered_items:
        if item is lead_item or not item.latest_time or len(audit_rows) >= 4:
            continue
        lane = _feed_lane(item)
        status_label = "paused" if not item.enabled else item.status
        audit_rows.append(
            f"""
<div class="audit-row">
  <time>{_safe_text(item.latest_time.strftime('%H:%M'))}</time>
  <div><b>{_safe_text(item.name)}</b><span>{_safe_text(_feed_lane_label(lane))} · {_safe_text(status_label)} · {_safe_text(item.job_id)}</span></div>
</div>"""
        )

    def _feed_section(lane: str, title: str, subtitle: str) -> str:
        lane_rows = ''.join(rows_by_lane[lane])
        if not lane_rows:
            empty_message = "No additional outputs in this lane yet." if lead_item and _feed_lane(lead_item) == lane else "No visible outputs in this lane yet."
            lane_rows = f'<p class="empty-feed">{html.escape(empty_message)}</p>'
        return f"""
<div class="feed-section lane-section-{lane}">
  <div class="feed-section-title"><b>{html.escape(title)}</b><span>{html.escape(subtitle)}</span></div>
  <section class="feed" data-feed-lane="{html.escape(lane, quote=True)}">{lane_rows}</section>
</div>"""

    daily_section = _feed_section("daily", "Daily life feed", "news, newsletters, weather, sports, lunch")
    dev_section = _feed_section("dev", "Development sprint cycles", "Acta, Vesta, QA, security, app agents")
    system_section = (
        _feed_section("system", "System/local jobs", "refreshers and silent maintenance")
        if system_items or (lead_item and _feed_lane(lead_item) == "system")
        else ""
    )

    jobs_rows = _render_jobs_rows(ordered_items, now)

    lead_title = lead_item.name if lead_item else "No briefing output yet"
    lead_excerpt = lead_item.excerpt if lead_item else "Acta is waiting for the next generated briefing packet."
    lead_artifact_url = (
        lead_item.artifact_url
        if lead_item and lead_item.enabled and _is_safe_signed_acta_artifact_url(lead_item.artifact_url)
        else None
    )
    lead_href = html.escape(lead_artifact_url, quote=True) if lead_artifact_url else ""
    lead_href_attr = f' data-open-url="{lead_href}"' if lead_href else ' aria-disabled="true"'
    lead_open_overlay = (
        f'<a class="row-open-overlay" href="{lead_href}" aria-label="Open briefing: {html.escape(lead_item.name, quote=True)}"></a>'
        if lead_item and lead_href
        else ""
    )
    lead_telegram_url = lead_item.telegram_url if lead_item and _is_safe_telegram_url(lead_item.telegram_url) else None
    lead_ask_link = (
        f'<a class="ask-label" href="{html.escape(lead_telegram_url, quote=True)}" target="_blank" rel="noopener">ASK TELEGRAM</a>'
        if lead_telegram_url
        else ""
    )
    lead_read_key = html.escape(_read_key(lead_item), quote=True) if lead_item and lead_href else ""
    lead_class = "lead readable unread" if lead_href else "lead"
    lead_read_attr = f' data-read-key="{lead_read_key}"' if lead_href else ""
    lead_label_read_state = '<span class="read-dot"></span><span class="read-state">UNREAD</span>' if lead_href else ""
    lead_open_hint = "open first" if lead_href else "no page"
    lead_row_meta = "row opens" if lead_href else "signed rows only"
    lead_confidence = _confidence_label(lead_item, now) if lead_item else "CONF LOW/GAP"
    dashboard_script = _dashboard_inline_script()
    dashboard_csp_placeholder = "__ACTA_DASHBOARD_CSP__"

    page_html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, viewport-fit=cover, user-scalable=no">
<meta http-equiv="Content-Security-Policy" content="{dashboard_csp_placeholder}">
<title>Acta Situation Room</title>
<meta name="description" content="Acta Imperatr situation room: private briefing packets, source provenance, jobs, and archive in a compact operator surface.">
<style>
:root {{ color-scheme: dark; --black:#03060b; --bg:#03060b; --bg2:#071018; --panel:rgba(255,255,255,.055); --panel2:rgba(255,255,255,.085); --line:rgba(255,255,255,.105); --line-soft:rgba(255,255,255,.07); --text:#f5f7fb; --body:rgba(245,247,251,.86); --muted:rgba(245,247,251,.66); --faint:rgba(245,247,251,.42); --accent:#756cff; --acta:#756cff; --acta2:#23a7ff; --green:#57a773; --red:#d05a4e; --ui:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif; --read:'Iowan Old Style','Charter','Source Serif Pro',Georgia,serif; --mono:'SFMono-Regular','Roboto Mono','IBM Plex Mono',Consolas,monospace; }}
* {{ box-sizing:border-box; }}
html {{ width:100%; min-width:320px; overflow-x:hidden; background:#03060b; }}
body {{ margin:0; width:100%; min-width:320px; overflow-x:hidden; background:radial-gradient(circle at 18% 8%, rgba(117,108,255,.20), transparent 30%), radial-gradient(circle at 82% 12%, rgba(35,167,255,.12), transparent 28%), linear-gradient(145deg,#020408,#071018 52%,#030509); color:var(--body); font:14px/1.45 var(--ui); -webkit-font-smoothing:antialiased; text-rendering:optimizeLegibility; -webkit-text-size-adjust:100%; touch-action:pan-y; }}
body:before {{ content:""; position:fixed; inset:0; pointer-events:none; opacity:.16; background-image:linear-gradient(rgba(255,255,255,.03) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.025) 1px,transparent 1px); background-size:52px 52px; mask-image:radial-gradient(circle at 50% 34%,black,transparent 78%); }}
a {{ color:inherit; }}
.shell {{ min-height:100vh; display:grid; grid-template-columns:220px minmax(0,1fr); background:transparent; position:relative; }}
.rail {{ border-right:1px solid var(--line); background:rgba(3,6,11,.72); backdrop-filter:blur(22px) saturate(145%); display:flex; flex-direction:column; }}
.brand {{ height:58px; border-bottom:1px solid var(--line); display:flex; align-items:center; gap:10px; padding:0 14px; }}
.logo {{ width:32px; height:32px; border-radius:12px; border:1px solid rgba(255,255,255,.18); display:grid; place-items:center; font:600 18px var(--read); color:#fff; background:linear-gradient(135deg,rgba(117,108,255,.62),rgba(35,167,255,.34)); box-shadow:0 0 24px rgba(117,108,255,.28); }}
.brand b {{ font:600 20px var(--read); letter-spacing:-.04em; color:#fff; text-transform:none; }}
.brand small {{ display:block; font:10px var(--mono); color:var(--muted); letter-spacing:.08em; }}
.nav-side {{ padding:14px 8px; }}
.nav-side h4 {{ margin:16px 8px 7px; color:var(--faint); font:700 10px var(--mono); letter-spacing:.12em; text-transform:uppercase; }}
.nav-side a {{ display:flex; align-items:center; gap:8px; color:rgba(245,247,251,.72); padding:9px 10px; border-radius:14px; text-decoration:none; }}
.nav-side a.active {{ background:rgba(117,108,255,.20); color:#fff; border:1px solid rgba(117,108,255,.34); box-shadow:inset 3px 0 0 rgba(35,167,255,.54); padding-left:9px; }}
.nav-side span {{ margin-left:auto; color:var(--faint); font:10px var(--mono); }}
.railfoot {{ margin-top:auto; border-top:1px solid var(--line); padding:11px 14px; color:var(--muted); font:11px var(--mono); }}
.live {{ display:inline-block; width:7px; height:7px; border-radius:50%; background:var(--green); margin-right:7px; }}
.main {{ min-width:0; }}
.top {{ height:58px; border-bottom:1px solid var(--line); display:flex; align-items:center; gap:12px; padding:0 16px; background:rgba(3,6,11,.78); backdrop-filter:blur(22px) saturate(145%); position:sticky; top:0; z-index:2; }}
.ticker {{ color:#fff; font:800 11px var(--mono); letter-spacing:.11em; }}
.ticker em {{ font-style:normal; color:var(--acta2); text-shadow:0 0 18px rgba(35,167,255,.32); }}
.search {{ height:32px; flex:1; max-width:520px; border:1px solid var(--line); border-radius:999px; background:rgba(255,255,255,.045); color:var(--faint); display:flex; align-items:center; padding:0 12px; font:12px var(--mono); }}
.topstats {{ display:flex; gap:9px; margin-left:auto; }}
.topstats div {{ font:10px var(--mono); color:var(--muted); border:1px solid var(--line); border-radius:999px; background:rgba(255,255,255,.045); padding:5px 8px; }}
.topstats b {{ color:var(--text); font-weight:700; }}
.content {{ padding:18px; display:grid; grid-template-columns:minmax(0,1fr) 340px; gap:18px; }}
.panel-title {{ display:flex; align-items:center; justify-content:space-between; gap:10px; margin:4px 0 10px; color:#fff; font:800 11px var(--mono); letter-spacing:.12em; text-transform:uppercase; }}
.panel-title span {{ color:var(--faint); font-weight:600; letter-spacing:.08em; }}
.lead {{ display:grid; grid-template-columns:minmax(0,1fr) auto; gap:10px 12px; border:1px solid var(--line); border-radius:16px; padding:12px 13px; margin-bottom:10px; text-decoration:none; color:inherit; cursor:pointer; position:relative; overflow:hidden; background:rgba(255,255,255,.035); box-shadow:0 10px 32px rgba(0,0,0,.26); }}
.lead:before {{ content:""; position:absolute; inset:0 auto 0 0; width:3px; background:linear-gradient(180deg,var(--acta2),var(--acta)); box-shadow:0 0 14px rgba(117,108,255,.32); pointer-events:none; }}
.lead > * {{ position:relative; }}
.lead[aria-disabled='true'] {{ cursor:default; }}
.lead:hover h1 {{ color:#fff; }}
.row-open-overlay {{ position:absolute; inset:0; z-index:1; border:0; text-decoration:none; }}
.lead .ask-label, .brief-row .ask-label, .card-actions, .lead > :not(.row-open-overlay), .brief-row > .swipe-content {{ position:relative; z-index:2; }}
.row-open-overlay:focus-visible {{ outline:2px solid var(--acta2); outline-offset:2px; box-shadow:0 0 0 4px rgba(35,167,255,.14),0 10px 32px rgba(0,0,0,.26); border-radius:inherit; }}
.label {{ grid-column:1/-1; display:flex; align-items:center; flex-wrap:wrap; gap:6px; font:750 10px var(--mono); letter-spacing:.09em; color:var(--muted); text-transform:uppercase; }}
h1 {{ grid-column:1; font:720 clamp(18px,2.3vw,24px)/1.08 var(--ui); letter-spacing:-.025em; margin:0; color:#fff; max-width:980px; }}
.lead p {{ grid-column:1; font:13px/1.32 var(--ui); color:var(--body); max-width:940px; margin:0; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; }}
.meta {{ grid-column:1/-1; display:flex; flex-wrap:wrap; gap:6px; margin-top:0; color:var(--muted); font:10px var(--mono); text-transform:uppercase; }}
.meta span, .meta .ask-label {{ border:1px solid var(--line-soft); border-radius:999px; padding:3px 6px; background:rgba(255,255,255,.026); }}
.meta b {{ color:#fff; font-weight:700; }}
.output-summary {{ grid-column:2; grid-row:2 / span 2; align-self:center; justify-self:end; display:grid; gap:2px; min-width:58px; text-align:right; font:10px var(--mono); color:var(--muted); text-transform:uppercase; }}
.output-summary b {{ color:#fff; font:760 16px/1 var(--ui); }}
.output-summary span {{ color:var(--faint); letter-spacing:.08em; }}
.read-dot {{ width:8px; height:8px; border-radius:50%; background:var(--acta2); display:inline-block; margin-right:7px; box-shadow:0 0 0 2px rgba(117,108,255,.16),0 0 16px rgba(35,167,255,.28); }}
.readable.read {{ opacity:.68; }}
.readable.read .read-dot {{ background:transparent; box-shadow:inset 0 0 0 1px var(--faint); }}
.readable.read h1, .readable.read h2 {{ color:#c8c8c8; }}
.feed {{ display:flex; flex-direction:column; gap:6px; border-top:0; }}
.feed-section {{ margin:0 0 14px; }}
.feed-section-title {{ display:flex; align-items:flex-end; justify-content:space-between; gap:10px; margin:10px 0 7px; color:#fff; font:800 10px var(--mono); letter-spacing:.11em; text-transform:uppercase; }}
.feed-section-title span {{ color:var(--muted); font-weight:600; letter-spacing:.06em; text-align:right; }}
.empty-feed {{ margin:0; border:1px dashed var(--line-soft); border-radius:13px; padding:12px; color:var(--muted); font:12px var(--mono); text-transform:uppercase; letter-spacing:.06em; background:rgba(255,255,255,.018); }}
.lane-chip {{ border-color:rgba(117,108,255,.44) !important; color:#fff; background:rgba(117,108,255,.18) !important; }}
.brief-row[data-feed-lane='dev'] .swipe-content:before {{ background:linear-gradient(180deg,#ffb86b,var(--acta)); }}
.brief-row[data-feed-lane='system'] .swipe-content:before {{ background:rgba(245,247,251,.34); }}
.brief-row[data-feed-lane='dev'] {{ background:rgba(255,184,107,.035); }}
.brief-row {{ display:block; border:1px solid var(--line-soft); border-radius:13px; text-decoration:none; color:inherit; cursor:pointer; position:relative; overflow:hidden; background:rgba(255,255,255,.028); }}
.swipe-content {{ display:grid; grid-template-columns:34px minmax(0,1fr) auto; gap:9px; align-items:center; padding:8px 10px; min-height:64px; position:relative; z-index:1; background:transparent; transition:transform .22s cubic-bezier(.2,.8,.2,1), opacity .18s ease, background .18s ease; will-change:transform; }}
.swipe-content:before {{ content:""; width:2px; height:36px; border-radius:999px; background:rgba(117,108,255,.78); position:absolute; left:0; top:50%; transform:translateY(-50%); }}
.swipe-action {{ position:absolute; inset:0 auto 0 0; width:118px; display:flex; align-items:center; padding-left:16px; color:#fff; background:linear-gradient(135deg,var(--acta),var(--acta2)); font:800 10px var(--mono); letter-spacing:.08em; z-index:0; opacity:0; transition:opacity .12s ease; }}
.brief-row.swiping .swipe-content {{ transition:none; }}
.brief-row.swipe-peek .swipe-content {{ transform:translateX(92px); }}
.brief-row.swipe-peek .swipe-action, .brief-row.swiping .swipe-action {{ opacity:1; }}
.brief-row:hover .swipe-content {{ background:rgba(255,255,255,.045); }}
.brief-row[aria-disabled='true'] {{ cursor:default; }}
.row-signal {{ display:grid; gap:4px; justify-items:center; align-content:center; color:var(--muted); font:760 8px var(--mono); letter-spacing:.08em; text-transform:uppercase; }}
.row-signal span:empty {{ display:none; }}
.silent .row-signal, .missing .row-signal {{ color:#fff; }}
.paused {{ opacity:.6; }}
.confidence-chip {{ display:inline-flex; align-items:center; gap:4px; border:1px solid rgba(35,167,255,.34); border-radius:999px; padding:2px 6px; color:#fff; background:rgba(117,108,255,.13); letter-spacing:.08em; }}
.row-kicker {{ color:var(--muted); font:9.5px var(--mono); text-transform:uppercase; letter-spacing:.06em; margin-top:4px; display:flex; flex-wrap:wrap; align-items:center; gap:4px 6px; line-height:1.25; }}
.row-kicker .read-state {{ color:#fff; }}
h2 {{ font:680 15px/1.12 var(--ui); margin:0 0 2px; color:#fff; letter-spacing:-.015em; }}
.brief-copy p {{ font:13px/1.25 var(--ui); color:var(--body); margin:0; max-width:860px; display:-webkit-box; -webkit-line-clamp:1; -webkit-box-orient:vertical; overflow:hidden; }}
.source-line {{ display:block; font:9.5px var(--mono); color:var(--faint); margin-top:4px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.open-label {{ justify-self:end; align-self:center; border:1px solid var(--line-soft); border-radius:999px; color:#fff; padding:5px 7px; font:10px var(--mono); background:rgba(255,255,255,.035); }}
.card-actions {{ justify-self:end; align-self:center; display:flex; gap:6px; align-items:center; }}
.ask-label {{ border:1px solid rgba(35,167,255,.38); border-radius:999px; color:#fff; background:rgba(117,108,255,.34); text-decoration:none; padding:5px 7px; font:760 10px var(--mono); }}
.brief-row[data-open-url]:hover .open-label {{ border-color:var(--acta2); color:#fff; }}
.jobs-panel {{ margin-top:22px; border-top:1px solid var(--line); scroll-margin-top:112px; }}
.jobs-head {{ display:flex; align-items:flex-end; gap:12px; padding:16px 0 8px; border-bottom:1px solid var(--line-soft); }}
.jobs-head h2 {{ margin:0; font:800 13px var(--mono); letter-spacing:.12em; text-transform:uppercase; color:#fff; }}
.jobs-head span {{ margin-left:auto; color:var(--muted); font:11px var(--mono); text-transform:uppercase; }}
.job-row {{ display:grid; grid-template-columns:42px minmax(0,1.2fr) minmax(120px,.7fr) minmax(180px,.9fr); gap:12px; align-items:center; padding:13px 0; border-bottom:1px solid var(--line-soft); position:relative; overflow:hidden; }}
.job-row[data-open-url] {{ cursor:pointer; }}
.job-row[aria-disabled='true'] {{ cursor:default; }}
.job-row[data-open-url]:hover {{ border-color:rgba(35,167,255,.55); }}
.job-open-overlay {{ z-index:2; }}
.job-row > :not(.row-open-overlay) {{ position:relative; }}
.job-main .thread-link {{ position:relative; z-index:3; pointer-events:auto; }}
.job-rank {{ color:var(--accent); font:800 11px var(--mono); }}
.job-main b {{ display:block; color:#fff; font:700 15px/1.2 var(--ui); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.job-main span, .job-schedule, .job-last {{ color:var(--muted); font:11px var(--mono); min-width:0; }}
.thread-link {{ color:var(--accent); text-decoration:none; border-bottom:1px solid rgba(35,167,255,.55); font-weight:800; }}
.job-schedule, .job-last {{ display:grid; gap:3px; }}
.job-schedule em, .job-last em {{ color:var(--faint); font-style:normal; font-size:9px; letter-spacing:.1em; }}
.job-last time {{ color:#fff; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.job-last small {{ color:var(--accent); font:10px var(--mono); text-transform:uppercase; display:flex; flex-wrap:wrap; gap:4px; align-items:center; line-height:1.25; }}
.job-last small span {{ border:1px solid var(--line-soft); border-radius:999px; padding:2px 5px; background:rgba(255,255,255,.026); }}
.job-last small .confidence-chip {{ color:#fff; border-color:rgba(35,167,255,.34); background:rgba(117,108,255,.16); letter-spacing:.08em; }}
.job-last small .job-open-state.signed {{ color:#fff; border-color:rgba(117,108,255,.46); background:rgba(117,108,255,.22); }}
.job-last small .job-open-state.muted {{ color:var(--muted); }}
.side {{ display:grid; gap:16px; align-content:start; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:24px; overflow:hidden; backdrop-filter:blur(18px) saturate(145%); }}
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
.chip {{ border:1px solid var(--line); border-radius:999px; color:var(--muted); font:10px var(--mono); padding:4px 7px; background:rgba(255,255,255,.045); }}
.date-nav {{ display:flex; gap:8px; flex-wrap:wrap; padding:0 18px 16px; border-bottom:1px solid var(--line); }}
.nav-link {{ color:var(--muted); text-decoration:none; border:1px solid var(--line); border-radius:999px; padding:7px 11px; font:11px var(--mono); background:rgba(255,255,255,.045); }}
.nav-link.primary {{ color:#fff; border-color:rgba(117,108,255,.46); background:rgba(117,108,255,.18); }}
footer {{ color:var(--faint); margin:24px 16px 36px; font:12px var(--mono); text-align:center; }}
.mobilebar {{ display:none; }}
.pull-refresh {{ display:none; position:fixed; left:50%; top:calc(8px + env(safe-area-inset-top, 0px)); transform:translate(-50%,-130%); min-width:150px; padding:9px 12px; border:1px solid var(--line); border-radius:999px; background:rgba(3,6,11,.96); color:var(--acta2); font:800 10px var(--mono); letter-spacing:.12em; text-align:center; z-index:5; opacity:0; transition:transform .18s ease, opacity .18s ease; box-shadow:0 12px 32px rgba(0,0,0,.55); }}
.pull-refresh.ready {{ color:#fff; background:linear-gradient(135deg,var(--acta),var(--acta2)); border-color:transparent; }}
.pull-refresh.visible {{ opacity:1; transform:translate(-50%,0); }}
@media (max-width:980px) {{ .pull-refresh {{ display:block; }} .shell {{ display:block; min-width:0; width:100%; }} .rail {{ display:none; }} .main {{ width:100%; min-width:0; }} .top {{ height:50px; padding:0 max(14px, env(safe-area-inset-left, 0px)) 0 max(14px, env(safe-area-inset-left, 0px)); }} .date-nav {{ position:static; background:rgba(3,6,11,.82); padding:8px 14px; gap:8px; }} .nav-link {{ min-height:38px; display:inline-flex; align-items:center; padding:0 12px; }} .content {{ display:block; padding:12px 14px calc(132px + env(safe-area-inset-bottom, 0px)); }} .panel-title {{ margin-top:12px; }} .side {{ display:none; }} .topstats {{ display:none; }} .lead {{ margin-bottom:8px; touch-action:pan-y; }} .lead p {{ display:-webkit-box; -webkit-line-clamp:1; -webkit-box-orient:vertical; overflow:hidden; }} .meta {{ gap:9px; }} .feed {{ border-top:0; }} .brief-row {{ min-height:60px; touch-action:pan-y; }} .swipe-content {{ grid-template-columns:32px minmax(0,1fr) auto; gap:8px; min-height:60px; padding:7px 10px; touch-action:pan-y; }} .brief-row:hover .swipe-content {{ background:rgba(255,255,255,.05); outline:0; }} .row-signal {{ font-size:8px; }} .row-kicker {{ font-size:10px; }} .brief-copy p {{ display:-webkit-box; -webkit-line-clamp:1; -webkit-box-orient:vertical; overflow:hidden; }} .source-line {{ display:block; font-size:9px; margin-top:3px; }} .open-label {{ display:none; }} .card-actions {{ grid-column:3; justify-self:end; }} .ask-label {{ padding:5px 7px; font-size:10px; }} .jobs-panel {{ margin-top:18px; scroll-margin-top:100px; }} .jobs-head {{ padding-top:14px; }} .job-row {{ grid-template-columns:34px minmax(0,1fr); gap:8px 10px; padding:13px 0; }} .job-schedule, .job-last {{ grid-column:2; }} .job-main b {{ font-size:14px; }} .search {{ max-width:none; }} .mobilebar {{ display:grid; position:fixed; left:max(10px, env(safe-area-inset-left, 0px)); right:max(10px, env(safe-area-inset-right, 0px)); bottom:calc(14px + env(safe-area-inset-bottom, 0px)); min-height:62px; background:linear-gradient(180deg, rgba(7,16,24,.96), rgba(3,6,11,.94)), radial-gradient(circle at 18% 0%, rgba(117,108,255,.28), transparent 42%), radial-gradient(circle at 86% 20%, rgba(35,167,255,.18), transparent 48%); backdrop-filter:blur(18px) saturate(145%); border:1px solid rgba(117,108,255,.28); grid-template-columns:repeat(5,1fr); z-index:3; box-shadow:0 -16px 38px rgba(0,0,0,.62), 0 0 26px rgba(117,108,255,.13); opacity:0; transform:translateY(calc(100% + 24px)); pointer-events:none; transition:opacity .18s ease, transform .22s cubic-bezier(.2,.8,.2,1); }} .mobilebar.visible {{ opacity:1; transform:translateY(0); pointer-events:auto; }} .mobilebar a {{ display:grid; place-items:center; min-height:62px; color:#ddd; text-decoration:none; font:11px var(--mono); touch-action:manipulation; -webkit-tap-highlight-color:rgba(117,108,255,.18); }} .mobilebar a:first-child {{ color:var(--accent); }} }}
@media (max-width:620px) {{ .top {{ gap:8px; }} .ticker {{ font-size:11px; }} .search {{ display:none; }} .lead {{ grid-template-columns:1fr; }} .output-summary {{ grid-column:1; grid-row:auto; justify-self:start; text-align:left; display:flex; align-items:center; gap:6px; min-width:0; }} h1 {{ font-size:19px; max-width:100%; }} .lead p {{ font-size:13px; line-height:1.3; }} .label {{ line-height:1.7; }} .swipe-content {{ grid-template-columns:28px minmax(0,1fr); }} h2 {{ font-size:15px; }} .row-kicker {{ flex-wrap:wrap; overflow:visible; line-height:1.25; }} .row-kicker span, .row-kicker .read-state {{ min-height:auto; display:inline-flex; align-items:center; }} .source-line {{ white-space:normal; overflow:visible; text-overflow:clip; line-height:1.25; word-break:break-word; color:var(--muted); }} .card-actions {{ grid-column:2; justify-self:start; margin-top:2px; }} .brief-copy p {{ font-size:13px; line-height:1.25; }} footer {{ font-size:11px; line-height:1.45; }} }}
</style>
</head>
<body>
<div class="pull-refresh" aria-live="polite">PULL TO REFRESH</div>
<div class="shell">
  <aside class="rail">
    <div class="brand"><div class="logo">A</div><div><b>Acta</b><small>IMPERATR SITUATION ROOM</small></div></div>
    <nav class="nav-side">
      <h4>Today</h4>
      <a class="active" href="/">Today <span>{total}</span></a>
      <a href="/outputs">Outputs <span>{total}</span></a>
      <a href="/runs">Runs <span>{active}</span></a>
      <a href="/jobs">Jobs <span>{len(jobs_rows)}</span></a>
      <a href="/archive">Archive <span>{len(archive_dates)}</span></a>
      <h4>Trace</h4>
      <a href="/runs">Source Runs <span>{active}</span></a>
      <a href="/archive">Audit Trail <span>{len(archive_dates)}</span></a>
    </nav>
    <div class="railfoot"><span class="live"></span>LIVE {html.escape(now.strftime('%H:%M UTC'))}<br>DAY {html.escape(day_label)}</div>
  </aside>
  <main class="main">
    <header class="top">
      <div class="ticker"><em>ACTA</em> / OUTPUTS</div>
      <div class="search">Search briefings, sources, jobs, archive…</div>
      <div class="topstats"><div>VISIBLE <b>{visible}</b></div><div>SILENT <b>{silent}</b></div><div>MISSING <b>{missing}</b></div></div>
    </header>
    <nav class="date-nav"><a class="nav-link primary" href="/">Today</a><a class="nav-link" href="/outputs">Outputs</a><a class="nav-link" href="/runs">Runs</a><a class="nav-link" href="/jobs">Jobs</a><a class="nav-link" href="/archive">Archive</a></nav>
    <section class="content">
      <div>
        <article class="{lead_class}"{lead_read_attr}{lead_href_attr}>
          {lead_open_overlay}
          <div class="label">{lead_label_read_state}<span>{_safe_text(lead_confidence)}</span><span>today</span><span>{lead_open_hint}</span></div>
          <h1>{_safe_text(lead_title)}</h1>
          <p>{_safe_text(lead_excerpt)}</p>
          <div class="output-summary"><b>{visible}/{active}</b><span>fresh</span><span>{missing} gaps</span></div>
          <div class="meta"><span>{html.escape(day_label)}</span><span>{_safe_text(lead_confidence)}</span><span>{silent} silent</span><span>{len(archive_dates)} archive days</span><span>{lead_row_meta}</span>{lead_ask_link}</div>
        </article>
        <div class="panel-title"><b>Output Streams</b><span>daily life separated from dev cycles</span></div>
        {daily_section}
        {dev_section}
        {system_section}
      </div>
      <aside class="side">
        <section class="card"><div class="card-head">Read Order <span>PRIORITIZED</span></div><div class="readorder">{''.join(read_order) or '<p class="prompt">No outputs yet.</p>'}</div></section>
        <section class="card"><div class="card-head">Audit Trail <span>PROVENANCE</span></div><div class="audit">{''.join(audit_rows) or '<p class="prompt">No runs yet.</p>'}</div></section>
        <section class="card"><div class="card-head">Operator Assist <span>NEXT PASS</span></div><div class="assist"><div class="prompt">Open the top signed briefing packet first. Use silent, missing, or no-page rows as source gaps, not as promoted content. Keep raw prompts and local paths out of the human surface.</div><div class="chiprow"><span class="chip">TEXT FIRST</span><span class="chip">NO RAW PROMPTS</span><span class="chip">SIGNED ROWS</span></div></div></section>
      </aside>
    </section>
    <footer>Generated {html.escape(now.isoformat())}. Signed Acta links expire automatically.</footer>
  </main>
</div>
<nav class="mobilebar"><a href="/">TODAY</a><a href="/outputs">OUTPUTS</a><a href="/runs">RUNS</a><a href="/jobs">JOBS</a><a href="/archive">ARCHIVE</a></nav>
<script>{dashboard_script}</script>
</body>
</html>
"""
    style_match = re.search(r"<style>(.*?)</style>", page_html, re.S)
    if not style_match:
        raise RuntimeError("dashboard style block missing")
    dashboard_csp = _inline_style_and_script_csp(style_match.group(1), dashboard_script)
    return page_html.replace(dashboard_csp_placeholder, html.escape(dashboard_csp, quote=False), 1)



def _acta_page_css() -> str:
    return """
:root { color-scheme: dark; --black:#03060b; --bg:#03060b; --bg2:#071018; --panel:rgba(255,255,255,.045); --panel2:rgba(255,255,255,.07); --line:rgba(255,255,255,.105); --line-soft:rgba(255,255,255,.07); --text:#f5f7fb; --body:rgba(245,247,251,.84); --muted:rgba(245,247,251,.62); --faint:rgba(245,247,251,.42); --accent:#756cff; --acta:#756cff; --acta2:#23a7ff; --green:#57a773; --red:#d05a4e; --ui:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif; --read:'Iowan Old Style','Charter','Source Serif Pro',Georgia,serif; --mono:'SFMono-Regular','Roboto Mono','IBM Plex Mono',Consolas,monospace; }
* { box-sizing:border-box; }
html { width:100%; min-width:320px; overflow-x:hidden; background:var(--black); }
body { margin:0; width:100%; min-width:320px; overflow-x:hidden; background:radial-gradient(circle at 18% 8%, rgba(117,108,255,.20), transparent 30%), radial-gradient(circle at 82% 12%, rgba(35,167,255,.12), transparent 28%), linear-gradient(145deg,#020408,#071018 52%,#030509); color:var(--body); font:14px/1.45 var(--ui); -webkit-font-smoothing:antialiased; text-rendering:optimizeLegibility; -webkit-text-size-adjust:100%; }
body:before { content:""; position:fixed; inset:0; pointer-events:none; opacity:.14; background-image:linear-gradient(rgba(255,255,255,.03) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.025) 1px,transparent 1px); background-size:52px 52px; mask-image:radial-gradient(circle at 50% 34%,black,transparent 78%); }
a { color:inherit; }
.top { height:54px; border-bottom:1px solid var(--line); display:flex; align-items:center; gap:12px; padding:0 16px; background:rgba(3,6,11,.82); backdrop-filter:blur(22px) saturate(145%); position:sticky; top:0; z-index:2; }
.ticker { color:#fff; font:800 11px var(--mono); letter-spacing:.11em; text-decoration:none; text-transform:uppercase; white-space:nowrap; }
.ticker em { font-style:normal; color:var(--acta2); text-shadow:0 0 18px rgba(35,167,255,.32); }
.nav { margin-left:auto; display:flex; gap:8px; min-width:0; overflow:auto; scrollbar-width:none; }
.nav::-webkit-scrollbar { display:none; }
.nav a, .quick-nav a, .back { color:rgba(245,247,251,.78); text-decoration:none; border:1px solid var(--line-soft); background:rgba(255,255,255,.035); padding:7px 10px; border-radius:999px; font:11px var(--mono); text-transform:uppercase; letter-spacing:.08em; white-space:nowrap; }
.nav a.active, .nav a:hover, .quick-nav a.active, .quick-nav a:hover, .back:hover { color:#fff; border-color:rgba(117,108,255,.46); background:rgba(117,108,255,.20); box-shadow:inset 0 0 0 1px rgba(35,167,255,.12), 0 0 18px rgba(117,108,255,.12); }
.mobilebar { display:none; }
main { width:min(1180px, calc(100vw - 28px)); margin:0 auto; padding:18px 0 88px; position:relative; }
.kicker { margin:0; color:var(--acta2); font:800 10px var(--mono); text-transform:uppercase; letter-spacing:.13em; }
h1 { margin:6px 0 8px; color:var(--text); font:720 clamp(22px,3.6vw,34px)/1.02 var(--ui); letter-spacing:-.035em; }
.lede { max-width:820px; margin:0 0 14px; color:var(--body); font:14px/1.4 var(--ui); }
.stats, .quick-nav { display:flex; flex-wrap:wrap; gap:7px; margin:12px 0; }
.stat, .archive-card, .job-row, .output-row, .report-shell, .detail-card { border:1px solid var(--line-soft); background:rgba(255,255,255,.032); border-radius:14px; box-shadow:0 12px 34px rgba(0,0,0,.22); }
.stat { padding:6px 8px; color:var(--muted); font:10px var(--mono); text-transform:uppercase; letter-spacing:.06em; }
.stat b { color:#fff; font-size:13px; margin-left:5px; }
.jobs-panel { margin-top:12px; border-top:1px solid var(--line-soft); }
.jobs-head { display:flex; align-items:flex-end; gap:12px; padding:12px 0 7px; border-bottom:1px solid var(--line-soft); }
.jobs-head h2 { margin:0; font:800 12px var(--mono); letter-spacing:.12em; text-transform:uppercase; color:#fff; }
.jobs-head span { margin-left:auto; color:var(--muted); font:10px var(--mono); text-transform:uppercase; }
.job-row { display:grid; grid-template-columns:34px minmax(0,1.2fr) minmax(112px,.64fr) minmax(164px,.86fr); gap:10px; align-items:center; padding:10px 12px; margin:7px 0; position:relative; overflow:hidden; }
.job-row[data-open-url] { cursor:pointer; }
.job-row[aria-disabled='true'] { cursor:default; }
.job-row[data-open-url]:hover { border-color:rgba(35,167,255,.55); box-shadow:0 0 24px rgba(35,167,255,.10); }
.job-open-overlay { position:absolute; inset:0; z-index:2; border:0; text-decoration:none; }
.job-open-overlay:focus-visible { outline:2px solid var(--acta2); outline-offset:2px; border-radius:inherit; }
.job-row > :not(.row-open-overlay) { position:relative; }
.job-main .thread-link { position:relative; z-index:3; pointer-events:auto; }
.job-row:before { content:""; width:2px; height:36px; border-radius:999px; background:rgba(117,108,255,.78); position:absolute; left:0; top:50%; transform:translateY(-50%); z-index:4; pointer-events:none; }
.job-rank { color:var(--acta2); font:800 10px var(--mono); }
.job-main b { display:block; color:#fff; font:700 14px/1.15 var(--ui); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.job-main span, .job-schedule, .job-last { color:var(--muted); font:10px var(--mono); min-width:0; }
.thread-link { color:var(--acta2); text-decoration:none; border-bottom:1px solid rgba(35,167,255,.55); font-weight:800; }
.job-schedule, .job-last { display:grid; gap:3px; }
.job-schedule em, .job-last em { color:var(--faint); font-style:normal; font-size:9px; letter-spacing:.1em; }
.job-last time { color:#fff; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.job-last small { color:var(--acta2); font:10px var(--mono); text-transform:uppercase; display:flex; flex-wrap:wrap; gap:4px; align-items:center; line-height:1.25; }
.job-last small span { border:1px solid var(--line-soft); border-radius:999px; padding:2px 5px; background:rgba(255,255,255,.026); }
.job-last small .confidence-chip { color:#fff; border-color:rgba(35,167,255,.34); background:rgba(117,108,255,.16); letter-spacing:.08em; }
.job-last small .job-open-state.signed { color:#fff; border-color:rgba(117,108,255,.46); background:rgba(117,108,255,.22); }
.job-last small .job-open-state.muted { color:var(--muted); }
.silent .job-rank, .missing .job-rank, .silent .output-rank, .missing .output-rank { color:var(--red); }
.outputs-panel { margin-top:12px; display:flex; flex-direction:column; gap:7px; }
.output-row { display:grid; grid-template-columns:34px minmax(0,1fr) auto; gap:10px; align-items:center; padding:9px 12px; position:relative; overflow:hidden; min-height:62px; }
.output-row[data-open-url] { cursor:pointer; }
.output-row[aria-disabled='true'] { cursor:default; }
.output-row[data-open-url]:hover { border-color:rgba(35,167,255,.55); box-shadow:0 0 24px rgba(35,167,255,.10); }
.output-row.read { opacity:.70; }
.output-row.read .output-main b { color:#c8c8c8; }
.output-row .read-state { color:#fff; }
.output-open-overlay { position:absolute; inset:0; z-index:1; border:0; text-decoration:none; }
.output-actions { position:relative; z-index:2; pointer-events:none; }
.output-actions a, .output-actions button { pointer-events:auto; }
.output-meta .followup-meta { position:relative; z-index:2; }
.output-row:before { content:""; width:2px; height:36px; border-radius:999px; background:linear-gradient(180deg,var(--acta),var(--acta2)); position:absolute; left:0; top:50%; transform:translateY(-50%); z-index:3; }
.output-rank { color:var(--acta2); font:800 10px var(--mono); }
.output-main b { display:block; color:#fff; font:700 15px/1.14 var(--ui); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.output-main p { margin:2px 0 0; color:var(--body); font:13px/1.28 var(--ui); display:-webkit-box; -webkit-line-clamp:1; -webkit-box-orient:vertical; overflow:hidden; }
.catalog-output-row .output-main p { -webkit-line-clamp:2; line-height:1.32; }
.output-meta { display:flex; flex-wrap:wrap; gap:5px; margin-top:6px; color:var(--muted); font:9.5px var(--mono); text-transform:uppercase; }
.output-meta span, .output-meta .followup-meta, .meta span { border:1px solid var(--line-soft); border-radius:999px; padding:3px 6px; background:rgba(255,255,255,.026); }
.output-meta .confidence-chip { color:#fff; border-color:rgba(35,167,255,.34); background:rgba(117,108,255,.16); letter-spacing:.08em; }
.output-meta .followup-meta { color:var(--body); text-decoration:none; border-color:rgba(35,167,255,.38); }
.output-actions { display:flex; gap:6px; align-items:center; }
.output-actions a, .output-actions span, .output-actions button { color:#fff; text-decoration:none; border:1px solid var(--line-soft); border-radius:999px; padding:6px 8px; font:760 10px var(--mono); text-transform:uppercase; white-space:nowrap; }
.output-actions button { appearance:none; background:rgba(255,255,255,.045); cursor:pointer; }
.output-actions button:focus-visible { outline:2px solid var(--acta2); outline-offset:2px; }
.output-actions .open { background:rgba(117,108,255,.22); border-color:rgba(117,108,255,.42); }
.output-actions .ask { background:rgba(117,108,255,.34); border-color:rgba(35,167,255,.38); }
.output-actions .muted { color:var(--muted); background:rgba(255,255,255,.026); }
.prompt { font:14px/1.45 var(--ui); color:var(--body); border-left:2px solid var(--accent); padding-left:10px; }
.grid { display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:10px; }
.archive-card { display:block; text-decoration:none; padding:14px; min-height:86px; position:relative; overflow:hidden; }
.archive-card:before { content:""; position:absolute; inset:0 auto 0 0; width:2px; background:linear-gradient(180deg,var(--acta),var(--acta2)); opacity:.82; }
.archive-card:hover { border-color:rgba(35,167,255,.55); box-shadow:0 0 24px rgba(35,167,255,.10); }
.archive-card span { display:block; color:var(--muted); font:800 10px/1 var(--mono); text-transform:uppercase; letter-spacing:.12em; margin-bottom:8px; }
.archive-card strong { color:var(--text); font:720 20px/1.08 var(--ui); letter-spacing:-.02em; }
.actions { margin-left:auto; display:flex; gap:8px; align-items:center; min-width:0; }
.followup { color:#fff; text-decoration:none; border:1px solid rgba(35,167,255,.38); background:rgba(117,108,255,.34); border-radius:999px; padding:7px 9px; font:800 10px var(--mono); text-transform:uppercase; letter-spacing:.08em; white-space:nowrap; }
.report-shell { padding:14px; overflow:hidden; }
h1.report-title { margin:6px 0 10px; max-width:930px; color:var(--text); font:720 clamp(22px,3.5vw,34px)/1.04 var(--ui); letter-spacing:-.035em; }
.meta { display:flex; flex-wrap:wrap; gap:6px; margin:0 0 12px; color:var(--muted); font:10px var(--mono); text-transform:uppercase; }
.meta b { color:#fff; font-weight:700; }
article.report-body { border-top:1px solid var(--line-soft); padding-top:12px; color:var(--body); font:15px/1.52 var(--ui); }
.report-section { margin:14px 0; padding:14px 0; border-top:1px solid var(--line-soft); }
.section-title { display:flex; gap:9px; align-items:flex-start; margin:0 0 8px; color:#fff; font:720 19px/1.12 var(--ui); letter-spacing:-.015em; }
.section-title:before { content:""; flex:0 0 5px; width:5px; height:18px; margin-top:2px; border-radius:999px; background:linear-gradient(180deg,var(--acta),var(--acta2)); box-shadow:0 0 18px rgba(117,108,255,.28); }
h3 { margin:14px 0 7px; color:#fff; font:700 16px/1.2 var(--ui); }
p { margin:.72em 0; }
ul, ol { margin:.7em 0; padding-left:0; list-style:none; }
li { position:relative; margin:.55em 0; padding-left:21px; }
li:before { content:""; position:absolute; left:4px; top:.72em; width:5px; height:5px; border-radius:50%; background:var(--acta2); }
ol { counter-reset:item; }
ol li { counter-increment:item; padding-left:32px; }
ol li:before { content:counter(item); top:.05em; left:0; width:21px; height:21px; display:grid; place-items:center; color:#fff; background:linear-gradient(135deg,var(--acta),var(--acta2)); font:800 10px/1 var(--mono); }
strong { color:#fff; font-weight:700; }
em { color:#b9dfff; font-style:normal; }
article a { color:#fff; text-decoration:none; border-bottom:1px solid rgba(35,167,255,.6); }
pre { overflow-x:auto; background:rgba(0,0,0,.24); border:1px solid var(--line); padding:14px; border-radius:12px; font-size:13px; }
code { font-family:var(--mono); color:#b9dfff; }
p code, li code { background:rgba(255,255,255,.055); border:1px solid var(--line); padding:1px 5px; border-radius:6px; }
footer { color:var(--faint); margin-top:22px; font:11px var(--mono); text-align:center; }
@media (max-width:760px) { .top { height:50px; padding:0 14px; gap:8px; } .ticker { font-size:11px; } .top .nav { display:none; } .nav { gap:6px; } .nav a { min-height:36px; display:inline-flex; align-items:center; padding:0 10px; } main { width:100%; padding:14px 14px calc(118px + env(safe-area-inset-bottom, 0px)); } h1 { font-size:22px; } .lede { font-size:13px; line-height:1.36; } .grid { grid-template-columns:1fr; } .job-row, .output-row { grid-template-columns:32px minmax(0,1fr); gap:8px 9px; padding:9px 10px; min-height:60px; } .job-schedule, .job-last, .output-actions { grid-column:2; } .job-main b, .output-main b { font-size:14px; } .catalog-output-row .output-main p { -webkit-line-clamp:3; line-height:1.34; } .output-actions { justify-self:start; } .output-meta { flex-wrap:wrap; overflow:visible; } .output-meta .followup-meta { display:inline-flex; } .actions { gap:6px; margin-left:0; overflow:auto; } .followup { max-width:132px; overflow:hidden; text-overflow:ellipsis; padding:7px 9px; } .back { padding:7px 9px; } .report-shell { border-radius:16px; padding:12px; } h1.report-title { font-size:22px; } article.report-body { font-size:14.5px; line-height:1.5; } .section-title { font-size:18px; } .mobilebar { display:grid; position:fixed; left:max(10px, env(safe-area-inset-left, 0px)); right:max(10px, env(safe-area-inset-right, 0px)); bottom:calc(14px + env(safe-area-inset-bottom, 0px)); min-height:62px; background:linear-gradient(180deg, rgba(7,16,24,.96), rgba(3,6,11,.94)), radial-gradient(circle at 18% 0%, rgba(117,108,255,.28), transparent 42%), radial-gradient(circle at 86% 20%, rgba(35,167,255,.18), transparent 48%); backdrop-filter:blur(18px) saturate(145%); border:1px solid rgba(117,108,255,.28); border-radius:0; grid-template-columns:repeat(5,1fr); z-index:3; box-shadow:0 -16px 38px rgba(0,0,0,.62), 0 0 26px rgba(117,108,255,.13); } .mobilebar a { display:grid; place-items:center; min-height:62px; color:#ddd; text-decoration:none; font:11px var(--mono); touch-action:manipulation; -webkit-tap-highlight-color:rgba(117,108,255,.18); } .mobilebar a.active { color:#fff; background:rgba(117,108,255,.18); box-shadow:inset 0 2px 0 var(--acta2); } }
""".strip()


def _acta_nav_links() -> list[tuple[str, str, str]]:
    return [("/", "Today", "today"), ("/outputs", "Outputs", "outputs"), ("/runs", "Runs", "runs"), ("/jobs", "Jobs", "jobs"), ("/archive", "Archive", "archive")]


def _acta_top_nav(active: str, label: str) -> str:
    links = _acta_nav_links()
    nav = "".join(
        f'<a class="active" href="{href}">{text}</a>' if key == active else f'<a href="{href}">{text}</a>'
        for href, text, key in links
    )
    return f'<header class="top"><a class="ticker" href="/"><em>ACTA</em> / {html.escape(label.upper())}</a><nav class="nav">{nav}</nav></header>'


def _acta_mobile_module_nav(active: str | None = None) -> str:
    links = [(href, text.upper(), key) for href, text, key in _acta_nav_links()]
    nav = "".join(
        f'<a class="active" href="{href}">{text}</a>' if key == active else f'<a href="{href}">{text}</a>'
        for href, text, key in links
    )
    return f'<nav class="mobilebar" aria-label="Acta mobile module navigation">{nav}</nav>'


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
<style>{_acta_page_css()}</style>
</head>
<body>
{_acta_top_nav('jobs', 'Jobs')}
<main>
  <p class="kicker">Acta Situation Room · Operations</p>
  <h1>Source runs.</h1>
  <p class="lede">Operational visibility for active relevant Hermes jobs: schedule, delivery route, latest run timestamp, and freshness status.</p>
  <section class="stats"><div class="stat">Relevant <b>{len(jobs_rows)}</b></div><div class="stat">Fresh <b>{fresh}</b></div><div class="stat">Silent <b>{silent}</b></div><div class="stat">Missing <b>{missing}</b></div></section>
  <section class="jobs-panel">
    <div class="jobs-head"><h2>Active Cron Jobs</h2><span>{len(jobs_rows)} relevant</span></div>
    {''.join(jobs_rows) or '<p class="prompt">No active relevant jobs.</p>'}
  </section>
  <footer>Generated {html.escape(now.isoformat())}.</footer>
</main>
{_acta_mobile_module_nav('jobs')}
</body>
</html>
"""


def publish_catalog_output_artifacts(
    outputs: Sequence[ActaOutputItem],
    hermes_home: Path,
    publish_settings: Mapping[str, Any],
) -> None:
    """Publish backing persistent output HTML at stable /outputs/<id>/ URLs.

    Catalog rows intentionally keep stable root-relative hrefs. This helper makes
    those hrefs resolvable during Acta publishing without trusting arbitrary
    catalog paths: it only reads sanitized filenames from the known artifacts
    shelf and uploads them under slug-shaped object keys.
    """
    artifacts_dir = default_outputs_dir(hermes_home).resolve()
    staging_dir = hermes_home / "acta" / "published-outputs"
    staging_dir.mkdir(parents=True, exist_ok=True)
    for item in outputs:
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", item.id):
            continue
        if not item.source_name.endswith(".html") or Path(item.source_name).name != item.source_name:
            continue
        source_path = artifacts_dir / item.source_name
        try:
            resolved = source_path.resolve()
            resolved.relative_to(artifacts_dir)
        except (OSError, ValueError):
            continue
        if not resolved.exists() or resolved.is_symlink() or not resolved.is_file():
            continue
        try:
            wrapped_path = staging_dir / f"{item.id}.html"
            wrapped_path.write_text(
                render_acta_detail_report(
                    _html_artifact_markdown_body(resolved, title=item.title),
                    HtmlReportMetadata(
                        job_id=f"catalog:{item.id}",
                        job_name=item.title,
                        run_time=item.updated_at or item.created_at,
                        source_filename=resolved.name,
                    ),
                ),
                encoding="utf-8",
            )
            publish_html_artifact(
                wrapped_path,
                {"id": "acta-output"},
                {**publish_settings, "object_key": f"public/outputs/{item.id}.html"},
            )
        except (OSError, HtmlArtifactPublishError):
            continue


def render_outputs_page(
    items: Sequence[CronSituationItem],
    generated_at: datetime | None = None,
    feed_preferences: Mapping[str, Any] | None = None,
) -> str:
    now = generated_at or datetime.now(timezone.utc)
    ordered_items = apply_feed_preferences(items, feed_preferences)
    rows: list[str] = []
    signed = 0
    fresh = 0
    silent = 0
    missing = 0
    for index, item in enumerate(ordered_items, start=1):
        artifact_url = item.artifact_url if _is_safe_signed_acta_artifact_url(item.artifact_url) else ""
        telegram_url = item.telegram_url if _is_safe_telegram_url(item.telegram_url) else ""
        status_label = "paused" if not item.enabled else item.status
        if item.status == "fresh":
            fresh += 1
        elif item.status == "silent":
            silent += 1
        elif item.status == "missing":
            missing += 1
        if artifact_url:
            signed += 1
        latest = item.latest_time.isoformat() if item.latest_time else "No run yet"
        age = _age_label(item.latest_time, now)
        confidence = _confidence_label(item, now)
        category = "system" if _is_system_item(item) else "brief"
        source = item.latest_md.name if item.latest_md else (item.latest_html.name if item.latest_html else item.job_id)
        signed_href = html.escape(artifact_url, quote=True) if artifact_url else ""
        row_open_attr = f' data-open-url="{signed_href}"' if signed_href else ' aria-disabled="true"'
        open_overlay = (
            f'<a class="output-open-overlay" href="{signed_href}" aria-label="Open artifact for {html.escape(item.name, quote=True)}"></a>'
            if signed_href
            else ""
        )
        open_action = '<span class="open">SIGNED</span>' if signed_href else '<span class="muted">No signed link</span>'
        ask_action = (
            f'<a class="ask" href="{html.escape(telegram_url, quote=True)}" target="_blank" rel="noopener" '
            'aria-label="Ask follow-up in Telegram">ASK</a>'
            if telegram_url
            else ""
        )
        followup_meta = (
            f'<a class="followup-meta" href="{html.escape(telegram_url, quote=True)}" target="_blank" rel="noopener" '
            'aria-label="Ask follow-up in Telegram" title="Ask follow-up in Telegram">FOLLOW-UP</a>'
            if telegram_url
            else '<span>NO FOLLOW-UP</span>'
        )
        read_key = f"output:{item.job_id}:{latest}"
        read_class = " readable unread" if signed_href else ""
        read_attr = f' data-read-key="{_safe_text(read_key)}"' if signed_href else ""
        read_state_chip = '<span class="read-state">UNREAD</span>' if signed_href else ""
        rows.append(
            f"""
<article class="output-row{read_class} {_status_class(item)}"{read_attr}{row_open_attr}>
  {open_overlay}
  <div class="output-rank">{index:02d}</div>
  <div class="output-main">
    <b>{_safe_text(item.name)}</b>
    <p>{_safe_text(item.excerpt or "No visible response was produced for this run.")}</p>
    <div class="output-meta">{read_state_chip}<span class="confidence-chip">{_safe_text(confidence)}</span><span>{_safe_text(status_label)}</span><span>{_safe_text(category)}</span><span>{_safe_text(age)}</span><span>SCHEDULE {_safe_text(item.schedule or "manual")}</span><span>SOURCE {_safe_text(source)}</span><span>JOB {_safe_text(item.job_id)}</span><span>{_safe_text(item.deliver or "local")}</span><span>{_safe_text(latest)}</span>{followup_meta}</div>
  </div>
  <div class="output-actions">{open_action}{ask_action}</div>
</article>"""
        )
    read_state_script = _outputs_read_state_script()
    outputs_csp = _inline_script_csp(read_state_script)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, viewport-fit=cover, user-scalable=no">
<meta http-equiv="Content-Security-Policy" content="{html.escape(outputs_csp, quote=False)}">
<title>Acta Outputs</title>
<style>{_acta_page_css()}</style>
</head>
<body>
{_acta_top_nav('outputs', 'Outputs')}
<main>
  <p class="kicker">Acta Situation Room · Outputs</p>
  <h1>Signed source objects.</h1>
  <p class="lede">Compact source-backed output rows with artifact links, Telegram follow-up, freshness, category, and provenance in the Acta Imperatr surface.</p>
  <section class="stats"><div class="stat">Outputs <b>{len(rows)}</b></div><div class="stat">Signed <b>{signed}</b></div><div class="stat">Unread <b data-unread-count="{signed}">{signed}</b></div><div class="stat">Fresh <b>{fresh}</b></div><div class="stat">Silent <b>{silent}</b></div><div class="stat">Missing <b>{missing}</b></div></section>
  <section class="outputs-panel">
    {''.join(rows) or '<p class="prompt">No source outputs yet.</p>'}
  </section>
  <footer>Generated {html.escape(now.isoformat())}. Signed Acta links expire automatically.</footer>
</main>
{_acta_mobile_module_nav('outputs')}
<script>{read_state_script}</script>
</body>
</html>
"""


def render_catalog_outputs_page(
    outputs: Sequence[ActaOutputItem],
    generated_at: datetime | None = None,
    local_artifact_base: str | None = None,
) -> str:
    now = generated_at or datetime.now(timezone.utc)
    rows: list[str] = []
    pinned = sum(1 for item in outputs if item.pinned)
    unread = sum(1 for item in outputs if not item.read and _is_safe_catalog_href(item.href))
    for index, item in enumerate(outputs, start=1):
        href = item.href if _is_safe_catalog_href(item.href) else ""
        if (
            href
            and local_artifact_base
            and Path(item.source_name).name == item.source_name
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*\.html", item.source_name)
        ):
            href = f"{local_artifact_base.rstrip('/')}/{item.source_name}"
        escaped_href = html.escape(href, quote=True) if href else ""
        updated = _parse_iso_datetime(item.updated_at)
        age = _age_label(updated, now) if updated else "catalog"
        tags = " ".join(f"<span>#{_safe_text(tag)}</span>" for tag in item.tags[:6])
        open_overlay = (
            f'<a class="output-open-overlay" href="{escaped_href}" aria-label="Open output: {html.escape(item.title, quote=True)}"></a>'
            if escaped_href
            else ""
        )
        row_open_attr = f' data-open-url="{escaped_href}"' if escaped_href else ' aria-disabled="true"'
        read_class = f" readable{' read' if item.read else ' unread'}" if escaped_href else ""
        read_key_attr = f' data-read-key="{html.escape(f"output:{item.id}", quote=True)}"' if escaped_href else ""
        read_initial_attr = f' data-read-initial="{str(bool(item.read)).lower()}"' if escaped_href else ""
        read_title_attr = f' data-read-title="{html.escape(item.title, quote=True)}"' if escaped_href else ""
        read_state_chip = f"<span class=\"read-state\">{'READ' if item.read else 'UNREAD'}</span>" if escaped_href else ""
        open_action = '<span class="open">OPEN</span>' if escaped_href else '<span class="muted">No public link</span>'
        read_toggle = (
            f'<button class="read-toggle" type="button" aria-label="Mark output {"unread" if item.read else "read"}: {html.escape(item.title, quote=True)}">'
            f'{"Mark unread" if item.read else "Mark read"}</button>'
            if escaped_href
            else ""
        )
        rows.append(
            f"""
<article class="output-row catalog-output-row{read_class} fresh"{read_key_attr}{read_initial_attr}{read_title_attr}{row_open_attr}>
  {open_overlay}
  <div class="output-rank">{index:02d}</div>
  <div class="output-main">
    <b>{_safe_text(item.title)}</b>
    <p>{_safe_text(item.summary or "Persistent Acta output.")}</p>
    <div class="output-meta">{read_state_chip}<span class="confidence-chip">CATALOG</span><span>{'PINNED' if item.pinned else 'OUTPUT'}</span><span>{_safe_text(age)}</span><span>SOURCE {_safe_text(item.source_name)}</span><span>ID {_safe_text(item.id)}</span>{tags}</div>
  </div>
  <div class="output-actions">{read_toggle}{open_action}</div>
</article>"""
        )
    read_state_script = _outputs_read_state_script()
    outputs_csp = _inline_script_csp(read_state_script)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, viewport-fit=cover, user-scalable=no">
<meta http-equiv="Content-Security-Policy" content="{html.escape(outputs_csp, quote=False)}">
<title>Acta Outputs</title>
<style>{_acta_page_css()}</style>
</head>
<body>
{_acta_top_nav('outputs', 'Outputs')}
<main>
  <p class="kicker">Acta Situation Room · Outputs</p>
  <h1>Persistent catalog.</h1>
  <p class="lede">Durable Acta outputs imported from the persistent catalog, separate from run history.</p>
  <nav class="quick-nav"><a href="/">Today</a><a class="active" href="/outputs">Outputs</a><a href="/runs">Runs</a><a href="/jobs">Jobs</a><a href="/archive">Archive</a></nav>
  <section class="stats"><div class="stat">Outputs <b>{len(rows)}</b></div><div class="stat">Pinned <b>{pinned}</b></div><div class="stat">Unread <b data-unread-count="{unread}">{unread}</b></div></section>
  <section class="outputs-panel">
    {''.join(rows) or '<p class="prompt">No persistent Acta outputs yet.</p>'}
  </section>
  <footer>Generated {html.escape(now.isoformat())}. Catalog rows expose only public-safe output hrefs.</footer>
</main>
{_acta_mobile_module_nav('outputs')}
<script>{read_state_script}</script>
</body>
</html>
"""


def render_runs_page(runs: Sequence[ActaRunItem], generated_at: datetime | None = None) -> str:
    now = generated_at or datetime.now(timezone.utc)
    rows: list[str] = []
    fresh = sum(1 for item in runs if item.status == "fresh")
    silent = sum(1 for item in runs if item.status == "silent")
    signed = sum(1 for item in runs if _is_safe_signed_acta_artifact_url(item.artifact_url))
    for index, item in enumerate(runs, start=1):
        run_time = item.run_time.isoformat() if item.run_time else "unknown"
        age = _age_label(item.run_time, now) if item.run_time else "unknown"
        confidence = _confidence_bucket(item.enabled, item.status, item.run_time, now)
        telegram_url = item.telegram_url if _is_safe_telegram_url(item.telegram_url) else ""
        followup = (
            f'<a class="followup-meta" href="{html.escape(telegram_url, quote=True)}" target="_blank" rel="noopener" aria-label="Ask follow-up in Telegram">FOLLOW-UP</a>'
            if telegram_url
            else '<span>NO FOLLOW-UP</span>'
        )
        kind = "+".join(part for part, present in (("MD", item.has_markdown), ("HTML", item.has_html)) if present) or "OUTPUT"
        artifact_url = item.artifact_url if _is_safe_signed_acta_artifact_url(item.artifact_url) else ""
        signed_href = html.escape(artifact_url, quote=True) if artifact_url else ""
        row_open_attr = f' data-open-url="{signed_href}"' if signed_href else ' aria-disabled="true"'
        open_overlay = (
            f'<a class="output-open-overlay" href="{signed_href}" aria-label="Open run history for {html.escape(item.name, quote=True)}"></a>'
            if signed_href
            else ""
        )
        read_key = f"run:{item.job_id}:{item.run_id}"
        read_class = " readable unread" if signed_href else ""
        read_key_attr = f' data-read-key="{html.escape(read_key, quote=True)}"' if signed_href else ""
        read_title_attr = f' data-read-title="{html.escape(item.name, quote=True)}"' if signed_href else ""
        read_state_chip = '<span class="read-state">UNREAD</span>' if signed_href else ""
        open_action = '<span class="open">SIGNED</span>' if signed_href else '<span class="muted">HISTORY</span>'
        rows.append(
            f"""
<article class="output-row{read_class} {_safe_text(item.status)}"{read_key_attr}{read_title_attr}{row_open_attr}>
  {open_overlay}
  <div class="output-rank">{index:02d}</div>
  <div class="output-main">
    <b>{_safe_text(item.name)}</b>
    <p>{_safe_text(item.excerpt)}</p>
    <div class="output-meta">{read_state_chip}<span class="confidence-chip">{_safe_text(confidence)}</span><span>{_safe_text(item.status)}</span><span>{_safe_text(kind)}</span><span>{_safe_text(age)}</span><span>RUN {_safe_text(run_time)}</span><span>SOURCE {_safe_text(Path(item.source_name).name)}</span><span>JOB {_safe_text(item.job_id)}</span><span>SCHEDULE {_safe_text(item.schedule or 'manual')}</span><span>{_safe_text(item.deliver or 'local')}</span>{followup}</div>
  </div>
  <div class="output-actions">{open_action}</div>
</article>"""
        )
    read_state_script = _outputs_read_state_script()
    runs_csp = _inline_script_csp(read_state_script)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, viewport-fit=cover, user-scalable=no">
<meta http-equiv="Content-Security-Policy" content="{html.escape(runs_csp, quote=False)}">
<title>Acta Runs</title>
<style>{_acta_page_css()}</style>
</head>
<body>
{_acta_top_nav('runs', 'Runs')}
<main>
  <p class="kicker">Acta Situation Room · Runs</p>
  <h1>Run history.</h1>
  <p class="lede">Chronological cron output history scanned from run files across jobs. Local filesystem paths are not exposed.</p>
  <nav class="quick-nav"><a href="/">Today</a><a href="/outputs">Outputs</a><a class="active" href="/runs">Runs</a><a href="/jobs">Jobs</a><a href="/archive">Archive</a></nav>
  <section class="stats"><div class="stat">Runs <b>{len(rows)}</b></div><div class="stat">Unread <b data-unread-count="{signed}">{signed}</b></div><div class="stat">Fresh <b>{fresh}</b></div><div class="stat">Silent <b>{silent}</b></div></section>
  <section class="outputs-panel">
    {''.join(rows) or '<p class="prompt">No cron run history yet.</p>'}
  </section>
  <footer>Generated {html.escape(now.isoformat())}.</footer>
</main>
{_acta_mobile_module_nav('runs')}
<script>{read_state_script}</script>
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
<style>{_acta_page_css()}</style>
</head>
<body>
{_acta_top_nav('archive', 'Archive')}
<main>
<p class="kicker">Acta · Archive</p>
<h1>Previous days.</h1>
<p class="lede">Browse prior Situation Room snapshots by day. Historical pages now use the same Imperatr surface as the live feed.</p>
<nav class="quick-nav"><a href="/">Today</a><a href="/outputs">Outputs</a><a href="/runs">Runs</a><a href="/jobs">Jobs</a><a class="active" href="/archive">Archive</a></nav>
<section class="grid">{cards}</section>
<footer>Generated {html.escape(now.isoformat())}.</footer>
</main>
{_acta_mobile_module_nav('archive')}
</body>
</html>
"""


def render_acta_detail_report(
    body: str,
    metadata: HtmlReportMetadata | Mapping[str, str],
    telegram_url: str | None = None,
    detail_signals: Mapping[str, str] | None = None,
) -> str:
    """Render a standalone Acta detail page using the Imperatr Acta visual system."""
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
    for label, value in (detail_signals or {}).items():
        clean_label = re.sub(r"[^A-Za-z0-9 _/-]+", "", str(label)).strip().upper()
        clean_value = str(value or "").strip()
        if clean_label and clean_value:
            footer_bits.append(f"<span><b>{html.escape(clean_label)}</b> {html.escape(clean_value)}</span>")
    followup_link = ""
    safe_telegram_url = telegram_url if _is_safe_telegram_url(telegram_url) else ""
    if safe_telegram_url:
        followup_link = (
            f'<a class="followup" href="{html.escape(safe_telegram_url, quote=True)}" '
            'target="_blank" rel="noopener" aria-label="Ask follow-up in Telegram" title="Ask follow-up in Telegram">Ask</a>'
        )
    actions = f'<div class="actions">{followup_link}<a class="back" href="/outputs">Outputs</a><a class="back" href="/">Back</a></div>'
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, viewport-fit=cover, user-scalable=no">
<meta http-equiv="Content-Security-Policy" content="{html.escape(CSP, quote=False)}">
<title>{html.escape(title)} · Acta Situation Room</title>
<style>{_acta_page_css()}</style>
</head>
<body>
<header class="top"><a class="ticker" href="/"><em>ACTA</em> / BRIEF</a>{actions}</header>
<main>
  <section class="report-shell">
    <p class="kicker">Acta Situation Room · Source-backed drill-in</p>
    <h1 class="report-title">{html.escape(title)}</h1>
    <div class="meta">{''.join(footer_bits)}</div>
    <article class="report-body">{rendered}</article>
  </section>
  <footer>Signed Acta detail. Same Imperatr app shell across every drill-in.</footer>
</main>
{_acta_mobile_module_nav('outputs')}
</body>
</html>
"""


def _cron_detail_signals(item: CronSituationItem, now: datetime | None = None) -> dict[str, str]:
    """Return source-derived operator signals for signed detail pages."""
    current = now or datetime.now(timezone.utc)
    return {
        "signed status": "paused" if not item.enabled else item.status,
        "signed conf": _confidence_label(item, current),
        "signed age": _age_label(item.latest_time, current),
    }


def _run_detail_signals(item: ActaRunItem, now: datetime | None = None) -> dict[str, str]:
    """Return source-derived operator signals for run-history detail pages."""
    current = now or datetime.now(timezone.utc)
    return {
        "signed status": "paused" if not item.enabled else item.status,
        "signed conf": _confidence_bucket(item.enabled, item.status, item.run_time, current),
        "signed age": _age_label(item.run_time, current),
    }


def _detail_body(item: CronSituationItem) -> str:
    if not item.latest_md:
        return f"# {item.name}\n\nNo Markdown output exists yet."
    text = item.latest_md.read_text(encoding="utf-8", errors="replace")
    response = _extract_response_if_present(text) or ""
    if not response or response.strip() == "[SILENT]":
        response = "No visible response was produced for this run."
    return response


def _html_artifact_markdown_body(path: Path, *, title: str) -> str:
    """Extract useful text from an HTML artifact for the current Acta shell.

    Historical catalog artifacts are often full standalone HTML pages with their
    own amber/terminal skin. They are source material, not the UI contract. Acta
    republishes the visible content inside the current Imperatr detail shell so a
    persistent output cannot resurrect the old design on click-through.
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return f"# {title}\n\nThe HTML artifact could not be read."
    body_match = re.search(r"<body[^>]*>(.*?)</body>", raw, flags=re.IGNORECASE | re.DOTALL)
    visible = body_match.group(1) if body_match else raw
    visible = re.sub(r"<script\b[^>]*>.*?</script>", "\n", visible, flags=re.IGNORECASE | re.DOTALL)
    visible = re.sub(r"<style\b[^>]*>.*?</style>", "\n", visible, flags=re.IGNORECASE | re.DOTALL)
    visible = re.sub(r"<(?:br|/p|/div|/section|/article|/li|/h[1-6])\b[^>]*>", "\n", visible, flags=re.IGNORECASE)
    visible = re.sub(r"<[^>]+>", " ", visible)
    visible = html.unescape(visible)
    legacy_markers = ("#f5a400", "--amber", "bloomberg", "palantir", "generated files", "generated-file")
    lines = []
    for line in visible.splitlines():
        cleaned = re.sub(r"\s+", " ", line).strip()
        if not cleaned:
            continue
        if any(marker in cleaned.lower() for marker in legacy_markers):
            continue
        lines.append(cleaned)
    summary = "\n\n".join(lines[:80]).strip()
    if not summary:
        summary = "HTML artifact normalized into the current Acta detail shell."
    return f"# {title}\n\n{summary}"


def _html_detail_body(item: CronSituationItem) -> str:
    """Extract useful text from an HTML-only artifact for the current v9 detail shell.

    Markdown artifacts are the normal Acta path and render directly through
    ``_detail_body``. Some historical cron runs only have an HTML artifact,
    though; publishing that file directly can resurrect archived amber/terminal
    styling on click-through. This fallback keeps the signed detail route in the
    current Imperatr shell by carrying over visible text while dropping legacy
    presentation markup and design-contract residue.
    """
    if not item.latest_html:
        return f"# {item.name}\n\nNo output artifact exists yet."
    return _html_artifact_markdown_body(item.latest_html, title=item.name)


def _run_detail_body(item: ActaRunItem) -> str:
    if item.md_path is not None:
        try:
            text = item.md_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        response = _extract_response_if_present(text) or ""
        if not response or response.strip() == "[SILENT]":
            return "No visible response was produced for this run."
        return response
    if item.html_path is not None:
        return _html_artifact_markdown_body(item.html_path, title=item.name)
    return "No visible response was produced for this run."


def attach_run_artifact_urls(
    runs: Sequence[ActaRunItem],
    publish_settings: Mapping[str, Any],
    output_dir: Path,
) -> list[ActaRunItem]:
    output_dir.mkdir(parents=True, exist_ok=True)
    linked: list[ActaRunItem] = []
    for item in runs:
        url: str | None = None
        source_path = item.md_path or item.html_path
        if source_path is not None:
            try:
                safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{item.job_id}-{item.run_id}").strip(".-_") or "run"
                temp_html = output_dir / f"{safe_stem}.html"
                detail_html = render_acta_detail_report(
                    _run_detail_body(item),
                    HtmlReportMetadata(
                        job_id=item.job_id,
                        job_name=item.name,
                        run_time=item.run_time.isoformat() if item.run_time else "",
                        source_filename=Path(item.source_name).name,
                    ),
                    telegram_url=item.telegram_url,
                    detail_signals=_run_detail_signals(item),
                )
                temp_html.write_text(detail_html, encoding="utf-8")
                url = publish_html_artifact(
                    temp_html,
                    {"id": item.job_id},
                    {**publish_settings, "object_key": f"r/run-details/{safe_stem}.html"},
                )
            except (OSError, HtmlArtifactPublishError):
                url = None
        linked.append(
            ActaRunItem(
                job_id=item.job_id,
                name=item.name,
                schedule=item.schedule,
                deliver=item.deliver,
                enabled=item.enabled,
                run_id=item.run_id,
                run_time=item.run_time,
                status=item.status,
                excerpt=item.excerpt,
                source_name=Path(item.source_name).name,
                has_markdown=item.has_markdown,
                has_html=item.has_html,
                telegram_url=item.telegram_url,
                artifact_url=url,
                md_path=item.md_path,
                html_path=item.html_path,
            )
        )
    return linked


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
                detail_signals=_cron_detail_signals(item),
            )
            temp_html.write_text(detail_html, encoding="utf-8")
            source_html = temp_html
        elif item.latest_html is not None:
            # HTML-only outputs may carry historical CSS/copy. Wrap their
            # visible content in the current Acta v9 shell instead of publishing
            # raw generated-file UI as the signed click-through target.
            temp_html = output_dir / f"{item.job_id}-{item.latest_html.stem}.html"
            detail_html = render_acta_detail_report(
                _html_detail_body(item),
                HtmlReportMetadata(
                    job_id=item.job_id,
                    job_name=item.name,
                    run_time=item.latest_time.isoformat() if item.latest_time else "",
                    source_filename=item.latest_html.name,
                ),
                telegram_url=item.telegram_url,
                detail_signals=_cron_detail_signals(item),
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
        outputs_path = output_dir / "outputs.html"
        catalog_outputs = collect_catalog_outputs(home)
        publish_catalog_output_artifacts(catalog_outputs, home, publish_settings)
        local_artifact_base = os.path.relpath(default_outputs_dir(home), output_dir)
        outputs_path.write_text(
            render_catalog_outputs_page(
                catalog_outputs,
                generated_at=generated_at,
                local_artifact_base=local_artifact_base,
            ),
            encoding="utf-8",
        )
        publish_outputs_path = output_dir / "outputs.publish.html"
        publish_outputs_path.write_text(render_catalog_outputs_page(catalog_outputs, generated_at=generated_at), encoding="utf-8")
        publish_html_artifact(
            publish_outputs_path,
            {"id": "acta-situation-room"},
            {**publish_settings, "object_key": "public/outputs/index.html"},
        )
        runs_path = output_dir / "runs.html"
        runs = collect_run_history(home)
        runs = attach_run_artifact_urls(runs, publish_settings, output_dir / "run-details")
        runs_path.write_text(render_runs_page(runs, generated_at=generated_at), encoding="utf-8")
        publish_html_artifact(
            runs_path,
            {"id": "acta-situation-room"},
            {**publish_settings, "object_key": "public/runs/index.html"},
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
