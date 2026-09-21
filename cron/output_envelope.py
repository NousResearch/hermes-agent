"""Delivery-layer output envelope for cron jobs.

The de-noise standard (plans/2026-06-20-discord-output-standard.md) defines one
shape for every proactive Discord message: a severity tag, a title, a 2-3 line
summary, and all detail pushed into an attachment. The first build wired this
into the agent-job path via the ``cron-output-contract`` skill, but skills only
constrain agent reasoning. Script-mode crons (worker-failure-analysis,
governance-crossref, the denji-profile-review trio, hermaguard-gate, ...) emit
raw stdout that is delivered verbatim, so most of the #governance and #content
noise was never governed.

This module governs ALL cron output at the single delivery choke point
(`cron/scheduler.py::_process_job`), regardless of agent vs script:

- 🟢 LOG  -> suppressed live, appended to the briefing log only.
- 🟡 FYI  -> posts to its channel, no ping, enveloped + attachment-offloaded.
- 🔴 ACT  -> posts with @-mention, keeps one inline action line.
- repeats -> deduped on a stable key: silent for 12h, then one re-post.

Severity is read from a leading marker (🔴/🟡/🟢 or [SEV:ACT|FYI|LOG]) the job
emits, else a per-job ``severity`` config field, else defaulted (failed run ->
ACT, otherwise FYI). It is intentionally conservative: short clean output is
passed through almost untouched; only walls of text get offloaded to an
attachment so the channel body stays at title + summary.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

HERMES_HOME = Path("/home/kensei/.hermes")
DETAIL_DIR = HERMES_HOME / "governance" / "cron-detail"
BRIEFING_LOG_DIR = HERMES_HOME / "governance" / "briefing-log"
DEDUP_FILE = HERMES_HOME / "state" / "cron-envelope-dedup.json"
TZ = ZoneInfo("Europe/London")

# How long a repeat of the same summary stays silent before one re-post.
DEDUP_WINDOW = dt.timedelta(hours=12)

# Body-size thresholds above which detail is offloaded to an attachment so the
# channel only ever carries title + summary.
MAX_BODY_LINES = 8
MAX_BODY_CHARS = 700
# Link-bearing output (deal scans, job listings) keeps its links CLICKABLE in the
# channel rather than being buried in an attachment, up to Discord's message
# limit (2000 chars; we leave headroom for the title and severity prefix).
DISCORD_SAFE_LIMIT = 1850
# Excludes quote/angle chars so a crafted URL can't break out of an HTML attribute.
_URL_RE = re.compile(r"https?://[^\s\"'<>]+")
# Summary shown in-channel when content is offloaded.
SUMMARY_MAX_LINES = 3
SUMMARY_MAX_CHARS = 320
# Hard cap on the offloaded attachment so a runaway job cannot exhaust disk.
MAX_DETAIL_BYTES = 512 * 1024
# Delete offloaded detail files older than this on each write (cheap rolling GC).
DETAIL_RETENTION = dt.timedelta(days=14)

# Serialises the read-modify-write of the dedup state file; cron jobs run in a
# parallel thread pool, so without this two simultaneous runs can both miss a
# key and double-post.
_DEDUP_LOCK = threading.Lock()

# Discord mass/role/user mention patterns. Cron stdout is untrusted (script jobs,
# third-party tools), so these are neutralised before any text reaches a channel
# to stop a job triggering an @everyone-style ping.
_MENTION_RE = re.compile(r"@everyone|@here|<@[!&]?\d+>", re.IGNORECASE)

ACT, FYI, LOG = "ACT", "FYI", "LOG"
_EMOJI = {ACT: "🔴", FYI: "🟡", LOG: "🟢"}
_MARKER_TO_SEV = {"🔴": ACT, "🟡": FYI, "🟢": LOG}
_TAG_RE = re.compile(r"\[SEV:(ACT|FYI|LOG)\]", re.IGNORECASE)
_SILENT_RE = re.compile(r"\[SILENT\]", re.IGNORECASE)
# Matches dates already present in a title so we don't append a redundant one.
_DATE_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:19|20)\d{2}\b|\b\d{1,2} \w{3}\b")


def _strip_mentions(text: str) -> str:
    """Neutralise Discord pings embedded in untrusted job output."""
    return _MENTION_RE.sub(lambda m: m.group(0).replace("@", "@​").replace("<", "(").replace(">", ")"), text)


@dataclass
class EnvelopeDecision:
    suppress: bool
    text: Optional[str] = None
    severity: Optional[str] = None
    reason: str = ""


def _now() -> dt.datetime:
    return dt.datetime.now(TZ)


def detect_severity(content: str, job: dict, success: bool) -> tuple[str, str]:
    """Return (severity, content_without_marker)."""
    stripped = content.lstrip()

    for marker, sev in _MARKER_TO_SEV.items():
        if stripped.startswith(marker):
            return sev, stripped[len(marker):].lstrip()

    tag = _TAG_RE.search(content)
    if tag:
        return tag.group(1).upper(), _TAG_RE.sub("", content, count=1).strip()

    configured = str(job.get("severity") or "").strip().upper()
    if configured in (ACT, FYI, LOG):
        return configured, content

    # No explicit signal: a failed run needs eyes, success is informational.
    return (ACT if not success else FYI), content


def _nonempty_lines(text: str) -> list[str]:
    return [ln for ln in (l.strip() for l in text.splitlines()) if ln]


def _job_title(job: dict) -> str:
    name = job.get("name") or job.get("id") or "Cron job"
    return str(name).strip()


def _build_title(content: str, job: dict) -> str:
    """Return the plain title text (no leading emoji)."""
    lines = _nonempty_lines(content)
    first = lines[0] if lines else _job_title(job)
    # A short, heading-like first line becomes the title; a long sentence does
    # not, so fall back to the job name to keep the title scannable.
    if len(first) > 80 or first.endswith((".", "!", "?")):
        first = _job_title(job)
    # Don't append a date if the chosen title already carries one (e.g. a job
    # that prints its own "· 22/06/2026 10:18:00" header).
    if _DATE_RE.search(first):
        return first
    date = _now().strftime("%-d %b")
    return f"{first} · {date}"


def _build_summary(content: str, title_line: str) -> str:
    lines = _nonempty_lines(content)
    # Drop a leading line that the title already consumed.
    if lines and (lines[0] in title_line or title_line.endswith(lines[0])):
        lines = lines[1:]
    summary_lines: list[str] = []
    seen: set[str] = set()
    total = 0
    for ln in lines:
        key = re.sub(r"\d+", "#", ln.strip())  # collapse near-identical repeats
        if key in seen:
            continue
        seen.add(key)
        if len(summary_lines) >= SUMMARY_MAX_LINES:
            break
        if total + len(ln) > SUMMARY_MAX_CHARS and summary_lines:
            break
        summary_lines.append(ln)
        total += len(ln)
    return "\n".join(summary_lines).strip()


def _action_line(content: str) -> Optional[str]:
    for ln in _nonempty_lines(content):
        if ln.lower().lstrip().startswith(("reply:", "reply ", "action:", "decision:")):
            return ln.strip()
    return None


def _gc_detail() -> None:
    """Drop offloaded detail files past the retention window."""
    cutoff = (_now() - DETAIL_RETENTION).timestamp()
    for old in list(DETAIL_DIR.glob("*.html")) + list(DETAIL_DIR.glob("*.md")):
        try:
            if old.stat().st_mtime < cutoff:
                old.unlink()
        except OSError:
            pass


def _html_escape(text: str) -> str:
    import html as _html
    return _html.escape(str(text), quote=True)


def _md_inline(s: str) -> str:
    """Escape, then render inline markdown (code, bold, italic, links)."""
    import html as _html
    s = _html.escape(s, quote=True)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\*\w])\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", s)
    # URLs are already escaped (no raw quotes/angles can survive), so a bounded
    # match is safe to wrap in an anchor.
    s = re.sub(r"(https?://[^\s<]+)", r'<a href="\1">\1</a>', s)
    return s


def _render_markdown(text: str) -> str:
    """Render the common markdown a cron emits (headings, bold, bullets, code,
    links) to safe HTML so the fallback attachment reads cleanly instead of
    showing raw ``**`` / ``-`` / backticks."""
    out: list[str] = []
    in_list = False
    for raw in text.splitlines():
        ln = raw.rstrip()
        if not ln.strip():
            if in_list:
                out.append("</ul>"); in_list = False
            continue
        h = re.match(r"\s*(#{1,6})\s+(.*)", ln)
        if h:
            if in_list:
                out.append("</ul>"); in_list = False
            lvl = min(len(h.group(1)) + 1, 4)
            out.append(f"<h{lvl}>{_md_inline(h.group(2))}</h{lvl}>")
            continue
        b = re.match(r"\s*[-*•]\s+(.*)", ln)
        if b:
            if not in_list:
                out.append("<ul>"); in_list = True
            out.append(f"<li>{_md_inline(b.group(1))}</li>")
            continue
        if in_list:
            out.append("</ul>"); in_list = False
        out.append(f"<p>{_md_inline(ln)}</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


def _linkify_html(text: str) -> str:
    """Escape HTML then turn bare URLs into clickable anchors."""
    import html as _html

    def _repl(m: re.Match) -> str:
        raw = m.group(0)
        # Trailing punctuation shouldn't be swallowed into the href.
        trail = ""
        while raw and raw[-1] in ").,;]":
            trail = raw[-1] + trail
            raw = raw[:-1]
        # Escape the URL for the attribute/text context — cron stdout is untrusted.
        safe = _html.escape(raw, quote=True)
        return f'<a href="{safe}">{safe}</a>{trail}'

    out, last = [], 0
    for m in _URL_RE.finditer(text):
        out.append(_html.escape(text[last:m.start()]))
        out.append(_repl(m))
        last = m.end()
    out.append(_html.escape(text[last:]))
    return "".join(out)


def _write_detail(job: dict, severity: str, content: str) -> Path:
    """Write the offloaded detail as clickable HTML (links stay usable)."""
    DETAIL_DIR.mkdir(parents=True, exist_ok=True)
    _gc_detail()
    now = _now()
    name = _job_title(job)
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-").lower() or "cron"
    path = (DETAIL_DIR / f"{safe}-{now.strftime('%Y%m%d-%H%M%S')}.html").resolve()
    # Defence in depth: never write outside DETAIL_DIR even if the slug or a
    # symlinked dir tries to escape.
    if path.parent != DETAIL_DIR.resolve():
        path = DETAIL_DIR.resolve() / f"cron-{now.strftime('%Y%m%d-%H%M%S')}.html"
    body = content.strip()
    if len(body.encode("utf-8")) > MAX_DETAIL_BYTES:
        body = body.encode("utf-8")[:MAX_DETAIL_BYTES].decode("utf-8", "ignore") + "\n\n[truncated]"
    # Dark KENSEI theme — the agreed cron-output-contract palette
    # (#11100f background, #fbbf24 accent, #f5f5f4 text). Keep it consistent with
    # the HTML agents generate themselves so attachments look uniform.
    doc = (
        "<!doctype html><html lang=en><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>{_html_escape(name)}</title>"
        "<style>"
        ":root{color-scheme:dark}"
        "body{font:15px/1.6 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;"
        "background:#11100f;color:#f5f5f4;max-width:780px;margin:0 auto;padding:2.2rem 1.2rem}"
        ".hd{border-bottom:1px solid #2a2826;padding-bottom:.9rem;margin-bottom:1.3rem}"
        "h1{font-size:1.25rem;margin:0;color:#fbbf24;font-weight:650;letter-spacing:.2px}"
        ".meta{color:#8a847c;font-size:.82rem;margin-top:.35rem}"
        ".body{background:#1a1816;border:1px solid #2a2826;border-radius:10px;padding:.4rem 1.3rem 1rem}"
        ".body h2{color:#fbbf24;font-size:1.05rem;margin:1.3rem 0 .5rem;font-weight:600}"
        ".body h3{color:#e7b53a;font-size:.95rem;margin:1.1rem 0 .4rem;font-weight:600}"
        ".body h4{color:#caa84f;font-size:.9rem;margin:1rem 0 .3rem}"
        ".body p{margin:.5rem 0}.body ul{margin:.4rem 0 .7rem;padding-left:1.25rem}.body li{margin:.25rem 0}"
        "code{font:13px ui-monospace,SFMono-Regular,Menlo,monospace;background:#2a2826;color:#fbd968;"
        "padding:.1rem .35rem;border-radius:4px}"
        "strong{color:#fff;font-weight:650}em{color:#d6cfc6}"
        "a{color:#60a5fa;text-decoration:none}a:hover{text-decoration:underline}"
        "</style>"
        f"<div class=hd><h1>{_EMOJI[severity]} {_html_escape(name)}</h1>"
        f"<div class=meta>{now.strftime('%d/%m/%Y %H:%M %Z')} · {severity}</div></div>"
        f"<div class=body>{_render_markdown(body)}</div>\n"
    )
    path.write_text(doc, encoding="utf-8")
    return path


def _append_briefing_log(job: dict, severity: str, content: str) -> None:
    BRIEFING_LOG_DIR.mkdir(parents=True, exist_ok=True)
    now = _now()
    line = {
        "ts": now.isoformat(),
        "job_id": job.get("id"),
        "job": _job_title(job),
        "severity": severity,
        "summary": " ".join(_nonempty_lines(content))[:300],
    }
    with (BRIEFING_LOG_DIR / f"{now.strftime('%Y-%m-%d')}.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(line, ensure_ascii=False) + "\n")


def _dedup_key(job: dict, summary: str) -> str:
    norm = re.sub(r"\d+", "#", summary.lower())  # ignore varying counts/ids
    norm = re.sub(r"\s+", " ", norm).strip()
    return hashlib.sha1(f"{job.get('id','')}::{norm}".encode("utf-8")).hexdigest()


def _load_dedup() -> dict:
    try:
        return json.loads(DEDUP_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_dedup(state: dict) -> None:
    DEDUP_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = DEDUP_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    tmp.replace(DEDUP_FILE)


def _is_duplicate(key: str) -> bool:
    """True if this key was posted within the dedup window. Records the post.

    Held under ``_DEDUP_LOCK`` so concurrent cron threads cannot both miss the
    same key (which would double-post) or clobber each other's writes.
    """
    with _DEDUP_LOCK:
        state = _load_dedup()
        now = _now()
        last = state.get(key)
        if last:
            try:
                last_ts = dt.datetime.fromisoformat(last)
                if now - last_ts < DEDUP_WINDOW:
                    return True
            except Exception:
                pass
        state[key] = now.isoformat()
        # Garbage-collect entries older than twice the window.
        cutoff = now - 2 * DEDUP_WINDOW
        state = {k: v for k, v in state.items() if _safe_after(v, cutoff)}
        _save_dedup(state)
        return False


def _safe_after(iso: str, cutoff: dt.datetime) -> bool:
    try:
        return dt.datetime.fromisoformat(iso) >= cutoff
    except Exception:
        return False


def build_envelope(
    job: dict,
    content: str,
    success: bool = True,
    mention: str = "@Sahil",
    enable_dedup: bool = True,
) -> EnvelopeDecision:
    """Transform raw cron output into the standard envelope.

    Returns an :class:`EnvelopeDecision`. ``suppress=True`` means do not deliver.
    """
    if content is None or not content.strip():
        return EnvelopeDecision(suppress=True, reason="empty")

    if _SILENT_RE.search(content):
        return EnvelopeDecision(suppress=True, reason="silent-marker")

    severity, body = detect_severity(content, job, success)
    # Untrusted stdout: neutralise embedded Discord pings before it goes near a
    # channel, a title, a summary or the attachment.
    body = _strip_mentions(body.strip())

    if severity == LOG:
        _append_briefing_log(job, severity, body)
        # Carry a compact one-liner so the scheduler can mirror it to the
        # #cron-outputs audit sink. It still never posts to the domain channel.
        log_title = _build_title(body, job)
        log_summary = _build_summary(body, f"🟢 {log_title}")
        log_line = f"🟢 {log_title}" + (f"\n{log_summary}" if log_summary else "")
        return EnvelopeDecision(suppress=True, severity=LOG, text=log_line, reason="log-to-briefing")

    plain_title = _build_title(body, job)
    title = f"{_EMOJI[severity]} {plain_title}"
    summary = _build_summary(body, title)

    # Dedup before doing any work (attachment writes) so a suppressed repeat is
    # cheap and leaves no orphaned detail files.
    if enable_dedup:
        key = _dedup_key(job, summary or title)
        if _is_duplicate(key):
            return EnvelopeDecision(suppress=True, severity=severity, reason="deduped-12h")

    # The job already produced its own HTML attachment (the cron-output-contract
    # skill mandates one). Never offload over it — that would create a second,
    # competing attachment. Keep our title + summary and pass the MEDIA tag(s)
    # through untouched so the agent's own artifact is what gets delivered.
    media_tags = re.findall(r"(?m)^\s*MEDIA:\S.*$", body)
    if media_tags:
        clean_body = re.sub(r"(?m)^\s*MEDIA:\S.*$", "", body).strip()
        summary = _build_summary(clean_body, title)
        parts = [f"{_EMOJI[ACT]} {mention} · {plain_title}" if severity == ACT else title]
        if summary:
            parts.append(summary)
        parts.extend(t.strip() for t in media_tags)
        return EnvelopeDecision(
            suppress=False, text="\n".join(p for p in parts if p).strip(),
            severity=severity, reason="passthrough-media",
        )

    # Link-bearing output (deal scans, job alerts) must keep its links clickable
    # in the channel, so it stays inline up to Discord's message limit instead of
    # being buried in an attachment. Only genuinely oversized link output, or
    # link-free walls, get offloaded.
    head_len = len(mention) + len(plain_title) + 8
    if _URL_RE.search(body):
        offload = head_len + len(body) > DISCORD_SAFE_LIMIT
    else:
        offload = len(_nonempty_lines(body)) > MAX_BODY_LINES or len(body) > MAX_BODY_CHARS

    parts: list[str] = []
    if severity == ACT:
        parts.append(f"{_EMOJI[ACT]} {mention} · {plain_title}")
    else:
        parts.append(title)

    if offload:
        if summary:
            parts.append(summary)
        action = _action_line(body)
        if severity == ACT and action and action not in summary:
            parts.append(action)
        detail = _write_detail(job, severity, body)
        parts.append(f"📎 {detail.name}")
        parts.append(f"MEDIA:{detail}")
    else:
        # Short output: keep it inline, just under the enveloped title.
        remainder = body.strip()
        # Avoid duplicating the title line if the body led with it.
        if remainder and remainder.splitlines()[0] not in title:
            parts.append(remainder)
        elif len(remainder.splitlines()) > 1:
            parts.append("\n".join(remainder.splitlines()[1:]).strip())

    text = "\n".join(p for p in parts if p).strip()

    return EnvelopeDecision(suppress=False, text=text, severity=severity, reason="enveloped")
