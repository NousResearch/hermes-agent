"""Safe local HTML artifacts for selected cron job reports.

This module deliberately renders only an explicit human-report section.  It
must never render the full cron Markdown archive because that archive can
contain prompts, internal context, and audit metadata.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, Optional
from urllib.parse import urlparse

REPORT_START = "<!-- HERMES_HTML_REPORT_START -->"
REPORT_END = "<!-- HERMES_HTML_REPORT_END -->"

CSP = "default-src 'none'; img-src data:; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'"

_LOCAL_PATH_PATTERNS = (
    re.compile(r"(?<![:\w/])/(?:Users|home|tmp|var/folders|private/tmp|Volumes)/[^\s<>()\"']+"),
    re.compile(r"(?<![:\w/])/(?!/)(?:[A-Za-z0-9._-]+/)+[^\s<>()\"']+"),
    re.compile(r"\b[A-Za-z]:[\\/][^\s<>()\"']+"),
)
_EVENT_HANDLER_ASSIGNMENT_RE = re.compile(
    r"(^|\s)on[a-zA-Z][\w:-]*\s*=\s*(\"[^\"]*\"|'[^']*'|[^\s>]+)",
    re.IGNORECASE,
)


def _redact_sensitive(text: str) -> str:
    """Redact secrets before text is persisted into an HTML artifact.

    This boundary is fail-closed: if the shared redactor cannot be loaded or
    raises unexpectedly, do not return the original text to the artifact.
    """
    if not text:
        return text
    try:
        from agent.redact import redact_sensitive_text
        return redact_sensitive_text(text, force=True)
    except Exception:
        return "[REDACTED]"


def _redact_local_paths(text: str) -> str:
    """Remove absolute local filesystem paths from user-facing HTML text."""
    if not text:
        return text
    redacted = text
    for pattern in _LOCAL_PATH_PATTERNS:
        redacted = pattern.sub("[LOCAL_PATH]", redacted)
    return redacted


def _strip_event_handlers(text: str) -> str:
    """Remove literal event-handler assignments from rendered text."""
    if not text:
        return text
    return _EVENT_HANDLER_ASSIGNMENT_RE.sub(lambda m: m.group(1), text)


def _sanitize_text(text: str) -> str:
    """Apply fail-closed secret redaction, local-path redaction, and cleanup."""
    return _strip_event_handlers(_redact_local_paths(_redact_sensitive(text)))


def _source_filename_only(value: str) -> str:
    """Return only the final filename segment for artifact metadata."""
    value = str(value or "")
    if not value:
        return ""
    value = value.replace("\\", "/")
    return value.rsplit("/", 1)[-1]


@dataclass(frozen=True)
class HtmlReportMetadata:
    """Non-sensitive metadata displayed in the rendered report footer."""

    job_id: str
    job_name: str = ""
    run_time: str = ""
    source_filename: str = ""


def extract_html_report_body(text: str) -> Optional[str]:
    """Extract exactly one non-empty human-report section from text.

    Returns None when markers are absent, duplicated, reversed, or enclose only
    whitespace.  There is intentionally no fallback to the full input.
    """
    if not isinstance(text, str) or not text:
        return None
    if text.count(REPORT_START) != 1 or text.count(REPORT_END) != 1:
        return None
    start = text.find(REPORT_START)
    end = text.find(REPORT_END)
    if start < 0 or end < 0 or end <= start:
        return None
    body = text[start + len(REPORT_START):end].strip()
    return body or None


def strip_html_report_section(text: str) -> str:
    """Remove the explicit HTML report block from chat-facing output.

    This lets a cron response carry two surfaces in one final response: a short
    Telegram/chat summary plus a delimited long-form report for local HTML.
    If markers are missing or ambiguous, return the original text unchanged.
    """
    if not isinstance(text, str) or not text:
        return text
    if text.count(REPORT_START) != 1 or text.count(REPORT_END) != 1:
        return text
    start = text.find(REPORT_START)
    end = text.find(REPORT_END)
    if start < 0 or end < 0 or end <= start:
        return text
    stripped = (text[:start] + text[end + len(REPORT_END):]).strip()
    return re.sub(r"\n{3,}", "\n\n", stripped)


def _safe_href(raw: str) -> str:
    parsed = urlparse(raw)
    if parsed.scheme.lower() in {"http", "https", "mailto"} and parsed.netloc:
        return raw
    return ""


def _emphasis_escaped(text: str) -> str:
    """Escape text and render a tiny safe Markdown emphasis subset."""
    escaped = html.escape(text)
    escaped = re.sub(r"\*\*([^*\n]+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"<em>\1</em>", escaped)
    return escaped


def _linkify_escaped(text: str) -> str:
    """Escape text, convert safe Markdown/plain links, and render emphasis."""
    markdown_link = re.compile(r"\[([^\]\n]{1,240})\]\(([^)\s]+)\)")
    chunks: list[str] = []
    pos = 0
    for match in markdown_link.finditer(text):
        chunks.append(_linkify_plain_urls(text[pos:match.start()]))
        label = match.group(1)
        raw_href = match.group(2)
        href = _safe_href(raw_href)
        if href:
            chunks.append(
                f'<a href="{html.escape(href, quote=True)}" rel="noreferrer">{_emphasis_escaped(label)}</a>'
            )
        else:
            chunks.append(_emphasis_escaped(match.group(0)))
        pos = match.end()
    chunks.append(_linkify_plain_urls(text[pos:]))
    return "".join(chunks)


def _linkify_plain_urls(text: str) -> str:
    """Escape text and convert plain safe URLs to anchors."""
    pattern = re.compile(r"(?P<url>https?://[^\s<>()]+)")
    parts: list[str] = []
    pos = 0
    for match in pattern.finditer(text):
        parts.append(_emphasis_escaped(text[pos:match.start()]))
        url = match.group("url").rstrip(".,;:!?")
        suffix = match.group("url")[len(url):]
        href = _safe_href(url)
        if href:
            escaped_url = html.escape(url)
            escaped_href = html.escape(href, quote=True)
            parts.append(f'<a href="{escaped_href}" rel="noreferrer">{escaped_url}</a>')
        else:
            parts.append(_emphasis_escaped(url))
        parts.append(_emphasis_escaped(suffix))
        pos = match.end()
    parts.append(_emphasis_escaped(text[pos:]))
    return "".join(parts)


def _render_inline(text: str) -> str:
    sanitized = _sanitize_text(text)
    parts: list[str] = []
    pos = 0
    for match in re.finditer(r"`([^`\n]+)`", sanitized):
        parts.append(_linkify_escaped(sanitized[pos:match.start()]))
        parts.append(f"<code>{html.escape(match.group(1))}</code>")
        pos = match.end()
    parts.append(_linkify_escaped(sanitized[pos:]))
    return "".join(parts)


def _render_code_block(code_lines: list[str]) -> str:
    code_text = _sanitize_text("\n".join(code_lines))
    return "<pre><code>" + html.escape(code_text) + "</code></pre>"


def render_report_body(body: str) -> str:
    """Render a conservative Markdown-ish subset into sanitized HTML."""
    lines = body.splitlines()
    out: list[str] = []
    paragraph: list[str] = []
    in_code = False
    code_lines: list[str] = []
    list_tag: str | None = None
    section_open = False

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            text = " ".join(part.strip() for part in paragraph if part.strip())
            if text:
                out.append(f"<p>{_render_inline(text)}</p>")
            paragraph = []

    def close_list() -> None:
        nonlocal list_tag
        if list_tag:
            out.append(f"</{list_tag}>")
            list_tag = None

    def close_section() -> None:
        nonlocal section_open
        flush_paragraph()
        close_list()
        if section_open:
            out.append("</section>")
            section_open = False

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        stripped = line.strip()

        if stripped.startswith("```"):
            if in_code:
                out.append(_render_code_block(code_lines))
                code_lines = []
                in_code = False
            else:
                flush_paragraph()
                close_list()
                in_code = True
                code_lines = []
            continue

        if in_code:
            code_lines.append(line)
            continue

        if not stripped:
            flush_paragraph()
            close_list()
            continue

        heading = re.match(r"^(#{1,3})\s+(.+)$", stripped)
        if heading:
            flush_paragraph()
            close_list()
            level = len(heading.group(1))
            heading_html = _render_inline(heading.group(2).strip())
            if level == 1:
                close_section()
                out.append(f'<h1 class="report-title">{heading_html}</h1>')
            elif level == 2:
                close_section()
                out.append('<section class="report-section">')
                section_open = True
                out.append(f'<h2 class="section-title">{heading_html}</h2>')
            else:
                if not section_open:
                    out.append('<section class="report-section">')
                    section_open = True
                out.append(f"<h3>{heading_html}</h3>")
            continue

        bullet = re.match(r"^[-*]\s+(.+)$", stripped)
        if bullet:
            flush_paragraph()
            if list_tag != "ul":
                close_list()
                out.append("<ul>")
                list_tag = "ul"
            out.append(f"<li>{_render_inline(bullet.group(1).strip())}</li>")
            continue

        numbered = re.match(r"^\d+[.)]\s+(.+)$", stripped)
        if numbered:
            flush_paragraph()
            if list_tag != "ol":
                close_list()
                out.append("<ol>")
                list_tag = "ol"
            out.append(f"<li>{_render_inline(numbered.group(1).strip())}</li>")
            continue

        close_list()
        paragraph.append(line)

    if in_code:
        out.append(_render_code_block(code_lines))
    flush_paragraph()
    close_list()
    close_section()
    return "\n".join(out)


def render_html_report(body: str, metadata: HtmlReportMetadata | Mapping[str, str]) -> str:
    """Render a standalone, sanitized HTML report."""
    if isinstance(metadata, Mapping):
        meta = HtmlReportMetadata(**{k: str(v) for k, v in metadata.items() if k in HtmlReportMetadata.__annotations__})
    else:
        meta = metadata
    title = _sanitize_text(meta.job_name or f"Cron report {meta.job_id}")
    job_id = _sanitize_text(meta.job_id)
    run_time = _sanitize_text(meta.run_time or datetime.now(timezone.utc).isoformat())
    source_filename = _source_filename_only(_redact_sensitive(meta.source_filename))
    rendered = render_report_body(body)

    footer_items = [
        ("Job", title),
        ("Job ID", job_id),
        ("Run time", run_time),
    ]
    if source_filename:
        footer_items.append(("Source", source_filename))
    footer = "".join(
        f"<dt>{html.escape(label)}</dt><dd>{html.escape(value)}</dd>"
        for label, value in footer_items
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="{html.escape(CSP, quote=False)}">
<title>{html.escape(title)}</title>
<style>
:root {{ color-scheme: dark; --bg:#08090a; --panel:#0f1011; --surface:rgba(255,255,255,.035); --surface-2:rgba(255,255,255,.055); --border:rgba(255,255,255,.08); --border-soft:rgba(255,255,255,.05); --text:#f7f8f8; --muted:#8a8f98; --soft:#d0d6e0; --accent:#7170ff; --accent-2:#a9a7ff; --gold:#c6a15b; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-feature-settings:'cv01','ss03'; line-height:1.58; background: radial-gradient(circle at 20% -10%, rgba(113,112,255,.22), transparent 34rem), radial-gradient(circle at 85% 0%, rgba(198,161,91,.11), transparent 30rem), var(--bg); color:var(--soft); }}
main {{ max-width:1040px; margin:0 auto; padding:42px 20px 64px; }}
.masthead {{ display:flex; justify-content:space-between; gap:20px; align-items:flex-end; margin:0 0 20px; padding:0 4px; }}
.brand {{ display:flex; gap:12px; align-items:center; color:var(--text); font-weight:590; letter-spacing:-.03em; }}
.mark {{ width:36px; height:36px; display:grid; place-items:center; border-radius:10px; color:#111; background:linear-gradient(135deg, var(--gold), #f1d798); box-shadow:0 0 32px rgba(198,161,91,.18); font-family:ui-serif, Georgia, serif; font-weight:700; }}
.kicker {{ margin:0 0 6px; color:var(--accent-2); font:510 12px/1 ui-monospace, SFMono-Regular, Menlo, monospace; text-transform:uppercase; letter-spacing:.14em; }}
.meta-chip {{ color:var(--muted); border:1px solid var(--border-soft); border-radius:999px; padding:7px 10px; background:rgba(255,255,255,.025); font-size:12px; white-space:nowrap; }}
article {{ position:relative; overflow:hidden; background:linear-gradient(180deg, rgba(255,255,255,.052), rgba(255,255,255,.026)); border:1px solid var(--border); border-radius:24px; padding:34px; box-shadow:0 24px 80px rgba(0,0,0,.38), inset 0 1px 0 rgba(255,255,255,.06); }}
article:before {{ content:""; position:absolute; inset:0 0 auto; height:1px; background:linear-gradient(90deg, transparent, rgba(198,161,91,.55), rgba(113,112,255,.55), transparent); }}
.report-title {{ margin:0 0 24px; max-width:820px; color:var(--text); font-size:clamp(34px, 6vw, 64px); line-height:.98; letter-spacing:-.055em; font-weight:590; }}
.report-section {{ margin:16px 0; padding:22px; border:1px solid var(--border-soft); border-radius:18px; background:linear-gradient(180deg, rgba(255,255,255,.045), rgba(255,255,255,.022)); box-shadow:inset 0 0 18px rgba(0,0,0,.16); }}
.section-title {{ display:flex; align-items:center; gap:10px; margin:0 0 14px; color:var(--text); font-size:20px; line-height:1.2; letter-spacing:-.02em; font-weight:590; }}
.section-title:before {{ content:""; width:7px; height:18px; border-radius:99px; background:linear-gradient(180deg, var(--gold), var(--accent)); box-shadow:0 0 18px rgba(113,112,255,.25); }}
h3 {{ margin:18px 0 8px; color:var(--text); font-size:16px; letter-spacing:-.01em; }}
p {{ margin:.78em 0; }}
ul, ol {{ margin:.7em 0; padding-left:0; list-style:none; }}
li {{ position:relative; margin:.58em 0; padding-left:22px; }}
li:before {{ content:""; position:absolute; left:4px; top:.72em; width:6px; height:6px; border-radius:50%; background:var(--accent); box-shadow:0 0 12px rgba(113,112,255,.45); }}
ol {{ counter-reset:item; }}
ol li {{ counter-increment:item; padding-left:30px; }}
ol li:before {{ content:counter(item); top:.15em; left:0; width:20px; height:20px; display:grid; place-items:center; font:510 11px/1 ui-monospace, SFMono-Regular, Menlo, monospace; color:var(--text); background:rgba(113,112,255,.22); }}
strong {{ color:var(--text); font-weight:590; }}
em {{ color:#e7ddc9; font-style:normal; }}
a {{ color:var(--accent-2); text-decoration:none; border-bottom:1px solid rgba(169,167,255,.35); }}
a:hover {{ color:#fff; border-bottom-color:#fff; }}
pre {{ overflow-x:auto; background:#050608; border:1px solid var(--border); border-radius:14px; padding:16px; }}
code {{ font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:.92em; color:#e7ddc9; }}
p code, li code {{ background:rgba(255,255,255,.065); border:1px solid var(--border-soft); border-radius:6px; padding:1px 5px; }}
footer {{ margin-top:22px; color:var(--muted); font-size:.86rem; border-top:1px solid var(--border-soft); padding-top:18px; }}
dl {{ display:grid; grid-template-columns:max-content 1fr; gap:6px 14px; }}
dt {{ color:var(--soft); font-weight:590; }}
dd {{ margin:0; overflow-wrap:anywhere; }}
@media (max-width:700px) {{ main {{ padding:24px 12px 44px; }} .masthead {{ align-items:flex-start; flex-direction:column; }} article {{ padding:22px 16px; border-radius:18px; }} .report-section {{ padding:17px 15px; }} }}
</style>
</head>
<body>
<main>
<header class="masthead">
  <div>
    <p class="kicker">Acta Diurna</p>
    <div class="brand"><span class="mark">A</span><span>{html.escape(title)}</span></div>
  </div>
  <div class="meta-chip">{html.escape(run_time)}</div>
</header>
<article>
{rendered}
<footer><dl>{footer}</dl></footer>
</article>
</main>
</body>
</html>
"""


def render_report_from_output(output: str, metadata: HtmlReportMetadata | Mapping[str, str]) -> Optional[str]:
    """Extract and render a report section from a cron output string."""
    body = extract_html_report_body(output)
    if body is None:
        return None
    return render_html_report(body, metadata)
