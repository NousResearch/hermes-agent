#!/usr/bin/env python3
"""Format calendar events as concise Discord text plus an attached HTML file."""
import html
import json
import datetime as dt
from pathlib import Path

BASE_DIR = Path("/home/kensei/.hermes/runbooks/calendar-brief")
EVENTS_FILE = BASE_DIR / "events.json"
TZ = dt.timezone(dt.timedelta(hours=1))
now = dt.datetime.now(TZ)
stamp_dir = now.strftime("%d-%m-%y")
out_dir = BASE_DIR / stamp_dir
out_dir.mkdir(parents=True, exist_ok=True)
html_path = out_dir / f"calendar-brief-{now.strftime('%d%m%y-%H%M%S')}.html"

if not EVENTS_FILE.exists():
    print(f"⚠️ Calendar brief · {now.strftime('%d/%m/%Y %H:%M:%S')}")
    print("failed · 0 events · fetch data missing")
    print("• Calendar fetch output was not found.")
    raise SystemExit(0)

data = json.loads(EVENTS_FILE.read_text())
today = data.get("today", []) or []
week = data.get("week", []) or []
issues = data.get("issues", []) or []

if not today and not week and not issues:
    raise SystemExit(0)

def parse_dt(value):
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None

def fmt_range(ev):
    start = parse_dt(ev.get("start", ""))
    end = parse_dt(ev.get("end", ""))
    if not start:
        return "all-day"
    if end and end.date() == start.date() and (end - start).total_seconds() < 86400:
        return f"{start.strftime('%H:%M:%S')}-{end.strftime('%H:%M:%S')}"
    return start.strftime("%H:%M:%S")

def fmt_day(ev):
    start = parse_dt(ev.get("start", ""))
    return start.strftime("%d/%m/%Y") if start else "unknown date"

def clean(value):
    return str(value or "").strip()

status = "⚠️" if issues else "✅"
count = len(today)
signal = f"{len(issues)} fetch issue(s)" if issues else ("clear day" if count == 0 else "today mapped")

lines = [
    f"{status} Calendar brief · {now.strftime('%d/%m/%Y %H:%M:%S')}",
    f"checked · {count} today · {signal}",
    "",
]
for ev in today[:5]:
    name = clean(ev.get("summary")) or "Untitled"
    loc = clean(ev.get("location"))
    loc_part = f" @ {loc}" if loc else ""
    lines.append(f"• {fmt_range(ev)} · {name}{loc_part}")
if not today and not issues:
    lines.append("• No events scheduled today.")
if issues:
    lines.append(f"• Fetch issues: {len(issues)}. Full detail attached.")

html_lines = [
    "<!doctype html><html><head><meta charset='utf-8'>",
    "<title>Calendar brief</title>",
    "<style>body{background:#11100f;color:#f5f5f4;font-family:Inter,Arial,sans-serif;margin:32px;line-height:1.5} h1,h2{color:#fbbf24} .card{background:#1c1917;border:1px solid #3f3f46;border-radius:14px;padding:16px;margin:14px 0}.muted{color:#a8a29e}</style>",
    "</head><body>",
    f"<h1>Calendar brief · {html.escape(now.strftime('%d/%m/%Y %H:%M:%S'))}</h1>",
    f"<p class='muted'>{len(today)} today · {len(week)} week ahead · {len(issues)} issue(s)</p>",
    "<h2>Today</h2>",
]
if today:
    for ev in today:
        name = html.escape(clean(ev.get("summary")) or "Untitled")
        loc = html.escape(clean(ev.get("location")))
        link = clean(ev.get("link"))
        title = f"<a href='{html.escape(link)}'>{name}</a>" if link else name
        html_lines.append(f"<div class='card'><strong>{fmt_range(ev)}</strong> · {title}<br><span class='muted'>{loc}</span></div>")
else:
    html_lines.append("<div class='card'>No events scheduled today.</div>")
html_lines.append("<h2>Week ahead</h2>")
if week:
    for ev in week:
        name = html.escape(clean(ev.get("summary")) or "Untitled")
        html_lines.append(f"<div class='card'><strong>{fmt_day(ev)} {fmt_range(ev)}</strong> · {name}</div>")
else:
    html_lines.append("<div class='card'>No week-ahead events found.</div>")
if issues:
    html_lines.append("<h2>Fetch issues</h2>")
    for issue in issues:
        html_lines.append(f"<div class='card'>{html.escape(str(issue))}</div>")
html_lines.append("</body></html>")
html_path.write_text("\n".join(html_lines))
if not html_path.exists():
    raise SystemExit("HTML attachment was not written")

lines += ["", f"MEDIA:{html_path}"]
print("\n".join(lines))
