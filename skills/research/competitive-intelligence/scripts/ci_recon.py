#!/usr/bin/env python3
"""
ci_recon.py — stdlib-only helper for the competitive-intelligence skill.

Subcommands
-----------
validate <url>
    Check HTTP accessibility and robots.txt compliance.
    Prints a JSON status object. Exits 0 if accessible and allowed, 1 otherwise.

save_report <slug>
    Read a markdown report from stdin and save it atomically under
    ~/.hermes/competitive-intelligence/<slug>/<ISO-datetime>/.
    Prints the saved path to stdout.

export_html <slug>
    Generate a self-contained interactive HTML report from the latest saved
    report for <slug>. Prints the path to report.html.

export_pdf <slug>
    Generate a PDF from the latest saved report for <slug>.
    Tries wkhtmltopdf, then pandoc. Falls back to HTML + browser instructions.
    Prints the path to the output file.

export_csv <slug>
    Extract every markdown table from the latest saved report as individual
    CSV files. Prints one path per file.
"""

from __future__ import annotations

import csv
import html
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


_TIMEOUT = 10
_USER_AGENT = "HermesAgent/1.0 (+https://github.com/NousResearch/hermes-agent)"


def _hermes_home() -> Path:
    env = os.environ.get("HERMES_HOME", "")
    if env:
        return Path(env)
    return Path.home() / ".hermes"


def _safe_slug(slug: str) -> str:
    """Sanitize slug to prevent path traversal and filesystem issues.

    Dots are intentionally excluded — they enable '..' traversal sequences
    even after stripping leading dots.
    """
    # Allow only alphanumeric, dash, underscore (no dots)
    safe = re.sub(r"[^\w\-]", "_", slug.strip())
    # Strip leading underscores/dashes (handles inputs like '___etc')
    safe = re.sub(r"^[_\-]+", "", safe)
    # Collapse repeated separators
    safe = re.sub(r"[_\-]{2,}", "-", safe)
    safe = safe[:80].rstrip("-_") or "report"
    return safe


def _find_latest_report_dir(slug: str) -> Optional[Path]:
    base = _hermes_home() / "competitive-intelligence" / _safe_slug(slug)
    if not base.exists():
        return None
    dirs = sorted(
        (d for d in base.iterdir() if d.is_dir() and (d / "report.md").exists()),
        reverse=True,
    )
    return dirs[0] if dirs else None


def _atomic_write(path: Path, content: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

def cmd_validate(url: str) -> int:
    result: dict = {
        "url": url,
        "accessible": False,
        "robots_allows_crawl": False,
        "status_code": None,
        "error": None,
    }

    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            result["status_code"] = resp.status
            result["accessible"] = True
    except urllib.error.HTTPError as exc:
        result["status_code"] = exc.code
        # Treat redirects as accessible; content gate check is the agent's job
        if exc.code in (301, 302, 303, 307, 308):
            result["accessible"] = True
        else:
            result["error"] = f"HTTP {exc.code}: {exc.reason}"
    except urllib.error.URLError as exc:
        result["error"] = str(exc.reason)
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)

    if not result["accessible"]:
        print(json.dumps(result, indent=2))
        return 1

    parsed = urllib.parse.urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
        allowed = rp.can_fetch(_USER_AGENT, url) or rp.can_fetch("*", url)
        result["robots_allows_crawl"] = bool(allowed)
        if not result["robots_allows_crawl"]:
            result["error"] = "robots.txt disallows crawling for this user-agent"
    except Exception:  # noqa: BLE001
        # Unreachable robots.txt → assume allowed (common convention)
        result["robots_allows_crawl"] = True

    print(json.dumps(result, indent=2))
    return 0 if result["robots_allows_crawl"] else 1


# ---------------------------------------------------------------------------
# save_report
# ---------------------------------------------------------------------------

def cmd_save_report(slug: str) -> int:
    slug = _safe_slug(slug)
    report_md = sys.stdin.read()
    if not report_md.strip():
        print("ERROR: no report content received on stdin", file=sys.stderr)
        return 1

    # Include time in directory name to avoid same-day overwrites
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = _hermes_home() / "competitive-intelligence" / slug / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "slug": slug,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_urls": [],
    }

    _atomic_write(out_dir / "report.md", report_md)
    _atomic_write(out_dir / "meta.json", json.dumps(meta, indent=2, ensure_ascii=False) + "\n")

    print(str(out_dir / "report.md"))
    return 0


# ---------------------------------------------------------------------------
# export_html
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Competitive Intelligence — {slug_title}</title>
<script src="https://cdn.jsdelivr.net/npm/marked@9/marked.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#f0f2f5;color:#1a1a2e;display:flex;min-height:100vh}}
#sidebar{{width:260px;min-width:260px;background:#1e2329;color:#c9d1d9;padding:20px;position:sticky;top:0;height:100vh;overflow-y:auto;flex-shrink:0}}
#brand{{color:#58a6ff;font-weight:700;font-size:1.05rem;margin-bottom:18px;padding-bottom:14px;border-bottom:1px solid #30363d}}
#toc-title{{color:#8b949e;font-size:0.7rem;text-transform:uppercase;letter-spacing:1.2px;margin-bottom:10px}}
#toc{{list-style:none}}
#toc li.h2{{margin-top:7px}}
#toc li.h3{{padding-left:14px;font-size:0.82rem}}
#toc a{{color:#8b949e;text-decoration:none;display:block;padding:3px 0;line-height:1.4}}
#toc a:hover{{color:#58a6ff}}
#meta{{color:#6e7681;font-size:0.72rem;margin-top:18px;padding-top:14px;border-top:1px solid #30363d;line-height:1.6}}
#content{{flex:1;padding:48px 56px;max-width:1000px}}
#report h1{{font-size:1.75rem;color:#0066cc;border-bottom:3px solid #0066cc;padding-bottom:10px;margin:32px 0 20px}}
#report h2{{font-size:1.2rem;color:#1a1a2e;border-bottom:2px solid #e0e6ed;padding-bottom:7px;margin:28px 0 14px;cursor:pointer;user-select:none}}
#report h2::after{{content:' ▾';font-size:0.75em;color:#aaa}}
#report h2.collapsed::after{{content:' ▸'}}
#report h3{{font-size:1rem;color:#333;margin:18px 0 9px}}
#report p{{line-height:1.75;margin-bottom:12px}}
#report ul,#report ol{{padding-left:22px;margin-bottom:12px;line-height:1.75}}
#report table{{border-collapse:collapse;width:100%;margin:16px 0;font-size:0.88rem;box-shadow:0 1px 4px rgba(0,0,0,.1);border-radius:6px;overflow:hidden}}
#report th{{background:#0055b3;color:#fff;padding:10px 14px;text-align:left;font-weight:600;font-size:0.82rem;text-transform:uppercase;letter-spacing:.5px}}
#report td{{padding:9px 14px;border-bottom:1px solid #e8edf2}}
#report tr:nth-child(even) td{{background:#f5f8fc}}
#report tr:hover td{{background:#eaf0ff}}
#report code{{background:#eef0f3;padding:2px 6px;border-radius:4px;font-size:.87em;font-family:'SFMono-Regular',Consolas,monospace}}
#report pre{{background:#1e2329;color:#c9d1d9;padding:16px 20px;border-radius:8px;overflow-x:auto;margin:14px 0;font-size:.85em}}
#report pre code{{background:none;color:inherit;padding:0}}
#report blockquote{{border-left:4px solid #0066cc;margin:14px 0;padding:8px 16px;background:#eef4ff;color:#444;border-radius:0 4px 4px 0}}
#report hr{{border:none;border-top:2px solid #e0e6ed;margin:28px 0}}
.badge{{display:inline-block;padding:1px 7px;border-radius:10px;font-size:.72em;font-weight:700;margin:0 1px;vertical-align:middle}}
.badge.high{{background:#d4edda;color:#155724}}
.badge.med{{background:#fff3cd;color:#856404}}
.badge.low{{background:#fde8cc;color:#7a4000}}
.cdn-warn{{color:#856404;background:#fff3cd;padding:10px 16px;border-radius:6px;margin-bottom:20px;font-size:.88rem;border:1px solid #ffc107}}
noscript pre{{white-space:pre-wrap;word-wrap:break-word;padding:32px;line-height:1.6;font-size:.9rem}}
@media print{{
  #sidebar{{display:none}}
  #content{{padding:20px;max-width:100%}}
  #report h2::after,#report h2.collapsed::after{{content:''}}
  body{{background:#fff}}
}}
@media(max-width:768px){{
  body{{flex-direction:column}}
  #sidebar{{width:100%;height:auto;position:static;overflow:visible}}
  #content{{padding:20px}}
}}
</style>
</head>
<body>
<aside id="sidebar">
  <div id="brand">&#9889; Herm&egrave;s CI</div>
  <div id="toc-title">Contents</div>
  <ul id="toc"></ul>
  <div id="meta">
    <strong>Slug:</strong> {slug_escaped}<br>
    <strong>Generated:</strong> {date}
  </div>
</aside>
<main id="content">
  <div id="report"></div>
</main>
<noscript>
  <pre>{escaped_md}</pre>
</noscript>
<script>
const src = {json_md};
try {{
  document.getElementById('report').innerHTML = marked.parse(src);
}} catch (e) {{
  const el = document.getElementById('report');
  const warn = document.createElement('p');
  warn.className = 'cdn-warn';
  warn.textContent = '⚠ Markdown renderer unavailable (CDN unreachable or JS disabled). Showing raw markdown.';
  const pre = document.createElement('pre');
  pre.style.cssText = 'white-space:pre-wrap;font-size:.9rem;line-height:1.7;padding:20px;background:#f8f9fa;border-radius:6px';
  pre.textContent = src;
  el.appendChild(warn);
  el.appendChild(pre);
}}

// Build sidebar TOC
const toc = document.getElementById('toc');
document.querySelectorAll('#report h2, #report h3').forEach(function(h, i) {{
  const id = 'sec-' + i;
  h.id = id;
  const li = document.createElement('li');
  li.className = h.tagName.toLowerCase();
  const a = document.createElement('a');
  a.href = '#' + id;
  a.textContent = h.textContent.replace(/ [▾▸]$/, '');
  li.appendChild(a);
  toc.appendChild(li);
}});

// Confidence badges
document.querySelectorAll('#report p, #report li, #report td').forEach(function(el) {{
  el.innerHTML = el.innerHTML
    .replace(/\[H\]/g, '<span class="badge high" title="High — direct evidence">H</span>')
    .replace(/\[M\]/g, '<span class="badge med"  title="Medium — indirect signals">M</span>')
    .replace(/\[L\]/g, '<span class="badge low"  title="Low — inference only">L</span>');
}});

// Collapsible H2 sections
document.querySelectorAll('#report h2').forEach(function(h2) {{
  h2.addEventListener('click', function() {{
    const collapsed = h2.classList.toggle('collapsed');
    let el = h2.nextElementSibling;
    while (el && el.tagName !== 'H2') {{
      el.style.display = collapsed ? 'none' : '';
      el = el.nextElementSibling;
    }}
  }});
}});

// Smooth scroll for TOC links
document.querySelectorAll('#toc a').forEach(function(a) {{
  a.addEventListener('click', function(e) {{
    e.preventDefault();
    const target = document.querySelector(a.getAttribute('href'));
    if (target) target.scrollIntoView({{behavior: 'smooth', block: 'start'}});
  }});
}});
</script>
</body>
</html>
"""


def cmd_export_html(slug: str) -> int:
    slug = _safe_slug(slug)
    out_dir = _find_latest_report_dir(slug)
    if not out_dir:
        print(f"ERROR: no saved report found for slug {slug!r}", file=sys.stderr)
        return 1

    content = (out_dir / "report.md").read_text(encoding="utf-8")
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    rendered = _HTML_TEMPLATE.format(
        slug_title=html.escape(slug),
        slug_escaped=html.escape(slug),
        date=html.escape(date_str),
        json_md=json.dumps(content),
        escaped_md=html.escape(content),
    )

    html_path = out_dir / "report.html"
    _atomic_write(html_path, rendered)
    print(str(html_path))
    return 0


# ---------------------------------------------------------------------------
# export_pdf
# ---------------------------------------------------------------------------

def _try_pdf_tool(args: list[str], label: str) -> Optional[Path]:
    """Run a PDF conversion command; return output path on success, None on failure."""
    pdf_path = Path(args[-1])
    r = subprocess.run(args, capture_output=True)
    if r.returncode == 0 and pdf_path.exists() and pdf_path.stat().st_size > 0:
        return pdf_path
    stderr = r.stderr.decode(errors="replace").strip()
    if stderr:
        print(f"{label} failed: {stderr}", file=sys.stderr)
    else:
        print(f"{label} failed (exit {r.returncode})", file=sys.stderr)
    return None


def cmd_export_pdf(slug: str) -> int:
    slug = _safe_slug(slug)
    out_dir = _find_latest_report_dir(slug)
    if not out_dir:
        print(f"ERROR: no saved report found for slug {slug!r}", file=sys.stderr)
        return 1

    md_path = out_dir / "report.md"
    pdf_path = out_dir / "report.pdf"

    # Ensure HTML artifact exists (needed by wkhtmltopdf)
    html_path = out_dir / "report.html"
    if not html_path.exists():
        if cmd_export_html(slug) != 0:
            print("WARNING: could not generate HTML for PDF conversion", file=sys.stderr)

    # 1. wkhtmltopdf (HTML → PDF, no LaTeX dependency)
    if shutil.which("wkhtmltopdf") and html_path.exists():
        result = _try_pdf_tool(
            ["wkhtmltopdf", "--quiet", str(html_path), str(pdf_path)],
            "wkhtmltopdf",
        )
        if result:
            print(str(result))
            return 0

    # 2. pandoc with wkhtmltopdf engine
    if shutil.which("pandoc"):
        result = _try_pdf_tool(
            ["pandoc", str(md_path), "-o", str(pdf_path), "--pdf-engine=wkhtmltopdf"],
            "pandoc (wkhtmltopdf engine)",
        )
        if result:
            print(str(result))
            return 0
        # pandoc with default engine (may use pdflatex/xelatex if installed)
        result = _try_pdf_tool(
            ["pandoc", str(md_path), "-o", str(pdf_path)],
            "pandoc (default engine)",
        )
        if result:
            print(str(result))
            return 0

    # 3. weasyprint (Python package, may or may not be installed)
    if shutil.which("weasyprint") and html_path.exists():
        result = _try_pdf_tool(
            ["weasyprint", str(html_path), str(pdf_path)],
            "weasyprint",
        )
        if result:
            print(str(result))
            return 0

    # 4. Fallback — HTML is a usable substitute
    if html_path.exists():
        print(
            "WARNING: No PDF tool found (tried wkhtmltopdf, pandoc, weasyprint).\n"
            f"HTML report saved at: {html_path}\n"
            "To create a PDF: open in browser → File → Print → Save as PDF.",
            file=sys.stderr,
        )
        print(str(html_path))
        return 0

    print("ERROR: could not produce PDF or HTML output", file=sys.stderr)
    return 1


# ---------------------------------------------------------------------------
# export_csv
# ---------------------------------------------------------------------------

_TABLE_SEP_RE = re.compile(r"^\|[-|: ]*-[-|: ]*\|")  # requires at least one dash


def _parse_markdown_tables(content: str) -> list[tuple[str, list[list[str]]]]:
    """Return (heading_slug, rows) for every markdown table in content."""
    results: list[tuple[str, list[list[str]]]] = []
    lines = content.split("\n")
    current_heading = "table"
    heading_count: dict[str, int] = {}
    i = 0

    while i < len(lines):
        stripped = lines[i].strip()

        if re.match(r"^#{1,4}\s", stripped):
            text = re.sub(r"^#+\s*", "", stripped)
            slug = re.sub(r"[^\w\s-]", "", text).strip().lower()
            slug = re.sub(r"[\s_]+", "_", slug)[:40].rstrip("_")
            current_heading = slug or "table"

        if stripped.startswith("|") and i + 1 < len(lines):
            sep = lines[i + 1].strip()
            if _TABLE_SEP_RE.match(sep):
                rows: list[list[str]] = []
                j = i
                while j < len(lines) and lines[j].strip().startswith("|"):
                    row_stripped = lines[j].strip()
                    if _TABLE_SEP_RE.match(row_stripped):
                        j += 1
                        continue
                    cells = [c.strip() for c in row_stripped.split("|")[1:-1]]
                    rows.append(cells)
                    j += 1

                if len(rows) >= 2:
                    count = heading_count.get(current_heading, 0)
                    heading_count[current_heading] = count + 1
                    name = current_heading if count == 0 else f"{current_heading}_{count + 1}"
                    results.append((name, rows))
                i = j
                continue

        i += 1

    return results


def cmd_export_csv(slug: str) -> int:
    slug = _safe_slug(slug)
    out_dir = _find_latest_report_dir(slug)
    if not out_dir:
        print(f"ERROR: no saved report found for slug {slug!r}", file=sys.stderr)
        return 1

    content = (out_dir / "report.md").read_text(encoding="utf-8")
    tables = _parse_markdown_tables(content)

    if not tables:
        print("WARNING: no markdown tables found in report", file=sys.stderr)
        return 0

    csv_dir = out_dir / "csv"
    csv_dir.mkdir(exist_ok=True)

    for name, rows in tables:
        csv_path = csv_dir / f"{name}.csv"
        buf = io.StringIO()
        writer = csv.writer(buf, quoting=csv.QUOTE_MINIMAL)
        writer.writerows(rows)
        _atomic_write(csv_path, buf.getvalue())
        print(str(csv_path))

    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 1

    sub = args[0]

    if sub == "validate":
        if len(args) < 2:
            print("Usage: ci_recon.py validate <url>", file=sys.stderr)
            return 1
        return cmd_validate(args[1])

    if sub == "save_report":
        if len(args) < 2:
            print("Usage: ci_recon.py save_report <slug>", file=sys.stderr)
            return 1
        return cmd_save_report(args[1])

    if sub == "export_html":
        if len(args) < 2:
            print("Usage: ci_recon.py export_html <slug>", file=sys.stderr)
            return 1
        return cmd_export_html(args[1])

    if sub == "export_pdf":
        if len(args) < 2:
            print("Usage: ci_recon.py export_pdf <slug>", file=sys.stderr)
            return 1
        return cmd_export_pdf(args[1])

    if sub == "export_csv":
        if len(args) < 2:
            print("Usage: ci_recon.py export_csv <slug>", file=sys.stderr)
            return 1
        return cmd_export_csv(args[1])

    print(f"Unknown subcommand: {sub!r}", file=sys.stderr)
    print("Available: validate, save_report, export_html, export_pdf, export_csv", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
