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
    ~/.hermes/competitive-intelligence/<slug>/<ISO-date>/.
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


def _find_latest_report_dir(slug: str) -> Optional[Path]:
    base = _hermes_home() / "competitive-intelligence" / slug
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

    from urllib.parse import urlparse
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
        allowed = rp.can_fetch(_USER_AGENT, url) or rp.can_fetch("*", url)
        result["robots_allows_crawl"] = bool(allowed)
    except Exception:  # noqa: BLE001
        result["robots_allows_crawl"] = True

    print(json.dumps(result, indent=2))
    return 0 if result["robots_allows_crawl"] else 1


# ---------------------------------------------------------------------------
# save_report
# ---------------------------------------------------------------------------

def cmd_save_report(slug: str) -> int:
    report_md = sys.stdin.read()
    if not report_md.strip():
        print("ERROR: no report content received on stdin", file=sys.stderr)
        return 1

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = _hermes_home() / "competitive-intelligence" / slug / date_str
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
<title>Competitive Intelligence — {slug}</title>
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
  <div id="brand">⚡ Hermès CI</div>
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
<noscript><pre>{escaped_md}</pre></noscript>
<script>
const src = {json_md};
document.getElementById('report').innerHTML = marked.parse(src);

// Build sidebar TOC
const toc = document.getElementById('toc');
document.querySelectorAll('#report h2, #report h3').forEach((h, i) => {{
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
document.querySelectorAll('#report p, #report li, #report td').forEach(el => {{
  el.innerHTML = el.innerHTML
    .replace(/\[H\]/g, '<span class="badge high" title="High confidence — direct evidence">H</span>')
    .replace(/\[M\]/g, '<span class="badge med"  title="Medium confidence — indirect signals">M</span>')
    .replace(/\[L\]/g, '<span class="badge low"  title="Low confidence — inference only">L</span>');
}});

// Collapsible H2 sections
document.querySelectorAll('#report h2').forEach(h2 => {{
  h2.addEventListener('click', () => {{
    const collapsed = h2.classList.toggle('collapsed');
    let el = h2.nextElementSibling;
    while (el && el.tagName !== 'H2') {{
      el.style.display = collapsed ? 'none' : '';
      el = el.nextElementSibling;
    }}
  }});
}});

// Smooth scroll for TOC links
document.querySelectorAll('#toc a').forEach(a => {{
  a.addEventListener('click', e => {{
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
    out_dir = _find_latest_report_dir(slug)
    if not out_dir:
        print(f"ERROR: no saved report found for slug {slug!r}", file=sys.stderr)
        return 1

    content = (out_dir / "report.md").read_text(encoding="utf-8")
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    rendered = _HTML_TEMPLATE.format(
        slug=slug,
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

def cmd_export_pdf(slug: str) -> int:
    out_dir = _find_latest_report_dir(slug)
    if not out_dir:
        print(f"ERROR: no saved report found for slug {slug!r}", file=sys.stderr)
        return 1

    md_path = out_dir / "report.md"
    pdf_path = out_dir / "report.pdf"

    # Ensure HTML artifact exists for wkhtmltopdf fallback
    html_path = out_dir / "report.html"
    if not html_path.exists():
        cmd_export_html(slug)

    # 1. Try wkhtmltopdf (HTML → PDF, no LaTeX dependency)
    if shutil.which("wkhtmltopdf"):
        r = subprocess.run(
            ["wkhtmltopdf", "--quiet", str(html_path), str(pdf_path)],
            capture_output=True,
        )
        if r.returncode == 0 and pdf_path.exists():
            print(str(pdf_path))
            return 0

    # 2. Try pandoc (markdown → PDF)
    if shutil.which("pandoc"):
        # Try with wkhtmltopdf engine first (no LaTeX needed)
        r = subprocess.run(
            ["pandoc", str(md_path), "-o", str(pdf_path), "--pdf-engine=wkhtmltopdf"],
            capture_output=True,
        )
        if r.returncode == 0 and pdf_path.exists():
            print(str(pdf_path))
            return 0
        # Try default engine (may use pdflatex/xelatex if installed)
        r = subprocess.run(
            ["pandoc", str(md_path), "-o", str(pdf_path)],
            capture_output=True,
        )
        if r.returncode == 0 and pdf_path.exists():
            print(str(pdf_path))
            return 0

    # 3. Try weasyprint (Python package — may or may not be installed)
    if shutil.which("weasyprint"):
        r = subprocess.run(
            ["weasyprint", str(html_path), str(pdf_path)],
            capture_output=True,
        )
        if r.returncode == 0 and pdf_path.exists():
            print(str(pdf_path))
            return 0

    # 4. Fallback — return the HTML and advise browser print
    print(
        f"WARNING: No PDF tool found (tried wkhtmltopdf, pandoc, weasyprint).\n"
        f"HTML report is at: {html_path}\n"
        f"To create a PDF: open that file in a browser → File → Print → Save as PDF.",
        file=sys.stderr,
    )
    print(str(html_path))
    return 0  # HTML is a usable substitute; don't hard-fail


# ---------------------------------------------------------------------------
# export_csv
# ---------------------------------------------------------------------------

def _parse_markdown_tables(content: str) -> list[tuple[str, list[list[str]]]]:
    """Return (heading_slug, rows) for every markdown table in content."""
    results: list[tuple[str, list[list[str]]]] = []
    lines = content.split("\n")
    current_heading = "table"
    heading_count: dict[str, int] = {}
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Track nearest preceding heading for naming
        if re.match(r"^#{1,4}\s", stripped):
            text = re.sub(r"^#+\s*", "", stripped)
            slug = re.sub(r"[^\w\s-]", "", text).strip().lower()
            slug = re.sub(r"[\s_]+", "_", slug)[:40].rstrip("_")
            current_heading = slug or "table"

        # Detect table start: a line of cells followed by a separator line
        if stripped.startswith("|") and i + 1 < len(lines):
            sep = lines[i + 1].strip()
            if re.match(r"^\|[-|: ]+\|$", sep):
                rows: list[list[str]] = []
                j = i
                while j < len(lines) and lines[j].strip().startswith("|"):
                    row_stripped = lines[j].strip()
                    if re.match(r"^\|[-|: ]+\|$", row_stripped):
                        j += 1
                        continue  # skip separator row
                    cells = [c.strip() for c in row_stripped.split("|")[1:-1]]
                    rows.append(cells)
                    j += 1

                if len(rows) >= 2:  # header + at least one data row
                    count = heading_count.get(current_heading, 0)
                    heading_count[current_heading] = count + 1
                    name = current_heading if count == 0 else f"{current_heading}_{count + 1}"
                    results.append((name, rows))
                i = j
                continue

        i += 1

    return results


def cmd_export_csv(slug: str) -> int:
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
