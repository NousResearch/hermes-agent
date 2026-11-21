#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
"""recon_kit — deterministic helpers for bmad-deep-recon.

The mechanical half of the research workflow: everything here is exact,
repeatable work the LLM should never re-derive by hand. All subcommands
print one JSON object to stdout; diagnostics go to stderr. Exit codes:
0 = pass, 1 = findings that need attention, 2 = usage/parse error.

Subcommands:
  citations RESEARCH_MD
      Cross-check inline [n] markers against the source-appendix table:
      dangling markers (no appendix row) and orphaned rows (never cited).
  tally MEMLOG_MD
      Count memlog entries by type, and claim entries by status.
      Claim lines carry `status=<word>` and optionally `ref=[n]`; for a
      given ref the LAST status wins, so status changes are appends.
  staleness CLAIMS_JSON --windows JSON [--today YYYY-MM-DD]
      Given claims [{claim, class, pub_date}] and a months-per-class map
      (e.g. '{"size/growth": 18, "pricing": 3}'), compute each claim's
      re-check date, flag stale ones, and report the earliest re-check.
  slug TOPIC --type TYPE [--pattern P] [--date YYYY-MM-DD]
      Expand the run-folder pattern deterministically so the same topic
      always lands in the same folder across draft -> process -> refresh.
  escape-sources RESEARCH_MD
      Emit the source-appendix table as HTML with every cell escaped and
      only validated http(s) URLs turned into links, for the briefing.
"""
from __future__ import annotations

import argparse
import calendar
import html
import json
import re
import sys
import unicodedata
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urlparse

MARKER_RE = re.compile(r"\[(\d+)\](?!\()")  # [3] but not a [3](url) link
MD_LINK_RE = re.compile(r"\[([^\]]*)\]\((\S+?)\)")
BARE_URL_RE = re.compile(r"https?://[^\s|)\]]+")


def out(payload: dict, exit_code: int) -> int:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return exit_code


def read_text(path_arg: str) -> str:
    if path_arg == "-":
        return sys.stdin.read()
    return Path(path_arg).read_text(encoding="utf-8")


def strip_fences(text: str) -> str:
    """Blank out fenced code blocks so their contents never count as markers or rows."""
    lines, fenced = [], False
    for ln in text.splitlines():
        if ln.lstrip().startswith("```"):
            fenced = not fenced
            lines.append("")
            continue
        lines.append("" if fenced else ln)
    return "\n".join(lines)


def table_cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def appendix_rows(text: str) -> dict[int, list[str]]:
    """Source-appendix rows: markdown table rows whose first cell is a bare [n] / n."""
    rows: dict[int, list[str]] = {}
    for ln in text.splitlines():
        stripped = ln.strip()
        if not stripped.startswith("|"):
            continue
        cells = table_cells(stripped)
        if not cells or len(cells) < 2:
            continue
        m = re.fullmatch(r"\[?(\d+)\]?", cells[0])
        if m:
            rows[int(m.group(1))] = cells
    return rows


# --- citations ---------------------------------------------------------------

def cmd_citations(args) -> int:
    text = strip_fences(read_text(args.file))
    rows = appendix_rows(text)
    markers: set[int] = set()
    for ln in text.splitlines():
        stripped = ln.strip()
        if stripped.startswith("|"):
            cells = table_cells(stripped)
            if cells and re.fullmatch(r"\[?(\d+)\]?", cells[0]):
                continue  # an appendix row is not a citation of itself
        markers.update(int(n) for n in MARKER_RE.findall(ln))
    dangling = sorted(markers - set(rows))
    orphaned = sorted(set(rows) - markers)
    ok = not dangling and not orphaned
    return out({
        "markers": sorted(markers),
        "appendix_rows": sorted(rows),
        "dangling_markers": dangling,
        "orphaned_rows": orphaned,
        "ok": ok,
    }, 0 if ok else 1)


# --- tally -------------------------------------------------------------------

ENTRY_RE = re.compile(r"^- (?:\(([\w-]+)(?: by [^)]*)?\)\s*)?(.*)$")


def cmd_tally(args) -> int:
    text = read_text(args.file)
    body = text.split("---", 2)[-1] if text.startswith("---") else text
    by_type: dict[str, int] = {}
    by_ref: dict[int, str] = {}
    unref_status: dict[str, int] = {}
    entries = 0
    for ln in body.splitlines():
        m = ENTRY_RE.match(ln)
        if not m or not ln.startswith("- "):
            continue
        entries += 1
        etype = m.group(1) or "note"
        by_type[etype] = by_type.get(etype, 0) + 1
        if etype == "claim":
            status_m = re.search(r"status=([\w-]+)", m.group(2))
            status = status_m.group(1) if status_m else "unknown"
            ref_m = re.search(r"ref=\[?(\d+)\]?", m.group(2))
            if ref_m:
                by_ref[int(ref_m.group(1))] = status  # last status wins per ref
            else:
                unref_status[status] = unref_status.get(status, 0) + 1
    claims: dict[str, int] = dict(unref_status)
    for status in by_ref.values():
        claims[status] = claims.get(status, 0) + 1
    return out({
        "entries": entries,
        "by_type": dict(sorted(by_type.items())),
        "claims": dict(sorted(claims.items())),
        "claims_total": sum(claims.values()),
    }, 0)


# --- staleness ---------------------------------------------------------------

def parse_date(raw: str) -> date:
    raw = raw.strip()
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unparseable date: {raw!r} (want YYYY[-MM[-DD]])")


def add_months(d: date, months: int) -> date:
    total = d.month - 1 + months
    year, month = d.year + total // 12, total % 12 + 1
    return date(year, month, min(d.day, calendar.monthrange(year, month)[1]))


def cmd_staleness(args) -> int:
    try:
        payload = json.loads(read_text(args.file))
        windows = {k.lower(): int(v) for k, v in json.loads(args.windows).items()}
        today = parse_date(args.today) if args.today else date.today()
    except (ValueError, json.JSONDecodeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    claims = payload["claims"] if isinstance(payload, dict) else payload
    results, no_window, stale_count = [], set(), 0
    earliest: date | None = None
    for c in claims:
        cls = str(c.get("class", "")).lower()
        try:
            pub = parse_date(str(c["pub_date"]))
        except (KeyError, ValueError) as e:
            print(f"error in claim {c!r}: {e}", file=sys.stderr)
            return 2
        months = windows.get(cls)
        if months is None:
            no_window.add(cls)
            results.append({**c, "recheck": None, "stale": None})
            continue
        recheck = add_months(pub, months)
        stale = recheck <= today
        stale_count += stale
        earliest = recheck if earliest is None or recheck < earliest else earliest
        results.append({**c, "recheck": recheck.isoformat(), "stale": stale})
    return out({
        "today": today.isoformat(),
        "claims": results,
        "stale_count": stale_count,
        "earliest_recheck": earliest.isoformat() if earliest else None,
        "no_window_classes": sorted(no_window),
    }, 1 if stale_count else 0)


# --- slug --------------------------------------------------------------------

def slugify(text: str, max_len: int = 40) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return re.sub(r"-{2,}", "-", text)[:max_len].rstrip("-")


def cmd_slug(args) -> int:
    slug = slugify(args.topic)
    if not slug:
        print("error: topic slugified to an empty string", file=sys.stderr)
        return 2
    folder = (args.pattern
              .replace("{research_type}", args.type)
              .replace("{topic_slug}", slug)
              .replace("{date}", args.date or date.today().isoformat()))
    return out({"topic_slug": slug, "folder": folder}, 0)


# --- escape-sources ----------------------------------------------------------

def safe_url(raw: str) -> str | None:
    parsed = urlparse(raw)
    return raw if parsed.scheme in ("http", "https") and parsed.netloc else None


def cell_html(cell: str, invalid: list[str]) -> str:
    """Escape a cell; a markdown link or bare URL becomes an <a> only when http(s)."""
    link = MD_LINK_RE.search(cell)
    if link:
        url = safe_url(link.group(2))
        label = html.escape(link.group(1) or link.group(2))
        if url:
            return html.escape(cell[:link.start()]) + \
                f'<a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener">{label}</a>' + \
                html.escape(cell[link.end():])
        invalid.append(link.group(2))
        return html.escape(cell.replace(link.group(0), link.group(1) or link.group(2)))
    bare = BARE_URL_RE.search(cell)
    if bare:
        url = safe_url(bare.group(0))
        if url:
            escaped = html.escape(url, quote=True)
            return html.escape(cell[:bare.start()]) + \
                f'<a href="{escaped}" target="_blank" rel="noopener">{escaped}</a>' + \
                html.escape(cell[bare.end():])
        invalid.append(bare.group(0))
    return html.escape(cell)


def cmd_escape_sources(args) -> int:
    text = strip_fences(read_text(args.file))
    rows = appendix_rows(text)
    if not rows:
        print("error: no source-appendix table rows found", file=sys.stderr)
        return 2
    invalid: list[str] = []
    body_rows = []
    for n in sorted(rows):
        cells = rows[n]
        tds = "".join(f"<td>{cell_html(c, invalid)}</td>" for c in cells[1:])
        body_rows.append(f'<tr id="src-{n}"><td>[{n}]</td>{tds}</tr>')
    table = ('<table class="sources"><tbody>' + "".join(body_rows) + "</tbody></table>")
    return out({"rows": len(rows), "invalid_urls": invalid, "html": table},
               1 if invalid else 0)


# --- entry point -------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("citations", help="cross-check [n] markers vs the source appendix")
    pc.add_argument("file", help="path to research.md (or - for stdin)")
    pc.set_defaults(func=cmd_citations)

    pt = sub.add_parser("tally", help="count memlog entries by type and claims by status")
    pt.add_argument("file", help="path to .memlog.md (or - for stdin)")
    pt.set_defaults(func=cmd_tally)

    ps = sub.add_parser("staleness", help="compute re-check dates from freshness windows")
    ps.add_argument("file", help="claims JSON: [{claim, class, pub_date}] (or - for stdin)")
    ps.add_argument("--windows", required=True,
                    help='JSON months-per-class map, e.g. \'{"pricing": 3}\'')
    ps.add_argument("--today", help="override today's date (YYYY-MM-DD)")
    ps.set_defaults(func=cmd_staleness)

    pg = sub.add_parser("slug", help="expand the run-folder pattern deterministically")
    pg.add_argument("topic", help="research topic text")
    pg.add_argument("--type", required=True, help="research type code (e.g. market)")
    pg.add_argument("--pattern", default="{research_type}-{topic_slug}-{date}",
                    help="folder pattern (default: {research_type}-{topic_slug}-{date})")
    pg.add_argument("--date", help="override date (YYYY-MM-DD; default today)")
    pg.set_defaults(func=cmd_slug)

    pe = sub.add_parser("escape-sources",
                        help="source appendix as escaped HTML with validated links")
    pe.add_argument("file", help="path to research.md (or - for stdin)")
    pe.set_defaults(func=cmd_escape_sources)

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    if sys.platform == "win32":
        # Piped output on Windows defaults to a legacy code page, not UTF-8.
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    sys.exit(main())
