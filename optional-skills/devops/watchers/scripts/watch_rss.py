#!/usr/bin/env python3
"""Watch an RSS 2.0 or Atom feed; print new items to stdout, silent on empty.

Usage (via cron with --no-agent):

    hermes cron create my-feed \\
      --schedule "*/15 * * * *" --no-agent \\
      --script "$HERMES_HOME/skills/devops/watchers/scripts/watch_rss.py" \\
      --script-args "--name hn --url https://news.ycombinator.com/rss"

First run records a baseline (emits nothing).  Subsequent runs emit only
items whose <guid> / <id> isn't in the watermark.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).parent))
from _watermark import Watermark, format_items_as_markdown  # type: ignore


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def _parse_feed(xml_bytes: bytes):
    """Return a list of {id, title, url, summary} dicts.

    Handles both RSS 2.0 ``<item>`` and Atom ``<entry>``.
    """
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        print(f"watch_rss: invalid XML: {e}", file=sys.stderr)
        sys.exit(2)

    entries = []
    for item in root.iter():
        tag = _strip_ns(item.tag)
        if tag not in ("item", "entry"):
            continue
        # ElementTree Elements without children are *falsy* — use `is not None`.
        children = {_strip_ns(c.tag): c for c in item}

        guid_el = children.get("guid")
        if guid_el is None:
            guid_el = children.get("id")
        link_el = children.get("link")
        if link_el is not None:
            href = link_el.attrib.get("href") or (link_el.text or "").strip()
        else:
            href = ""
        guid = (guid_el.text or "").strip() if guid_el is not None else ""
        guid = guid or href
        if not guid:
            continue

        title_el = children.get("title")
        title = (title_el.text or "").strip() if title_el is not None else ""

        summ_el = children.get("description")
        if summ_el is None:
            summ_el = children.get("summary")
        summary = (summ_el.text or "").strip() if summ_el is not None else ""

        entries.append(
            {"id": guid, "title": title, "url": href, "summary": summary}
        )
    return entries


def main() -> int:
    p = argparse.ArgumentParser(description="Watch an RSS/Atom feed.")
    p.add_argument("--name", required=True, help="Watcher name (used for state file)")
    p.add_argument("--url", required=True, help="Feed URL")
    p.add_argument("--max", type=int, default=10,
                   help="Max new items to emit per tick (default: 10)")
    p.add_argument("--with-summary", action="store_true",
                   help="Include <description>/<summary> snippet under each item")
    p.add_argument("--timeout", type=float, default=20.0,
                   help="HTTP timeout in seconds (default: 20)")
    args = p.parse_args()

    try:
        req = urllib.request.Request(args.url, headers={"User-Agent": "Hermes-Watcher/1.0"})
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            xml_bytes = resp.read()
    except urllib.error.HTTPError as e:
        print(f"watch_rss: HTTP {e.code} from {args.url}", file=sys.stderr)
        return 2
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        print(f"watch_rss: network error: {e}", file=sys.stderr)
        return 2

    entries = _parse_feed(xml_bytes)

    wm = Watermark.load(args.name)
    new_items = wm.filter_new(entries, id_key="id")
    wm.save()

    # Cap emitted items (watermark still records all seen IDs so we don't
    # re-emit them next tick).
    if args.max > 0:
        new_items = new_items[: args.max]

    body_key = "summary" if args.with_summary else None
    output = format_items_as_markdown(new_items, body_key=body_key)
    if output:
        sys.stdout.write(output)
    # Empty stdout on no-new — cron treats that as silent.
    return 0


if __name__ == "__main__":
    sys.exit(main())
