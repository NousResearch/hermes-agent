"""Emit stable, primary-source BAC filing and OCC enforcement snapshots for Hermes cron."""

from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


BAC_FILINGS_URL = "https://investor.bankofamerica.com/regulatory-and-other-filings"
OCC_NEWS_URL = "https://www.occ.gov/rss/occ_news.xml"
FINCEN_ACTIONS_URL = "https://www.fincen.gov/news/enforcement-actions"
_MAX_RESPONSE_BYTES = 2_000_000
_TIMEOUT_SECONDS = 20
_TARGET_HEADINGS = {
    "latest current report",
    "latest quarterly or annual filing",
}


class _BACFilingsParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.section: str | None = None
        self._heading: list[str] | None = None
        self._form: list[str] | None = None
        self.records: dict[str, dict[str, str]] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "h2":
            self.section = None
            self._heading = []
        elif tag == "h3" and self.section:
            self._form = []
        elif tag == "time" and self.section:
            published = values.get("datetime")
            if published:
                self.records[self.section]["published"] = published[:10]
        elif tag == "a" and self.section:
            href = values.get("href", "")
            if (
                "url" not in self.records[self.section]
                and "/select-sec-filings/content/" in href
                and href.lower().endswith(".htm")
            ):
                self.records[self.section]["url"] = urljoin(BAC_FILINGS_URL, href)

    def handle_data(self, data: str) -> None:
        if self._heading is not None:
            self._heading.append(data)
        if self._form is not None:
            self._form.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "h2" and self._heading is not None:
            heading = " ".join("".join(self._heading).split()).casefold()
            self._heading = None
            if heading in _TARGET_HEADINGS:
                self.section = heading
                self.records[heading] = {}
        elif tag == "h3" and self._form is not None:
            form = " ".join("".join(self._form).split())
            self._form = None
            if self.section and "form" not in self.records[self.section]:
                self.records[self.section]["form"] = form


class _FinCENActionsParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._row: list[dict[str, str]] | None = None
        self._cell: dict[str, str] | None = None
        self.actions: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = {"text": "", "url": ""}
        elif tag == "a" and self._cell is not None:
            self._cell["url"] = values.get("href", "")

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell["text"] += data

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._cell is not None and self._row is not None:
            self._cell["text"] = " ".join(self._cell["text"].split())
            self._row.append(self._cell)
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if len(self._row) >= 3 and "bank of america" in self._row[0]["text"].casefold():
                self.actions.append({
                    "source": "FinCEN enforcement actions",
                    "published": self._row[1]["text"],
                    "matter": self._row[2]["text"],
                    "title": self._row[0]["text"],
                    "url": urljoin(FINCEN_ACTIONS_URL, self._row[0]["url"]),
                })
            self._row = None


def parse_bac_filings(page: str) -> list[dict[str, str]]:
    parser = _BACFilingsParser()
    parser.feed(page)
    result = []
    for heading in ("latest current report", "latest quarterly or annual filing"):
        record = parser.records.get(heading, {})
        if not all(record.get(key) for key in ("published", "form", "url")):
            raise ValueError(f"BAC IR filing section is missing expected fields: {heading}")
        result.append({
            "source": "Bank of America IR / SEC filing",
            "published": record["published"],
            "form": record["form"],
            "url": record["url"],
        })
    return result


def parse_occ_news(feed: str) -> list[dict[str, str]]:
    root = ET.fromstring(feed)
    channel = root.find("channel")
    if channel is None:
        raise ValueError("OCC RSS feed has no channel")
    result = []
    for item in channel.findall("item"):
        title = " ".join((item.findtext("title") or "").split())
        description = " ".join(unescape(re.sub(r"<[^>]*>", " ", item.findtext("description") or "")).split())
        if not title or "bank of america" not in f"{title} {description}".casefold():
            continue
        link = (item.findtext("link") or "").strip()
        if urlparse(link).hostname not in {"occ.gov", "www.occ.gov"}:
            raise ValueError(f"OCC item has an unexpected source URL: {link!r}")
        result.append({
            "source": "Office of the Comptroller of the Currency",
            "published": " ".join((item.findtext("pubDate") or "").split()),
            "title": title,
            "description": description[:1200],
            "url": link,
        })
    return result


def parse_fincen_actions(page: str) -> list[dict[str, str]]:
    parser = _FinCENActionsParser()
    parser.feed(page)
    if not parser.actions:
        raise ValueError("FinCEN enforcement list has no Bank of America entry")
    for action in parser.actions:
        if urlparse(action["url"]).hostname not in {"fincen.gov", "www.fincen.gov"}:
            raise ValueError(f"FinCEN item has an unexpected source URL: {action['url']!r}")
    return parser.actions


def _fetch(url: str, allowed_hosts: set[str]) -> str:
    request = Request(url, headers={"User-Agent": "Hermes-BAC-CIRT-Monitor/1.0"})
    with urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
        host = urlparse(response.geturl()).hostname
        if host not in allowed_hosts:
            raise ValueError(f"Source redirected to an unexpected host: {host!r}")
        body = response.read(_MAX_RESPONSE_BYTES + 1)
    if len(body) > _MAX_RESPONSE_BYTES:
        raise ValueError(f"Source response exceeded {_MAX_RESPONSE_BYTES} bytes: {url}")
    return body.decode("utf-8-sig")


def collect() -> dict[str, object]:
    filings = parse_bac_filings(_fetch(BAC_FILINGS_URL, {"investor.bankofamerica.com"}))
    occ = parse_occ_news(_fetch(OCC_NEWS_URL, {"www.occ.gov", "occ.gov"}))
    fincen = parse_fincen_actions(_fetch(FINCEN_ACTIONS_URL, {"www.fincen.gov", "fincen.gov"}))
    return {
        "monitor_release": Path(__file__).resolve().parent.parent.name,
        "sources": [BAC_FILINGS_URL, OCC_NEWS_URL, FINCEN_ACTIONS_URL],
        "items": sorted(
            filings + occ + fincen,
            key=lambda item: (item["source"], item["published"], item.get("url", "")),
        ),
    }


def main() -> int:
    try:
        snapshot = collect()
    except (OSError, URLError, ET.ParseError, ValueError) as exc:
        print(f"BAC CIRT source check failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
