#!/usr/bin/env python3
"""Import public X/Twitter post details into a JSONL ledger via oEmbed.

This module deliberately uses only the public ``publish.twitter.com/oembed``
endpoint. It does not use credentials, browser automation, publishing APIs, or
media downloads. oEmbed does not provide trustworthy media-analysis data, so
every imported record explicitly reports
``media_analysis: not_available_via_oembed``.

Existing ledger fields are copied unchanged. Imported information is stored in
a dedicated ``x_oembed_import`` mapping (or a suffixed variant if that key is
already occupied), so the importer never replaces a ledger field.

Example:
    python3 content_engine/x_oembed_import.py \
        --input input-ledger.jsonl \
        --output enriched-ledger.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from html import unescape
from html.parser import HTMLParser
from http.client import HTTPException
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

OEMBED_ENDPOINT = "https://publish.twitter.com/oembed"
RETRIEVAL_METHOD = "public_publish_twitter_oembed"
MEDIA_ANALYSIS_UNAVAILABLE = "not_available_via_oembed"
DEFAULT_INTER_RECORD_DELAY = 0.5
DEFAULT_RETRY_BACKOFF = 0.5
_MAX_HTTP_ATTEMPTS = 2
_TRANSIENT_HTTP_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})

_COMMON_URL_FIELDS = ("tweet_url", "source_url", "x_url", "url")
_TCO_URL_PATTERN = re.compile(
    r'https?://t\.co(?=[/:?#\s"\'<>]|$)[^\s"\'<>]*',
    re.IGNORECASE,
)

HttpGet = Callable[[str, float], bytes]
Clock = Callable[[], datetime]
Sleeper = Callable[[float], None]


class OEmbedImportError(ValueError):
    """Raised when a public oEmbed response cannot produce a ledger record."""


@dataclass(frozen=True)
class ImportSummary:
    """Counts returned after importing one JSONL ledger."""

    records_total: int
    records_succeeded: int
    records_failed: int


def _normalise_paragraph_text(value: str) -> str:
    """Normalise inline whitespace while retaining explicit and source newlines."""
    normalised = value.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(
        " ".join(line.split()) for line in normalised.split("\n")
    ).strip()


class _TweetHTMLParser(HTMLParser):
    """Extract tweet paragraph text and t.co links from Twitter's oEmbed HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._inside_paragraph = False
        self._paragraph_parts: list[list[str]] = []
        self.tco_urls: list[str] = []

    @property
    def text(self) -> str:
        paragraphs: list[str] = []
        for parts in self._paragraph_parts:
            paragraph = _normalise_paragraph_text("".join(parts))
            if paragraph:
                paragraphs.append(paragraph)
        return "\n\n".join(paragraphs)

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        normalised_tag = tag.lower()

        if normalised_tag == "p":
            # A new <p> implicitly starts a new paragraph even if malformed
            # markup omitted a prior closing tag.
            self._paragraph_parts.append([])
            self._inside_paragraph = True
            return

        if normalised_tag == "br":
            if self._inside_paragraph and self._paragraph_parts:
                self._paragraph_parts[-1].append("\n")
            return

        if normalised_tag != "a":
            return

        for name, value in attrs:
            if name.lower() == "href" and value and _is_tco_url(value):
                self.tco_urls.append(value)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "p":
            self._inside_paragraph = False

    def handle_data(self, data: str) -> None:
        if self._inside_paragraph and self._paragraph_parts:
            self._paragraph_parts[-1].append(data)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _timestamp(clock: Clock) -> str:
    value = clock()
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def _nonnegative_seconds(value: float, *, label: str) -> float:
    try:
        seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite non-negative number") from exc

    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"{label} must be a finite non-negative number")
    return seconds


def _is_tco_url(value: str) -> bool:
    parsed = urlsplit(value)
    return (
        parsed.scheme in {"http", "https"}
        and (parsed.hostname or "").lower() == "t.co"
    )


def _deduplicate(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def _tco_urls_from_html(html: str, parser_urls: list[str]) -> list[str]:
    """Return t.co URLs in their first-seen order from links and rendered text."""
    urls = list(parser_urls)
    for match in _TCO_URL_PATTERN.finditer(unescape(html)):
        candidate = match.group(0).rstrip(".,;:!?)]}")
        if _is_tco_url(candidate):
            urls.append(candidate)
    return _deduplicate(urls)


def _author_handle(author_url: str | None) -> str | None:
    if not author_url:
        return None

    path_parts = [part for part in urlsplit(author_url).path.split("/") if part]
    if not path_parts:
        return None
    return path_parts[0]


def build_oembed_url(tweet_url: str) -> str:
    """Build the only public endpoint URL used by this importer."""
    return (
        f"{OEMBED_ENDPOINT}?"
        f"{urlencode({'url': tweet_url, 'omit_script': 'true', 'dnt': 'true'})}"
    )


def _default_http_get(url: str, timeout: float) -> bytes:
    """Fetch a public oEmbed response using the Python standard library."""
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "KenseiAgent-XOEmbedImporter/1.0",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def _validate_tweet_url(value: str) -> str:
    parsed = urlsplit(value)
    hostname = (parsed.hostname or "").lower()
    valid_hosts = {
        "x.com",
        "www.x.com",
        "twitter.com",
        "www.twitter.com",
        "mobile.twitter.com",
    }

    if parsed.scheme not in {"http", "https"} or hostname not in valid_hosts:
        raise OEmbedImportError(
            "source URL must be an http(s) X/Twitter status URL"
        )

    path_parts = [part for part in parsed.path.split("/") if part]
    is_user_status = len(path_parts) >= 3 and path_parts[1] == "status"
    is_web_status = (
        len(path_parts) >= 4
        and path_parts[0] == "i"
        and path_parts[1] == "web"
        and path_parts[2] == "status"
    )
    if not (is_user_status or is_web_status):
        raise OEmbedImportError(
            "source URL must identify an X/Twitter status post"
        )

    return value


def _source_url(record: Mapping[str, Any], url_field: str | None) -> str:
    fields = (url_field,) if url_field else _COMMON_URL_FIELDS

    for field in fields:
        if not field:
            continue
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return _validate_tweet_url(value.strip())

    label = f"'{url_field}'" if url_field else ", ".join(
        f"'{field}'" for field in _COMMON_URL_FIELDS
    )
    raise OEmbedImportError(f"record has no usable X/Twitter URL in {label}")


def parse_oembed_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Extract public text, author details, and t.co URLs from an oEmbed payload."""
    html = payload.get("html")
    if not isinstance(html, str) or not html.strip():
        raise OEmbedImportError("oEmbed response did not contain HTML")

    parser = _TweetHTMLParser()
    parser.feed(html)
    parser.close()

    text = parser.text
    if not text:
        raise OEmbedImportError(
            "oEmbed response did not contain tweet text paragraphs"
        )

    author_name = payload.get("author_name")
    if author_name is not None and not isinstance(author_name, str):
        raise OEmbedImportError("oEmbed response author_name was not a string")

    author_url = payload.get("author_url")
    if author_url is not None and not isinstance(author_url, str):
        raise OEmbedImportError("oEmbed response author_url was not a string")

    return {
        "text": text,
        "author": author_name.strip() if isinstance(author_name, str) else None,
        "author_url": author_url,
        "author_handle": _author_handle(author_url),
        "tco_urls": _tco_urls_from_html(html, parser.tco_urls),
        "oembed_provider_name": payload.get("provider_name"),
        "oembed_provider_url": payload.get("provider_url"),
        "oembed_version": payload.get("version"),
    }


def _is_transient_http_error(exc: OSError | HTTPException) -> bool:
    """Return whether an HTTP/network error is worth the sole retry."""
    if isinstance(exc, HTTPError):
        return exc.code in _TRANSIENT_HTTP_STATUS_CODES
    return True


def fetch_oembed(
    tweet_url: str,
    *,
    http_get: HttpGet = _default_http_get,
    timeout: float = 15.0,
    sleeper: Sleeper = time.sleep,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
) -> dict[str, Any]:
    """Retrieve and parse one public X/Twitter oEmbed response.

    A transient HTTP/network failure receives one bounded retry. Validation and
    oEmbed parse failures occur outside the retry block and are never retried.
    """
    retry_backoff = _nonnegative_seconds(
        retry_backoff,
        label="retry_backoff",
    )

    response_body: bytes
    for attempt in range(_MAX_HTTP_ATTEMPTS):
        try:
            response_body = http_get(build_oembed_url(tweet_url), timeout)
            break
        except (OSError, HTTPException) as exc:
            if (
                attempt == _MAX_HTTP_ATTEMPTS - 1
                or not _is_transient_http_error(exc)
            ):
                raise

            delay = retry_backoff * (2**attempt)
            if delay:
                sleeper(delay)
    else:  # Defensive: the loop always breaks or raises.
        raise RuntimeError("oEmbed request attempts unexpectedly exhausted")

    if not isinstance(response_body, bytes):
        raise OEmbedImportError("HTTP getter must return response bytes")

    try:
        payload = json.loads(response_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OEmbedImportError(f"invalid oEmbed JSON: {exc}") from exc

    if not isinstance(payload, Mapping):
        raise OEmbedImportError("oEmbed response must be a JSON object")

    return parse_oembed_payload(payload)


def _base_metadata(timestamp: str) -> dict[str, Any]:
    return {
        "retrieved_at": timestamp,
        "retrieval_method": RETRIEVAL_METHOD,
        "media_analysis": MEDIA_ANALYSIS_UNAVAILABLE,
    }


def _error_metadata(timestamp: str, exc: Exception) -> dict[str, Any]:
    metadata = _base_metadata(timestamp)
    metadata.update(
        {
            "status": "error",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
    )
    return metadata


def _available_result_key(record: Mapping[str, Any]) -> str:
    key = "x_oembed_import"
    suffix = 2
    while key in record:
        key = f"x_oembed_import_{suffix}"
        suffix += 1
    return key


def enrich_ledger_record(
    record: Mapping[str, Any],
    *,
    http_get: HttpGet = _default_http_get,
    clock: Clock = _utc_now,
    timeout: float = 15.0,
    url_field: str | None = None,
    sleeper: Sleeper = time.sleep,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
) -> tuple[dict[str, Any], bool]:
    """Copy and enrich one ledger record without allowing its failure to escape."""
    output = dict(record)
    result_key = _available_result_key(output)
    timestamp = _timestamp(clock)

    try:
        tweet_url = _source_url(record, url_field)
        parsed = fetch_oembed(
            tweet_url,
            http_get=http_get,
            timeout=timeout,
            sleeper=sleeper,
            retry_backoff=retry_backoff,
        )
        metadata = _base_metadata(timestamp)
        metadata.update(
            {
                "status": "ok",
                "source_url": tweet_url,
                **parsed,
            }
        )
        output[result_key] = metadata
        return output, True
    except Exception as exc:  # Per-record isolation is intentional.
        output[result_key] = _error_metadata(timestamp, exc)
        return output, False


def _invalid_jsonl_record(
    raw_line: str,
    line_number: int,
    *,
    clock: Clock,
    exc: Exception,
) -> dict[str, Any]:
    """Represent malformed JSONL as an isolated, auditable output record."""
    metadata = _error_metadata(_timestamp(clock), exc)
    metadata["line_number"] = line_number
    return {
        "raw_jsonl": raw_line.rstrip("\r\n"),
        "x_oembed_import": metadata,
    }


def import_jsonl(
    input_path: Path,
    output_path: Path,
    *,
    http_get: HttpGet = _default_http_get,
    clock: Clock = _utc_now,
    timeout: float = 15.0,
    url_field: str | None = None,
    inter_record_delay: float = DEFAULT_INTER_RECORD_DELAY,
    sleeper: Sleeper = time.sleep,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
) -> ImportSummary:
    """Read a JSONL ledger and write one enriched JSON object per input record."""
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output paths must be different")

    inter_record_delay = _nonnegative_seconds(
        inter_record_delay,
        label="inter_record_delay",
    )
    retry_backoff = _nonnegative_seconds(
        retry_backoff,
        label="retry_backoff",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    succeeded = 0
    failed = 0
    first_record = True

    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as destination:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue

            # Sleeping before each record after the first guarantees that delay
            # occurs only between records and never after the final record.
            if not first_record and inter_record_delay:
                sleeper(inter_record_delay)
            first_record = False

            total += 1
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                result = _invalid_jsonl_record(
                    raw_line,
                    line_number,
                    clock=clock,
                    exc=exc,
                )
                failed += 1
            else:
                if not isinstance(record, Mapping):
                    result = _invalid_jsonl_record(
                        raw_line,
                        line_number,
                        clock=clock,
                        exc=OEmbedImportError("JSONL record must be an object"),
                    )
                    failed += 1
                else:
                    result, imported = enrich_ledger_record(
                        record,
                        http_get=http_get,
                        clock=clock,
                        timeout=timeout,
                        url_field=url_field,
                        sleeper=sleeper,
                        retry_backoff=retry_backoff,
                    )
                    if imported:
                        succeeded += 1
                    else:
                        failed += 1

            destination.write(json.dumps(result, ensure_ascii=False, sort_keys=True))
            destination.write("\n")

    return ImportSummary(
        records_total=total,
        records_succeeded=succeeded,
        records_failed=failed,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Enrich a JSONL X/Twitter ledger through the public "
            "publish.twitter.com/oembed endpoint."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Source JSONL ledger path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSONL path; must differ from --input.",
    )
    parser.add_argument(
        "--url-field",
        default=None,
        help=(
            "Explicit ledger field containing the X/Twitter post URL. "
            "Defaults to trying tweet_url, source_url, x_url, and url."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="Per-record public oEmbed HTTP timeout in seconds (default: 15).",
    )
    parser.add_argument(
        "--inter-record-delay",
        type=float,
        default=DEFAULT_INTER_RECORD_DELAY,
        help=(
            "Seconds to wait between non-empty JSONL records "
            f"(default: {DEFAULT_INTER_RECORD_DELAY})."
        ),
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    http_get: HttpGet = _default_http_get,
    clock: Clock = _utc_now,
    sleeper: Sleeper = time.sleep,
) -> int:
    """Run the JSONL importer CLI and emit a JSON summary to stdout."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    if (
        not math.isfinite(args.inter_record_delay)
        or args.inter_record_delay < 0
    ):
        parser.error("--inter-record-delay must be a finite non-negative number")

    try:
        summary = import_jsonl(
            args.input,
            args.output,
            http_get=http_get,
            clock=clock,
            timeout=args.timeout,
            url_field=args.url_field,
            inter_record_delay=args.inter_record_delay,
            sleeper=sleeper,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(json.dumps(asdict(summary), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
