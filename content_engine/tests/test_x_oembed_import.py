"""Hermetic fixture-based tests for the public X oEmbed JSONL importer."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

import x_oembed_import as importer


@pytest.fixture
def fixed_clock():
    return lambda: datetime(2026, 7, 26, 12, 30, tzinfo=UTC)


@pytest.fixture
def oembed_payload() -> dict[str, str]:
    return {
        "author_name": "Ada Lovelace",
        "author_url": "https://twitter.com/ada",
        "provider_name": "Twitter",
        "provider_url": "https://twitter.com",
        "version": "1.0",
        "html": (
            '<blockquote class="twitter-tweet">'
            '<p lang="en" dir="ltr">'
            "Shipping the ledger importer today. "
            '<a href="https://t.co/Launch123">https://t.co/Launch123</a>'
            "</p>&mdash; Ada Lovelace (@ada) "
            '<a href="https://twitter.com/ada/status/1234567890">'
            "July 26, 2026"
            "</a></blockquote>"
        ),
    }


@pytest.fixture
def ledger_paths(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "source.jsonl", tmp_path / "enriched.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_import_jsonl_preserves_ledger_fields_and_extracts_public_oembed_data(
    ledger_paths: tuple[Path, Path],
    oembed_payload: dict[str, str],
    fixed_clock,
) -> None:
    source, destination = ledger_paths
    original = {
        "ledger_id": "candidate-17",
        "source_url": "https://x.com/ada/status/1234567890",
        "campaign": "build-in-public",
        "nested_provenance": {"origin": "manual-research", "rank": 3},
    }
    source.write_text(json.dumps(original) + "\n", encoding="utf-8")

    observed_requests: list[tuple[str, float]] = []

    def fake_http_get(url: str, timeout: float) -> bytes:
        observed_requests.append((url, timeout))
        return json.dumps(oembed_payload).encode("utf-8")

    summary = importer.import_jsonl(
        source,
        destination,
        http_get=fake_http_get,
        clock=fixed_clock,
        timeout=9.5,
    )

    assert summary == importer.ImportSummary(
        records_total=1,
        records_succeeded=1,
        records_failed=0,
    )
    assert len(observed_requests) == 1

    request_url, timeout = observed_requests[0]
    parsed_request = urlsplit(request_url)
    assert parsed_request.scheme == "https"
    assert parsed_request.netloc == "publish.twitter.com"
    assert parsed_request.path == "/oembed"
    assert parse_qs(parsed_request.query) == {
        "dnt": ["true"],
        "omit_script": ["true"],
        "url": ["https://x.com/ada/status/1234567890"],
    }
    assert timeout == 9.5

    result = _read_jsonl(destination)
    assert len(result) == 1
    assert result[0]["ledger_id"] == original["ledger_id"]
    assert result[0]["campaign"] == original["campaign"]
    assert result[0]["nested_provenance"] == original["nested_provenance"]

    metadata = result[0]["x_oembed_import"]
    assert metadata == {
        "author": "Ada Lovelace",
        "author_handle": "ada",
        "author_url": "https://twitter.com/ada",
        "media_analysis": "not_available_via_oembed",
        "oembed_provider_name": "Twitter",
        "oembed_provider_url": "https://twitter.com",
        "oembed_version": "1.0",
        "retrieval_method": "public_publish_twitter_oembed",
        "retrieved_at": "2026-07-26T12:30:00+00:00",
        "source_url": "https://x.com/ada/status/1234567890",
        "status": "ok",
        "tco_urls": ["https://t.co/Launch123"],
        "text": "Shipping the ledger importer today. https://t.co/Launch123",
    }


def test_import_jsonl_isolates_a_failed_record_and_continues(
    ledger_paths: tuple[Path, Path],
    oembed_payload: dict[str, str],
    fixed_clock,
) -> None:
    source, destination = ledger_paths
    records = [
        {
            "ledger_id": "unavailable",
            "source_url": "https://x.com/ada/status/111",
            "keep": "failed-record-fields-stay-present",
        },
        {
            "ledger_id": "available",
            "tweet_url": "https://twitter.com/ada/status/222",
            "keep": "successful-record-fields-stay-present",
        },
    ]
    source.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    def fake_http_get(url: str, timeout: float) -> bytes:
        requested_tweet_url = parse_qs(urlsplit(url).query)["url"][0]
        if requested_tweet_url.endswith("/111"):
            raise OSError("fixture endpoint unavailable")
        return json.dumps(oembed_payload).encode("utf-8")

    summary = importer.import_jsonl(
        source,
        destination,
        http_get=fake_http_get,
        clock=fixed_clock,
        sleeper=lambda _: None,
    )

    assert summary == importer.ImportSummary(
        records_total=2,
        records_succeeded=1,
        records_failed=1,
    )

    failed, succeeded = _read_jsonl(destination)

    assert failed["ledger_id"] == "unavailable"
    assert failed["keep"] == "failed-record-fields-stay-present"
    assert failed["x_oembed_import"]["status"] == "error"
    assert failed["x_oembed_import"]["error"] == {
        "type": "OSError",
        "message": "fixture endpoint unavailable",
    }
    assert failed["x_oembed_import"]["retrieval_method"] == (
        "public_publish_twitter_oembed"
    )
    assert failed["x_oembed_import"]["media_analysis"] == (
        "not_available_via_oembed"
    )

    assert succeeded["ledger_id"] == "available"
    assert succeeded["keep"] == "successful-record-fields-stay-present"
    assert succeeded["x_oembed_import"]["status"] == "ok"
    assert succeeded["x_oembed_import"]["text"].startswith(
        "Shipping the ledger importer today."
    )


def test_import_jsonl_uses_default_delay_only_between_nonempty_records(
    ledger_paths: tuple[Path, Path],
    oembed_payload: dict[str, str],
    fixed_clock,
) -> None:
    source, destination = ledger_paths
    records = [
        {"source_url": "https://x.com/ada/status/1"},
        {"source_url": "https://x.com/ada/status/2"},
        {"source_url": "https://x.com/ada/status/3"},
    ]
    source.write_text(
        json.dumps(records[0])
        + "\n\n"
        + json.dumps(records[1])
        + "\n"
        + json.dumps(records[2])
        + "\n",
        encoding="utf-8",
    )

    observed_sleeps: list[float] = []
    observed_requests: list[str] = []

    def fake_http_get(url: str, timeout: float) -> bytes:
        observed_requests.append(url)
        return json.dumps(oembed_payload).encode("utf-8")

    summary = importer.import_jsonl(
        source,
        destination,
        http_get=fake_http_get,
        clock=fixed_clock,
        sleeper=observed_sleeps.append,
    )

    assert summary == importer.ImportSummary(
        records_total=3,
        records_succeeded=3,
        records_failed=0,
    )
    assert len(observed_requests) == 3
    assert observed_sleeps == [
        importer.DEFAULT_INTER_RECORD_DELAY,
        importer.DEFAULT_INTER_RECORD_DELAY,
    ]


def test_fetch_oembed_retries_one_transient_network_error_with_backoff(
    oembed_payload: dict[str, str],
) -> None:
    attempts = 0
    observed_sleeps: list[float] = []

    def fake_http_get(url: str, timeout: float) -> bytes:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("temporary fixture outage")
        return json.dumps(oembed_payload).encode("utf-8")

    parsed = importer.fetch_oembed(
        "https://x.com/ada/status/123",
        http_get=fake_http_get,
        sleeper=observed_sleeps.append,
        retry_backoff=0.25,
    )

    assert attempts == 2
    assert observed_sleeps == [0.25]
    assert parsed["text"] == "Shipping the ledger importer today. https://t.co/Launch123"


def test_fetch_oembed_does_not_retry_parse_errors() -> None:
    requests = 0
    observed_sleeps: list[float] = []

    def fake_http_get(url: str, timeout: float) -> bytes:
        nonlocal requests
        requests += 1
        return b"{not valid json"

    with pytest.raises(importer.OEmbedImportError, match="invalid oEmbed JSON"):
        importer.fetch_oembed(
            "https://x.com/ada/status/123",
            http_get=fake_http_get,
            sleeper=observed_sleeps.append,
        )

    assert requests == 1
    assert observed_sleeps == []


def test_parse_oembed_payload_preserves_paragraphs_breaks_and_source_newlines() -> None:
    parsed = importer.parse_oembed_payload(
        {
            "html": (
                "<blockquote>"
                "<p>First line\n  continues<br>"
                "second line https://t.co/a-b_C~d?ref=1</p>"
                "<p>Third\t line</p>"
                "&mdash; Attribution that is not tweet text"
                "</blockquote>"
            )
        }
    )

    assert parsed["text"] == (
        "First line\n"
        "continues\n"
        "second line https://t.co/a-b_C~d?ref=1\n\n"
        "Third line"
    )
    assert parsed["tco_urls"] == ["https://t.co/a-b_C~d?ref=1"]


def test_parse_oembed_payload_rejects_attribution_only_html_without_paragraphs() -> None:
    with pytest.raises(
        importer.OEmbedImportError,
        match="tweet text paragraphs",
    ):
        importer.parse_oembed_payload(
            {
                "html": (
                    "<blockquote>"
                    "&mdash; Ada Lovelace (@ada) "
                    '<a href="https://twitter.com/ada/status/123">'
                    "July 26, 2026"
                    "</a>"
                    "</blockquote>"
                )
            }
        )


def test_enrich_ledger_record_rejects_invalid_source_urls_without_http_access(
    fixed_clock,
) -> None:
    requests: list[str] = []

    def fake_http_get(url: str, timeout: float) -> bytes:
        requests.append(url)
        raise AssertionError("invalid source URL must never reach HTTP")

    result, imported = importer.enrich_ledger_record(
        {"source_url": "https://example.invalid/ada/status/123"},
        http_get=fake_http_get,
        clock=fixed_clock,
    )

    assert not imported
    assert requests == []
    assert result["x_oembed_import"]["status"] == "error"
    assert result["x_oembed_import"]["error"]["type"] == "OEmbedImportError"
    assert "X/Twitter status URL" in result["x_oembed_import"]["error"]["message"]


def test_import_jsonl_records_malformed_jsonl_and_continues(
    ledger_paths: tuple[Path, Path],
    oembed_payload: dict[str, str],
    fixed_clock,
) -> None:
    source, destination = ledger_paths
    source.write_text(
        json.dumps({"source_url": "https://x.com/ada/status/123"})
        + "\n"
        + '{"ledger_id":\n',
        encoding="utf-8",
    )

    def fake_http_get(url: str, timeout: float) -> bytes:
        return json.dumps(oembed_payload).encode("utf-8")

    summary = importer.import_jsonl(
        source,
        destination,
        http_get=fake_http_get,
        clock=fixed_clock,
        sleeper=lambda _: None,
    )

    assert summary == importer.ImportSummary(
        records_total=2,
        records_succeeded=1,
        records_failed=1,
    )

    succeeded, malformed = _read_jsonl(destination)
    assert succeeded["x_oembed_import"]["status"] == "ok"
    assert malformed["raw_jsonl"] == '{"ledger_id":'
    assert malformed["x_oembed_import"]["status"] == "error"
    assert malformed["x_oembed_import"]["line_number"] == 2
    assert malformed["x_oembed_import"]["error"]["type"] == "JSONDecodeError"


def test_cli_uses_explicit_url_field_and_configured_inter_record_delay(
    ledger_paths: tuple[Path, Path],
    oembed_payload: dict[str, str],
    fixed_clock,
    capsys,
) -> None:
    source, destination = ledger_paths
    records = [
        {
            "ledger_id": "cli-first",
            "url": "https://example.invalid/not-used",
            "post_link": "https://x.com/ada/status/333",
        },
        {
            "ledger_id": "cli-second",
            "url": "https://example.invalid/not-used",
            "post_link": "https://twitter.com/ada/status/444",
        },
    ]
    source.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    observed_urls: list[str] = []
    observed_sleeps: list[float] = []

    def fake_http_get(url: str, timeout: float) -> bytes:
        observed_urls.append(parse_qs(urlsplit(url).query)["url"][0])
        return json.dumps(oembed_payload).encode("utf-8")

    exit_code = importer.main(
        [
            "--input",
            str(source),
            "--output",
            str(destination),
            "--url-field",
            "post_link",
            "--inter-record-delay",
            "0.25",
        ],
        http_get=fake_http_get,
        clock=fixed_clock,
        sleeper=observed_sleeps.append,
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {
        "records_failed": 0,
        "records_succeeded": 2,
        "records_total": 2,
    }
    assert observed_urls == [
        "https://x.com/ada/status/333",
        "https://twitter.com/ada/status/444",
    ]
    assert observed_sleeps == [0.25]

    result = _read_jsonl(destination)
    assert [record["x_oembed_import"]["status"] for record in result] == [
        "ok",
        "ok",
    ]


def test_cli_rejects_negative_inter_record_delay(
    ledger_paths: tuple[Path, Path],
    capsys,
) -> None:
    source, destination = ledger_paths

    with pytest.raises(SystemExit) as exc_info:
        importer.main(
            [
                "--input",
                str(source),
                "--output",
                str(destination),
                "--inter-record-delay",
                "-0.1",
            ]
        )

    assert exc_info.value.code == 2
    assert "--inter-record-delay must be a finite non-negative number" in (
        capsys.readouterr().err
    )
