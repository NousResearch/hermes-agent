from cron.source_health import (
    MONITOR_STATUS_PREFIX,
    SourceObservation,
    evaluate_source_health,
    monitor_status_line,
)


def test_verified_primary_allows_delivery():
    status = evaluate_source_health(
        [
            SourceObservation(
                name="official-rss",
                access_ok=True,
                verification="verified",
                confidence="high",
                items_seen=3,
            )
        ]
    )

    assert status.source_access_ok is True
    assert status.fallback_used is False
    assert status.confidence == "high"
    assert status.delivery_allowed is True
    assert status.delivery_suppressed is False
    assert status.suppression_reason is None


def test_verified_fallback_allows_delivery_but_records_fallback():
    status = evaluate_source_health(
        [
            SourceObservation(
                name="reddit-json",
                access_ok=False,
                verification="verified",
                confidence="high",
                failure_reason="HTTP 403",
            ),
            SourceObservation(
                name="public-rss-fallback",
                access_ok=True,
                verification="single_source",
                confidence="medium",
                fallback=True,
                items_seen=1,
            ),
        ]
    )

    assert status.source_access_ok is True
    assert status.fallback_used is True
    assert status.confidence == "medium"
    assert status.delivery_allowed is True


def test_discovery_only_fallback_suppresses_delivery():
    status = evaluate_source_health(
        [
            SourceObservation(
                name="firecrawl-primary",
                access_ok=False,
                verification="verified",
                confidence="high",
                failure_reason="extraction failed",
            ),
            SourceObservation(
                name="web-search-fallback",
                access_ok=True,
                verification="discovery_only",
                confidence="low",
                fallback=True,
                items_seen=4,
            ),
        ]
    )

    assert status.source_access_ok is True
    assert status.fallback_used is True
    assert status.confidence == "low"
    assert status.delivery_allowed is False
    assert status.delivery_suppressed is True
    assert status.suppression_reason == "discovery_only_source"


def test_unverified_access_suppresses_delivery():
    status = evaluate_source_health(
        [
            SourceObservation(
                name="reddit-scrape",
                access_ok=True,
                verification="unverified",
                confidence="medium",
                items_seen=2,
            )
        ]
    )

    assert status.source_access_ok is True
    assert status.delivery_suppressed is True
    assert status.suppression_reason == "unverified_source"


def test_no_access_records_failure_reason():
    status = evaluate_source_health(
        [
            SourceObservation(
                name="reddit-json",
                access_ok=False,
                verification="verified",
                confidence="high",
                failure_reason="HTTP 429",
            )
        ]
    )

    assert status.source_access_ok is False
    assert status.confidence == "none"
    assert status.delivery_suppressed is True
    assert status.suppression_reason == "source_access_failed"
    assert "HTTP 429" in (status.failure_reason or "")


def test_monitor_status_line_is_parseable_json_payload():
    line = monitor_status_line(
        [
            SourceObservation(
                name="official-rss",
                access_ok=True,
                verification="verified",
                confidence="high",
            )
        ]
    )

    assert line.startswith(MONITOR_STATUS_PREFIX)
    payload = line.removeprefix(MONITOR_STATUS_PREFIX)
    assert '"source_access_ok": true' in payload
    assert '"delivery_allowed": true' in payload


def test_scheduler_source_health_gate_suppresses_discovery_only_delivery():
    from cron.scheduler import SILENT_MARKER, _apply_monitor_source_health_gate

    line = monitor_status_line(
        [
            SourceObservation(
                name="web-search-fallback",
                access_ok=True,
                verification="discovery_only",
                confidence="low",
                fallback=True,
                items_seen=2,
            )
        ]
    )

    assert _apply_monitor_source_health_gate(True, f"Candidate found\n{line}") == SILENT_MARKER


def test_scheduler_source_health_gate_allows_verified_delivery():
    from cron.scheduler import _apply_monitor_source_health_gate

    response = monitor_status_line(
        [
            SourceObservation(
                name="official-rss",
                access_ok=True,
                verification="verified",
                confidence="high",
                items_seen=1,
            )
        ]
    )

    assert _apply_monitor_source_health_gate(True, response) == response
