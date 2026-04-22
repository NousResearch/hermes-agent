from __future__ import annotations

import re

from market_monitor.fetchers.base import content_hash
from market_monitor.models import FetchResult, ObservationRecord, ParseOutput
from market_monitor.parsers.common import attach_snapshot_metadata, extract_period_label, html_text, make_obs_id, make_snapshot


class CadaNevReportMetaParser:
    source_id = "cada"
    parser_version = "0.1.0"

    def parse(self, fetch_result: FetchResult, dataset_id: str | None = None) -> ParseOutput:
        dataset_id = dataset_id or fetch_result.dataset_id or "cada_nev_report_meta"
        raw = fetch_result.text or ""
        text = html_text(raw)
        period_label = _safe_period(text, fetch_result.period_hint)
        title_match = re.search(r"<h1[^>]*>(.*?)</h1>", raw, re.S)
        title = html_text(title_match.group(1)) if title_match else text[:120]
        published_at = fetch_result.fetch_time[:10]
        observations = [
            ObservationRecord(
                obs_id=make_obs_id(dataset_id, period_label, "report_title", "retail"),
                dataset_id=dataset_id,
                source_id=self.source_id,
                period_label=period_label,
                period_type="month",
                metric_name="report_title",
                metric_scope="retail",
                metric_type="absolute",
                value_text=title,
                published_at=published_at,
                source_url=fetch_result.source_url,
            )
        ]
        output = ParseOutput(snapshot=make_snapshot(fetch_result, parser_version=self.parser_version), observations=observations)
        return attach_snapshot_metadata(output, content_hash=content_hash(fetch_result.content_bytes))


def _safe_period(text: str, period_hint: str | None) -> str:
    try:
        return extract_period_label(text)
    except ValueError:
        if period_hint:
            return period_hint
        raise
