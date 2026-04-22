from __future__ import annotations

from market_monitor.fetchers.base import content_hash
from market_monitor.models import FetchResult, ObservationRecord, ParseOutput
from market_monitor.parsers.common import attach_snapshot_metadata, extract_period_label, html_text, make_obs_id, make_snapshot, normalize_period_label


class CpcaMonthlyMarketParser:
    source_id = "cpca"
    parser_version = "0.1.0"

    def parse(self, fetch_result: FetchResult, dataset_id: str | None = None) -> ParseOutput:
        dataset_id = dataset_id or fetch_result.dataset_id or "cpca_monthly_market"
        text = html_text(fetch_result.text)
        period_label = _safe_period(text, fetch_result.period_hint)
        published_at = fetch_result.fetch_time[:10]
        observations: list[ObservationRecord] = []

        patterns = {
            "retail": r"零售(?:销量)?(?:为)?\s*(\d+(?:\.\d+)?)\s*万辆",
            "wholesale": r"批发(?:销量)?(?:为)?\s*(\d+(?:\.\d+)?)\s*万辆",
            "export": r"出口(?:量)?(?:为)?\s*(\d+(?:\.\d+)?)\s*万辆",
        }
        for scope, pattern in patterns.items():
            import re

            match = re.search(pattern, text)
            if not match:
                continue
            observations.append(
                ObservationRecord(
                    obs_id=make_obs_id(dataset_id, period_label, "sales_volume", scope),
                    dataset_id=dataset_id,
                    source_id=self.source_id,
                    period_label=period_label,
                    period_type="month",
                    metric_name="sales_volume",
                    metric_scope=scope,
                    metric_type="absolute",
                    energy_type="nev_total",
                    value_numeric=int(round(float(match.group(1)) * 10000)),
                    unit="vehicles",
                    published_at=published_at,
                    source_url=fetch_result.source_url,
                )
            )

        output = ParseOutput(snapshot=make_snapshot(fetch_result, parser_version=self.parser_version), observations=observations)
        return attach_snapshot_metadata(output, content_hash=content_hash(fetch_result.content_bytes))


def _safe_period(text: str, period_hint: str | None) -> str:
    try:
        return extract_period_label(text)
    except ValueError:
        if period_hint:
            return normalize_period_label(period_hint)
        raise
