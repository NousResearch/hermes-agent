from __future__ import annotations

from market_monitor.models import FetchResult, ObservationRecord, ParseOutput
from market_monitor.parsers.common import (
    attach_snapshot_metadata,
    extract_float,
    extract_period_label,
    html_text,
    make_obs_id,
    make_snapshot,
    validate_non_negative,
)
from market_monitor.fetchers.base import content_hash


class CaamNevProdSalesParser:
    source_id = "caam"
    parser_version = "0.1.0"

    def parse(self, fetch_result: FetchResult, dataset_id: str | None = None) -> ParseOutput:
        dataset_id = dataset_id or fetch_result.dataset_id or "caam_nev_prod_sales"
        text = html_text(fetch_result.text)
        period_label = extract_period_label(text)
        published_at = fetch_result.fetch_time[:10]

        production_abs = _wan_liang(text, r"产量为\s*(-?\d+(?:\.\d+)?)\s*万辆")
        sales_abs = _wan_liang(text, r"销量为\s*(-?\d+(?:\.\d+)?)\s*万辆")
        production_yoy = extract_float(r"产量为\s*-?\d+(?:\.\d+)?\s*万辆，同比(?:增长|下降)\s*(-?\d+(?:\.\d+)?)%", text)
        sales_yoy = extract_float(r"销量为\s*-?\d+(?:\.\d+)?\s*万辆，同比(?:增长|下降)\s*(-?\d+(?:\.\d+)?)%", text)
        market_share = extract_float(r"总销量的\s*(-?\d+(?:\.\d+)?)%", text)

        observations = [
            ObservationRecord(
                obs_id=make_obs_id(dataset_id, period_label, "production_volume", "production"),
                dataset_id=dataset_id,
                source_id=self.source_id,
                period_label=period_label,
                period_type="month",
                metric_name="production_volume",
                metric_scope="production",
                metric_type="absolute",
                energy_type="nev_total",
                value_numeric=production_abs,
                unit="vehicles",
                published_at=published_at,
                source_url=fetch_result.source_url,
            ),
            ObservationRecord(
                obs_id=make_obs_id(dataset_id, period_label, "sales_volume", "production"),
                dataset_id=dataset_id,
                source_id=self.source_id,
                period_label=period_label,
                period_type="month",
                metric_name="sales_volume",
                metric_scope="production",
                metric_type="absolute",
                energy_type="nev_total",
                value_numeric=sales_abs,
                unit="vehicles",
                published_at=published_at,
                source_url=fetch_result.source_url,
            ),
            ObservationRecord(
                obs_id=make_obs_id(dataset_id, period_label, "production_volume", "production", "yoy"),
                dataset_id=dataset_id,
                source_id=self.source_id,
                period_label=period_label,
                period_type="month",
                metric_name="production_volume",
                metric_scope="production",
                metric_type="yoy",
                energy_type="nev_total",
                value_numeric=production_yoy,
                unit="pct",
                published_at=published_at,
                source_url=fetch_result.source_url,
            ),
            ObservationRecord(
                obs_id=make_obs_id(dataset_id, period_label, "sales_volume", "production", "yoy"),
                dataset_id=dataset_id,
                source_id=self.source_id,
                period_label=period_label,
                period_type="month",
                metric_name="sales_volume",
                metric_scope="production",
                metric_type="yoy",
                energy_type="nev_total",
                value_numeric=sales_yoy,
                unit="pct",
                published_at=published_at,
                source_url=fetch_result.source_url,
            ),
            ObservationRecord(
                obs_id=make_obs_id(dataset_id, period_label, "market_share", "production"),
                dataset_id=dataset_id,
                source_id=self.source_id,
                period_label=period_label,
                period_type="month",
                metric_name="market_share",
                metric_scope="production",
                metric_type="share",
                energy_type="nev_total",
                value_numeric=market_share,
                unit="pct",
                published_at=published_at,
                source_url=fetch_result.source_url,
            ),
        ]
        validate_non_negative([obs for obs in observations if obs.metric_type == "absolute"])
        output = ParseOutput(snapshot=make_snapshot(fetch_result, parser_version=self.parser_version), observations=observations)
        return attach_snapshot_metadata(output, content_hash=content_hash(fetch_result.content_bytes))


def _wan_liang(text: str, pattern: str) -> int:
    value = extract_float(pattern, text)
    vehicles = int(round(value * 10000))
    if vehicles < 0:
        raise ValueError("negative vehicle count not allowed")
    return vehicles
