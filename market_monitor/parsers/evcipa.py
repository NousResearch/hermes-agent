from __future__ import annotations

from market_monitor.fetchers.base import content_hash
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


class EvcipaInfraParser:
    source_id = "evcipa"
    parser_version = "0.1.0"

    def parse(self, fetch_result: FetchResult, dataset_id: str | None = None) -> ParseOutput:
        dataset_id = dataset_id or fetch_result.dataset_id or "evcipa_monthly_infra"
        text = html_text(fetch_result.text)
        period_label = extract_period_label(text)
        published_at = fetch_result.fetch_time[:10]

        total = _wan_tai(text, r"累计数量为\s*(-?\d+(?:\.\d+)?)\s*万台")
        public = _wan_tai(text, r"公共充电桩\s*(-?\d+(?:\.\d+)?)\s*万台")
        private = _wan_tai(text, r"(?:私人充电桩|随车配建私人充电桩)\s*(-?\d+(?:\.\d+)?)\s*万台")
        increment = _wan_tai(text, r"增加\s*(-?\d+(?:\.\d+)?)\s*万台")
        ratio = extract_float(r"桩车增量比为\s*1\s*[:：]\s*(-?\d+(?:\.\d+)?)", text)
        if ratio < 0:
            raise ValueError("negative pile vehicle ratio not allowed")

        observations = [
            _obs(dataset_id, self.source_id, period_label, "charging_piles_total", total, fetch_result.source_url, published_at),
            _obs(dataset_id, self.source_id, period_label, "public_charging_piles", public, fetch_result.source_url, published_at),
            _obs(dataset_id, self.source_id, period_label, "private_charging_piles", private, fetch_result.source_url, published_at),
            _obs(dataset_id, self.source_id, period_label, "monthly_increment", increment, fetch_result.source_url, published_at),
            ObservationRecord(
                obs_id=make_obs_id(dataset_id, period_label, "pile_vehicle_ratio", "charging_infrastructure"),
                dataset_id=dataset_id,
                source_id=self.source_id,
                period_label=period_label,
                period_type="month",
                metric_name="pile_vehicle_ratio",
                metric_scope="charging_infrastructure",
                metric_type="absolute",
                value_numeric=ratio,
                unit="ratio",
                published_at=published_at,
                source_url=fetch_result.source_url,
            ),
        ]
        validate_non_negative(observations)
        output = ParseOutput(snapshot=make_snapshot(fetch_result, parser_version=self.parser_version), observations=observations)
        return attach_snapshot_metadata(output, content_hash=content_hash(fetch_result.content_bytes))


def _wan_tai(text: str, pattern: str) -> int:
    value = extract_float(pattern, text)
    piles = int(round(value * 10000))
    if piles < 0:
        raise ValueError("negative infrastructure value not allowed")
    return piles


def _obs(dataset_id: str, source_id: str, period_label: str, metric_name: str, value: int, source_url: str, published_at: str) -> ObservationRecord:
    return ObservationRecord(
        obs_id=make_obs_id(dataset_id, period_label, metric_name, "charging_infrastructure"),
        dataset_id=dataset_id,
        source_id=source_id,
        period_label=period_label,
        period_type="month",
        metric_name=metric_name,
        metric_scope="charging_infrastructure",
        metric_type="absolute",
        value_numeric=value,
        unit="piles",
        published_at=published_at,
        source_url=source_url,
    )
