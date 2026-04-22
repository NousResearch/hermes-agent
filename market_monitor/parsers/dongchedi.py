from __future__ import annotations

from market_monitor.fetchers.base import content_hash
from market_monitor.models import FetchResult, ObservationRecord, ParseOutput
from market_monitor.parsers.common import (
    attach_snapshot_metadata,
    build_brand_and_model_entities,
    build_entity_link,
    make_model_entity_id,
    make_obs_id,
    make_snapshot,
    parse_embedded_json,
    parse_script_assignment_json,
    validate_non_negative,
    validate_ranking_continuity,
)


class DongchediModelRankParser:
    source_id = "dongchedi"
    parser_version = "0.1.0"

    def parse(self, fetch_result: FetchResult, dataset_id: str | None = None) -> ParseOutput:
        dataset_id = dataset_id or fetch_result.dataset_id or "dongchedi_model_rank"
        payload = _load_payload(fetch_result.text or "")
        period_label = payload["period"]
        published_at = fetch_result.fetch_time[:10]

        entities = []
        observations = []
        links = []
        for item in payload.get("items", []):
            brand = item["brand"].strip()
            model = item["model"].strip()
            ranking = int(item["rank"])
            sales = float(item["sales"])
            if sales < 0:
                raise ValueError("negative sales value not allowed")
            brand_entity, model_entity = build_brand_and_model_entities(brand, model)
            entities.extend([brand_entity, model_entity])
            obs_id = make_obs_id(dataset_id, period_label, "sales_volume", "retail", f"rank-{ranking}")
            observations.append(
                ObservationRecord(
                    obs_id=obs_id,
                    dataset_id=dataset_id,
                    source_id=self.source_id,
                    period_label=period_label,
                    period_type="month",
                    metric_name="sales_volume",
                    metric_scope="retail",
                    metric_type="ranking",
                    value_numeric=sales,
                    ranking=ranking,
                    unit="vehicles",
                    published_at=published_at,
                    source_url=fetch_result.source_url,
                )
            )
            links.append(build_entity_link(obs_id, brand_entity.entity_id, "brand"))
            links.append(build_entity_link(obs_id, make_model_entity_id(brand, model), "model"))

        validate_non_negative(observations)
        validate_ranking_continuity(observations)
        output = ParseOutput(
            snapshot=make_snapshot(fetch_result, parser_version=self.parser_version),
            entities=entities,
            observations=observations,
            observation_entities=links,
        )
        return attach_snapshot_metadata(output, content_hash=content_hash(fetch_result.content_bytes))


def _load_payload(text: str) -> dict:
    for element_id in ("__DATA__", "__NEXT_DATA__"):
        try:
            return parse_embedded_json(text, element_id=element_id)
        except ValueError:
            continue
    return parse_script_assignment_json(text, "window.__INITIAL_STATE__")
