from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass(frozen=True)
class DatasetConfig:
    dataset_id: str
    source_id: str
    dataset_name: str
    data_url: str
    category: str
    metric_scope: str
    entity_granularity: str
    time_granularity: str


@dataclass(frozen=True)
class SourceConfig:
    source_id: str
    source_name: str
    homepage_url: str
    source_level: str
    update_frequency: str
    access_mode: str
    active: bool
    datasets: tuple[DatasetConfig, ...]


@dataclass(frozen=True)
class SourceCatalog:
    sources: tuple[SourceConfig, ...]
    datasets_by_id: dict[str, DatasetConfig]


def load_source_catalog(path: Path | str) -> SourceCatalog:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    sources: list[SourceConfig] = []
    datasets_by_id: dict[str, DatasetConfig] = {}
    for source in raw.get("sources", []):
        datasets: list[DatasetConfig] = []
        for dataset in source.get("datasets", []):
            dataset_config = DatasetConfig(
                dataset_id=dataset["dataset_id"],
                source_id=source["source_id"],
                dataset_name=dataset["dataset_name"],
                data_url=dataset.get("data_url", source["homepage_url"]),
                category=dataset["category"],
                metric_scope=dataset["metric_scope"],
                entity_granularity=dataset["entity_granularity"],
                time_granularity=dataset["time_granularity"],
            )
            datasets.append(dataset_config)
            datasets_by_id[dataset_config.dataset_id] = dataset_config
        sources.append(
            SourceConfig(
                source_id=source["source_id"],
                source_name=source["source_name"],
                homepage_url=source["homepage_url"],
                source_level=source["source_level"],
                update_frequency=source["update_frequency"],
                access_mode=source["access_mode"],
                active=bool(source.get("active", True)),
                datasets=tuple(datasets),
            )
        )
    return SourceCatalog(sources=tuple(sources), datasets_by_id=datasets_by_id)
