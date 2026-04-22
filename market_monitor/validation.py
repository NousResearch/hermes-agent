from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from market_monitor.db import Database
from market_monitor.models import ObservationRecord
from market_monitor.parsers.common import validate_non_negative, validate_ranking_continuity


@dataclass(frozen=True)
class ValidationFinding:
    code: str
    message: str
    observation_key: str | None = None


@dataclass(frozen=True)
class ValidationResult:
    errors: tuple[ValidationFinding, ...] = ()
    warnings: tuple[ValidationFinding, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class HistoricalObservation:
    observation_key: str | None
    dataset_id: str | None
    source_id: str | None
    energy_type: str | None
    metric_name: str
    metric_scope: str
    metric_type: str | None
    value_numeric: float | None
    period_label: str


def validate_parse_output(observations: Sequence[ObservationRecord]) -> ValidationResult:
    findings_errors: list[ValidationFinding] = []
    try:
        validate_non_negative(observations)
    except ValueError as exc:
        findings_errors.append(ValidationFinding(code="negative_value", message=str(exc)))
    try:
        validate_ranking_continuity(observations)
    except ValueError as exc:
        findings_errors.append(ValidationFinding(code="ranking_discontinuity", message=str(exc)))
    return ValidationResult(errors=tuple(findings_errors), warnings=())


def validate_observations_against_history(
    db: Database,
    observations: Sequence[ObservationRecord | Mapping[str, object]],
    *,
    jump_warning_ratio: float = 2.0,
    jump_error_ratio: float = 3.5,
) -> ValidationResult:
    errors: list[ValidationFinding] = []
    warnings: list[ValidationFinding] = []
    for observation in observations:
        current = _to_historical_observation(observation)
        if current.value_numeric is None or current.observation_key is None:
            continue
        previous = db.query(
            """
            SELECT observation_key, metric_name, metric_scope, value_numeric, period_label
            FROM observations
            WHERE observation_key = ? AND is_latest = 1
            LIMIT 1
            """,
            (current.observation_key,),
        )
        if not previous:
            previous = db.query(
                """
                SELECT observation_key, metric_name, metric_scope, value_numeric, period_label
                FROM observations
                WHERE dataset_id = ? AND source_id = ? AND metric_name = ? AND metric_scope = ?
                  AND metric_type = ? AND COALESCE(energy_type, '') = COALESCE(?, '')
                  AND is_latest = 1 AND period_label < ?
                ORDER BY period_label DESC
                LIMIT 1
                """,
                (current.dataset_id, current.source_id, current.metric_name, current.metric_scope, current.metric_type, current.energy_type, current.period_label),
            )
        if not previous:
            continue
        previous_value = previous[0]["value_numeric"]
        if previous_value in (None, 0):
            continue
        ratio = current.value_numeric / float(previous_value)
        inverse_ratio = float(previous_value) / current.value_numeric if current.value_numeric else None
        breached_error = ratio >= jump_error_ratio or (inverse_ratio is not None and inverse_ratio >= jump_error_ratio)
        breached_warning = ratio >= jump_warning_ratio or (inverse_ratio is not None and inverse_ratio >= jump_warning_ratio)
        if not breached_warning:
            continue
        finding = ValidationFinding(
            code="jump_outlier" if breached_error else "suspicious_jump",
            observation_key=current.observation_key,
            message=(
                f"{current.metric_name}/{current.metric_scope} moved from {previous_value} to "
                f"{current.value_numeric} between {previous[0]['period_label']} and {current.period_label}"
            ),
        )
        if breached_error:
            errors.append(finding)
        else:
            warnings.append(finding)
    return ValidationResult(errors=tuple(errors), warnings=tuple(warnings))


def _to_historical_observation(observation: ObservationRecord | Mapping[str, object]) -> HistoricalObservation:
    if isinstance(observation, ObservationRecord):
        return HistoricalObservation(
            observation_key=observation.observation_key or observation.obs_id,
            dataset_id=observation.dataset_id,
            source_id=observation.source_id,
            energy_type=observation.energy_type,
            metric_name=observation.metric_name,
            metric_scope=observation.metric_scope,
            metric_type=observation.metric_type,
            value_numeric=observation.value_numeric,
            period_label=observation.period_label,
        )
    return HistoricalObservation(
        observation_key=observation.get("observation_key") if isinstance(observation, Mapping) else None,
        dataset_id=str(observation.get("dataset_id")) if observation.get("dataset_id") is not None else None,
        source_id=str(observation.get("source_id")) if observation.get("source_id") is not None else None,
        energy_type=str(observation.get("energy_type")) if observation.get("energy_type") is not None else None,
        metric_name=str(observation.get("metric_name")) if observation.get("metric_name") is not None else "",
        metric_scope=str(observation.get("metric_scope")) if observation.get("metric_scope") is not None else "",
        metric_type=str(observation.get("metric_type")) if observation.get("metric_type") is not None else None,
        value_numeric=float(observation["value_numeric"]) if observation.get("value_numeric") is not None else None,
        period_label=str(observation.get("period_label")) if observation.get("period_label") is not None else "",
    )
