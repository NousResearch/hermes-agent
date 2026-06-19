"""PydanticAI-driven calibration for Futu stock-filter probes.

The calibration stage sits between theme discovery and candidate-pool
compression. The AI agent proposes how to turn broad screener ideas into
concrete Futu ``get_stock_filter`` trials; deterministic code executes those
trials and records what actually came back; a second AI pass selects whether
each probe should become a calibrated filter, remain rank-then-score evidence,
or be skipped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from .futu_theme_discovery import (
    _build_futu_stock_filters,
    _normalize_market,
    load_futu_screener_catalog_snapshot,
)
from .pydantic_runtime import (
    create_pydantic_agent,
    pydantic_event_stream_handler,
    usage_metadata,
)
from .schemas import (
    CalibratedFilter,
    CalibrationInputProbe,
    CalibrationTrial,
    CalibrationTrialResult,
    FilterCalibrationArtifact,
)
from .storage import utc_now
from .theme_discovery import normalize_futu_symbol


_CALIBRATION_PLANNER_INSTRUCTIONS = """
You are the investment assistant's Futu filter-calibration planner.

Your job is to convert the user's theme-level screener probes into concrete
Futu get_stock_filter trials. Do not recommend a portfolio and do not select
stocks. You only design filter trials that deterministic code will execute.

Hard rules:
- Use only the probes and stock_filter_specs supplied in the payload.
- Preserve the intent of each probe, but create multiple trial variants when a
  field needs calibration: baseline/ranking-only, relaxed threshold, and strict
  threshold are typical variants.
- For numeric filters, choose explicit filter_min/filter_max values only when
  the field meaning supports a numeric threshold. Explain the rationale.
- Do not introduce unsupported Futu fields, new symbols, or new plate codes.
- Keep result_limit between 20 and 200.
- Write rationale fields in Chinese. Keep enum values and ticker symbols
  unchanged.
""".strip()


_CALIBRATION_SELECTOR_INSTRUCTIONS = """
You are the investment assistant's Futu filter-calibration selector.

You receive the original probes, the exact trials that were run, and the
deterministic Futu results. Decide for each probe whether it should become:
- calibrated_filter: a concrete Futu filter is useful and not too narrow.
- rank_then_score: the signal is useful for ranking/enrichment, but hard
  filtering would hide important candidates.
- skip_probe: the probe failed, is unsupported, or does not add useful evidence.

Hard rules:
- Do not recommend portfolio weights or trades.
- Base decisions only on trial_results and warnings in the payload.
- For calibrated_filter, select one existing trial_id.
- Prefer rank_then_score over calibrated_filter when important focus symbols
  disappear or the result set is too narrow.
- Write selection_reason fields in Chinese.
""".strip()


class CalibrationTrialPlan(BaseModel):
    """AI-authored trial plan before deterministic Futu execution."""

    theme: str
    market: str = "US"
    trials: list[CalibrationTrial] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CalibrationSelectionPlan(BaseModel):
    """AI-authored interpretation of deterministic calibration results."""

    calibrated_filters: list[CalibratedFilter] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


@dataclass
class FutuCalibrationExecutor:
    """Execute calibration trials through the live Futu OpenD API."""

    config: Any | None = None

    def execute_trials(
        self,
        trials: list[CalibrationTrial],
        *,
        market: str = "US",
        focus_symbols: list[str] | None = None,
    ) -> list[CalibrationTrialResult]:
        from .adapters import (
            FutuOpenDConfig,
            MarketDataAdapter,
            _check_ret,
            _import_futu,
            _parse_quote_market,
        )

        market = _normalize_market(market)
        focus = _normalize_symbols(focus_symbols or [], market)
        config = self.config or FutuOpenDConfig.from_env()
        adapter = MarketDataAdapter(config)
        adapter._check_opend()  # noqa: SLF001 - shared adapter boundary inside this plugin.
        futu = _import_futu()
        quote_ctx = futu.OpenQuoteContext(host=config.host, port=config.port)
        try:
            results: list[CalibrationTrialResult] = []
            for trial in trials:
                validation_errors = validate_stock_filter_specs_against_catalog(
                    trial.stock_filter_specs,
                    market=market,
                )
                if validation_errors:
                    results.append(
                        _failed_trial_result(
                            trial,
                            focus,
                            "Unsupported stock_filter_specs: " + "; ".join(validation_errors),
                        )
                    )
                    continue
                try:
                    filters = _build_futu_stock_filters(futu, trial.stock_filter_specs)
                    ret, data = adapter._quote_call(  # noqa: SLF001
                        quote_ctx.get_stock_filter,
                        _parse_quote_market(futu, market),
                        filters,
                        plate_code=trial.plate_code,
                        begin=0,
                        num=trial.result_limit,
                    )
                    _check_ret(futu, ret, data, f"get_stock_filter({trial.trial_id})")
                    _last_page, all_count, stock_list = data
                    results.append(
                        _trial_result_from_stock_list(
                            trial,
                            all_count=int(all_count),
                            stock_list=list(stock_list),
                            market=market,
                            focus_symbols=focus,
                        )
                    )
                except Exception as exc:
                    results.append(_failed_trial_result(trial, focus, str(exc)))
            return results
        finally:
            quote_ctx.close()


def calibrate_filter_probes(
    *,
    theme: str,
    market: str = "US",
    probes: list[CalibrationInputProbe],
    focus_symbols: list[str] | None = None,
    executor: FutuCalibrationExecutor | None = None,
) -> FilterCalibrationArtifact:
    """Run the AI-planned, Futu-executed filter-calibration loop."""

    market = _normalize_market(market)
    focus = _normalize_symbols(focus_symbols or _probe_focus_symbols(probes), market)
    trial_plan, planner_runtime = _run_calibration_planner_agent(
        theme=theme,
        market=market,
        probes=probes,
        focus_symbols=focus,
    )
    trial_results = (executor or FutuCalibrationExecutor()).execute_trials(
        trial_plan.trials,
        market=market,
        focus_symbols=focus,
    )
    selection_plan, selector_runtime = _run_calibration_selector_agent(
        theme=theme,
        market=market,
        probes=probes,
        trials=trial_plan.trials,
        trial_results=trial_results,
        focus_symbols=focus,
    )
    return build_filter_calibration_artifact(
        theme=theme,
        market=market,
        probes=probes,
        trials=trial_plan.trials,
        trial_results=trial_results,
        calibrated_filters=selection_plan.calibrated_filters,
        warnings=[*trial_plan.warnings, *selection_plan.warnings],
        pydantic_ai={
            "planner": planner_runtime,
            "selector": selector_runtime,
        },
    )


def build_filter_calibration_artifact(
    *,
    theme: str,
    market: str,
    probes: list[CalibrationInputProbe],
    trials: list[CalibrationTrial],
    trial_results: list[CalibrationTrialResult],
    calibrated_filters: list[CalibratedFilter],
    warnings: list[str] | None = None,
    pydantic_ai: dict[str, Any] | None = None,
) -> FilterCalibrationArtifact:
    """Build and validate a calibration artifact from executed trials."""

    market = _normalize_market(market)
    trial_by_id = {trial.trial_id: trial for trial in trials}
    normalized_filters: list[CalibratedFilter] = []
    for item in calibrated_filters:
        if item.selected_trial_id and not item.selected_filters and item.selected_trial_id in trial_by_id:
            item = item.model_copy(
                update={"selected_filters": trial_by_id[item.selected_trial_id].stock_filter_specs}
            )
        normalized_filters.append(item)

    artifact = FilterCalibrationArtifact(
        theme=theme,
        market=market,
        generated_at=utc_now(),
        input_probe_count=len(probes),
        probes=probes,
        trials=trials,
        trial_results=trial_results,
        calibrated_filters=normalized_filters,
        focus_symbol_audit=_focus_symbol_audit(probes, trial_results, market),
        warnings=_dedupe(warnings or []),
        pydantic_ai=pydantic_ai or {},
    )
    _validate_filter_calibration_artifact(artifact)
    return artifact


def validate_stock_filter_specs_against_catalog(
    specs: list[dict[str, Any]],
    *,
    market: str = "US",
) -> list[str]:
    """Validate stock_filter_specs against the cached Futu screener catalog."""

    if not specs:
        return ["stock_filter_specs cannot be empty for calibration trials."]

    catalog = load_futu_screener_catalog_snapshot(market=market)
    stock_catalog = (
        catalog.get("markets", {})
        .get(_normalize_market(market), {})
        .get("stock_filter_catalog", {})
    )
    filter_types = stock_catalog.get("filter_types", {})
    if not isinstance(filter_types, dict) or not filter_types:
        return ["Futu stock-filter catalog snapshot is missing filter_types."]

    errors: list[str] = []
    sort_values = set(stock_catalog.get("sort_dir") or [])
    quarter_values = set(stock_catalog.get("financial_quarter") or [])
    relative_positions = set(stock_catalog.get("relative_position") or [])
    supported_ktype = set(stock_catalog.get("supported_pattern_ktype") or [])

    for index, spec in enumerate(specs):
        if not isinstance(spec, dict):
            errors.append(f"spec[{index}] must be an object.")
            continue
        filter_type = str(spec.get("type") or "simple").strip().lower()
        type_catalog = filter_types.get(filter_type)
        if not isinstance(type_catalog, dict):
            errors.append(f"spec[{index}] has unsupported type: {filter_type}")
            continue
        allowed_fields = set(type_catalog.get("fields") or [])
        if filter_type == "custom_indicator":
            for key in ("stock_field1", "stock_field2"):
                field = str(spec.get(key) or "").strip().upper()
                if field not in allowed_fields:
                    errors.append(f"spec[{index}].{key} is unsupported: {spec.get(key)}")
            if spec.get("relative_position") not in relative_positions:
                errors.append(
                    f"spec[{index}].relative_position is unsupported: {spec.get('relative_position')}"
                )
        else:
            field = str(spec.get("stock_field") or "").strip().upper()
            if field not in allowed_fields:
                errors.append(f"spec[{index}].stock_field is unsupported: {spec.get('stock_field')}")
        if spec.get("sort") is not None and spec.get("sort") not in sort_values:
            errors.append(f"spec[{index}].sort is unsupported: {spec.get('sort')}")
        if filter_type == "financial" and spec.get("quarter") is not None:
            if spec.get("quarter") not in quarter_values:
                errors.append(f"spec[{index}].quarter is unsupported: {spec.get('quarter')}")
        if filter_type in {"pattern", "custom_indicator"} and spec.get("ktype") is not None:
            if spec.get("ktype") not in supported_ktype:
                errors.append(f"spec[{index}].ktype is unsupported: {spec.get('ktype')}")
    return errors


def diagnose_trial_result(returned_count: int, all_count: int, *, result_limit: int) -> str:
    """Classify a Futu calibration result for the selector agent."""

    if returned_count <= 0 or all_count <= 0:
        return "zero_result"
    if all_count < 5:
        return "too_narrow"
    if all_count > max(result_limit * 3, 300):
        return "too_broad"
    return "usable"


def _run_calibration_planner_agent(
    *,
    theme: str,
    market: str,
    probes: list[CalibrationInputProbe],
    focus_symbols: list[str],
) -> tuple[CalibrationTrialPlan, dict[str, Any]]:
    from pydantic_ai import ModelRetry

    agent, _model_config, runtime = create_pydantic_agent(
        output_type=CalibrationTrialPlan,
        instructions=_CALIBRATION_PLANNER_INSTRUCTIONS,
        agent_kind="filter_calibration_planner",
        output_retries=2,
    )

    @agent.output_validator
    def validate_planner_output(data: CalibrationTrialPlan) -> CalibrationTrialPlan:
        data.theme = theme
        data.market = market
        try:
            _validate_trial_plan(probes, data, market=market)
        except ValueError as exc:
            raise ModelRetry(str(exc)) from exc
        return data

    result = agent.run_sync(
        _planner_prompt(theme, market, probes, focus_symbols),
        event_stream_handler=pydantic_event_stream_handler("filter_calibration_planner"),
    )
    plan = result.output
    plan.theme = theme
    plan.market = market
    return plan, {**runtime, "usage": usage_metadata(result)}


def _run_calibration_selector_agent(
    *,
    theme: str,
    market: str,
    probes: list[CalibrationInputProbe],
    trials: list[CalibrationTrial],
    trial_results: list[CalibrationTrialResult],
    focus_symbols: list[str],
) -> tuple[CalibrationSelectionPlan, dict[str, Any]]:
    from pydantic_ai import ModelRetry

    agent, _model_config, runtime = create_pydantic_agent(
        output_type=CalibrationSelectionPlan,
        instructions=_CALIBRATION_SELECTOR_INSTRUCTIONS,
        agent_kind="filter_calibration_selector",
        output_retries=2,
    )

    @agent.output_validator
    def validate_selector_output(data: CalibrationSelectionPlan) -> CalibrationSelectionPlan:
        try:
            _validate_selection_plan(probes, trials, data)
        except ValueError as exc:
            raise ModelRetry(str(exc)) from exc
        return data

    result = agent.run_sync(
        _selector_prompt(theme, market, probes, trials, trial_results, focus_symbols),
        event_stream_handler=pydantic_event_stream_handler("filter_calibration_selector"),
    )
    return result.output, {**runtime, "usage": usage_metadata(result)}


def _planner_prompt(
    theme: str,
    market: str,
    probes: list[CalibrationInputProbe],
    focus_symbols: list[str],
) -> str:
    payload = {
        "task": "Create concrete Futu get_stock_filter calibration trials.",
        "theme": theme,
        "market": market,
        "focus_symbols": focus_symbols,
        "probes": [probe.model_dump(mode="json") for probe in probes],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _selector_prompt(
    theme: str,
    market: str,
    probes: list[CalibrationInputProbe],
    trials: list[CalibrationTrial],
    trial_results: list[CalibrationTrialResult],
    focus_symbols: list[str],
) -> str:
    payload = {
        "task": "Select calibrated filters from deterministic Futu trial results.",
        "theme": theme,
        "market": market,
        "focus_symbols": focus_symbols,
        "probes": [probe.model_dump(mode="json") for probe in probes],
        "trials": [trial.model_dump(mode="json") for trial in trials],
        "trial_results": [item.model_dump(mode="json") for item in trial_results],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _validate_trial_plan(
    probes: list[CalibrationInputProbe],
    plan: CalibrationTrialPlan,
    *,
    market: str,
) -> None:
    probe_names = {probe.name for probe in probes}
    if probes and not plan.trials:
        raise ValueError("Calibration planner returned no trials.")
    seen: set[str] = set()
    errors: list[str] = []
    for trial in plan.trials:
        if trial.trial_id in seen:
            errors.append(f"Duplicate trial_id: {trial.trial_id}")
        seen.add(trial.trial_id)
        if trial.probe_name not in probe_names:
            errors.append(f"Trial {trial.trial_id} references unknown probe_name: {trial.probe_name}")
        if not trial.stock_filter_specs:
            errors.append(f"Trial {trial.trial_id} has empty stock_filter_specs.")
        errors.extend(
            f"Trial {trial.trial_id}: {error}"
            for error in validate_stock_filter_specs_against_catalog(
                trial.stock_filter_specs,
                market=market,
            )
        )
    if errors:
        raise ValueError("; ".join(errors))


def _validate_selection_plan(
    probes: list[CalibrationInputProbe],
    trials: list[CalibrationTrial],
    selection: CalibrationSelectionPlan,
) -> None:
    probe_names = {probe.name for probe in probes}
    trial_ids = {trial.trial_id for trial in trials}
    selected_probe_names = {item.probe_name for item in selection.calibrated_filters}
    errors: list[str] = []
    missing = sorted(probe_names - selected_probe_names)
    if missing:
        errors.append("Selection is missing probes: " + ", ".join(missing))
    for item in selection.calibrated_filters:
        if item.probe_name not in probe_names:
            errors.append(f"Selection references unknown probe_name: {item.probe_name}")
        if item.selected_mode in {"calibrated_filter", "rank_then_score"}:
            if not item.selected_trial_id:
                errors.append(f"{item.probe_name} requires selected_trial_id for {item.selected_mode}.")
            elif item.selected_trial_id not in trial_ids:
                errors.append(
                    f"{item.probe_name} references unknown selected_trial_id: {item.selected_trial_id}"
                )
    if errors:
        raise ValueError("; ".join(errors))


def _validate_filter_calibration_artifact(artifact: FilterCalibrationArtifact) -> None:
    _validate_trial_plan(
        artifact.probes,
        CalibrationTrialPlan(
            theme=artifact.theme,
            market=artifact.market,
            trials=artifact.trials,
        ),
        market=artifact.market,
    )
    _validate_selection_plan(
        artifact.probes,
        artifact.trials,
        CalibrationSelectionPlan(calibrated_filters=artifact.calibrated_filters),
    )


def _trial_result_from_stock_list(
    trial: CalibrationTrial,
    *,
    all_count: int,
    stock_list: list[Any],
    market: str,
    focus_symbols: list[str],
) -> CalibrationTrialResult:
    sample_symbols: list[str] = []
    sample_names: dict[str, str] = {}
    for item in stock_list[: trial.result_limit]:
        raw_code = str(getattr(item, "stock_code", "") or "")
        symbol = normalize_futu_symbol(raw_code, market)
        if not symbol:
            continue
        sample_symbols.append(symbol)
        name = str(getattr(item, "stock_name", "") or "")
        if name:
            sample_names[symbol] = name
    included = [symbol for symbol in focus_symbols if symbol in sample_symbols]
    missing = [symbol for symbol in focus_symbols if symbol not in sample_symbols]
    returned_count = len(sample_symbols)
    return CalibrationTrialResult(
        trial_id=trial.trial_id,
        probe_name=trial.probe_name,
        status="completed",
        diagnosis=diagnose_trial_result(returned_count, all_count, result_limit=trial.result_limit),
        all_count=max(0, int(all_count)),
        returned_count=returned_count,
        sample_symbols=sample_symbols,
        sample_names=sample_names,
        focus_symbols_included=included,
        focus_symbols_missing=missing,
    )


def _failed_trial_result(
    trial: CalibrationTrial,
    focus_symbols: list[str],
    error: str,
) -> CalibrationTrialResult:
    return CalibrationTrialResult(
        trial_id=trial.trial_id,
        probe_name=trial.probe_name,
        status="failed",
        diagnosis="error",
        focus_symbols_missing=focus_symbols,
        warnings=[error],
        error=error,
    )


def _focus_symbol_audit(
    probes: list[CalibrationInputProbe],
    trial_results: list[CalibrationTrialResult],
    market: str,
) -> dict[str, Any]:
    focus_symbols = _normalize_symbols(_probe_focus_symbols(probes), market)
    included_any = []
    missing_all = []
    for symbol in focus_symbols:
        if any(symbol in result.focus_symbols_included for result in trial_results):
            included_any.append(symbol)
        else:
            missing_all.append(symbol)
    return {
        "focus_symbols": focus_symbols,
        "included_any": included_any,
        "missing_all": missing_all,
    }


def _probe_focus_symbols(probes: list[CalibrationInputProbe]) -> list[str]:
    symbols: list[str] = []
    for probe in probes:
        symbols.extend(probe.focus_symbols)
    return _dedupe(symbols)


def _normalize_symbols(symbols: list[str], market: str) -> list[str]:
    return _dedupe(
        [
            normalize_futu_symbol(symbol, market)
            for symbol in symbols
            if normalize_futu_symbol(symbol, market)
        ]
    )


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output
