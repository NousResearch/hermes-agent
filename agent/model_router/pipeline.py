"""Staged routing pipeline — port of pipeline/router-pipeline.ts control flow.

Ordered stages with early exit: the moment any stage reaches a routing
decision, subsequent stages are skipped. Any stage exception produces a
fallback decision (safe default, else current model) with the prompt redacted
from the error — routing never raises into the gateway.

Stage order (adapted from PIPELINE_STAGE_ORDER; Pi-only stages — hardware
probe, speculative prewarm, planning delegate — are explicit non-goals):

  1. force_model            — explicit operator override short-circuit
  2. loop_escalation        — repeated tool failures escalate to frontier
  3. context_fit            — filter fleet by context window (never decides)
  4. low_intensity          — low-intensity turns pick cheapest adequate model
  5. session_pin            — valid pin holds; break conditions evaluated
  6. triage                 — trivial/complex fast paths
  7. local_zero             — opt-in local inference (default disabled)
  8. triage_cloud_fallback  — trivial turns not claimed locally → economical
  9. hydra_match            — requirement vector + multi-objective selection
 10. safe_default           — first healthy safe-tier model
 11. context_overflow       — largest-fit model when economical cannot fit
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

from . import hydra
from .context_fit import (
    DEFAULT_MIN_OUTPUT_TOKENS,
    DEFAULT_SAFETY_MARGIN,
    filter_fleet_by_context_fit,
    resolve_context_overflow_fallback,
    select_lowest_cost_model,
)
from .envelope import classify_turn_envelope
from .escalation import evaluate_loop_escalation
from .health import HealthConfig
from .local_zero import LocalZeroConfig, ping_local_services
from .low_intensity import DEFAULT_HIGH_THRESHOLD, DEFAULT_LOW_THRESHOLD, score_low_intensity
from .pinning import FlipFlopGuard, PIN_BREAK_OVERFLOW, evaluate_pin
from .safe_default import safe_default
from .scoring import FrugalityWeights
from .triage import VERDICT_COMPLEX, VERDICT_TRIVIAL, triage
from .types import (
    PIN_REASON_AUTO,
    PIN_REASON_LOOP_ESCALATION,
    CandidateScore,
    ModelProfile,
    RoutingDecision,
    RoutingRequest,
    SessionPin,
    TIER_ECONOMICAL,
    TIER_FRONTIER,
    TIER_LOCAL,
    normalize_tier,
)

MODE_OFF = "off"
MODE_SUGGEST = "suggest"
MODE_AUTO = "auto"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RouterConfig:
    frugality: FrugalityWeights = FrugalityWeights()
    loop_escalation_threshold: int = 3
    pin_enabled: bool = True
    dwell_turns: int = 3
    switch_margin: float = 0.25
    safe_tier: str = TIER_ECONOMICAL
    context_safety_margin: float = DEFAULT_SAFETY_MARGIN
    low_intensity_high: float = DEFAULT_HIGH_THRESHOLD
    low_intensity_low: float = DEFAULT_LOW_THRESHOLD
    local_zero: LocalZeroConfig = LocalZeroConfig()
    hydra_enabled: bool = True
    flip_flop_threshold: int = 3
    min_output_tokens: int = DEFAULT_MIN_OUTPUT_TOKENS
    health: HealthConfig = HealthConfig()
    stream_failover_enabled: bool = True
    stream_failover_max_alternates: int = 1


def _redact_error(error: BaseException, prompt_text: str) -> str:
    """Error string with prompt content removed (port of redactPromptFromError)."""
    message = str(error) or type(error).__name__
    if prompt_text:
        message = message.replace(prompt_text[:200], "<redacted>")
    return message[:300]


class RouterPipeline:
    """Ordered early-exit routing pipeline over an explicit candidate fleet."""

    def __init__(
        self,
        fleet,
        config: RouterConfig = RouterConfig(),
        state=None,
        telemetry=None,
        health=None,
    ):
        self._fleet = tuple(fleet)
        self._config = config
        self._state = state
        self._telemetry = telemetry
        self._health = health
        self._flip_flop = FlipFlopGuard(config.flip_flop_threshold)

    @property
    def fleet(self):
        return self._fleet

    @property
    def health(self):
        return self._health

    # ─── Public entry ────────────────────────────────────────────────────────

    def route(
        self,
        request: RoutingRequest,
        *,
        current_model: str,
        mode: str = MODE_OFF,
        dry_run: bool = False,
    ) -> RoutingDecision:
        """Choose at most one model for a turn. Modes: off, suggest, auto.

        ``dry_run`` executes the same stages while suppressing pin, health,
        flip-flop, and telemetry writes; it powers ``hermes router explain``.
        """
        if mode not in {MODE_OFF, MODE_SUGGEST, MODE_AUTO}:
            mode = MODE_OFF
        start = time.monotonic()

        if mode == MODE_OFF:
            return RoutingDecision(current_model, "disabled", "routing_disabled", explanation="routing disabled")

        try:
            health_excluded: set[str] = set()
            while True:
                decision = self._run_pipeline(
                    request,
                    current_model,
                    health_excluded=health_excluded,
                    mutate_state=not dry_run,
                )
                if (
                    dry_run
                    or mode != MODE_AUTO
                    or self._health is None
                    or decision.stage in {"fallback", "pipeline_error", "force_model"}
                ):
                    break
                selected = next(
                    (model for model in self._fleet if model.id == decision.selected_model),
                    None,
                )
                if selected is None or self._health.claim_dispatch(
                    selected.provider, selected.id
                ):
                    break
                health_excluded.add(selected.id)
                if len(health_excluded) >= len(self._fleet):
                    decision = self._decide_named(
                        current_model,
                        "fallback",
                        "health_probe_busy",
                        request,
                    )
                    break
        except Exception as exc:  # belt-and-braces; stages already guard
            decision = self._fallback_decision(
                request, current_model, "pipeline_error", _redact_error(exc, request.prompt_text)
            )

        latency_ms = (time.monotonic() - start) * 1000.0
        decision = replace(decision, routing_latency_ms=round(latency_ms, 3))

        # Mode semantics: suggest computes and records but retains the model.
        if mode == MODE_SUGGEST and decision.selected_model != current_model:
            decision = replace(
                decision,
                selected_model=current_model,
                suggestion=decision.selected_model,
                pinned=False,
                explanation=f"{decision.explanation}; suggestion only — current model retained",
            )
        elif mode == MODE_SUGGEST:
            decision = replace(decision, pinned=False)

        # Pin bookkeeping only applies to turns that will actually run on the
        # selected model (auto mode).
        if (
            not dry_run
            and mode == MODE_AUTO
            and self._state is not None
            and self._config.pin_enabled
        ):
            self._update_pin(request, decision)

        if not dry_run and self._telemetry is not None:
            try:
                self._telemetry.record(
                    request,
                    decision,
                    mode=mode,
                    session_id=request.session_id,
                )
            except Exception as exc:
                # Telemetry is never a routing gate. Avoid logging the exception
                # message because third-party recorders may include prompt text.
                logger.warning(
                    "Model router telemetry write failed: mode=%s stage=%s error_type=%s",
                    mode,
                    decision.stage,
                    type(exc).__name__,
                )
        return decision

    # ─── Pipeline ────────────────────────────────────────────────────────────

    def _run_pipeline(
        self,
        request: RoutingRequest,
        current_model: str,
        *,
        health_excluded: set[str] | None = None,
        mutate_state: bool = True,
    ) -> RoutingDecision:
        turn_type = request.turn_type or (
            classify_turn_envelope(request.messages) if request.messages else "main_loop"
        )
        request = replace(request, turn_type=turn_type)
        cfg = self._config
        basic_features = {
            "estimated_input_tokens": request.estimated_tokens(),
            "min_output_tokens": cfg.min_output_tokens,
            "has_images": bool(request.has_images),
        }

        # 1. force_model — explicit operator override short-circuits gates.
        if request.force_model_id:
            for model in self._fleet:
                if model.id == request.force_model_id:
                    return self._decide(
                        model,
                        "force_model",
                        "operator_override",
                        request,
                        features=basic_features,
                    )
            return self._decide_named(
                current_model,
                "force_model",
                "forced_model_not_in_fleet",
                request,
                features=basic_features,
            )

        # Compute candidate eligibility before an early escalation decision so
        # every automatic target (including a frontier escalation) reserves
        # context plus output headroom and respects an open health circuit.
        fit = filter_fleet_by_context_fit(
            self._fleet,
            request,
            cfg.context_safety_margin,
            cfg.min_output_tokens,
        )
        excluded = health_excluded or set()
        health_rejected = []
        eligible = []
        for profile in fit.effective_fleet:
            unavailable = profile.id in excluded
            if not unavailable and self._health is not None:
                unavailable = not self._health.is_available(profile.provider, profile.id)
            if unavailable:
                health_rejected.append(
                    CandidateScore(profile.id, rejected_reason="health_circuit_open")
                )
            else:
                eligible.append(profile)
        fleet = tuple(eligible)
        all_rejected = tuple(fit.rejected) + tuple(health_rejected)
        rejected_notes = tuple(
            f"{candidate.model_id}: {candidate.rejected_reason}"
            for candidate in all_rejected
        )

        pin = self._state.load_pin(request.session_id) if (
            self._state and cfg.pin_enabled
        ) else None

        # 2. loop_escalation — repeated tool failures escalate to an eligible frontier.
        if pin is not None:
            result = self._stage(
                "loop_escalation",
                evaluate_loop_escalation,
                pin,
                request,
                fleet,
                cfg.loop_escalation_threshold,
            )
            if result is not None:
                if (
                    mutate_state
                    and result.updated_pin is not None
                    and self._state is not None
                ):
                    self._state.save_pin(result.updated_pin)
                    pin = result.updated_pin
                if result.should_escalate and result.escalation_target is not None:
                    if mutate_state and self._state is not None:
                        self._state.save_pin(replace(
                            pin,
                            pinned_model_id=result.escalation_target.id,
                            pin_reason=PIN_REASON_LOOP_ESCALATION,
                            turns_held=0,
                            updated_at=time.time(),
                        ))
                    return self._decide(
                        result.escalation_target,
                        "loop_escalation",
                        result.reason,
                        request,
                        pinned=True,
                        rejected=rejected_notes,
                        features=basic_features,
                    )

        # 3. context_fit and health have narrowed ``fleet`` but never decide.
        guard_tier = self._flip_flop.is_tier_pinned(request.session_id)
        if guard_tier:
            tier_fleet = tuple(model for model in fleet if model.tier == guard_tier)
            if tier_fleet:
                fleet = tier_fleet

        triage_result = triage(request.prompt_text)
        requirements = hydra.build_requirement_vector(request, triage_result)
        low = score_low_intensity(
            request,
            triage_result,
            requirements.magnitude,
            high_threshold=cfg.low_intensity_high,
            low_threshold=cfg.low_intensity_low,
        )
        features = {
            **basic_features,
            "low_intensity_score": low.score,
            "triage": {
                "verdict": triage_result.verdict,
                "reason": triage_result.reason_code,
            },
            "requirements": {
                "reasoning": requirements.reasoning,
                "code_gen": requirements.code_gen,
                "tool_use": requirements.tool_use,
            },
        }
        mo = hydra.hydra_match(fleet, requirements, request, cfg.frugality)
        candidates = mo.candidates + tuple(
            # Keep eligibility rejections visible beside scored candidates.
            # Multi-objective candidates already include unhealthy/capability rejects.
            CandidateScore(
                candidate.model_id,
                rejected_reason=candidate.rejected_reason,
                shortfall=candidate.shortfall,
            )
            for candidate in all_rejected
        )

        # 4. low_intensity — cheap adequate pick for low-intensity turns.
        if low.is_high:
            economical = [model for model in fleet if model.tier == TIER_ECONOMICAL]
            pick = select_lowest_cost_model(economical)
            if pick is not None:
                return self._decide(
                    pick,
                    "low_intensity",
                    "low_intensity_gate",
                    request,
                    rejected=rejected_notes,
                    candidates=candidates,
                    features=features,
                )

        # 5. session_pin — valid pin holds unless a break condition fires.
        if pin is not None:
            active_pinned = next(
                (model for model in fleet if model.id == pin.pinned_model_id),
                None,
            )
            best_alt = next(
                (model for model in fleet if mo.selected and model.id == mo.selected.model_id),
                None,
            )
            best_alt_score = mo.selected.composite_score if mo.selected else 0.0
            pinned_score = next(
                (
                    scored.composite_score
                    for scored in mo.candidates
                    if scored.model_id == pin.pinned_model_id
                ),
                0.0,
            )
            evaluation = evaluate_pin(
                pin,
                active_pinned,
                best_alt,
                compaction_flag=request.compaction_flag,
                pinned_fits_context=active_pinned is not None,
                dwell_turns=cfg.dwell_turns,
                switch_margin=cfg.switch_margin,
                pinned_score=pinned_score,
                alternative_score=best_alt_score,
                estimated_input_tokens=request.estimated_tokens(),
                warm_prefix_tokens=request.estimated_tokens(),
            )
            if evaluation.hold and active_pinned is not None:
                return self._decide(
                    active_pinned,
                    "session_pin",
                    "pin_hold",
                    request,
                    pinned=True,
                    rejected=rejected_notes,
                    candidates=candidates,
                    features=features,
                )
            if evaluation.switch_to is not None and evaluation.reason != PIN_BREAK_OVERFLOW:
                return self._decide(
                    evaluation.switch_to,
                    "session_pin",
                    evaluation.reason,
                    request,
                    rejected=rejected_notes,
                    candidates=candidates,
                    features=features,
                )
            if mutate_state and self._state is not None:
                self._state.clear_pin(request.session_id)
            pin = None

        # 6. triage — fast paths.
        if triage_result.verdict == VERDICT_COMPLEX:
            frontier = [
                model for model in fleet
                if model.tier == TIER_FRONTIER and model.healthy
            ]
            if frontier:
                pick = sorted(frontier, key=lambda model: (-model.quality, model.id))[0]
                return self._decide(
                    pick,
                    "triage",
                    triage_result.reason_code,
                    request,
                    rejected=rejected_notes,
                    candidates=candidates,
                    features=features,
                )

        # 7. local_zero — opt-in local inference for trivial/low-intensity turns.
        if cfg.local_zero.enabled and (
            triage_result.verdict == VERDICT_TRIVIAL or low.is_high
        ):
            local_model = next(
                (
                    model for model in fleet
                    if model.tier == TIER_LOCAL
                    and model.healthy
                    and (not cfg.local_zero.model or model.id == cfg.local_zero.model)
                ),
                None,
            )
            if local_model is not None and self._stage(
                "local_zero_ping", ping_local_services, cfg.local_zero
            ):
                return self._decide(
                    local_model,
                    "local_zero",
                    "local_ready",
                    request,
                    rejected=rejected_notes,
                    candidates=candidates,
                    features=features,
                )

        # 8. triage_cloud_fallback — trivial turns not claimed locally.
        if triage_result.verdict == VERDICT_TRIVIAL:
            economical = [model for model in fleet if model.tier == TIER_ECONOMICAL]
            pick = select_lowest_cost_model(economical)
            if pick is not None:
                return self._decide(
                    pick,
                    "triage_cloud_fallback",
                    triage_result.reason_code,
                    request,
                    rejected=rejected_notes,
                    candidates=candidates,
                    features=features,
                )

        # 9. hydra_match — requirement vector + multi-objective selection.
        if cfg.hydra_enabled and mo.selected is not None:
            pick = next(model for model in fleet if model.id == mo.selected.model_id)
            decision = self._decide(
                pick,
                "hydra_match",
                "multi_objective",
                request,
                rejected=rejected_notes,
                candidates=candidates,
                features=features,
            )
            if mutate_state:
                self._flip_flop.observe_tier(request.session_id, pick.tier)
            return decision

        # 10. safe_default — context, headroom, and health aware.
        pick = safe_default(
            fleet,
            request,
            safe_tier=cfg.safe_tier,
            safety_margin=cfg.context_safety_margin,
            min_output_tokens=cfg.min_output_tokens,
        )
        if pick is not None:
            return self._decide(
                pick,
                "safe_default",
                "safe_tier_default",
                request,
                rejected=rejected_notes,
                candidates=candidates,
                features=features,
            )

        # 11. context_overflow — try a larger healthy circuit without ever
        # dispatching a model that still lacks the required output floor.
        if fit.rejected:
            overflow_fleet = tuple(
                model for model in self._fleet
                if model.id not in excluded
                and (
                    self._health is None
                    or self._health.is_available(model.provider, model.id)
                )
            )
            overflow = resolve_context_overflow_fallback(
                overflow_fleet,
                request.estimated_tokens(),
                preferred_provider=None,
                safety_margin=cfg.context_safety_margin,
                min_output_tokens=cfg.min_output_tokens,
            )
            if overflow.kind == "selected" and overflow.model is not None:
                return self._decide(
                    overflow.model,
                    "context_overflow",
                    overflow.reason_code,
                    request,
                    rejected=rejected_notes,
                    candidates=candidates,
                    features=features,
                )

        return self._decide_named(
            current_model,
            "fallback",
            "no_candidate_eligible",
            request,
            rejected=rejected_notes,
            candidates=candidates,
            features=features,
        )

    def failover_candidates(
        self,
        request: RoutingRequest,
        selected_model: str,
        *,
        limit: int | None = None,
    ) -> tuple[ModelProfile, ...]:
        """Return deterministic, currently eligible stream-failover targets.

        This is selection only: it performs no health probe claim and no state,
        telemetry, pin, or network mutation. The gateway passes these targets
        to Hermes' existing provider-failure retry loop.
        """
        if not self._config.stream_failover_enabled:
            return ()
        cap = self._config.stream_failover_max_alternates if limit is None else limit
        cap = max(0, int(cap))
        if cap == 0:
            return ()
        fit = filter_fleet_by_context_fit(
            self._fleet,
            request,
            self._config.context_safety_margin,
            self._config.min_output_tokens,
        )
        fleet = tuple(
            model for model in fit.effective_fleet
            if model.id != selected_model
            and model.healthy
            and (
                self._health is None
                or self._health.is_available(model.provider, model.id)
            )
        )
        triage_result = triage(request.prompt_text)
        requirements = hydra.build_requirement_vector(request, triage_result)
        scored = hydra.hydra_match(
            fleet, requirements, request, self._config.frugality
        )
        score_by_id = {
            candidate.model_id: candidate.composite_score
            for candidate in scored.candidates
            if candidate.rejected_reason is None
        }
        selected = next(
            (model for model in self._fleet if model.id == selected_model),
            None,
        )
        viable = [model for model in fleet if model.id in score_by_id]
        viable.sort(
            key=lambda model: (
                0 if selected is not None and model.tier == selected.tier else 1,
                -score_by_id[model.id],
                model.id,
            )
        )
        return tuple(viable[:cap])

    # ─── Stage guard ─────────────────────────────────────────────────────────

    def _stage(self, name: str, fn, *args):
        """Run a pure stage function; exceptions degrade to None (no decision)."""
        try:
            return fn(*args)
        except Exception:
            return None

    # ─── Decision builders ───────────────────────────────────────────────────

    def _decide(self, model: ModelProfile, stage: str, reason: str, request: RoutingRequest,
                *, pinned: bool = False, rejected: tuple = (), candidates: tuple = (), features: dict = None) -> RoutingDecision:
        return RoutingDecision(
            selected_model=model.id,
            stage=stage,
            reason_code=reason,
            explanation=f"stage={stage} reason={reason} tier={model.tier} provider={model.provider or 'default'}",
            rejected=rejected,
            candidates=candidates,
            turn_type=request.turn_type or "unknown",
            pinned=pinned,
            features=features or {},
        )

    def _decide_named(
        self,
        model_id: str,
        stage: str,
        reason: str,
        request: RoutingRequest,
        *,
        rejected: tuple = (),
        candidates: tuple = (),
        features: dict | None = None,
    ) -> RoutingDecision:
        return RoutingDecision(
            selected_model=model_id,
            stage=stage,
            reason_code=reason,
            explanation=f"stage={stage} reason={reason}",
            rejected=rejected,
            candidates=candidates,
            turn_type=request.turn_type or "unknown",
            features=features or {},
        )

    def _fallback_decision(self, request: RoutingRequest, current_model: str, stage: str, error: str) -> RoutingDecision:
        pick = None
        try:
            eligible = tuple(
                model for model in self._fleet
                if self._health is None
                or self._health.is_available(model.provider, model.id)
            )
            pick = safe_default(
                eligible,
                request,
                safe_tier=self._config.safe_tier,
                safety_margin=self._config.context_safety_margin,
                min_output_tokens=self._config.min_output_tokens,
            )
        except Exception:
            pick = None
        selected = pick.id if pick is not None else current_model
        return RoutingDecision(
            selected_model=selected,
            stage=stage,
            reason_code="stage_error",
            explanation=f"routing stage failed ({error}); fell back to {selected}",
            turn_type=request.turn_type or "unknown",
        )

    # ─── Pin bookkeeping ─────────────────────────────────────────────────────

    def _update_pin(self, request: RoutingRequest, decision: RoutingDecision) -> None:
        """Create/refresh the session pin after an auto-mode decision."""
        if not request.session_id or self._state is None:
            return
        try:
            existing = self._state.load_pin(request.session_id)
            if decision.stage in {"fallback", "disabled"}:
                return
            if existing and existing.pinned_model_id == decision.selected_model:
                self._state.save_pin(replace(existing, turns_held=existing.turns_held + 1, updated_at=time.time()))
            else:
                reason = existing.pin_reason if existing and decision.pinned else PIN_REASON_AUTO
                self._state.save_pin(SessionPin(
                    session_id=request.session_id,
                    pinned_model_id=decision.selected_model,
                    pin_reason=reason,
                    turns_held=1,
                    updated_at=time.time(),
                ))
        except Exception:
            pass


# ─── Config factory ───────────────────────────────────────────────────────────


def profile_from_config(item: dict) -> Optional[ModelProfile]:
    """Build a ModelProfile from one config candidate entry (tolerant)."""
    if not isinstance(item, dict):
        return None
    model = item.get("model")
    if not isinstance(model, str) or not model.strip():
        return None
    quality = float(item.get("quality", 0.5) or 0.5)
    tier = item.get("tier")
    if tier is None:
        tier = TIER_FRONTIER if quality >= 0.9 else TIER_ECONOMICAL
    cost_per_1m = item.get("cost_per_1m")
    return ModelProfile(
        id=model.strip(),
        provider=str(item.get("provider", "") or ""),
        tier=normalize_tier(tier),
        context_window=int(item.get("context_window", 0) or 0),
        reasoning=bool(item.get("reasoning", False)),
        vision=bool(item.get("vision", False)),
        quality=quality,
        cost=float(item.get("cost", 0.5) or 0.5),
        cost_per_1m=float(cost_per_1m) if cost_per_1m is not None else None,
        est_latency_ms=float(item.get("est_latency_ms", 0.0) or 0.0),
        verbosity=float(item.get("verbosity", 1.0) or 1.0),
        healthy=bool(item.get("healthy", True)),
    )


def router_config_from_dict(cfg: dict) -> RouterConfig:
    """Parse the extended model_router config sections (all optional)."""
    cfg = cfg if isinstance(cfg, dict) else {}
    frugality = cfg.get("frugality") or {}
    escalation = cfg.get("loop_escalation") or {}
    pin = cfg.get("session_pin") or {}
    local = cfg.get("local_zero") or {}
    hydra_cfg = cfg.get("hydra") or {}
    headroom = cfg.get("output_headroom") or {}
    health = cfg.get("health") or {}
    stream_failover = cfg.get("stream_failover") or {}
    return RouterConfig(
        frugality=FrugalityWeights(
            lambda_cost=float(frugality.get("lambda_cost", 0.5)),
            lambda_latency=float(frugality.get("lambda_latency", 0.1)),
            lambda_verbosity=float(frugality.get("lambda_verbosity", 0.15)),
        ),
        loop_escalation_threshold=max(1, int(escalation.get("threshold", 3))),
        pin_enabled=bool(pin.get("enabled", True)),
        dwell_turns=max(0, int(pin.get("dwell_turns", 3))),
        switch_margin=float(pin.get("switch_margin", 0.25)),
        safe_tier=normalize_tier(cfg.get("safe_tier"), default=TIER_ECONOMICAL),
        context_safety_margin=max(
            0.0, min(1.0, float(cfg.get("context_safety_margin", DEFAULT_SAFETY_MARGIN)))
        ),
        local_zero=LocalZeroConfig(
            enabled=bool(local.get("enabled", False)),
            endpoints=tuple(local.get("endpoints") or ()),
            model=str(local.get("model") or ""),
            timeout_ms=int(local.get("timeout_ms", 1500)),
        ),
        hydra_enabled=bool(hydra_cfg.get("enabled", True)),
        min_output_tokens=max(
            0, int(headroom.get("min_output_tokens", DEFAULT_MIN_OUTPUT_TOKENS))
        ),
        health=HealthConfig(
            enabled=bool(health.get("enabled", True)),
            failure_threshold=max(1, int(health.get("failure_threshold", 3))),
            reset_timeout_seconds=max(
                0.0, float(health.get("reset_timeout_seconds", 30.0))
            ),
            half_open_successes=max(1, int(health.get("half_open_successes", 2))),
            max_entries=max(1, min(4096, int(health.get("max_entries", 256)))),
        ),
        stream_failover_enabled=bool(stream_failover.get("enabled", True)),
        stream_failover_max_alternates=max(
            0, min(8, int(stream_failover.get("max_alternates", 1)))
        ),
    )


def default_db_path(hermes_home=None) -> Path:
    """Resolve the router state/telemetry DB path under $HERMES_HOME."""
    if hermes_home is None:
        try:
            from hermes_cli.config import get_hermes_home
            hermes_home = get_hermes_home()
        except Exception:
            hermes_home = Path.home() / ".hermes"
    return Path(hermes_home) / "state" / "model_router" / "router.db"


def pipeline_from_config(
    router_cfg: dict,
    *,
    hermes_home=None,
    read_only: bool = False,
) -> Optional["RouterPipeline"]:
    """Build a pipeline from the model_router config section.

    Returns None when no usable candidates exist. State and telemetry are
    created best-effort; their failure never blocks routing.
    """
    if not isinstance(router_cfg, dict):
        return None
    raw_candidates = router_cfg.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        return None

    profiles = [p for p in (profile_from_config(item) for item in raw_candidates) if p is not None]
    if not profiles:
        return None

    config = router_config_from_dict(router_cfg)
    telemetry_cfg = router_cfg.get("telemetry") or {}
    db_path = telemetry_cfg.get("db_path") or default_db_path(hermes_home)

    state = None
    telemetry = None
    health = None
    if config.pin_enabled:
        try:
            from .state import RouterStateStore
            store = RouterStateStore(db_path, read_only=read_only)
            state = store if store.available else None
        except Exception:
            state = None
    if not read_only and bool(telemetry_cfg.get("enabled", True)):
        try:
            from .telemetry import RouterTelemetry
            recorder = RouterTelemetry(db_path)
            telemetry = recorder if recorder.available else None
        except Exception:
            telemetry = None
    if config.health.enabled:
        try:
            from .health import RouterHealthStore
            health_store = RouterHealthStore(
                db_path, config.health, read_only=read_only
            )
            health = health_store if health_store.available else None
        except Exception:
            health = None

    return RouterPipeline(
        profiles,
        config,
        state=state,
        telemetry=telemetry,
        health=health,
    )
