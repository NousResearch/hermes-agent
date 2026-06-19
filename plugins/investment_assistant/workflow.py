"""Recoverable workflow state machine for theme discovery."""

from __future__ import annotations

import re
from typing import Any

from .adapters import FutuAdapterError, MarketDataAdapter, canonical_theme_key, normalize_market_symbol
from .candidate_triage import (
    CandidateTriageArtifact,
    CandidateTriagePlanArtifact,
    build_candidate_triage_artifact,
    build_candidate_triage_plan,
    select_triage_strategy,
)
from .discovery_v1 import build_ai_discovery_v1_plan
from .lightweight_enrichment import LightweightEnrichmentArtifact, build_lightweight_enrichment_artifact
from .portfolio_architect import PortfolioArchitectResult, build_portfolio_maps_from_triage
from .portfolio_revision import (
    PortfolioMapRevision,
    PortfolioRevisionPatch,
    build_portfolio_revision_from_artifacts,
)
from .schemas import (
    HumanActionKind,
    InvestmentPolicy,
    PortfolioMap,
    ThemeDiscoveryPlan,
    WorkflowState,
    WorkflowStatus,
)
from .sec_provider import SecFilingsProvider
from .storage import InvestmentAssistantStore
from .websearch_discovery import (
    build_native_websearch_discovery_plan,
    build_websearch_discovery_plan,
    is_native_websearch_discovery_mode,
    is_websearch_discovery_mode,
)


class InvestmentAssistantWorkflow:
    def __init__(
        self,
        store: InvestmentAssistantStore | None = None,
        market_data: MarketDataAdapter | None = None,
        portfolio: Any | None = None,
        sec_filings: Any | None = None,
    ):
        self.store = store or InvestmentAssistantStore()
        self.market_data = market_data or MarketDataAdapter()
        # Kept for constructor compatibility with earlier MVP experiments.
        # The public workflow does not read holdings in discovery-only mode.
        self.portfolio = portfolio
        self.sec_filings = sec_filings or SecFilingsProvider()

    def run(
        self,
        tenant: str,
        action: str,
        session_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = payload or {}
        action = (action or "").strip().lower()
        if action == "start":
            if payload.get("discovery_only"):
                return self._discover_only(tenant, payload)
            return self._start_candidate_triage_plan(tenant, payload)
        if action in {"discover", "theme_discovery", "discovery"}:
            return self._discover_only(tenant, payload)
        if action in {"answer_human_input", "select_option", "select_triage_strategy"}:
            session = self._require_session(tenant, session_id)
            if not self.store.pending_human_action(session["session_id"]) and session.get("state") in {
                WorkflowState.NEEDS_PORTFOLIO_MAP_REVIEW.value,
                WorkflowState.NEEDS_PORTFOLIO_MAP_SELECTION.value,
            }:
                return self._select_portfolio_map(session, payload)
            return self._answer_human_input(session, payload)
        if action in {"build_portfolio_maps", "portfolio_maps", "build_portfolio", "architect"}:
            session = self._require_session(tenant, session_id)
            return self._build_portfolio_maps(session, payload)
        if action in {"revise_portfolio_map", "revise_map", "modify_portfolio_map", "adjust_portfolio_map"}:
            session = self._require_session(tenant, session_id)
            return self._revise_portfolio_map(session, payload)
        if action == "continue":
            session = self._require_session(tenant, session_id)
            if self.store.pending_human_action(session["session_id"]):
                if payload:
                    return self._answer_human_input(session, payload)
                return self._status(session)
            if session.get("state") in {
                WorkflowState.CANDIDATE_TRIAGE_COMPLETE.value,
                WorkflowState.DRAFTING_TARGET_PORTFOLIO_MAPS.value,
            } and session.get("status") in {WorkflowStatus.COMPLETED.value, WorkflowStatus.FAILED.value}:
                return self._build_portfolio_maps(session, payload)
            if payload:
                return self._answer_human_input(session, payload)
            return self._status(session)
        if action == "status":
            session = self._require_session(tenant, session_id)
            return self._status(session)
        if action == "cancel":
            session = self._require_session(tenant, session_id)
            updated = self.store.update_session(
                session["session_id"],
                state=WorkflowState.CANCELLED.value,
                status=WorkflowStatus.CANCELLED.value,
            )
            return self._response(updated, data={"cancelled": True})
        return {
            "success": False,
            "error": f"Unknown ia_portfolio_workflow action: {action}",
            "allowed_actions": [
                "start",
                "discover",
                "answer_human_input",
                "select_option",
                "build_portfolio_maps",
                "revise_portfolio_map",
                "continue",
                "status",
                "cancel",
            ],
        }

    def _parse_start_payload(self, payload: dict[str, Any]) -> dict[str, Any] | dict[str, bool | str]:
        theme_raw = str(payload.get("theme_key") or payload.get("theme") or payload.get("topic") or "").strip()
        if not theme_raw:
            return {
                "success": False,
                "error": "payload.theme or payload.theme_key is required",
            }
        try:
            theme = canonical_theme_key(theme_raw)
        except FutuAdapterError:
            theme = theme_raw
        market = _market_from_payload(payload)
        theme_description = str(payload.get("theme_description") or payload.get("description") or "").strip()
        if payload.get("topic") and not theme_description:
            theme_description = str(payload.get("topic") or "").strip()
        required_symbols = _normalize_symbols(payload.get("required_symbols") or payload.get("base_symbols") or [], market)
        if not required_symbols and isinstance(payload.get("required_symbol"), str):
            required_symbols = _normalize_symbols([payload["required_symbol"]], market)
        if not required_symbols and isinstance(payload.get("base_symbol"), str):
            required_symbols = _normalize_symbols([payload["base_symbol"]], market)
        if not required_symbols and isinstance(payload.get("required_symbols"), str):
            required_symbols = _normalize_symbols([payload["required_symbols"]], market)
        if not required_symbols and isinstance(payload.get("base_symbols"), str):
            required_symbols = _normalize_symbols([payload["base_symbols"]], market)
        return {
            "theme_raw": theme_raw,
            "theme": theme,
            "market": market,
            "theme_description": theme_description,
            "required_symbols": required_symbols,
        }

    def _start_candidate_triage_plan(self, tenant: str, payload: dict[str, Any]) -> dict[str, Any]:
        parsed = self._parse_start_payload(payload)
        if parsed.get("success") is False:
            return parsed
        theme_raw = str(parsed["theme_raw"])
        theme = str(parsed["theme"])
        market = str(parsed["market"])
        theme_description = str(parsed["theme_description"])
        required_symbols = list(parsed["required_symbols"])

        session = self.store.create_session(
            tenant_id=tenant,
            theme=theme,
            state=WorkflowState.NEW.value,
            status=WorkflowStatus.ACTIVE.value,
        )
        session_id = session["session_id"]
        initial_artifact = self.store.add_artifact(
            session_id,
            "initial_request",
            {
                "theme": theme,
                "market": market,
                "theme_description": theme_description,
                "required_symbols": required_symbols,
                "payload": {
                    **payload,
                    "theme": theme,
                    "market": market,
                    "theme_description": theme_description,
                    "required_symbols": required_symbols,
                },
                "candidate_pool_must_ignore_holdings": True,
                "v1_scope": "discovery_lightweight_triage_plan",
            },
        )
        policy = InvestmentPolicy(
            theme=theme,
            theme_description=theme_description,
            required_symbols=required_symbols,
            objective=_choice(payload.get("objective"), {"balanced", "growth", "income"}, "balanced"),
            risk_level=_choice(payload.get("risk_level"), {"conservative", "moderate", "aggressive"}, "moderate"),
            target_portfolio_weight=_float_default(payload.get("target_portfolio_weight"), 0.15),
            cash_reserve=_float_default(payload.get("cash_reserve"), 0.10),
            single_name_limit=_float_default(payload.get("single_name_limit"), 0.15),
            allow_options=bool(payload.get("allow_options", False)),
            notes=str(payload.get("notes") or ""),
        )
        policy_artifact = self.store.add_artifact(session_id, "policy", policy)
        self._record_completed_state_run(
            session_id,
            WorkflowState.NEW,
            input_payload={"action": "start", "theme_raw": theme_raw, "market": market},
            output_payload={
                "initial_request_artifact_id": initial_artifact["artifact_id"],
                "policy_artifact_id": policy_artifact["artifact_id"],
                "next_state": WorkflowState.EXPANDING_THEME.value,
            },
            artifact_ids=[initial_artifact["artifact_id"], policy_artifact["artifact_id"]],
        )

        discovery_run = self._begin_state_run(
            session,
            WorkflowState.EXPANDING_THEME,
            {
                "theme": theme,
                "market": market,
                "theme_description": theme_description,
                "required_symbols": required_symbols,
                "mode": _discovery_mode_label(payload, suffix="to_triage_plan"),
            },
        )
        try:
            discovery = self._build_theme_discovery(
                payload,
                theme,
                market=market,
                theme_description=theme_description,
                required_symbols=required_symbols,
            )
            discovery_artifact = self.store.add_artifact(session_id, "theme_discovery", discovery)
            self._finish_state_run(
                discovery_run,
                output_payload={
                    "theme_discovery_artifact_id": discovery_artifact["artifact_id"],
                    "domain_count": len(discovery.domain_tree),
                    "coverage_requirement_count": len(discovery.coverage_requirements),
                    "seed_symbol_count": len(discovery.seed_symbols),
                    "research_source_count": len(discovery.research_trace),
                    "search_query_count": len(discovery.search_queries),
                    "filter_plan_count": len(discovery.filter_plans_by_layer),
                    "executed_filter_probe_count": len(discovery.executed_filter_probes),
                    "layer_filter_audit_count": len(discovery.layer_filter_audits),
                    "omission_count": len(discovery.omissions_to_investigate),
                    "warnings": discovery.warnings,
                    "next_state": WorkflowState.BUILDING_MARKET_ARTIFACTS.value,
                },
                artifact_ids=[discovery_artifact["artifact_id"]],
            )
        except Exception as exc:
            return self._fail_workflow(session, discovery_run, WorkflowState.EXPANDING_THEME, exc)

        lightweight_run = self._begin_state_run(
            session,
            WorkflowState.BUILDING_MARKET_ARTIFACTS,
            {
                "theme_discovery_artifact_id": discovery_artifact["artifact_id"],
                "mode": "futu_lightweight_enrichment",
            },
        )
        try:
            lightweight = build_lightweight_enrichment_artifact(discovery)
            lightweight_artifact = self.store.add_artifact(session_id, "futu_lightweight_enrichment", lightweight)
            self._finish_state_run(
                lightweight_run,
                output_payload={
                    "futu_lightweight_enrichment_artifact_id": lightweight_artifact["artifact_id"],
                    "candidate_count": len(lightweight.candidates),
                    "check_summary": lightweight.check_summary,
                    "warnings": lightweight.warnings,
                    "next_state": WorkflowState.BUILDING_UNBIASED_CANDIDATE_POOL.value,
                },
                artifact_ids=[lightweight_artifact["artifact_id"]],
            )
        except Exception as exc:
            return self._fail_workflow(session, lightweight_run, WorkflowState.BUILDING_MARKET_ARTIFACTS, exc)

        plan_run = self._begin_state_run(
            session,
            WorkflowState.BUILDING_UNBIASED_CANDIDATE_POOL,
            {
                "theme_discovery_artifact_id": discovery_artifact["artifact_id"],
                "futu_lightweight_enrichment_artifact_id": lightweight_artifact["artifact_id"],
                "mode": "candidate_triage_plan",
            },
        )
        try:
            triage_plan = build_candidate_triage_plan(discovery, lightweight)
            triage_plan_artifact = self.store.add_artifact(session_id, "candidate_triage_plan", triage_plan)
            self._finish_state_run(
                plan_run,
                output_payload={
                    "candidate_triage_plan_artifact_id": triage_plan_artifact["artifact_id"],
                    "strategy_option_count": len(triage_plan.strategy_options),
                    "recommended_option_id": triage_plan.recommended_option_id,
                    "next_state": WorkflowState.NEEDS_CANDIDATE_TRIAGE_STRATEGY.value,
                },
                artifact_ids=[triage_plan_artifact["artifact_id"]],
            )
            human_action = self.store.create_human_action(
                session_id,
                WorkflowState.NEEDS_CANDIDATE_TRIAGE_STRATEGY.value,
                HumanActionKind.SELECT_CANDIDATE_TRIAGE_STRATEGY.value,
                prompt=_triage_plan_human_prompt(triage_plan.model_dump(mode="json")),
                response_schema=_triage_strategy_response_schema(triage_plan.model_dump(mode="json")),
            )
            session = self.store.update_session(
                session_id,
                state=WorkflowState.NEEDS_CANDIDATE_TRIAGE_STRATEGY.value,
                status=WorkflowStatus.WAITING_FOR_HUMAN.value,
                current_artifact_id=triage_plan_artifact["artifact_id"],
            )
            return self._response(
                session,
                data={
                    "message": "Candidate triage strategy plan generated. Waiting for user strategy selection before final triage.",
                    "theme_discovery_artifact_id": discovery_artifact["artifact_id"],
                    "futu_lightweight_enrichment_artifact_id": lightweight_artifact["artifact_id"],
                    "candidate_triage_plan_artifact_id": triage_plan_artifact["artifact_id"],
                    "candidate_triage_plan": triage_plan.model_dump(mode="json"),
                    "candidate_symbols": _symbols_from_discovery_and_lightweight(discovery, lightweight),
                    "state_runs": self._state_run_summaries(session_id),
                },
                warnings=[*discovery.warnings, *lightweight.warnings, *triage_plan.warnings],
            )
        except Exception as exc:
            return self._fail_workflow(session, plan_run, WorkflowState.BUILDING_UNBIASED_CANDIDATE_POOL, exc)

    def _discover_only(self, tenant: str, payload: dict[str, Any]) -> dict[str, Any]:
        parsed = self._parse_start_payload(payload)
        if parsed.get("success") is False:
            return parsed
        theme_raw = str(parsed["theme_raw"])
        theme = str(parsed["theme"])
        market = str(parsed["market"])
        theme_description = str(parsed["theme_description"])
        required_symbols = list(parsed["required_symbols"])

        session = self.store.create_session(
            tenant_id=tenant,
            theme=theme,
            state=WorkflowState.NEW.value,
            status=WorkflowStatus.ACTIVE.value,
        )
        session_id = session["session_id"]
        initial_artifact = self.store.add_artifact(
            session_id,
            "initial_request",
            {
                "theme": theme,
                "market": market,
                "theme_description": theme_description,
                "required_symbols": required_symbols,
                "payload": {
                    **payload,
                    "theme": theme,
                    "market": market,
                    "theme_description": theme_description,
                    "required_symbols": required_symbols,
                    "discovery_only": True,
                },
                "candidate_pool_must_ignore_holdings": True,
                "v1_scope": "theme_discovery_only",
            },
        )
        policy = InvestmentPolicy(
            theme=theme,
            theme_description=theme_description,
            required_symbols=required_symbols,
            objective=_choice(payload.get("objective"), {"balanced", "growth", "income"}, "balanced"),
            risk_level=_choice(payload.get("risk_level"), {"conservative", "moderate", "aggressive"}, "moderate"),
            target_portfolio_weight=_float_default(payload.get("target_portfolio_weight"), 0.15),
            cash_reserve=_float_default(payload.get("cash_reserve"), 0.10),
            single_name_limit=_float_default(payload.get("single_name_limit"), 0.15),
            allow_options=bool(payload.get("allow_options", False)),
            notes=str(payload.get("notes") or ""),
        )
        policy_artifact = self.store.add_artifact(session_id, "policy", policy)
        self._record_completed_state_run(
            session_id,
            WorkflowState.NEW,
            input_payload={"action": "discover", "theme_raw": theme_raw, "market": market},
            output_payload={
                "initial_request_artifact_id": initial_artifact["artifact_id"],
                "policy_artifact_id": policy_artifact["artifact_id"],
                "next_state": WorkflowState.EXPANDING_THEME.value,
            },
            artifact_ids=[initial_artifact["artifact_id"], policy_artifact["artifact_id"]],
        )

        discovery_run = self._begin_state_run(
            session,
            WorkflowState.EXPANDING_THEME,
            {
                "theme": theme,
                "market": market,
                "theme_description": theme_description,
                "required_symbols": required_symbols,
                "mode": _discovery_mode_label(payload, suffix="only"),
            },
        )
        try:
            discovery = self._build_theme_discovery(
                payload,
                theme,
                market=market,
                theme_description=theme_description,
                required_symbols=required_symbols,
            )
            discovery_artifact = self.store.add_artifact(session_id, "theme_discovery", discovery)
            self._finish_state_run(
                discovery_run,
                output_payload={
                    "theme_discovery_artifact_id": discovery_artifact["artifact_id"],
                    "domain_count": len(discovery.domain_tree),
                    "coverage_requirement_count": len(discovery.coverage_requirements),
                    "seed_symbol_count": len(discovery.seed_symbols),
                    "research_source_count": len(discovery.research_trace),
                    "search_query_count": len(discovery.search_queries),
                    "filter_plan_count": len(discovery.filter_plans_by_layer),
                    "executed_filter_probe_count": len(discovery.executed_filter_probes),
                    "layer_filter_audit_count": len(discovery.layer_filter_audits),
                    "omission_count": len(discovery.omissions_to_investigate),
                    "warnings": discovery.warnings,
                    "stopped_before": [
                        WorkflowState.BUILDING_UNBIASED_CANDIDATE_POOL.value,
                        WorkflowState.VALIDATING_EVIDENCE.value,
                        WorkflowState.DRAFTING_TARGET_PORTFOLIO_MAPS.value,
                    ],
                },
                artifact_ids=[discovery_artifact["artifact_id"]],
            )
            self._record_completed_state_run(
                session_id,
                WorkflowState.THEME_DISCOVERY_COMPLETE,
                input_payload={"action": "stop_after_theme_discovery"},
                output_payload={
                    "theme_discovery_artifact_id": discovery_artifact["artifact_id"],
                    "v1_scope": "theme_discovery_only",
                },
                artifact_ids=[discovery_artifact["artifact_id"]],
            )
            session = self.store.update_session(
                session_id,
                state=WorkflowState.THEME_DISCOVERY_COMPLETE.value,
                status=WorkflowStatus.COMPLETED.value,
                current_artifact_id=discovery_artifact["artifact_id"],
            )
            return self._response(
                session,
                data={
                    "message": "Theme discovery generated. Workflow stopped before downstream enrichment and portfolio-map generation.",
                    "theme_discovery_artifact_id": discovery_artifact["artifact_id"],
                    "theme_discovery": discovery.model_dump(mode="json"),
                    "state_runs": self._state_run_summaries(session_id),
                    "stopped_before": [
                        WorkflowState.BUILDING_UNBIASED_CANDIDATE_POOL.value,
                        WorkflowState.VALIDATING_EVIDENCE.value,
                        WorkflowState.BUILDING_MARKET_ARTIFACTS.value,
                        WorkflowState.REFLECTING_CANDIDATE_POOL.value,
                        WorkflowState.DRAFTING_TARGET_PORTFOLIO_MAPS.value,
                    ],
                },
                warnings=discovery.warnings,
            )
        except Exception as exc:
            self._fail_state_run(discovery_run, exc)
            error_payload = {
                "message": str(exc),
                "failed_state": WorkflowState.EXPANDING_THEME.value,
                "recoverable": True,
            }
            self.store.add_artifact(session_id, "workflow_error", error_payload)
            self.store.add_event(session_id, "error", WorkflowState.EXPANDING_THEME.value, WorkflowState.EXPANDING_THEME.value, error_payload)
            failed_session = self.store.update_session(
                session_id,
                status=WorkflowStatus.FAILED.value,
            )
            return self._response(
                failed_session,
                data={
                    "artifacts": [
                        {
                            "artifact_id": item["artifact_id"],
                            "type": item["type"],
                            "version": item["version"],
                            "created_at": item["created_at"],
                        }
                        for item in self.store.list_artifacts(session_id)
                    ],
                    "latest_error": error_payload,
                    "state_runs": self._state_run_summaries(session_id),
                },
            )

    def _build_theme_discovery(
        self,
        payload: dict[str, Any],
        theme: str,
        *,
        market: str,
        theme_description: str,
        required_symbols: list[str],
    ) -> ThemeDiscoveryPlan:
        if is_websearch_discovery_mode(payload):
            builder = (
                build_native_websearch_discovery_plan
                if is_native_websearch_discovery_mode(payload)
                else build_websearch_discovery_plan
            )
            return builder(
                theme,
                market=market,
                theme_description=theme_description,
                required_symbols=required_symbols,
                max_searches=_optional_int(payload.get("max_searches") or payload.get("websearch_max_searches")),
                max_results=_optional_int(payload.get("max_results") or payload.get("websearch_max_results")),
            )
        return build_ai_discovery_v1_plan(
            theme,
            market=market,
            theme_description=theme_description,
            required_symbols=required_symbols,
        )

    def _answer_human_input(self, session: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        pending = self.store.pending_human_action(session["session_id"])
        if not pending:
            return {
                "success": False,
                "error": "No pending investment assistant human action found for this session.",
                "allowed_actions": ["start", "discover", "status", "cancel"],
            }
        if pending.get("kind") != HumanActionKind.SELECT_CANDIDATE_TRIAGE_STRATEGY.value:
            if pending.get("kind") == HumanActionKind.SELECT_PORTFOLIO_MAP.value:
                return self._select_portfolio_map(session, payload, pending)
            if pending.get("kind") == HumanActionKind.CLARIFY_PORTFOLIO_REVISION.value:
                return self._answer_portfolio_revision_clarification(session, payload, pending)
            if pending.get("kind") == HumanActionKind.CONFIRM_PORTFOLIO_REVISION.value:
                return self._answer_portfolio_revision_confirmation(session, payload, pending)
            return {
                "success": False,
                "error": f"Unsupported human action kind: {pending.get('kind')}",
                "allowed_actions": ["status", "cancel"],
            }

        plan_artifact = self.store.latest_artifact(session["session_id"], "candidate_triage_plan")
        discovery_artifact = self.store.latest_artifact(session["session_id"], "theme_discovery")
        lightweight_artifact = self.store.latest_artifact(session["session_id"], "futu_lightweight_enrichment")
        if not plan_artifact or not discovery_artifact or not lightweight_artifact:
            return {
                "success": False,
                "error": "Cannot continue triage because required prior artifacts are missing.",
                "allowed_actions": ["status", "cancel"],
            }

        plan = CandidateTriagePlanArtifact.model_validate(plan_artifact["payload"])
        selection = _normalize_triage_strategy_answer(payload, plan)
        selection_payload = selection.model_dump(mode="json")
        answered_action = self.store.answer_human_action(pending["action_id"], selection_payload)
        selection_artifact = self.store.add_artifact(session["session_id"], "triage_strategy_selection", selection)
        self.store.add_event(
            session["session_id"],
            "answer_human_input",
            WorkflowState.NEEDS_CANDIDATE_TRIAGE_STRATEGY.value,
            WorkflowState.BUILDING_UNBIASED_CANDIDATE_POOL.value,
                {"action_id": answered_action["action_id"], "selection_artifact_id": selection_artifact["artifact_id"]},
            )

        discovery = ThemeDiscoveryPlan.model_validate(discovery_artifact["payload"])
        lightweight = LightweightEnrichmentArtifact.model_validate(lightweight_artifact["payload"])
        triage_run = self._begin_state_run(
            session,
            WorkflowState.BUILDING_UNBIASED_CANDIDATE_POOL,
            {
                "candidate_triage_plan_artifact_id": plan_artifact["artifact_id"],
                "triage_strategy_selection_artifact_id": selection_artifact["artifact_id"],
                "selected_option_id": selection_payload.get("selected_option_id"),
                "mode": "candidate_triage_from_confirmed_strategy",
            },
        )
        try:
            triage = build_candidate_triage_artifact(discovery, lightweight, triage_strategy=selection)
            triage_artifact = self.store.add_artifact(session["session_id"], "candidate_triage", triage)
            self._finish_state_run(
                triage_run,
                output_payload={
                    "candidate_triage_artifact_id": triage_artifact["artifact_id"],
                    "deep_enrichment_count": len(triage.deep_enrichment_queue),
                    "watchlist_count": len(triage.watchlist),
                    "deferred_count": len(triage.deferred),
                    "rejected_count": len(triage.rejected),
                    "next_state": WorkflowState.CANDIDATE_TRIAGE_COMPLETE.value,
                },
                artifact_ids=[selection_artifact["artifact_id"], triage_artifact["artifact_id"]],
            )
            self._record_completed_state_run(
                session["session_id"],
                WorkflowState.CANDIDATE_TRIAGE_COMPLETE,
                input_payload={"action": "stop_after_candidate_triage"},
                output_payload={
                    "candidate_triage_artifact_id": triage_artifact["artifact_id"],
                    "v1_scope": "candidate_triage_only",
                },
                artifact_ids=[triage_artifact["artifact_id"]],
            )
            updated = self.store.update_session(
                session["session_id"],
                state=WorkflowState.CANDIDATE_TRIAGE_COMPLETE.value,
                status=WorkflowStatus.COMPLETED.value,
                current_artifact_id=triage_artifact["artifact_id"],
            )
            return self._response(
                updated,
                data={
                    "message": "Candidate triage completed. Workflow stopped before SEC/fundamental deep research and portfolio-map generation.",
                    "candidate_triage_plan_artifact_id": plan_artifact["artifact_id"],
                    "triage_strategy_selection_artifact_id": selection_artifact["artifact_id"],
                    "candidate_triage_artifact_id": triage_artifact["artifact_id"],
                    "selected_triage_strategy": selection_payload,
                    "candidate_triage": triage.model_dump(mode="json"),
                    "candidate_symbols": _symbols_from_discovery_and_lightweight(discovery, lightweight),
                    "state_runs": self._state_run_summaries(session["session_id"]),
                    "stopped_before": [
                        WorkflowState.VALIDATING_EVIDENCE.value,
                        WorkflowState.REFLECTING_CANDIDATE_POOL.value,
                        WorkflowState.DRAFTING_TARGET_PORTFOLIO_MAPS.value,
                    ],
                },
                warnings=triage.warnings,
            )
        except Exception as exc:
            return self._fail_workflow(session, triage_run, WorkflowState.BUILDING_UNBIASED_CANDIDATE_POOL, exc)

    def _build_portfolio_maps(self, session: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        if self.store.pending_human_action(session["session_id"]):
            return {
                "success": False,
                "error": "Cannot build portfolio maps while a human action is still pending.",
                "allowed_actions": ["answer_human_input", "select_option", "status", "cancel"],
            }

        triage_artifact = self.store.latest_artifact(session["session_id"], "candidate_triage")
        policy_artifact = self.store.latest_artifact(session["session_id"], "policy")
        lightweight_artifact = self.store.latest_artifact(session["session_id"], "futu_lightweight_enrichment")
        deep_research_artifact = self.store.latest_artifact(session["session_id"], "deep_research_report")
        if not triage_artifact or not policy_artifact:
            return {
                "success": False,
                "error": "Cannot build portfolio maps because candidate_triage or policy artifact is missing.",
                "allowed_actions": ["start", "discover", "status", "cancel"],
            }

        triage = CandidateTriageArtifact.model_validate(triage_artifact["payload"])
        policy = InvestmentPolicy.model_validate(policy_artifact["payload"])
        policy = _policy_with_payload_overrides(policy, payload)
        lightweight_payload = lightweight_artifact["payload"] if lightweight_artifact else {}
        deep_research_payload = deep_research_artifact["payload"] if deep_research_artifact else {}

        architect_run = self._begin_state_run(
            session,
            WorkflowState.DRAFTING_TARGET_PORTFOLIO_MAPS,
            {
                "candidate_triage_artifact_id": triage_artifact["artifact_id"],
                "policy_artifact_id": policy_artifact["artifact_id"],
                "futu_lightweight_enrichment_artifact_id": (
                    lightweight_artifact["artifact_id"] if lightweight_artifact else None
                ),
                "deep_research_report_artifact_id": (
                    deep_research_artifact["artifact_id"] if deep_research_artifact else None
                ),
                "mode": (
                    "portfolio_architect_from_deep_research"
                    if deep_research_artifact
                    else "portfolio_architect_from_candidate_triage"
                ),
                "target_portfolio_weight": policy.target_portfolio_weight,
                "cash_reserve": policy.cash_reserve,
                "single_name_limit": policy.single_name_limit,
                "required_symbols": policy.required_symbols,
            },
        )
        try:
            architect_result, run_metadata = build_portfolio_maps_from_triage(
                policy=policy,
                triage=triage,
                lightweight=lightweight_payload if isinstance(lightweight_payload, dict) else {},
                deep_research=deep_research_payload if isinstance(deep_research_payload, dict) else {},
            )
            result_artifact = self.store.add_artifact(
                session["session_id"],
                "portfolio_architect_result",
                architect_result,
            )
            run_artifact = self.store.add_artifact(
                session["session_id"],
                "portfolio_architect_run",
                run_metadata,
            )
            map_ids = [item.map_id for item in architect_result.portfolio_maps.maps]
            selected_symbols = [item.symbol for item in architect_result.selection.selected_for_portfolio]
            self._finish_state_run(
                architect_run,
                output_payload={
                    "portfolio_architect_result_artifact_id": result_artifact["artifact_id"],
                    "portfolio_architect_run_artifact_id": run_artifact["artifact_id"],
                    "map_count": len(map_ids),
                    "map_ids": map_ids,
                    "selected_symbol_count": len(selected_symbols),
                    "eligible_symbol_count": len(run_metadata.eligible_symbols),
                    "next_state": WorkflowState.NEEDS_PORTFOLIO_MAP_REVIEW.value,
                    "warnings": run_metadata.warnings,
                },
                artifact_ids=[result_artifact["artifact_id"], run_artifact["artifact_id"]],
            )
            human_action = self.store.create_human_action(
                session["session_id"],
                WorkflowState.NEEDS_PORTFOLIO_MAP_REVIEW.value,
                HumanActionKind.SELECT_PORTFOLIO_MAP.value,
                prompt=_portfolio_map_human_prompt(architect_result.model_dump(mode="json")),
                response_schema=_portfolio_map_response_schema(architect_result.model_dump(mode="json")),
            )
            updated = self.store.update_session(
                session["session_id"],
                state=WorkflowState.NEEDS_PORTFOLIO_MAP_REVIEW.value,
                status=WorkflowStatus.WAITING_FOR_HUMAN.value,
                current_artifact_id=result_artifact["artifact_id"],
            )
            warnings = _dedupe(
                [
                    *run_metadata.warnings,
                    *architect_result.warnings,
                    *architect_result.selection.warnings,
                    *architect_result.portfolio_maps.warnings,
                ]
            )
            return self._response(
                updated,
                data={
                    "message": (
                        "Portfolio-map architect completed from deep-research report and saved artifacts."
                        if deep_research_artifact
                        else "Portfolio-map architect completed from candidate triage and saved artifacts."
                    ),
                    "policy_artifact_id": policy_artifact["artifact_id"],
                    "futu_lightweight_enrichment_artifact_id": (
                        lightweight_artifact["artifact_id"] if lightweight_artifact else None
                    ),
                    "deep_research_report_artifact_id": (
                        deep_research_artifact["artifact_id"] if deep_research_artifact else None
                    ),
                    "candidate_triage_artifact_id": triage_artifact["artifact_id"],
                    "portfolio_architect_result_artifact_id": result_artifact["artifact_id"],
                    "portfolio_architect_run_artifact_id": run_artifact["artifact_id"],
                    "portfolio_architect_result": architect_result.model_dump(mode="json"),
                    "portfolio_architect_run": run_metadata.model_dump(mode="json"),
                    "portfolio_maps": architect_result.portfolio_maps.model_dump(mode="json"),
                    "map_ids": map_ids,
                    "selected_symbols": selected_symbols,
                    "portfolio_map_selection_action_id": human_action["action_id"],
                    "state_runs": self._state_run_summaries(session["session_id"]),
                },
                warnings=warnings,
            )
        except Exception as exc:
            return self._fail_workflow(session, architect_run, WorkflowState.DRAFTING_TARGET_PORTFOLIO_MAPS, exc)

    def _select_portfolio_map(
        self,
        session: dict[str, Any],
        payload: dict[str, Any],
        pending: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result_artifact = self.store.latest_artifact(session["session_id"], "portfolio_architect_result")
        if not result_artifact:
            return {
                "success": False,
                "error": "Cannot select a portfolio map because portfolio_architect_result is missing.",
                "allowed_actions": ["build_portfolio_maps", "continue", "status", "cancel"],
            }
        result_payload = result_artifact["payload"]
        selected_map = _select_portfolio_map_from_payload(result_payload, payload)
        if selected_map is None:
            map_ids = [
                item.get("map_id")
                for item in ((result_payload.get("portfolio_maps") or {}).get("maps") or [])
                if isinstance(item, dict) and item.get("map_id")
            ]
            return {
                "success": False,
                "error": "Cannot resolve selected portfolio map from payload.",
                "available_map_ids": map_ids,
                "allowed_actions": ["select_option", "status", "cancel"],
            }
        answer_payload = {
            "selected_map_id": selected_map.get("map_id"),
            "selected_map": selected_map,
            "raw_payload": payload,
            "portfolio_architect_result_artifact_id": result_artifact["artifact_id"],
        }
        if pending:
            self.store.answer_human_action(pending["action_id"], answer_payload)
        selection_artifact = self.store.add_artifact(
            session["session_id"],
            "selected_portfolio_map",
            answer_payload,
        )
        self.store.add_event(
            session["session_id"],
            "select_portfolio_map",
            session.get("state"),
            WorkflowState.TARGET_PORTFOLIO_MAP_SELECTED.value,
            {
                "selected_map_id": selected_map.get("map_id"),
                "selected_portfolio_map_artifact_id": selection_artifact["artifact_id"],
            },
        )
        updated = self.store.update_session(
            session["session_id"],
            state=WorkflowState.TARGET_PORTFOLIO_MAP_SELECTED.value,
            status=WorkflowStatus.COMPLETED.value,
            current_artifact_id=selection_artifact["artifact_id"],
        )
        return self._response(
            updated,
            data={
                "message": "Portfolio map selection recorded.",
                "portfolio_architect_result_artifact_id": result_artifact["artifact_id"],
                "selected_portfolio_map_artifact_id": selection_artifact["artifact_id"],
                "selected_map_id": selected_map.get("map_id"),
                "selected_portfolio_map": selected_map,
                "state_runs": self._state_run_summaries(session["session_id"]),
            },
        )

    def _revise_portfolio_map(self, session: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        if self.store.pending_human_action(session["session_id"]):
            return {
                "success": False,
                "error": "Cannot revise portfolio map while a human action is still pending.",
                "allowed_actions": ["answer_human_input", "select_option", "continue", "status", "cancel"],
            }
        request = _revision_request_from_payload(payload)
        if not request:
            return {
                "success": False,
                "error": "payload.request, payload.answer, payload.message, or payload.text is required.",
                "allowed_actions": ["revise_portfolio_map", "status", "cancel"],
            }
        result_artifact = self.store.latest_artifact(session["session_id"], "portfolio_architect_result")
        policy_artifact = self.store.latest_artifact(session["session_id"], "policy")
        if not result_artifact or not policy_artifact:
            return {
                "success": False,
                "error": "Cannot revise portfolio map because portfolio_architect_result or policy artifact is missing.",
                "allowed_actions": ["build_portfolio_maps", "continue", "status", "cancel"],
            }
        architect_result = PortfolioArchitectResult.model_validate(result_artifact["payload"])
        policy = _policy_with_payload_overrides(
            InvestmentPolicy.model_validate(policy_artifact["payload"]),
            payload,
        )
        base_map = self._base_map_for_revision(session, architect_result, payload)
        deep_research_artifact = self.store.latest_artifact(session["session_id"], "deep_research_report")
        deep_research_payload = deep_research_artifact["payload"] if deep_research_artifact else {}

        revision_run = self._begin_state_run(
            session,
            WorkflowState.REVISING_PORTFOLIO_MAP,
            {
                "portfolio_architect_result_artifact_id": result_artifact["artifact_id"],
                "policy_artifact_id": policy_artifact["artifact_id"],
                "deep_research_report_artifact_id": (
                    deep_research_artifact["artifact_id"] if deep_research_artifact else None
                ),
                "base_map_id": base_map.map_id,
                "request": request,
                "mode": "portfolio_revision",
            },
        )
        try:
            patch, revision, run_metadata = build_portfolio_revision_from_artifacts(
                user_request=request,
                base_map=base_map,
                architect_result=architect_result,
                policy=policy,
                deep_research=deep_research_payload if isinstance(deep_research_payload, dict) else {},
            )
            patch_artifact = self.store.add_artifact(session["session_id"], "portfolio_revision_patch", patch)
            run_artifact = self.store.add_artifact(session["session_id"], "portfolio_revision_run", run_metadata)
            if patch.needs_clarification or revision is None:
                self._finish_state_run(
                    revision_run,
                    output_payload={
                        "portfolio_revision_patch_artifact_id": patch_artifact["artifact_id"],
                        "portfolio_revision_run_artifact_id": run_artifact["artifact_id"],
                        "base_map_id": base_map.map_id,
                        "next_state": WorkflowState.NEEDS_PORTFOLIO_REVISION_CLARIFICATION.value,
                        "warnings": run_metadata.warnings,
                    },
                    artifact_ids=[patch_artifact["artifact_id"], run_artifact["artifact_id"]],
                )
                human_action = self.store.create_human_action(
                    session["session_id"],
                    WorkflowState.NEEDS_PORTFOLIO_REVISION_CLARIFICATION.value,
                    HumanActionKind.CLARIFY_PORTFOLIO_REVISION.value,
                    prompt=_portfolio_revision_clarification_prompt(patch.model_dump(mode="json")),
                    response_schema=_portfolio_revision_clarification_schema(),
                )
                updated = self.store.update_session(
                    session["session_id"],
                    state=WorkflowState.NEEDS_PORTFOLIO_REVISION_CLARIFICATION.value,
                    status=WorkflowStatus.WAITING_FOR_HUMAN.value,
                    current_artifact_id=patch_artifact["artifact_id"],
                )
                return self._response(
                    updated,
                    data={
                        "message": "Portfolio revision needs clarification before generating a revised map.",
                        "portfolio_architect_result_artifact_id": result_artifact["artifact_id"],
                        "portfolio_revision_patch_artifact_id": patch_artifact["artifact_id"],
                        "portfolio_revision_run_artifact_id": run_artifact["artifact_id"],
                        "portfolio_revision_patch": patch.model_dump(mode="json"),
                        "portfolio_revision_clarification_action_id": human_action["action_id"],
                        "state_runs": self._state_run_summaries(session["session_id"]),
                    },
                    warnings=run_metadata.warnings,
                )

            revision_artifact = self.store.add_artifact(session["session_id"], "portfolio_map_revision", revision)
            self._finish_state_run(
                revision_run,
                output_payload={
                    "portfolio_revision_patch_artifact_id": patch_artifact["artifact_id"],
                    "portfolio_map_revision_artifact_id": revision_artifact["artifact_id"],
                    "portfolio_revision_run_artifact_id": run_artifact["artifact_id"],
                    "base_map_id": base_map.map_id,
                    "revision_id": revision.revision_id,
                    "next_state": WorkflowState.NEEDS_PORTFOLIO_REVISION_REVIEW.value,
                    "warnings": run_metadata.warnings,
                },
                artifact_ids=[patch_artifact["artifact_id"], revision_artifact["artifact_id"], run_artifact["artifact_id"]],
            )
            human_action = self.store.create_human_action(
                session["session_id"],
                WorkflowState.NEEDS_PORTFOLIO_REVISION_REVIEW.value,
                HumanActionKind.CONFIRM_PORTFOLIO_REVISION.value,
                prompt=_portfolio_revision_review_prompt(revision.model_dump(mode="json")),
                response_schema=_portfolio_revision_review_schema(),
            )
            updated = self.store.update_session(
                session["session_id"],
                state=WorkflowState.NEEDS_PORTFOLIO_REVISION_REVIEW.value,
                status=WorkflowStatus.WAITING_FOR_HUMAN.value,
                current_artifact_id=revision_artifact["artifact_id"],
            )
            return self._response(
                updated,
                data={
                    "message": "Portfolio map revision generated. Waiting for user confirmation.",
                    "portfolio_architect_result_artifact_id": result_artifact["artifact_id"],
                    "portfolio_revision_patch_artifact_id": patch_artifact["artifact_id"],
                    "portfolio_map_revision_artifact_id": revision_artifact["artifact_id"],
                    "portfolio_revision_run_artifact_id": run_artifact["artifact_id"],
                    "portfolio_revision_patch": patch.model_dump(mode="json"),
                    "portfolio_map_revision": revision.model_dump(mode="json"),
                    "portfolio_revision_review_action_id": human_action["action_id"],
                    "state_runs": self._state_run_summaries(session["session_id"]),
                },
                warnings=run_metadata.warnings,
            )
        except Exception as exc:
            return self._fail_workflow(session, revision_run, WorkflowState.REVISING_PORTFOLIO_MAP, exc)

    def _answer_portfolio_revision_clarification(
        self,
        session: dict[str, Any],
        payload: dict[str, Any],
        pending: dict[str, Any],
    ) -> dict[str, Any]:
        answer = _revision_request_from_payload(payload)
        if not answer:
            return {
                "success": False,
                "error": "Clarification answer is required.",
                "allowed_actions": ["answer_human_input", "status", "cancel"],
            }
        patch_artifact = self.store.latest_artifact(session["session_id"], "portfolio_revision_patch")
        patch_payload = patch_artifact["payload"] if patch_artifact else {}
        original = str(patch_payload.get("user_request") or "").strip()
        combined = original
        if combined:
            combined += f"\nUser clarification: {answer}"
        else:
            combined = answer
        self.store.answer_human_action(
            pending["action_id"],
            {"answer": answer, "combined_request": combined, "raw_payload": payload},
        )
        self.store.add_event(
            session["session_id"],
            "clarify_portfolio_revision",
            session.get("state"),
            WorkflowState.REVISING_PORTFOLIO_MAP.value,
            {"action_id": pending["action_id"]},
        )
        updated = self.store.update_session(
            session["session_id"],
            state=WorkflowState.TARGET_PORTFOLIO_MAP_SELECTED.value,
            status=WorkflowStatus.COMPLETED.value,
        )
        return self._revise_portfolio_map(
            updated,
            {
                **payload,
                "request": combined,
                "base_map_id": patch_payload.get("base_map_id") or payload.get("base_map_id") or "",
            },
        )

    def _answer_portfolio_revision_confirmation(
        self,
        session: dict[str, Any],
        payload: dict[str, Any],
        pending: dict[str, Any],
    ) -> dict[str, Any]:
        decision = _portfolio_revision_confirmation_decision(payload)
        if decision == "revise_again":
            answer = _revision_request_from_payload(payload)
            self.store.answer_human_action(
                pending["action_id"],
                {"decision": decision, "answer": answer, "raw_payload": payload},
            )
            updated = self.store.update_session(
                session["session_id"],
                state=WorkflowState.TARGET_PORTFOLIO_MAP_SELECTED.value,
                status=WorkflowStatus.COMPLETED.value,
            )
            return self._revise_portfolio_map(updated, {**payload, "request": answer, "use_latest_revision": True})
        if decision != "accept":
            self.store.answer_human_action(
                pending["action_id"],
                {"decision": decision, "raw_payload": payload},
            )
            updated = self.store.update_session(
                session["session_id"],
                state=WorkflowState.TARGET_PORTFOLIO_MAP_SELECTED.value,
                status=WorkflowStatus.COMPLETED.value,
            )
            return self._response(
                updated,
                data={
                    "message": "Portfolio map revision was not confirmed. The prior selected map remains unchanged.",
                    "state_runs": self._state_run_summaries(session["session_id"]),
                },
            )

        revision_artifact = self.store.latest_artifact(session["session_id"], "portfolio_map_revision")
        if not revision_artifact:
            return {
                "success": False,
                "error": "Cannot confirm portfolio revision because portfolio_map_revision is missing.",
                "allowed_actions": ["revise_portfolio_map", "status", "cancel"],
            }
        revision = PortfolioMapRevision.model_validate(revision_artifact["payload"])
        answer_payload = {
            "revision_id": revision.revision_id,
            "selected_map_id": revision.revised_map.map_id,
            "selected_map": revision.revised_map.model_dump(mode="json"),
            "portfolio_map_revision": revision.model_dump(mode="json"),
            "portfolio_map_revision_artifact_id": revision_artifact["artifact_id"],
            "raw_payload": payload,
        }
        self.store.answer_human_action(pending["action_id"], answer_payload)
        selection_artifact = self.store.add_artifact(
            session["session_id"],
            "selected_portfolio_map_revision",
            answer_payload,
        )
        self.store.add_event(
            session["session_id"],
            "confirm_portfolio_revision",
            session.get("state"),
            WorkflowState.TARGET_PORTFOLIO_MAP_REVISION_SELECTED.value,
            {
                "revision_id": revision.revision_id,
                "selected_portfolio_map_revision_artifact_id": selection_artifact["artifact_id"],
            },
        )
        updated = self.store.update_session(
            session["session_id"],
            state=WorkflowState.TARGET_PORTFOLIO_MAP_REVISION_SELECTED.value,
            status=WorkflowStatus.COMPLETED.value,
            current_artifact_id=selection_artifact["artifact_id"],
        )
        return self._response(
            updated,
            data={
                "message": "Portfolio map revision confirmed and recorded.",
                "portfolio_map_revision_artifact_id": revision_artifact["artifact_id"],
                "selected_portfolio_map_revision_artifact_id": selection_artifact["artifact_id"],
                "selected_portfolio_map_revision": answer_payload,
                "state_runs": self._state_run_summaries(session["session_id"]),
            },
        )

    def _base_map_for_revision(
        self,
        session: dict[str, Any],
        architect_result: PortfolioArchitectResult,
        payload: dict[str, Any],
    ) -> PortfolioMap:
        if payload.get("use_latest_revision"):
            latest_revision = self.store.latest_artifact(session["session_id"], "portfolio_map_revision")
            if latest_revision:
                revision = PortfolioMapRevision.model_validate(latest_revision["payload"])
                return revision.revised_map
        map_id = str(payload.get("base_map_id") or payload.get("map_id") or payload.get("option_id") or "").strip()
        if map_id:
            for item in architect_result.portfolio_maps.maps:
                if item.map_id == map_id:
                    return item
            latest_revision = self.store.latest_artifact(session["session_id"], "portfolio_map_revision")
            if latest_revision:
                revision = PortfolioMapRevision.model_validate(latest_revision["payload"])
                if revision.revised_map.map_id == map_id:
                    return revision.revised_map
            raise ValueError(
                "Cannot revise portfolio map because base_map_id is unknown: "
                f"{map_id}. Available map ids: "
                + ", ".join(item.map_id for item in architect_result.portfolio_maps.maps)
            )
        selected_revision = self.store.latest_artifact(session["session_id"], "selected_portfolio_map_revision")
        if selected_revision:
            selected = selected_revision["payload"].get("selected_map") or {}
            if selected:
                return PortfolioMap.model_validate(selected)
        selected_artifact = self.store.latest_artifact(session["session_id"], "selected_portfolio_map")
        if selected_artifact:
            selected = selected_artifact["payload"].get("selected_map") or {}
            if selected:
                return PortfolioMap.model_validate(selected)
        if len(architect_result.portfolio_maps.maps) == 1:
            return architect_result.portfolio_maps.maps[0]
        raise ValueError("Cannot revise portfolio map without a selected map or payload.base_map_id.")

    def _require_session(self, tenant: str, session_id: str | None) -> dict[str, Any]:
        session = self.store.get_session(session_id) if session_id else self.store.latest_session(tenant)
        if not session and _is_cli_tenant(tenant):
            session = self.store.latest_session_by_tenant_prefix("cli:")
        if not session:
            raise ValueError("No investment assistant workflow session found")
        if not _tenant_matches(session["tenant_id"], tenant):
            raise ValueError("Workflow session does not belong to the current tenant")
        return session

    def _transition(self, session: dict[str, Any], state: WorkflowState) -> None:
        old = session.get("state")
        self.store.add_event(session["session_id"], "transition", old, state.value, {})
        self.store.update_session(
            session["session_id"],
            state=state.value,
            status=WorkflowStatus.ACTIVE.value,
        )
        session["state"] = state.value

    def _begin_state_run(
        self,
        session: dict[str, Any],
        state: WorkflowState,
        input_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._transition(session, state)
        return self.store.start_state_run(
            session["session_id"],
            state.value,
            input_payload or {},
        )

    def _record_completed_state_run(
        self,
        session_id: str,
        state: WorkflowState,
        *,
        input_payload: dict[str, Any] | None = None,
        output_payload: dict[str, Any] | None = None,
        artifact_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        run = self.store.start_state_run(session_id, state.value, input_payload or {})
        return self.store.finish_state_run(
            run["run_id"],
            status="completed",
            output_payload=output_payload or {},
            artifact_ids=artifact_ids or [],
        )

    def _finish_state_run(
        self,
        run: dict[str, Any],
        *,
        output_payload: dict[str, Any] | None = None,
        artifact_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        return self.store.finish_state_run(
            run["run_id"],
            status="completed",
            output_payload=output_payload or {},
            artifact_ids=artifact_ids or [],
        )

    def _fail_state_run(self, run: dict[str, Any], exc: Exception) -> dict[str, Any]:
        return self.store.finish_state_run(
            run["run_id"],
            status="failed",
            output_payload={},
            artifact_ids=run.get("artifact_ids") or [],
            error_payload=_state_error_payload(exc),
        )

    def _fail_workflow(
        self,
        session: dict[str, Any],
        run: dict[str, Any],
        failed_state: WorkflowState,
        exc: Exception,
    ) -> dict[str, Any]:
        self._fail_state_run(run, exc)
        error_payload = {
            "message": str(exc),
            "failed_state": failed_state.value,
            "recoverable": True,
        }
        self.store.add_artifact(session["session_id"], "workflow_error", error_payload)
        self.store.add_event(
            session["session_id"],
            "error",
            failed_state.value,
            failed_state.value,
            error_payload,
        )
        failed_session = self.store.update_session(
            session["session_id"],
            state=failed_state.value,
            status=WorkflowStatus.FAILED.value,
        )
        return self._response(
            failed_session,
            data={
                "artifacts": [
                    {
                        "artifact_id": item["artifact_id"],
                        "type": item["type"],
                        "version": item["version"],
                        "created_at": item["created_at"],
                    }
                    for item in self.store.list_artifacts(session["session_id"])
                ],
                "latest_error": error_payload,
                "state_runs": self._state_run_summaries(session["session_id"]),
            },
        )

    def _state_run_summaries(self, session_id: str) -> list[dict[str, Any]]:
        return [_state_run_summary(item) for item in self.store.list_state_runs(session_id)]

    def _status(self, session: dict[str, Any]) -> dict[str, Any]:
        latest_error = self.store.latest_artifact(session["session_id"], "workflow_error")
        artifacts = [
            {
                "artifact_id": item["artifact_id"],
                "type": item["type"],
                "version": item["version"],
                "created_at": item["created_at"],
            }
            for item in self.store.list_artifacts(session["session_id"])
        ]
        return self._response(
            session,
            data={
                "artifacts": artifacts,
                "latest_error": latest_error["payload"] if latest_error else None,
                "state_runs": self._state_run_summaries(session["session_id"]),
            },
        )

    def _response(
        self,
        session: dict[str, Any],
        data: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
    ) -> dict[str, Any]:
        data = data or {}
        warnings = warnings or []
        human_action = self.store.pending_human_action(session["session_id"])
        allowed_actions = ["status", "cancel"]
        if human_action:
            allowed_actions = ["answer_human_input", "select_option", "continue", "status", "cancel"]
        elif session.get("state") in {
            WorkflowState.CANDIDATE_TRIAGE_COMPLETE.value,
            WorkflowState.DRAFTING_TARGET_PORTFOLIO_MAPS.value,
        } and session.get("status") in {WorkflowStatus.COMPLETED.value, WorkflowStatus.FAILED.value}:
            allowed_actions = ["build_portfolio_maps", "continue", "status", "cancel"]
        elif session.get("state") in {
            WorkflowState.NEEDS_PORTFOLIO_MAP_REVIEW.value,
            WorkflowState.NEEDS_PORTFOLIO_MAP_SELECTION.value,
        }:
            allowed_actions = ["select_option", "revise_portfolio_map", "status", "cancel"]
        elif session.get("state") in {
            WorkflowState.TARGET_PORTFOLIO_MAP_SELECTED.value,
            WorkflowState.TARGET_PORTFOLIO_MAP_REVISION_SELECTED.value,
        }:
            allowed_actions = ["revise_portfolio_map", "build_portfolio_maps", "status", "cancel"]
        response = {
            "success": True,
            "session_id": session["session_id"],
            "tenant_id": session["tenant_id"],
            "state": session["state"],
            "status": session["status"],
            "theme": session["theme"],
            "data": data,
            "human_action": _public_human_action(human_action),
            "allowed_actions": allowed_actions,
            "warnings": warnings,
            "next_instruction_for_agent": _next_instruction(human_action, data),
        }
        response["answer_contract"] = _answer_contract(response)
        response["display_response"] = _display_response(response)
        response["fallback_response"] = response["display_response"]
        response["agent_brief"] = _agent_brief(response)
        response["next_instruction_for_agent"] = (
            f"{response['next_instruction_for_agent']} Use agent_brief to write a natural "
            "Chinese user-facing reply. You may rephrase for clarity, but do not add "
            "symbols, weights, orders, option strategies, or risk claims that are not "
            "present in this workflow response."
        )
        return response


def _market_from_payload(payload: dict[str, Any]) -> str:
    return str(payload.get("market") or payload.get("quote_market") or "US").strip().upper() or "US"


def _normalize_symbols(raw: Any, market: str = "US") -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        values = re.split(r"[,，/、\s()（）]+", raw)
    elif isinstance(raw, list | tuple | set):
        values = []
        for item in raw:
            values.extend(_normalize_symbols(item, market))
    else:
        values = [str(raw)]

    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        symbol = normalize_market_symbol(value, market)
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        result.append(symbol)
    return result


def _state_error_payload(exc: Exception) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
    }


def _choice(value: Any, allowed: set[str], default: str) -> str:
    text = str(value or "").strip().lower()
    return text if text in allowed else default


def _float_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_cli_tenant(tenant: str) -> bool:
    return str(tenant or "").startswith("cli:")


def _tenant_matches(session_tenant: str, current_tenant: str) -> bool:
    if session_tenant == current_tenant:
        return True
    return _is_cli_tenant(session_tenant) and _is_cli_tenant(current_tenant)


def _policy_with_payload_overrides(policy: InvestmentPolicy, payload: dict[str, Any]) -> InvestmentPolicy:
    if not payload:
        return policy
    data = policy.model_dump(mode="json")
    market = str(payload.get("market") or "US")
    if payload.get("objective") is not None:
        data["objective"] = _choice(payload.get("objective"), {"balanced", "growth", "income"}, data["objective"])
    if payload.get("risk_level") is not None:
        data["risk_level"] = _choice(
            payload.get("risk_level"),
            {"conservative", "moderate", "aggressive"},
            data["risk_level"],
        )
    for key in ("target_portfolio_weight", "cash_reserve", "single_name_limit"):
        if payload.get(key) is not None:
            data[key] = _float_default(payload.get(key), data[key])
    if payload.get("required_symbols") is not None:
        data["required_symbols"] = _normalize_symbols(payload.get("required_symbols"), market)
    if payload.get("allow_options") is not None:
        data["allow_options"] = bool(payload.get("allow_options"))
    if payload.get("notes") is not None:
        data["notes"] = str(payload.get("notes") or "")
    return InvestmentPolicy.model_validate(data)


def _state_run_summary(run: dict[str, Any]) -> dict[str, Any]:
    output = run.get("output") if isinstance(run.get("output"), dict) else {}
    return {
        "run_id": run.get("run_id"),
        "state": run.get("state"),
        "status": run.get("status"),
        "started_at": run.get("started_at"),
        "ended_at": run.get("ended_at"),
        "duration_ms": run.get("duration_ms"),
        "artifact_ids": run.get("artifact_ids") or [],
        "output": _compact_state_output(output),
        "error": run.get("error"),
    }


def _public_human_action(human_action: dict[str, Any] | None) -> dict[str, Any] | None:
    if not human_action:
        return None
    return {
        "action_id": human_action.get("action_id"),
        "kind": human_action.get("kind"),
        "state": human_action.get("state"),
        "status": human_action.get("status"),
        "prompt": human_action.get("prompt"),
        "response_schema": human_action.get("response_schema"),
        "created_at": human_action.get("created_at"),
    }


def _triage_plan_human_prompt(plan: dict[str, Any]) -> dict[str, Any]:
    options = []
    for option in plan.get("strategy_options") or []:
        if not isinstance(option, dict):
            continue
        options.append(
            {
                "option_id": option.get("option_id"),
                "name": option.get("name"),
                "deep_research_total": option.get("deep_research_total"),
                "expected_watchlist_count": option.get("expected_watchlist_count"),
                "best_for": option.get("best_for"),
                "tradeoffs": option.get("tradeoffs") or [],
            }
        )
    return {
        "question": plan.get("prompt_to_user") or "请选择一个候选粗筛策略，或提出修改意见。",
        "recommended_option_id": plan.get("recommended_option_id"),
        "options": options,
        "freeform_allowed": True,
        "examples": [
            "选 1",
            "选 bottleneck_momentum，但 SNDK/COHR/LITE/WDC/MRVL 必须 deep research",
            "用平衡覆盖，但把 deep research 控制在 25 个以内",
        ],
    }


def _triage_strategy_response_schema(plan: dict[str, Any]) -> dict[str, Any]:
    option_ids = [
        str(option.get("option_id"))
        for option in plan.get("strategy_options") or []
        if isinstance(option, dict) and option.get("option_id")
    ]
    return {
        "type": "object",
        "properties": {
            "option_id": {
                "type": "string",
                "enum": option_ids,
                "description": "Preferred strategy option id from candidate_triage_plan.",
            },
            "answer": {
                "type": "string",
                "description": "Natural-language user instruction such as '选 2' or modifications.",
            },
            "modifications": {
                "type": "string",
                "description": "Optional free-form changes to budgets, inclusions, exclusions, or priorities.",
            },
            "must_include_symbols": {"type": "array", "items": {"type": "string"}},
            "exclude_symbols": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": True,
    }


def _portfolio_map_human_prompt(result: dict[str, Any]) -> dict[str, Any]:
    maps = []
    for portfolio_map in (result.get("portfolio_maps") or {}).get("maps") or []:
        if not isinstance(portfolio_map, dict):
            continue
        maps.append(
            {
                "map_id": portfolio_map.get("map_id"),
                "name": portfolio_map.get("name"),
                "objective": portfolio_map.get("objective"),
                "sleeve_weight": portfolio_map.get("sleeve_weight"),
                "cash_weight": portfolio_map.get("cash_weight"),
                "best_for": portfolio_map.get("best_for"),
            }
        )
    return {
        "question": "请选择一个目标组合版图作为后续建仓计划的基准，或提出修改意见。",
        "options": maps,
        "freeform_allowed": True,
        "examples": [
            "选 1",
            "选 2：AI_MAP_2_SEMI_COMPUTE_BOTTLENECK",
            "选 AI_MAP_2_SEMI_COMPUTE_BOTTLENECK",
        ],
    }


def _portfolio_map_response_schema(result: dict[str, Any]) -> dict[str, Any]:
    map_ids = [
        str(item.get("map_id"))
        for item in (result.get("portfolio_maps") or {}).get("maps") or []
        if isinstance(item, dict) and item.get("map_id")
    ]
    return {
        "type": "object",
        "properties": {
            "map_id": {
                "type": "string",
                "enum": map_ids,
                "description": "Preferred portfolio map id from portfolio_architect_result.",
            },
            "option_id": {
                "type": "string",
                "enum": map_ids,
                "description": "Alias of map_id.",
            },
            "answer": {
                "type": "string",
                "description": "Natural-language user instruction such as '选 2' or a map id.",
            },
        },
        "additionalProperties": True,
    }


def _portfolio_revision_clarification_prompt(patch: dict[str, Any]) -> dict[str, Any]:
    options = [
        str(value)
        for value in patch.get("clarification_options") or []
        if str(value).strip()
    ]
    return {
        "question": patch.get("clarification_question") or "请补充你希望如何调整这张目标版图。",
        "revision_intent": patch.get("revision_intent"),
        "base_map_id": patch.get("base_map_id"),
        "options": options,
        "freeform_allowed": True,
        "examples": [
            "让 AI 在风险预算内决定具体比例",
            "MU 和 SNDK 各提高 2%，从软件层和低确信标的里挪",
            "先不要改现金，保持 5% 现金",
        ],
    }


def _portfolio_revision_clarification_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "Natural-language clarification for the pending portfolio-map revision.",
            },
            "request": {"type": "string"},
            "message": {"type": "string"},
        },
        "additionalProperties": True,
    }


def _portfolio_revision_review_prompt(revision: dict[str, Any]) -> dict[str, Any]:
    revised_map = revision.get("revised_map") or {}
    return {
        "question": "请确认是否采用这版目标组合修订，或继续提出修改意见。",
        "revision_id": revision.get("revision_id"),
        "base_map_id": revision.get("base_map_id"),
        "map_id": revised_map.get("map_id"),
        "name": revised_map.get("name"),
        "change_summary": revision.get("change_summary") or [],
        "freeform_allowed": True,
        "examples": [
            "确认",
            "采用这版",
            "再把 LITE 加一点",
            "不要采用，保持原版",
        ],
    }


def _portfolio_revision_review_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "decision": {
                "type": "string",
                "enum": ["accept", "reject", "revise_again"],
                "description": "accept confirms the revised map; revise_again treats the answer as another revision request.",
            },
            "answer": {"type": "string"},
            "request": {"type": "string"},
            "message": {"type": "string"},
        },
        "additionalProperties": True,
    }


def _normalize_triage_strategy_answer(payload: dict[str, Any], plan: CandidateTriagePlanArtifact):
    raw_answer = payload.get("answer")
    if raw_answer is None and payload.get("selection") is not None:
        raw_answer = payload.get("selection")
    if raw_answer is None:
        raw_answer = payload.get("text")
    if raw_answer is None:
        raw_answer = payload.get("message")
    return select_triage_strategy(
        plan,
        option_id=str(
            payload.get("option_id")
            or payload.get("selected_option_id")
            or payload.get("strategy_option_id")
            or ""
        ).strip(),
        answer=str(raw_answer or "").strip(),
        modifications=str(payload.get("modifications") or "").strip(),
        must_include_symbols=payload.get("must_include_symbols") or [],
        exclude_symbols=payload.get("exclude_symbols") or [],
    )


def _revision_request_from_payload(payload: dict[str, Any]) -> str:
    return str(
        payload.get("request")
        or payload.get("answer")
        or payload.get("message")
        or payload.get("text")
        or payload.get("selection")
        or payload.get("modifications")
        or ""
    ).strip()


def _portfolio_revision_confirmation_decision(payload: dict[str, Any]) -> str:
    explicit = str(payload.get("decision") or "").strip().lower()
    if explicit in {"accept", "reject", "revise_again"}:
        return explicit
    answer = _revision_request_from_payload(payload).lower()
    if not answer:
        return "reject"
    accept_terms = {"确认", "采用", "接受", "同意", "ok", "yes", "accept", "confirm", "就这版", "可以"}
    reject_terms = {"拒绝", "不要", "不采用", "取消", "reject", "cancel", "保持原版"}
    if any(term in answer for term in accept_terms):
        return "accept"
    if any(term in answer for term in reject_terms):
        return "reject"
    return "revise_again"


def _select_portfolio_map_from_payload(result: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any] | None:
    maps = [
        item
        for item in (result.get("portfolio_maps") or {}).get("maps") or []
        if isinstance(item, dict) and item.get("map_id")
    ]
    if not maps:
        return None
    raw = str(
        payload.get("map_id")
        or payload.get("option_id")
        or payload.get("selected_map_id")
        or payload.get("selected_option_id")
        or payload.get("answer")
        or payload.get("selection")
        or payload.get("text")
        or payload.get("message")
        or ""
    ).strip()
    if not raw:
        return None
    raw_upper = raw.upper()
    for item in maps:
        map_id = str(item.get("map_id") or "")
        if map_id and map_id.upper() == raw_upper:
            return item
    for item in maps:
        map_id = str(item.get("map_id") or "")
        if map_id and map_id.upper() in raw_upper:
            return item
    match = re.search(r"(?:选|第|option|map)?\s*([0-9]+)", raw, flags=re.IGNORECASE)
    if match:
        index = int(match.group(1)) - 1
        if 0 <= index < len(maps):
            return maps[index]
    return None


def _symbols_from_discovery_and_lightweight(
    discovery: ThemeDiscoveryPlan,
    lightweight: LightweightEnrichmentArtifact,
) -> list[str]:
    symbols: list[str] = []
    for seed in discovery.seed_symbols:
        if seed.symbol and seed.symbol not in symbols:
            symbols.append(seed.symbol)
    for item in lightweight.candidates:
        if item.symbol and item.symbol not in symbols:
            symbols.append(item.symbol)
    return symbols


def _compact_state_output(output: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in output.items():
        if key in {
            "warnings",
            "defaults",
            "market_artifact_ids",
            "map_ids",
            "required_symbols",
            "macro_context_keys",
        }:
            compact[key] = value
            continue
        if key.endswith("_artifact_id") or key.endswith("_count"):
            compact[key] = value
            continue
        if isinstance(value, str | int | float | bool) or value is None:
            compact[key] = value
    return compact


def _next_instruction(human_action: dict[str, Any] | None, data: dict[str, Any] | None = None) -> str:
    data = data or {}
    if data.get("selected_portfolio_map_revision"):
        return (
            "Explain that the user's revised portfolio map has been confirmed and recorded. "
            "Do not generate construction plans, orders, holdings analysis, or option strategies."
        )
    if data.get("portfolio_map_revision"):
        return (
            "Explain the portfolio-map revision from the workflow artifact and ask the user "
            "to confirm it or provide further modifications. Do not add new symbols, weights, "
            "orders, or risk claims outside the revision artifact."
        )
    if data.get("portfolio_revision_patch"):
        return (
            "Explain the revision intent artifact and ask the clarification question if present. "
            "Do not revise the map yourself; the revision must come from workflow artifacts."
        )
    if data.get("selected_portfolio_map"):
        return (
            "Explain that the user's portfolio map selection has been recorded. "
            "Do not generate construction plans, orders, holdings analysis, or option strategies."
        )
    if data.get("portfolio_architect_result"):
        return (
            "Explain the portfolio architect result from the workflow artifact. "
            "Make clear that this is an AI-authored target portfolio-map draft "
            "based only on saved research artifacts, not current holdings, orders, "
            "execution instructions, or option strategies."
        )
    if data.get("candidate_triage_plan"):
        return (
            "Explain the candidate triage strategy plan and ask the user to choose "
            "one strategy option or provide modifications. Make clear that final "
            "triage, SEC/fundamental deep research, portfolio maps, holdings, "
            "orders, and options were not generated yet."
        )
    if data.get("candidate_triage"):
        return (
            "Explain the completed candidate triage artifact. Make clear that this "
            "is a research queue for later deep research, not a target portfolio, "
            "trade plan, holdings review, or order recommendation."
        )
    if data.get("theme_discovery"):
        return (
            "Explain the theme discovery artifact. Make clear that downstream "
            "candidate enrichment, SEC validation, thesis synthesis, portfolio "
            "maps, holdings, orders, and options were not generated."
        )
    return "Explain the workflow status. Do not claim that holdings, orders, or portfolio maps were generated."


def _answer_contract(response: dict[str, Any]) -> dict[str, Any]:
    data = response.get("data") or {}
    allowed_symbols = sorted(_collect_symbols(response))
    allowed_map_ids = sorted(_collect_map_ids(response))
    return {
        "mode": "artifact_only",
        "agent_may_rephrase": True,
        "fallback_response_on_validation_failure": True,
        "source_artifact_ids": _source_artifact_ids(data),
        "allowed_symbols": allowed_symbols,
        "allowed_map_ids": allowed_map_ids,
        "allowed_claims": [
            "Explain only the workflow response fields, source artifact ids, warnings, and current workflow artifact.",
            "For discovery-only results, explain only the theme_discovery artifact and the fact that downstream planning stopped.",
            "For candidate_triage_plan results, explain only the strategy options and ask the user to choose or modify one.",
            "For candidate_triage results, explain only the research queues and the fact that deep research/portfolio maps have not run.",
            "For portfolio_architect_result results, explain only the selection, map weights, rationales, warnings, and map ids present in the artifact.",
            "For portfolio_map_revision results, explain only the revised map, weight changes, funding sources, tradeoffs, and confirmation request present in the artifact.",
        ],
        "forbidden_claims": [
            "Do not introduce extra symbols, sectors, weights, option legs, or orders.",
            "Do not claim current holdings were read.",
            "Do not claim execution plans, tax advice, real orders, or option strategies were generated.",
            "Do not override stale-data warnings or missing-data warnings.",
        ],
    }


def _agent_brief(response: dict[str, Any]) -> str:
    contract = response.get("answer_contract") or {}
    lines = [
        "请把下面事实转成自然、易懂的中文回复；只能解释这些事实，不能新增标的、权重、订单或期权腿。",
        "",
        response.get("display_response") or _display_response(response),
        "",
        "表达边界：",
        "- 可以解释 discovery 的领域分层、覆盖检查项和种子标的。",
        "- 可以解释 candidate triage plan 的几个筛选策略，以及为什么需要用户先选一个。",
        "- 可以解释 candidate triage 的 deep research queue / watchlist / deferred / rejected。",
        "- 如果已有 portfolio_architect_result，可以解释 AI architect 选择了哪些标的、生成了哪些版图、每个版图的权重和取舍理由。",
        "- 如果已有 portfolio_map_revision，可以解释修订了哪些权重、资金从哪里挪出、风险变化是什么，并请用户确认。",
        "- 不要输出内部字段名。",
        "- 不要引入不在 allowed_symbols 里的股票代码。",
        "- 没有交易计划 artifact 时，不要给买卖建议或下单参数。",
    ]
    allowed_symbols = contract.get("allowed_symbols") or []
    allowed_map_ids = contract.get("allowed_map_ids") or []
    if allowed_symbols:
        lines.append(f"- allowed_symbols: {', '.join(allowed_symbols)}")
    if allowed_map_ids:
        lines.append(f"- allowed_map_ids: {', '.join(allowed_map_ids)}")
    return "\n".join(lines)


def _display_response(response: dict[str, Any]) -> str:
    data = response.get("data") or {}
    if data.get("selected_portfolio_map_revision"):
        return _display_selected_portfolio_map_revision(response)
    if data.get("portfolio_map_revision"):
        return _display_portfolio_map_revision(response)
    if data.get("portfolio_revision_patch"):
        return _display_portfolio_revision_patch(response)
    if data.get("selected_portfolio_map"):
        return _display_selected_portfolio_map(response)
    if data.get("portfolio_architect_result"):
        return _display_portfolio_architect_result(response)
    if data.get("candidate_triage_plan"):
        return _display_candidate_triage_plan(response)
    if data.get("candidate_triage"):
        return _display_candidate_triage(response)
    if data.get("theme_discovery"):
        return _display_theme_discovery(response)
    if data.get("artifacts") is not None:
        return _display_status(response)
    if data.get("cancelled"):
        return f"投资组合版图 workflow 已取消。session_id: {response['session_id']}"
    return _display_generic(response)


def _display_portfolio_revision_patch(response: dict[str, Any]) -> str:
    data = response.get("data") or {}
    patch = data.get("portfolio_revision_patch") or {}
    lines = [
        "我已把你的修改意见解析成目标版图修订意图，但还需要你补充一点信息。",
    ]
    intent = str(patch.get("revision_intent") or "").strip()
    if intent:
        lines.append(f"- 修订意图：{intent}")
    edits = patch.get("edits") or []
    if edits:
        lines.append("- 已识别的修改点：")
        for edit in edits[:8]:
            if not isinstance(edit, dict):
                continue
            target = edit.get("target") or {}
            target_label = target.get("label") or target.get("id") or target.get("kind") or ""
            lines.append(
                f"  - {edit.get('edit_type')} {edit.get('direction')} {target_label}"
            )
    question = str(patch.get("clarification_question") or "").strip()
    if question:
        lines.extend(["", f"需要你确认：{question}"])
    options = patch.get("clarification_options") or []
    for index, option in enumerate(options[:5], start=1):
        lines.append(f"{index}. {option}")
    lines.extend(_warning_lines(response.get("warnings") or patch.get("warnings") or []))
    lines.extend(
        [
            "",
            "你可以直接补充一句话；我会继续用同一张 base map 生成修订版。",
            f"portfolio_revision_patch_artifact_id: {data.get('portfolio_revision_patch_artifact_id')}",
            f"内部编号：{response['session_id']}",
        ]
    )
    return "\n".join(lines)


def _display_portfolio_map_revision(response: dict[str, Any]) -> str:
    data = response.get("data") or {}
    revision = data.get("portfolio_map_revision") or {}
    revised_map = revision.get("revised_map") or {}
    lines = [
        "已生成目标组合版图修订稿，等待你确认。",
        "这仍然只是目标版图：没有读取当前持仓，也没有生成买卖单、建仓计划或期权策略。",
        f"- revision_id: {revision.get('revision_id')}",
        f"- base_map_id: {revision.get('base_map_id')}",
    ]
    name = str(revised_map.get("name") or "").strip()
    if name:
        lines.append(f"- 修订后名称：{name}")
    sleeve_weight = _pct(revised_map.get("sleeve_weight"))
    cash_weight = _pct(revised_map.get("cash_weight"))
    if sleeve_weight or cash_weight:
        lines.append(f"- 风险资产：{sleeve_weight or 'N/A'}，现金：{cash_weight or 'N/A'}")
    changes = revision.get("change_summary") or []
    if changes:
        lines.extend(["", "改动摘要：", *[f"- {item}" for item in changes[:8]]])
    holdings = revised_map.get("holdings") or []
    if holdings:
        holding_text = ", ".join(
            f"{holding.get('symbol')} {_pct(holding.get('target_weight'))}"
            for holding in holdings
            if isinstance(holding, dict) and holding.get("symbol")
        )
        if holding_text:
            lines.extend(["", f"修订后配置：{holding_text}"])
    weight_changes = revision.get("weight_changes") or []
    if weight_changes:
        lines.extend(["", "权重变化："])
        for item in weight_changes[:12]:
            if not isinstance(item, dict):
                continue
            old = _pct(item.get("old_weight")) or "0.0%"
            new = _pct(item.get("new_weight")) or "0.0%"
            reason = str(item.get("reason") or "").strip()
            line = f"- {item.get('symbol')}: {old} -> {new} ({item.get('direction')})"
            if reason:
                line += f"。{reason}"
            lines.append(line)
    funding = revision.get("funding_sources") or []
    if funding:
        lines.extend(["", "权重资金来源：", *[f"- {item}" for item in funding[:8]]])
    tradeoffs = revision.get("tradeoff_explanation") or []
    if tradeoffs:
        lines.extend(["", "主要取舍：", *[f"- {item}" for item in tradeoffs[:8]]])
    risk_delta = revision.get("risk_delta") or []
    if risk_delta:
        lines.extend(["", "风险变化：", *[f"- {item}" for item in risk_delta[:8]]])
    lines.extend(_warning_lines(response.get("warnings") or revision.get("warnings") or []))
    lines.extend(
        [
            "",
            "下一步：回复“确认”采用这版修订，或继续说你想怎么改。",
            f"portfolio_map_revision_artifact_id: {data.get('portfolio_map_revision_artifact_id')}",
            f"portfolio_revision_patch_artifact_id: {data.get('portfolio_revision_patch_artifact_id')}",
            f"内部编号：{response['session_id']}",
        ]
    )
    return "\n".join(lines)


def _display_selected_portfolio_map_revision(response: dict[str, Any]) -> str:
    data = response.get("data") or {}
    selected = data.get("selected_portfolio_map_revision") or {}
    selected_map = selected.get("selected_map") or {}
    lines = [
        "已确认并记录你的目标组合版图修订版。",
        f"- revision_id: {selected.get('revision_id')}",
        f"- map_id: {selected_map.get('map_id') or selected.get('selected_map_id')}",
    ]
    name = str(selected_map.get("name") or "").strip()
    if name:
        lines.append(f"- 名称：{name}")
    holdings = selected_map.get("holdings") or []
    if holdings:
        holding_text = ", ".join(
            f"{holding.get('symbol')} {_pct(holding.get('target_weight'))}"
            for holding in holdings
            if isinstance(holding, dict) and holding.get("symbol")
        )
        if holding_text:
            lines.append(f"- 配置：{holding_text}")
    lines.extend(
        [
            "",
            "当前仍未读取真实持仓，也没有生成建仓计划或订单。",
            f"selected_portfolio_map_revision_artifact_id: {data.get('selected_portfolio_map_revision_artifact_id')}",
            f"portfolio_map_revision_artifact_id: {data.get('portfolio_map_revision_artifact_id')}",
            f"内部编号：{response['session_id']}",
        ]
    )
    return "\n".join(lines)


def _display_selected_portfolio_map(response: dict[str, Any]) -> str:
    data = response.get("data") or {}
    selected = data.get("selected_portfolio_map") or {}
    lines = [
        "已记录你的目标组合版图选择。",
        f"- map_id: {selected.get('map_id') or data.get('selected_map_id')}",
    ]
    name = str(selected.get("name") or "").strip()
    if name:
        lines.append(f"- 名称：{name}")
    objective = str(selected.get("objective") or "").strip()
    if objective:
        lines.append(f"- 目标：{objective}")
    sleeve_weight = _pct(selected.get("sleeve_weight"))
    cash_weight = _pct(selected.get("cash_weight"))
    if sleeve_weight:
        lines.append(f"- 目标风险资产：{sleeve_weight}")
    if cash_weight:
        lines.append(f"- 现金：{cash_weight}")
    holdings = selected.get("holdings") or []
    if holdings:
        holding_text = ", ".join(
            f"{holding.get('symbol')} {_pct(holding.get('target_weight'))}"
            for holding in holdings
            if isinstance(holding, dict) and holding.get("symbol")
        )
        if holding_text:
            lines.append(f"- 配置：{holding_text}")
    lines.extend(
        [
            "",
            "下一步可以基于这张目标版图生成建仓计划；当前还没有读取你的真实持仓，也没有生成订单。",
            f"selected_portfolio_map_artifact_id: {data.get('selected_portfolio_map_artifact_id')}",
            f"portfolio_architect_result_artifact_id: {data.get('portfolio_architect_result_artifact_id')}",
            f"内部编号：{response['session_id']}",
        ]
    )
    return "\n".join(lines)


def _display_portfolio_architect_result(response: dict[str, Any]) -> str:
    data = response.get("data") or {}
    result = data.get("portfolio_architect_result") or {}
    selection = result.get("selection") or {}
    portfolio_maps = result.get("portfolio_maps") or data.get("portfolio_maps") or {}
    maps = portfolio_maps.get("maps") or []
    run = data.get("portfolio_architect_run") or {}
    lines = [
        f"已生成“{_theme_label(str(response.get('theme') or ''))}”的目标组合版图草案。",
        "这是基于已保存 candidate triage、Futu 轻量数据和离线材料生成的 AI architect 结果；没有读取当前持仓，也没有生成买卖单、建仓计划或期权策略。",
    ]

    summary = str(selection.get("selection_summary") or "").strip()
    if summary:
        lines.extend(["", f"筛选后判断：{summary}"])

    selected = selection.get("selected_for_portfolio") or []
    watchlist = selection.get("watchlist_after_enrichment") or []
    deferred = selection.get("deferred_after_enrichment") or []
    rejected = selection.get("rejected_after_enrichment") or []
    if selected or watchlist or deferred or rejected:
        lines.extend(
            [
                "",
                "进入组合前的取舍：",
                f"- 入选组合候选：{len(selected)} 个",
                f"- watchlist：{len(watchlist)} 个，deferred：{len(deferred)} 个，rejected：{len(rejected)} 个",
            ]
        )

    tradeoffs = selection.get("peer_tradeoffs") or []
    if tradeoffs:
        lines.extend(["", "关键同层取舍："])
        for item in tradeoffs[:6]:
            if not isinstance(item, dict):
                continue
            selected_symbols = ", ".join(str(value) for value in item.get("selected_symbols") or [] if value)
            non_selected = ", ".join(str(value) for value in item.get("non_selected_symbols") or [] if value)
            rationale = str(item.get("rationale") or "").strip()
            line = f"- {item.get('layer_key') or 'peer group'}"
            if selected_symbols:
                line += f"：选 {selected_symbols}"
            if non_selected:
                line += f"，暂不选 {non_selected}"
            if rationale:
                line += f"。{rationale}"
            lines.append(line)

    if maps:
        lines.extend(["", f"组合版图候选（{len(maps)} 个）："])
        for index, item in enumerate(maps, start=1):
            if not isinstance(item, dict):
                continue
            map_id = str(item.get("map_id") or f"map_{index}")
            name = str(item.get("name") or map_id)
            objective = str(item.get("objective") or "")
            sleeve_weight = _pct(item.get("sleeve_weight"))
            cash_weight = _pct(item.get("cash_weight"))
            lines.append(f"{index}. {name} ({map_id})")
            details = []
            if objective:
                details.append(f"目标：{objective}")
            if sleeve_weight:
                details.append(f"风险资产：{sleeve_weight}")
            if cash_weight:
                details.append(f"现金：{cash_weight}")
            if details:
                lines.append(f"   - {'，'.join(details)}")
            positioning = str(item.get("positioning") or "").strip()
            if positioning:
                lines.append(f"   - 定位：{positioning}")
            holdings = item.get("holdings") or []
            if holdings:
                holding_text = ", ".join(
                    f"{holding.get('symbol')} {_pct(holding.get('target_weight'))}"
                    for holding in holdings
                    if isinstance(holding, dict) and holding.get("symbol")
                )
                if holding_text:
                    lines.append(f"   - 配置：{holding_text}")
            thesis = str(item.get("thesis") or "").strip()
            if thesis:
                lines.append(f"   - 主线：{thesis}")

    rationale_by_map = {
        str(item.get("map_id")): item
        for item in result.get("map_weight_rationales") or []
        if isinstance(item, dict) and item.get("map_id")
    }
    if rationale_by_map:
        lines.extend(["", "权重说明："])
        for map_id, rationale in list(rationale_by_map.items())[:4]:
            notes = []
            holding_count = str(rationale.get("holding_count_rationale") or "").strip()
            if holding_count:
                notes.append(holding_count)
            sleeve_notes = [
                str(value).strip()
                for value in rationale.get("sleeve_weight_rationale") or []
                if str(value).strip()
            ]
            notes.extend(sleeve_notes[:2])
            if notes:
                lines.append(f"- {map_id}：" + "；".join(notes[:3]))

    paths = [run.get("context_path"), run.get("portfolio_maps_path")]
    paths = [str(path) for path in paths if path]
    if paths:
        lines.extend(["", "本地审计文件：", *[f"- {path}" for path in paths]])

    lines.extend(_warning_lines(response.get("warnings") or result.get("warnings") or []))
    lines.extend(
        [
            "",
            "下一步：你可以选择一个 map_id，或提出修改意见；当前版本仍不会下单。",
            f"portfolio_architect_result_artifact_id: {data.get('portfolio_architect_result_artifact_id')}",
            f"portfolio_architect_run_artifact_id: {data.get('portfolio_architect_run_artifact_id')}",
            f"candidate_triage_artifact_id: {data.get('candidate_triage_artifact_id')}",
            f"内部编号：{response['session_id']}",
        ]
    )
    return "\n".join(lines)


def _display_theme_discovery(response: dict[str, Any]) -> str:
    data = response.get("data") or {}
    discovery = data.get("theme_discovery") or {}
    domains = discovery.get("domain_tree") or []
    seeds = discovery.get("seed_symbols") or []
    coverage = discovery.get("coverage_requirements") or []
    filter_plans = discovery.get("filter_plans_by_layer") or []
    executed_probes = discovery.get("executed_filter_probes") or []
    layer_audits = discovery.get("layer_filter_audits") or []
    omissions = discovery.get("omissions_to_investigate") or []
    next_enrichment = discovery.get("next_enrichment_needed") or []
    futu_tool_calls = ((discovery.get("pydantic_ai") or {}).get("futu_tool_calls") or [])
    lines = [
        f"已只生成“{_theme_label(str(response.get('theme') or ''))}”的 discovery。",
        "后续步骤已暂停：没有生成候选池 enrichment、SEC/财报验证、投资主线综合、组合版图、建仓计划或订单。",
    ]
    thesis = str(discovery.get("initial_thesis") or "").strip()
    if thesis:
        lines.extend(["", f"初步主题判断：{thesis}"])
    if filter_plans or executed_probes:
        lines.extend(
            [
                "",
                "Discovery v1 摘要：",
                f"- 分层筛选计划：{len(filter_plans)} 个",
                f"- 已执行 Futu probe：{len(executed_probes)} 个",
                f"- 候选种子：{len(seeds)} 个",
                f"- 待继续排查：{len(omissions)} 项",
            ]
        )
    if filter_plans:
        lines.extend(["", "分层筛选计划："])
        for plan in filter_plans[:8]:
            layer_name = plan.get("layer_name") or plan.get("layer_key")
            use_now = []
            deferred = []
            for decision in plan.get("filter_decisions") or []:
                category = str(decision.get("category") or "")
                fields = ", ".join(str(item) for item in decision.get("planned_fields") or [] if item)
                detail = f"{category}({fields})" if fields else category
                if decision.get("decision") == "use_now":
                    use_now.append(detail)
                elif decision.get("decision") == "defer_to_later_enrichment":
                    deferred.append(detail)
            line = f"- {layer_name}"
            if use_now:
                line += "；使用：" + "、".join(use_now[:6])
            if deferred:
                line += "；后续补充：" + "、".join(deferred[:4])
            lines.append(line)
    if executed_probes:
        lines.extend(["", "已执行的 Futu 筛选 probe："])
        for probe in executed_probes[:10]:
            symbols = ", ".join(str(value) for value in probe.get("candidate_symbols") or [] if value)
            specs = probe.get("stock_filter_specs") or []
            fields = _stock_filter_field_summary(specs)
            line = f"- {probe.get('layer_key')}: {probe.get('probe_type')}"
            if probe.get("plate_code"):
                line += f"，plate={probe.get('plate_code')}"
            if fields:
                line += f"，fields={fields}"
            if probe.get("result_count") is not None:
                line += f"，结果数={probe.get('result_count')}"
            if symbols:
                line += f"，候选={symbols}"
            lines.append(line)
    if futu_tool_calls:
        latest_call = futu_tool_calls[-1]
        lines.extend(
            [
                "",
                "Futu 探索调用：",
                f"- 板块关键词：{', '.join(str(value) for value in latest_call.get('plate_keywords') or []) or 'N/A'}",
                f"- 主动检查标的：{', '.join(str(value) for value in latest_call.get('must_check_symbols') or []) or 'N/A'}",
                f"- 返回候选数：{latest_call.get('candidate_count', 0)}，匹配板块数：{latest_call.get('plate_match_count', 0)}",
            ],
        )
    if domains:
        lines.extend(["", "领域分层："])
        for domain in domains[:8]:
            domain_name = domain.get("name") or domain.get("key")
            domain_thesis = str(domain.get("thesis") or "").strip()
            lines.append(f"- {domain_name}：{domain_thesis or _importance_label(domain.get('importance'))}")
            for subdomain in (domain.get("subdomains") or [])[:6]:
                candidates = subdomain.get("candidates") or []
                candidate_text = ", ".join(
                    str(candidate.get("symbol"))
                    for candidate in candidates[:8]
                    if candidate.get("symbol")
                )
                thesis_text = str(subdomain.get("thesis") or "").strip()
                suffix = f"（{candidate_text}）" if candidate_text else ""
                lines.append(f"  - {subdomain.get('name') or subdomain.get('key')}{suffix}：{thesis_text}")
    if coverage:
        lines.extend(["", "覆盖检查项："])
        for item in coverage[:10]:
            candidates = ", ".join(str(value) for value in item.get("candidate_symbols") or [])
            must = ", ".join(str(value) for value in item.get("must_consider_symbols") or [])
            line = f"- {item.get('name') or item.get('key')}"
            if candidates:
                line += f"：{candidates}"
            if must:
                line += f"；必须解释取舍：{must}"
            lines.append(line)
    if seeds:
        lines.extend(["", f"种子标的（{len(seeds)} 个）："])
        for seed in seeds[:40]:
            role = str(seed.get("role") or "").strip()
            rationale = str(seed.get("rationale") or "").strip()
            text = f"- {seed.get('symbol')}"
            if role:
                text += f"：{role}"
            if rationale:
                text += f"。{rationale}"
            lines.append(text)
        if len(seeds) > 40:
            lines.append(f"- 其余 {len(seeds) - 40} 个已保存在 artifact 中。")
    if layer_audits:
        lines.extend(["", "筛选审计要点："])
        for audit in layer_audits[:6]:
            summary = str(audit.get("result_summary") or "").strip()
            if summary:
                lines.append(f"- {audit.get('layer_name') or audit.get('layer_key')}：{summary}")
    if omissions:
        lines.extend(["", "待继续排查："])
        for item in omissions[:8]:
            lines.append(f"- {item}")
    if next_enrichment:
        lines.extend(["", "后续需要补的数据："])
        for item in next_enrichment[:8]:
            lines.append(f"- {item}")
    lines.extend(_warning_lines(response.get("warnings") or discovery.get("warnings") or []))
    lines.extend(
        [
            "",
            f"theme_discovery_artifact_id: {data.get('theme_discovery_artifact_id')}",
            f"内部编号：{response['session_id']}",
        ],
    )
    return "\n".join(lines)


def _display_status(response: dict[str, Any]) -> str:
    data = response.get("data") or {}
    artifacts = data.get("artifacts") or []
    state = str(response.get("state") or "")
    status = str(response.get("status") or "")
    lines = [_status_headline(status, state)]

    if status == WorkflowStatus.FAILED.value:
        lines.append(f"- 停在：{_state_label(state)}")
        latest_error = data.get("latest_error") or {}
        if latest_error.get("message"):
            lines.append(f"- 原因：{latest_error['message']}")
        if state == WorkflowState.DRAFTING_TARGET_PORTFOLIO_MAPS.value:
            lines.append("已保存候选粗筛结果；你可以说“继续生成组合版图”来重试这一段。")
        else:
            lines.append("你可以说“重新开始 AI 版图”，或“取消这个 workflow”。")
    elif status == WorkflowStatus.ACTIVE.value:
        lines.append(f"- 当前步骤：{_state_label(state)}")
        lines.append("如果这个状态停留很久，通常表示上次生成被中断；可以让我重新开始或取消后重试。")
    elif status == WorkflowStatus.WAITING_FOR_HUMAN.value:
        lines.append(f"- 当前步骤：{_state_label(state)}")
        human_action = response.get("human_action") or {}
        prompt = human_action.get("prompt") or {}
        question = prompt.get("question")
        if question:
            lines.append(f"- 需要你确认：{question}")
        options = prompt.get("options") or []
        for index, option in enumerate(options[:4], start=1):
            if not isinstance(option, dict):
                continue
            lines.append(
                f"  {index}. {option.get('name') or option.get('option_id')} "
                f"({option.get('option_id')})"
            )
    else:
        lines.append(f"- 当前步骤：{_state_label(state)}")

    if artifacts:
        latest = artifacts[-5:]
        lines.append("- 已保存资料：" + "、".join(_artifact_type_label(item["type"]) for item in latest))
    lines.append(f"内部编号：{response['session_id']}")
    return "\n".join(lines)


def _display_generic(response: dict[str, Any]) -> str:
    lines = [
        "投资组合版图 workflow 已更新。",
        f"- 当前状态：{_state_label(str(response.get('state') or ''))}",
        f"- 主题：{_theme_label(str(response.get('theme') or ''))}",
    ]
    message = (response.get("data") or {}).get("message")
    if message:
        lines.append(f"- message: {message}")
    lines.append(f"内部编号：{response['session_id']}")
    return "\n".join(lines)


def _display_candidate_triage_plan(response: dict[str, Any]) -> str:
    data = response.get("data") or {}
    plan = data.get("candidate_triage_plan") or {}
    lines = [
        f"已生成“{_theme_label(str(response.get('theme') or ''))}”的候选粗筛策略计划，正在等你选择。",
        "当前还没有生成最终 deep research 队列，也没有做 SEC/财报深研、组合权重、买卖计划或订单。",
    ]
    summary = str(plan.get("planning_summary") or "").strip()
    if summary:
        lines.extend(["", f"计划摘要：{summary}"])
    lines.extend(
        [
            "",
            "你可以选一个策略继续：",
        ]
    )
    for index, option in enumerate(plan.get("strategy_options") or [], start=1):
        if not isinstance(option, dict):
            continue
        lines.append(
            f"{index}. {option.get('name') or option.get('option_id')} "
            f"({option.get('option_id')})"
        )
        description = str(option.get("description") or "").strip()
        if description:
            lines.append(f"   - 方向：{description}")
        lines.append(
            f"   - 研究预算：deep {option.get('deep_research_total', 0)} 个，"
            f"watchlist 约 {option.get('expected_watchlist_count', 0)} 个"
        )
        best_for = str(option.get("best_for") or "").strip()
        if best_for:
            lines.append(f"   - 适合：{best_for}")
        budgets = option.get("layer_budgets") or []
        if budgets:
            top = [
                f"{item.get('layer_name') or item.get('layer_key')} {item.get('deep_research_count', 0)}"
                for item in budgets[:6]
                if isinstance(item, dict)
            ]
            if top:
                lines.append(f"   - 主要层级预算：{'; '.join(top)}")
        tradeoffs = [str(item) for item in option.get("tradeoffs") or [] if item]
        if tradeoffs:
            lines.append(f"   - 取舍：{tradeoffs[0]}")
    recommended = str(plan.get("recommended_option_id") or "").strip()
    if recommended:
        lines.extend(["", f"默认推荐：{recommended}"])
    prompt = str(plan.get("prompt_to_user") or "").strip()
    if prompt:
        lines.extend(["", f"下一步：{prompt}"])
    lines.extend(
        [
            "",
            "你可以直接回复：",
            "- 选 1",
            "- 选 bottleneck_momentum，但 SNDK/COHR/LITE/WDC/MRVL 必须进入 deep research",
            "- 用平衡覆盖，但 deep research 控制在 25 个以内",
        ]
    )
    lines.extend(_warning_lines(response.get("warnings") or plan.get("warnings") or []))
    lines.extend(
        [
            "",
            f"candidate_triage_plan_artifact_id: {data.get('candidate_triage_plan_artifact_id')}",
            f"futu_lightweight_enrichment_artifact_id: {data.get('futu_lightweight_enrichment_artifact_id')}",
            f"theme_discovery_artifact_id: {data.get('theme_discovery_artifact_id')}",
            f"内部编号：{response['session_id']}",
        ],
    )
    return "\n".join(lines)


def _display_candidate_triage(response: dict[str, Any]) -> str:
    data = response.get("data") or {}
    triage = data.get("candidate_triage") or {}
    selection = data.get("selected_triage_strategy") or {}
    lines = [
        f"已完成“{_theme_label(str(response.get('theme') or ''))}”的候选粗筛。",
        "这是给后续 SEC/财报/事件 deep research 使用的研究队列，不是目标组合、权重、交易计划或订单。",
    ]
    selected = str(selection.get("selected_option_id") or "").strip()
    if selected:
        lines.append(f"- 使用的筛选策略：{selected}")
    summary = str(triage.get("triage_summary") or "").strip()
    if summary:
        lines.extend(["", f"粗筛摘要：{summary}"])
    deep = triage.get("deep_enrichment_queue") or []
    watch = triage.get("watchlist") or []
    deferred = triage.get("deferred") or []
    rejected = triage.get("rejected") or []
    lines.extend(
        [
            "",
            f"队列规模：deep research {len(deep)} 个，watchlist {len(watch)} 个，deferred {len(deferred)} 个，rejected {len(rejected)} 个。",
        ]
    )
    if deep:
        lines.extend(["", "Deep research 队列："])
        for item in deep[:40]:
            if not isinstance(item, dict):
                continue
            needs = ", ".join(str(value) for value in item.get("research_needs") or [] if value)
            rationale = str(item.get("rationale") or "").strip()
            line = f"- {item.get('symbol')}：{item.get('role') or item.get('priority')}"
            if needs:
                line += f"；需要补：{needs}"
            if rationale:
                line += f"。{rationale}"
            lines.append(line)
        if len(deep) > 40:
            lines.append(f"- 其余 {len(deep) - 40} 个已保存在 artifact 中。")
    if watch:
        watch_symbols = ", ".join(str(item.get("symbol")) for item in watch[:40] if isinstance(item, dict))
        if watch_symbols:
            lines.extend(["", f"Watchlist：{watch_symbols}"])
    lines.extend(_warning_lines(response.get("warnings") or triage.get("warnings") or []))
    lines.extend(
        [
            "",
            "下一步应进入 deep research：只对 deep research 队列补 SEC/财报数字、事件、新闻和更深的基本面验证。",
            f"candidate_triage_artifact_id: {data.get('candidate_triage_artifact_id')}",
            f"内部编号：{response['session_id']}",
        ]
    )
    return "\n".join(lines)


def _theme_label(value: str) -> str:
    labels = {
        "ai": "AI 主题",
        "storage": "存储主题",
        "semiconductor": "半导体主题",
        "power": "电力主题",
    }
    return labels.get(value, value)


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return ""


def _importance_label(value: Any) -> str:
    labels = {
        "core": "核心",
        "important": "重要",
        "optional": "可选",
        "high": "高优先级",
        "medium": "中优先级",
        "low": "低优先级",
    }
    return labels.get(str(value), str(value or ""))


def _status_headline(status: str, state: str) -> str:
    if status == WorkflowStatus.FAILED.value:
        return "这次组合版图生成没有完成。"
    if status == WorkflowStatus.ACTIVE.value:
        return "组合版图还在处理中，暂时没有生成最终建议。"
    if status == WorkflowStatus.WAITING_FOR_HUMAN.value:
        return "组合版图正在等你确认下一步。"
    if status == WorkflowStatus.COMPLETED.value:
        return "组合版图流程已完成。"
    if status == WorkflowStatus.CANCELLED.value:
        return "组合版图流程已取消。"
    return f"组合版图当前处于：{_state_label(state)}"


def _state_label(value: str) -> str:
    labels = {
        WorkflowState.NEW.value: "刚创建",
        WorkflowState.NEEDS_POLICY_CONFIRMATION.value: "等待确认策略边界",
        WorkflowState.EXPANDING_THEME.value: "拆解主题范围",
        WorkflowState.THEME_DISCOVERY_COMPLETE.value: "已完成主题 discovery",
        WorkflowState.BUILDING_MARKET_ARTIFACTS.value: "收集行情和市场资料",
        WorkflowState.BUILDING_UNBIASED_CANDIDATE_POOL.value: "生成候选标的池",
        WorkflowState.NEEDS_CANDIDATE_TRIAGE_STRATEGY.value: "等待选择候选粗筛策略",
        WorkflowState.CANDIDATE_TRIAGE_COMPLETE.value: "已完成候选粗筛",
        WorkflowState.VALIDATING_EVIDENCE.value: "检查证据和数据质量",
        WorkflowState.REFLECTING_CANDIDATE_POOL.value: "复核候选池覆盖是否完整",
        WorkflowState.DRAFTING_TARGET_PORTFOLIO_MAPS.value: "生成目标组合版图",
        WorkflowState.REFLECTING_MAPS.value: "复核组合版图",
        WorkflowState.NEEDS_PORTFOLIO_MAP_SELECTION.value: "等待选择组合版图",
        WorkflowState.NEEDS_PORTFOLIO_MAP_REVIEW.value: "等待查看并选择组合版图",
        WorkflowState.TARGET_PORTFOLIO_MAP_SELECTED.value: "已选择目标组合版图",
        WorkflowState.REVISING_PORTFOLIO_MAP.value: "修订目标组合版图",
        WorkflowState.NEEDS_PORTFOLIO_REVISION_CLARIFICATION.value: "等待补充修订要求",
        WorkflowState.NEEDS_PORTFOLIO_REVISION_REVIEW.value: "等待确认修订版图",
        WorkflowState.TARGET_PORTFOLIO_MAP_REVISION_SELECTED.value: "已选择目标组合修订版",
        WorkflowState.READING_CURRENT_PORTFOLIO.value: "读取当前持仓",
        WorkflowState.BUILDING_CONSTRUCTION_PLAN.value: "生成建仓/减仓计划",
        WorkflowState.TRACKING_PLAN.value: "跟踪执行计划",
        WorkflowState.CANCELLED.value: "已取消",
    }
    return labels.get(value, value or "未知状态")


def _artifact_type_label(value: str) -> str:
    labels = {
        "initial_request": "初始需求",
        "policy": "策略边界",
        "theme_discovery": "主题 discovery",
        "futu_lightweight_enrichment": "Futu 轻量数据",
        "candidate_triage_plan": "候选粗筛策略计划",
        "triage_strategy_selection": "候选粗筛策略选择",
        "candidate_triage": "候选粗筛结果",
        "candidate_pool": "候选池",
        "research_trace": "研究来源",
        "market_context": "市场环境",
        "theme_exposure_map": "主题暴露",
        "technical_summary": "技术面摘要",
        "liquidity_context": "流动性资料",
        "options_surface": "期权资料",
        "sec_filings_context": "SEC 披露资料",
        "fundamental_quality": "基本面质量",
        "valuation_context": "估值资料",
        "earnings_event_calendar": "财报/事件日历",
        "correlation_and_diversification": "相关性和分散度",
        "benchmark_context": "基准对照",
        "market_regime": "市场状态",
        "positioning_and_sentiment": "情绪和定位",
        "analyst_revision_context": "分析师修正",
        "risk_scenario": "风险情景",
        "candidate_pool_reflection": "候选池复核",
        "thesis_synthesis": "投资主线综合",
        "portfolio_architect_result": "目标组合版图",
        "portfolio_architect_run": "组合版图生成记录",
        "selected_portfolio_map": "已选择组合版图",
        "portfolio_revision_patch": "组合版图修订意图",
        "portfolio_map_revision": "组合版图修订稿",
        "portfolio_revision_run": "组合版图修订记录",
        "selected_portfolio_map_revision": "已确认组合版图修订稿",
        "workflow_error": "错误记录",
    }
    return labels.get(value, value)


def _warning_lines(warnings: list[str]) -> list[str]:
    if not warnings:
        return []
    return ["", "数据提示：", *[f"- {warning}" for warning in _dedupe(warnings)[:6]]]


def _source_artifact_ids(data: dict[str, Any]) -> list[str]:
    artifact_ids: list[str] = []
    for key in (
        "theme_discovery_artifact_id",
        "futu_lightweight_enrichment_artifact_id",
        "candidate_triage_plan_artifact_id",
        "triage_strategy_selection_artifact_id",
        "candidate_triage_artifact_id",
        "candidate_pool_artifact_id",
        "candidate_pool_reflection_artifact_id",
        "thesis_synthesis_artifact_id",
        "portfolio_architect_result_artifact_id",
        "portfolio_architect_run_artifact_id",
        "selected_portfolio_map_artifact_id",
        "portfolio_revision_patch_artifact_id",
        "portfolio_map_revision_artifact_id",
        "portfolio_revision_run_artifact_id",
        "selected_portfolio_map_revision_artifact_id",
    ):
        value = data.get(key)
        if isinstance(value, str) and value:
            artifact_ids.append(value)
    market_ids = data.get("market_artifact_ids")
    if isinstance(market_ids, dict):
        artifact_ids.extend(str(value) for value in market_ids.values() if value)
    artifacts = data.get("artifacts")
    if isinstance(artifacts, list):
        artifact_ids.extend(
            str(item.get("artifact_id"))
            for item in artifacts
            if isinstance(item, dict) and item.get("artifact_id")
        )
    return _dedupe(artifact_ids)


def _collect_symbols(response: dict[str, Any]) -> set[str]:
    symbols: set[str] = set()
    data = response.get("data") or {}
    symbols.update(str(symbol) for symbol in data.get("candidate_symbols") or [] if symbol)
    discovery = data.get("theme_discovery") or {}
    for seed in discovery.get("seed_symbols") or []:
        if isinstance(seed, dict) and seed.get("symbol"):
            symbols.add(str(seed["symbol"]))
    for requirement in discovery.get("coverage_requirements") or []:
        if not isinstance(requirement, dict):
            continue
        symbols.update(str(symbol) for symbol in requirement.get("candidate_symbols") or [] if symbol)
        symbols.update(str(symbol) for symbol in requirement.get("must_consider_symbols") or [] if symbol)
    for probe in discovery.get("executed_filter_probes") or []:
        if not isinstance(probe, dict):
            continue
        symbols.update(str(symbol) for symbol in probe.get("candidate_symbols") or [] if symbol)
    triage = data.get("candidate_triage") or {}
    for key in ("deep_enrichment_queue", "watchlist", "deferred", "rejected", "high_salience_omissions"):
        for item in triage.get(key) or []:
            if isinstance(item, dict) and item.get("symbol"):
                symbols.add(str(item["symbol"]))
    architect_result = data.get("portfolio_architect_result") or {}
    selection = architect_result.get("selection") or {}
    for key in (
        "selected_for_portfolio",
        "watchlist_after_enrichment",
        "deferred_after_enrichment",
        "rejected_after_enrichment",
    ):
        for item in selection.get(key) or []:
            if isinstance(item, dict) and item.get("symbol"):
                symbols.add(str(item["symbol"]))
    portfolio_maps = architect_result.get("portfolio_maps") or data.get("portfolio_maps") or {}
    for portfolio_map in portfolio_maps.get("maps") or []:
        if not isinstance(portfolio_map, dict):
            continue
        for holding in portfolio_map.get("holdings") or []:
            if isinstance(holding, dict) and holding.get("symbol"):
                symbols.add(str(holding["symbol"]))
    selected_map = data.get("selected_portfolio_map") or {}
    for holding in selected_map.get("holdings") or []:
        if isinstance(holding, dict) and holding.get("symbol"):
            symbols.add(str(holding["symbol"]))
    revision = data.get("portfolio_map_revision") or {}
    revised_map = revision.get("revised_map") or {}
    for holding in revised_map.get("holdings") or []:
        if isinstance(holding, dict) and holding.get("symbol"):
            symbols.add(str(holding["symbol"]))
    selected_revision = data.get("selected_portfolio_map_revision") or {}
    selected_revision_map = selected_revision.get("selected_map") or {}
    for holding in selected_revision_map.get("holdings") or []:
        if isinstance(holding, dict) and holding.get("symbol"):
            symbols.add(str(holding["symbol"]))
    return {symbol for symbol in symbols if symbol}


def _collect_map_ids(response: dict[str, Any]) -> set[str]:
    data = response.get("data") or {}
    map_ids = {str(map_id) for map_id in data.get("map_ids") or [] if map_id}
    architect_result = data.get("portfolio_architect_result") or {}
    portfolio_maps = architect_result.get("portfolio_maps") or data.get("portfolio_maps") or {}
    for portfolio_map in portfolio_maps.get("maps") or []:
        if isinstance(portfolio_map, dict) and portfolio_map.get("map_id"):
            map_ids.add(str(portfolio_map["map_id"]))
    selected_map = data.get("selected_portfolio_map") or {}
    if isinstance(selected_map, dict) and selected_map.get("map_id"):
        map_ids.add(str(selected_map["map_id"]))
    revision = data.get("portfolio_map_revision") or {}
    revised_map = revision.get("revised_map") or {}
    if isinstance(revised_map, dict) and revised_map.get("map_id"):
        map_ids.add(str(revised_map["map_id"]))
    selected_revision = data.get("selected_portfolio_map_revision") or {}
    selected_revision_map = selected_revision.get("selected_map") or {}
    if isinstance(selected_revision_map, dict) and selected_revision_map.get("map_id"):
        map_ids.add(str(selected_revision_map["map_id"]))
    return {map_id for map_id in map_ids if map_id}


def _stock_filter_field_summary(specs: list[dict[str, Any]]) -> str:
    fields: list[str] = []
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        field = spec.get("stock_field") or spec.get("stock_field1") or spec.get("field")
        if not field:
            continue
        text = str(field)
        if spec.get("filter_min") is not None:
            text += f">={spec.get('filter_min')}"
        if spec.get("filter_max") is not None:
            text += f"<={spec.get('filter_max')}"
        if spec.get("days") is not None:
            text += f"/{spec.get('days')}d"
        fields.append(text)
    return ", ".join(_dedupe(fields)[:4])


def _discovery_mode_label(payload: dict[str, Any], *, suffix: str) -> str:
    prefix = "websearch_discovery" if is_websearch_discovery_mode(payload) else "ai_discovery_v1"
    return f"{prefix}_{suffix}"


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
