"""Read-only model-router inspection commands."""
from __future__ import annotations

import json
from pathlib import Path


def _router_db_path():
    from agent.model_router.pipeline import default_db_path
    return default_db_path()


def _load_full_cfg():
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly() or {}
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def _load_cfg():
    cfg = _load_full_cfg()
    router = cfg.get("model_router", {})
    return router if isinstance(router, dict) else {}


def _score_value(candidate) -> float:
    return float(
        getattr(candidate, "composite_score", getattr(candidate, "score", 0.0))
    )


def _rejected_payload(decision) -> list[dict]:
    rejected = {}
    for note in decision.rejected:
        model, separator, reason = str(note).partition(":")
        rejected[model.strip()] = reason.strip() if separator else "rejected"
    for candidate in decision.candidates:
        reason = getattr(candidate, "rejected_reason", None)
        if reason:
            rejected[str(candidate.model_id)] = str(reason)
    return [
        {"model": model, "reason": rejected[model]}
        for model in sorted(rejected)
    ]


def explain_router(
    router_cfg: dict,
    *,
    prompt: str,
    current_model: str,
    session_id: str = "",
    estimated_input_tokens=None,
    has_images: bool = False,
    turn_type=None,
    force_model_id=None,
):
    """Run the production pipeline without inference or mutable side effects."""
    from agent.model_router import RoutingRequest, pipeline_from_config

    pipeline = pipeline_from_config(router_cfg, read_only=True)
    mode = str(router_cfg.get("mode", "off") or "off")
    if pipeline is None:
        return {
            "mode": mode,
            "stage": "fallback",
            "reason": "no_configured_candidates",
            "selected_model": current_model,
            "suggested_model": None,
            "rejected_candidates": [],
            "features": {},
            "scores": [],
            "pin": None,
            "fallback_reason": "no_configured_candidates",
        }

    request = RoutingRequest(
        prompt_text=prompt,
        session_id=session_id,
        estimated_input_tokens=estimated_input_tokens,
        has_images=has_images,
        turn_type=turn_type,
        force_model_id=force_model_id,
    )
    pin = pipeline._state.load_pin(session_id) if (
        session_id and pipeline._state is not None
    ) else None
    decision = pipeline.route(
        request,
        current_model=current_model,
        mode=mode,
        dry_run=True,
    )
    scores = [
        {
            "model": str(candidate.model_id),
            "score": round(_score_value(candidate), 6),
            "rejected_reason": getattr(candidate, "rejected_reason", None),
        }
        for candidate in decision.candidates
    ]
    fallback_reason = (
        decision.reason_code
        if decision.stage in {"fallback", "pipeline_error", "context_overflow"}
        else None
    )
    return {
        "mode": mode,
        "stage": decision.stage,
        "reason": decision.reason_code,
        "selected_model": decision.selected_model,
        "suggested_model": decision.suggestion or None,
        "rejected_candidates": _rejected_payload(decision),
        "features": decision.features,
        "scores": scores,
        "pin": (
            {
                "model": pin.pinned_model_id,
                "reason": pin.pin_reason,
                "turns_held": pin.turns_held,
            }
            if pin is not None
            else None
        ),
        "fallback_reason": fallback_reason,
    }


def cmd_router(args):
    from agent.model_router.state import RouterStateStore
    from agent.model_router.telemetry import RouterTelemetry

    action = getattr(args, "router_action", None) or "status"
    cfg = _load_cfg()
    db_path = _router_db_path()

    if action == "status":
        state = RouterStateStore(db_path)
        payload = {
            "mode": cfg.get("mode", "off"),
            "candidate_count": len(cfg.get("candidates", [])) if isinstance(cfg.get("candidates", []), list) else 0,
            "pinned_sessions": len(state.list_pins()),
            "db_path": str(db_path),
            "db_size": db_path.stat().st_size if db_path.exists() else 0,
        }
    elif action == "history":
        telemetry = RouterTelemetry(db_path)
        payload = telemetry.history(
            limit=max(1, min(1000, int(getattr(args, "limit", 20)))),
            session_id=getattr(args, "session", "") or "",
        )
    elif action == "stats":
        telemetry = RouterTelemetry(db_path)
        payload = telemetry.stats()
    elif action == "explain":
        full_cfg = _load_full_cfg()
        configured_model = full_cfg.get("model", "")
        if isinstance(configured_model, dict):
            configured_model = configured_model.get("model", "")
        raw_candidates = cfg.get("candidates") or []
        first_model = next(
            (
                item.get("model")
                for item in raw_candidates
                if isinstance(item, dict) and item.get("model")
            ),
            "",
        )
        current_model = (
            getattr(args, "current_model", "")
            or str(configured_model or "")
            or str(first_model or "")
        )
        payload = explain_router(
            cfg,
            prompt=str(getattr(args, "prompt", "") or ""),
            current_model=current_model,
            session_id=str(getattr(args, "session", "") or ""),
            estimated_input_tokens=getattr(args, "estimated_input_tokens", None),
            has_images=bool(getattr(args, "has_images", False)),
            turn_type=getattr(args, "turn_type", None),
            force_model_id=getattr(args, "force_model", None),
        )
    else:
        raise ValueError(f"unknown router action: {action}")

    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
