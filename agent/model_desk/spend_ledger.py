"""C08 — Spend/token ledger snapshot (SessionDB + account usage views)."""

from __future__ import annotations

from typing import Any, Dict, Optional


def _session_db_spend(limit_sessions: int = 20) -> Optional[Dict[str, Any]]:
    """Aggregate recent session token usage from SessionDB when available."""
    try:
        from hermes_state import SessionDB

        db = SessionDB()
    except Exception:
        return None

    try:
        sessions = []
        if hasattr(db, "list_sessions_rich"):
            sessions = db.list_sessions_rich(limit=limit_sessions) or []
        elif hasattr(db, "list_sessions"):
            sessions = db.list_sessions(limit=limit_sessions) or []
        elif hasattr(db, "get_recent_sessions"):
            sessions = db.get_recent_sessions(limit=limit_sessions) or []

        totals = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "reasoning_tokens": 0,
            "sessions": 0,
        }
        for row in sessions:
            if not isinstance(row, dict):
                continue
            totals["sessions"] += 1
            for key in (
                "input_tokens",
                "output_tokens",
                "cache_read_tokens",
                "cache_write_tokens",
                "reasoning_tokens",
                # Common SessionDB aliases
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
            ):
                if key not in totals and key in (
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                ):
                    continue
                try:
                    val = int(row.get(key) or 0)
                except (TypeError, ValueError):
                    continue
                if key == "prompt_tokens":
                    totals["input_tokens"] += val
                elif key == "completion_tokens":
                    totals["output_tokens"] += val
                elif key in totals:
                    totals[key] += val
        totals["total_tokens"] = (
            totals["input_tokens"]
            + totals["output_tokens"]
            + totals["cache_read_tokens"]
            + totals["cache_write_tokens"]
        )
        return {"ok": True, "source": "session_db", "totals": totals, "rows": len(sessions)}
    except Exception as exc:
        return {"ok": False, "source": "session_db", "error": str(exc)[:160]}
    finally:
        try:
            if hasattr(db, "close"):
                db.close()
        except Exception:
            pass


def _credits_layer() -> Dict[str, Any]:
    """Soft credits/account view — never raises, never hits network hard-fail."""
    try:
        from agent.account_usage import build_credits_view

        view = build_credits_view(markdown=False, timeout=2.0)
        payload: Dict[str, Any] = {"ok": True, "source": "credits_view"}
        if view is None:
            payload["empty"] = True
            return payload
        # CreditsView is a dataclass-like object — serialize softly
        if hasattr(view, "__dict__"):
            payload["view"] = {
                k: v
                for k, v in vars(view).items()
                if not callable(v) and not str(k).startswith("_")
            }
        elif isinstance(view, dict):
            payload["view"] = view
        else:
            payload["view"] = str(view)[:200]
        return payload
    except Exception as exc:
        return {"ok": False, "source": "credits_view", "error": str(exc)[:120]}


def _account_usage_layer() -> Dict[str, Any]:
    try:
        from agent.account_usage import fetch_account_usage
        from hermes_cli.config import load_config

        cfg = load_config() if callable(load_config) else {}
        model_cfg = (cfg.get("model") or {}) if isinstance(cfg, dict) else {}
        provider = str(model_cfg.get("provider") or "").strip() or None
        snap = fetch_account_usage(
            provider,
            base_url=model_cfg.get("base_url"),
        )
        if snap is None:
            return {"ok": True, "source": "account_usage", "empty": True}
        data: Dict[str, Any] = {"ok": True, "source": "account_usage"}
        if hasattr(snap, "__dict__"):
            data["usage"] = {
                k: v
                for k, v in vars(snap).items()
                if not callable(v) and not str(k).startswith("_")
            }
        else:
            data["usage"] = str(snap)[:200]
        return data
    except Exception as exc:
        return {"ok": False, "source": "account_usage", "error": str(exc)[:120]}


def spend_ledger_snapshot(*, include_credits: bool = True) -> dict[str, Any]:
    """Return a soft spend/token ledger snapshot without hard API failures.

    Layers (best-effort, never raises):
    1. SessionDB recent token aggregates
    2. Credits view (Nous / local credits state)
    3. Account usage fetch (provider-dependent; soft-timeout)
    """
    out: Dict[str, Any] = {"ok": True, "skipped": False, "layers": []}

    session_layer = _session_db_spend()
    if session_layer:
        out["layers"].append(session_layer)
        if session_layer.get("ok") and session_layer.get("totals"):
            out["totals"] = session_layer["totals"]

    if include_credits:
        credits = _credits_layer()
        out["layers"].append(credits)
        if credits.get("ok") and credits.get("view"):
            out["credits"] = credits["view"]

    usage = _account_usage_layer()
    out["layers"].append(usage)
    if usage.get("ok") and usage.get("usage"):
        out["usage"] = usage["usage"]

    useful = bool(out.get("totals") or out.get("credits") or out.get("usage"))
    # Also useful if SessionDB returned rows even with zero tokens
    if not useful and session_layer and session_layer.get("ok"):
        useful = int(session_layer.get("rows") or 0) > 0 or bool(
            session_layer.get("totals")
        )
    if not useful:
        out["skipped"] = True
        out["note"] = "No local spend data yet — run a chat turn first."
    return out
