"""ModelArk Coding Plan: $0 "modelark subscription" cost + a cap-equivalent for the $1 card cap.

Richie, 2026-09-11: "they will report no cost, so that is why you need to adjust reporting and
calculation." Version 1.x stored a notional DeepSeek-list price AS the cost, which every native Hermes
surface (per-turn label, insights, dashboard) then showed as money. 2.0 records the truth and moves the
notional figure into the only place it is needed:

* **Reporting** — ``usage_pricing.resolve_billing_route`` is wrapped so any call to the Coding Plan
  endpoint resolves as ``billing_mode="subscription_included"`` (Hermes' own subscription mode, as for
  openai-codex). ``estimate_usage_cost`` then returns $0 / ``included``; our wrapper relabels that as
  ``cost_source = label = "modelark subscription"``. Rows land as estimated_cost_usd 0,
  cost_status "included", billing_mode "subscription_included".
* **Calculation** — ``kanban_db._session_cost_in_db`` (called by ``_cumulative_session_cost`` →
  ``enforce_max_cost`` and the overwatch escalator) is wrapped to ADD a cap-equivalent for Coding Plan
  rows: their tokens priced at DeepSeek's own list rate (bundled ``deepseek`` snapshot). Without it a
  worker looping on ModelArk would never trip the $1 cap — the 2026-09-08 failure.

2.1: the cap-equivalent rates (and extra model names) come from the fleet's single model file,
``~/.hermes/fleet/models.json`` (written by plugins/fleet-models), falling back to the bundled list.

Only the Coding Plan path (``/api/coding/``) is treated as subscription: ``/api/v3`` on the same host
bills per call and must keep reporting as unpriced.
"""
from __future__ import annotations

import dataclasses
import functools
import logging
import sqlite3
import sys
from decimal import Decimal

logger = logging.getLogger(__name__)

LABEL = "modelark subscription"
ARK_HOST = "ark.ap-southeast.bytepluses.com"
ARK_SUB_PATH = "/api/coding/"
PROVIDER_KEYS = {"modelark", "custom:modelark"}
# Ark model ids AND the served names the Coding Plan answers with -> DeepSeek list-price family.
FAMILY = {
    "deepseek-v4-flash-ga-260731": "deepseek-v4-flash", "deepseek-v4-flash-260425": "deepseek-v4-flash",
    "deepseek-v4-flash": "deepseek-v4-flash",
    "deepseek-v4-pro-ga-260813": "deepseek-v4-pro", "deepseek-v4-pro-260425": "deepseek-v4-pro",
    "deepseek-v4-pro": "deepseek-v4-pro",
}
_MARK = "_modelark_subscription_wrapped"
_REG = [None, {}]


def fleet_rates() -> dict:
    """{model name (lower): (in, out, cache_read) $/M} for every ModelArk model in the fleet's single model
    file (~/.hermes/fleet/models.yaml — ids AND served names), mtime-cached. The cap-equivalent is priced
    from there so a rate change on the dashboard reaches the $1 cap without a deploy. {} when the file is
    absent or unreadable: the bundled DeepSeek list (below) then applies."""
    import json
    from pathlib import Path
    try:
        from hermes_constants import get_default_hermes_root
        root = Path(get_default_hermes_root())
    except Exception:  # noqa: BLE001
        root = Path.home() / ".hermes"
    p = root / "fleet" / "models.json"
    try:
        mt = p.stat().st_mtime
    except OSError:
        return {}
    if _REG[0] == mt:
        return _REG[1]
    out = {}
    try:
        for m in (json.loads(p.read_text()).get("models") or {}).values():
            if not isinstance(m, dict) or m.get("provider") != "modelark":
                continue
            ce = m.get("cap_equivalent") or {}
            r = tuple(Decimal(str(ce.get(k) or 0)) for k in ("input", "output", "cache_read"))
            for name in [m.get("id"), *(m.get("served_as") or [])]:
                if name:
                    out[str(name).strip().lower()] = r
    except Exception:  # noqa: BLE001
        logger.warning("modelark-pricing: fleet/models.json unreadable — using the bundled DeepSeek list")
        out = {}
    _REG[0], _REG[1] = mt, out
    return out


def _is_ark_model(model: str) -> bool:
    m = (model or "").strip().lower()
    return m in FAMILY or m in fleet_rates()


def is_subscription_url(base_url) -> bool:
    b = str(base_url or "").lower()
    return ARK_HOST in b and ARK_SUB_PATH in b


def _wrap_route(orig):
    @functools.wraps(orig)
    def resolve_billing_route(model_name, provider=None, base_url=None):
        route = orig(model_name, provider=provider, base_url=base_url)
        model = (model_name or "").strip()
        if is_subscription_url(base_url) or (
                not base_url and (provider or "").strip().lower() in PROVIDER_KEYS and _is_ark_model(model)):
            return dataclasses.replace(route, provider="modelark", model=model,
                                       billing_mode="subscription_included")
        return route
    setattr(resolve_billing_route, _MARK, True)
    return resolve_billing_route


def _wrap_estimate(orig, up):
    @functools.wraps(orig)
    def estimate_usage_cost(model_name, usage, *, provider=None, base_url=None, api_key=None):
        res = orig(model_name, usage, provider=provider, base_url=base_url, api_key=api_key)
        try:
            route = up.resolve_billing_route(model_name, provider=provider, base_url=base_url)
        except Exception:  # noqa: BLE001
            return res
        if res.status == "included" and route.provider == "modelark":
            return dataclasses.replace(res, source=LABEL, label=LABEL, pricing_version="modelark-subscription",
                                       notes=("ModelArk Coding Plan — flat subscription, no per-call invoice",))
        return res
    setattr(estimate_usage_cost, _MARK, True)
    return estimate_usage_cost


def _rates():
    from agent import usage_pricing as up
    out = {}
    for fam in set(FAMILY.values()):
        e = up._OFFICIAL_DOCS_PRICING.get(("deepseek", fam))
        if e is not None:
            out[fam] = (e.input_cost_per_million or Decimal(0), e.output_cost_per_million or Decimal(0),
                        e.cache_read_cost_per_million if e.cache_read_cost_per_million is not None
                        else (e.input_cost_per_million or Decimal(0)))
    return out


def cap_equivalent_rows(conn, where: str, params) -> tuple[float, int]:
    """(cap-equivalent $, calls) for Coding Plan rows of the sessions matching `where` (alias s)."""
    rates = _rates()
    usd, calls = Decimal(0), 0
    rows = conn.execute(
        "SELECT u.model, COALESCE(SUM(u.input_tokens),0), COALESCE(SUM(u.output_tokens),0), "
        "COALESCE(SUM(u.cache_read_tokens),0), COALESCE(SUM(u.cache_write_tokens),0), "
        "COALESCE(SUM(u.api_call_count),0) FROM session_model_usage u JOIN sessions s ON s.id = u.session_id "
        f"WHERE ({where}) AND u.billing_base_url LIKE ? AND COALESCE(u.cost_source,'') != 'modelark-proxy' "
        "GROUP BY u.model", [*params, f"%{ARK_HOST}{ARK_SUB_PATH}%"]).fetchall()
    fleet = fleet_rates()
    for model, inp, out, cr, cw, n in rows:
        fam = FAMILY.get((model or "").lower()) or ("deepseek-v4-pro" if "pro" in (model or "") else "deepseek-v4-flash")
        r_in, r_out, r_cr = fleet.get((model or "").strip().lower()) or rates.get(fam, (Decimal(0), Decimal(0), Decimal(0)))
        usd += (Decimal(inp) + Decimal(cw)) * r_in / 1_000_000 + Decimal(out) * r_out / 1_000_000 \
            + Decimal(cr) * r_cr / 1_000_000
        calls += int(n or 0)
    return float(usd), calls


def _wrap_session_cost(orig, kdb):
    @functools.wraps(orig)
    def _session_cost_in_db(state_db_path, prefix_for=None, task_id=None):
        base = orig(state_db_path, prefix_for=prefix_for, task_id=task_id)
        try:
            prefix = str(prefix_for).rstrip("/\\") if prefix_for else ""
            clauses, params = [], []
            if prefix:
                clauses.append("(s.cwd = ? OR s.cwd LIKE ? ESCAPE '\\')")
                params += [prefix, kdb._escape_like(prefix) + "/%"]
            if task_id:
                clauses.append("s.title LIKE ? ESCAPE '\\'")
                params.append("%" + kdb._escape_like(str(task_id)) + "%")
            if not clauses:
                return base
            conn = sqlite3.connect(f"file:{state_db_path}?mode=ro", uri=True, timeout=10)
            try:
                extra, _ = cap_equivalent_rows(conn, " OR ".join(clauses), params)
            finally:
                conn.close()
            return base + extra
        except Exception:  # noqa: BLE001 — the cap sum is fail-open by design; never raise from it
            return base
    setattr(_session_cost_in_db, _MARK, True)
    return _session_cost_in_db


def modelark_cap_equivalent(task_id: str) -> tuple[float, int]:
    """(cap-equivalent $, calls) of a card's Coding Plan usage across every ledger — for briefs/reports."""
    from pathlib import Path
    try:
        from hermes_constants import get_default_hermes_root
        root = Path(get_default_hermes_root())
    except Exception:  # noqa: BLE001
        root = Path.home() / ".hermes"
    usd, calls = 0.0, 0
    for db in [root / "state.db", *sorted((root / "profiles").glob("*/state.db"))]:
        try:
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=10)
            try:
                u, n = cap_equivalent_rows(conn, "s.title LIKE ?", [f"%{task_id}%"])
            finally:
                conn.close()
            usd += u; calls += n
        except Exception:  # noqa: BLE001
            continue
    return usd, calls


def install() -> list:
    """Wrap the three seams. Idempotent. Returns the names wrapped this call."""
    done = []
    from agent import usage_pricing as up
    if not getattr(up.resolve_billing_route, _MARK, False):
        up.resolve_billing_route = _wrap_route(up.resolve_billing_route); done.append("resolve_billing_route")
    if not getattr(up.estimate_usage_cost, _MARK, False):
        up.estimate_usage_cost = _wrap_estimate(up.estimate_usage_cost, up); done.append("estimate_usage_cost")
    # Modules that bound estimate_usage_cost by name before this plugin loaded.
    for modname in ("agent.turn_usage", "agent.insights"):
        mod = sys.modules.get(modname)
        if mod is not None and getattr(mod, "estimate_usage_cost", None) is not None \
                and not getattr(mod.estimate_usage_cost, _MARK, False):
            mod.estimate_usage_cost = up.estimate_usage_cost; done.append(modname)
    try:
        from hermes_cli import kanban_db as kdb
        if not getattr(kdb._session_cost_in_db, _MARK, False):
            kdb._session_cost_in_db = _wrap_session_cost(kdb._session_cost_in_db, kdb); done.append("_session_cost_in_db")
        kdb.modelark_cap_equivalent = modelark_cap_equivalent
    except Exception as exc:  # noqa: BLE001
        logger.warning("modelark-pricing: kanban cap seam not installed (%s) — the $1 cap is blind to ModelArk", exc)
    return done


def register(ctx) -> None:  # noqa: ARG001 — plugin loader entry point
    logger.info("modelark-pricing 2.1: wrapped %s", install())
