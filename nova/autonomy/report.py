"""The autonomy report: what earned autonomy did for one tenant, in numbers a customer reads.

Built from the same ledger the Control Centre shows, plus the audit log's promotions,
demotions and triage timings. Every number is either **measured** (counted from records)
or an **estimate** (a measurement times an assumption the reader can see and change), and
the report says which. Where something was not measured it says so rather than printing
zero: provider cost, for instance, is only computed when the per-token price is given,
because the provider does not publish one.
"""

from __future__ import annotations

import statistics
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional

from nova.autonomy import ledger as ledger_mod

DEFAULT_MINUTES_PER_APPROVAL = 3.0


def _percentile(values: list[int], share: float) -> Optional[int]:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(share * (len(ordered) - 1))))
    return ordered[index]


def build_report(
    events: Iterable[Mapping[str, Any]],
    board_answers: Iterable[Mapping[str, Any]],
    *,
    tenant_id: str,
    company: str = "",
    window: int = 100,
    minutes_per_approval: float = DEFAULT_MINUTES_PER_APPROVAL,
    price_input_per_mtok: Optional[float] = None,
    price_output_per_mtok: Optional[float] = None,
    states: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> dict[str, Any]:
    events = [e for e in events if (e.get("tenant_id") or "") == tenant_id]
    built = ledger_mod.build(events, board_answers, tenant_id=tenant_id, window=window)

    triage = [e for e in events if e.get("kind") == "policy.triage"]
    latencies = [int(e["detail"]["latency_ms"]) for e in triage
                 if isinstance((e.get("detail") or {}).get("latency_ms"), int) and (e["detail"].get("model_version"))]
    unavailable = sum(1 for e in triage
                      if any(str(f.get("why", "")).startswith("provider_unavailable")
                             for f in (e.get("detail") or {}).get("failed") or ()))
    tokens_in = sum(int(((e.get("detail") or {}).get("usage") or {}).get("input_tokens") or 0) for e in triage)
    tokens_out = sum(int(((e.get("detail") or {}).get("usage") or {}).get("output_tokens") or 0) for e in triage)
    priced = price_input_per_mtok is not None and price_output_per_mtok is not None
    cost = (tokens_in * price_input_per_mtok + tokens_out * price_output_per_mtok) / 1_000_000 if priced else None

    def changes(kind: str) -> list[dict[str, Any]]:
        return [{"ts": e.get("ts"), "action": (e.get("detail") or {}).get("action") or e.get("subject"),
                 "reason": (e.get("detail") or {}).get("reason", ""), "by": e.get("actor", ""),
                 "model_version": (e.get("detail") or {}).get("model_version", "")}
                for e in events if e.get("kind") == kind and e.get("phase") == "committed"]

    actions = []
    total_auto = total_escalated = 0
    for name, book in sorted(built["actions"].items()):
        t = book.all_time
        handled = t.autonomous_executed + t.escalations
        total_auto += t.autonomous_executed
        total_escalated += t.escalations
        actions.append({
            "action": name,
            "state": (states or {}).get(name, {}).get("state", "supervised"),
            "approval_gated_calls": handled,
            "handled_autonomously": t.autonomous_executed,
            "handled_autonomously_share": (t.autonomous_executed / handled) if handled else None,
            "escalated_to_a_person": t.escalations,
            "shadow_agreement": t.agreement,
            "shadow_agreement_window": book.window.agreement,
            "false_safe": t.false_safe,
            "false_safe_rate": t.false_safe_rate,
            "reviewed_safe_verdicts": t.shadow_reviewed,
            "incidents": t.incidents,
            "autonomous_outcome_unknown": t.autonomous_open,
        })
    gated = total_auto + total_escalated
    stamps = sorted(str(e.get("ts")) for e in events if e.get("ts"))
    return {
        "tenant_id": tenant_id,
        "company": company,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "period": {"from": stamps[0] if stamps else None, "to": stamps[-1] if stamps else None},
        "totals": {
            "approval_gated_calls": gated,
            "handled_autonomously": total_auto,
            "handled_autonomously_share": (total_auto / gated) if gated else None,
            "escalated_to_a_person": total_escalated,
        },
        "estimates": {
            "human_hours_saved": total_auto * minutes_per_approval / 60.0,
            "assumption": f"{minutes_per_approval:g} minutes of a person's time per approval avoided",
        },
        "triage": {
            "calls": len(triage),
            "provider_unavailable": unavailable,
            "latency_ms_median": int(statistics.median(latencies)) if latencies else None,
            "latency_ms_p95": _percentile(latencies, 0.95),
            "tokens": {"input": tokens_in, "output": tokens_out},
            "cost_usd": cost,
            "cost_note": ("estimate from the prices given" if priced else
                          "not computed: the provider publishes no per-token price; pass "
                          "--price-input-per-mtok and --price-output-per-mtok to estimate it"),
        },
        "actions": actions,
        "promotions": changes("autonomy.promoted"),
        "demotions": changes("autonomy.demoted"),
        "caveats": list(built["caveats"]) + [
            "Hours saved is an estimate: calls run without a person times the assumed minutes per approval.",
        ],
    }


def _pct(value: Optional[float]) -> str:
    return "not measured yet" if value is None else f"{value * 100:.1f}%"


def to_markdown(report: Mapping[str, Any]) -> str:
    totals, est, tri = report["totals"], report["estimates"], report["triage"]
    who = report.get("company") or report["tenant_id"]
    period = report["period"]
    lines = [
        f"# Autonomy report — {who}",
        "",
        f"Tenant `{report['tenant_id']}` · generated {report['generated_at']}"
        + (f" · records from {period['from']} to {period['to']}" if period["from"] else " · no records yet"),
        "",
        "## Summary",
        "",
        "| | Value | Kind |",
        "|---|---|---|",
        f"| Approval-gated calls | {totals['approval_gated_calls']} | measured |",
        f"| Handled without a person | {totals['handled_autonomously']} ({_pct(totals['handled_autonomously_share'])}) | measured |",
        f"| Sent to a person | {totals['escalated_to_a_person']} | measured |",
        f"| Human hours saved | {est['human_hours_saved']:.1f} | **estimate** ({est['assumption']}) |",
        f"| Triage calls | {tri['calls']} ({tri['provider_unavailable']} with the provider unavailable) | measured |",
        f"| Triage latency, median / p95 | {tri['latency_ms_median'] if tri['latency_ms_median'] is not None else '—'} ms"
        f" / {tri['latency_ms_p95'] if tri['latency_ms_p95'] is not None else '—'} ms | measured |",
        f"| Provider tokens, in / out | {tri['tokens']['input']} / {tri['tokens']['output']} | measured |",
        f"| Provider cost | {'$%.2f' % tri['cost_usd'] if tri['cost_usd'] is not None else 'not computed'} | "
        f"{'**estimate**' if tri['cost_usd'] is not None else '—'} ({tri['cost_note']}) |",
        "",
        "## By action",
        "",
        "| Action | State | False-safe | Shadow agreement (all / window) | Reviewed safe verdicts | Without a person | Incidents |",
        "|---|---|---|---|---|---|---|",
    ]
    for a in report["actions"]:
        lines.append(
            f"| {a['action']} | {a['state']} | {a['false_safe']} ({_pct(a['false_safe_rate'])}) | "
            f"{_pct(a['shadow_agreement'])} / {_pct(a['shadow_agreement_window'])} | {a['reviewed_safe_verdicts']} | "
            f"{a['handled_autonomously']} ({_pct(a['handled_autonomously_share'])}) | {a['incidents']} |")
    if not report["actions"]:
        lines.append("| — | no triaged action has been used yet | | | | | |")
    for title, rows in (("Promotions", report["promotions"]), ("Demotions", report["demotions"])):
        lines += ["", f"## {title}", ""]
        if rows:
            for r in rows:
                model = f" on `{r['model_version']}`" if r.get("model_version") else ""
                lines.append(f"- {r['ts']} — **{r['action']}**{model} by {r['by'] or 'NOVA'}: {r['reason'] or 'no reason recorded'}")
        else:
            lines.append("- none")
    lines += ["", "## What these numbers cannot show", ""] + [f"- {c}" for c in report["caveats"]] + [""]
    return "\n".join(lines)
