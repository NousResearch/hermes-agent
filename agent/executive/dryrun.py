"""Dry-run helper for Executive v2.

Renders an ObjectiveStateData as a human-readable string for the CLI
handler. Pure function: no side effects, no I/O.
"""

from __future__ import annotations

from typing import Any


def render_dry_run(state: Any) -> str:
    """Render an ObjectiveStateData as a human-readable dry-run output.

    ``state`` is an ObjectiveStateData (or any object with the same
    attribute names). The output is meant to be printed to a TTY or
    used in tests as a snapshot.
    """
    lines: list[str] = []
    sep_top = "─" * 64
    sep_bot = "─" * 64
    lines.append(f"╭─ Executive v2 Dry-Run ─{sep_top[27:]}╮")
    lines.append(f"│ objective_id: {state.objective_id}")
    lines.append(f"│ fingerprint:  {state.fingerprint or '(none yet)'}")
    lines.append(f"│ state:        {state.state.value} (not persisted)")
    lines.append("│")

    if state.normalized:
        n = state.normalized
        lines.append(f"│ Goal Class:        {n.get('goal_class', 'OTHER')}")
        lines.append(f"│ Risk Profile:      {n.get('risk_profile', 'low')}")
        lines.append(f"│ Complexity:        {n.get('estimated_complexity', 'XS')}")
        sc = n.get("success_criteria") or []
        if sc:
            lines.append("│ Success Criteria:")
            for i, criterion in enumerate(sc, 1):
                lines.append(f"│   {i}. {criterion}")

    if state.discovered:
        d = state.discovered
        candidates = d.get("candidates") or []
        if candidates:
            lines.append(f"│ Required Capabilities ({len(candidates)}):")
            for c in candidates[:5]:
                lines.append(
                    f"│   - {c.get('id', '?')} (score {c.get('match_score', 0):.2f})"
                )
        lines.append(f"│ Reuse Decision:    {d.get('reuse_decision', 'generate')}")
        gaps = d.get("gaps") or []
        if gaps:
            lines.append(f"│ Gaps:               {', '.join(gaps)}")

    if state.contract:
        c = state.contract
        rc = c.get("risk_components", {})
        if rc:
            lines.append("│ Risk Components:")
            lines.append(
                f"│   financial={rc.get('financial', 0)} "
                f"regulatory={rc.get('regulatory', 0)} "
                f"customer_facing={rc.get('customer_facing', 0)}"
            )
            lines.append(
                f"│   irreversibility={rc.get('irreversibility', 0)} "
                f"data_sensitivity={rc.get('data_sensitivity', 0)} "
                f"TOTAL={c.get('risk_score', 0):.2f}"
            )
        approvals = c.get("approval_requirements") or []
        if approvals:
            lines.append("│ Approval Requirements:")
            for ar in approvals:
                lines.append(
                    f"│   - {ar.get('gate', '?')} "
                    f"(approver={ar.get('approver', '?')}, "
                    f"ttl={ar.get('ttl_hours', '?')}h)"
                )
        b = c.get("budget", {})
        if b:
            lines.append(
                f"│ Budget: policy={b.get('policy', '?')} "
                f"max_iter={b.get('max_iterations', '?')} "
                f"max_dur={b.get('max_duration_minutes', '?')}min "
                f"max=${b.get('max_cost_usd', '?')}"
            )
        # ── B1 EvidencePack section (Gate C; default OFF; omitted when absent) ──
        ep_summary = (
            c.get("evidence_pack_summary")
            or getattr(state, "evidence_pack_summary", None)
        )
        ep_ref = (
            c.get("evidence_pack_ref")
            or getattr(state, "evidence_pack_ref", None)
        )
        if ep_summary:
            lines.append("│ Evidence Pack:")
            lines.append(f"│   ref:      {ep_ref or '(none)'}")
            lines.append(f"│   summary:  {ep_summary}")

    lines.append("│")
    lines.append("│ /objective persist <objective_id>  to save this")
    lines.append("│ /objective cancel                  to discard")
    lines.append(f"╰{sep_bot}╯")
    return "\n".join(lines)
