from __future__ import annotations

from market_monitor.db import Database
from market_monitor.queries import get_brand_ranking, get_latest_market_snapshot


def render_monthly_summary(db: Database, *, period_label: str) -> str:
    market_rows = db.query(
        """
        SELECT value_numeric, unit, source_id
        FROM observations
        WHERE period_label = ? AND metric_name = 'sales_volume' AND metric_scope = 'retail' AND metric_type = 'absolute'
        ORDER BY published_at DESC
        LIMIT 1
        """,
        (period_label,),
    )
    market = dict(market_rows[0]) if market_rows else None
    brand_rows = get_brand_ranking(db, period_label=period_label, top_n=5)

    lines = [f"China EV Market Summary {period_label}"]
    if market:
        lines.append(f"- Retail sales: {int(market['value_numeric'])} {market['unit']} ({market['source_id']})")
    else:
        lines.append("- Retail sales: missing")

    if brand_rows:
        lines.append("- Top brands:")
        for row in brand_rows:
            lines.append(f"  {row['ranking']}. {row['brand_name']} — {int(row['value_numeric'])}")
    else:
        lines.append("- Top brands: missing")

    latest_charging = get_latest_market_snapshot(
        db,
        metric_name="charging_piles_total",
        metric_scope="charging_infrastructure",
    )
    if latest_charging and latest_charging["period_label"] == period_label:
        lines.append(f"- Charging piles total: {int(latest_charging['value_numeric'])}")

    return "\n".join(lines)
