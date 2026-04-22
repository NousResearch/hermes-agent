from __future__ import annotations

from market_monitor.db import Database


def get_metric_series(db: Database, *, metric_name: str, metric_scope: str, energy_type: str | None = None) -> list[dict]:
    sql = """
        SELECT period_label, value_numeric, unit, source_id, published_at
        FROM observations
        WHERE metric_name = ? AND metric_scope = ? AND is_latest = 1
    """
    params: list = [metric_name, metric_scope]
    if energy_type is not None:
        sql += " AND energy_type = ?"
        params.append(energy_type)
    sql += " ORDER BY period_label ASC"
    return [dict(row) for row in db.query(sql, params)]


def get_latest_market_snapshot(db: Database, *, metric_name: str, metric_scope: str, energy_type: str | None = None) -> dict | None:
    sql = """
        SELECT period_label, value_numeric, unit, source_id, published_at
        FROM observations
        WHERE metric_name = ? AND metric_scope = ? AND is_latest = 1
    """
    params: list = [metric_name, metric_scope]
    if energy_type is not None:
        sql += " AND energy_type = ?"
        params.append(energy_type)
    sql += " ORDER BY period_label DESC LIMIT 1"
    rows = db.query(sql, params)
    return dict(rows[0]) if rows else None


def get_brand_ranking(db: Database, *, period_label: str, top_n: int = 20) -> list[dict]:
    sql = """
        SELECT o.period_label, o.ranking, o.value_numeric, e.name_norm AS brand_name
        FROM observations o
        JOIN observation_entities oe ON oe.obs_id = o.obs_id AND oe.entity_role = 'brand'
        JOIN entities e ON e.entity_id = oe.entity_id
        WHERE o.period_label = ? AND o.metric_type = 'ranking' AND o.is_latest = 1
        ORDER BY o.ranking ASC
        LIMIT ?
    """
    return [dict(row) for row in db.query(sql, (period_label, top_n))]
