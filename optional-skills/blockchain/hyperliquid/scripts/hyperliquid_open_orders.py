"""Read-only frontend orders with separate, exchange-reported position PnL."""
from __future__ import annotations

from typing import Any


def normalize_open_orders(payload: Any, parent_oid: Any = None) -> list[dict[str, Any]]:
    """Flatten frontend order trees while retaining the parent order ID."""
    if not isinstance(payload, list):
        return []
    rows = []
    for order in payload:
        if not isinstance(order, dict):
            continue
        rows.append({
            "coin": order.get("coin", "-"),
            "oid": order.get("oid"),
            "parent_oid": parent_oid,
            "side": order.get("side", "-"),
            "limit_px": order.get("limitPx"),
            "size": order.get("sz"),
            "orig_size": order.get("origSz"),
            "timestamp": order.get("timestamp"),
            "order_type": order.get("orderType"),
            "is_trigger": order.get("isTrigger"),
            "trigger_px": order.get("triggerPx"),
            "trigger_condition": order.get("triggerCondition"),
            "reduce_only": order.get("reduceOnly"),
            "is_position_tpsl": order.get("isPositionTpsl"),
            "tif": order.get("tif"),
        })
        rows.extend(normalize_open_orders(order.get("children"), order.get("oid")))
    return rows


def run_open_orders(args: Any, client: Any) -> dict[str, Any]:
    user = client._resolve_user(args.user)
    scope = {"user": user, "dex": args.dex}
    orders = normalize_open_orders(client._post_info({"type": "frontendOpenOrders", **scope}))
    state = client._normalize_positions(client._post_info({"type": "clearinghouseState", **scope}))
    return {**scope, "count": len(orders), "orders": client._limit_items(orders, args.limit), **state}


def render_open_orders(data: dict[str, Any], client: Any) -> str:
    rows = []
    for order in data["orders"]:
        row = {key: "-" if value is None else value for key, value in order.items()}
        row["limit_px"] = client._format_price(order["limit_px"])
        row["trigger_px"] = client._format_price(order["trigger_px"])
        rows.append(row)
    headers = [
        ("Coin", "coin"), ("OID", "oid"), ("Parent", "parent_oid"),
        ("Side", "side"), ("Size", "size"), ("Limit", "limit_px"),
        ("Order Type", "order_type"), ("Trigger", "trigger_px"),
        ("Trigger Condition", "trigger_condition"), ("Reduce Only", "reduce_only"),
        ("Position TP/SL", "is_position_tpsl"),
    ]
    return "\n".join([
        "Current positions (uPnL from clearinghouseState, not per-order PnL)",
        client.render_state(data), "",
        f"Open orders returned: {len(rows)} of {data['count']} (including children)",
        client._render_table(headers, rows), "",
        "Unfilled orders and TP/SL trigger prices are not realized or projected PnL.",
    ])
