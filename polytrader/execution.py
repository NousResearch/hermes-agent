from __future__ import annotations

from importlib import import_module
from typing import Any

from .models import ExecutionReceipt, MarketMetadata, TradeDecision


def _load_clob_v2_types():
    # py-clob-client-v2 exposes its public order types from the package root.
    # Keep OrderArgsV2 as a compatibility alias for older docs/skills, but use
    # OrderArgs when that is what the installed SDK exports.
    clob_v2 = import_module("py_clob_client_v2")
    order_args = getattr(clob_v2, "OrderArgsV2", None) or getattr(clob_v2, "OrderArgs")
    return order_args, clob_v2.OrderType, clob_v2.PartialCreateOrderOptions, clob_v2.Side


def _order_type_value(order_type_cls: Any, order_type: str) -> Any:
    return getattr(order_type_cls, order_type.upper(), order_type)


def _side_value(side_cls: Any, side: str) -> Any:
    return getattr(side_cls, side.upper(), side.upper())


class ClobV2ExecutionClient:
    def __init__(self, *, client: Any, dry_run: bool, order_type: str = "GTC", post_only: bool = False) -> None:
        self.client = client
        self.dry_run = dry_run
        self.order_type = order_type
        self.post_only = post_only

    def place_order(self, decision: TradeDecision, market: MarketMetadata) -> ExecutionReceipt:
        if decision.action != "BUY":
            return ExecutionReceipt("skipped", self.dry_run, {"reason": decision.reason})
        if decision.token_id is None or decision.price is None or decision.collateral_size <= 0:
            return ExecutionReceipt("rejected", self.dry_run, {"reason": "incomplete buy decision"})

        payload = {
            "token_id": decision.token_id,
            "side": decision.side or "BUY",
            "price": decision.price,
            "size": decision.collateral_size,
            "tick_size": market.tick_size,
            "neg_risk": market.neg_risk,
            "strategy": decision.strategy,
        }
        if self.dry_run:
            return ExecutionReceipt("dry_run", True, payload)

        OrderArgsV2, OrderType, PartialCreateOrderOptions, Side = _load_clob_v2_types()
        order_args = OrderArgsV2(
            token_id=decision.token_id,
            price=decision.price,
            size=decision.collateral_size,
            side=_side_value(Side, decision.side or "BUY"),
        )
        options = PartialCreateOrderOptions(tick_size=f"{market.tick_size:g}", neg_risk=market.neg_risk)
        response = self.client.create_and_post_order(
            order_args=order_args,
            options=options,
            order_type=_order_type_value(OrderType, self.order_type),
            post_only=self.post_only,
        )
        return ExecutionReceipt("posted", False, response if isinstance(response, dict) else {"response": response})


def build_clob_v2_client(settings: Any) -> Any:
    settings.validate_for_live()
    try:
        clob_v2 = import_module("py_clob_client_v2")
    except ImportError as exc:
        raise RuntimeError("py_clob_client_v2 is required for live Polymarket CLOB v2 execution") from exc

    creds = None
    if settings.clob_api_key and settings.clob_api_secret and settings.clob_api_passphrase:
        creds = clob_v2.ApiCreds(
            api_key=settings.clob_api_key,
            api_secret=settings.clob_api_secret,
            api_passphrase=settings.clob_api_passphrase,
        )
    return clob_v2.ClobClient(
        host=settings.clob_host,
        chain_id=settings.chain_id,
        key=settings.private_key,
        creds=creds,
        signature_type=settings.signature_type if settings.private_key else None,
        funder=settings.funder_address,
    )
