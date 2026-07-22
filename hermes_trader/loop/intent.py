"""Parse and validate structured TradeIntent JSON from LLM output."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from hermes_trader.risk.gate import TradeIntent

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_JSON_OBJECT_RE = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)


def _schema_path() -> Path:
    return Path(__file__).resolve().parent / "intent_schema.json"


def hold_intent(chain: str, *, reasoning: str = "No actionable opportunity") -> TradeIntent:
    return TradeIntent(
        action="hold",
        chain=chain.strip().lower(),
        token_address="",
        size_usd=0.0,
        confidence=1.0,
        reasoning=reasoning,
    )


def _loads_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("empty TradeIntent payload")
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        block = _JSON_BLOCK_RE.search(stripped)
        if block:
            data = json.loads(block.group(1))
        else:
            match = _JSON_OBJECT_RE.search(stripped)
            if not match:
                raise ValueError("no JSON object found in TradeIntent payload")
            data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("TradeIntent must be a JSON object")
    return data


def parse_trade_intent(payload: str | dict[str, Any]) -> TradeIntent:
    """Parse TradeIntent from raw JSON, markdown fence, or mapping."""
    if isinstance(payload, dict):
        data = payload
    else:
        data = _loads_object(str(payload))
    return TradeIntent.from_mapping(data)


def validate_trade_intent(intent: TradeIntent | dict[str, Any]) -> None:
    """Validate against intent_schema.json when jsonschema is available."""
    try:
        import jsonschema  # type: ignore[import-untyped]
    except ImportError:
        return

    schema = json.loads(_schema_path().read_text(encoding="utf-8"))
    if isinstance(intent, TradeIntent):
        data = {
            "action": intent.action,
            "chain": intent.chain,
            "token_address": intent.token_address,
            "size_usd": intent.size_usd,
            "confidence": intent.confidence,
            "reasoning": intent.reasoning,
            "stop_loss_pct": intent.stop_loss_pct,
            "take_profit_pct": intent.take_profit_pct,
            "strategy_tag": intent.strategy_tag,
            "pool_liquidity_usd": intent.pool_liquidity_usd,
            "slippage_bps": intent.slippage_bps,
        }
    else:
        data = dict(intent)
    data = {k: v for k, v in data.items() if v is not None}
    jsonschema.validate(data, schema)