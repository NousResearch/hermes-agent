from __future__ import annotations

from decimal import Decimal
from typing import Dict


MODEL_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-4.5-preview": {"input": 75.00, "output": 150.00},
    "gpt-5": {"input": 10.00, "output": 30.00},
    "gpt-5.4": {"input": 10.00, "output": 30.00},
    "o3": {"input": 10.00, "output": 40.00},
    "o3-mini": {"input": 1.10, "output": 4.40},
    "o4-mini": {"input": 1.10, "output": 4.40},
    "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    "deepseek-chat": {"input": 0.14, "output": 0.28},
    "deepseek-reasoner": {"input": 0.55, "output": 2.19},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "llama-4-maverick": {"input": 0.50, "output": 0.70},
    "llama-4-scout": {"input": 0.20, "output": 0.30},
    "glm-5": {"input": 0.0, "output": 0.0},
    "glm-4.7": {"input": 0.0, "output": 0.0},
    "glm-4.5": {"input": 0.0, "output": 0.0},
    "glm-4.5-flash": {"input": 0.0, "output": 0.0},
    "kimi-k2.5": {"input": 0.0, "output": 0.0},
    "kimi-k2-thinking": {"input": 0.0, "output": 0.0},
    "kimi-k2-turbo-preview": {"input": 0.0, "output": 0.0},
    "kimi-k2-0905-preview": {"input": 0.0, "output": 0.0},
    "MiniMax-M2.5": {"input": 0.0, "output": 0.0},
    "MiniMax-M2.5-highspeed": {"input": 0.0, "output": 0.0},
    "MiniMax-M2.1": {"input": 0.0, "output": 0.0},
}

DEFAULT_PRICING = {"input": 0.0, "output": 0.0}


def get_pricing(model_name: str) -> Dict[str, float]:
    if not model_name:
        return DEFAULT_PRICING

    bare = model_name.split("/")[-1].lower()
    if bare in MODEL_PRICING:
        return MODEL_PRICING[bare]

    best_match = None
    best_len = 0
    for key, price in MODEL_PRICING.items():
        if bare.startswith(key) and len(key) > best_len:
            best_match = price
            best_len = len(key)
    if best_match:
        return best_match

    if "opus" in bare:
        return {"input": 15.00, "output": 75.00}
    if "sonnet" in bare:
        return {"input": 3.00, "output": 15.00}
    if "haiku" in bare:
        return {"input": 0.80, "output": 4.00}
    if "gpt-4o-mini" in bare:
        return {"input": 0.15, "output": 0.60}
    if "gpt-4o" in bare:
        return {"input": 2.50, "output": 10.00}
    if "gpt-5" in bare:
        return {"input": 10.00, "output": 30.00}
    if "deepseek" in bare:
        return {"input": 0.14, "output": 0.28}
    if "gemini" in bare:
        return {"input": 0.15, "output": 0.60}

    return DEFAULT_PRICING


def has_known_pricing(model_name: str) -> bool:
    pricing = get_pricing(model_name)
    return pricing is not DEFAULT_PRICING and any(
        float(value) > 0 for value in pricing.values()
    )


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = get_pricing(model)
    total = (
        Decimal(input_tokens) * Decimal(str(pricing["input"]))
        + Decimal(output_tokens) * Decimal(str(pricing["output"]))
    ) / Decimal("1000000")
    return float(total)


def format_duration_compact(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.0f}m"
    hours = minutes / 60
    if hours < 24:
        remaining_min = int(minutes % 60)
        return f"{int(hours)}h {remaining_min}m" if remaining_min else f"{int(hours)}h"
    days = hours / 24
    return f"{days:.1f}d"


def format_token_count_compact(value: int) -> str:
    abs_value = abs(int(value))
    if abs_value < 1_000:
        return str(int(value))

    sign = "-" if value < 0 else ""
    units = ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K"))
    for threshold, suffix in units:
        if abs_value >= threshold:
            scaled = abs_value / threshold
            if scaled < 10:
                text = f"{scaled:.2f}"
            elif scaled < 100:
                text = f"{scaled:.1f}"
            else:
                text = f"{scaled:.0f}"
            text = text.rstrip("0").rstrip(".")
            return f"{sign}{text}{suffix}"

    return f"{value:,}"
