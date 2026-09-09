"""Validated configuration for the optional native plugin side-run capability."""
from dataclasses import asdict, dataclass
import math
import re
from typing import Any


def bounded_number(value, name, low, high, *, integer=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not low <= value <= high or not math.isfinite(value)
            or (integer and not isinstance(value, int))):
        raise ValueError(f"{name} must be {'an integer' if integer else 'a finite number'} between {low} and {high}")
    return value


@dataclass(frozen=True)
class SideRunConfig:
    provider: str
    model: str
    tools: tuple[str, ...] | None = ()
    reasoning: dict | None = None
    max_iterations: int = 20
    max_tokens: int = 4096
    run_budget_seconds: float = 300

    @classmethod
    def from_mapping(cls, value: Any):
        if not isinstance(value, dict) or set(value) - set(cls.__dataclass_fields__):
            raise ValueError("Side-run configuration contains unsupported fields")
        data = dict(value)
        for name in ("provider", "model"):
            v = data.get(name)
            if (not isinstance(v, str) or not v or v != v.strip()
                    or len(v) > 256 or any(ord(c) < 32 for c in v)):
                raise ValueError(f"{name} must be an explicit nonempty identifier without surrounding whitespace")
            data[name] = v
        if data["provider"] in {"auto", "default"} or not re.fullmatch(r"[a-z0-9][a-z0-9_.:@-]*", data["provider"]):
            raise ValueError("provider must be explicit")
        tools = data.get("tools", [])
        if tools is not None:
            if not isinstance(tools, (list, tuple)) or len(tools) > 64 or any(
                not isinstance(t, str) or not re.fullmatch(r"[a-zA-Z0-9_.:-]+", t) for t in tools
            ):
                raise ValueError("tools must be null or a list of toolset names")
            data["tools"] = tuple(dict.fromkeys(tools))
        reasoning = data.get("reasoning")
        if reasoning is not None:
            if not isinstance(reasoning, dict) or set(reasoning) - {"enabled", "effort", "max_tokens"}:
                raise ValueError("reasoning supports enabled, effort and max_tokens")
            if "enabled" in reasoning and type(reasoning["enabled"]) is not bool:
                raise ValueError("reasoning.enabled must be boolean")
            if "effort" in reasoning and (
                not isinstance(reasoning["effort"], str)
                or reasoning["effort"] not in {"none", "minimal", "low", "medium", "high", "xhigh"}
            ):
                raise ValueError("Unsupported reasoning effort")
            if "max_tokens" in reasoning:
                bounded_number(reasoning["max_tokens"], "reasoning.max_tokens", 1, 131072, integer=True)
            data["reasoning"] = dict(reasoning)
        for name, low, high, integer in (
            ("max_iterations", 1, 500, True), ("max_tokens", 1, 131072, True),
            ("run_budget_seconds", 1, 3600, False),
        ):
            if name in data:
                bounded_number(data[name], name, low, high, integer=integer)
        return cls(**data)

    def to_dict(self):
        data = asdict(self)
        if self.tools is not None:
            data["tools"] = list(self.tools)
        return data
