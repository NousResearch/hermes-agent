"""Settings: defaults, config.yaml section, env overrides."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping

DEFAULTS: dict[str, Any] = {
    # report = measure only (default) | enforce = actually clip
    "mode": "report",
    # fallback cap per tool result, in tokens
    "default_cap": 1200,
    # nothing is ever clipped above this, even if a tool asks for more
    "hard_cap": 6000,
    # head share of the kept window; the tail keeps the rest (errors live at the edges)
    "head_ratio": 0.6,
    # characters per token, used for the estimate (Arabic is denser than English)
    "chars_per_token": 3.0,
    # per-tool overrides; big-output tools get a smaller cap on purpose
    "per_tool": {
        "web_extract": 900,
        "skill_view": 900,
        "session_search": 900,
        "memory_recall": 900,
        "read_file": 2000,
        "terminal": 1500,
        "execute_code": 1500,
    },
    # where the savings log goes (JSONL, one line per call)
    "stats_path": "~/.hermes/cache-engine/stats.jsonl",
}


@dataclass(frozen=True)
class Settings:
    mode: str = "report"
    default_cap: int = 1200
    hard_cap: int = 6000
    head_ratio: float = 0.6
    chars_per_token: float = 3.0
    per_tool: Mapping[str, int] = field(default_factory=dict)
    stats_path: str = "~/.hermes/cache-engine/stats.jsonl"

    def cap_for(self, tool_name: str | None) -> int:
        cap = self.per_tool.get(tool_name or "", self.default_cap)
        return max(64, min(int(cap), self.hard_cap))


def _merge(base: dict[str, Any], section: Mapping[str, Any] | None) -> dict[str, Any]:
    out = dict(base)
    for key, value in (section or {}).items():
        if key == "per_tool" and isinstance(value, Mapping):
            merged = dict(out.get("per_tool") or {})
            merged.update({str(k): int(v) for k, v in value.items()})
            out["per_tool"] = merged
        elif key in out:
            out[key] = value
    return out


def load_settings(section: Mapping[str, Any] | None = None) -> Settings:
    """Read ``cache_engine:`` from config.yaml; env vars win for quick experiments."""
    if section is None:
        try:
            from hermes_cli.config import load_config  # type: ignore

            section = (load_config() or {}).get("cache_engine") or {}
        except Exception:
            section = {}
    merged = _merge(DEFAULTS, section)
    if os.environ.get("HERMES_CACHE_ENGINE_MODE"):
        merged["mode"] = os.environ["HERMES_CACHE_ENGINE_MODE"]
    if os.environ.get("HERMES_CACHE_ENGINE_DEFAULT_CAP"):
        merged["default_cap"] = int(os.environ["HERMES_CACHE_ENGINE_DEFAULT_CAP"])
    return Settings(
        mode=str(merged["mode"]),
        default_cap=int(merged["default_cap"]),
        hard_cap=int(merged["hard_cap"]),
        head_ratio=float(merged["head_ratio"]),
        chars_per_token=float(merged["chars_per_token"]),
        per_tool=dict(merged["per_tool"]),
        stats_path=str(merged["stats_path"]),
    )
