#!/usr/bin/env python3
"""
Daily MoA Orchestrator Rotation + Free Reference Model Discovery (2x/day).

SAKANA-AI Fugu variant: rotates THREE strong orchestrators (gpt-5.6-luna,
gemini-3.1-flash, grok-4.5) in round-robin, AND refreshes the free reference
panel daily with live discovery + liveness probe.

Cron: 0 4,16 * * * (4am and 4pm JST)
"""

from __future__ import annotations

import random
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = Path.home() / ".hermes" / "config.yaml"

# Three strong orchestrators - round-robin daily
# Note: gemini-3.1-flash-preview not in catalog; use gemini-3.1-flash-lite (available)
ORCHESTRATORS = [
    {"provider": "openai-codex", "model": "gpt-5.6-luna"},
    {"provider": "gemini", "model": "gemini-3.1-flash-lite"},  # Google AI Studio (available)
    {"provider": "xai", "model": "grok-4.5"},
]

# Free reference model candidates - live discovery each run
CANDIDATE_ALIASES = [
    ("opencode-zen", "auto-free"),
    ("nvidia", "auto"),
    ("nous", "auto-free"),
    ("freellmapi", "auto"),
    ("freebuff", "deepseek/deepseek-v4-flash"),
]

# Static fallback if live discovery fails
STATIC_FALLBACK_REFS = [
    {"provider": "opencode-zen", "model": "big-pickle"},
    {"provider": "nvidia", "model": "nvidia/nemotron-3-ultra-550b-a55b"},
    {"provider": "nous", "model": "nvidia/nemotron-3-ultra-550b-a55b:free"},
    {"provider": "freellmapi", "model": "auto"},
    {"provider": "freebuff", "model": "deepseek/deepseek-v4-flash"},
]


def resolve_real_id(provider: str, alias: str) -> str | None:
    try:
        from hermes_cli import models as model_catalog
        return model_catalog.resolve_config_model_id(provider, alias, force_refresh=True)
    except Exception:
        return None


def liveness(provider: str, model: str) -> bool:
    try:
        from agent.auxiliary_client import call_llm
        call_llm(
            provider=provider,
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
        return True
    except Exception:
        return False


def _write(data: dict) -> None:
    import yaml

    class _D(yaml.SafeDumper):
        pass

    def _str_rep(d, s):
        if "\n" in s:
            return d.represent_scalar("tag:yaml.org,2002:str", s, style=">")
        return d.represent_scalar("tag:yaml.org,2002:str", s)

    _D.add_representer(str, _str_rep)
    CONFIG_PATH.write_text(
        yaml.dump(data, Dumper=_D, default_flow_style=False, sort_keys=False,
                  allow_unicode=True, width=4096),
        encoding="utf-8",
    )
    print(f"[ok] wrote {CONFIG_PATH}")


def get_current_preset(data: dict) -> tuple[dict, str]:
    moa = data.get("moa") or {}
    presets = moa.get("presets") or {}
    active = moa.get("active_preset") or moa.get("default_preset")
    if not active or active not in presets:
        raise ValueError("no active MoA preset to rotate")
    return presets[active], active


def discover_free_references() -> list[dict]:
    """Discover live free reference models with liveness probe."""
    alive = []
    print(f"[info] {date.today()} live free-model discovery:")
    for provider, alias in CANDIDATE_ALIASES:
        real = resolve_real_id(provider, alias)
        if not real:
            print(f"  [skip] {provider}:{alias} -> unresolvable")
            continue
        ok = liveness(provider, real)
        print(f"  [{'alive' if ok else 'dead '}] {provider}:{alias} -> {real}")
        if ok:
            alive.append({"provider": provider, "model": real})

    if not alive:
        print("[warn] no live free models; using static fallback")
        return STATIC_FALLBACK_REFS
    return alive


def select_orchestrator_today() -> dict:
    """Round-robin select orchestrator based on date."""
    day_index = int(date.today().strftime("%Y%m%d")) % len(ORCHESTRATORS)
    return ORCHESTRATORS[day_index]


def round_robin_shuffle(models: list[dict]) -> list[dict]:
    """Shuffle reference models with date-based seed for daily rotation."""
    rng = random.Random(int(date.today().strftime("%Y%m%d")))
    shuffled = models.copy()
    rng.shuffle(shuffled)
    return shuffled


def main() -> int:
    import yaml

    if not CONFIG_PATH.exists():
        print(f"[error] config not found: {CONFIG_PATH}")
        return 1

    data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    moa = data.get("moa") or {}
    presets = moa.get("presets") or {}

    # Ensure we have the main fugu preset
    preset_name = "hakuapulse-orchestrator"
    if preset_name not in presets:
        print(f"[error] preset '{preset_name}' not found in config")
        return 1

    preset = presets[preset_name]

    # 1. Select today's orchestrator (round-robin among 3 strong models)
    orchestrator = select_orchestrator_today()
    preset["aggregator"] = dict(orchestrator)
    print(f"[orchestrator] {date.today()} -> {orchestrator['provider']}:{orchestrator['model']}")

    # 2. Discover and rotate free reference models
    live_refs = discover_free_references()
    rotated_refs = round_robin_shuffle(live_refs)
    preset["reference_models"] = rotated_refs
    print(f"[ok] rotated {len(rotated_refs)} free reference models (round-robin):")
    for r in rotated_refs:
        print(f"      - {r['provider']}:{r['model']}")

    # 3. Ensure model.provider=moa and model.default=preset_name
    data.setdefault("model", {})["provider"] = "moa"
    data["model"]["default"] = preset_name
    moa["default_preset"] = preset_name
    moa["active_preset"] = preset_name

    _write(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())