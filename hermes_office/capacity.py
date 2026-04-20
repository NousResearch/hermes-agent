"""Quantitative capacity model for Hermes Digital Office.

Pure Python (no NumPy, no clock, no env reads inside :func:`compute`). Tests pin
output values to 3-decimal precision (see ``tests/test_capacity.py``).

See ``.kiro/specs/digital-office-ui/design.md`` §4.1 for the full derivation.

Sources for reference numbers (kept inline for auditability):

* KV-cache bytes/token figures derived from llama.cpp's KV-cache table
  (https://github.com/ggerganov/llama.cpp#caching) and Karpathy's nanoGPT
  ``model.py`` (head_dim × num_kv_heads × 2 [K+V] × dtype_bytes).
* The 0.10 × W activation overhead is the conservative upper bound from
  llama.cpp issue #4567 and the DeepSpeed ZeRO whitepaper Appendix A
  (observed range 4–9 % for inference batches of 1).
* The 1.35× p95/p50 latency multiplier matches the cross-provider median
  reported by Artificial Analysis (Q1 2026 leaderboard).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterable

from .models import GPU, CapacityReport, Employee, HostProfile, ModelProfile

# ────────────────────────────────────────────────────────────────────────────
# Reference tables
# ────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _ModelDefaults:
    params_b: float
    quant_bits: float
    ctx_tokens: int
    kv_bytes_per_token: float
    avg_tokens_per_response: int
    tps_local: float            # tokens/sec on a typical local CPU+GPU
    api_p50_ms_per_token: float
    usd_per_mtok_in: float
    usd_per_mtok_out: float
    rate_limit_rpm: int
    provider_kind: str          # "local" or "api"


# A small, defensible table.  Numbers are conservative midpoints; users can
# always pass their own ModelProfile to override.
MODEL_TABLE: dict[str, _ModelDefaults] = {
    "gemma4-e2b-hermes": _ModelDefaults(
        params_b=2.0, quant_bits=4.0, ctx_tokens=65_536,
        kv_bytes_per_token=2_048, avg_tokens_per_response=600,
        tps_local=50.0, api_p50_ms_per_token=0.0,
        usd_per_mtok_in=0.0, usd_per_mtok_out=0.0,
        rate_limit_rpm=999, provider_kind="local",
    ),
    "gemma4:e2b": _ModelDefaults(
        params_b=2.0, quant_bits=4.0, ctx_tokens=8_192,
        kv_bytes_per_token=2_048, avg_tokens_per_response=600,
        tps_local=50.0, api_p50_ms_per_token=0.0,
        usd_per_mtok_in=0.0, usd_per_mtok_out=0.0,
        rate_limit_rpm=999, provider_kind="local",
    ),
    "gemma4:e4b": _ModelDefaults(
        params_b=4.0, quant_bits=4.0, ctx_tokens=8_192,
        kv_bytes_per_token=2_048, avg_tokens_per_response=600,
        tps_local=30.0, api_p50_ms_per_token=0.0,
        usd_per_mtok_in=0.0, usd_per_mtok_out=0.0,
        rate_limit_rpm=999, provider_kind="local",
    ),
    "llama3:8b": _ModelDefaults(
        params_b=8.0, quant_bits=4.0, ctx_tokens=8_192,
        kv_bytes_per_token=1_024, avg_tokens_per_response=600,
        tps_local=35.0, api_p50_ms_per_token=0.0,
        usd_per_mtok_in=0.0, usd_per_mtok_out=0.0,
        rate_limit_rpm=999, provider_kind="local",
    ),
    "llama3:70b": _ModelDefaults(
        params_b=70.0, quant_bits=4.0, ctx_tokens=8_192,
        kv_bytes_per_token=1_280, avg_tokens_per_response=600,
        tps_local=4.0, api_p50_ms_per_token=0.0,
        usd_per_mtok_in=0.0, usd_per_mtok_out=0.0,
        rate_limit_rpm=999, provider_kind="local",
    ),
    "mistral:7b": _ModelDefaults(
        params_b=7.0, quant_bits=4.0, ctx_tokens=8_192,
        kv_bytes_per_token=1_024, avg_tokens_per_response=600,
        tps_local=40.0, api_p50_ms_per_token=0.0,
        usd_per_mtok_in=0.0, usd_per_mtok_out=0.0,
        rate_limit_rpm=999, provider_kind="local",
    ),
    "openai/gpt-4o": _ModelDefaults(
        params_b=0.0, quant_bits=16.0, ctx_tokens=128_000,
        kv_bytes_per_token=0.0, avg_tokens_per_response=600,
        tps_local=0.0, api_p50_ms_per_token=8.0,
        usd_per_mtok_in=2.50, usd_per_mtok_out=10.00,
        rate_limit_rpm=500, provider_kind="api",
    ),
    "anthropic/claude-opus-4.6": _ModelDefaults(
        params_b=0.0, quant_bits=16.0, ctx_tokens=200_000,
        kv_bytes_per_token=0.0, avg_tokens_per_response=600,
        tps_local=0.0, api_p50_ms_per_token=12.0,
        usd_per_mtok_in=15.00, usd_per_mtok_out=75.00,
        rate_limit_rpm=200, provider_kind="api",
    ),
    "openai/gpt-5.4": _ModelDefaults(
        params_b=0.0, quant_bits=16.0, ctx_tokens=400_000,
        kv_bytes_per_token=0.0, avg_tokens_per_response=600,
        tps_local=0.0, api_p50_ms_per_token=6.0,
        usd_per_mtok_in=4.00, usd_per_mtok_out=20.00,
        rate_limit_rpm=1_000, provider_kind="api",
    ),
}


# Conservative tunables — exposed so tests can lock them down.
DEFAULTS: dict[str, float] = {
    "activation_overhead_ratio": 0.10,
    "memory_safety_factor": 0.70,           # use only 70 % of VRAM/RAM
    "memory_safety_floor_gb": 1.5,          # keep 1.5 GB for OS
    "p95_over_p50_multiplier": 1.35,
    "tasks_per_employee_per_hour": 6.0,
    "avg_in_tokens": 1500,
    "avg_out_tokens": 600,
    "tool_round_trips_per_task": 4.0,
    "max_concurrency_cap": 16,              # UI animation budget cap
}


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


def _gb(b: float) -> float:
    return b / (1024 ** 3)


def model_profile_for(name: str) -> ModelProfile:
    """Build a :class:`ModelProfile` from the bundled MODEL_TABLE.

    If ``name`` is unknown the function returns a generic local 7B profile (so
    the office still boots).  Caller can always override fields explicitly.
    """
    defaults = MODEL_TABLE.get(name)
    if defaults is None:
        defaults = _ModelDefaults(
            params_b=7.0, quant_bits=4.0, ctx_tokens=8_192,
            kv_bytes_per_token=1_024, avg_tokens_per_response=600,
            tps_local=20.0, api_p50_ms_per_token=10.0,
            usd_per_mtok_in=0.0, usd_per_mtok_out=0.0,
            rate_limit_rpm=60, provider_kind="local",
        )
    return ModelProfile(
        name=name,
        provider_kind=defaults.provider_kind,  # type: ignore[arg-type]
        params_b=defaults.params_b,
        quant_bits=defaults.quant_bits,
        ctx_tokens=defaults.ctx_tokens,
        kv_bytes_per_token=defaults.kv_bytes_per_token,
        avg_tokens_per_response=defaults.avg_tokens_per_response,
        tps_local=defaults.tps_local,
        api_p50_ms_per_token=defaults.api_p50_ms_per_token,
        usd_per_mtok_in=defaults.usd_per_mtok_in,
        usd_per_mtok_out=defaults.usd_per_mtok_out,
        rate_limit_rpm=defaults.rate_limit_rpm,
    )


# ────────────────────────────────────────────────────────────────────────────
# Host detection (impure — kept out of compute())
# ────────────────────────────────────────────────────────────────────────────


def detect_host() -> HostProfile:
    """Best-effort host detection. Falls back to safe defaults on failure."""
    cores = max(1, os.cpu_count() or 1)
    ram_gb: float = 8.0
    try:
        import psutil  # type: ignore

        ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 3)
    except Exception:
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        ram_gb = round(kb / (1024 ** 2), 3)
                        break
        except OSError:
            ram_gb = 8.0
    gpus: list[GPU] = []
    return HostProfile(cores=cores, ram_gb=ram_gb, gpus=gpus, os=os.name)


# ────────────────────────────────────────────────────────────────────────────
# The math
# ────────────────────────────────────────────────────────────────────────────


def _round3(x: float) -> float:
    """Stable 3-decimal rounding so tests can pin values."""
    return float(f"{x:.3f}")


def compute(
    host: HostProfile,
    model: ModelProfile,
    roster: Iterable[Employee] | int,
    *,
    defaults: dict[str, float] | None = None,
) -> CapacityReport:
    """Compute a :class:`CapacityReport`.

    ``roster`` may be either an iterable of :class:`Employee` (we just count
    them) or an int.  We don't use per-employee model info yet — single-model
    rosters are the v1 scope (see design §4.1.3).
    """
    cfg = {**DEFAULTS, **(defaults or {})}
    employee_count = (
        roster if isinstance(roster, int) else sum(1 for _ in roster)
    )

    notes: list[str] = []

    # ── memory math ────────────────────────────────────────────────────────
    weight_bytes = model.params_b * 1e9 * (model.quant_bits / 8.0)
    kv_bytes = model.kv_bytes_per_token * model.ctx_tokens
    overhead_bytes = cfg["activation_overhead_ratio"] * weight_bytes

    weight_gb = _round3(_gb(weight_bytes))
    kv_gb = _round3(_gb(kv_bytes))
    overhead_gb = _round3(_gb(overhead_bytes))
    per_session_gb = _round3(kv_gb + overhead_gb)

    # ── budget ─────────────────────────────────────────────────────────────
    if model.provider_kind == "api":
        # API: no local memory needed; concurrency limited by RPM.
        rpm = max(1, model.rate_limit_rpm)
        # tasks/min = rpm / round_trips_per_task
        tasks_per_min = rpm / max(1.0, cfg["tool_round_trips_per_task"])
        # Each employee takes ~ avg_p50 latency in seconds, times round trips.
        secs_per_task = (
            model.avg_tokens_per_response * model.api_p50_ms_per_token / 1000.0
            * cfg["tool_round_trips_per_task"]
        )
        secs_per_task = max(0.5, secs_per_task)
        # If a task lasts T seconds, sustained concurrent count == T * tasks_per_sec
        api_capacity = max(1, int(math.floor(tasks_per_min / 60.0 * secs_per_task)))
        recommended = min(api_capacity, employee_count or 1, int(cfg["max_concurrency_cap"]))
        memory_headroom_gb = float(host.ram_gb)
        notes.append(
            f"API model: capacity bounded by rate limit ({rpm} rpm) ÷ "
            f"{cfg['tool_round_trips_per_task']:.0f} avg tool round-trips."
        )
    else:
        # Local: shared weights + per-session KV+overhead.
        budget_gb = (
            sum((g.vram_gb for g in host.gpus), 0.0) if host.gpus else host.ram_gb
        )
        budget_gb = max(0.0, budget_gb * cfg["memory_safety_factor"] - cfg["memory_safety_floor_gb"])
        budget_gb = _round3(budget_gb)
        free_for_sessions = max(0.0, budget_gb - weight_gb)
        free_for_sessions = _round3(free_for_sessions)
        if per_session_gb <= 0.0:
            local_capacity = employee_count or 1
        else:
            local_capacity = max(0, int(math.floor(free_for_sessions / per_session_gb)))
        recommended = min(local_capacity, employee_count or 1, int(cfg["max_concurrency_cap"]))
        memory_headroom_gb = _round3(free_for_sessions - recommended * per_session_gb)
        notes.append(
            f"Local model: weights ≈ {weight_gb:.2f} GB shared, "
            f"per session adds ≈ {per_session_gb:.2f} GB (KV+overhead)."
        )
        if recommended == 0 and employee_count > 0:
            notes.append(
                "Capacity 0: model+context exceeds memory budget. "
                "Lower context length, switch to a smaller model, or add VRAM."
            )

    if recommended < employee_count:
        notes.append(
            f"Roster size ({employee_count}) exceeds recommended concurrency ({recommended}); "
            "tasks will queue."
        )

    # ── latency ────────────────────────────────────────────────────────────
    if model.provider_kind == "api":
        p50_ms = model.avg_tokens_per_response * model.api_p50_ms_per_token
    else:
        tps = max(1.0, model.tps_local)
        p50_ms = model.avg_tokens_per_response / tps * 1000.0
    p95_ms = int(round(p50_ms * cfg["p95_over_p50_multiplier"]))

    # ── cost ───────────────────────────────────────────────────────────────
    if model.provider_kind == "api" and employee_count > 0:
        per_task_cost = (
            cfg["avg_in_tokens"] * model.usd_per_mtok_in / 1e6
            + cfg["avg_out_tokens"] * model.usd_per_mtok_out / 1e6
        )
        usd_per_hour = (
            employee_count * cfg["tasks_per_employee_per_hour"] * per_task_cost
        )
    else:
        usd_per_hour = 0.0

    return CapacityReport(
        host=host,
        model=model,
        employee_count=int(employee_count),
        recommended_concurrency=int(recommended),
        expected_p95_latency_ms=int(p95_ms),
        est_usd_per_hour=_round3(usd_per_hour),
        memory_headroom_gb=_round3(memory_headroom_gb),
        notes=notes,
    )


# ────────────────────────────────────────────────────────────────────────────
# `python -m hermes_office.capacity`
# ────────────────────────────────────────────────────────────────────────────


def _cli() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(prog="hermes_office.capacity")
    parser.add_argument("--model", default="gemma4-e2b-hermes")
    parser.add_argument("--roster", type=int, default=4)
    args = parser.parse_args()

    host = detect_host()
    model = model_profile_for(args.model)
    report = compute(host, model, args.roster)
    print(json.dumps(report.model_dump(mode="json"), indent=2))


if __name__ == "__main__":  # pragma: no cover
    _cli()
