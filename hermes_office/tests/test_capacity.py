"""Capacity model tests with hand-computed locked values.

The numbers below come from the math in `design.md` §4.1 — re-derived in
`tests/test_capacity.py` doctring blocks before each test so a reviewer can
verify without leaving this file.

If you change ``DEFAULTS`` in ``capacity.py`` you must update these numbers.
"""
from __future__ import annotations

import math

import pytest

from hermes_office.capacity import (
    DEFAULTS,
    MODEL_TABLE,
    compute,
    detect_host,
    model_profile_for,
)
from hermes_office.models import GPU, HostProfile, ModelProfile


# ────────────────────────────────────────────────────────────────────────────
# Reference 1 — Gemma-4 e2b on a 16 GB laptop, no GPU, 5 employees
#
# weight_bytes  = 2.0e9 * 4/8 = 1.0e9          → 0.931 GB
# kv_bytes      = 2048 * 65536 = 134_217_728   → 0.125 GB
# overhead      = 0.10 * 1e9 = 1.0e8           → 0.093 GB
# per_session   = 0.125 + 0.093                = 0.218 GB
# budget        = 16 * 0.7 − 1.5               = 9.7 GB
# free          = 9.7 − 0.931                  = 8.769 GB
# local_capac   = floor(8.769 / 0.218)         = 40
# recommended   = min(40, 5, 16)               = 5
# headroom      = 8.769 − 5 * 0.218            = 7.679 GB
# p50_ms        = 600 / 50 * 1000              = 12000
# p95_ms        = round(12000 * 1.35)          = 16200
# usd/h         = 0  (local model)
# ────────────────────────────────────────────────────────────────────────────


def test_reference_1_gemma_e2b_16gb_no_gpu_5_emp():
    host = HostProfile(cores=8, ram_gb=16.0, gpus=[], os="linux")
    model = model_profile_for("gemma4-e2b-hermes")
    report = compute(host, model, 5)
    assert report.recommended_concurrency == 5
    assert report.expected_p95_latency_ms == 16_200
    assert report.est_usd_per_hour == 0.0
    assert report.memory_headroom_gb == 7.679
    assert report.employee_count == 5
    assert any("Local model" in n for n in report.notes)


# ────────────────────────────────────────────────────────────────────────────
# Reference 2 — GPT-4o on the same laptop, 4 employees (API model)
#
# rate_limit_rpm        = 500
# tool_round_trips      = 4
# tasks_per_min         = 500 / 4              = 125
# secs_per_task         = 600 * 8 / 1000 * 4   = 19.2
# api_capacity          = floor(125/60 * 19.2) = floor(40.0) = 40
# recommended           = min(40, 4, 16)       = 4
# p50_ms                = 600 * 8              = 4800
# p95_ms                = round(4800 * 1.35)   = 6480
# per_task_cost (USD)   = 1500*2.5/1e6 + 600*10/1e6  = 0.00375 + 0.006 = 0.00975
# usd/h                 = 4 * 6 * 0.00975      = 0.234
# ────────────────────────────────────────────────────────────────────────────


def test_reference_2_gpt4o_4_emp_cost_and_concurrency():
    host = HostProfile(cores=8, ram_gb=16.0, gpus=[], os="linux")
    model = model_profile_for("openai/gpt-4o")
    report = compute(host, model, 4)
    assert report.recommended_concurrency == 4
    assert report.expected_p95_latency_ms == 6_480
    assert report.est_usd_per_hour == pytest.approx(0.234, abs=1e-3)
    assert report.memory_headroom_gb == 16.0  # API → host RAM untouched


# ────────────────────────────────────────────────────────────────────────────
# Reference 3 — Llama3 70B on 32 GB host, NO GPU → must refuse (capacity 0)
#
# weight_gb     = 70e9 * 0.5 / 1024^3          = 32.596
# kv_bytes      = 1280 * 8192                   = 10_485_760  → 0.010 GB
# overhead      = 0.10 * 35e9                   = 3.5e9       → 3.260 GB
# per_session   = 0.010 + 3.260                 = 3.270 GB
# budget        = 32 * 0.7 − 1.5                = 20.9 GB
# free          = max(0, 20.9 − 32.596)         = 0
# local_capac   = floor(0 / 3.270)              = 0
# recommended   = min(0, 1, 16)                 = 0
# ────────────────────────────────────────────────────────────────────────────


def test_reference_3_llama70b_no_gpu_refused():
    host = HostProfile(cores=16, ram_gb=32.0, gpus=[], os="linux")
    model = model_profile_for("llama3:70b")
    report = compute(host, model, 1)
    assert report.recommended_concurrency == 0
    # latency math still computed:
    assert report.expected_p95_latency_ms == int(round(600 / 4.0 * 1000 * 1.35))
    assert report.est_usd_per_hour == 0.0
    assert any("Capacity 0" in n for n in report.notes)


# ────────────────────────────────────────────────────────────────────────────
# Reference 4 — Llama3 70B with an A100-80GB, 4 employees
#
# weights & per_session same as Ref-3
# budget        = 80 * 0.7 − 1.5                = 54.5 GB
# free          = 54.5 − 32.596                 = 21.904 GB
# local_capac   = floor(21.904 / 3.270)         = 6
# recommended   = min(6, 4, 16)                 = 4
# headroom      = 21.904 − 4 * 3.270            = 8.824 GB
# ────────────────────────────────────────────────────────────────────────────


def test_reference_4_llama70b_with_gpu():
    host = HostProfile(
        cores=16, ram_gb=128.0,
        gpus=[GPU(name="A100-80GB", vram_gb=80.0)], os="linux",
    )
    model = model_profile_for("llama3:70b")
    report = compute(host, model, 4)
    assert report.recommended_concurrency == 4
    assert report.memory_headroom_gb == 8.824


# ────────────────────────────────────────────────────────────────────────────
# Reference 5 — Same Gemma e2b laptop with 50 employees → cap at 16
# ────────────────────────────────────────────────────────────────────────────


def test_reference_5_caps_at_max_concurrency():
    host = HostProfile(cores=8, ram_gb=16.0, gpus=[], os="linux")
    model = model_profile_for("gemma4-e2b-hermes")
    report = compute(host, model, 50)
    assert report.recommended_concurrency == DEFAULTS["max_concurrency_cap"]
    assert any("queue" in n for n in report.notes)


# ────────────────────────────────────────────────────────────────────────────
# Reference 6 — Empty roster: capacity should be 0 employee_count, no NaN.
# ────────────────────────────────────────────────────────────────────────────


def test_reference_6_empty_roster_safe():
    host = HostProfile(cores=2, ram_gb=4.0, gpus=[], os="linux")
    model = model_profile_for("gemma4-e2b-hermes")
    report = compute(host, model, 0)
    assert report.employee_count == 0
    assert report.recommended_concurrency >= 0
    assert math.isfinite(report.est_usd_per_hour)


# ────────────────────────────────────────────────────────────────────────────
# Determinism — same inputs ⇒ same outputs (3-decimal precision contract).
# ────────────────────────────────────────────────────────────────────────────


def test_determinism():
    host = HostProfile(cores=8, ram_gb=16.0, gpus=[], os="linux")
    model = model_profile_for("gemma4-e2b-hermes")
    a = compute(host, model, 5)
    b = compute(host, model, 5)
    assert a == b


def test_unknown_model_falls_back_to_generic():
    profile = model_profile_for("does-not-exist:weird")
    assert profile.params_b == 7.0
    assert profile.provider_kind == "local"


def test_detect_host_returns_sensible_defaults():
    host = detect_host()
    assert host.cores >= 1
    assert host.ram_gb > 0
