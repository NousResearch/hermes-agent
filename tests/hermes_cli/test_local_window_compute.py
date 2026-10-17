"""Compute buffers that grow with the window are priced at the launch posture.

Qwen4Exp's QSA attention scores the whole window each microbatch. Under the router's unified KV
(``--parallel`` auto: every slot sees the full window) llama.cpp b11370 reserved ~46 bytes per
microbatch token per window token for each context, device plus host, and an MTP head is a second
context. At 256K that is 5.9 GiB per context at ``-ub 512`` and ~23 GiB at ``-ub 2048``; priced as
logits alone, the prefill posture loaded the target and then ran out of memory creating the MTP
context on a 128 GB DGX Spark.
"""

from __future__ import annotations

from hermes_cli.local_runtime.catalog import catalog_by_id
from hermes_cli.local_runtime.context_policy import launch_args, posture_profile
from hermes_cli.local_runtime.estimator import HardwareBudget, footprint_bytes

GIB = 1 << 30
MIB = 1 << 20
# A 128 GB GB10 (DGX Spark) as the probe budgets it: the CUDA pool, 20% headroom, lazy reads.
SPARK = HardwareBudget(usable_vram_bytes=int(121.7 * GIB * 0.8), total_device_bytes=int(121.7 * GIB),
                       ram_available_bytes=0, uma=True)
# What b11370 allocated for Flash Next UD-IQ4_XS + ggml-org's Q8_0 head + the BF16 projector at
# -c 262144, -ub 512, unified KV, q8_0 cache, --lazy-mode on (MiB, from its own buffer logs).
MEASURED_256K_LEAN = (61222.13 + 644.14              # target weights, device + host
                      + 3291.18 + 644.14             # MTP head weights
                      + 865.5 + 248.1                # projector + its compute buffer
                      + 3264 + 816 + 512 + 128       # target and head KV
                      + 5111.55 + 798.45             # target compute, device + host
                      + 5053.94 + 808.15) * MIB      # head compute, device + host


def test_flash_next_on_a_128_gb_spark_runs_256k_at_the_lean_microbatch_and_is_priced_as_measured():
    entry = catalog_by_id()["qwen3.8-flash-next"]
    variant = entry.variants[0]

    plan = entry.launch_plan(variant, SPARK)
    args = launch_args(plan.profile, plan.decision, mtp_capable=True, uma=True,
                       mtp_prefill=plan.mtp_prefill)
    priced = footprint_bytes(plan.profile, plan.decision.window, overhead_bytes=plan.overhead_bytes)

    assert plan.decision.window == 262144 and not plan.decision.spilled
    assert "-ub" not in args
    assert MEASURED_256K_LEAN <= priced <= MEASURED_256K_LEAN * 1.1


def test_the_prefill_microbatch_pays_four_times_the_window_compute_of_the_lean_one():
    entry = catalog_by_id()["qwen3.8-flash-next"]
    profile = entry.profile(entry.variants[0])

    lean = posture_profile(profile, mtp_capable=True, mtp_prefill=False)
    stacked = posture_profile(profile, mtp_capable=True, mtp_prefill=True)

    assert stacked.window_compute_per_token == 4 * lean.window_compute_per_token
    assert lean.window_compute_per_token == 46 * 512 * 2
