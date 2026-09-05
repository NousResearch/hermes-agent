"""Behavior contracts for advanced managed-runtime launch planning."""

from __future__ import annotations

from hermes_cli.local_runtime.advanced import LaunchRequest, plan_launch
from hermes_cli.local_runtime.context_policy import WindowDecision, launch_args
from hermes_cli.local_runtime.estimator import HardwareBudget, LayerKind, ModelProfile


def _profile() -> ModelProfile:
    return ModelProfile(
        name="test", weights_bytes=8 << 30, embd_table_bytes=0, n_ctx_train=128 * 1024,
        layers=[(LayerKind.FULL, 1024)], n_vocab=32_000,
    )


def _budget() -> HardwareBudget:
    return HardwareBudget(usable_vram_bytes=24 << 30, total_device_bytes=24 << 30,
                          ram_available_bytes=32 << 30)


def test_slot_plan_charges_weights_once_and_request_state_per_slot():
    one = plan_launch(_profile(), _budget(), LaunchRequest(context_tokens=64 * 1024, slots=1),
                      default_context_tokens=64 * 1024, mtp_supported=False)
    two = plan_launch(_profile(), _budget(), LaunchRequest(context_tokens=64 * 1024, slots=2),
                      default_context_tokens=64 * 1024, mtp_supported=False)
    assert two.aggregate_context_tokens == 128 * 1024
    # A second slot adds its own KV/work state, not another copy of model weights.
    assert 0 < two.estimated_bytes - one.estimated_bytes < _profile().weights_bytes


def test_mtp_is_rejected_when_no_validated_recipe_exists():
    plan = plan_launch(_profile(), _budget(), LaunchRequest(speculation="mtp"),
                       default_context_tokens=64 * 1024, mtp_supported=False)
    assert plan.fits is False
    assert "validated MTP" in plan.reasons[0]


def test_explicit_context_is_never_silently_reduced():
    plan = plan_launch(_profile(), _budget(), LaunchRequest(context_tokens=129 * 1024),
                       default_context_tokens=64 * 1024, mtp_supported=False)
    assert plan.fits is False
    assert "context_tokens" in plan.reasons[0]


def test_launch_args_allocate_aggregate_context_for_parallel_slots():
    args = launch_args(_profile(), WindowDecision(window=64 * 1024, spill_bytes=0, kv_on_gpu=True), slots=3)
    assert args[args.index("-c") + 1] == str(3 * 64 * 1024)
    assert args[args.index("--parallel") + 1] == "3"
