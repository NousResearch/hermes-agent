"""The managed launch must not force a q8_0 KV cache on architectures whose attention head
dims 32 cannot divide: llama.cpp refuses to boot those models
("V cache type q8_0 with block size 32 does not divide n_embd_head_v=72", as with
Kimi-Linear-48B-A3B's head_dim_v=72), and the managed runtime turns the refusal into an HTTP 500
while a hand-launched server with the default f16 cache runs fine.

The contract under test: profile_from_gguf flags such architectures (kv_q8_ok=False), the launch
flags fall back to an f16 KV cache, and the context estimator prices the doubled KV so the 64K
memory guarantee still holds.
"""

from __future__ import annotations

from hermes_cli.local_runtime.context_policy import WindowDecision, launch_args
from hermes_cli.local_runtime.estimator import ctx_bytes, kv_dtype_factor, profile_from_gguf
from hermes_cli.local_runtime.gguf import GGUFHeader

_ARCH = "kimi-linear-test"


def _header(**extra_metadata) -> GGUFHeader:
    metadata = {
        "general.architecture": _ARCH,
        f"{_ARCH}.block_count": 4,
        f"{_ARCH}.context_length": 131072,
        f"{_ARCH}.attention.head_count_kv": 8,
        f"{_ARCH}.attention.key_length": 576,
        f"{_ARCH}.attention.value_length": 72,  # 72 % 32 != 0: q8_0 blocks cannot serve it
        **extra_metadata,
    }
    return GGUFHeader(path="test.gguf", version=3, metadata=metadata,
                      n_tensors=0, tensor_bytes=0, embd_table_bytes=0)


def _args(profile, **kwargs) -> list[str]:
    decision = WindowDecision(window=131072, spill_bytes=0, kv_on_gpu=True)
    return launch_args(profile, decision, **kwargs)


def test_head_dim_not_divisible_by_32_disables_kv_q8():
    profile = profile_from_gguf(_header())
    assert profile.kv_q8_ok is False

    args = _args(profile)
    assert args[args.index("-ctk") + 1] == "f16"
    assert args[args.index("-ctv") + 1] == "f16"


def test_swa_head_dim_not_divisible_disables_kv_q8_too():
    profile = profile_from_gguf(_header(**{
        f"{_ARCH}.attention.sliding_window_pattern": [1, 1, 1, 1],
        f"{_ARCH}.attention.key_length_swa": 576,
        f"{_ARCH}.attention.value_length_swa": 72,
    }))
    assert profile.kv_q8_ok is False


def test_divisible_head_dims_keep_the_q8_kv_cache():
    profile = profile_from_gguf(_header(**{f"{_ARCH}.attention.value_length": 128}))
    assert profile.kv_q8_ok is True

    args = _args(profile)
    assert args[args.index("-ctk") + 1] == "q8_0"
    assert args[args.index("-ctv") + 1] == "q8_0"


def test_non_divisible_dims_are_priced_at_full_f16_kv():
    profile = profile_from_gguf(_header())
    window = 65536
    with_fa = ctx_bytes(profile, window, flash_attention=True)
    without_fa = ctx_bytes(profile, window, flash_attention=False)
    # Same memory plan as the non-FA posture: the physics check must see the doubled KV.
    assert with_fa == without_fa
    assert kv_dtype_factor(flash_attention=True, kv_q8_ok=profile.kv_q8_ok) == 1.0


def test_divisible_dims_keep_the_q8_memory_discount():
    profile = profile_from_gguf(_header(**{f"{_ARCH}.attention.value_length": 128}))
    window = 65536
    with_fa = ctx_bytes(profile, window, flash_attention=True)
    without_fa = ctx_bytes(profile, window, flash_attention=False)
    assert with_fa < without_fa
    assert kv_dtype_factor(flash_attention=True, kv_q8_ok=profile.kv_q8_ok) < 1.0
