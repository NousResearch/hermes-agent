"""Q4 symmetric per-channel quantization for embedding compression.

Methodology: Persistent Q4 KV Cache (arXiv, Feb 2026).
  - Per-channel: one float32 scale per block of elements
  - Symmetric 4-bit: values clamped to [-7, 7]
  - Two int4 values packed per uint8 byte
  - Configurable channel size for fidelity/compression trade-off

Note: Q4 is provided as an optional aggressive mode. The default
storage uses float16 (2x compression, zero quality loss). Q4 offers
4-7x compression with some ranking degradation — suitable for
large-scale storage where storage cost dominates retrieval quality.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


def quantize_q4_per_channel(
    tensor: np.ndarray,
    channel_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    FP32 → Q4 symmetric per-channel quantization.

    Reshapes tensor as (num_channels, channel_size), computes one
    float32 scale per channel, and packs two int4 values per uint8 byte.

    Args:
        tensor: float32/float16 array of shape (D,)
        channel_size: elements per quantization channel (default 128 = head_dim)

    Returns:
        q4_packed: uint8 array of shape (ceil(D/2),) — two int4s per byte
        scales: float32 array of shape (num_channels,) — one scale per channel
    """
    D = tensor.shape[0]
    if D % 2 != 0:
        tensor = np.pad(tensor, (0, 1), mode="constant")
        D = tensor.shape[0]

    num_channels = D // channel_size
    if D % channel_size != 0:
        # Pad to the next full channel
        padded_len = (num_channels + 1) * channel_size
        tensor = np.pad(tensor, (0, padded_len - D), mode="constant")
        D = padded_len
        num_channels = D // channel_size

    # Reshape: (num_channels, channel_size)
    reshaped = tensor.reshape(num_channels, channel_size)

    # Per-channel scale: max absolute value / 7.0
    abs_max = np.maximum(np.abs(reshaped).max(axis=1, keepdims=True), 1e-12)
    scales = (abs_max / 7.0).squeeze(axis=1).astype(np.float32)

    # Quantize each channel
    quantized = np.clip(np.round(reshaped / abs_max * 7.0), -7, 7).astype(np.int8)

    # Flatten and pack 2 int4 per uint8 byte
    flat = quantized.ravel()
    even = flat[0::2] & 0x0F
    odd = flat[1::2] & 0x0F
    packed = (even | (odd << 4)).astype(np.uint8)

    return packed, scales


def dequantize_q4_per_channel(
    packed: np.ndarray,
    scales: np.ndarray,
    channel_size: int = 128,
    original_len: int | None = None,
) -> np.ndarray:
    """
    Q4 → FP32 dequantization (inverse of quantize_q4_per_channel).

    Args:
        packed: uint8 array of shape (ceil(D/2),)
        scales: float32 array of shape (num_channels,)
        channel_size: elements per quantization channel
        original_len: original unpadded length (truncates output)

    Returns:
        float32 array of shape (original_len,) or (D,)
    """
    # Unpack nibbles
    even = (packed & 0x0F).astype(np.float32)
    odd = ((packed >> 4) & 0x0F).astype(np.float32)

    # Sign-extend: values 8-15 → -8 to -1
    even = np.where(even > 7, even - 16, even)
    odd = np.where(odd > 7, odd - 16, odd)

    # Interleave: even at 0,2,4,...; odd at 1,3,5,...
    D = len(scales) * channel_size
    flat = np.empty(D, dtype=np.float32)
    flat[0::2] = even[: D // 2 + D % 2]
    flat[1::2] = odd[: D // 2]

    # Reshape to (num_channels, channel_size) and dequantize
    num_channels = len(scales)
    reshaped = flat.reshape(num_channels, channel_size)
    dequant = reshaped * (scales[:, np.newaxis] / 7.0)

    result = dequant.ravel()
    if original_len is not None:
        result = result[:original_len]

    return result


def compute_q4_size(num_elements: int, channel_size: int = 128) -> dict:
    """Compute storage sizes for a given embedding dimension.

    Returns dict with fp16_bytes, q4_packed_bytes, scales_bytes,
    total_q4_bytes, and compression_ratio.
    """
    # Pad to channel boundary
    num_channels = (num_elements + channel_size - 1) // channel_size
    padded = num_channels * channel_size

    fp16_bytes = num_elements * 2
    q4_packed_bytes = padded // 2  # 4 bits per element → half byte
    scales_bytes = num_channels * 4  # float32 per channel

    return {
        "fp16_bytes": fp16_bytes,
        "q4_packed_bytes": q4_packed_bytes,
        "scales_bytes": scales_bytes,
        "total_q4_bytes": q4_packed_bytes + scales_bytes,
        "compression_ratio": fp16_bytes / (q4_packed_bytes + scales_bytes),
        "num_channels": num_channels,
        "channel_size": channel_size,
    }


# ── Legacy compatibility with Phase 0 naming ────────────────────────────

def quantize_q4_per_channel_legacy(
    tensor: np.ndarray,
    channel_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """Alias for quantize_q4_per_channel."""
    return quantize_q4_per_channel(tensor, channel_size)


def dequantize_q4_per_channel_legacy(
    packed: np.ndarray,
    scales: np.ndarray,
    channel_size: int = 128,
    original_len: int | None = None,
) -> np.ndarray:
    """Alias for dequantize_q4_per_channel."""
    return dequantize_q4_per_channel(packed, scales, channel_size, original_len)
