"""Shared synthetic-GGUF factory for the native model-settings tests.

Writes minimal-but-valid GGUFs the real reader (local_runtime.gguf) accepts:
magic, version 3, zero tensors, and a metadata table. The ``pad`` bytes land
after the metadata: they never affect header parsing but always affect the
artifact hash, so tests can change identity without touching metadata.
"""

from __future__ import annotations

import io
import struct
from pathlib import Path


def gguf_bytes(metadata: dict | None = None, pad: bytes = b"") -> bytes:
    buf = io.BytesIO()
    metadata = metadata or {}
    buf.write(b"GGUF")
    buf.write(struct.pack("<IQQ", 3, 0, len(metadata)))
    for key, value in metadata.items():
        raw = key.encode("utf-8")
        buf.write(struct.pack("<Q", len(raw)) + raw)
        if isinstance(value, bool):
            buf.write(struct.pack("<I", 7))
            buf.write(struct.pack("<?", value))
        elif isinstance(value, int):
            buf.write(struct.pack("<I", 4))
            buf.write(struct.pack("<I", value))
        elif isinstance(value, float):
            buf.write(struct.pack("<I", 6))
            buf.write(struct.pack("<f", value))
        else:  # str
            buf.write(struct.pack("<I", 8))
            enc = str(value).encode("utf-8")
            buf.write(struct.pack("<Q", len(enc)) + enc)
    return buf.getvalue() + pad


def write_gguf(path: Path, metadata: dict | None = None, pad: bytes = b"") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gguf_bytes(metadata, pad))
    return path
