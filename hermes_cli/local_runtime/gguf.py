"""GGUF metadata + tensor-table reader (stdlib only).

Feeds the per-layer context estimator: architecture, layer count, per-layer
KV head counts (0 = recurrent layer — the hybrid discriminator), head dims,
sliding-window config, trained context, and exact weight bytes summed from
the tensor table (validated to within 0.01% of the loader's buffer).

Reads the header only (metadata + tensor infos); never touches tensor data,
so it is fast enough to run at picker time on multi-GB files.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

_GGUF_MAGIC = b"GGUF"

# ggml tensor type sizes: type_id -> (block_bytes, block_elems).
# IQ-family sizes verified against ggml-common.h.
_GGML_TYPE_SIZES = {
    0: (4, 1), 1: (2, 1), 2: (18, 32), 3: (20, 32), 6: (22, 32), 7: (24, 32),
    8: (34, 32), 9: (36, 32), 10: (84, 256), 11: (110, 256), 12: (144, 256),
    13: (176, 256), 14: (210, 256), 15: (292, 256), 16: (66, 256),
    17: (74, 256), 18: (98, 256), 19: (50, 256), 20: (18, 32),
    21: (110, 256), 22: (82, 256), 23: (136, 256), 24: (1, 1), 25: (2, 1),
    26: (4, 1), 27: (8, 1), 28: (8, 1), 29: (56, 256), 30: (2, 1),
}

# GGUF metadata value types.
_V_UINT8, _V_INT8, _V_UINT16, _V_INT16 = 0, 1, 2, 3
_V_UINT32, _V_INT32, _V_FLOAT32, _V_BOOL = 4, 5, 6, 7
_V_STRING, _V_ARRAY, _V_UINT64, _V_INT64, _V_FLOAT64 = 8, 9, 10, 11, 12

_SCALAR_FMT = {
    _V_UINT8: "<B", _V_INT8: "<b", _V_UINT16: "<H", _V_INT16: "<h",
    _V_UINT32: "<I", _V_INT32: "<i", _V_FLOAT32: "<f", _V_BOOL: "<?",
    _V_UINT64: "<Q", _V_INT64: "<q", _V_FLOAT64: "<d",
}


@dataclass
class GGUFHeader:
    path: str
    version: int
    metadata: dict = field(default_factory=dict)
    n_tensors: int = 0
    tensor_bytes: int = 0          # exact sum over the tensor table
    embd_table_bytes: int = 0      # token_embd.weight (duplicated host-side
                                   # when fully offloaded)

    # ── typed accessors ──────────────────────────────────────

    @property
    def architecture(self) -> str:
        return str(self.metadata.get("general.architecture", ""))

    def _arch_key(self, suffix: str):
        return self.metadata.get(f"{self.architecture}.{suffix}")

    @property
    def n_layer(self) -> int:
        return int(self._arch_key("block_count") or 0)

    @property
    def n_vocab(self) -> int:
        """Vocabulary size: prices the GPU logits buffers (they scale
        ubatch x vocab). vocab_size metadata when present, else the
        tokenizer list length."""
        v = self._arch_key("vocab_size")
        if v:
            return int(v)
        toks = self.metadata.get("tokenizer.ggml.tokens")
        return len(toks) if isinstance(toks, list) else 0

    @property
    def n_ctx_train(self) -> int:
        return int(self._arch_key("context_length") or 0)

    @property
    def sampling_defaults(self) -> dict:
        """Upstream's recommended sampling, when the file carries it.

        Model publishers bake general.sampling.* keys into the GGUF
        (llama-server reads them as that model's default generation
        settings), so the file itself is the source of truth for how its
        publisher wants it run — it arrives with the download and updates
        with every re-upload, no catalog required. Returned as preset INI
        keys; empty when the file carries none.
        """
        ini_key = {"temp": "temp", "temperature": "temp", "top_p": "top-p",
                   "top_k": "top-k", "min_p": "min-p",
                   "repeat_penalty": "repeat-penalty",
                   "presence_penalty": "presence-penalty"}
        out = {}
        for key, value in self.metadata.items():
            if not key.startswith("general.sampling."):
                continue
            name = ini_key.get(key.rsplit(".", 1)[-1])
            if name is not None and isinstance(value, (int, float)):
                num = round(float(value), 4)
                out[name] = str(int(num)) if num == int(num) else str(num)
        return out

    @property
    def n_embd(self) -> int:
        return int(self._arch_key("embedding_length") or 0)

    @property
    def n_head(self) -> int:
        v = self._arch_key("attention.head_count")
        if isinstance(v, list):
            return int(max(v))
        return int(v or 0)

    @property
    def full_attention_interval(self) -> int:
        """GDN-hybrid discriminator (qwen35 family): every Nth layer is full
        attention, the rest are linear/recurrent. 0 = not present."""
        return int(self._arch_key("full_attention_interval") or 0)

    def head_counts_kv(self) -> list[int]:
        """Per-layer KV head counts; 0 marks a recurrent/linear layer (the
        n_head_kv == 0 discriminator).

        Three GGUF shapes, each verified against real files:
        - per-layer array (nemotron_h_moe): use as-is;
        - scalar + full_attention_interval (qwen35): the scalar applies to
          every INTERVAL-th layer (1-indexed: layers where (i+1) % N == 0),
          zero elsewhere — pricing all layers as attention was a 4x
          overestimate on Qwen3.6-27B;
        - plain scalar (dense): broadcast to every layer.
        """
        v = self._arch_key("attention.head_count_kv")
        if isinstance(v, list):
            return [int(x) for x in v]
        scalar = int(v or 0)
        interval = self.full_attention_interval
        if interval > 1:
            return [scalar if (i + 1) % interval == 0 else 0
                    for i in range(self.n_layer)]
        return [scalar] * self.n_layer

    @property
    def head_dim_k(self) -> int:
        v = self._arch_key("attention.key_length")
        if v:
            return int(v)
        return self.n_embd // self.n_head if self.n_head else 0

    @property
    def head_dim_v(self) -> int:
        v = self._arch_key("attention.value_length")
        if v:
            return int(v)
        return self.head_dim_k

    @property
    def sliding_window(self) -> int:
        return int(self._arch_key("attention.sliding_window") or 0)

    @property
    def expert_count(self) -> int:
        return int(self._arch_key("expert_count") or 0)


def read_gguf_header(path: str | Path) -> GGUFHeader:
    path = Path(path)

    def read_str(f) -> str:
        (n,) = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8", errors="replace")

    def read_value(f, vtype: int):
        if vtype == _V_STRING:
            return read_str(f)
        if vtype == _V_ARRAY:
            (etype,) = struct.unpack("<I", f.read(4))
            (n,) = struct.unpack("<Q", f.read(8))
            return [read_value(f, etype) for _ in range(n)]
        fmt = _SCALAR_FMT[vtype]
        (value,) = struct.unpack(fmt, f.read(struct.calcsize(fmt)))
        return value

    with open(path, "rb") as f:
        if f.read(4) != _GGUF_MAGIC:
            raise ValueError(f"not a GGUF file: {path}")
        (version,) = struct.unpack("<I", f.read(4))
        n_tensors, n_kv = struct.unpack("<QQ", f.read(16))

        metadata: dict = {}
        for _ in range(n_kv):
            key = read_str(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            metadata[key] = read_value(f, vtype)

        tensor_bytes = 0
        embd_bytes = 0
        for _ in range(n_tensors):
            name = read_str(f)
            (n_dims,) = struct.unpack("<I", f.read(4))
            dims = struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims))
            (ttype,) = struct.unpack("<I", f.read(4))
            f.read(8)  # offset
            size = _GGML_TYPE_SIZES.get(ttype)
            if size is None:
                raise ValueError(f"unknown ggml tensor type {ttype} in {path}")
            block_bytes, block_elems = size
            elems = 1
            for d in dims:
                elems *= d
            nbytes = (elems // block_elems) * block_bytes
            tensor_bytes += nbytes
            if name == "token_embd.weight":
                embd_bytes = nbytes

    return GGUFHeader(path=str(path), version=version, metadata=metadata,
                      n_tensors=n_tensors, tensor_bytes=tensor_bytes,
                      embd_table_bytes=embd_bytes)
