"""GGUF metadata + tensor-table reader (stdlib only).

Reads the header only (metadata + tensor infos); never touches tensor data, so it is fast enough to
run at picker time on multi-GB files.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

_GGUF_MAGIC = b"GGUF"

# Split GGUF naming: "<stem>-00001-of-00003.gguf"; the part suffix is not part of the model id.
SPLIT_PART_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")
_PART_SUFFIX_RE = re.compile(r"-\d{5}-of-\d{5}$")


def model_id_from_stem(stem: str) -> str:
    """Model id from a GGUF file stem (split-part suffix stripped)."""
    return _PART_SUFFIX_RE.sub("", stem)


# ggml tensor type sizes: type_id -> (block_bytes, block_elems). IQ-family verified against
# ggml-common.h.
_GGML_TYPE_SIZES = {
    0: (4, 1), 1: (2, 1), 2: (18, 32), 3: (20, 32), 6: (22, 32), 7: (24, 32),
    8: (34, 32), 9: (36, 32), 10: (84, 256), 11: (110, 256), 12: (144, 256),
    13: (176, 256), 14: (210, 256), 15: (292, 256), 16: (66, 256),
    17: (74, 256), 18: (98, 256), 19: (50, 256), 20: (18, 32),
    21: (110, 256), 22: (82, 256), 23: (136, 256), 24: (1, 1), 25: (2, 1),
    26: (4, 1), 27: (8, 1), 28: (8, 1), 29: (56, 256), 30: (2, 1),
}

# GGUF metadata value types -> struct format; STRING (8) and ARRAY (9) are variable-length.
_V_STRING, _V_ARRAY = 8, 9
_SCALAR_FMT = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h",       # uint8 int8 uint16 int16
    4: "<I", 5: "<i", 6: "<f", 7: "<?",       # uint32 int32 float32 bool
    10: "<Q", 11: "<q", 12: "<d",             # uint64 int64 float64
}

# general.sampling.* metadata key -> preset INI key.
_SAMPLING_INI_KEY = {"temp": "temp", "temperature": "temp", "top_p": "top-p",
                     "top_k": "top-k", "min_p": "min-p",
                     "repeat_penalty": "repeat-penalty",
                     "presence_penalty": "presence-penalty"}


@dataclass
class GGUFHeader:
    path: str
    version: int
    metadata: dict = field(default_factory=dict)
    n_tensors: int = 0
    tensor_bytes: int = 0          # exact sum over the tensor table
    embd_table_bytes: int = 0      # token_embd.weight (duplicated host-side when fully offloaded)

    # ── typed accessors ──────────────────────────────────────

    @property
    def architecture(self) -> str:
        return str(self.metadata.get("general.architecture", ""))

    def _arch_key(self, suffix: str):
        return self.metadata.get(f"{self.architecture}.{suffix}")

    def _arch_int(suffix: str, doc: str = ""):  # noqa: N805 — property factory, deleted below
        return property(lambda self: int(self._arch_key(suffix) or 0), doc=doc)

    n_layer = _arch_int("block_count")
    n_ctx_train = _arch_int("context_length")
    n_embd = _arch_int("embedding_length")
    sliding_window = _arch_int("attention.sliding_window")
    expert_count = _arch_int("expert_count")
    full_attention_interval = _arch_int(
        "full_attention_interval",
        "GDN-hybrid discriminator (qwen35 family): every Nth layer is full attention, the rest "
        "are linear/recurrent. 0 = not present.")
    del _arch_int

    @property
    def n_vocab(self) -> int:
        """Vocabulary size (prices the GPU logits buffers): vocab_size metadata when present, else
        the tokenizer list length."""
        v = self._arch_key("vocab_size")
        if v:
            return int(v)
        toks = self.metadata.get("tokenizer.ggml.tokens")
        return len(toks) if isinstance(toks, list) else 0

    @property
    def sampling_defaults(self) -> dict:
        """Upstream's recommended sampling as preset INI keys, when the file carries it.

        Publishers bake ``general.sampling.*`` keys into the GGUF (llama-server reads them as that
        model's defaults), so the file is the source of truth — it ships with the download and
        updates with every re-upload, no catalog needed. Empty if absent.
        """
        out = {}
        for key, value in self.metadata.items():
            if not key.startswith("general.sampling."):
                continue
            name = _SAMPLING_INI_KEY.get(key.rsplit(".", 1)[-1])
            if name is not None and isinstance(value, (int, float)):
                num = round(float(value), 4)
                out[name] = str(int(num)) if num == int(num) else str(num)
        return out

    @property
    def n_head(self) -> int:
        v = self._arch_key("attention.head_count")
        if isinstance(v, list):
            return int(max(v))
        return int(v or 0)

    def head_counts_kv(self) -> list[int]:
        """Per-layer KV head counts; 0 marks a recurrent/linear layer (n_head_kv == 0).

        Three GGUF shapes: a per-layer array (nemotron_h_moe) is used as-is; a scalar plus
        ``full_attention_interval`` (qwen35) applies to every N-th layer (1-indexed) and is zero
        elsewhere — pricing all layers as attention was a 4x overestimate; a plain scalar (dense)
        broadcasts to every layer.
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


def read_gguf_header(path: str | Path) -> GGUFHeader:
    path = Path(path)

    def read(f, fmt: str):
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))

    def read_str(f) -> str:
        (n,) = read(f, "<Q")
        return f.read(n).decode("utf-8", errors="replace")

    def read_value(f, vtype: int):
        if vtype == _V_STRING:
            return read_str(f)
        if vtype == _V_ARRAY:
            etype, n = read(f, "<IQ")
            return [read_value(f, etype) for _ in range(n)]
        return read(f, _SCALAR_FMT[vtype])[0]

    with open(path, "rb") as f:
        if f.read(4) != _GGUF_MAGIC:
            raise ValueError(f"not a GGUF file: {path}")
        version, n_tensors, n_kv = read(f, "<IQQ")

        metadata: dict = {}
        for _ in range(n_kv):
            key = read_str(f)
            (vtype,) = read(f, "<I")
            metadata[key] = read_value(f, vtype)

        tensor_bytes = 0
        embd_bytes = 0
        for _ in range(n_tensors):
            name = read_str(f)
            (n_dims,) = read(f, "<I")
            dims = read(f, f"<{n_dims}Q")
            (ttype,) = read(f, "<I")
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
