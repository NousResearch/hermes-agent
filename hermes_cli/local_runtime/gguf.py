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

# llama.cpp b10679 added automatic, mmap-backed lookup for a large PLE table. The engine chooses
# this only when the individual tensor is strictly larger than 4 GiB; at or below the boundary it
# remains an ordinary resident tensor. Keep the parser's fact separate from the engine capability
# check below — an older installed build must price the same file normally.
AUTOMATIC_LAZY_TENSOR_THRESHOLD_BYTES = 4 << 30
LAZY_LOOKUP_MIN_ENGINE_BUILD = 10679
_LAZY_LOOKUP_TENSOR_NAME = "per_layer_token_embd.weight"


def model_id_from_stem(stem: str) -> str:
    """Model id from a GGUF file stem (split-part suffix stripped)."""
    return _PART_SUFFIX_RE.sub("", stem)


def supports_automatic_lazy_lookup(engine_tag: str | None) -> bool:
    """Whether this exact llama.cpp build uses the automatic PLE mmap path.

    ``engine_tag`` is deliberately supplied by the caller rather than read from configuration:
    boot can fall back to an older installed build while a newer configured build is pending.
    Unknown or malformed tags take the safe, resident path.
    """
    match = re.search(r"(\d+)", engine_tag or "")
    return bool(match and int(match.group(1)) >= LAZY_LOOKUP_MIN_ENGINE_BUILD)


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
    lazy_table_bytes: int = 0      # large per_layer_token_embd.weight, eligible for automatic mmap lookup

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


def _read_gguf_part(path: Path) -> GGUFHeader:
    """Read one GGUF header. Split aggregation belongs in ``read_gguf_header``."""

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
        lazy_bytes = 0
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
            if name == _LAZY_LOOKUP_TENSOR_NAME and nbytes > AUTOMATIC_LAZY_TENSOR_THRESHOLD_BYTES:
                lazy_bytes += nbytes

    return GGUFHeader(path=str(path), version=version, metadata=metadata,
                      n_tensors=n_tensors, tensor_bytes=tensor_bytes,
                      embd_table_bytes=embd_bytes, lazy_table_bytes=lazy_bytes)


def _split_parts(path: Path) -> tuple[Path, ...]:
    """Return every part for a split GGUF, refusing an incomplete set.

    A caller may hand us any continuation part, but the model's metadata lives in part one and
    accounting only makes sense with the complete set. Silently reading one shard would make a
    metadata-only first shard look like a zero-byte model.
    """
    match = SPLIT_PART_RE.search(path.name)
    if match is None:
        return (path,)
    stem = path.name[:match.start()]
    total = int(match.group(2))
    parts = tuple(path.with_name(f"{stem}-{index:05d}-of-{total:05d}.gguf")
                  for index in range(1, total + 1))
    missing = [part.name for part in parts if not part.is_file()]
    if missing:
        raise ValueError(f"incomplete split GGUF for {path.name}: missing {', '.join(missing)}")
    return parts


def read_gguf_header(path: str | Path) -> GGUFHeader:
    """Read metadata and tensor tables for one GGUF or every shard of a complete split model."""
    path = Path(path)
    headers = [_read_gguf_part(part) for part in _split_parts(path)]
    first = headers[0]
    if any(header.version != first.version for header in headers[1:]):
        raise ValueError(f"split GGUF parts disagree on version: {path}")
    if len(headers) == 1:
        return first
    return GGUFHeader(
        path=str(path), version=first.version, metadata=first.metadata,
        n_tensors=sum(header.n_tensors for header in headers),
        tensor_bytes=sum(header.tensor_bytes for header in headers),
        embd_table_bytes=sum(header.embd_table_bytes for header in headers),
        lazy_table_bytes=sum(header.lazy_table_bytes for header in headers),
    )
