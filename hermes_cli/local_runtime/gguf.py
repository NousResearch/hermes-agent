"""GGUF metadata + tensor-table reader (stdlib only).

Reads the header only (metadata + tensor infos); never touches tensor data, so it is fast enough to
run at picker time on multi-GB files.
"""

from __future__ import annotations

import logging
import re
import struct
from dataclasses import dataclass, field, replace
from pathlib import Path

logger = logging.getLogger(__name__)

_GGUF_MAGIC = b"GGUF"

# Split GGUF naming: "<stem>-00001-of-00003.gguf"; the part suffix is not part of the model id.
SPLIT_PART_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")
_PART_SUFFIX_RE = re.compile(r"-\d{5}-of-\d{5}$")
# Same tensor selection as context_policy's per-block FFN -ot override.
_FFN_WEIGHT = re.compile(r"blk\.(\d+)\.ffn_.*\.weight")


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
    39: (17, 32),  # MXFP4 — 32 elements per 17-byte block (gpt-oss family)
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
    # block index -> bytes of that block's FFN weights (the tensors a `blk\.N\.ffn_.*\.weight`
    # -ot override moves), so spill placement can move only as many blocks as it needs.
    ffn_block_bytes: dict[int, int] = field(default_factory=dict)

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
    key_length_swa = _arch_int(
        "attention.key_length_swa",
        "Per-token key size for sliding-window layers when it differs from the global "
        "attention.key_length (e.g. gemma3/gemma4). 0 = not present.")
    value_length_swa = _arch_int(
        "attention.value_length_swa",
        "Per-token value size for sliding-window layers when it differs from the global "
        "attention.value_length. 0 = not present.")
    del _arch_int

    @property
    def sliding_window_pattern(self) -> list[int] | None:
        """Per-layer SWA pattern declared by the file itself: a truthy entry marks a
        sliding-window layer, falsy marks global. None when the file doesn't declare the per-layer
        array form (older GGUFs, architectures without per-layer SWA metadata, or files that use
        the scalar period form instead — see `sliding_window_pattern_period`)."""
        v = self._arch_key("attention.sliding_window_pattern")
        return [int(x) for x in v] if isinstance(v, list) else None

    @property
    def sliding_window_pattern_period(self) -> int:
        """Scalar SWA period declared by the file: llama.cpp also permits
        `attention.sliding_window_pattern` as a single integer N (every Nth layer is full
        attention, e.g. Gemma-family writers) rather than a per-layer array. 0 when absent or when
        the file uses the array form instead — callers should fall back to a coarser signal."""
        v = self._arch_key("attention.sliding_window_pattern")
        return int(v) if isinstance(v, int) and not isinstance(v, bool) else 0

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


def _split_parts(path: Path) -> "list[Path] | None":
    """Every on-disk part of the split ``path`` belongs to, first part first; None when ``path`` is
    not a split member or no other part is present.

    A split is priced as the set, never as one file: publishers lay shards out so the first part
    can hold little more than metadata while the bulk sits in the later ones, so reading one part
    prices the model at whatever fraction of its weights that part happens to hold."""
    m = SPLIT_PART_RE.search(path.name)
    if m is None:
        return None
    stem, total = path.name[: m.start()], int(m.group(2))
    parts = [p for p in (path.with_name(f"{stem}-{i:05d}-of-{total:05d}.gguf")
                         for i in range(1, total + 1)) if p.is_file()]
    return parts if len(parts) > 1 else None


def _read_part(path: Path) -> GGUFHeader:
    """One file's own header: metadata and that file's tensor table."""
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
        ffn_block_bytes: dict[int, int] = {}
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
            elif m := _FFN_WEIGHT.match(name):
                block = int(m.group(1))
                ffn_block_bytes[block] = ffn_block_bytes.get(block, 0) + nbytes

    return GGUFHeader(path=str(path), version=version, metadata=metadata,
                      n_tensors=n_tensors, tensor_bytes=tensor_bytes,
                      embd_table_bytes=embd_bytes, ffn_block_bytes=ffn_block_bytes)


def read_gguf_header(path: str | Path) -> GGUFHeader:
    """Header for a MODEL, not for a file: a split GGUF is priced as the sum of its parts.

    Weights, the host-side embedding-table duplicate and the per-block FFN map are all summed
    across the shards on disk, because a split is a layout choice, not a smaller model — pricing
    part 1 alone underprices every model whose first shard is a metadata stub (the Hugging Face
    layout), which then makes the physics check and the residency cap admit giants the card cannot
    hold. A split writer repeats the metadata in every shard, so architecture facts (block count,
    train context, the per-layer SWA pattern, vocab) come from the part named by the caller.

    A sibling that has gone missing or become unreadable is skipped rather than fatal: a
    half-arrived split prices at what is actually on disk. Refusing it is
    ``staged_in(require_complete=True)``'s job — an incomplete split is never servable, it is only
    underpriced here. The part the caller named is the exception: if that one cannot be read there
    is no header to answer with."""
    path = Path(path)
    named = _read_part(path)
    parts = _split_parts(path)
    if parts is None:
        return named

    readable = [named]
    for part in parts:
        if part == path:
            continue
        try:
            readable.append(_read_part(part))
        except (ValueError, OSError, struct.error) as exc:
            # struct.error is neither ValueError nor OSError, and it is what a header cut
            # mid-stream actually raises. A part we cannot parse is priced out, not fatal.
            logger.debug("split part unreadable %s: %s", part.name, exc)

    ffn_block_bytes: dict[int, int] = {}
    for header in readable:
        for block, nbytes in header.ffn_block_bytes.items():
            ffn_block_bytes[block] = ffn_block_bytes.get(block, 0) + nbytes
    return replace(named,
                   n_tensors=sum(h.n_tensors for h in readable),
                   tensor_bytes=sum(h.tensor_bytes for h in readable),
                   embd_table_bytes=sum(h.embd_table_bytes for h in readable),
                   ffn_block_bytes=ffn_block_bytes)
