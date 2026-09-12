"""Browse Hugging Face for GGUF models the user can run.

The curated catalog is the front page; this module is the firehose behind it — day-0 models not yet
in the catalog, community quants, anything. Acquisition only: nothing here serves a model. A browsed
download lands in the machine-scoped models dir and from that moment the normal machinery owns it —
staleness bounce, preset generation from the real GGUF header, fit policy, placement pills.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, replace

logger = logging.getLogger(__name__)

_HF = "https://huggingface.co"
_TIMEOUT_S = 15
# Rough-fit fill-in for pre-download pricing: a mid-size model's 64K-floor KV plus runtime
# overhead. Deliberately round — the verdict bands are coarse, not window grants.
_ROUGH_KV_AND_OVERHEAD = 4 << 30

# Tiny TTL cache: the pane fires a search per keystroke pause and re-opens repos the user flips
# between. Process-local, size-capped; upstream truth changes slowly at this granularity.
_CACHE: dict[str, tuple[float, object]] = {}
_CACHE_TTL_S = 300
_CACHE_MAX = 128


def _get_json(url: str) -> object:
    now = time.monotonic()
    hit = _CACHE.get(url)
    if hit and now - hit[0] < _CACHE_TTL_S:
        return hit[1]
    req = urllib.request.Request(url, headers={"User-Agent": "hermes-local-models"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as r:
        data = json.load(r)
    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.pop(min(_CACHE, key=lambda k: _CACHE[k][0]))
    _CACHE[url] = (now, data)
    return data


@dataclass(frozen=True)
class HFModelHit:
    repo: str                   # e.g. "unsloth/Qwen3.8-27B-GGUF"
    downloads: int
    likes: int
    updated: str                # ISO date from HF
    gated: bool


@dataclass(frozen=True)
class HFFileGroup:
    """One downloadable quant: a single GGUF or all parts of a split one."""

    label: str                  # e.g. "Q4_K_M" or the file stem
    paths: tuple[str, ...]      # repo-relative, split parts in order
    total_bytes: int
    fit: str = "unknown"        # fits-gpu | needs-ram | too-big | unknown
    # A repository tree exposes file sizes, not a GGUF tensor table. In particular, a large
    # PLE/Engram table can be mmap-backed by supported llama.cpp builds, so this is never placement.
    fit_is_estimate: bool = True


_QUANT_RE = re.compile(r"(?:IQ|Q)\d[_A-Z0-9]*|F16|BF16|F32", re.IGNORECASE)
_SPLIT_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)
_GGUF_SUFFIX = ".gguf"


def search_models(query: str, limit: int = 20) -> list[HFModelHit]:
    """Full-text search over HF models that ship GGUF files, most downloaded first (the closest
    public signal to 'trending')."""
    q = urllib.parse.quote(query.strip())
    url = (f"{_HF}/api/models?search={q}&filter=gguf&sort=downloads"
           f"&direction=-1&limit={max(1, min(int(limit), 50))}")
    return [HFModelHit(repo=str(m.get("id", "")),
                       downloads=int(m.get("downloads") or 0),
                       likes=int(m.get("likes") or 0),
                       updated=str(m.get("lastModified") or ""),
                       gated=bool(m.get("gated")))
            for m in _get_json(url)]


def _quant_label(filename: str) -> str:
    m = _QUANT_RE.search(filename)
    return m.group(0).upper() if m else filename


def _group_source_stem(group: HFFileGroup) -> str:
    """Human-readable identity for a group when two quants have the same short label.

    Repositories commonly put every quant in its own directory (``Q4/model.gguf``).  The
    quant label alone then is not a stable UI key, so retain the repository-relative source
    identity only for duplicate labels.  The download route independently makes the staged
    filename collision-proof.
    """
    path = group.paths[0]
    match = _SPLIT_RE.search(path)
    return path[:match.start()] if match is not None else path[:-len(_GGUF_SUFFIX)]


def _with_unique_labels(groups: list[HFFileGroup]) -> list[HFFileGroup]:
    """Disambiguate only duplicate display labels, preserving the simple usual-case labels."""
    counts: dict[str, int] = {}
    for group in groups:
        counts[group.label] = counts.get(group.label, 0) + 1
    return [
        (replace(group, label=f"{group.label} · {_group_source_stem(group)}")
         if counts[group.label] > 1 else group)
        for group in groups
    ]


def repo_files(repo: str) -> list[HFFileGroup]:
    """The servable GGUFs in a repo, grouped: split parts collapse into one entry (first part is
    what llama.cpp loads); mmproj/draft companions are excluded. Largest quant first."""
    url = f"{_HF}/api/models/{urllib.parse.quote(repo)}/tree/main?recursive=true"
    files = _get_json(url)

    singles: list[tuple[str, int]] = []
    splits: dict[str, list[tuple[int, int, str, int]]] = {}
    for f in files:
        path = str(f.get("path", ""))
        if not path.casefold().endswith(_GGUF_SUFFIX):
            continue
        name = path.rsplit("/", 1)[-1].lower()
        if name.startswith(("mmproj", "dspark")) or "draft" in name:
            continue
        size = int(f.get("size") or 0)
        m = _SPLIT_RE.search(path)
        if m:
            splits.setdefault(path[: m.start()], []).append(
                (int(m.group(1)), int(m.group(2)), path, size))
        else:
            singles.append((path, size))

    groups = [HFFileGroup(label=_quant_label(path), paths=(path,), total_bytes=size)
              for path, size in singles]
    for stem, parts in splits.items():
        # An incomplete split must never be presented as a quant. llama.cpp can only load it
        # after every numbered shard exists, and accepting it here would make any header result
        # deceptively incomplete.
        totals = {total for _, total, _, _ in parts}
        expected = next(iter(totals), 0)
        indexed = {index: (path, size) for index, _, path, size in parts}
        if (len(totals) != 1 or len(parts) != expected
                or set(indexed) != set(range(1, expected + 1))):
            logger.warning("skipping incomplete GGUF split %s", stem)
            continue
        ordered = [indexed[index] for index in range(1, expected + 1)]
        groups.append(HFFileGroup(label=_quant_label(stem), paths=tuple(path for path, _ in ordered),
                                  total_bytes=sum(size for _, size in ordered)))
    groups = _with_unique_labels(groups)
    groups.sort(key=lambda g: g.total_bytes, reverse=True)
    return groups


def rough_fit(total_bytes: int, budget) -> str:
    """Coarse pre-download verdict from file size alone (GGUF file size ≈ in-memory weights);
    the GGUF header refines this after download. Bands match the catalog pills' language."""
    # This is deliberately not a placement claim: only the finished header says whether a PLE
    # lookup can stay disk-backed on the active engine.
    need = total_bytes + _ROUGH_KV_AND_OVERHEAD
    if need <= budget.usable_vram_bytes:
        return "fits-gpu"
    if need <= budget.usable_vram_bytes + budget.ram_available_bytes:
        return "needs-ram"
    return "too-big"


def priced_repo_files(repo: str, budget) -> list[HFFileGroup]:
    return [replace(g, fit=rough_fit(g.total_bytes, budget)) for g in repo_files(repo)]


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from dataclasses import field  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
