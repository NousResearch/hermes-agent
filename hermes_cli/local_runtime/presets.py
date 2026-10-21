"""Per-model preset generation (--models-preset INI) — the router-side carrier for context-policy
launch decisions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path

from hermes_cli.local_runtime.context_policy import (
    RUNTIME_OVERHEAD_BYTES, fit_to_free_memory, launch_args, plan_launch, ub_logits_bytes)
from hermes_cli.local_runtime.estimator import (
    HardwareBudget, PhysicsRefusal, as_loaded, ctx_bytes, footprint_bytes, profile_from_gguf)
from hermes_cli.local_runtime.gguf import model_id_from_stem, read_gguf_header

logger = logging.getLogger(__name__)

# args list -> INI keys. Flags the policy owns; everything else stays out of the preset.
_FLAG_TO_KEY = {
    "-c": "ctx-size", "-b": "batch-size", "-ub": "ubatch-size",
    "-ctk": "cache-type-k", "-ctv": "cache-type-v", "-fa": "flash-attn",
    "-ot": "override-tensor", "--spec-type": "spec-type", "--spec-draft-n-max": "spec-draft-n-max",
}


@dataclass
class PresetEntry:
    model_id: str
    window: int
    spilled: bool
    refusal: str | None = None
    keys: dict[str, str] | None = None


def _args_to_keys(args: list[str]) -> dict[str, str]:
    keys: dict[str, str] = {}
    i = 0
    while i < len(args):
        key = _FLAG_TO_KEY.get(args[i])
        if key is None:
            i += 1
            continue
        keys[key] = args[i + 1]
        i += 2
    return keys


def _asset_path(asset) -> Path | None:
    """On-disk path of a catalog companion asset, or None when it isn't downloaded."""
    from hermes_cli.local_runtime.bootstrap import assets_dir

    if asset is None:
        return None
    path = assets_dir() / asset.local_name
    return path if path.exists() else None


@dataclass(frozen=True)
class _Companions:
    """The files a catalog entry loads beside its weights, as far as they are on disk."""

    mmproj: Path | None = None
    mtp_head: Path | None = None
    nbytes: int = 0


def _companions(entry) -> _Companions:
    if entry is None:
        return _Companions()
    mmproj, head = _asset_path(entry.mmproj), _asset_path(entry.mtp_head)
    nbytes = sum(a.size_bytes for a, p in ((entry.mmproj, mmproj), (entry.mtp_head, head)) if p)
    return _Companions(mmproj=mmproj, mtp_head=head, nbytes=nbytes)


def _runs_mtp(entry, companions: _Companions) -> bool:
    """MTP spec decode runs for a catalog entry: built-in heads, or a shipped head on disk."""
    return entry is not None and (entry.mtp or companions.mtp_head is not None)


def _draft_fits(path: Path, profile, budget: HardwareBudget, window: int, overhead: int) -> bool:
    """Optional draft never shrinks the advertised window or displaces its GPU buffers.

    The catalog does not know the draft's context layout. Admit it only after reading the file;
    draft KV defaults to f16, independently of the target's q8 cache.
    """
    try:
        draft = profile_from_gguf(read_gguf_header(path))
        draft_need = footprint_bytes(
            draft, window, flash_attention=False,
            overhead_bytes=RUNTIME_OVERHEAD_BYTES + ub_logits_bytes(draft.n_vocab, mtp_capable=False))
    except (ValueError, OSError) as exc:
        logger.warning("draft omitted %s: %s", path.name, exc)
        return False
    return (footprint_bytes(profile, window, overhead_bytes=overhead) + draft_need
            <= budget.usable_vram_bytes + budget.ram_available_bytes
            and ctx_bytes(profile, window) + overhead + draft_need <= budget.usable_vram_bytes)


# The draft model's own context beside the target's: its KV and compute buffers. Without it the
# price below still covered every measured server (b11671, -ub 2048, one slot) by 0.2 GiB
# (Nemotron 3.5 Bolt) to 2.7 GiB (Qwen3.8 27B); this keeps the thinnest case clear.
_DRAFT_CONTEXT_BYTES = 512 << 20


def _engine_build(tag: str | None) -> int | None:
    try:
        return int(str(tag).removeprefix("b")) if tag else None
    except ValueError:
        return None


def _dflash_extra_bytes(entry, profile) -> int:
    """Fixed bytes a one-slot DFlash draft adds beside its target: the draft's weights, a copy of
    the recurrent state per draft position, and the draft's own context."""
    copies = entry.draft_depth + 1
    return entry.draft.size_bytes + profile.recurrent_state_bytes * copies + _DRAFT_CONTEXT_BYTES


def _runs_dflash(entry) -> Path | None:
    """The entry's DFlash draft on disk, or None."""
    if entry is None or entry.draft is None or entry.draft_spec_type != "draft-dflash":
        return None
    return _asset_path(entry.draft)


def _dflash_plan(entry, profile, budget: HardwareBudget, live: HardwareBudget | None, *,
                 fixed_overhead: int, requested_window: int | None, engine_tag: str | None,
                 window_without: int, residents: int = 1):
    """The launch plan with the entry's DFlash draft in place of MTP, or None.

    The draft runs on one slot: each slot keeps a recurrent-state copy per draft position, and
    the router's default four slots hold enough copies to push weights off the GPU. It may
    shrink the window to half native, never below what the model gets without it when that is
    smaller; a requested (grown) window it can't hold keeps the model as it was. On one slot a
    side request takes the chat's slot and the prompt cache brings the chat back, so where free
    commit sizes that cache the draft keeps its smallest useful size.
    """
    if _runs_dflash(entry) is None:
        return None
    build = _engine_build(engine_tag)
    if build is None or build < (_engine_build(entry.draft_min_engine) or 0):
        return None
    lean = replace(profile, kv_scale=1.0)
    overhead = fixed_overhead + _dflash_extra_bytes(entry, lean)
    plan = plan_launch(lean, budget, fixed_overhead=overhead, requested_window=requested_window)
    if live is not None and live.commit_room_bytes is not None:
        from hermes_cli.local_runtime.hardware import usable_beside_prompt_cache

        live = replace(live, usable_vram_bytes=usable_beside_prompt_cache(live, residents))
    if live is not None:
        # fit_to_free_memory holds a too-busy card at the floor rather than move weights off it on a
        # live reading. The draft never rides on that hold: it runs only where it fits now.
        now = plan_launch(lean, live, fixed_overhead=overhead)
        if isinstance(now.decision, PhysicsRefusal) or now.decision.spilled:
            return None
        plan = fit_to_free_memory(plan, lean, live, mtp_capable=False, fixed_overhead=overhead)
    decision = plan.decision
    if isinstance(decision, PhysicsRefusal) or decision.spilled:
        return None
    native = profile.n_ctx_train or decision.window
    if requested_window and decision.window < min(requested_window, native):
        return None
    if decision.window < min(window_without, native // 2):
        return None
    return plan


def preset_for_model(gguf: Path, budget: HardwareBudget,
                     mtp_capable: set[str], *, requested_window: int | None = None,
                     live: HardwareBudget | None = None, residents: int = 1,
                     engine_tag: str | None = None) -> PresetEntry | None:
    """The launch decision for one staged model, or None when its header is unreadable.

    ``budget`` is the card's capacity; ``live`` (from ``hardware.launch_budget``) narrows the
    window to what fits beside other programs now. ``residents`` is how many models the router may
    hold at once; on Windows unified memory their prompt caches share the commit they leave.
    ``engine_tag`` is the installed llama.cpp release; without it no draft that needs a minimum
    release runs.
    """
    from hermes_cli.local_runtime.catalog import entry_for_model
    from hermes_cli.local_runtime.growth import load_window_overrides

    model_id = model_id_from_stem(gguf.stem)
    try:
        header = read_gguf_header(gguf)
        profile = as_loaded(profile_from_gguf(header), budget)
    except (ValueError, OSError) as exc:
        logger.warning("preset skip %s: %s", gguf.name, exc)
        return None
    entry = entry_for_model(model_id)
    companions = _companions(entry)
    is_mtp = _runs_mtp(entry, companions) if entry is not None else model_id in mtp_capable

    fixed_overhead = RUNTIME_OVERHEAD_BYTES + companions.nbytes
    requested = load_window_overrides().get(model_id) if requested_window is None else requested_window
    plan = plan_launch(profile, budget, mtp_capable=is_mtp, fixed_overhead=fixed_overhead,
                       requested_window=requested)
    if live is not None:
        plan = fit_to_free_memory(plan, profile, live, mtp_capable=is_mtp,
                                  fixed_overhead=fixed_overhead)
    decision = plan.decision
    if isinstance(decision, PhysicsRefusal):
        return PresetEntry(model_id=model_id, window=0, spilled=False, refusal=decision.message)

    dflash = None if decision.spilled else _dflash_plan(
        entry, profile, budget, live, fixed_overhead=fixed_overhead, requested_window=requested,
        engine_tag=engine_tag, window_without=decision.window, residents=residents)
    if dflash is not None:
        plan, decision = dflash, dflash.decision
    # Router discovery is preset-only: refused files must never autoload with stock fit.
    keys = _args_to_keys(launch_args(
        profile, decision, mtp_capable=is_mtp and dflash is None, uma=budget.uma,
        mtp_prefill=plan.mtp_prefill,
        mtp_draft_depth=entry.mtp_draft_depth if entry is not None else 3))
    keys["model"] = str(gguf)
    from hermes_cli.local_runtime.hardware import prompt_cache_bytes
    cache = prompt_cache_bytes(live, footprint_bytes(plan.profile, decision.window,
                                                     overhead_bytes=plan.overhead_bytes), residents)
    if cache is not None:
        keys["cache-ram"] = str(cache >> 20)  # MiB; 0 turns the cache off
    if profile.lazy_bytes:
        # The price above leaves these tensors on disk; llama.cpp's own default loads them on
        # integrated GPUs.
        keys["lazy-mode"] = "on"
    if entry is not None and is_mtp:
        # Integrated-MTP targets sample on the backend, and so does the draft (pairing validated
        # against the vendor's published llama.cpp recipes).
        keys["backend-sampling"] = "on"
        keys["spec-draft-backend-sampling"] = "on"

    # Sampling deference ladder, under the policy keys (policy wins on clash): the GGUF's own
    # general.sampling.* metadata is the publisher's recommendation and covers models the catalog
    # has never heard of; catalog sampling applies only where the file is silent; a model
    # carrying neither runs llama.cpp defaults.
    for k, v in header.sampling_defaults.items():
        keys.setdefault(k, v)
    if entry is not None:
        for k, v in (entry.sampling or {}).items():
            keys.setdefault(k, v)
        if companions.mmproj is not None:
            keys["mmproj"] = str(companions.mmproj)
        if companions.mtp_head is not None:
            # The model carries no MTP layers: draft-mtp drafts from the shipped head instead.
            keys["model-draft"] = str(companions.mtp_head)
        if dflash is not None:
            keys["model-draft"] = str(_runs_dflash(entry))
            keys["spec-type"] = "draft-dflash"
            keys["spec-draft-n-max"] = str(entry.draft_depth)
            keys["parallel"] = "1"
        draft_path = (_asset_path(entry.draft) if decision.spilled
                      and entry.draft_spec_type == "draft-dspark" else None)
        if draft_path is not None and _draft_fits(draft_path, plan.profile, budget, decision.window,
                                                  plan.overhead_bytes):
            keys["model-draft"] = str(draft_path)
            keys["spec-type"] = "draft-dspark"
            # Unsloth's measured cliff: acceptance 83% at 2-3 drafts, collapses at 4.
            keys["spec-draft-n-max"] = "3"
    return PresetEntry(model_id=model_id, window=decision.window,
                       spilled=decision.spilled, keys=keys)


def resident_footprint(gguf: Path, budget: HardwareBudget, window: int, *, drafting: bool = False) -> int | None:
    """Estimated bytes one staged model holds while loaded at ``window``, or None when unreadable.
    ``drafting`` prices the one-slot DFlash launch its preset runs instead of MTP."""
    from hermes_cli.local_runtime.catalog import entry_for_model

    model_id = model_id_from_stem(gguf.stem)
    try:
        profile = as_loaded(profile_from_gguf(read_gguf_header(gguf)), budget)
    except (ValueError, OSError) as exc:
        logger.debug("footprint skip %s: %s", gguf.name, exc)
        return None
    entry = entry_for_model(model_id)
    companions = _companions(entry)
    fixed_overhead = RUNTIME_OVERHEAD_BYTES + companions.nbytes
    if drafting and _runs_dflash(entry) is not None:
        lean = replace(profile, kv_scale=1.0)
        plan = plan_launch(lean, budget, fixed_overhead=fixed_overhead + _dflash_extra_bytes(entry, lean),
                           requested_window=window)
    else:
        plan = plan_launch(profile, budget, mtp_capable=_runs_mtp(entry, companions),
                           fixed_overhead=fixed_overhead, requested_window=window)
    return footprint_bytes(plan.profile, window, overhead_bytes=plan.overhead_bytes)


def _launch_footprint(gguf: Path, budget: HardwareBudget, engine_tag: str | None = None) -> int | None:
    """Estimated resident bytes for one staged model at the window this policy grants it, or None
    when it cannot be priced: an unreadable header, or a model the physics check refuses outright
    (it never loads, so it must not shrink the residency cap)."""
    from hermes_cli.local_runtime.catalog import entry_for_model
    from hermes_cli.local_runtime.growth import load_window_overrides

    model_id = model_id_from_stem(gguf.stem)
    try:
        profile = as_loaded(profile_from_gguf(read_gguf_header(gguf)), budget)
    except (ValueError, OSError) as exc:
        logger.debug("footprint skip %s: %s", gguf.name, exc)
        return None
    entry = entry_for_model(model_id)
    companions = _companions(entry)
    fixed_overhead = RUNTIME_OVERHEAD_BYTES + companions.nbytes
    requested = load_window_overrides().get(model_id)
    plan = plan_launch(profile, budget, mtp_capable=_runs_mtp(entry, companions),
                       fixed_overhead=fixed_overhead, requested_window=requested)
    if isinstance(plan.decision, PhysicsRefusal):
        return None
    # Priced whole even when the plan spills: a spilled model still holds part of its weights on
    # the device, and over-counting errs toward the side that cannot thrash.
    price = footprint_bytes(plan.profile, plan.decision.window, overhead_bytes=plan.overhead_bytes)
    # Where the engine loads the draft, the preset runs it in place of MTP: the model holds the
    # larger of the two launches, never both.
    dflash = None if plan.decision.spilled else _dflash_plan(
        entry, profile, budget, None, fixed_overhead=fixed_overhead, requested_window=requested,
        engine_tag=engine_tag, window_without=plan.decision.window)
    if dflash is not None:
        price = max(price, footprint_bytes(dflash.profile, dflash.decision.window,
                                           overhead_bytes=dflash.overhead_bytes))
    return price


def admitted_residency_count(models_dir: Path, budget: HardwareBudget, configured: int,
                             live: HardwareBudget | None = None, engine_tag: str | None = None) -> int:
    """How many models the card may hold resident at once: priced against the budget, not a count.

    Residency used to be bounded by a count alone, so a second model was admitted against an
    already-full card. On Windows/WDDM that over-commit is not refused — the allocation is paged
    to host memory, and that child decodes at a third of its speed for the rest of its life: no
    error, no UI hint, and ejecting the incumbent afterwards does not repair it (only a clean
    reload does). Capping the count instead has llama.cpp evict its LRU *before* the incoming
    child allocates, which is the only placement that fits.

    The cap rises above one only while the LARGEST staged model still fits TWICE — any pair of
    staged models then fits by construction. ``configured`` stays a ceiling (a user's smaller
    number is honoured), and an unpriceable input (no usable device memory, no readable model)
    keeps today's behaviour. On Windows unified memory ``live`` (free commit) bounds it too: the
    models the router holds together must fit in commit, not just in the pool.
    """
    from hermes_cli.local_runtime.bootstrap import staged_in

    if configured <= 1 or budget.usable_vram_bytes <= 0:
        return configured
    largest = 0
    for gguf in staged_in(models_dir):
        need = _launch_footprint(gguf, budget, engine_tag)
        if need:
            largest = max(largest, need)
    if largest <= 0:
        return configured
    usable = budget.usable_vram_bytes
    if live is not None and live.commit_room_bytes is not None:
        usable = min(usable, live.usable_vram_bytes)
    return max(1, min(configured, usable // largest))


def plan_presets(models_dir: Path, budget: HardwareBudget, mtp_capable: set[str] | None = None,
                 *, live: HardwareBudget | None = None, residents: int = 1,
                 engine_tag: str | None = None) -> list[PresetEntry]:
    """The launch decision for every staged model; unreadable headers are skipped."""
    from hermes_cli.local_runtime.bootstrap import staged_in

    entries = (preset_for_model(gguf, budget, mtp_capable or set(), live=live, residents=residents,
                                engine_tag=engine_tag)
               for gguf in staged_in(models_dir))
    return [entry for entry in entries if entry is not None]


def generate_presets(models_dir: Path, budget: HardwareBudget, preset_path: Path,
                     mtp_capable: set[str] | None = None, *,
                     live: HardwareBudget | None = None, residents: int = 1,
                     engine_tag: str | None = None) -> list[PresetEntry]:
    """Plan every staged model and write one INI. Refused models get no section (the picker
    surfaces the refusal from the returned entries)."""
    entries = plan_presets(models_dir, budget, mtp_capable, live=live, residents=residents,
                           engine_tag=engine_tag)
    write_presets(entries, preset_path)
    return entries


def write_presets(entries: list[PresetEntry], preset_path: Path) -> None:
    from utils import atomic_write_text

    atomic_write_text(preset_path, render_presets(entries), tmp_prefix=f".{preset_path.name}_", mode=0o600)
    logger.info("wrote %d preset sections to %s", sum(e.keys is not None for e in entries), preset_path)


def render_presets(entries: list[PresetEntry]) -> str:
    sections: list[str] = []
    for entry in entries:
        # INI comments preserve non-flag facts atomically with the launch policy.
        sections.append("# hermes-decision: " + json.dumps({
            "model_id": entry.model_id, "window": entry.window,
            "spilled": entry.spilled, "refusal": entry.refusal}) + "\n")
        if entry.keys is not None:
            body = "\n".join(f"{k} = {v}" for k, v in entry.keys.items())
            sections.append(f"[{entry.model_id}]\n{body}\n")
    return "\n".join(sections)


def read_preset_decisions(preset_path: Path | None = None) -> dict[str, PresetEntry]:
    """The launch decisions the running server was actually given, read back from the preset INI
    (the INI is the record — it's what spawned the children). Missing/unparseable -> {}."""
    import configparser

    if preset_path is None:
        from hermes_cli.local_runtime.binaries import runtimes_root

        preset_path = runtimes_root() / "presets.ini"
    out: dict[str, PresetEntry] = {}
    try:
        parser = configparser.ConfigParser(interpolation=None)
        text = preset_path.read_text(encoding="utf-8-sig")
        parser.read_string(text)
        recorded = {}
        for line in text.splitlines():
            if line.startswith("# hermes-decision: "):
                fact = json.loads(line.removeprefix("# hermes-decision: "))
                recorded[fact["model_id"]] = fact
                if fact.get("refusal"):
                    out[fact["model_id"]] = PresetEntry(**fact)
        for section in parser.sections():
            out[section] = PresetEntry(
                model_id=section, window=parser.getint(section, "ctx-size", fallback=0),
                spilled=recorded.get(section, {}).get("spilled", parser.has_option(section, "override-tensor")),
                keys=dict(parser[section]))
    except Exception as exc:
        logger.debug("preset read-back failed: %s", exc)
    return out
