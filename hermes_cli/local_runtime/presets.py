"""Per-model preset generation (--models-preset INI) — the router-side carrier for context-policy
launch decisions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path

from hermes_cli.local_runtime.context_policy import (
    RUNTIME_OVERHEAD_BYTES, WindowDecision, initial_window, launch_args, ub_logits_bytes)
from hermes_cli.local_runtime.estimator import (
    HardwareBudget, ModelProfile, PhysicsRefusal, ctx_bytes, profile_from_gguf)
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


def _asset_path(asset) -> "Path | None":
    """On-disk path of a catalog companion asset, or None when it isn't downloaded."""
    from hermes_cli.local_runtime.bootstrap import assets_dir

    if asset is None:
        return None
    path = assets_dir() / asset.local_name
    return path if path.exists() else None


def _choose_mtp_posture(profile: ModelProfile, budget: HardwareBudget,
                        fixed_overhead: int) -> tuple[bool, int]:
    """(mtp_prefill, logits_bytes) for an MTP model — window first, prefill second.

    Price the launch under both postures and keep whichever grants the larger window: the stacked
    posture's bigger compute buffer buys ~3x short-prompt prefill but costs ~2 GiB that would
    otherwise be window (measured at 256K the ub512 posture still prefills at 2.7K tok/s), so
    never trade context away for prefill. Same window -> stacked.
    """
    plain_logits = ub_logits_bytes(profile.n_vocab, mtp_capable=True)
    stacked_logits = ub_logits_bytes(profile.n_vocab, mtp_capable=True, mtp_prefill=True)
    stacked = initial_window(profile, budget, overhead_bytes=fixed_overhead + stacked_logits)
    plain = initial_window(profile, budget, overhead_bytes=fixed_overhead + plain_logits)
    if (not isinstance(stacked, PhysicsRefusal) and not stacked.spilled
            and (isinstance(plain, PhysicsRefusal) or stacked.window >= plain.window)):
        return True, stacked_logits
    return False, plain_logits


def _restore_grown_window(model_id: str, profile: ModelProfile, budget: HardwareBudget,
                          decision: WindowDecision, overhead: int) -> WindowDecision:
    """Session growth (growth.py): a persisted override lifts the launch window to where the ladder
    last grew it — capped at native, and only when physics still clears the bigger window on THIS
    boot's budget (a smaller-VRAM day re-fits honestly back down)."""
    try:
        from hermes_cli.local_runtime.growth import load_window_overrides

        override = load_window_overrides().get(model_id)
        native = profile.n_ctx_train or decision.window
        if override and override > decision.window:
            target = min(int(override), native)
            kv = ctx_bytes(profile, target)
            need = profile.weights_bytes + kv + overhead
            if need <= budget.usable_vram_bytes + budget.ram_available_bytes:
                return WindowDecision(
                    window=target, spill_bytes=max(0, need - budget.usable_vram_bytes),
                    kv_on_gpu=kv <= budget.usable_vram_bytes,
                    reasons=[f"grown window restored ({target // 1024}K)"])
    except Exception as exc:  # noqa: BLE001 — overrides are advisory
        logger.debug("window override skipped for %s: %s", model_id, exc)
    return decision


def _preset_for(gguf: Path, budget: HardwareBudget,
                mtp_capable: set[str]) -> PresetEntry | None:
    """The launch decision for one staged model, or None when its header is unreadable."""
    from hermes_cli.local_runtime.catalog import entry_for_model

    model_id = model_id_from_stem(gguf.stem)
    try:
        header = read_gguf_header(gguf)
        profile = profile_from_gguf(header)
    except (ValueError, OSError) as exc:
        logger.warning("preset skip %s: %s", gguf.name, exc)
        return None
    entry = entry_for_model(model_id)
    is_mtp = entry.mtp if entry is not None else model_id in mtp_capable
    if is_mtp and profile.kv_scale == 1.0:
        # Header-derived profiles don't know about MTP's draft context; apply the calibrated KV
        # multiplier so the launch fit prices what the server will actually allocate.
        profile = replace(profile, kv_scale=1.2)
    mmproj_path = _asset_path(entry.mmproj) if entry is not None else None
    # Overhead beyond weights+KV: runtime buffers, the vision projector when present, and the
    # logits buffers of whichever microbatch/MTP posture launch_args will choose — flag and price
    # decided together, from the same facts.
    fixed_overhead = RUNTIME_OVERHEAD_BYTES + (
        entry.mmproj.size_bytes if entry is not None and mmproj_path is not None else 0)
    if is_mtp:
        mtp_prefill, logits_bytes = _choose_mtp_posture(profile, budget, fixed_overhead)
    else:
        mtp_prefill, logits_bytes = False, ub_logits_bytes(profile.n_vocab, mtp_capable=False)
    overhead = fixed_overhead + logits_bytes
    decision = initial_window(profile, budget, overhead_bytes=overhead)
    if isinstance(decision, PhysicsRefusal):
        return PresetEntry(model_id=model_id, window=0, spilled=False, refusal=decision.message)
    decision = _restore_grown_window(model_id, profile, budget, decision, overhead)

    # The launch flags MUST match the pricing above (same entry/is_mtp/posture).
    keys = _args_to_keys(launch_args(
        profile, decision, mtp_capable=is_mtp, uma=budget.uma, mtp_prefill=mtp_prefill,
        mtp_draft_depth=entry.mtp_draft_depth if entry is not None else 3))
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
        if mmproj_path is not None:
            keys["mmproj"] = str(mmproj_path)
        draft_path = _asset_path(entry.draft) if decision.spilled else None
        if draft_path is not None:
            keys["model-draft"] = str(draft_path)
            keys["spec-type"] = "draft-dspark"
            # Unsloth's measured cliff: acceptance 83% at 2-3 drafts, collapses at 4.
            keys["spec-draft-n-max"] = "3"
    return PresetEntry(model_id=model_id, window=decision.window,
                       spilled=decision.spilled, keys=keys)


def generate_presets(models_dir: Path, budget: HardwareBudget, preset_path: Path,
                     mtp_capable: set[str] | None = None) -> list[PresetEntry]:
    """Walk the staged models, run the launch decision per model, and write one INI. Refused
    models get no section (the picker surfaces the refusal from the returned entries)."""
    from hermes_cli.local_runtime.bootstrap import staged_in

    entries: list[PresetEntry] = []
    sections: list[str] = []
    for gguf in staged_in(models_dir, require_complete=False):
        entry = _preset_for(gguf, budget, mtp_capable or set())
        if entry is None:
            continue
        entries.append(entry)
        if entry.keys is not None:
            body = "\n".join(f"{k} = {v}" for k, v in entry.keys.items())
            sections.append(f"[{entry.model_id}]\n{body}\n")

    preset_path.parent.mkdir(parents=True, exist_ok=True)
    preset_path.write_text("\n".join(sections), encoding="utf-8")
    logger.info("wrote %d preset sections to %s", len(sections), preset_path)
    return entries


def read_preset_decisions(preset_path: Path | None = None) -> dict[str, PresetEntry]:
    """The launch decisions the running server was actually given, read back from the preset INI
    (the INI is the record — it's what spawned the children). Missing/unparseable -> {}."""
    import configparser

    if preset_path is None:
        from hermes_cli.local_runtime.binaries import runtimes_root

        preset_path = runtimes_root() / "presets.ini"
    out: dict[str, PresetEntry] = {}
    try:
        parser = configparser.ConfigParser()
        parser.read(preset_path, encoding="utf-8")
        for section in parser.sections():
            out[section] = PresetEntry(
                model_id=section, window=parser.getint(section, "ctx-size", fallback=0),
                spilled=parser.has_option(section, "override-tensor"))
    except Exception as exc:  # noqa: BLE001
        logger.debug("preset read-back failed: %s", exc)
    return out
