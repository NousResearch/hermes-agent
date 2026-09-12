"""Per-model preset generation (--models-preset INI) — the router-side carrier for context-policy
launch decisions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from hermes_cli.local_runtime.context_policy import (
    RUNTIME_OVERHEAD_BYTES, launch_args, plan_launch, ub_logits_bytes)
from hermes_cli.local_runtime.estimator import (
    HardwareBudget, PhysicsRefusal, ctx_bytes, footprint_bytes, profile_from_gguf)
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


def preset_for_model(gguf: Path, budget: HardwareBudget,
                     mtp_capable: set[str], *, requested_window: int | None = None) -> PresetEntry | None:
    """The launch decision for one staged model, or None when its header is unreadable."""
    from hermes_cli.local_runtime.catalog import entry_for_model
    from hermes_cli.local_runtime.growth import load_window_overrides

    model_id = model_id_from_stem(gguf.stem)
    try:
        header = read_gguf_header(gguf)
        profile = profile_from_gguf(header)
    except (ValueError, OSError) as exc:
        logger.warning("preset skip %s: %s", gguf.name, exc)
        return None
    entry = entry_for_model(model_id)
    is_mtp = entry.mtp if entry is not None else model_id in mtp_capable

    mmproj_path = _asset_path(entry.mmproj) if entry is not None else None
    fixed_overhead = RUNTIME_OVERHEAD_BYTES + (
        entry.mmproj.size_bytes if entry is not None and mmproj_path is not None else 0)
    plan = plan_launch(profile, budget, mtp_capable=is_mtp, fixed_overhead=fixed_overhead,
                       requested_window=(load_window_overrides().get(model_id)
                                         if requested_window is None else requested_window))
    decision = plan.decision
    if isinstance(decision, PhysicsRefusal):
        return PresetEntry(model_id=model_id, window=0, spilled=False, refusal=decision.message)

    # Router discovery is preset-only: refused files must never autoload with stock fit.
    keys = _args_to_keys(launch_args(
        profile, decision, mtp_capable=is_mtp, uma=budget.uma, mtp_prefill=plan.mtp_prefill,
        mtp_draft_depth=entry.mtp_draft_depth if entry is not None else 3))
    keys["model"] = str(gguf)
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
        if draft_path is not None and _draft_fits(draft_path, profile, budget, decision.window, plan.overhead_bytes):
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
    for gguf in staged_in(models_dir):
        entry = preset_for_model(gguf, budget, mtp_capable or set())
        if entry is None:
            continue
        entries.append(entry)
        # INI comments preserve non-flag facts atomically with the launch policy.
        sections.append("# hermes-decision: " + json.dumps({
            "model_id": entry.model_id, "window": entry.window,
            "spilled": entry.spilled, "refusal": entry.refusal}) + "\n")
        if entry.keys is not None:
            body = "\n".join(f"{k} = {v}" for k, v in entry.keys.items())
            sections.append(f"[{entry.model_id}]\n{body}\n")

    preset_path.parent.mkdir(parents=True, exist_ok=True)
    import os
    import tempfile

    fd, tmp = tempfile.mkstemp(prefix=preset_path.name, suffix=".tmp", dir=preset_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write("\n".join(sections))
        os.replace(tmp, preset_path)
    finally:
        Path(tmp).unlink(missing_ok=True)
    logger.info("wrote %d preset sections to %s", sum(e.keys is not None for e in entries), preset_path)
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
        parser = configparser.ConfigParser(interpolation=None)
        text = preset_path.read_text(encoding="utf-8")
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
    except Exception as exc:  # noqa: BLE001
        logger.debug("preset read-back failed: %s", exc)
    return out
