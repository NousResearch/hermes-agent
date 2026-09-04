"""Per-model preset generation (--models-preset INI) — the router-side
carrier for context-policy launch decisions.

The INI shape is what the router itself generates per child: a
[model-id] section whose keys are long-form
llama-server flag names without the leading dashes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from hermes_cli.local_runtime.context_policy import (
    RUNTIME_OVERHEAD_BYTES,
    WindowDecision,
    initial_window,
    launch_args,
    ub_logits_bytes,
)
from hermes_cli.local_runtime.estimator import (
    HardwareBudget,
    PhysicsRefusal,
    profile_from_gguf,
)
from hermes_cli.local_runtime.gguf import read_gguf_header

logger = logging.getLogger(__name__)

# args list -> INI keys. Flags the policy owns; everything else stays out
# of the preset (recipe sampling defaults merge in a later pass).
_FLAG_TO_KEY = {
    "-c": "ctx-size",
    "-b": "batch-size",
    "-ub": "ubatch-size",
    "-ctk": "cache-type-k",
    "-ctv": "cache-type-v",
    "-fa": "flash-attn",
    "-ot": "override-tensor",
    "--spec-type": "spec-type",
    "--spec-draft-n-max": "spec-draft-n-max",
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
        flag = args[i]
        key = _FLAG_TO_KEY.get(flag)
        if key is None:
            i += 1
            continue
        keys[key] = args[i + 1]
        i += 2
    return keys


def generate_presets(models_dir: Path, budget: HardwareBudget,
                     preset_path: Path,
                     mtp_capable: set[str] | None = None) -> list[PresetEntry]:
    """Walk the staged models, run the launch decision per model, and
    write one INI. Refused models get no section (the router simply won't
    have policy for them; the picker surfaces the refusal + smaller-quant
    suggestion from the returned entries).

    Catalog-declared companions merge in here: sampling defaults (policy
    keys always win), the vision projector when present, and a spec-decode
    draft model iff the decision spilled — the rule: speculative
    decode is a spill amplifier, so a resident draft accelerates a spilled
    main model; a zero-spill model doesn't pay the draft's memory."""
    from hermes_cli.local_runtime.bootstrap import assets_dir
    from hermes_cli.local_runtime.catalog import find_entry_for_model

    entries: list[PresetEntry] = []
    sections: list[str] = []
    for gguf in _staged_in(models_dir):
        model_id = _strip_part(gguf.stem)
        try:
            header = read_gguf_header(gguf)
            profile = profile_from_gguf(header)
        except (ValueError, OSError) as exc:
            logger.warning("preset skip %s: %s", gguf.name, exc)
            continue
        # Overhead beyond weights+KV: runtime buffers, the vision projector
        # when this model ships one, and the logits buffers of whichever
        # microbatch/MTP posture launch_args will choose — flag and price
        # decided together, from the same facts.
        hit = find_entry_for_model(model_id)
        entry = hit[0] if hit is not None else None
        is_mtp = (entry.mtp if entry is not None
                  else model_id in (mtp_capable or set()))
        if is_mtp and profile.kv_scale == 1.0:
            # Header-derived profiles don't know about MTP's draft
            # context; apply the calibrated KV multiplier here so the
            # launch fit prices what the server will actually allocate.
            import dataclasses

            profile = dataclasses.replace(profile, kv_scale=1.2)
        mmproj_bytes = 0
        if entry is not None and entry.mmproj is not None:
            mmproj_path = assets_dir() / entry.mmproj.local_name
            if mmproj_path.exists():
                mmproj_bytes = entry.mmproj.size_bytes
        # MTP posture ladder — window first, prefill second: price the
        # launch under both postures and keep whichever grants the larger
        # window (the stacked posture's bigger compute buffer buys ~3x
        # short-prompt prefill but costs ~2 GiB that would otherwise be
        # window; measured at 256K the ub512 posture still prefills at
        # 2.7K tok/s, so window wins ties only one way: never trade
        # context away for prefill). Same window -> stacked.
        mtp_prefill = False
        logits_bytes = ub_logits_bytes(profile.n_vocab, mtp_capable=is_mtp)
        if is_mtp:
            stacked_logits = ub_logits_bytes(profile.n_vocab, mtp_capable=True,
                                             mtp_prefill=True)
            stacked_probe = initial_window(
                profile, budget,
                overhead_bytes=(RUNTIME_OVERHEAD_BYTES + mmproj_bytes
                                + stacked_logits))
            plain_probe = initial_window(
                profile, budget,
                overhead_bytes=(RUNTIME_OVERHEAD_BYTES + mmproj_bytes
                                + logits_bytes))
            if (not isinstance(stacked_probe, PhysicsRefusal)
                    and not stacked_probe.spilled
                    and (isinstance(plain_probe, PhysicsRefusal)
                         or stacked_probe.window >= plain_probe.window)):
                mtp_prefill = True
                logits_bytes = stacked_logits
        decision = initial_window(
            profile, budget,
            overhead_bytes=RUNTIME_OVERHEAD_BYTES + mmproj_bytes + logits_bytes)
        if isinstance(decision, PhysicsRefusal):
            entries.append(PresetEntry(model_id=model_id, window=0,
                                       spilled=False, refusal=decision.message))
            continue

        # Session growth (growth.py): a persisted override lifts the launch
        # window to where the ladder last grew it — capped at native, and
        # only when physics still clears the bigger window on THIS boot's
        # budget (a smaller-VRAM day re-fits honestly back down).
        try:
            from hermes_cli.local_runtime.estimator import ctx_bytes
            from hermes_cli.local_runtime.growth import load_window_overrides

            override = load_window_overrides().get(model_id)
            native = profile.n_ctx_train or decision.window
            if override and override > decision.window:
                target = min(int(override), native)
                kv = ctx_bytes(profile, target)
                need = (profile.weights_bytes + kv
                        + RUNTIME_OVERHEAD_BYTES + mmproj_bytes + logits_bytes)
                if need <= budget.usable_vram_bytes + budget.ram_available_bytes:
                    spill = max(0, need - budget.usable_vram_bytes)
                    decision = WindowDecision(
                        window=target, spill_bytes=spill,
                        kv_on_gpu=kv <= budget.usable_vram_bytes,
                        reasons=[f"grown window restored ({target // 1024}K)"])
        except Exception as exc:  # noqa: BLE001 — overrides are advisory
            logger.debug("window override skipped for %s: %s", model_id, exc)

        # (entry and is_mtp resolved above, where the overhead was priced —
        # the launch flags below MUST match that pricing.)
        args = launch_args(profile, decision, mtp_capable=is_mtp,
                           mtp_draft_depth=(entry.mtp_draft_depth
                                            if entry is not None else 3),
                           uma=budget.uma, mtp_prefill=mtp_prefill)
        keys = _args_to_keys(args)

        if entry is not None and is_mtp:
            # Integrated-MTP targets sample on the backend, and so does
            # the draft (pairing validated against the vendor's published
            # llama.cpp recipes for these models).
            keys["backend-sampling"] = "on"
            keys["spec-draft-backend-sampling"] = "on"

        # Sampling deference ladder, under the policy keys (policy wins
        # on clash). The GGUF's own general.sampling.* metadata is the
        # publisher's recommendation — it arrives with the file, updates
        # with every re-upload, and covers models the catalog has never
        # heard of. Catalog sampling applies only where the file is
        # silent; a model carrying neither runs llama.cpp defaults.
        for k, v in header.sampling_defaults.items():
            keys.setdefault(k, v)
        if entry is not None:
            for k, v in (entry.sampling or {}).items():
                keys.setdefault(k, v)
            if entry.mmproj is not None:
                mmproj_path = assets_dir() / entry.mmproj.local_name
                if mmproj_path.exists():
                    keys["mmproj"] = str(mmproj_path)
            if entry.draft is not None and decision.spilled:
                draft_path = assets_dir() / entry.draft.local_name
                if draft_path.exists():
                    keys["model-draft"] = str(draft_path)
                    keys["spec-type"] = "draft-dspark"
                    # Unsloth's measured cliff: acceptance 83% at 2-3
                    # drafts, collapses at 4.
                    keys["spec-draft-n-max"] = "3"

        entries.append(PresetEntry(model_id=model_id, window=decision.window,
                                   spilled=decision.spilled, keys=keys))
        body = "\n".join(f"{k} = {v}" for k, v in keys.items())
        sections.append(f"[{model_id}]\n{body}\n")

    preset_path.parent.mkdir(parents=True, exist_ok=True)
    preset_path.write_text("\n".join(sections), encoding="utf-8")
    logger.info("wrote %d preset sections to %s", len(sections), preset_path)
    return entries


def read_preset_decisions(preset_path: Path | None = None) -> dict[str, PresetEntry]:
    """The launch decisions the running server was actually given, read
    back from the preset INI (the INI is the record — it's what spawned
    the children). Missing/unparseable file returns {}."""
    import configparser

    if preset_path is None:
        from hermes_cli.local_runtime.binaries import runtimes_root

        preset_path = runtimes_root() / "presets.ini"
    out: dict[str, PresetEntry] = {}
    try:
        parser = configparser.ConfigParser()
        parser.read(preset_path, encoding="utf-8")
        for section in parser.sections():
            window = parser.getint(section, "ctx-size", fallback=0)
            spilled = parser.has_option(section, "override-tensor")
            out[section] = PresetEntry(model_id=section, window=window,
                                       spilled=spilled)
    except Exception as exc:  # noqa: BLE001
        logger.debug("preset read-back failed: %s", exc)
    return out


def _strip_part(stem: str) -> str:
    import re

    return re.sub(r"-\d{5}-of-\d{5}$", "", stem)


def _staged_in(models_dir: Path) -> "list[Path]":
    """Servable models in an arbitrary directory (split first-parts only) —
    the validation harness points at non-default dirs."""
    import re

    part = re.compile(r"-(\d{5})-of-\d{5}\.gguf$")
    out = []
    for p in sorted(models_dir.glob("*.gguf")):
        m = part.search(p.name)
        if m and m.group(1) != "00001":
            continue
        out.append(p)
    return out
