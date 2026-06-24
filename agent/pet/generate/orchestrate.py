"""Pet generation orchestration — the base-draft → hatch flow.

Two steps, mirroring the UX across every surface:

1. :func:`generate_base_drafts` — a handful of prompt-only "what should this pet
   look like" variants. Cheap; the user picks one (or retries for a fresh set).
2. :func:`hatch_pet` — takes the chosen base and generates one grounded row
   strip per Hermes state, slices each into frames, composes the atlas, validates
   it, and writes the pet into the store.

Splitting it this way bounds cost (4 cheap base calls per round; the ~6 row
calls happen once, on the pet you actually keep) and gives each UI a natural
preview/loading point.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from agent.pet.generate import atlas, imagegen, prompts
from agent.pet.generate.imagegen import GenerationError, SpriteProvider

logger = logging.getLogger(__name__)

# (event, detail) — e.g. ("row", "idle"), ("compose", ""), ("save", "<slug>").
ProgressFn = Callable[[str, str], None]

# Image generations are independent network calls, so we fan them out instead of
# blocking on each in turn — a hatch is ~8 row calls that would otherwise run
# back-to-back and routinely blow past the client's RPC timeout. Capped so we
# don't hammer the provider's rate limit (one cold call can still be slow).
_MAX_PARALLEL_GENERATIONS = 4
_MIN_FILLED_STATES = 6
_REQUIRED_STATES = frozenset({"idle", "running-right", "waving"})


@dataclass(frozen=True)
class HatchResult:
    """Outcome of a successful :func:`hatch_pet`."""

    slug: str
    display_name: str
    spritesheet: Path
    states: list[str]
    validation: dict


def _harden_transparency(path: Path) -> Path:
    """Key out any solid backdrop the provider painted; save as an RGBA PNG.

    ``background=transparent`` is requested on every call, but image models honor
    it inconsistently — some still paint a flat (often near-white) backdrop. We
    run the same chroma-key pass the row extractor uses so every base draft the
    user picks between (and the reference the rows are grounded on) is a clean
    cutout. Best-effort: a decode failure leaves the original untouched.
    """
    from PIL import Image

    try:
        with Image.open(path) as opened:
            keyed = atlas.remove_background(opened.convert("RGBA"))
        # Zero the RGB of any leftover semi-transparent edge pixels so a keyed
        # draft has no colored halo when composited on the dark UI.
        keyed = atlas._clear_transparent_rgb(keyed)
        out = path.with_suffix(".png")
        keyed.save(out, format="PNG")
        return out
    except Exception as exc:  # noqa: BLE001 - cosmetic; fall back to the raw image
        logger.debug("base draft transparency hardening failed for %s: %s", path, exc)
        return path


def generate_base_drafts(
    concept: str,
    *,
    n: int = 4,
    style: str = "auto",
    provider: SpriteProvider | None = None,
    on_draft: Callable[[int, Path], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> list[Path]:
    """Generate *n* candidate base looks for *concept*; returns image paths.

    Each draft is hardened to a transparent cutout (see :func:`_harden_transparency`).
    Drafts are generated concurrently and *on_draft(index, path)* fires as each
    one finishes (not at the end) so callers can stream previews to the UI
    instead of leaving it blank until the whole batch is done.

    *is_cancelled*, when supplied, is polled cooperatively: a draft that hasn't
    started yet is skipped, and once it trips we stop staging/streaming further
    drafts and cancel any queued work (already-in-flight provider calls can't be
    hard-killed, but their results are dropped).
    """
    sprite = provider or imagegen.resolve_provider(require_references=False)
    cancelled = is_cancelled or (lambda: False)

    # Each draft is its own one-shot generation, run concurrently so the user
    # waits for one image, not N. A single draft failing must not sink the set.
    # Each gets a distinct variation nudge so the options aren't near-duplicates.
    logger.info("pet generate: drafting %d base looks for %r (style=%s)", n, concept, style)

    def _one(index: int) -> tuple[int, Path | None]:
        if cancelled():
            return index, None
        t0 = time.monotonic()
        variation = prompts.BASE_VARIATIONS[index % len(prompts.BASE_VARIATIONS)]
        prompt = prompts.build_base_prompt(concept, style=style, variation=variation)
        try:
            out = imagegen.generate(prompt, n=1, provider=sprite, prefix="pet_base")
        except Exception as exc:  # noqa: BLE001 - tolerate a single failed draft
            logger.warning("pet generate: draft %d failed after %.1fs: %s", index, time.monotonic() - t0, exc)
            return index, None
        if not out:
            logger.warning("pet generate: draft %d produced no image", index)
            return index, None
        logger.info("pet generate: draft %d ready in %.1fs", index, time.monotonic() - t0)
        return index, _harden_transparency(out[0])

    workers = max(1, min(n, _MAX_PARALLEL_GENERATIONS))
    results: dict[int, Path] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one, i) for i in range(n)]
        # as_completed runs in *this* (the caller's) thread, so on_draft — and any
        # gateway event it emits — inherits the request's bound transport, unlike
        # the worker threads above.
        for fut in as_completed(futures):
            if cancelled():
                logger.info("pet generate: cancelled — dropping remaining drafts")
                for pending in futures:
                    pending.cancel()
                break
            index, path = fut.result()
            if path is None:
                continue
            results[index] = path
            if on_draft is not None:
                try:
                    on_draft(index, path)
                except Exception as exc:  # noqa: BLE001 - progress is best-effort
                    logger.debug("on_draft callback failed: %s", exc)

    drafts = [results[i] for i in sorted(results)]
    if not drafts and not cancelled():
        raise GenerationError("image generation produced no usable drafts")
    return drafts


def hatch_pet(
    *,
    base_image: str | Path,
    slug: str,
    display_name: str = "",
    description: str = "",
    concept: str = "",
    style: str = "auto",
    on_progress: ProgressFn | None = None,
    provider: SpriteProvider | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> HatchResult:
    """Turn an approved base image into a full, installed Hermes pet.

    Generates a grounded row strip per state, extracts frames, composes +
    validates the atlas, and registers it. The idle row falls back to the base
    look so the pet always renders. Raises :class:`GenerationError` on failure.

    *is_cancelled*, when supplied, is polled cooperatively: rows that haven't
    started are skipped, queued rows are cancelled, and once every row is done we
    abort (raising :class:`GenerationError`) before composing/saving so a stopped
    hatch never writes a half-built pet.
    """
    base = Path(base_image)
    if not base.is_file():
        raise GenerationError(f"base image not found: {base}")

    sprite = provider or imagegen.resolve_provider(require_references=True)
    progress = on_progress or (lambda *_: None)
    cancelled = is_cancelled or (lambda: False)
    label = concept or display_name or slug

    frames_by_state: dict[str, list] = {}
    total_rows = len(atlas.ROW_SPECS)
    logger.info("pet hatch %r: generating %d animation rows", slug, total_rows)

    # Generate every state's row strip concurrently — they're independent
    # grounded calls, so the hatch waits for the slowest row, not their sum. A
    # single row failing is tolerated (idle is guaranteed below).
    def _gen_row(spec: tuple[str, int, int]) -> tuple[str, list | None]:
        state, _row, count = spec
        if cancelled():
            return state, None
        t0 = time.monotonic()
        try:
            strips = imagegen.generate(
                prompts.build_row_prompt(state, count, label, style=style),
                n=1,
                reference_images=[base],
                provider=sprite,
                prefix=f"pet_row_{state}",
            )
            # One image call per row (the expensive part). ``auto`` validates by
            # connected components with an equal-slot fallback; raw (fit=False) so
            # normalize_cells registers the whole pet at once. We deliberately do
            # NOT re-generate a ragged row — the registration pass salvages it far
            # cheaper than another image-model round-trip.
            frames = atlas.extract_strip_frames(strips[0], count, method="auto", fit=False)
            logger.info("pet hatch %r: row %r ready in %.1fs", slug, state, time.monotonic() - t0)
            return state, frames
        except Exception as exc:  # noqa: BLE001 - one bad row is tolerated (idle guaranteed)
            logger.warning("pet hatch %r: row %r failed after %.1fs: %s", slug, state, time.monotonic() - t0, exc)
            return state, None

    # running-left is derived by mirroring running-right (guaranteed-consistent
    # and one fewer generation), so we don't generate it directly.
    generated_specs = [spec for spec in atlas.ROW_SPECS if spec[0] != "running-left"]

    workers = max(1, min(len(generated_specs), _MAX_PARALLEL_GENERATIONS))
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_gen_row, spec) for spec in generated_specs]
        # as_completed runs on the caller (request) thread, so progress events
        # emitted here inherit the request transport — unlike the worker threads.
        for fut in as_completed(futures):
            if cancelled():
                logger.info("pet hatch %r: cancelled — dropping remaining rows", slug)
                for pending in futures:
                    pending.cancel()
                break
            state, frames = fut.result()
            done += 1
            progress("row", f"{state}:{done}:{total_rows}")
            if frames:
                frames_by_state[state] = frames

    if cancelled():
        raise GenerationError("hatch cancelled")

    # Derive running-left from the approved running-right row (per-frame mirror,
    # preserving order/timing). Missing running-right is rejected below; a pet
    # without its canonical walk cycle is a failed hatch, not a shippable mascot.
    right = frames_by_state.get("running-right")
    if right:
        done += 1
        progress("row", f"running-left:{done}:{total_rows}")
        frames_by_state["running-left"] = atlas.mirror_frames(right)
        logger.info("pet hatch %r: row 'running-left' mirrored from running-right", slug)
    else:
        logger.warning("pet hatch %r: no running-right to mirror; left walk left empty", slug)

    # Idle is the resting state the renderer falls back to — guarantee it.
    if not frames_by_state.get("idle"):
        progress("row", "idle-fallback")
        frames_by_state["idle"] = [atlas.single_frame(base, fit=False)]

    progress("compose", "")
    logger.info("pet hatch %r: composing atlas from %d states", slug, len(frames_by_state))
    # One shared scale + baseline across every state so the pet never slides or
    # pulses size between frames; compose just packs the normalized cells.
    sheet = atlas.compose_atlas(atlas.normalize_cells(frames_by_state))
    validation = atlas.validate_atlas(sheet)
    if not validation["ok"]:
        raise GenerationError("; ".join(validation["errors"]) or "atlas validation failed")
    filled_states = set(validation["filled_states"])
    missing_required = sorted(_REQUIRED_STATES - filled_states)
    if missing_required:
        raise GenerationError(f"missing required animation row(s): {', '.join(missing_required)}")
    if len(filled_states) < _MIN_FILLED_STATES:
        raise GenerationError(
            f"only {len(filled_states)}/{len(atlas.ROW_SPECS)} animation rows were usable; regenerate"
        )

    from agent.pet import store

    progress("save", slug)
    logger.info("pet hatch %r: saving pet", slug)
    pet = store.register_local_pet(
        sheet,
        slug=slug,
        display_name=display_name or slug,
        description=description,
    )
    return HatchResult(
        slug=pet.slug,
        display_name=pet.display_name,
        spritesheet=pet.spritesheet,
        states=validation["filled_states"],
        validation=validation,
    )
