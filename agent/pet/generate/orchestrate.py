"""Pet generation orchestration — the base-draft → hatch flow.

:func:`generate_base_drafts` makes a few cheap prompt-only look variants the user
picks between; :func:`hatch_pet` grounds one row strip per state on the chosen
base, slices frames, composes + validates the atlas, and writes it to the store.
Splitting bounds cost (the ~8 row calls happen once, on the pet you keep) and
gives each UI a natural preview/loading point.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

from agent.pet.generate import atlas, imagegen, prompts
from agent.pet.generate.imagegen import GenerationError, SpriteProvider

logger = logging.getLogger(__name__)

# (event, detail) — e.g. ("row", "idle"), ("compose", ""), ("save", "<slug>").
ProgressFn = Callable[[str, str], None]

# Fan-out so a hatch (~8 rows) doesn't blow the client's RPC timeout; capped for provider rate limits.
_MAX_PARALLEL_GENERATIONS = 4
# Row attempts: early ones demand clean per-pose gutters, the last is lenient so a stubborn row still yields.
_ROW_GEN_ATTEMPTS = 3
_MIN_FILLED_STATES = 6
_REQUIRED_STATES = frozenset({"idle", "running-right", "waving"})

# (substrings, friendly message), first match wins. Moderation is the big one:
# image models refuse trademarked characters / real people as an opaque 400.
_IMAGE_ERROR_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("moderation_blocked", "safety system", "content policy", "content_policy"),
     "The image provider blocked this prompt — its safety filter rejects "
     "trademarked characters and real people. Try an original description."),
    (("api key", "unauthorized", "401", "auth"),
     "The image provider rejected the request — check your API key in Settings → Providers."),
    (("rate limit", "429"), "The image provider is rate-limiting — wait a moment and try again."),
)


@dataclass(frozen=True)
class HatchResult:
    """Outcome of a successful :func:`hatch_pet`."""

    slug: str
    display_name: str
    spritesheet: Path
    states: list[str]
    validation: dict


def _unlink_quietly(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)


def _harden_transparency(path: Path) -> Path:
    """Key out any solid backdrop the provider painted; save as an RGBA PNG.

    ``background=transparent`` is honored inconsistently, so every base draft gets
    the chroma-key pass. Best-effort: a decode failure leaves the original untouched.
    """
    try:
        keyed = atlas._clear_transparent_rgb(atlas.remove_background(atlas._load_rgba(path)))  # no halo on the dark UI
        # PNGs (any case) are hardened in place: with_suffix(".png") on ".PNG" would
        # name the same file on case-insensitive filesystems and the unlink below
        # would delete the hardened output.
        out = path if path.suffix.lower() == ".png" else path.with_suffix(".png")
        keyed.save(out, format="PNG")
        if out != path:
            _unlink_quietly(path)  # nothing else prunes cache/images outside the gateway loop
        return out
    except Exception as exc:  # noqa: BLE001 - cosmetic; fall back to the raw image
        logger.debug("base draft transparency hardening failed for %s: %s", path, exc)
        return path


def _run_parallel(fn, items, *, cancelled, on_cancel_log: str) -> Iterator:
    """Fan *fn* over *items* in a pool, yielding results in completion order.

    ``as_completed`` runs on the caller's thread, so result callbacks inherit the
    request's bound transport (workers don't). Once *cancelled* trips, queued work
    is cancelled and in-flight results dropped.
    """
    workers = max(1, min(len(items), _MAX_PARALLEL_GENERATIONS))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fn, item) for item in items]
        for fut in as_completed(futures):
            if cancelled():
                logger.info(on_cancel_log)
                for pending in futures:
                    pending.cancel()
                break
            yield fut.result()


def generate_base_drafts(
    concept: str,
    *,
    n: int = 4,
    style: str = "auto",
    reference_images: list[Path] | None = None,
    provider: SpriteProvider | None = None,
    on_draft: Callable[[int, Path], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> list[Path]:
    """Generate *n* candidate base looks for *concept* concurrently; returns image paths.

    Drafts are hardened to transparent cutouts; *on_draft(index, path)* fires as each
    finishes so UIs can stream previews. *is_cancelled* is polled cooperatively.
    """
    # A user reference image grounds every draft, so it needs a reference-capable provider.
    refs = reference_images or None
    sprite = provider or imagegen.resolve_provider(require_references=bool(refs))
    cancelled = is_cancelled or (lambda: False)
    logger.info("pet generate: drafting %d base looks for %r (style=%s)", n, concept, style)

    def _one(index: int) -> tuple[int, Path | None, str | None]:
        if cancelled():
            return index, None, None
        t0 = time.monotonic()
        variation = prompts.BASE_VARIATIONS[index % len(prompts.BASE_VARIATIONS)]  # distinct look per draft
        prompt = prompts.build_base_prompt(concept, style=style, variation=variation)
        try:
            out = imagegen.generate(prompt, n=1, reference_images=refs, provider=sprite, prefix="pet_base")
        except Exception as exc:  # noqa: BLE001 - tolerate a single failed draft
            logger.warning("pet generate: draft %d failed after %.1fs: %s", index, time.monotonic() - t0, exc)
            return index, None, str(exc)
        if not out:
            logger.warning("pet generate: draft %d produced no image", index)
            return index, None, "the image provider returned no image"
        logger.info("pet generate: draft %d ready in %.1fs", index, time.monotonic() - t0)
        return index, _harden_transparency(out[0]), None

    results: dict[int, Path] = {}
    errors: list[str] = []
    cancel_log = "pet generate: cancelled — dropping remaining drafts"
    for index, path, err in _run_parallel(_one, range(n), cancelled=cancelled, on_cancel_log=cancel_log):
        if path is None:
            errors += [err] if err else []
            continue
        results[index] = path
        if on_draft is not None:
            try:
                on_draft(index, path)
            except Exception as exc:  # noqa: BLE001 - progress is best-effort
                logger.debug("on_draft callback failed: %s", exc)
    drafts = [results[i] for i in sorted(results)]
    if not drafts and not cancelled():
        # Surface *why*: the most common failure reason is the representative cause.
        if not errors:
            raise GenerationError("image generation produced no usable drafts")
        raise GenerationError(_humanize_image_error(Counter(errors).most_common(1)[0][0]))
    return drafts


def _humanize_image_error(error: str) -> str:
    """Turn a raw provider error into a friendly, actionable sentence."""
    low = error.lower()
    hint = next((message for needles, message in _IMAGE_ERROR_HINTS if any(s in low for s in needles)), None)
    return hint or error.splitlines()[0].strip()[:200]  # first line, sans provider envelope


def _generate_row(spec: tuple[str, int, int], *, base: Path, label: str, style: str, slug: str, sprite, cancelled) -> tuple[str, list | None]:
    """Generate + slice one animation row, retrying up to ``_ROW_GEN_ATTEMPTS`` times.

    Self-healing: a roll whose poses touch (no gutters) slices badly, so
    ``components`` (raises on touching poses) drives regeneration and only the
    final attempt uses lenient ``auto`` slicing. Returns ``(state, None)`` when
    cancelled or every attempt failed.
    """
    state, _row, count = spec
    t0 = time.monotonic()
    last_exc: Exception | None = None
    for attempt in range(_ROW_GEN_ATTEMPTS):
        if cancelled():
            return state, None
        strips: list[Path] = []
        try:
            # Landscape: each frame gets real horizontal room and clean gutters.
            strips = imagegen.generate(
                prompts.build_row_prompt(state, count, label, style=style), n=1, reference_images=[base],
                provider=sprite, prefix=f"pet_row_{state}", aspect_ratio="landscape",
            )
            # fit=False keeps raw columns so normalize_cells registers the whole pet at once.
            method = "components" if attempt < _ROW_GEN_ATTEMPTS - 1 else "auto"
            frames = atlas.extract_strip_frames(strips[0], count, method=method, fit=False)
            logger.info("pet hatch %r: row %r ready in %.1fs (attempt %d)", slug, state, time.monotonic() - t0, attempt + 1)
            return state, frames
        except Exception as exc:  # noqa: BLE001 - retried; one bad row is tolerated
            last_exc = exc
            logger.warning("pet hatch %r: row %r attempt %d/%d failed: %s", slug, state, attempt + 1, _ROW_GEN_ATTEMPTS, exc)
        finally:
            # Strips are intermediates already decoded into memory; nothing
            # prunes cache/images outside the gateway loop, so drop them now.
            for strip in strips:
                _unlink_quietly(Path(strip))
    logger.warning("pet hatch %r: row %r gave up after %.1fs: %s", slug, state, time.monotonic() - t0, last_exc)
    return state, None


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

    Idle falls back to the base look so the pet always renders. Raises
    :class:`GenerationError` on failure. Once *is_cancelled* trips, aborts before
    composing/saving so a stopped hatch never writes a half-built pet.
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

    def _gen_row(spec: tuple[str, int, int]) -> tuple[str, list | None]:
        return _generate_row(spec, base=base, label=label, style=style, slug=slug, sprite=sprite, cancelled=cancelled)

    # running-left is mirrored from running-right (consistent, one fewer generation).
    generated_specs = [spec for spec in atlas.ROW_SPECS if spec[0] != "running-left"]
    cancel_log = f"pet hatch {slug!r}: cancelled — dropping remaining rows"
    done = 0
    for state, frames in _run_parallel(_gen_row, generated_specs, cancelled=cancelled, on_cancel_log=cancel_log):
        done += 1
        progress("row", f"{state}:{done}:{total_rows}")
        if frames:
            frames_by_state[state] = frames
    if cancelled():
        raise GenerationError("hatch cancelled")

    # Per-frame mirror preserves order/timing. A missing running-right is
    # rejected below: a pet without its canonical walk cycle is a failed hatch.
    right = frames_by_state.get("running-right")
    if right:
        done += 1
        progress("row", f"running-left:{done}:{total_rows}")
        frames_by_state["running-left"] = atlas.mirror_frames(right)
        logger.info("pet hatch %r: row 'running-left' mirrored from running-right", slug)
    else:
        logger.warning("pet hatch %r: no running-right to mirror; left walk left empty", slug)

    if not frames_by_state.get("idle"):  # the renderer's resting fallback — guarantee it
        progress("row", "idle-fallback")
        frames_by_state["idle"] = [atlas.single_frame(base, fit=False)]

    progress("compose", "")
    logger.info("pet hatch %r: composing atlas from %d states", slug, len(frames_by_state))
    # One shared scale + baseline across states so the pet never slides or pulses.
    sheet = atlas.compose_atlas(atlas.normalize_cells(frames_by_state))
    validation = atlas.validate_atlas(sheet)
    if not validation["ok"]:
        raise GenerationError("; ".join(validation["errors"]) or "atlas validation failed")
    filled_states = set(validation["filled_states"])
    if missing_required := sorted(_REQUIRED_STATES - filled_states):
        raise GenerationError(f"missing required animation row(s): {', '.join(missing_required)}")
    if len(filled_states) < _MIN_FILLED_STATES:
        raise GenerationError(f"only {len(filled_states)}/{total_rows} animation rows were usable; regenerate")

    from agent.pet import store

    progress("save", slug)
    logger.info("pet hatch %r: saving pet", slug)
    pet = store.register_local_pet(sheet, slug=slug, display_name=display_name or slug, description=description)
    return HatchResult(pet.slug, pet.display_name, pet.spritesheet, validation["filled_states"], validation)
