"""Pet generation orchestration — the base-draft → hatch flow.

Two steps, mirroring the UX across every surface:

1. :func:`generate_base_drafts` — a handful of prompt-only "what should this pet
   look like" variants. Cheap; the user picks one (or retries for a fresh set).
2. :func:`hatch_pet` — takes the chosen base and generates 25 grounded one/two-
   pose assets, strictly isolates all 49 authored frames, mirrors the run cycle,
   composes the atlas, validates it, and writes the pet into the store.

Splitting it this way keeps the expensive quality pass on only the pet the user
actually keeps and gives each UI a natural preview/loading point.
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
# blocking on each in turn. Capped so we don't hammer the provider's rate limit
# (one cold call can still be slow).
_MAX_PARALLEL_GENERATIONS = 4
# A normal quality-first hatch creates 25 one/two-pose reference edits.  Retry a
# malformed segment once by default; unlike the former row path there is no
# best-effort slice that can leak a neighbour into a finished cell.
_POSE_GEN_ATTEMPTS = 2


@dataclass(frozen=True)
class HatchResult:
    """Outcome of a successful :func:`hatch_pet`."""

    slug: str
    display_name: str
    spritesheet: Path
    states: list[str]
    validation: dict


@dataclass(frozen=True)
class _PoseSegment:
    state: str
    row: int
    index: int
    phases: tuple[str, ...]


def _pose_segments() -> list[_PoseSegment]:
    """The exact 25 generation jobs needed for the 49 non-mirrored frames."""
    segments: list[_PoseSegment] = []
    for state, row, count in atlas.ROW_SPECS:
        if state == "running-left":
            continue
        phases = prompts.POSE_PHASES.get(state) or ()
        if len(phases) != count:
            raise GenerationError(
                f"pose phase contract for '{state}' has {len(phases)} entries; expected {count}"
            )
        for start in range(0, count, 2):
            segments.append(_PoseSegment(state, row, start // 2, phases[start : start + 2]))
    return segments


def _harden_transparency(path: Path) -> Path:
    """Key out any solid backdrop the provider painted; save as an RGBA PNG.

    ``background=transparent`` is requested on every call, but image models honor
    it inconsistently — some still paint a flat (often near-white) backdrop. We
    run the same chroma-key pass the row extractor uses so every base draft the
    user picks between (and the reference the poses are grounded on) is a clean
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
    reference_images: list[Path] | None = None,
    provider: SpriteProvider | None = None,
    on_draft: Callable[[int, Path], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    model: str | None = None,
    seed: int | None = None,
    concurrency: int = _MAX_PARALLEL_GENERATIONS,
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
    # A user reference image (e.g. their own pet) grounds every draft, so it
    # needs a reference-capable provider — same requirement as the pose passes.
    refs = reference_images or None
    sprite = provider or imagegen.resolve_provider(require_references=bool(refs), model=model)
    cancelled = is_cancelled or (lambda: False)

    # Each draft is its own one-shot generation, run concurrently so the user
    # waits for one image, not N. A single draft failing must not sink the set.
    # Each gets a distinct variation nudge so the options aren't near-duplicates.
    logger.info("pet generate: drafting %d base looks for %r (style=%s)", n, concept, style)

    def _one(index: int) -> tuple[int, Path | None, str | None]:
        if cancelled():
            return index, None, None
        t0 = time.monotonic()
        variation = prompts.BASE_VARIATIONS[index % len(prompts.BASE_VARIATIONS)]
        prompt = prompts.build_base_prompt(concept, style=style, variation=variation)
        try:
            kwargs = dict(n=1, reference_images=refs, provider=sprite, prefix="pet_base")
            if model:
                kwargs["model"] = model
            if seed is not None:
                kwargs["seed"] = seed + index
            out = imagegen.generate(prompt, **kwargs)
        except Exception as exc:  # noqa: BLE001 - tolerate a single failed draft
            logger.warning("pet generate: draft %d failed after %.1fs: %s", index, time.monotonic() - t0, exc)
            return index, None, str(exc)
        if not out:
            logger.warning("pet generate: draft %d produced no image", index)
            return index, None, "the image provider returned no image"
        if cancelled():
            for path in out:
                try:
                    path.unlink(missing_ok=True)
                except OSError as exc:
                    logger.debug("could not remove cancelled draft intermediate %s: %s", path, exc)
            return index, None, None
        source = out[0]
        hardened = _harden_transparency(source)
        if hardened != source:
            try:
                source.unlink(missing_ok=True)
            except OSError as exc:
                logger.debug("could not remove base draft intermediate %s: %s", source, exc)
        logger.info("pet generate: draft %d ready in %.1fs", index, time.monotonic() - t0)
        return index, hardened, None

    workers = max(1, min(n, int(concurrency), _MAX_PARALLEL_GENERATIONS))
    results: dict[int, Path] = {}
    errors: list[str] = []
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
            index, path, err = fut.result()
            if path is None:
                if err:
                    errors.append(err)
                continue
            results[index] = path
            if on_draft is not None:
                try:
                    on_draft(index, path)
                except Exception as exc:  # noqa: BLE001 - progress is best-effort
                    logger.debug("on_draft callback failed: %s", exc)

    drafts = [results[i] for i in sorted(results)]
    if not drafts and not cancelled():
        # Surface *why* — every draft failed for a reason (a content-policy refusal
        # on a name like "minion", a provider/auth error, …); the most common one
        # is the representative cause. Far more useful than "no usable drafts".
        raise GenerationError(_drafts_failed_reason(errors))
    return drafts


def _drafts_failed_reason(errors: list[str]) -> str:
    """The representative reason a draft round produced nothing, humanized."""
    if not errors:
        return "image generation produced no usable drafts"
    from collections import Counter

    return _humanize_image_error(Counter(errors).most_common(1)[0][0])


def _humanize_image_error(error: str) -> str:
    """Turn a raw provider error into a friendly, actionable sentence.

    The big one is moderation: image models refuse trademarked characters and
    real people (e.g. "minion"), which reads as an opaque 400 otherwise.
    """
    low = error.lower()
    if any(s in low for s in ("moderation_blocked", "safety system", "content policy", "content_policy")):
        return (
            "The image provider blocked this prompt — its safety filter rejects "
            "trademarked characters and real people. Try an original description."
        )
    if any(s in low for s in ("api key", "unauthorized", "401", "auth")):
        return "The image provider rejected the request — check your API key in Settings → Providers."
    if "rate limit" in low or "429" in low:
        return "The image provider is rate-limiting — wait a moment and try again."
    # Otherwise the first line, trimmed of the noisy provider envelope.
    return error.splitlines()[0].strip()[:200]


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
    model: str | None = None,
    seed: int | None = None,
    concurrency: int = _MAX_PARALLEL_GENERATIONS,
    pose_attempts: int = _POSE_GEN_ATTEMPTS,
    row_attempts: int | None = None,
) -> HatchResult:
    """Turn an approved base image into a full, installed Hermes pet.

    Generates reference-grounded one/two-pose assets, extracts frames, composes
    and validates the complete atlas, then registers it.  No partial or
    best-effort atlas is saved. Raises :class:`GenerationError` on failure.

    ``row_attempts`` is retained as a compatibility alias for callers predating
    the pair-pose pipeline; when supplied it overrides ``pose_attempts``.
    """
    base = Path(base_image)
    if not base.is_file():
        raise GenerationError(f"base image not found: {base}")

    sprite = provider or imagegen.resolve_provider(require_references=True, model=model)
    progress = on_progress or (lambda *_: None)
    cancelled = is_cancelled or (lambda: False)
    label = concept or display_name or slug

    segments = _pose_segments()
    total_segments = len(segments)
    attempts_limit = max(1, min(3, int(row_attempts if row_attempts is not None else pose_attempts)))
    logger.info(
        "pet hatch %r: generating %d strict pose segments (attempts=%d)",
        slug,
        total_segments,
        attempts_limit,
    )

    def _gen_segment(spec: _PoseSegment) -> tuple[_PoseSegment, list | None, int, str | None]:
        if cancelled():
            return spec, None, 0, "hatch cancelled"
        t0 = time.monotonic()
        last_exc: Exception | None = None
        retry_reason = ""
        for attempt in range(attempts_limit):
            if cancelled():
                return spec, None, attempt, "hatch cancelled"
            generated_path: Path | None = None
            try:
                kwargs = dict(
                    n=1,
                    reference_images=[base],
                    provider=sprite,
                    prefix=f"pet_pose_{spec.state}_{spec.index}",
                    aspect_ratio="landscape" if len(spec.phases) == 2 else "square",
                )
                if model:
                    kwargs["model"] = model
                if seed is not None:
                    kwargs["seed"] = seed + spec.row * 100 + spec.index * 10 + attempt
                generated = imagegen.generate(
                    prompts.build_pose_segment_prompt(
                        spec.state,
                        spec.phases,
                        label,
                        style=style,
                        retry_reason=retry_reason,
                    ),
                    **kwargs,
                )
                if not generated:
                    raise GenerationError("image provider returned no pose image")
                generated_path = generated[0]
                frames = atlas.extract_pose_segment(generated_path, len(spec.phases))
                logger.info(
                    "pet hatch %r: %s segment %d ready in %.1fs (attempt %d)",
                    slug,
                    spec.state,
                    spec.index + 1,
                    time.monotonic() - t0,
                    attempt + 1,
                )
                return spec, frames, attempt + 1, None
            except Exception as exc:  # noqa: BLE001 - bounded retry with QA feedback
                last_exc = exc
                retry_reason = str(exc).strip()[:300]
                logger.warning(
                    "pet hatch %r: %s segment %d attempt %d/%d failed: %s",
                    slug,
                    spec.state,
                    spec.index + 1,
                    attempt + 1,
                    attempts_limit,
                    exc,
                )
            finally:
                if generated_path is not None:
                    try:
                        generated_path.unlink(missing_ok=True)
                    except OSError as exc:
                        logger.debug("could not remove pose intermediate %s: %s", generated_path, exc)
        message = str(last_exc or "pose generation failed")
        return spec, None, attempts_limit, message

    workers = max(1, min(total_segments, int(concurrency), _MAX_PARALLEL_GENERATIONS))
    completed = 0
    segment_frames: dict[tuple[str, int], list] = {}
    segment_attempts: dict[tuple[str, int], int] = {}
    fatal: tuple[_PoseSegment, str] | None = None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_gen_segment, segment) for segment in segments]
        for future in as_completed(futures):
            if cancelled():
                for pending in futures:
                    pending.cancel()
                break
            spec, frames, used_attempts, error = future.result()
            if frames is None:
                fatal = (spec, error or "pose generation failed")
                for pending in futures:
                    pending.cancel()
                break
            segment_frames[(spec.state, spec.index)] = frames
            segment_attempts[(spec.state, spec.index)] = used_attempts
            completed += 1
            progress("pose", f"{spec.state}:{completed}:{total_segments}")

    if cancelled():
        raise GenerationError("hatch cancelled")
    if fatal is not None:
        spec, error = fatal
        raise GenerationError(
            f"{spec.state} pose segment {spec.index + 1} failed local quality checks: {error}"
        )

    frames_by_state: dict[str, list] = {}
    state_qa: dict[str, dict] = {}
    for state, _row, count in atlas.ROW_SPECS:
        if state == "running-left":
            continue
        state_segments = [segment for segment in segments if segment.state == state]
        frames = [
            frame
            for segment in sorted(state_segments, key=lambda item: item.index)
            for frame in segment_frames.get((state, segment.index), [])
        ]
        if len(frames) != count:
            raise GenerationError(f"{state} produced {len(frames)}/{count} accepted poses")
        frames_by_state[state] = frames
        state_qa[state] = {
            "pass": True,
            "segments": len(state_segments),
            "attempts": sum(segment_attempts[(state, segment.index)] for segment in state_segments),
        }

    progress("compose", "")
    logger.info("pet hatch %r: composing atlas from %d states", slug, len(frames_by_state))
    try:
        normalized = atlas.normalize_cells(frames_by_state)
    except ValueError as exc:
        raise GenerationError(str(exc)) from exc
    right = normalized.get("running-right")
    if not right:
        raise GenerationError("running-right did not produce normalized frames")
    normalized["running-left"] = atlas.mirror_frames(right)
    state_qa["running-left"] = {
        "pass": True,
        "segments": 0,
        "attempts": 0,
        "source": "mirrored locally from running-right",
    }
    sheet = atlas.compose_atlas(normalized)
    validation = atlas.validate_atlas(sheet)
    validation["states"] = state_qa
    validation["poseSegments"] = total_segments
    if not validation["ok"]:
        raise GenerationError("; ".join(validation["errors"]) or "atlas validation failed")
    filled_states = set(validation["filled_states"])
    expected_states = {state for state, _row, _count in atlas.ROW_SPECS}
    missing = sorted(expected_states - filled_states)
    if missing:
        raise GenerationError(f"missing animation row(s): {', '.join(missing)}")

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
