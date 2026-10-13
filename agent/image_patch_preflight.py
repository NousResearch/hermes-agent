"""Provider-bound image patch-budget preflight.

When explicitly enabled, this module protects image-capable provider requests
from per-image patch ceilings unless the selected model is configured as an
exclusion. It rewrites only the per-request message copy, leaving the persisted
conversation and source files untouched.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from fnmatch import fnmatchcase
import io
import logging
from math import ceil
import re
from typing import Any, Callable, Optional, TypeVar, cast


logger = logging.getLogger(__name__)

# OpenAI vision inputs are tokenized as 32px x 32px patches for patch-based
# models: https://platform.openai.com/docs/guides/images-vision#calculating-costs
OPENAI_IMAGE_PATCH_SIZE = 32
OPENAI_IMAGE_MAX_PATCHES = 30_000
# Leave headroom for provider-side preprocessing and future rounding changes.
OPENAI_IMAGE_TARGET_PATCHES = 27_000


@dataclass(frozen=True)
class ImagePatchPreflightSettings:
    enabled: bool = False
    max_patches: int = OPENAI_IMAGE_MAX_PATCHES
    target_patches: int = OPENAI_IMAGE_TARGET_PATCHES
    excluded_models: tuple[str, ...] = ()


_T = TypeVar("_T")


def _positive_int(value: Any, fallback: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed > 0 else fallback


def image_patch_preflight_settings() -> ImagePatchPreflightSettings:
    """Load the active profile's opt-in preflight settings.

    Invalid or unreadable configuration leaves the feature disabled. Threshold
    overrides are independently fail-safe and fall back to documented defaults.
    An explicit target patch count wins over the safety ratio.
    """
    try:
        from hermes_cli.config import load_config, read_user_config_raw

        config = load_config()
        raw_config = read_user_config_raw()
        agent_config = config.get("agent", {}) if isinstance(config, dict) else {}
        raw_agent_config = raw_config.get("agent", {}) if isinstance(raw_config, dict) else {}
        if not isinstance(agent_config, dict):
            agent_config = {}
        if not isinstance(raw_agent_config, dict):
            raw_agent_config = {}
    except Exception as exc:
        logger.warning(
            "image patch preflight config could not be read; leaving disabled: %s",
            exc,
        )
        return ImagePatchPreflightSettings()

    raw_enabled = agent_config.get("openai_image_patch_preflight", False)
    enabled = raw_enabled if isinstance(raw_enabled, bool) else (
        str(raw_enabled).strip().lower() in {"1", "true", "yes", "on"}
    )
    max_patches = _positive_int(
        agent_config.get("openai_image_patch_max_patches"),
        OPENAI_IMAGE_MAX_PATCHES,
    )
    explicit_target = agent_config.get("openai_image_patch_target_patches")
    target_was_explicit = "openai_image_patch_target_patches" in raw_agent_config
    try:
        ratio = float(agent_config.get("openai_image_patch_target_ratio", 0.9))
    except (TypeError, ValueError):
        ratio = 0.9
    if not 0 < ratio <= 1:
        ratio = 0.9
    if target_was_explicit:
        target_patches = _positive_int(
            explicit_target,
            min(OPENAI_IMAGE_TARGET_PATCHES, max_patches),
        )
    else:
        target_patches = max(1, int(max_patches * ratio))
    target_patches = min(target_patches, max_patches)

    raw_exclusions = agent_config.get("openai_image_patch_excluded_models", ())
    if isinstance(raw_exclusions, str):
        raw_exclusions = [raw_exclusions]
    exclusions = tuple(
        str(item).strip().lower()
        for item in raw_exclusions if str(item).strip()
    ) if isinstance(raw_exclusions, (list, tuple, set)) else ()
    return ImagePatchPreflightSettings(enabled, max_patches, target_patches, exclusions)


def is_openai_image_patch_preflight_enabled() -> bool:
    """Return the active profile's explicit preflight opt-in.

    ``load_config`` resolves through the context-local ``HERMES_HOME``
    override used by multiplex gateways, so this remains profile-scoped even
    when several bots share one process. Fail closed: an unreadable config
    must not silently enable a behavior change for another bot.
    """
    return image_patch_preflight_settings().enabled


def is_image_patch_preflight_excluded(model: Any, exclusions: tuple[str, ...]) -> bool:
    """Return whether a model matches a configured case-insensitive glob."""
    normalized = str(model or "").strip().lower()
    return any(fnmatchcase(normalized, pattern) for pattern in exclusions)


def is_gpt56_image_patch_limited(model: Any) -> bool:
    """Return True for the GPT-5.6 family covered by the 30k patch limit."""
    bare = str(model or "").strip().lower().rsplit("/", 1)[-1]
    return (
        bare == "gpt-5.6"
        or bare.startswith("gpt-5.6-")
        or bare.startswith("gpt-5.6.")
    )


def is_openai_image_patch_limited(model: Any) -> bool:
    """Match the model families with native-detail 30k patch rejection.

    Keep the older predicate available to callers, but use this shared
    predicate at the request boundary so GPT-6 Astra and its snapshots
    receive the same protection as GPT-5.6.
    """
    bare = str(model or "").strip().lower().rsplit("/", 1)[-1]
    return is_gpt56_image_patch_limited(model) or (
        bare == "gpt-6-astra" or bare.startswith("gpt-6-astra-")
    )


def image_patch_count(
    width: int,
    height: int,
    *,
    patch_size: int = OPENAI_IMAGE_PATCH_SIZE,
) -> int:
    """Return the provider patch grid size for an image."""
    if width < 1 or height < 1 or patch_size < 1:
        raise ValueError("image dimensions and patch_size must be positive")
    return ceil(width / patch_size) * ceil(height / patch_size)


def fit_dimensions_to_patch_budget(
    width: int,
    height: int,
    *,
    target_patches: int = OPENAI_IMAGE_TARGET_PATCHES,
    patch_size: int = OPENAI_IMAGE_PATCH_SIZE,
) -> tuple[int, int]:
    """Largest aspect-preserving integer dimensions within ``target_patches``.

    A binary search over the long side avoids the excessive 50% downscale used
    by the generic byte/dimension recovery helper and keeps as much visual
    detail as the patch budget permits.
    """
    if width < 1 or height < 1 or target_patches < 1 or patch_size < 1:
        raise ValueError("dimensions, target_patches, and patch_size must be positive")
    if image_patch_count(width, height, patch_size=patch_size) <= target_patches:
        return width, height

    longest = max(width, height)

    def _dimensions_for_long_side(long_side: int) -> tuple[int, int]:
        scale = long_side / longest
        return (
            max(1, min(width, round(width * scale))),
            max(1, min(height, round(height * scale))),
        )

    low, high = 1, longest - 1
    best = (1, 1)
    while low <= high:
        middle = (low + high) // 2
        candidate = _dimensions_for_long_side(middle)
        if image_patch_count(*candidate, patch_size=patch_size) <= target_patches:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    return best


def _data_url_mime_and_payload(data_url: str) -> tuple[Optional[str], Optional[bytes]]:
    if not isinstance(data_url, str) or not data_url.startswith("data:image/"):
        return None, None
    header, separator, encoded = data_url.partition(",")
    if not separator or ";base64" not in header.lower() or not encoded:
        return None, None
    mime = header[5:].split(";", 1)[0].strip().lower()
    if mime == "image/jpg":
        mime = "image/jpeg"
    try:
        return mime, base64.b64decode(encoded)
    except Exception:
        return None, None


def _resize_data_url_to_patch_budget(
    data_url: str,
    *,
    max_patches: int,
    target_patches: int,
    patch_size: int,
) -> Optional[str]:
    """Return a resized data URL, or None when no rewrite is needed/possible."""
    mime, payload = _data_url_mime_and_payload(data_url)
    if mime is None or payload is None:
        return None

    try:
        from PIL import Image, ImageOps

        with Image.open(io.BytesIO(payload)) as opened:
            # A 90-degree EXIF rotation swaps the grid dimensions without
            # changing the patch count. Inspect the header first so images
            # already within budget do not need to decode their pixels.
            original_width, original_height = opened.size
            original_patches = image_patch_count(
                original_width,
                original_height,
                patch_size=patch_size,
            )
            if original_patches <= max_patches:
                return None

            image = ImageOps.exif_transpose(opened)
            original_width, original_height = image.size
            new_width, new_height = fit_dimensions_to_patch_budget(
                original_width,
                original_height,
                target_patches=min(target_patches, max_patches),
                patch_size=patch_size,
            )
            resampling = getattr(Image, "Resampling", Image).LANCZOS
            resized = image.resize((new_width, new_height), resampling)

            has_alpha = resized.mode in {"RGBA", "LA"} or (
                resized.mode == "P" and "transparency" in resized.info
            )
            output = io.BytesIO()
            if mime == "image/png" or has_alpha:
                output_mime = "image/png"
                resized.save(output, format="PNG", optimize=True)
            elif mime == "image/webp":
                output_mime = "image/webp"
                if resized.mode not in {"RGB", "RGBA"}:
                    resized = resized.convert("RGB")
                resized.save(output, format="WEBP", quality=90, method=4)
            else:
                output_mime = "image/jpeg"
                if resized.mode not in {"RGB", "L"}:
                    resized = resized.convert("RGB")
                resized.save(output, format="JPEG", quality=90, optimize=True)

            new_patches = image_patch_count(
                new_width,
                new_height,
                patch_size=patch_size,
            )
            if new_patches > max_patches:
                logger.warning(
                    "image patch preflight produced an over-budget result: "
                    "%dx%d=%d patches (limit=%d)",
                    new_width,
                    new_height,
                    new_patches,
                    max_patches,
                )
                return None

            logger.info(
                "image patch preflight: resized %dx%d (%d patches) to "
                "%dx%d (%d patches); source bytes remain unchanged",
                original_width,
                original_height,
                original_patches,
                new_width,
                new_height,
                new_patches,
            )
            encoded = base64.b64encode(output.getvalue()).decode("ascii")
            return f"data:{output_mime};base64,{encoded}"
    except Exception as exc:
        # Fail open: the normal provider error path remains available, and the
        # exact exception is kept in local logs rather than user-facing chat.
        logger.warning("image patch preflight could not inspect/resize image: %s", exc)
        return None


def resize_image_parts_to_patch_budget(
    api_messages: list,
    *,
    max_patches: int = OPENAI_IMAGE_MAX_PATCHES,
    target_patches: int = OPENAI_IMAGE_TARGET_PATCHES,
    patch_size: int = OPENAI_IMAGE_PATCH_SIZE,
) -> int:
    """Resize over-budget base64 image parts in a provider-bound message copy.

    Handles chat ``content`` parts, Responses ``input_image`` parts, Anthropic
    base64 image blocks, and images nested in ``function_call_output.output``.
    Rewrites containers copy-on-write so aliased persisted-history parts and
    the original files/data URLs remain untouched. Remote HTTP(S) images are
    intentionally skipped because this synchronous boundary does not fetch
    network content.
    """
    if not isinstance(api_messages, list) or not api_messages:
        return 0
    if max_patches < 1 or target_patches < 1 or patch_size < 1:
        raise ValueError("patch budgets and patch_size must be positive")

    def _rewrite(node: Any) -> tuple[Any, int]:
        if isinstance(node, list):
            rewritten_items = None
            changed_total = 0
            for index, item in enumerate(node):
                rewritten, changed = _rewrite(item)
                if changed:
                    if rewritten_items is None:
                        rewritten_items = list(node)
                    rewritten_items[index] = rewritten
                    changed_total += changed
            return (rewritten_items if rewritten_items is not None else node), changed_total

        if not isinstance(node, dict):
            return node, 0

        part_type = node.get("type")
        if part_type in {"image_url", "input_image"}:
            image_value = node.get("image_url")
            if isinstance(image_value, dict):
                url = image_value.get("url")
                resized = _resize_data_url_to_patch_budget(
                    url,
                    max_patches=max_patches,
                    target_patches=target_patches,
                    patch_size=patch_size,
                )
                if resized:
                    return {
                        **node,
                        "image_url": {**image_value, "url": resized},
                    }, 1
            elif isinstance(image_value, str):
                resized = _resize_data_url_to_patch_budget(
                    image_value,
                    max_patches=max_patches,
                    target_patches=target_patches,
                    patch_size=patch_size,
                )
                if resized:
                    return {**node, "image_url": resized}, 1
            return node, 0

        if part_type == "image":
            source = node.get("source")
            if isinstance(source, dict) and source.get("type") == "base64":
                media_type = str(source.get("media_type") or "image/jpeg")
                data = source.get("data")
                if isinstance(data, str) and data:
                    resized = _resize_data_url_to_patch_budget(
                        f"data:{media_type};base64,{data}",
                        max_patches=max_patches,
                        target_patches=target_patches,
                        patch_size=patch_size,
                    )
                    if resized:
                        header, _, encoded = resized.partition(",")
                        output_mime = header[5:].split(";", 1)[0]
                        return {
                            **node,
                            "source": {
                                **source,
                                "type": "base64",
                                "media_type": output_mime,
                                "data": encoded,
                            },
                        }, 1
            return node, 0

        rewritten_node = None
        changed_total = 0
        for key, value in node.items():
            if not isinstance(value, (dict, list)):
                continue
            rewritten, changed = _rewrite(value)
            if changed:
                if rewritten_node is None:
                    rewritten_node = dict(node)
                rewritten_node[key] = rewritten
                changed_total += changed
        return (rewritten_node if rewritten_node is not None else node), changed_total

    changed_count = 0
    for index, message in enumerate(list(api_messages)):
        rewritten, changed = _rewrite(message)
        if not changed:
            continue
        # The top-level message is already the request-local copy. Mutate that
        # shell so callers holding a reference to it observe the rewritten
        # nested containers, without touching any aliased history containers.
        if isinstance(message, dict) and isinstance(rewritten, dict):
            message.clear()
            message.update(rewritten)
        else:
            api_messages[index] = rewritten
        changed_count += changed
    return changed_count


def is_patch_overage_error(exc: BaseException) -> bool:
    """Recognize provider HTTP 400 image-patch budget rejections."""
    if getattr(exc, "status_code", None) != 400:
        return False
    text = str(exc).lower()
    response = getattr(exc, "response", None)
    with_body = getattr(response, "text", None)
    if with_body:
        text = f"{text} {str(with_body).lower()}"
    return "patch" in text and any(word in text for word in ("maximum", "max", "limit", "exceed", "too many"))


def _annotate_patch_overage(exc: BaseException, *, enabled: bool) -> None:
    marker = f"image patch preflight enabled={'true' if enabled else 'false'}"
    if marker in str(exc).lower():
        return
    if exc.args:
        exc.args = (f"{exc.args[0]} ({marker})", *exc.args[1:])
    else:
        exc.args = (marker,)


def _reported_patch_counts(exc: BaseException) -> tuple[Optional[int], Optional[int]]:
    """Extract provider-reported ``(required, limit)`` patch counts when present."""
    text = str(exc).lower().replace(",", "")
    response = getattr(exc, "response", None)
    response_text = getattr(response, "text", None)
    if response_text:
        text = f"{text} {str(response_text).lower().replace(',', '')}"

    def _number_near(labels: tuple[str, ...]) -> Optional[int]:
        for label in labels:
            match = re.search(rf"\b{label}\b[^0-9]{{0,24}}([0-9]+)", text)
            if match:
                return int(match.group(1))
        return None

    required = _number_near(("requires", "required", "actual", "has", "is"))
    limit = _number_near(("maximum(?: is)?", "max(?:imum)?", "limit(?: is)?", "allowed"))
    return required, limit


def _recovery_patch_budget(
    exc: BaseException, *, current_target: int, current_max: int,
) -> int:
    """Choose a retry ceiling strictly below the successful preflight ceiling."""
    required, reported_limit = _reported_patch_counts(exc)
    if reported_limit is not None:
        if required and required > reported_limit:
            scaled = int(current_target * reported_limit / required * 0.9)
            return max(1, min(reported_limit, current_target - 1, scaled))
        return max(1, min(reported_limit, current_target - 1, int(reported_limit * 0.9)))
    conservative = int(min(current_target, current_max) * 0.8)
    return max(1, min(current_target - 1, conservative))


def _request_local_copy(api_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Copy only request containers the recursive rewriter may replace."""
    copied = dict(api_kwargs)
    for key in ("messages", "input"):
        value = copied.get(key)
        if isinstance(value, list):
            copied[key] = [dict(item) if isinstance(item, dict) else item for item in value]
    return copied


def _resize_request_images(
    api_kwargs: dict[str, Any], *, max_patches: int, target_patches: int,
) -> tuple[dict[str, Any], int]:
    request = _request_local_copy(api_kwargs)
    changed = 0
    for key in ("messages", "input"):
        value = request.get(key)
        if isinstance(value, list):
            changed += resize_image_parts_to_patch_budget(
                value,
                max_patches=max_patches,
                target_patches=target_patches,
            )
    return request, changed


def request_with_image_patch_preflight(
    api_kwargs: dict[str, Any],
    send: Callable[[dict[str, Any]], _T],
    *,
    max_patches: Optional[int] = None,
    target_patches: Optional[int] = None,
    retry_max_patches: Optional[int] = None,
) -> _T:
    """Send one provider request with opt-in patch preflight and one recovery retry.

    The caller's request/history remains untouched. Inspection and resize failures
    fail open into the normal provider path. A recognized HTTP 400 patch-overage
    response gets at most one resized retry, and final overage errors state whether
    preflight was enabled for the request.
    """
    settings = image_patch_preflight_settings()
    model = api_kwargs.get("model")
    eligible = settings.enabled and not is_image_patch_preflight_excluded(model, settings.excluded_models)
    effective_max = max_patches or settings.max_patches
    effective_target = min(target_patches or settings.target_patches, effective_max)
    request = api_kwargs
    if eligible:
        request, _ = _resize_request_images(
            api_kwargs,
            max_patches=effective_max,
            target_patches=effective_target,
        )
    def _retry_request(exc: BaseException) -> tuple[dict[str, Any], int]:
        if effective_target <= 1:
            return api_kwargs, 0
        if retry_max_patches is not None:
            retry_limit = retry_target = min(
                retry_max_patches,
                effective_target - 1,
            )
        else:
            retry_limit = retry_target = _recovery_patch_budget(
                exc,
                current_target=effective_target,
                current_max=effective_max,
            )
        return _resize_request_images(
            api_kwargs,
            max_patches=retry_limit,
            target_patches=retry_target,
        )

    def _retry_after(exc: BaseException) -> _T:
        retry_request, changed = _retry_request(exc)
        if not changed:
            _annotate_patch_overage(exc, enabled=True)
            raise exc
        try:
            return send(retry_request)
        except Exception as retry_exc:
            if is_patch_overage_error(retry_exc):
                _annotate_patch_overage(retry_exc, enabled=True)
            raise

    try:
        result = send(request)
    except Exception as exc:
        if not is_patch_overage_error(exc):
            raise
        if not eligible:
            _annotate_patch_overage(exc, enabled=settings.enabled)
            raise
        return _retry_after(exc)

    if not api_kwargs.get("stream"):
        return result
    result_stream = cast(Any, result)
    try:
        iter(result_stream)
    except TypeError:
        # Some adapters ignore stream=True and return a completed response.
        return result

    class _PatchRetryStream:
        def __init__(self, initial_stream: Any):
            self.stream = initial_stream
            self.iterator = iter(initial_stream)
            self.yielded = False
            self.retried = False

        def __iter__(self):
            return self

        def __getattr__(self, name: str):
            # Preserve SDK stream metadata (notably ``response`` and transport
            # handles) so downstream error reporting and drain-timeout logic
            # behave exactly as they do for the unwrapped provider stream.
            return getattr(self.stream, name)

        def __next__(self):
            while True:
                try:
                    event = next(self.iterator)
                    self.yielded = True
                    return event
                except StopIteration:
                    raise
                except Exception as exc:
                    if not is_patch_overage_error(exc):
                        raise
                    if self.yielded or self.retried or not eligible:
                        _annotate_patch_overage(exc, enabled=settings.enabled)
                        raise
                    self.close()
                    self.stream = cast(Any, _retry_after(exc))
                    self.iterator = iter(self.stream)
                    self.retried = True

        def close(self):
            close = getattr(self.stream, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.debug("image patch stream close failed", exc_info=True)

    return cast(_T, _PatchRetryStream(result_stream))


__all__ = [
    "OPENAI_IMAGE_MAX_PATCHES",
    "OPENAI_IMAGE_PATCH_SIZE",
    "OPENAI_IMAGE_TARGET_PATCHES",
    "ImagePatchPreflightSettings",
    "fit_dimensions_to_patch_budget",
    "image_patch_count",
    "image_patch_preflight_settings",
    "is_gpt56_image_patch_limited",
    "is_image_patch_preflight_excluded",
    "is_openai_image_patch_limited",
    "is_openai_image_patch_preflight_enabled",
    "is_patch_overage_error",
    "request_with_image_patch_preflight",
    "resize_image_parts_to_patch_budget",
]
