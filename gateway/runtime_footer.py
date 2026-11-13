"""Gateway runtime-metadata footer (model · context % · cwd), off by default to keep replies
minimal. Config: ``display.runtime_footer: {enabled: bool, fields: [model, context_pct, cwd]}``
(order shown; drop any to hide), per-platform override ``display.platforms.<p>.runtime_footer``,
toggled by ``/footer on|off``. Fields: ``model`` (vendor prefix dropped), ``context_pct`` (last-call
occupancy), ``latency`` (turn wall-clock, opt-in — NOT in the default set so an unset ``fields``
renders exactly as before), ``served_model`` (opt-in, ``alias → served``: the deployment a routing
proxy reported via ``x-litellm-model-id`` / ``x-litellm-model-api-base``, or Hermes' own fallback
route; skipped when the served model is the requested one), ``cwd`` (home-relative). ``gateway/run.py`` appends the footer to the
final response only (never to tool-progress or streaming partials); when streaming already
delivered the text, it goes out as a trailing message via ``send_trailing_footer()``.

Photo replies (#74547): where the footer would be a message of its own (the trailing send after a
streamed reply, or a reply whose only text is the footer) it rides the last photo's caption instead,
on adapters that declare ``media_caption_limit`` (the caption length one ``send_multiple_images``
photo renders; Telegram: 1024) and only when it fits; otherwise it is sent as before.
``FooteredReply`` carries the footer next to the reply text to the adapter's send path."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import quote

_DEFAULT_FIELDS: tuple[str, ...] = ("model", "context_pct", "cwd")
_SEP = " · "


def _home_relative_cwd(cwd: str) -> str:
    """Return *cwd* with ``$HOME`` collapsed to ``~``.  Empty string if unset."""
    if not cwd:
        return ""
    try:
        home = os.path.expanduser("~")
        p = os.path.abspath(cwd)
        if home and (p == home or p.startswith(home + os.sep)):
            return "~" + p[len(home):]
        return p
    except Exception:
        return cwd


def _model_short(model: Optional[str]) -> str:
    """Drop ``vendor/`` prefix (``openai/gpt-5.4`` → ``gpt-5.4``)."""
    return model.rsplit("/", 1)[-1] if model else ""


def _env_cwd() -> str:
    try:
        from tools.terminal_scope import terminal_env
    except ImportError:
        return os.environ.get("TERMINAL_CWD", "")
    return terminal_env("TERMINAL_CWD", "")


def resolve_footer_config(user_config: dict[str, Any] | None, platform_key: str | None = None) -> dict[str, Any]:
    """Resolve effective footer config: defaults (enabled=False) <
    ``display.runtime_footer`` < ``display.platforms.<platform_key>.runtime_footer``."""
    resolved = {"enabled": False, "fields": list(_DEFAULT_FIELDS)}
    cfg = (user_config or {}).get("display") or {}
    plat_cfg = (cfg.get("platforms") or {}).get(platform_key) if platform_key else None
    sections = [cfg.get("runtime_footer"), plat_cfg.get("runtime_footer") if isinstance(plat_cfg, dict) else None]
    for section in sections:
        if not isinstance(section, dict):
            continue
        if "enabled" in section:
            resolved["enabled"] = bool(section.get("enabled"))
        if isinstance(section.get("fields"), list) and section["fields"]:
            resolved["fields"] = [str(f) for f in section["fields"]]
    return resolved


def _format_latency(seconds: float) -> str:
    """Humanize a turn duration: ``<1s``, ``22s``, ``1m05s``."""
    if seconds < 1:
        return "<1s"
    total = round(seconds)
    if total < 60:
        return f"{total}s"
    m, sec = divmod(total, 60)
    return f"{m}m{sec:02d}s"


def format_runtime_footer(*, model: Optional[str], context_tokens: int,
                          context_length: Optional[int], cwd: Optional[str] = None,
                          turn_seconds: Optional[float] = None,
                          requested_model: Optional[str] = None, served_model: Optional[str] = None,
                          fields: Iterable[str] = _DEFAULT_FIELDS) -> str:
    """Render the footer line, or "" if no fields have data. Fields whose data is missing (and
    unknown field names) are skipped silently — a partial footer beats ``?%`` or empty slots."""
    def context_pct() -> str:
        if context_length and context_length > 0 and context_tokens >= 0:
            return f"{max(0, min(100, round((context_tokens / context_length) * 100)))}%"
        return ""

    def served() -> str:
        requested = requested_model or model
        alias = _model_short(requested)
        if served_model and served_model not in (alias, requested):
            return f"{alias} → {served_model}"
        return ""

    renderers = {
        "model": lambda: _model_short(model),
        "served_model": served,
        "context_pct": context_pct,
        # Skipped when the caller did not measure (None) or the value is negative.
        "latency": lambda: _format_latency(turn_seconds) if turn_seconds is not None and turn_seconds >= 0 else "",
        "cwd": lambda: _home_relative_cwd(cwd or _env_cwd()),
    }
    return _SEP.join(v for field in fields if (render := renderers.get(field)) and (v := render()))


def build_footer_line(*, user_config: dict[str, Any] | None, platform_key: str | None,
                      model: Optional[str], context_tokens: int, context_length: Optional[int],
                      cwd: Optional[str] = None, turn_seconds: Optional[float] = None,
                      requested_model: Optional[str] = None, served_model: Optional[str] = None) -> str:
    """Entry point for gateway/run.py: footer text, or "" when disabled / no data. Callers append it
    to the final response themselves, preserving a single blank line of separation.
    ``turn_seconds`` is the caller-measured (``time.monotonic()``) run duration; ``None`` skips the
    ``latency`` field."""
    cfg = resolve_footer_config(user_config, platform_key)
    if not cfg.get("enabled"):
        return ""
    return format_runtime_footer(model=model, context_tokens=context_tokens,
                                 context_length=context_length, cwd=cwd, turn_seconds=turn_seconds,
                                 requested_model=requested_model, served_model=served_model,
                                 fields=cfg.get("fields") or _DEFAULT_FIELDS)


class FooteredReply(str):
    """A final reply whose text ends with ``\n\n<runtime_footer>``. Plain text everywhere; any
    rewrite yields a plain ``str`` again, so the footer then simply stays in the text."""
    runtime_footer: str

    def __new__(cls, text: str, footer: str):
        instance = super().__new__(cls, f"{text}\n\n{footer}")
        instance.runtime_footer = footer
        return instance


class _FooterCaption(str):
    """A photo caption that carries the runtime footer (``caption_with_footer``)."""
    runtime_footer: str


def caption_with_footer(adapter: Any, images: list, footer: str) -> tuple[list, bool]:
    """``(images, True)`` with ``footer`` on the last ``(url, alt)`` photo's caption when the
    adapter declares ``media_caption_limit`` and the caption fits it (UTF-16 units). Never on an
    animation: a GIF goes out on its own path, whose failure the batch result does not list."""
    limit = int(getattr(adapter, "media_caption_limit", 0) or 0)
    target = next((i for i in range(len(images) - 1, -1, -1)
                   if not adapter._is_animation_url(images[i][0])), None) if footer and limit > 0 else None
    if target is None:
        return images, False
    url, alt = images[target]
    caption = _FooterCaption(f"{alt}\n\n{footer}" if alt else footer)
    if len(caption.encode("utf-16-le")) // 2 > limit:
        return images, False
    caption.runtime_footer = footer
    return [*images[:target], (url, caption), *images[target + 1:]], True


def caption_landed(result: Any, images: list) -> bool:
    """The photo whose caption carries the footer was delivered, or is queued for a re-upload that
    carries it (``raw_response["deferred"]``: flood control; a separate footer would repeat it).
    ``success`` means at least one photo landed, so that photo listed in ``raw_response["failed"]``
    (``(path_or_url, error)``) took the footer with it. False when no caption carries it."""
    url = next((u for u, alt in images if getattr(alt, "runtime_footer", "")), None)
    raw = getattr(result, "raw_response", None)
    raw = raw if isinstance(raw, dict) else {}
    if url is None or url in (raw.get("deferred") or ()):
        return url is not None
    if not getattr(result, "success", False):
        return False
    from urllib.parse import unquote
    lost = {str(item[0]) for item in raw.get("failed") or () if item}
    return url not in lost and unquote(url.removeprefix("file://")) not in lost


async def send_footer_unless_captioned(adapter: Any, chat_id: str, images: list, result: Any,
                                       metadata: Any, record_delivery: Any) -> None:
    """Non-streamed send path: the footer a photo caption carried, sent as its own message (as
    before #74547) when that photo did not land."""
    footer = next((alt.runtime_footer for _u, alt in images if getattr(alt, "runtime_footer", "")), "")
    if footer and not caption_landed(result, images):
        record_delivery(await adapter._send_with_retry(chat_id=chat_id, content=footer, metadata=metadata))


def footer_onto_photo_caption(adapter: Any, extracted: Any, image_exts: Iterable[str]) -> Any:
    """The adapter send path's ``_ExtractedResponse``, with a footer-only text moved onto the last
    photo's caption (MEDIA photos join the photo batch): otherwise the footer is its own message."""
    footer = getattr(extracted.pre_extract, "runtime_footer", "")
    if not footer or extracted.force_document_attachments or extracted.text_content.strip() != footer.strip():
        return extracted
    photos = [p for p, voice in extracted.media_files if not voice and Path(p).suffix.lower() in image_exts]
    images, attached = caption_with_footer(
        adapter, [*extracted.images, *((f"file://{quote(p)}", "") for p in photos)], footer)
    if not attached:
        return extracted
    return dataclasses.replace(extracted, text_content="", images=images,
                               media_files=[m for m in extracted.media_files if m[0] not in photos])
