"""Helpers for live-turn redirect/steer payloads that include images.

Text-only steer still appends to the last tool result. Images cannot ride
that string, so they are delivered as a normal multimodal user correction
at the next role-safe loop boundary (after the current tool batch, or after
cancelling an in-flight model request).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

RedirectUserContent = Union[str, List[dict]]


def normalize_image_paths(image_paths: Optional[Sequence[Any]]) -> List[str]:
    """Return existing local image paths, dropping blanks and missing files."""
    out: List[str] = []
    seen: set[str] = set()
    for raw in image_paths or []:
        path = str(raw or "").strip()
        if not path or path in seen:
            continue
        if not Path(path).exists():
            continue
        seen.add(path)
        out.append(path)
    return out


def supports_image_redirect(agent: Any) -> bool:
    """Codex app-server turn/steer is text-only; queue images instead."""
    return getattr(agent, "api_mode", None) != "codex_app_server"


def persist_image_correction_text(user_text: str, image_paths: Sequence[str]) -> str:
    """UI/session form: caption plus ``@image:`` refs (desktop renders these)."""
    from agent.context_references import format_reference_value

    text = (user_text or "").strip()
    refs = "\n".join(
        f"@image:{format_reference_value(p)}" for p in image_paths if Path(p).exists()
    )
    if not refs:
        return text
    return f"{text}\n{refs}" if text else refs


def text_mode_image_correction(user_text: str, image_paths: Sequence[str]) -> str:
    """Model-facing text when native vision is off: inspect via vision_analyze."""
    parts: List[str] = []
    for path in image_paths:
        p = Path(path)
        if not p.exists():
            continue
        parts.append(
            f"[The user attached an image: {p.name}]\n"
            f"[Examine it with the vision_analyze tool using image_url: {p}]"
        )
    text = (user_text or "").strip()
    prefix = "\n\n".join(parts)
    if prefix:
        return f"{prefix}\n\n{text}" if text else prefix
    return text or "What do you see in this image?"


def build_redirect_user_payload(
    agent: Any,
    text: str,
    image_paths: Sequence[str],
) -> Tuple[str, RedirectUserContent]:
    """Return (display_text, api_content_suffix) for an active-turn correction."""
    paths = normalize_image_paths(image_paths)
    cleaned = (text or "").strip()
    if not paths:
        return cleaned, cleaned

    display = persist_image_correction_text(cleaned, paths)
    mode = "text"
    try:
        from hermes_cli.config import load_config
        from agent.image_routing import decide_image_input_mode

        cfg = load_config()
        mode = decide_image_input_mode(
            getattr(agent, "provider", "") or "",
            getattr(agent, "model", "") or "",
            cfg,
            requested_provider=getattr(agent, "requested_provider", "") or "",
        )
        if getattr(agent, "api_mode", "") == "codex_app_server":
            mode = "text"
    except Exception:
        mode = "text"

    if mode == "native":
        try:
            from agent.image_routing import build_native_content_parts

            parts, _skipped = build_native_content_parts(cleaned, list(paths))
            if any(isinstance(p, dict) and p.get("type") == "image_url" for p in parts):
                return display, parts
        except Exception:
            pass
    return display, text_mode_image_correction(cleaned, paths)


def parse_redirect_image_paths(params: Any) -> List[str]:
    raw = params.get("image_paths") if isinstance(params, dict) else None
    if not isinstance(raw, list):
        return []
    return [str(p).strip() for p in raw if str(p).strip()]


def call_redirect_like(fn: Any, text: str, image_paths: Sequence[str]) -> Any:
    try:
        return fn(text, image_paths=list(image_paths) or None)
    except TypeError:
        return fn(text)


def merge_redirect_text(existing: Optional[str], incoming: str) -> str:
    cleaned = (incoming or "").strip()
    prior = (existing or "").strip()
    if not cleaned:
        return prior
    if not prior:
        return cleaned
    return f"{prior}\n\n[Additional user correction]\n{cleaned}"
