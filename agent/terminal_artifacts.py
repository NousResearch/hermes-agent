"""Materialize explicit terminal attachments while their producing environment is still live."""

from pathlib import Path


def _host_artifact(path: str, home: Path) -> str | None:
    from gateway.platforms.base import validate_media_delivery_path

    candidate = Path(path)
    # Do not probe the host filesystem for paths owned by the terminal.
    if not candidate.is_absolute() or not any(candidate.is_relative_to(root) for root in (home, home.resolve())):
        return None
    safe = validate_media_delivery_path(path)
    return safe if safe and Path(safe).is_relative_to(home.resolve()) else None


def materialize_terminal_media(text: str, task_id: str | None) -> str:
    """Turn remote ``MEDIA:`` references into host files before persistence and backend cleanup.

    Only the current task's existing environment is consulted. Ordinary terminal paths belong to
    it even when the host happens to contain a readable namesake. Profile-managed host artifacts
    (image generation, uploads, earlier exports) keep their already deliverable host path.
    """
    if not isinstance(text, str) or "media:" not in text.lower() or not isinstance(task_id, str) or not task_id:
        return text
    from tools.environments.base import env_uses_host_filesystem
    from tools.terminal_tool_lifecycle import get_active_env

    env = get_active_env(task_id)
    if env is None or env_uses_host_filesystem(env):
        return text
    from gateway.media_fetch import fetch_remote_media
    from gateway.platforms.base import (
        MEDIA_EXTENSIONLESS_TAG_RE, MEDIA_TAG_CLEANUP_RE, _mask_media_scan_text,
        _match_extensionless_path, _normalize_media_tag_path,
    )
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    destination = home / "cache" / "generated" / "files"
    masked = _mask_media_scan_text(text)
    spans = []
    for pattern in (MEDIA_TAG_CLEANUP_RE, MEDIA_EXTENSIONLESS_TAG_RE):
        for match in pattern.finditer(masked):
            if any(start <= match.start() < end for start, end, _, _ in spans):
                continue
            spans.append((match.start(), match.end(), match, pattern is MEDIA_EXTENSIONLESS_TAG_RE))
    replacements = []
    exported = {}
    def export(path):
        if path not in exported:
            exported[path] = _host_artifact(path, home) or fetch_remote_media(
                path, env=env, destination_dir=destination,
            )
        return exported[path]

    for start, end, match, expandable in sorted(spans, key=lambda item: item[0]):
        path = _normalize_media_tag_path(match.group("path"))
        resolved = _match_extensionless_path(masked, match, validate=export) if expandable else None
        local, path_end = resolved if resolved else (export(path), match.end("path"))
        if local == _normalize_media_tag_path(masked[match.start("path"):path_end]):
            continue
        if local is None:
            # Keep the failure visible, but not as a clickable host-path attachment.
            replacements.append((start, end, f"Attachment unavailable: could not export `{path}` from the terminal."))
        else:
            replacements.append((match.start("path"), path_end, f'"{local}"'))
    for start, end, replacement in sorted(replacements, reverse=True):
        text = text[:start] + replacement + text[end:]
    return text
