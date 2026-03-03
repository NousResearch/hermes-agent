"""Shared .env loading helpers with encoding fallbacks and token validation."""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path
from typing import Iterable, Optional, Tuple

from dotenv import load_dotenv


DEFAULT_DOTENV_ENCODINGS: tuple[str, ...] = (
    "utf-8",
    "utf-8-sig",
    "cp1252",
    "latin-1",
)

# Secrets and IDs that should be printable ASCII only. If these contain
# mojibake after fallback decoding, tool auth fails in opaque ways.
SENSITIVE_ENV_KEY_RE = re.compile(
    r"(?:_API_KEY|_BOT_TOKEN|_APP_TOKEN|_ACCESS_TOKEN|_REFRESH_TOKEN|_PROJECT_ID|_CLIENT_SECRET)$"
)
PRINTABLE_ASCII_RE = re.compile(r"^[\x20-\x7E]+$")


def read_text_with_fallback(
    path: Path,
    encodings: Optional[Iterable[str]] = None,
) -> Tuple[str, str]:
    """Read a text file using fallback encodings.

    Returns:
        (text, encoding_used)
    """
    encoding_candidates = tuple(encodings or DEFAULT_DOTENV_ENCODINGS)
    data = path.read_bytes()

    for encoding in encoding_candidates:
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue

    # Last-resort path: keep process alive and preserve as much content as possible.
    return data.decode("utf-8", errors="replace"), "utf-8-replace"


def _parse_env_line(raw_line: str) -> tuple[str, str] | None:
    line = raw_line.strip()
    if not line or line.startswith("#") or "=" not in line:
        return None
    key, _, value = line.partition("=")
    key = key.strip()
    value = value.strip()
    if value and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    return key, value


def _validate_sensitive_values(
    env_text: str, path: Path, *, encoding_used: str, logger: Optional[logging.Logger] = None
) -> None:
    invalid_keys: list[str] = []
    for raw in env_text.splitlines():
        parsed = _parse_env_line(raw)
        if not parsed:
            continue
        key, value = parsed
        if not value or not SENSITIVE_ENV_KEY_RE.search(key):
            continue
        if not PRINTABLE_ASCII_RE.fullmatch(value):
            invalid_keys.append(key)

    if not invalid_keys:
        return
    key_list = ", ".join(sorted(set(invalid_keys)))
    if encoding_used not in ("utf-8", "utf-8-sig"):
        if logger:
            logger.warning(
                "Non-ASCII in sensitive .env keys (%s) at %s (decoded as %s). "
                "Re-save ~/.hermes/.env as UTF-8 to fix.",
                key_list, path, encoding_used,
            )
    raise ValueError(
        f"Invalid non-ASCII bytes detected in sensitive .env values ({key_list}) at {path}. "
        "Re-save ~/.hermes/.env as UTF-8 and re-enter affected keys."
    )


def read_env_text_with_fallback(
    path: Path,
    encodings: Optional[Iterable[str]] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[str, str]:
    """Read `.env` text with fallback decoding and secret validation."""
    text, encoding_used = read_text_with_fallback(path, encodings=encodings)
    _validate_sensitive_values(text, path, encoding_used=encoding_used, logger=logger)
    return text, encoding_used


def load_dotenv_with_fallback(
    dotenv_path: Path | str,
    *,
    override: bool = False,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """Load dotenv content with robust decoding and secret validation.

    This avoids startup crashes when ~/.hermes/.env contains non-UTF8 bytes
    while still failing clearly for corrupted API/token values.
    """
    path = Path(dotenv_path).expanduser()
    if not path.exists():
        return False

    text, encoding_used = read_env_text_with_fallback(path, logger=logger)

    changed = load_dotenv(stream=io.StringIO(text), override=override)

    if logger and encoding_used not in ("utf-8", "utf-8-sig"):
        logger.warning(
            "Loaded %s with fallback encoding '%s'. Consider saving it as UTF-8.",
            path,
            encoding_used,
        )

    return changed
