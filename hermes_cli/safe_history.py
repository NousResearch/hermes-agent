"""``FileHistory`` that survives unencodable input and write failures."""
from __future__ import annotations

import logging

from prompt_toolkit.history import FileHistory

from hermes_cli.input_sanitize import sanitize_user_prompt_text

logger = logging.getLogger(__name__)


class SafeFileHistory(FileHistory):
    """Normalizes lone surrogates before writing; a failed append only logs."""

    def store_string(self, string: str) -> None:
        try:
            super().store_string(sanitize_user_prompt_text(string))
        except (UnicodeEncodeError, UnicodeDecodeError, OSError, ValueError) as exc:
            logger.warning("Could not append to shell history: %s", exc)
