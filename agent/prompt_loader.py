"""Prompt file resolution and loading.

Supports the ``cronjob`` tool's ``prompt_file`` parameter by resolving
a relative path against several well-known search directories.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def load_prompt_text(
    path: str,
    fallback_text: Optional[str] = None,
) -> Optional[str]:
    """Read a prompt file, searching multiple locations.

    Search order (first match wins):

    1. As an absolute path, or relative to the process CWD.
    2. ``<cwd>/prompts/<basename>``
    3. ``$HERMES_PROMPTS_ROOT/<basename>`` (if set)
    4. ``$HERMES_HOME/prompts/<basename>`` (if set)

    Parameters
    ----------
    path:
        File path to search for. May be an absolute path or a relative
        basename.
    fallback_text:
        If provided and the file cannot be found, returned as-is instead
        of ``None``.  Use this to supply a default empty-string or
        fallback prompt.

    Returns
    -------
    File contents (stripped of leading/trailing whitespace) on success,
    ``None`` (or *fallback_text*) when no matching file was found.
    """
    candidates: list[Path] = []

    p = Path(path)

    # 1. As given — absolute or relative to CWD
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(Path.cwd() / p)

    # 2. CWD/prompts/<basename>
    candidates.append(Path.cwd() / "prompts" / p.name)

    # 3. $HERMES_PROMPTS_ROOT/<basename>
    prompts_root = os.environ.get("HERMES_PROMPTS_ROOT")
    if prompts_root:
        candidates.append(Path(prompts_root) / p.name)

    # 4. $HERMES_HOME/prompts/<basename>
    hermes_home = os.environ.get("HERMES_HOME")
    if hermes_home:
        candidates.append(Path(hermes_home) / "prompts" / p.name)

    for candidate in candidates:
        try:
            text = candidate.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError):
            continue
        return text.strip()

    return fallback_text
