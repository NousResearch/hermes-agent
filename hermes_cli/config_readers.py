"""Raw configuration readers for decisions that must preserve invalid input."""

from pathlib import Path
from typing import Any, Dict, Optional

from utils import fast_safe_load


def read_user_config_for_authority(
    config_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Read a user ``config.yaml`` preserving ROOT SHAPE, for authority decisions.

    Companion to :func:`read_user_config_raw`, added rather than folded into
    it because that function's normalize-to-``{}`` contract is depended on by
    its existing callers.

    Use this — and only this — when the answer *"was anything selected here?"*
    decides which physical resource the process will use, and where guessing
    wrong is unsafe rather than merely inconvenient. The motivating case is
    the session-state backend: a config that exists but is structurally
    unusable cannot be read as "the operator chose SQLite", because that
    silently splits session history across two stores.

    The distinction this preserves, which every other reader flattens:

      * missing file        → ``None``   (nothing was written: no selection)
      * empty / whitespace  → ``None``   (nothing was written: no selection)
      * comments only       → ``None``   (nothing was written: no selection)
      * mapping root        → the ``dict`` exactly as written
      * NON-mapping root
        (list, scalar, ...) → raises ``ValueError`` — the operator wrote
                              something, and what they intended cannot be
                              determined; it must not be read as "nothing"
      * unparseable YAML    → raises (the underlying parser error)

    ``None`` and ``{}`` are therefore NOT interchangeable here: ``None`` means
    "no configuration exists", while a returned ``{}`` is an explicitly empty
    mapping (an empty document yields ``None``). Callers may treat ``None`` as "no selection"
    and must let the exceptions propagate.

    No DEFAULT_CONFIG merge, no managed-scope overlay, no ``${ENV_VAR}``
    expansion, no migration, no caching — same as ``read_user_config_raw``.

    ``config_path`` defaults to :func:`get_config_path` (profile-aware).
    """
    if config_path is None:
        from hermes_cli.config import get_config_path

        config_path = get_config_path()
    try:
        with open(config_path, encoding="utf-8") as f:
            data = fast_safe_load(f)
    except FileNotFoundError:
        return None

    # An empty document — truly empty, whitespace, or comments only — parses to
    # None. That is a genuine "operator wrote nothing", distinct from a
    # structurally invalid document.
    if data is None:
        return None

    if not isinstance(data, dict):
        raise ValueError(
            f"{config_path} has a {type(data).__name__} at its top level, but "
            f"a Hermes config must be a mapping. Refusing to interpret a "
            f"structurally unusable file as 'nothing configured', because "
            f"that answer decides which physical store this process uses."
        )
    return data
