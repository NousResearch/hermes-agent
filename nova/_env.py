"""Reading which variables a dotenv file defines, without reading their values.

Lives in the platform layer rather than beside the runtime adapter because two unrelated
things need it — deployment readiness and backup — and only one of them is runtime-shaped.
A ``.env`` file is a format, not a Hermes concept.

**Values are read and immediately dropped.** Only the presence of a name is ever returned,
so nothing in NOVA can log, audit or display a credential it happened to parse on the way
to answering "is this set?".
"""

from __future__ import annotations

from pathlib import Path

#: The conventional per-agent environment file. On ``NEVER_WRITE``: the operator owns it.
ENV_FILENAME = ".env"


def read_env_file(path: Path) -> dict[str, str]:
    """Variable names defined in a dotenv file, with every value discarded.

    Deliberately a small parser rather than a dependency: NOVA loads with the standard
    library and PyYAML, and the question here is only which keys exist.
    """
    names: dict[str, str] = {}
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return names
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export "):].lstrip()
        name, separator, _value = stripped.partition("=")
        name = name.strip()
        if separator and name:
            names[name] = ""  # presence only; the value is not kept
    return names
