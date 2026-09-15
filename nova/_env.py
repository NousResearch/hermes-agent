"""Reading which variables a dotenv file defines — and, deliberately, writing them.

Lives in the platform layer rather than beside the runtime adapter because two unrelated
things need it — deployment readiness and backup — and only one of them is runtime-shaped.
A ``.env`` file is a format, not a Hermes concept.

**Values are read and immediately dropped.** Only the presence of a name is ever returned,
so nothing in NOVA can log, audit or display a credential it happened to parse on the way
to answering "is this set?". :func:`read_env_file` has no mode that returns a value, which
is why no caller can be talked into returning one.

**Writing is a separate, narrow act.** ``.env`` remains on the materialiser's
``NEVER_WRITE`` list, so ``nova apply`` still cannot touch it — a deployment's credentials
are not something a configuration push should be able to overwrite. :func:`write_env_values`
exists for one caller: an administrator deliberately setting a credential through the
control plane. It merges rather than replaces, so a variable NOVA does not know about
survives, and it writes 0600.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Optional

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


def _quote(value: str) -> str:
    """Render a value so a dotenv reader gets back exactly what was set.

    Double quotes, with backslashes and quotes escaped. Bare values break on spaces, ``#``
    and trailing whitespace, and a credential is exactly the kind of string that contains
    them — a token with a ``#`` in it silently truncated at the comment marker is a failure
    nobody would think to look for.
    """
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    escaped = escaped.replace("\n", "\\n").replace("\r", "")
    return f'"{escaped}"'


def write_env_values(path: Path, values: Mapping[str, Optional[str]]) -> tuple[str, ...]:
    """Set or clear variables in a dotenv file. Returns the names that changed.

    A value of ``None`` removes the variable. Every other line in the file — comments,
    blank lines, variables NOVA has never heard of — is preserved verbatim and in order:
    this file is the operator's, and an edit through the control plane must not quietly
    reformat or drop the parts of it NOVA does not model.

    Written atomically at mode 0600. The temporary file is created with that mode from the
    outset rather than chmod-ed afterwards, so the value is never briefly world-readable.
    """
    path = Path(path)
    existing = ""
    try:
        existing = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        existing = ""

    changed: list[str] = []
    remaining = dict(values)
    out: list[str] = []

    for line in existing.splitlines():
        stripped = line.strip()
        body = stripped[len("export "):].lstrip() if stripped.startswith("export ") else stripped
        name, separator, _ = body.partition("=")
        name = name.strip()
        if not separator or not name or stripped.startswith("#") or name not in remaining:
            out.append(line)
            continue
        new = remaining.pop(name)
        changed.append(name)
        if new is not None:
            out.append(f"{name}={_quote(new)}")
        # None: the line is dropped, which is how a credential is cleared.

    for name, new in remaining.items():
        if new is None:
            continue  # clearing something that was never set is not a change
        out.append(f"{name}={_quote(new)}")
        changed.append(name)

    if not changed:
        return ()

    text = "\n".join(out).rstrip("\n") + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = os.open(
        str(path.with_name(path.name + ".nova-tmp")),
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        0o600,
    )
    tmp = path.with_name(path.name + ".nova-tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return tuple(sorted(changed))
