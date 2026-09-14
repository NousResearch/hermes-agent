"""Editing a tenant bundle safely, so the Control Centre can change one.

Until now the bundle was written by hand and read by NOVA. The Control Centre needs to
change it — create an agent, rewrite a Soul, grant a channel — and the bundle has to stay
the single source of truth while that happens. This module is the only place NOVA writes
one, and it holds three properties that a naive "write the YAML and hope" would not:

**Validated before committed.** Every edit is applied to a private copy, and that copy is
loaded through the ordinary :func:`nova.spec.load_bundle` — the same function, with the
same cross-checks, that validates a bundle on disk. If the result does not load, nothing is
written and the caller gets the ordinary :class:`~nova.errors.SpecError` naming the field.
There is no second, weaker validator for edits made through the UI.

**All or nothing.** A change that touches an agent file and a prompt file lands as both or
neither. A half-applied edit would leave a bundle that does not load, which is the one state
an operator cannot recover from through the interface they were using.

**Never outside the bundle.** Paths are built from validated ids and resolved against the
bundle root; anything that escapes is refused before a byte is written. A control plane that
can be talked into writing `../../.ssh/authorized_keys` is not a control plane.

What this module deliberately does **not** do is write to a runtime profile. An agent's
``SOUL.md`` is derived — ``materialize`` builds it from the bundle and overwrites it on
every apply — so an edit written there would save, persist, survive a reload, and then
vanish the next time anyone applied the bundle. Edits go to the bundle; the runtime is
updated by applying it.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import yaml

from nova._fields import _ID_PATTERN
from nova.errors import SpecError

#: Directories inside a bundle the writer may touch. Everything else — knowledge corpora in
#: particular — is the customer's and is copied, never rewritten.
WRITABLE_DIRS = ("agents", "prompts", "objectives", "automations")

#: Files at the bundle root the writer may replace.
WRITABLE_FILES = (
    "organization.yaml",
    "identity.yaml",
    "policy.yaml",
    "knowledge.yaml",
    "channels.yaml",
    "deployment.yaml",
)


def validate_id(value: str, *, what: str = "id") -> str:
    """The id, or a refusal.

    Reuses the pattern every bundle id is already held to (``nova/_fields.py``) rather than
    inventing a second one: lowercase letters, digits and hyphens, alphanumeric at both
    ends, 64 characters. That shape cannot contain ``/``, ``\\`` or ``..``, which is what
    makes every path built from it safe by construction rather than by a later check.
    """
    value = (value or "").strip()
    if not _ID_PATTERN.match(value):
        raise SpecError(
            f"{value!r} is not a valid {what} — use lowercase letters, digits and hyphens, "
            "starting and ending alphanumeric (max 64 characters)"
        )
    return value


def safe_relative(root: Path, relative: str) -> Path:
    """Resolve ``relative`` inside ``root``, or refuse.

    Belt to :func:`validate_id`'s braces. Ids are already traversal-proof, but not every
    path in a bundle comes from an id — ``instructions: prompts/ops.md`` is author-supplied
    — so the resolved path is checked against the root before it is used.
    """
    root = Path(root).resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise SpecError(
            f"{relative!r} resolves outside the bundle. Paths in a bundle are relative to it"
        ) from None
    return candidate


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".nova-")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def dump_yaml(document: Any) -> str:
    """Bundle YAML: block style, key order preserved, unicode intact.

    ``sort_keys=False`` because a bundle is read by people. An agent file whose keys got
    alphabetised on every save would churn the diff and bury the field that changed.
    """
    return yaml.safe_dump(document, sort_keys=False, allow_unicode=True, default_flow_style=False)


def _iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file():
            yield path


def _sync(staged: Path, root: Path) -> list[str]:
    """Copy the staged bundle over the real one. Returns the paths that changed.

    Writes are atomic per file, and a file removed in the staging copy is removed here. The
    window where the two disagree is one ``os.replace`` per file — the alternative, swapping
    whole directories, would break anything holding the bundle path open.
    """
    changed: list[str] = []
    staged_rel = {p.relative_to(staged).as_posix(): p for p in _iter_files(staged)}
    live_rel = {p.relative_to(root).as_posix(): p for p in _iter_files(root)}

    for rel, source in sorted(staged_rel.items()):
        target = root / rel
        new = source.read_bytes()
        if rel in live_rel and live_rel[rel].read_bytes() == new:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(target, new.decode("utf-8"))
        changed.append(rel)

    for rel, existing in sorted(live_rel.items()):
        if rel not in staged_rel:
            existing.unlink(missing_ok=True)
            changed.append(rel + " (removed)")
    return changed


class BundleEdit:
    """The staged copy an edit is made against. Passed to the caller's mutate function."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def read_yaml(self, relative: str) -> dict[str, Any]:
        path = safe_relative(self.root, relative)
        if not path.is_file():
            return {}
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {}

    def write_yaml(self, relative: str, document: Any) -> None:
        _atomic_write(safe_relative(self.root, relative), dump_yaml(document))

    def read_text(self, relative: str) -> str:
        path = safe_relative(self.root, relative)
        return path.read_text(encoding="utf-8") if path.is_file() else ""

    def write_text(self, relative: str, text: str) -> None:
        _atomic_write(safe_relative(self.root, relative), text)

    def remove(self, relative: str) -> bool:
        path = safe_relative(self.root, relative)
        if not path.is_file():
            return False
        path.unlink()
        return True

    def exists(self, relative: str) -> bool:
        return safe_relative(self.root, relative).exists()


def edit(root: Path, mutate: Callable[[BundleEdit], None], *, loader: Optional[Callable] = None):
    """Apply ``mutate`` to the bundle at ``root``, validated, all or nothing.

    Returns the freshly loaded :class:`~nova.spec.bundle.TenantBundle` so a caller can
    report what the edit produced rather than re-reading it.

    ``loader`` is injected only so tests can assert the validation actually runs; it
    defaults to the real bundle loader, and no caller should pass anything else.
    """
    from nova.spec import load_bundle

    root = Path(root)
    if not root.is_dir():
        raise SpecError(f"{root} is not a bundle directory")

    load = loader or load_bundle

    with tempfile.TemporaryDirectory(prefix="nova-edit-") as tmp:
        staged = Path(tmp) / "bundle"
        # copytree rather than a targeted copy: the loader cross-checks agents against
        # policy, knowledge and channels, so validating an edit needs the whole bundle
        # present. Bundles are declarations — kilobytes — and the corpora under knowledge/
        # are the only thing that could be large.
        shutil.copytree(root, staged, symlinks=False)
        mutate(BundleEdit(staged))
        try:
            # Raises SpecError → nothing below runs, and nothing is written.
            bundle = load(staged)
        except SpecError as exc:
            raise _relocate(exc, staged, root) from None
        changed = _sync(staged, root)

    return bundle, changed


def _relocate(exc: SpecError, staged: Path, root: Path) -> SpecError:
    """Re-point a validation failure at the real bundle.

    The loader reports the file it read, which is the staging copy — a path under /tmp that
    has already been deleted by the time anyone reads the message. Naming a directory the
    operator cannot open is worse than naming none, so the prefix is swapped back.
    """
    source = exc.source
    if source is not None:
        try:
            source = root / Path(source).resolve().relative_to(staged.resolve())
        except ValueError:
            source = None
    return SpecError(exc.message, field=exc.field, source=source)
