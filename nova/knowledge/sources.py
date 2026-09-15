"""What a knowledge source is, and which files belong to it.

A source is declared once per tenant, in ``knowledge.yaml``, and referenced by id from any
number of agents. The declaration is deliberately narrow — a root directory, glob filters,
a size cap and a classification label — because everything it does not say is something a
customer cannot get wrong.

The walk below is the security boundary of the whole capability. Everything downstream
assumes the file it is handed is one the tenant meant to publish, so this module refuses
rather than guesses: no symlinks, no paths that resolve outside the declared root, no files
above the cap, nothing from a dot-directory.
"""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

from nova._fields import Doc
from nova.errors import SpecError
from nova.knowledge.origin import Origin

#: The file a tenant bundle declares its corpora in.
KNOWLEDGE_FILE = "knowledge.yaml"

#: Default ceiling per document. Large enough for a long policy manual, small enough that a
#: stray database dump in a documents folder fails loudly instead of filling the index.
DEFAULT_MAX_FILE_BYTES = 8 * 1024 * 1024

#: Directory names never descended into, regardless of the include patterns. These hold
#: version-control internals, virtualenvs and build output — never customer knowledge, and
#: each one is capable of contributing tens of thousands of files.
SKIPPED_DIRECTORIES = frozenset(
    {
        ".git", ".hg", ".svn", ".venv", "venv", "node_modules", "__pycache__",
        ".mypy_cache", ".pytest_cache", ".tox", ".idea", ".vscode", "dist", "build",
    }
)

#: Files NOVA itself writes into a corpus root, which are therefore never documents.
#: Excluded by name rather than by a dot-file rule: a customer whose corpus legitimately
#: holds ``.eslintrc.md`` should get it indexed, and a rule broad enough to catch this
#: marker would quietly drop theirs too.
RESERVED_FILES = frozenset({".nova-sync.json"})

#: Free-form in the sense that NOVA does not interpret it, but constrained to a short list
#: so the label means the same thing in every tenant's dashboard and every citation.
CLASSIFICATIONS = ("public", "internal", "confidential", "restricted")


@dataclass(frozen=True)
class KnowledgeSource:
    """One declared corpus."""

    id: str
    root: Path
    title: str = ""
    description: str = ""
    include: tuple[str, ...] = ("**/*",)
    exclude: tuple[str, ...] = ()
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
    classification: str = "internal"
    #: Where the documents come from, when they are mirrored rather than local. A corpus
    #: with an origin is a mirror: NOVA syncs into ``root`` and the ingest runs over the
    #: result, so nothing downstream knows the difference.
    origin: Optional["Origin"] = None
    source: Optional[Path] = None

    @property
    def display_title(self) -> str:
        return self.title or self.id

    @classmethod
    def parse(
        cls,
        data: Any,
        *,
        source: Optional[Path] = None,
        base_dir: Optional[Path] = None,
        env: Optional[Mapping[str, str]] = None,
        prefix: str = "",
    ) -> "KnowledgeSource":
        doc = Doc(data, source=source, prefix=prefix, env=env)
        identifier = doc.identifier("id")
        raw_root = doc.str_("root", required=True)

        root = Path(raw_root).expanduser()
        if not root.is_absolute():
            if base_dir is None:
                raise SpecError(
                    f"{raw_root!r} is relative but there is no bundle directory to resolve it "
                    "against; give an absolute path",
                    field="root",
                    source=source,
                )
            root = (base_dir / root).resolve()

        spec = cls(
            id=identifier,
            root=root,
            title=doc.str_("title"),
            description=doc.str_("description"),
            include=tuple(doc.str_list("include", default=["**/*"])),
            exclude=tuple(doc.str_list("exclude")),
            max_file_bytes=doc.int_(
                "max_file_bytes", default=DEFAULT_MAX_FILE_BYTES, minimum=1
            ),
            classification=doc.choice("classification", CLASSIFICATIONS, default="internal"),
            origin=Origin.parse(doc.child("origin")),
            source=source,
        )
        doc.reject_unknown()
        return spec

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "root": str(self.root),
            "title": self.title,
            "description": self.description,
            "include": list(self.include),
            "exclude": list(self.exclude),
            "max_file_bytes": self.max_file_bytes,
            "classification": self.classification,
            **({"origin": self.origin.to_dict()} if self.origin else {}),
        }


@dataclass(frozen=True)
class KnowledgeCatalog:
    """Every corpus one tenant has declared."""

    sources: tuple[KnowledgeSource, ...] = ()
    source_path: Optional[Path] = None

    @property
    def ids(self) -> frozenset[str]:
        return frozenset(spec.id for spec in self.sources)

    def get(self, source_id: str) -> KnowledgeSource:
        for spec in self.sources:
            if spec.id == source_id:
                return spec
        known = ", ".join(sorted(self.ids)) or "(none)"
        raise SpecError(f"no knowledge source {source_id!r}; declared sources: {known}")

    def subset(self, source_ids: Sequence[str]) -> tuple[KnowledgeSource, ...]:
        """The declared sources named by *source_ids*, in declaration order.

        Declaration order rather than the caller's order, so an agent's readable corpora
        are listed the same way everywhere they appear.
        """
        wanted = set(source_ids)
        return tuple(spec for spec in self.sources if spec.id in wanted)

    def to_dict(self) -> dict[str, Any]:
        return {"sources": [spec.to_dict() for spec in self.sources]}

    @classmethod
    def parse(
        cls,
        data: Any,
        *,
        source: Optional[Path] = None,
        base_dir: Optional[Path] = None,
        env: Optional[Mapping[str, str]] = None,
    ) -> "KnowledgeCatalog":
        doc = Doc(data or {}, source=source, env=env)
        raw = doc._raw("sources", []) or []
        if not isinstance(raw, list):
            raise SpecError("must be a list of source declarations", field="sources", source=source)

        sources: list[KnowledgeSource] = []
        seen: set[str] = set()
        for index, entry in enumerate(raw):
            spec = KnowledgeSource.parse(
                entry, source=source, base_dir=base_dir, env=env, prefix=f"sources[{index}]"
            )
            if spec.id in seen:
                raise SpecError(
                    f"duplicate knowledge source id {spec.id!r}",
                    field=f"sources[{index}].id",
                    source=source,
                )
            seen.add(spec.id)
            sources.append(spec)
        doc.reject_unknown()
        return cls(sources=tuple(sources), source_path=source)


def load_catalog(
    bundle_root: Path,
    *,
    env: Optional[Mapping[str, str]] = None,
) -> KnowledgeCatalog:
    """Read ``knowledge.yaml`` from a tenant bundle. An absent file means no corpora.

    Absent is a valid state, not a degraded one: a tenant that has declared no knowledge
    gets agents with no knowledge tool, which is exactly how every agent behaved before
    this capability existed.
    """
    import yaml

    path = bundle_root / KNOWLEDGE_FILE
    if not path.is_file():
        return KnowledgeCatalog()
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SpecError(f"could not be read: {exc.strerror or exc}", source=path) from exc
    except yaml.YAMLError as exc:
        raise SpecError(f"is not valid YAML: {exc}", source=path) from exc
    return KnowledgeCatalog.parse(data or {}, source=path, base_dir=bundle_root, env=env)


@dataclass(frozen=True)
class SkippedFile:
    """One file the walk declined, and why. Surfaced in the ingest report."""

    path: Path
    reason: str


@dataclass
class WalkResult:
    """Everything one walk of a source produced."""

    files: list[Path] = field(default_factory=list)
    skipped: list[SkippedFile] = field(default_factory=list)


def iter_documents(spec: KnowledgeSource) -> WalkResult:
    """Every file belonging to *spec*, in a deterministic order, with refusals recorded.

    Sorted because ingestion order decides chunk ids, and a corpus that reindexes to a
    different set of ids on every run cannot be diffed, audited or incrementally updated.
    """
    result = WalkResult()
    root = spec.root
    if not root.is_dir():
        result.skipped.append(SkippedFile(root, "source root is not a directory"))
        return result

    resolved_root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        here = Path(dirpath)
        # Prune in place so os.walk never descends. Sorted for determinism.
        dirnames[:] = sorted(
            name
            for name in dirnames
            if name not in SKIPPED_DIRECTORIES and not name.startswith(".")
        )
        for name in sorted(filenames):
            path = here / name
            relative = path.relative_to(root).as_posix()
            if relative in RESERVED_FILES:
                # NOVA's own sync marker. Silent rather than reported as skipped: it is not
                # something the customer put there, so naming it in an ingest report would
                # be NOVA telling an operator about NOVA.
                continue
            if not _matches(relative, spec.include):
                continue
            if spec.exclude and _matches(relative, spec.exclude):
                result.skipped.append(SkippedFile(path, "excluded by pattern"))
                continue
            skip = _refuse(path, resolved_root, spec.max_file_bytes)
            if skip is not None:
                result.skipped.append(SkippedFile(path, skip))
                continue
            result.files.append(path)
    return result


def _matches(relative: str, patterns: Sequence[str]) -> bool:
    """True when *relative* matches any glob in *patterns*."""
    return any(
        fnmatch.fnmatch(relative, candidate)
        for pattern in patterns
        for candidate in _candidates(pattern)
    )


def _candidates(pattern: str) -> tuple[str, ...]:
    """One customer-written pattern, as the fnmatch patterns that implement it.

    ``fnmatch`` has no concept of ``**``: its ``*`` already spans separators, so ``**/*.md``
    behaves as ``*/*.md`` and silently fails to match ``refunds.md`` at the corpus root.
    Everyone writing ``**/*.md`` means "every .md file, at any depth, including this one",
    so the ``**/`` prefix is also tried with the prefix removed — and a mid-pattern
    ``/**/`` with the segment collapsed, which is the same expectation one level in.

    Worth the eight lines. The alternative is a customer declaring a corpus, seeing an
    ingest report that says "indexed 4", and never learning that the four documents at the
    top of their handbook were the ones it left out.
    """
    out = [pattern]
    if pattern.startswith("**/"):
        out.append(pattern[3:])
    if "/**/" in pattern:
        out.append(pattern.replace("/**/", "/"))
    return tuple(dict.fromkeys(out))


def _refuse(path: Path, resolved_root: Path, max_bytes: int) -> Optional[str]:
    """Why this file must not be ingested, or None when it may be.

    A symlink is refused even when it points somewhere harmless. Following one would mean
    the set of ingested files is decided by the filesystem rather than by the declaration,
    and a link planted in a documents folder is the cheapest possible way to read a
    customer's private key into a corpus their agents can search.
    """
    if path.is_symlink():
        return "symlink (not followed)"
    try:
        if path.resolve().parent != path.parent.resolve():
            return "resolves outside its directory"
        if resolved_root not in path.resolve().parents:
            return "resolves outside the source root"
        stat = path.stat()
    except OSError as exc:
        return f"could not be read: {exc.strerror or exc}"
    if not path.is_file():
        return "not a regular file"
    if stat.st_size == 0:
        return "empty"
    if stat.st_size > max_bytes:
        return f"{stat.st_size} bytes exceeds max_file_bytes ({max_bytes})"
    return None
