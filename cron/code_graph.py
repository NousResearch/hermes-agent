"""Per-service code knowledge graph — the server-side builder.

A ``service`` node in the cron dataflow graph declares the code behind it
(``source_files``). This module turns that file set into a knowledge graph of
the code — nodes for modules/class/functions, typed edges for imports/calls,
community clusters — by running ``graphify``'s deterministic AST extraction over
just those files (isolated in ``tools/code_graph_runner.py``), then mapping its
grammar onto the same ``{source, target, type, class}`` typed-edge shape the
Portal renderer already speaks.

Freshness is derive-on-read with a content-digest cache: ``source_files`` is
excluded from the configuration digest and services are excluded from the
changeset log, so a "regenerate when the definition changes" trigger cannot ride
changesets. Instead the graph is keyed on a digest of the file *contents*, so it
is rebuilt exactly when the code the service points at changes, and served from
cache otherwise (a graphify subprocess only spawns on a miss).

Everything here re-enforces the browse-root allowlist itself: graphify reads the
filesystem directly, bypassing ``files.read``, so the ``repo``/``hermes``
containment, the 1 MiB size cap and the binary-file skip are applied here or not
at all.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

# Mirrors ``tui_gateway.files_browse`` — the roots a source file may be opened
# under, and the read caps the file browser enforces.
_ALLOWED_ROOTS = {"repo", "hermes"}
_MAX_FILE_BYTES = 1024 * 1024

_RUNNER = Path(__file__).resolve().parent.parent / "tools" / "code_graph_runner.py"

# graphify's ``relation`` vocabulary, mapped onto our edge class. "Flow" is the
# system-flow subgraph (who imports/calls whom) both clients emphasise;
# "structure" is nesting/inheritance; everything else is a plain reference.
_FLOW_RELATIONS = {"calls", "imports", "imports_from"}
_STRUCTURE_RELATIONS = {
    "contains",
    "implements",
    "inherits",
    "extends",
    "subclass",
    "defines",
    "method_of",
}

_CODE_EXTENSIONS = (
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".go", ".swift", ".rb",
    ".rs", ".java", ".kt", ".c", ".h", ".cpp", ".cc", ".hpp", ".m", ".mm", ".sh",
)


class CodeGraphUnavailable(RuntimeError):
    """The code graph could not be built (no readable files, or graphify failed).

    Callers should treat this as a soft failure — "code graph unavailable" — not
    a server error: a graphify hiccup must never take down the dataflow surface.
    """


# --------------------------------------------------------------------------- #
# File selection + digests
# --------------------------------------------------------------------------- #
def _browse_roots() -> Dict[str, Path]:
    # Imported lazily to avoid a circular import (``cron.jobs`` imports this
    # module for the node stamp).
    from cron.jobs import _source_file_roots

    return _source_file_roots()


def _reverify_root(path: Path, roots: Dict[str, Path]):
    """``(root name, rel)`` of the allowlisted browse root containing ``path``.

    We never trust the ``root`` an entry claims — graphify will read the file
    directly, so containment is re-derived here against the live roots.
    ``(None, None)`` when no allowlisted root contains it.
    """
    from cron.jobs import _browse_root_for

    name, rel = _browse_root_for(path, roots)
    if name in _ALLOWED_ROOTS and rel is not None:
        return name, rel
    return None, None


def eligible_source_files(source_files: Any) -> List[Dict[str, Any]]:
    """The subset of resolved ``source_files`` entries safe to feed to graphify.

    Keeps only files that exist, resolve onto an allowlisted browse root, are
    within the 1 MiB cap, and decode as UTF-8 (binary skipped). Reads each file's
    bytes to compute a content hash. Returns ``{path, root, rel, sha, size}``
    sorted by ``(root, rel)`` for a deterministic digest and node mapping.
    """
    roots = _browse_roots()
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for entry in source_files or []:
        if not isinstance(entry, dict) or not entry.get("exists"):
            continue
        if entry.get("root") not in _ALLOWED_ROOTS or not entry.get("rel"):
            continue
        raw_path = entry.get("path")
        if not raw_path:
            continue
        path = Path(raw_path)
        key = str(path)
        if key in seen:
            continue
        root_name, rel = _reverify_root(path, roots)
        if root_name is None:
            continue
        try:
            if not path.is_file() or path.stat().st_size > _MAX_FILE_BYTES:
                continue
            data = path.read_bytes()
        except OSError:
            continue
        if b"\x00" in data:  # binary — same sniff the file browser uses
            continue
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        seen.add(key)
        out.append(
            {
                "path": key,
                "root": root_name,
                "rel": rel,
                "sha": hashlib.sha256(data).hexdigest(),
                "size": len(data),
            }
        )
    out.sort(key=lambda item: (item["root"], item["rel"]))
    return out


def _content_digest(eligible: List[Dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for item in eligible:
        digest.update(
            f"{item['root']}\x00{item['rel']}\x00{item['size']}\x00{item['sha']}\n".encode()
        )
    return digest.hexdigest()


def source_files_digest(source_files: Any) -> str:
    """Content digest of a service's resolved ``source_files`` — the cache key.

    Changes exactly when the bytes of the in-root files change, which is what
    makes the code graph track the current definition without a changeset.
    """
    return _content_digest(eligible_source_files(source_files))


def service_code_graph_stamp(source_files: Any) -> Optional[str]:
    """A cheap change-token for the service node — no file reads.

    Hashed over ``(root, rel, size, mtime_ns)`` of the in-root files, so it flips
    when the declared code changes without paying the content hash on every
    ``cron.graph`` fetch (the authoritative content digest is computed lazily in
    ``build_service_code_graph``). Over-invalidation is fine; under-invalidation
    is not, so mtime is included.
    """
    parts = []
    for entry in source_files or []:
        if not isinstance(entry, dict) or not entry.get("exists"):
            continue
        if entry.get("root") not in _ALLOWED_ROOTS or not entry.get("rel"):
            continue
        try:
            stat = os.stat(entry["path"])
        except OSError:
            continue
        parts.append((entry["root"], entry["rel"], stat.st_size, stat.st_mtime_ns))
    if not parts:
        return None
    parts.sort()
    digest = hashlib.sha256()
    for root, rel, size, mtime in parts:
        digest.update(f"{root}\x00{rel}\x00{size}\x00{mtime}\n".encode())
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# graphify subprocess
# --------------------------------------------------------------------------- #
def _graphify_python() -> str:
    """The interpreter to run the graphify runner with.

    Honours ``HERMES_GRAPHIFY_PYTHON`` (point it at a venv that has ``graphifyy``
    installed), else the current interpreter — which works when ``graphifyy`` is
    installed in the gateway's own environment.
    """
    override = os.environ.get("HERMES_GRAPHIFY_PYTHON")
    if override and Path(override).exists():
        return override
    return sys.executable


def _run_graphify(files: List[str], *, root: Optional[str], timeout: float) -> Dict[str, Any]:
    interpreter = _graphify_python()
    job = json.dumps({"files": files, "root": root, "directed": True})
    try:
        proc = subprocess.run(
            [interpreter, str(_RUNNER)],
            input=job,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise CodeGraphUnavailable(f"graphify timed out after {timeout}s") from exc
    except OSError as exc:
        raise CodeGraphUnavailable(f"cannot launch graphify: {exc}") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or "").strip()[:400]
        raise CodeGraphUnavailable(f"graphify failed: {detail}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise CodeGraphUnavailable("graphify produced invalid output") from exc


# --------------------------------------------------------------------------- #
# Grammar mapping
# --------------------------------------------------------------------------- #
def _edge_class(relation: str) -> str:
    if relation in _FLOW_RELATIONS:
        return "flow"
    if relation in _STRUCTURE_RELATIONS:
        return "structure"
    return "reference"


def _looks_like_module(label: str) -> bool:
    return label.endswith(_CODE_EXTENSIONS)


def _derive_kind(node: Dict[str, Any]) -> str:
    """Structural kind from graphify's node shape.

    graphify emits no explicit class/func kind, so derive it: a node with no
    ``source_file`` is an external/imported symbol; a filename label is the
    module node; ``_callable`` marks a function/method; an upper-cased label is a
    class; anything else is a plain symbol (module-level var/const).
    """
    source_file = node.get("source_file") or ""
    label = node.get("label") or ""
    if not source_file:
        return "external"
    if _looks_like_module(label):
        return "module"
    if node.get("_callable"):
        return "func"
    if label[:1].isupper():
        return "class"
    return "symbol"


def _parse_line(source_location: Any) -> Optional[int]:
    if not isinstance(source_location, str):
        return None
    match = re.match(r"L(\d+)", source_location)
    return int(match.group(1)) if match else None


def _assemble(
    service_id: str,
    digest: str,
    code_control: Optional[Dict[str, Any]],
    source_index: Dict[str, Dict[str, Any]],
    raw: Dict[str, Any],
) -> Dict[str, Any]:
    node_community: Dict[str, str] = {}
    for cid, members in (raw.get("communities") or {}).items():
        for member in members:
            node_community[member] = str(cid)

    nodes: List[Dict[str, Any]] = []
    for node in raw.get("nodes") or []:
        node_id = node.get("id")
        if not node_id:
            continue
        source_file = node.get("source_file") or ""
        entry = source_index.get(source_file)
        rel = entry["rel"] if entry else None
        nodes.append(
            {
                "id": node_id,
                "kind": _derive_kind(node),
                "type": _derive_kind(node),
                "label": node.get("label") or node_id,
                "path": rel,
                "root": entry["root"] if entry else None,
                "rel": rel,
                "line": _parse_line(node.get("source_location")),
                "community": node_community.get(node_id),
            }
        )
    nodes.sort(key=lambda item: item["id"])

    edges: List[Dict[str, Any]] = []
    for edge in raw.get("edges") or []:
        source = edge.get("source")
        target = edge.get("target")
        if not source or not target:
            continue
        relation = edge.get("relation") or "references"
        edges.append(
            {
                "source": source,
                "target": target,
                "type": relation,
                "class": _edge_class(relation),
                "confidence": edge.get("confidence") or "EXTRACTED",
            }
        )
    edges.sort(key=lambda item: (item["source"], item["target"], item["type"]))

    communities = {
        str(cid): sorted(members)
        for cid, members in (raw.get("communities") or {}).items()
    }

    return {
        "service": service_id,
        "digest": digest,
        "code_control": code_control,
        "nodes": nodes,
        "edges": edges,
        "communities": communities,
    }


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
def _cache_dir(service_id: str) -> Optional[Path]:
    try:
        from hermes_constants import get_hermes_home

        base = get_hermes_home().resolve() / "code_graphs"
    except Exception:
        return None
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", service_id) or "service"
    return base / safe


def _cache_key(digest: str, revision: Optional[str]) -> str:
    return f"{digest}_{revision or 'none'}"


def _cache_load(service_id: str, digest: str, revision: Optional[str]) -> Optional[Dict[str, Any]]:
    directory = _cache_dir(service_id)
    if directory is None:
        return None
    path = directory / f"{_cache_key(digest, revision)}.json"
    try:
        return json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _cache_store(service_id: str, digest: str, revision: Optional[str], payload: Dict[str, Any]) -> None:
    directory = _cache_dir(service_id)
    if directory is None:
        return
    try:
        directory.mkdir(parents=True, exist_ok=True)
        # Evict older versions for this service — only the current one matters.
        current = f"{_cache_key(digest, revision)}.json"
        for stale in directory.glob("*.json"):
            if stale.name != current:
                try:
                    stale.unlink()
                except OSError:
                    pass
        fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        os.replace(tmp, directory / current)
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# Public entry
# --------------------------------------------------------------------------- #
def build_service_code_graph(
    service_id: str,
    *,
    source_files: Any,
    code_control: Any = None,
    timeout: float = 30.0,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Build (or fetch from cache) the code knowledge graph for one service.

    Raises :class:`CodeGraphUnavailable` when there is nothing renderable — no
    in-root readable files, or graphify is missing/failed/timed out.
    """
    eligible = eligible_source_files(source_files)
    if not eligible:
        raise CodeGraphUnavailable("no readable in-root source files")

    digest = _content_digest(eligible)

    provenance: Optional[Dict[str, Any]] = None
    revision: Optional[str] = None
    if isinstance(code_control, dict) and code_control.get("status") == "verified":
        revision = code_control.get("revision")
        provenance = {
            "repository": code_control.get("repository"),
            "revision": code_control.get("revision"),
            "pull_request": code_control.get("pull_request"),
        }

    if use_cache:
        cached = _cache_load(service_id, digest, revision)
        if cached is not None:
            return cached

    # Report each node's source file relative to a common base so graphify does
    # not collapse absolute paths to a bare (possibly colliding) basename; index
    # the eligible entries by that same relative path to remap root/rel back on.
    paths = [item["path"] for item in eligible]
    common_base = os.path.commonpath([os.path.dirname(p) for p in paths])
    source_index = {os.path.relpath(item["path"], common_base): item for item in eligible}

    raw = _run_graphify(paths, root=common_base, timeout=timeout)
    payload = _assemble(service_id, digest, provenance, source_index, raw)

    if use_cache:
        _cache_store(service_id, digest, revision, payload)
    return payload
