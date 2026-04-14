#!/usr/bin/env python3
"""Build a canonical Obsidian memory registry surface from export INDEX.json.

This script creates a stable, query-friendly registry snapshot that links:
- source note path
- note id + aliases (when available)
- deterministic lookup tokens
- source->chunk cross references

Primary output is a latest alias surface under state/continuity/latest.
Optional history is appended as compact JSONL for drift/debug visibility.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_INDEX_REL = Path("memory/obsidian_export_chunks/INDEX.json")
DEFAULT_LATEST_REL = Path("state/continuity/latest/xk_obsidian_memory_registry_latest.json")
DEFAULT_HISTORY_REL = Path("state/continuity/obsidian/xk_obsidian_memory_registry_history.jsonl")

SCHEMA = "clawd.obsidian_memory_registry_surface.v1"

FRONTMATTER_KV_RE = re.compile(r"^([A-Za-z0-9_-]+)\s*:\s*(.*?)\s*$")
TOKEN_SPACE_RE = re.compile(r"\s+")
TOKEN_SAFE_RE = re.compile(r"[^a-z0-9._/\-]+")


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def load_json_object(path: Path, *, label: str) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"{label} not found: {path}")
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"failed to read {label} {path}: {exc}")

    if not isinstance(payload, dict):
        raise SystemExit(f"{label} must be a JSON object: {path}")

    return payload


def split_frontmatter(text: str) -> Tuple[Optional[str], str]:
    if not text.startswith("---\n") and text.strip() != "---":
        return None, text

    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return None, text

    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            fm = "\n".join(lines[1:i])
            body = "\n".join(lines[i + 1 :])
            return fm, body

    return None, text


def parse_frontmatter_fields(frontmatter: Optional[str]) -> Dict[str, str]:
    if not frontmatter:
        return {}

    out: Dict[str, str] = {}
    for raw in frontmatter.split("\n"):
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line.startswith(" ") or line.startswith("\t"):
            continue

        m = FRONTMATTER_KV_RE.match(line)
        if not m:
            continue

        key = m.group(1).strip().lower()
        value = m.group(2).strip()
        if not value:
            continue

        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]

        out[key] = value.strip()

    return out


def parse_inline_aliases(raw: str) -> List[str]:
    text = raw.strip()
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()

    if not text:
        return []

    out: List[str] = []
    for part in text.split(","):
        candidate = part.strip().strip('"').strip("'")
        if candidate:
            out.append(candidate)
    return out


def parse_frontmatter_aliases(frontmatter: Optional[str]) -> List[str]:
    if not frontmatter:
        return []

    lines = frontmatter.splitlines()
    aliases: List[str] = []

    i = 0
    while i < len(lines):
        raw = lines[i].rstrip()
        stripped = raw.strip()
        i += 1

        if not stripped or stripped.startswith("#"):
            continue

        if not stripped.lower().startswith(("aliases:", "alias:")):
            continue

        key, _, rest = stripped.partition(":")
        _ = key
        rest = rest.strip()

        if rest:
            aliases.extend(parse_inline_aliases(rest))
            continue

        while i < len(lines):
            candidate = lines[i].rstrip()
            if not candidate.strip():
                i += 1
                continue

            lstripped = candidate.lstrip()
            if not candidate.startswith(" ") and not candidate.startswith("\t"):
                break

            token = lstripped
            if token.startswith("- "):
                value = token[2:].strip().strip('"').strip("'")
                if value:
                    aliases.append(value)
            i += 1

    deduped = sorted({a.strip() for a in aliases if a and a.strip()})
    return deduped


def parse_note_frontmatter(note_path: Path) -> Tuple[Optional[str], List[str]]:
    if not note_path.exists() or not note_path.is_file():
        return None, []

    text = note_path.read_text(encoding="utf-8", errors="replace")
    frontmatter, _body = split_frontmatter(text)
    fields = parse_frontmatter_fields(frontmatter)
    aliases = parse_frontmatter_aliases(frontmatter)

    note_id = fields.get("id")
    note_id = note_id.strip() if isinstance(note_id, str) and note_id.strip() else None
    return note_id, aliases


def normalize_token(raw: str) -> str:
    text = (raw or "").strip().lower()
    if not text:
        return ""
    text = TOKEN_SPACE_RE.sub("-", text)
    text = text.replace("\\", "/")
    text = TOKEN_SAFE_RE.sub("", text)
    text = text.strip("./-")
    return text


def build_lookup_tokens(source_path: str, note_id: Optional[str], aliases: Sequence[str]) -> List[str]:
    tokens: Set[str] = set()

    rel = Path(source_path).with_suffix("").as_posix()
    stem = Path(source_path).stem

    for raw in [rel, stem, note_id or "", *aliases]:
        token = normalize_token(raw)
        if token:
            tokens.add(token)

    return sorted(tokens)


def coerce_aliases(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [x for x in parse_inline_aliases(raw) if x]
    if isinstance(raw, list):
        out = []
        for item in raw:
            text = str(item).strip()
            if text:
                out.append(text)
        return sorted(set(out))
    return []


def relpath_or_abs(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()).as_posix())
    except Exception:
        return str(path.resolve())


def build_surface(index_payload: Dict[str, Any], *, index_path: Path) -> Dict[str, Any]:
    source_files = index_payload.get("source_files")
    chunks = index_payload.get("chunks")

    if not isinstance(source_files, list):
        raise SystemExit("index payload missing list field: source_files")
    if not isinstance(chunks, list):
        raise SystemExit("index payload missing list field: chunks")

    vault_root_raw = str(index_payload.get("vault_root") or "").strip()
    vault_root = Path(vault_root_raw).expanduser().resolve() if vault_root_raw else None

    chunk_ids_by_source: Dict[str, List[str]] = {}
    for row in chunks:
        if not isinstance(row, dict):
            continue
        source_path = str(row.get("source_path") or "").strip()
        chunk_id = str(row.get("chunk_id") or "").strip()
        if not source_path or not chunk_id:
            continue
        chunk_ids_by_source.setdefault(source_path, []).append(chunk_id)

    for source_path in chunk_ids_by_source:
        chunk_ids_by_source[source_path] = sorted(set(chunk_ids_by_source[source_path]))

    source_registry: List[Dict[str, Any]] = []
    token_to_sources: Dict[str, Set[str]] = {}
    token_to_note_ids: Dict[str, Set[str]] = {}

    for row in sorted(source_files, key=lambda x: str(x.get("path") if isinstance(x, dict) else "")):
        if not isinstance(row, dict):
            continue

        source_path = str(row.get("path") or "").strip()
        if not source_path:
            continue

        row_note_id = str(row.get("note_id") or "").strip() or None
        row_aliases = coerce_aliases(row.get("aliases"))

        fm_note_id: Optional[str] = None
        fm_aliases: List[str] = []
        if vault_root is not None:
            note_path = (vault_root / source_path).resolve()
            fm_note_id, fm_aliases = parse_note_frontmatter(note_path)

        note_id = row_note_id or fm_note_id
        aliases = sorted({*row_aliases, *fm_aliases})

        lookup_tokens = build_lookup_tokens(source_path=source_path, note_id=note_id, aliases=aliases)

        for token in lookup_tokens:
            token_to_sources.setdefault(token, set()).add(source_path)
            if note_id:
                token_to_note_ids.setdefault(token, set()).add(note_id)

        chunk_ids = chunk_ids_by_source.get(source_path, [])

        entry: Dict[str, Any] = {
            "source_path": source_path,
            "note_id": note_id,
            "aliases": aliases,
            "lookup_tokens": lookup_tokens,
            "doc_type": row.get("doc_type"),
            "trust_level": row.get("trust_level"),
            "trust_score": row.get("trust_score"),
            "created": row.get("created"),
            "updated": row.get("updated"),
            "chunk_count": int(row.get("chunk_count") or len(chunk_ids)),
            "chunk_ids": chunk_ids,
        }

        if row.get("git_commit"):
            entry["git_commit"] = row.get("git_commit")

        source_registry.append(entry)

    lookup_registry: List[Dict[str, Any]] = []
    collisions: List[Dict[str, Any]] = []
    for token in sorted(token_to_sources.keys()):
        source_paths = sorted(token_to_sources[token])
        note_ids = sorted(token_to_note_ids.get(token, set()))
        row = {
            "token": token,
            "source_paths": source_paths,
            "source_count": len(source_paths),
            "note_ids": note_ids,
        }
        lookup_registry.append(row)
        if len(source_paths) > 1:
            collisions.append(row)

    doc_type_counts: Dict[str, int] = {}
    trust_level_counts: Dict[str, int] = {}
    for row in source_registry:
        doc_type = str(row.get("doc_type") or "unknown")
        trust_level = str(row.get("trust_level") or "unknown")
        doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + 1
        trust_level_counts[trust_level] = trust_level_counts.get(trust_level, 0) + 1

    index_sha = sha256_text(index_path.read_text(encoding="utf-8"))

    surface: Dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at_utc": now_iso(),
        "input": {
            "index_path": str(index_path),
            "index_sha256": f"sha256:{index_sha}",
            "vault_root": str(vault_root) if vault_root is not None else None,
        },
        "summary": {
            "source_count": len(source_registry),
            "chunk_count": len(chunks),
            "lookup_token_count": len(lookup_registry),
            "collision_token_count": len(collisions),
            "doc_type_counts": {k: doc_type_counts[k] for k in sorted(doc_type_counts.keys())},
            "trust_level_counts": {k: trust_level_counts[k] for k in sorted(trust_level_counts.keys())},
        },
        "source_registry": source_registry,
        "lookup_registry": lookup_registry,
        "collision_registry": collisions,
    }

    stable_copy = dict(surface)
    stable_copy.pop("generated_at_utc", None)
    fingerprint = sha256_text(stable_json(stable_copy))
    surface["registry_fingerprint"] = f"sha256:{fingerprint}"
    return surface


def normalized_for_compare(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload)
    out.pop("generated_at_utc", None)
    return out


def write_latest_if_changed(path: Path, payload: Dict[str, Any]) -> bool:
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(current, dict) and normalized_for_compare(current) == normalized_for_compare(payload):
                return False
        except Exception:  # noqa: BLE001
            pass

    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return True


def append_history(path: Path, payload: Dict[str, Any], *, repo_root: Path, latest_path: Path) -> None:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    input_blob = payload.get("input") if isinstance(payload.get("input"), dict) else {}

    row = {
        "generated_at_utc": payload.get("generated_at_utc"),
        "registry_fingerprint": payload.get("registry_fingerprint"),
        "index_sha256": input_blob.get("index_sha256"),
        "source_count": summary.get("source_count"),
        "chunk_count": summary.get("chunk_count"),
        "lookup_token_count": summary.get("lookup_token_count"),
        "collision_token_count": summary.get("collision_token_count"),
        "latest_surface": relpath_or_abs(latest_path, repo_root),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build canonical Obsidian memory registry surface from export INDEX")
    p.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT), help="Workspace repo root")
    p.add_argument("--index", default=str(DEFAULT_INDEX_REL), help="INDEX.json path (repo-relative allowed)")
    p.add_argument("--latest-out", default=str(DEFAULT_LATEST_REL), help="Latest registry output path")
    p.add_argument("--history-log", default=str(DEFAULT_HISTORY_REL), help="History JSONL log path")
    p.add_argument("--no-history", action="store_true", help="Do not append history log")
    p.add_argument("--pretty", action="store_true", help="Pretty-print summary output")
    return p.parse_args(argv)


def resolve_path(repo_root: Path, raw: str) -> Path:
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (repo_root / candidate).resolve()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    repo_root = Path(args.repo_root).expanduser().resolve()
    if not repo_root.exists() or not repo_root.is_dir():
        print(json.dumps({"ok": False, "error": f"repo root not found: {repo_root}"}, ensure_ascii=False), file=sys.stderr)
        return 2

    index_path = resolve_path(repo_root, args.index)
    latest_path = resolve_path(repo_root, args.latest_out)
    history_path = resolve_path(repo_root, args.history_log)

    index_payload = load_json_object(index_path, label="obsidian export index")
    surface = build_surface(index_payload, index_path=index_path)

    latest_written = write_latest_if_changed(latest_path, surface)

    history_appended = False
    if not args.no_history and latest_written:
        append_history(history_path, surface, repo_root=repo_root, latest_path=latest_path)
        history_appended = True

    summary = {
        "ok": True,
        "schema": SCHEMA,
        "repo_root": str(repo_root),
        "index_path": str(index_path),
        "latest_path": str(latest_path),
        "history_path": None if args.no_history else str(history_path),
        "latest_written": latest_written,
        "history_appended": history_appended,
        "registry_fingerprint": surface.get("registry_fingerprint"),
        "source_count": surface.get("summary", {}).get("source_count"),
        "chunk_count": surface.get("summary", {}).get("chunk_count"),
        "lookup_token_count": surface.get("summary", {}).get("lookup_token_count"),
        "collision_token_count": surface.get("summary", {}).get("collision_token_count"),
    }

    if args.pretty:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
