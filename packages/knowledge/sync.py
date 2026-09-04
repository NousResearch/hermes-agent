"""Incremental document synchronization (Step 7).

Sources: Obsidian vault, markdown trees, PDFs, git repos, MkDocs sites,
READMEs, and Hermes conversation transcripts. Detects new / updated /
deleted files via a checksum manifest and only pushes the delta.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .types import Document, SyncReport

logger = logging.getLogger("hermes.knowledge.sync")

TEXT_EXTS = {".md", ".markdown", ".txt", ".rst", ".mdx"}
CODE_EXTS = {".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java",
             ".kt", ".rb", ".sh", ".sql", ".yaml", ".yml", ".toml"}
PDF_EXTS = {".pdf"}
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".obsidian",
             "dist", "build", ".next", ".cache", "site-packages", ".sandbox"}
MAX_BYTES = 2_000_000


def _doc_id(path: str) -> str:
    return hashlib.sha1(os.path.abspath(path).encode()).hexdigest()[:20]


def _read_pdf(path: str) -> str:
    try:
        import fitz  # type: ignore  (pymupdf)

        with fitz.open(path) as doc:
            return "\n".join(page.get_text() for page in doc)
    except Exception:
        pass
    try:
        out = subprocess.run(["pdftotext", path, "-"], capture_output=True,
                             timeout=60, text=True)
        if out.returncode == 0:
            return out.stdout
    except Exception:
        pass
    logger.warning("sync: cannot extract PDF text from %s (install pymupdf or poppler)", path)
    return ""


def _title_from(path: str, content: str) -> str:
    m = re.search(r"^\s*#\s+(.+)$", content[:2000], re.M)
    if m:
        return m.group(1).strip()
    m = re.search(r"^title:\s*(.+)$", content[:800], re.M)
    if m:
        return m.group(1).strip().strip("\"'")
    return os.path.splitext(os.path.basename(path))[0]


def read_document(path: str, source: str, workspace: str,
                  root: str = "") -> Optional[Document]:
    ext = os.path.splitext(path)[1].lower()
    try:
        if os.path.getsize(path) > MAX_BYTES:
            return None
    except OSError:
        return None
    if ext in PDF_EXTS:
        content = _read_pdf(path)
    elif ext in TEXT_EXTS or ext in CODE_EXTS:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read()
        except OSError:
            return None
    else:
        return None
    if not content.strip():
        return None
    meta: Dict[str, Any] = {"ext": ext}
    if root:
        meta["relpath"] = os.path.relpath(path, root)
    if ext in CODE_EXTS:
        meta["kind"] = "code"
    return Document(
        id=_doc_id(path),
        title=_title_from(path, content),
        content=content,
        path=os.path.abspath(path),
        source=source,
        workspace=workspace,
        metadata=meta,
        mtime=os.path.getmtime(path),
    )


def walk_source(root: str, source: str, workspace: str,
                include_code: bool = False,
                exts: Optional[Sequence[str]] = None) -> Iterable[Document]:
    allowed = set(exts) if exts else (TEXT_EXTS | PDF_EXTS | (CODE_EXTS if include_code else set()))
    root = os.path.abspath(os.path.expanduser(root))
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".hermes")]
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() not in allowed:
                continue
            doc = read_document(os.path.join(dirpath, fn), source, workspace, root)
            if doc:
                yield doc


def conversation_documents(limit: int = 200, workspace: str = "default") -> List[Document]:
    """Index previous Hermes conversations from the SQLite session store."""
    docs: List[Document] = []
    try:
        from hermes_constants import get_hermes_home  # type: ignore

        db = os.path.join(str(get_hermes_home()), "state.db")
    except Exception:
        db = os.path.expanduser("~/.hermes/state.db")
    if not os.path.exists(db):
        return docs
    import sqlite3

    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=10)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT session_id, group_concat(content, '\n') AS body,"
            " max(rowid) AS r FROM messages WHERE role IN ('user','assistant')"
            " GROUP BY session_id ORDER BY r DESC LIMIT ?", (limit,),
        ).fetchall()
        conn.close()
    except Exception as exc:
        logger.warning("sync: conversation indexing skipped: %s", exc)
        return docs
    for r in rows:
        body = r["body"] or ""
        if not body.strip():
            continue
        docs.append(Document(
            id=f"conv-{r['session_id']}",
            title=f"Conversation {r['session_id']}",
            content=body[:200_000],
            path=f"session://{r['session_id']}",
            source="conversation",
            workspace=workspace,
        ))
    return docs


class SyncManifest:
    """Persistent {document_id: checksum} state for delta detection."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.data: Dict[str, Dict[str, Any]] = {}
        try:
            with open(path) as fh:
                self.data = json.load(fh)
        except Exception:
            self.data = {}

    def save(self) -> None:
        tmp = f"{self.path}.tmp"
        with open(tmp, "w") as fh:
            json.dump(self.data, fh, indent=1)
        os.replace(tmp, self.path)

    def ids_for(self, source_key: str) -> List[str]:
        return [k for k, v in self.data.items() if v.get("source_key") == source_key]


class DocumentSynchronizer:
    """Delta sync: only changed files hit the provider."""

    def __init__(self, service, manifest_path: Optional[str] = None):
        self.service = service
        if manifest_path is None:
            base = os.path.dirname(service.config.db_path)
            manifest_path = os.path.join(base, "sync_manifest.json")
        self.manifest = SyncManifest(manifest_path)

    def sync_documents(self, documents: Iterable[Document], source_key: str,
                       prune: bool = True) -> SyncReport:
        t0 = time.perf_counter()
        report = SyncReport()
        seen: set = set()
        for doc in documents:
            seen.add(doc.id)
            prev = self.manifest.data.get(doc.id)
            if prev and prev.get("checksum") == doc.checksum:
                report.unchanged.append(doc.path or doc.id)
                continue
            try:
                res = self.service.update(doc) if prev else self.service.index(doc)
                if not res.ok:
                    report.failed.append(f"{doc.path}: {res.detail}")
                    continue
                (report.updated if prev else report.added).append(doc.path or doc.id)
                self.manifest.data[doc.id] = {
                    "checksum": doc.checksum, "path": doc.path,
                    "source_key": source_key, "synced_at": time.time(),
                }
            except Exception as exc:
                report.failed.append(f"{doc.path}: {exc}")

        if prune:
            for stale in self.manifest.ids_for(source_key):
                if stale in seen:
                    continue
                entry = self.manifest.data.get(stale, {})
                p = entry.get("path", "")
                if p and not p.startswith("session://") and os.path.exists(p):
                    continue  # still on disk but filtered out; don't delete
                try:
                    self.service.delete(stale)
                    report.deleted.append(p or stale)
                    self.manifest.data.pop(stale, None)
                except Exception as exc:
                    report.failed.append(f"delete {stale}: {exc}")

        self.manifest.save()
        report.elapsed_ms = (time.perf_counter() - t0) * 1000
        return report

    # -- source-typed helpers -------------------------------------------
    def sync_path(self, root: str, source: str = "markdown",
                  workspace: Optional[str] = None, include_code: bool = False,
                  exts: Optional[Sequence[str]] = None) -> SyncReport:
        ws = workspace or self.service.config.workspace
        key = f"{source}:{os.path.abspath(os.path.expanduser(root))}"
        return self.sync_documents(
            walk_source(root, source, ws, include_code=include_code, exts=exts), key)

    def sync_obsidian(self, vault: str, workspace: Optional[str] = None) -> SyncReport:
        return self.sync_path(vault, source="obsidian", workspace=workspace)

    def sync_git_repo(self, repo: str, workspace: Optional[str] = None,
                      include_code: bool = True) -> SyncReport:
        return self.sync_path(repo, source="git", workspace=workspace,
                              include_code=include_code)

    def sync_mkdocs(self, root: str, workspace: Optional[str] = None) -> SyncReport:
        docs_dir = os.path.join(root, "docs")
        return self.sync_path(docs_dir if os.path.isdir(docs_dir) else root,
                              source="mkdocs", workspace=workspace)

    def sync_conversations(self, limit: int = 200,
                           workspace: Optional[str] = None) -> SyncReport:
        ws = workspace or self.service.config.workspace
        return self.sync_documents(conversation_documents(limit, ws),
                                   "conversation:hermes", prune=False)

    def sync_configured(self) -> Dict[str, Dict[str, Any]]:
        """Run every source listed in config ``knowledge.sync_sources``."""
        out: Dict[str, Dict[str, Any]] = {}
        for src in self.service.config.sync_sources or []:
            typ = src.get("type", "markdown")
            path = src.get("path", "")
            ws = src.get("workspace")
            try:
                if typ == "obsidian":
                    rep = self.sync_obsidian(path, ws)
                elif typ == "git":
                    rep = self.sync_git_repo(path, ws, src.get("include_code", True))
                elif typ == "mkdocs":
                    rep = self.sync_mkdocs(path, ws)
                elif typ == "conversations":
                    rep = self.sync_conversations(src.get("limit", 200), ws)
                else:
                    rep = self.sync_path(path, typ, ws, src.get("include_code", False))
                out[f"{typ}:{path or 'hermes'}"] = rep.to_dict()
            except Exception as exc:
                out[f"{typ}:{path}"] = {"error": str(exc)}
        return out
