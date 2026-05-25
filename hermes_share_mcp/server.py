"""Restricted Local Files / SMB Share MCP server for HermesShare.

The server intentionally exposes only a compact document-oriented API rooted at a
single share directory.  It never accepts absolute paths, resolves symlinks, and
rejects any path whose real location is outside the configured root.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import mimetypes
import os
import stat
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

DEFAULT_SHARE_ROOT = Path("/home/hermes/HermesShare")
DEFAULT_MAX_READ_BYTES = 256_000
DEFAULT_MAX_SEARCH_BYTES = 1_000_000
DEFAULT_MAX_WRITE_BYTES = 256_000
TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".log",
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".css",
    ".html",
    ".xml",
}
DOC_EXTENSIONS = TEXT_EXTENSIONS | {".pdf", ".doc", ".docx", ".odt", ".rtf", ".xlsx", ".xls"}
WRITE_EXTENSIONS = TEXT_EXTENSIONS - {".log"}


class SharePathError(ValueError):
    """Raised when a requested path is unsafe or outside the share root."""


@dataclass(frozen=True)
class ShareConfig:
    root: Path = DEFAULT_SHARE_ROOT
    allow_write: bool = True
    include_hidden: bool = False
    max_read_bytes: int = DEFAULT_MAX_READ_BYTES
    max_search_bytes: int = DEFAULT_MAX_SEARCH_BYTES
    max_write_bytes: int = DEFAULT_MAX_WRITE_BYTES

    @classmethod
    def from_env(cls) -> "ShareConfig":
        root = Path(os.environ.get("HERMES_SHARE_MCP_ROOT", str(DEFAULT_SHARE_ROOT))).expanduser()
        allow_write = _env_bool("HERMES_SHARE_MCP_ALLOW_WRITE", True)
        include_hidden = _env_bool("HERMES_SHARE_MCP_INCLUDE_HIDDEN", False)
        max_read_bytes = _env_int("HERMES_SHARE_MCP_MAX_READ_BYTES", DEFAULT_MAX_READ_BYTES)
        max_search_bytes = _env_int("HERMES_SHARE_MCP_MAX_SEARCH_BYTES", DEFAULT_MAX_SEARCH_BYTES)
        max_write_bytes = _env_int("HERMES_SHARE_MCP_MAX_WRITE_BYTES", DEFAULT_MAX_WRITE_BYTES)
        return cls(
            root=root,
            allow_write=allow_write,
            include_hidden=include_hidden,
            max_read_bytes=max_read_bytes,
            max_search_bytes=max_search_bytes,
            max_write_bytes=max_write_bytes,
        )

    @property
    def resolved_root(self) -> Path:
        return self.root.expanduser().resolve(strict=False)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _is_hidden(path: Path, root: Path) -> bool:
    try:
        rel_parts = path.relative_to(root).parts
    except ValueError:
        return True
    return any(part.startswith(".") for part in rel_parts if part not in {".", ".."})


def _looks_text(path: Path) -> bool:
    if path.suffix.lower() in TEXT_EXTENSIONS:
        return True
    guessed, _ = mimetypes.guess_type(path.name)
    return bool(guessed and guessed.startswith("text/"))


def _compact_error(code: str, message: str) -> dict[str, Any]:
    return {"ok": False, "error": {"code": code, "message": message}}


class ShareMCPService:
    """Document-oriented operations rooted at one local share directory."""

    def __init__(self, config: ShareConfig | None = None):
        self.config = config or ShareConfig.from_env()
        self.root = self.config.resolved_root

    def _resolve(self, relative_path: str | None = "", *, for_write: bool = False) -> Path:
        raw = (relative_path or "").strip()
        if raw in {"", "."}:
            candidate = self.root
        else:
            requested = Path(raw)
            if requested.is_absolute():
                raise SharePathError("absolute paths are not allowed; use a path relative to the share root")
            if any(part in {"..", ""} for part in requested.parts):
                raise SharePathError("path traversal segments are not allowed")
            candidate = (self.root / requested).resolve(strict=False)
        if not _is_relative_to(candidate, self.root):
            raise SharePathError("resolved path is outside the configured share root")
        if not self.config.include_hidden and _is_hidden(candidate, self.root):
            raise SharePathError("hidden files and directories are not exposed")
        if for_write and candidate.suffix.lower() not in WRITE_EXTENSIONS:
            raise SharePathError(f"writes are limited to text document extensions: {', '.join(sorted(WRITE_EXTENSIONS))}")
        return candidate

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix() or "."

    def _entry(self, path: Path) -> dict[str, Any]:
        st = path.stat()
        return {
            "path": self._relative(path),
            "name": path.name,
            "type": "directory" if path.is_dir() else "file",
            "size": st.st_size if path.is_file() else None,
            "modified": _utc_iso(st.st_mtime),
            "extension": path.suffix.lower() if path.is_file() else None,
        }

    def _safe_walk(self, base: Path) -> Iterable[Path]:
        for current, dirs, files in os.walk(base):
            current_path = Path(current).resolve(strict=False)
            if not _is_relative_to(current_path, self.root):
                dirs[:] = []
                continue
            if not self.config.include_hidden:
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                files = [f for f in files if not f.startswith(".")]
            for name in dirs + files:
                path = (current_path / name).resolve(strict=False)
                if _is_relative_to(path, self.root):
                    yield path

    def list_share_files(
        self,
        path: str = "",
        pattern: str = "*",
        recursive: bool = False,
        limit: int = 100,
        include_dirs: bool = True,
    ) -> dict[str, Any]:
        try:
            base = self._resolve(path)
            if not base.exists():
                return _compact_error("not_found", "path does not exist")
            if not base.is_dir():
                return _compact_error("not_directory", "path is not a directory")
            limit = max(1, min(int(limit or 100), 500))
            candidates = self._safe_walk(base) if recursive else ((base / child.name).resolve(strict=False) for child in base.iterdir())
            entries: list[dict[str, Any]] = []
            for item in candidates:
                if not _is_relative_to(item, self.root):
                    continue
                if not self.config.include_hidden and _is_hidden(item, self.root):
                    continue
                if item.is_dir() and not include_dirs:
                    continue
                rel = self._relative(item)
                if not fnmatch.fnmatch(item.name, pattern) and not fnmatch.fnmatch(rel, pattern):
                    continue
                entries.append(self._entry(item))
                if len(entries) >= limit:
                    break
            entries.sort(key=lambda e: (e["type"] != "directory", e["path"].lower()))
            return {"ok": True, "root": str(self.root), "path": self._relative(base), "entries": entries, "truncated": len(entries) >= limit}
        except SharePathError as exc:
            return _compact_error("unsafe_path", str(exc))
        except OSError as exc:
            return _compact_error("io_error", str(exc))

    def read_shared_doc(self, path: str, max_bytes: int | None = None) -> dict[str, Any]:
        try:
            target = self._resolve(path)
            if not target.exists():
                return _compact_error("not_found", "file does not exist")
            if not target.is_file():
                return _compact_error("not_file", "path is not a file")
            if not _looks_text(target):
                return _compact_error("unsupported_type", "only text-like documents can be read inline")
            max_len = max(1, min(int(max_bytes or self.config.max_read_bytes), self.config.max_read_bytes))
            data = target.read_bytes()
            truncated = len(data) > max_len
            text = data[:max_len].decode("utf-8", errors="replace")
            return {"ok": True, "path": self._relative(target), "bytes": len(data), "truncated": truncated, "content": text}
        except SharePathError as exc:
            return _compact_error("unsafe_path", str(exc))
        except OSError as exc:
            return _compact_error("io_error", str(exc))

    def write_shared_doc(self, path: str, content: str, overwrite: bool = False) -> dict[str, Any]:
        try:
            if not self.config.allow_write:
                return _compact_error("write_disabled", "writes are disabled for this server")
            target = self._resolve(path, for_write=True)
            encoded = content.encode("utf-8")
            if len(encoded) > self.config.max_write_bytes:
                return _compact_error("too_large", f"content exceeds max_write_bytes ({self.config.max_write_bytes})")
            if target.exists() and not overwrite:
                return _compact_error("exists", "file exists; pass overwrite=true to replace it")
            target.parent.mkdir(parents=True, exist_ok=True)
            parent = target.parent.resolve(strict=False)
            if not _is_relative_to(parent, self.root):
                return _compact_error("unsafe_path", "parent path escapes share root")
            target.write_text(content, encoding="utf-8")
            return {"ok": True, "path": self._relative(target), "bytes_written": len(encoded), "overwritten": overwrite}
        except SharePathError as exc:
            return _compact_error("unsafe_path", str(exc))
        except OSError as exc:
            return _compact_error("io_error", str(exc))

    def search_share(
        self,
        query: str,
        path: str = "",
        glob: str = "*",
        limit: int = 50,
        max_file_bytes: int | None = None,
    ) -> dict[str, Any]:
        try:
            if not query:
                return _compact_error("invalid_query", "query is required")
            base = self._resolve(path)
            if not base.exists():
                return _compact_error("not_found", "path does not exist")
            limit = max(1, min(int(limit or 50), 200))
            max_len = max(1, min(int(max_file_bytes or self.config.max_search_bytes), self.config.max_search_bytes))
            needle = query.lower()
            results: list[dict[str, Any]] = []
            files = [base] if base.is_file() else [p for p in self._safe_walk(base) if p.is_file()]
            for file_path in files:
                rel = self._relative(file_path)
                if not fnmatch.fnmatch(file_path.name, glob) and not fnmatch.fnmatch(rel, glob):
                    continue
                if not _looks_text(file_path) or file_path.stat().st_size > max_len:
                    continue
                text = file_path.read_text(encoding="utf-8", errors="replace")
                for line_no, line in enumerate(text.splitlines(), start=1):
                    idx = line.lower().find(needle)
                    if idx < 0:
                        continue
                    start = max(0, idx - 80)
                    end = min(len(line), idx + len(query) + 80)
                    results.append({"path": rel, "line": line_no, "snippet": line[start:end]})
                    if len(results) >= limit:
                        return {"ok": True, "query": query, "matches": results, "truncated": True}
            return {"ok": True, "query": query, "matches": results, "truncated": False}
        except SharePathError as exc:
            return _compact_error("unsafe_path", str(exc))
        except (OSError, UnicodeError) as exc:
            return _compact_error("io_error", str(exc))

    def get_recent_files(self, path: str = "", limit: int = 25, extensions: list[str] | None = None) -> dict[str, Any]:
        try:
            base = self._resolve(path)
            if not base.exists():
                return _compact_error("not_found", "path does not exist")
            limit = max(1, min(int(limit or 25), 100))
            allowed = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions or []}
            files = [base] if base.is_file() else [p for p in self._safe_walk(base) if p.is_file()]
            entries = [self._entry(p) for p in files if (not allowed or p.suffix.lower() in allowed)]
            entries.sort(key=lambda e: e["modified"], reverse=True)
            return {"ok": True, "files": entries[:limit], "truncated": len(entries) > limit}
        except SharePathError as exc:
            return _compact_error("unsafe_path", str(exc))
        except OSError as exc:
            return _compact_error("io_error", str(exc))

    def sync_status(self) -> dict[str, Any]:
        try:
            exists = self.root.exists()
            writable = bool(exists and os.access(self.root, os.W_OK))
            readable = bool(exists and os.access(self.root, os.R_OK))
            file_count = 0
            newest: str | None = None
            if exists and readable:
                mtimes: list[float] = []
                for path in self._safe_walk(self.root):
                    if path.is_file():
                        file_count += 1
                        mtimes.append(path.stat().st_mtime)
                        if file_count >= 10_000:
                            break
                if mtimes:
                    newest = _utc_iso(max(mtimes))
            mode = self.root.stat().st_mode if exists else 0
            return {
                "ok": True,
                "root": str(self.root),
                "exists": exists,
                "readable": readable,
                "writable": writable and self.config.allow_write,
                "write_configured": self.config.allow_write,
                "file_count_sample": file_count,
                "newest_file_modified": newest,
                "root_mode": stat.filemode(mode) if exists else None,
            }
        except OSError as exc:
            return _compact_error("io_error", str(exc))


def _register_tools(service: ShareMCPService):
    try:
        from mcp.server.fastmcp import FastMCP
    except Exception as exc:  # pragma: no cover - exercised only without optional dep
        raise SystemExit("The 'mcp' package is required. Install with: pip install 'hermes-agent[mcp]'") from exc

    mcp = FastMCP("HermesShare Local Files")

    @mcp.tool()
    def list_share_files(path: str = "", pattern: str = "*", recursive: bool = False, limit: int = 100, include_dirs: bool = True) -> dict[str, Any]:
        """List files/directories under the configured share root with path safety."""
        return service.list_share_files(path, pattern, recursive, limit, include_dirs)

    @mcp.tool()
    def search_share(query: str, path: str = "", glob: str = "*", limit: int = 50, max_file_bytes: int | None = None) -> dict[str, Any]:
        """Search text-like files in the share and return compact line snippets."""
        return service.search_share(query, path, glob, limit, max_file_bytes)

    @mcp.tool()
    def read_shared_doc(path: str, max_bytes: int | None = None) -> dict[str, Any]:
        """Read a UTF-8 text-like document from the share root."""
        return service.read_shared_doc(path, max_bytes)

    @mcp.tool()
    def write_shared_doc(path: str, content: str, overwrite: bool = False) -> dict[str, Any]:
        """Write a UTF-8 text document inside the share root if writes are enabled."""
        return service.write_shared_doc(path, content, overwrite)

    @mcp.tool()
    def get_recent_files(path: str = "", limit: int = 25, extensions: list[str] | None = None) -> dict[str, Any]:
        """Return recently modified files under the share root."""
        return service.get_recent_files(path, limit, extensions)

    @mcp.tool()
    def sync_status() -> dict[str, Any]:
        """Report share root readability/writability and a compact freshness sample."""
        return service.sync_status()

    return mcp


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Restricted MCP stdio server for HermesShare / SMB share files.")
    parser.add_argument("--root", default=os.environ.get("HERMES_SHARE_MCP_ROOT", str(DEFAULT_SHARE_ROOT)), help="Share root to expose (default: /home/hermes/HermesShare)")
    parser.add_argument("--read-only", action="store_true", help="Disable write_shared_doc")
    parser.add_argument("--include-hidden", action="store_true", help="Expose dotfiles/dot directories inside the root")
    parser.add_argument("--max-read-bytes", type=int, default=DEFAULT_MAX_READ_BYTES)
    parser.add_argument("--max-search-bytes", type=int, default=DEFAULT_MAX_SEARCH_BYTES)
    parser.add_argument("--max-write-bytes", type=int, default=DEFAULT_MAX_WRITE_BYTES)
    parser.add_argument("--print-config-snippet", action="store_true", help="Print a Hermes mcp_servers config snippet and exit")
    return parser


def config_snippet(root: str = str(DEFAULT_SHARE_ROOT), read_only: bool = False) -> str:
    args = ["hermes-share-mcp", "--root", root]
    if read_only:
        args.append("--read-only")
    return "mcp_servers:\n  hermesshare:\n    command: \"{}\"\n    args: {}\n    timeout: 30\n    connect_timeout: 30\n".format(args[0], json.dumps(args[1:]))


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.print_config_snippet:
        print(config_snippet(args.root, args.read_only))
        return
    service = ShareMCPService(
        ShareConfig(
            root=Path(args.root),
            allow_write=not args.read_only and _env_bool("HERMES_SHARE_MCP_ALLOW_WRITE", True),
            include_hidden=args.include_hidden or _env_bool("HERMES_SHARE_MCP_INCLUDE_HIDDEN", False),
            max_read_bytes=args.max_read_bytes,
            max_search_bytes=args.max_search_bytes,
            max_write_bytes=args.max_write_bytes,
        )
    )
    mcp = _register_tools(service)
    mcp.run()


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
