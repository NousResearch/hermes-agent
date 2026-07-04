"""Tool handlers and URI/resource helpers for OpenViking memory."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse
from urllib.request import url2pathname

from tools.registry import tool_error

from .constants import (
    _CATEGORY_SUBDIR_MAP,
    _DEFAULT_MEMORY_SUBDIR,
    _GENERATED_MEMORY_SUMMARY_FILENAMES,
    _READ_BATCH_FULL_LIMIT,
    _READ_BATCH_LIMIT,
    _REMOTE_RESOURCE_PREFIXES,
)

logger = logging.getLogger(__name__)


def _zip_directory(dir_path: Path) -> Path:
    """Create a temporary zip file containing a directory tree."""
    root = dir_path.resolve()
    zip_path = Path(tempfile.gettempdir()) / f"openviking_upload_{uuid.uuid4().hex}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in dir_path.rglob("*"):
            if file_path.is_symlink():
                continue
            if file_path.is_file():
                try:
                    file_path.resolve().relative_to(root)
                except ValueError:
                    continue
                arcname = str(file_path.relative_to(dir_path)).replace("\\", "/")
                zipf.write(file_path, arcname=arcname)
    return zip_path


def _is_windows_absolute_path(value: str) -> bool:
    return (
        len(value) >= 3
        and value[0].isalpha()
        and value[1] == ":"
        and value[2] in {"/", "\\"}
    )


def _is_remote_resource_source(value: str) -> bool:
    return value.startswith(_REMOTE_RESOURCE_PREFIXES)


def _memory_segment_index(parts: List[str]) -> Optional[int]:
    if len(parts) >= 2 and parts[0] == "user" and parts[1] == "memories":
        return 1
    if len(parts) >= 3 and parts[0] == "user" and parts[2] == "memories":
        return 2
    if len(parts) >= 4 and parts[0] == "user" and parts[1] == "peers" and parts[3] == "memories":
        return 3
    if len(parts) >= 5 and parts[0] == "user" and parts[2] == "peers" and parts[4] == "memories":
        return 4
    return None


def _validate_forget_memory_uri(raw_uri: Any) -> tuple[Optional[str], Optional[str]]:
    if not isinstance(raw_uri, str):
        return None, "uri is required"

    uri = raw_uri.strip()
    if not uri:
        return None, "uri is required"

    parsed = urlparse(uri)
    if parsed.scheme != "viking" or not uri.startswith("viking://"):
        return None, "viking_forget only accepts viking:// memory file URIs"
    if parsed.query or parsed.fragment:
        return None, "viking_forget requires an exact URI without query or fragment"
    if uri.endswith("/") or not uri.endswith(".md"):
        return None, "viking_forget only deletes concrete .md memory files"

    parts = [part for part in uri[len("viking://") :].split("/") if part]
    memories_idx = _memory_segment_index(parts)
    if memories_idx is None or len(parts) < memories_idx + 2:
        return None, "viking_forget only deletes user memory file URIs"

    filename = uri.rsplit("/", 1)[-1]
    if filename in _GENERATED_MEMORY_SUMMARY_FILENAMES:
        return None, "viking_forget cannot delete generated memory summary files"

    return uri, None


def _is_local_path_reference(value: str) -> bool:
    if not value or "\n" in value or "\r" in value:
        return False
    if _is_remote_resource_source(value):
        return False
    if _is_windows_absolute_path(value):
        return True
    return (
        value.startswith(("/", "./", "../", "~/", ".\\", "..\\", "~\\"))
        or "/" in value
        or "\\" in value
    )


def _path_from_file_uri(uri: str) -> Path | str:
    parsed = urlparse(uri)
    if parsed.netloc not in {"", "localhost"}:
        return f"Unsupported non-local file URI: {uri}"
    return Path(url2pathname(parsed.path)).expanduser()



class OpenVikingToolMixin:
    """OpenViking model-tool handlers used by OpenVikingMemoryProvider."""

    @staticmethod
    def _unwrap_result(resp: Any) -> Any:
        """Return OpenViking payload body regardless of wrapped/unwrapped shape."""
        if isinstance(resp, dict) and "result" in resp:
            return resp.get("result")
        return resp

    @staticmethod
    def _normalize_summary_uri(uri: str) -> str:
        """Map pseudo summary files to their parent directory URI for L0/L1 reads."""
        if not uri:
            return uri
        for suffix in ("/.abstract.md", "/.overview.md", "/.read.md", "/.full.md"):
            if uri.endswith(suffix):
                return uri[: -len(suffix)] or "viking://"
        return uri

    def _is_directory_uri(self, uri: str) -> bool | None:
        """Probe fs/stat to decide if a URI is a directory.

        Returns True/False when the server answers cleanly, and None when the
        probe itself fails (network error, unexpected shape). Callers should
        treat None as "unknown" and fall back to the exception-based path.
        """
        try:
            resp = self._client.get("/api/v1/fs/stat", params={"uri": uri})
        except Exception:
            return None
        result = self._unwrap_result(resp)
        if isinstance(result, dict):
            if "isDir" in result:
                return bool(result.get("isDir"))
            if "is_dir" in result:
                return bool(result.get("is_dir"))
            if result.get("type") == "dir":
                return True
            if result.get("type") == "file":
                return False
        return None

    def _tool_search(self, args: dict) -> str:
        query = args.get("query", "")
        if not query:
            return tool_error("query is required")

        payload: Dict[str, Any] = {"query": query}
        mode = args.get("mode", "auto")
        if args.get("scope"):
            payload["target_uri"] = args["scope"]
        if args.get("limit"):
            payload["limit"] = args["limit"]

        endpoint = "/api/v1/search/search" if mode == "deep" else "/api/v1/search/find"
        if endpoint == "/api/v1/search/search" and self._session_id:
            payload["session_id"] = self._session_id

        resp = self._client.post(endpoint, payload)
        result = resp.get("result", {})

        # Format results for the model — keep it concise
        scored_entries = []
        for ctx_type in ("memories", "resources", "skills"):
            items = result.get(ctx_type, [])
            for item in items:
                raw_score = item.get("score")
                sort_score = raw_score if raw_score is not None else 0.0
                entry = {
                    "uri": item.get("uri", ""),
                    "type": ctx_type.rstrip("s"),
                    "score": round(raw_score, 3) if raw_score is not None else 0.0,
                    "abstract": item.get("abstract", ""),
                }
                if item.get("relations"):
                    entry["related"] = [r.get("uri") for r in item["relations"][:3]]
                scored_entries.append((sort_score, entry))

        scored_entries.sort(key=lambda x: x[0], reverse=True)
        formatted = [entry for _, entry in scored_entries]

        return json.dumps({
            "results": formatted,
            "total": result.get("total", len(formatted)),
        }, ensure_ascii=False)

    def _read_uri_payload(
        self,
        uri: str,
        level: str,
        *,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        summary_level = level in {"abstract", "overview"}
        # OpenViking expects directory URIs for pseudo summary files
        # (e.g. viking://user/hermes/.overview.md).
        resolved_uri = self._normalize_summary_uri(uri) if summary_level else uri
        used_fallback = False

        # abstract/overview endpoints are directory-only on OpenViking
        # (v0.3.x returns 500/412 for file URIs). When the caller asks for a
        # summary level on a non-pseudo URI, probe fs/stat first and route
        # file URIs straight to /content/read instead of eating a failing
        # round-trip. The pseudo-URI path already points at a directory, so
        # skip the probe there.
        if summary_level and resolved_uri == uri:
            is_dir = self._is_directory_uri(uri)
            if is_dir is False:
                resolved_uri = uri
                used_fallback = True

        # Map our level names to OpenViking GET endpoints.
        endpoint = "/api/v1/content/read"
        if not used_fallback:
            if level == "abstract":
                endpoint = "/api/v1/content/abstract"
            elif level == "overview":
                endpoint = "/api/v1/content/overview"

        try:
            resp = self._client.get(endpoint, params={"uri": resolved_uri})
        except Exception:
            # OpenViking may return HTTP 500 for abstract/overview reads on normal
            # file URIs (mem_*.md). For those, gracefully fallback to full read.
            if not summary_level or resolved_uri != uri or used_fallback:
                raise
            resp = self._client.get("/api/v1/content/read", params={"uri": uri})
            used_fallback = True

        result = self._unwrap_result(resp)
        # Content endpoints may return either plain strings or objects.
        if isinstance(result, str):
            content = result
        elif isinstance(result, dict):
            content = result.get("content", "") or result.get("text", "")
        else:
            content = ""

        # Truncate long content to avoid flooding context.
        max_len = 8000
        if level == "overview":
            max_len = 4000
        elif level == "abstract":
            max_len = 1200
        if limit is not None:
            max_len = max(200, min(max_len, limit))

        if len(content) > max_len:
            content = content[:max_len] + "\n\n[... truncated, use a more specific URI or full level]"

        payload = {
            "uri": uri,
            "resolved_uri": resolved_uri,
            "level": level,
            "content": content,
        }
        if used_fallback:
            payload["fallback"] = "content/read"

        return payload

    def _tool_read(self, args: dict) -> str:
        level = args.get("level", "overview")
        uri_arg = args.get("uri", "")
        uris_arg = args.get("uris", [])

        raw_uris: List[Any]
        batch_requested = bool(uris_arg) or isinstance(uri_arg, list)
        if isinstance(uris_arg, list) and uris_arg:
            raw_uris = uris_arg
        elif isinstance(uri_arg, list):
            raw_uris = uri_arg
        elif isinstance(uri_arg, str) and uri_arg:
            raw_uris = [uri_arg]
        else:
            return tool_error("uri or uris is required")

        uris: List[str] = []
        seen: Set[str] = set()
        for raw_uri in raw_uris:
            if not isinstance(raw_uri, str):
                continue
            uri = raw_uri.strip()
            if not uri or uri in seen:
                continue
            seen.add(uri)
            uris.append(uri)

        if not uris:
            return tool_error("uri or uris is required")

        selected = uris[:_READ_BATCH_LIMIT]
        per_item_limit = (
            _READ_BATCH_FULL_LIMIT
            if len(selected) > 1 and level == "full"
            else None
        )
        if len(selected) == 1 and not batch_requested:
            return json.dumps(
                self._read_uri_payload(selected[0], level),
                ensure_ascii=False,
            )

        results: List[Dict[str, Any]] = []
        for uri in selected:
            try:
                results.append(
                    self._read_uri_payload(uri, level, limit=per_item_limit)
                )
            except Exception as e:
                results.append({"uri": uri, "level": level, "error": str(e)})

        return json.dumps(
            {
                "level": level,
                "results": results,
                "requested": len(uris),
                "returned": len(results),
                "truncated": len(uris) > len(selected),
            },
            ensure_ascii=False,
        )

    def _tool_browse(self, args: dict) -> str:
        action = args.get("action", "list")
        path = args.get("path", "viking://")

        # Map action to the correct fs endpoint (all GET with uri= param)
        endpoint_map = {"tree": "/api/v1/fs/tree", "list": "/api/v1/fs/ls", "stat": "/api/v1/fs/stat"}
        endpoint = endpoint_map.get(action, "/api/v1/fs/ls")
        resp = self._client.get(endpoint, params={"uri": path})
        result = self._unwrap_result(resp)

        # Format list/tree results for readability
        if action in {"list", "tree"}:
            raw_entries = result
            if isinstance(result, dict):
                raw_entries = result.get("entries") or result.get("items") or result.get("children") or []

            if isinstance(raw_entries, list):
                entries = []
                for e in raw_entries[:50]:  # cap at 50 entries
                    uri = e.get("uri", "")
                    name = e.get("rel_path") or e.get("name") or (uri.rsplit("/", 1)[-1] if uri else "")
                    is_dir = bool(e.get("isDir") or e.get("is_dir") or e.get("type") == "dir")
                    entries.append({
                        "name": name,
                        "uri": uri,
                        "type": "dir" if is_dir else "file",
                        "abstract": e.get("abstract", ""),
                    })
                return json.dumps({"path": path, "entries": entries}, ensure_ascii=False)

        return json.dumps(result, ensure_ascii=False)

    def _tool_remember(self, args: dict) -> str:
        content = args.get("content", "")
        if not content:
            return tool_error("content is required")

        category = args.get("category", "")
        subdir = _CATEGORY_SUBDIR_MAP.get(category, _DEFAULT_MEMORY_SUBDIR)
        uri = self._build_memory_uri(subdir)

        # Write directly via content/write API.
        # This creates the file, stores the content, and queues vector indexing
        # in a single call — no dependency on session commit / VLM extraction.
        try:
            result = self._client.post("/api/v1/content/write", {
                "uri": uri,
                "content": content,
                "mode": "create",
            })
            written = result.get("result", {}).get("written_bytes", 0)
            return json.dumps({
                "status": "stored",
                "message": f"Memory stored ({written}b) and queued for vector indexing.",
            })
        except Exception as e:
            logger.error("OpenViking content/write failed: %s", e)
            return tool_error(f"Failed to store memory: {e}")

    def _tool_forget(self, args: dict) -> str:
        uri, error = _validate_forget_memory_uri(args.get("uri"))
        if error:
            return tool_error(error)

        resp = self._client.delete(
            "/api/v1/fs",
            params={"uri": uri, "recursive": False},
        )
        result = self._unwrap_result(resp)
        payload: Dict[str, Any] = {"status": "deleted", "uri": uri}
        if isinstance(result, dict):
            payload["uri"] = result.get("uri") or uri
            for key in (
                "estimated_deleted_count",
                "memory_cleanup",
                "semantic_root_uri",
                "semantic_status",
                "queue_status",
            ):
                if key in result:
                    payload[key] = result[key]

        return json.dumps(payload, ensure_ascii=False)

    def _tool_add_resource(self, args: dict) -> str:
        url = args.get("url", "")
        if not url:
            return tool_error("url is required")

        if args.get("to") and args.get("parent"):
            return tool_error("Cannot specify both 'to' and 'parent'")

        payload: Dict[str, Any] = {}
        for key in ("reason", "to", "parent", "instruction", "wait", "timeout"):
            if key in args and args[key] not in {None, ""}:
                payload[key] = args[key]

        parsed_url = urlparse(url)
        if _is_remote_resource_source(url):
            source_path = None
        elif parsed_url.scheme == "file":
            source_path = _path_from_file_uri(url)
            if isinstance(source_path, str):
                return tool_error(source_path)
        elif parsed_url.scheme and not _is_windows_absolute_path(url):
            source_path = None
        else:
            source_path = Path(url).expanduser()

        cleanup_path: Optional[Path] = None
        try:
            if source_path is not None:
                if source_path.exists():
                    if source_path.is_dir():
                        payload["source_name"] = source_path.name
                        cleanup_path = _zip_directory(source_path)
                        upload_path = cleanup_path
                    elif source_path.is_file():
                        payload["source_name"] = source_path.name
                        upload_path = source_path
                    else:
                        return tool_error(f"Unsupported local resource path: {url}")
                    payload["temp_file_id"] = self._client.upload_temp_file(upload_path)
                elif _is_local_path_reference(url):
                    return tool_error(f"Local resource path does not exist: {url}")
                else:
                    payload["path"] = url
            else:
                payload["path"] = url

            resp = self._client.post("/api/v1/resources", payload)
            result = resp.get("result", {})
        finally:
            if cleanup_path:
                cleanup_path.unlink(missing_ok=True)

        return json.dumps({
            "status": "added",
            "root_uri": result.get("root_uri", ""),
            "message": "Resource queued for processing. Use viking_search after a moment to find it.",
        }, ensure_ascii=False)
