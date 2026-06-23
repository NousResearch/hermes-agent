"""SiYuan adapter for static knowledge documents.

Uses SiYuan's HTTP API only; it never edits workspace `.sy` files directly.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any, Protocol
import urllib.error
import urllib.request

from hermes_constants import get_hermes_home
from knowledge.adapters.base import ExistingKnowledge, WriteResult
from knowledge.types import KnowledgeWriteRequest

DEFAULT_ENDPOINT = "http://127.0.0.1:6806"
DEFAULT_NOTEBOOK = "Hermes"
MAX_RETURN_TEXT_CHARS = 20_000
MAX_SQL_CHARS = 2_000


@dataclass(frozen=True)
class SiYuanConfig:
    endpoint: str
    api_token: str
    default_notebook: str = DEFAULT_NOTEBOOK

    def redacted(self) -> dict[str, str]:
        return {
            "endpoint": self.endpoint,
            "default_notebook": self.default_notebook,
            "api_token": "[REDACTED]" if self.api_token else "",
        }


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    env: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def load_config() -> SiYuanConfig:
    data: dict[str, Any] = {}
    config_path = get_hermes_home() / "siyuan" / "config.json"
    if config_path.exists():
        data = json.loads(config_path.read_text(encoding="utf-8"))

    secret_env = _read_env_file(Path("/opt/siyuan/secrets/siyuan.env"))
    endpoint = (
        os.getenv("SIYUAN_ENDPOINT")
        or data.get("local_endpoint")
        or data.get("endpoint")
        or DEFAULT_ENDPOINT
    ).rstrip("/")
    token = (
        os.getenv("SIYUAN_API_TOKEN")
        or data.get("api_token")
        or secret_env.get("SIYUAN_API_TOKEN")
        or ""
    )
    notebook = os.getenv("SIYUAN_DEFAULT_NOTEBOOK") or data.get("default_notebook") or DEFAULT_NOTEBOOK
    return SiYuanConfig(endpoint=endpoint, api_token=token, default_notebook=notebook)


class SiYuanAPIError(RuntimeError):
    pass


class SiYuanPoster(Protocol):
    def post(self, path: str, payload: dict[str, Any]) -> Any: ...


class SiYuanClient:
    def __init__(self, config: SiYuanConfig | None = None):
        self.config = config or load_config()
        if not self.config.api_token:
            raise SiYuanAPIError("SiYuan API token is not configured")

    def post(self, path: str, payload: dict[str, Any]) -> Any:
        if not path.startswith("/"):
            path = "/" + path
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.config.endpoint + path,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Token {self.config.api_token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise SiYuanAPIError(f"HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise SiYuanAPIError(str(exc)) from exc

        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            raise SiYuanAPIError(f"Invalid JSON response: {raw[:200]}") from exc
        if isinstance(parsed, dict) and parsed.get("code", 0) != 0:
            raise SiYuanAPIError(parsed.get("msg") or f"SiYuan API error code={parsed.get('code')}")
        return parsed.get("data") if isinstance(parsed, dict) and "data" in parsed else parsed


def _slugify_title(title: str) -> str:
    text = re.sub(r"[\\/]+", "-", title.strip())
    text = re.sub(r'[\x00-\x1f:*?"<>|#\[\]{}]+', "-", text)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-.")
    return text[:120] or "Untitled"


def doc_path_for(title: str, path: str = "") -> str:
    if path:
        clean = path.strip()
        return clean if clean.startswith("/") else "/" + clean
    return "/" + _slugify_title(title)


def _truncate(text: str, limit: int = MAX_RETURN_TEXT_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


class SiYuanKnowledgeAdapter:
    def __init__(self, client: SiYuanClient | None = None):
        self.client = client or SiYuanClient(load_config())

    @property
    def name(self) -> str:
        return "siyuan"

    def ensure_notebook(self, name: str) -> str:
        data = self.client.post("/api/notebook/lsNotebooks", {})
        notebooks = data.get("notebooks", []) if isinstance(data, dict) else []
        for notebook in notebooks:
            if notebook.get("name") == name:
                return notebook["id"]
        created = self.client.post("/api/notebook/createNotebook", {"name": name})
        notebook = created.get("notebook", created) if isinstance(created, dict) else {}
        notebook_id = notebook.get("id")
        if not notebook_id:
            raise SiYuanAPIError(f"Failed to create notebook {name!r}")
        return notebook_id

    def search_existing(self, request: KnowledgeWriteRequest) -> list[ExistingKnowledge]:
        query = (request.title or request.idempotency_key or "").strip()
        if not query:
            return []
        data = self.client.post(
            "/api/search/fullTextSearchBlock",
            {"query": query[:500], "method": 0, "orderBy": 0, "groupBy": 0, "page": 1, "paths": []},
        )
        blocks = data.get("blocks", []) if isinstance(data, dict) else []
        results: list[ExistingKnowledge] = []
        normalized_title = (request.title or "").strip().casefold()
        for block in blocks[:10]:
            hpath = str(block.get("hPath") or block.get("path") or "")
            content = str(block.get("content") or "")
            score = 1.0 if normalized_title and normalized_title in (hpath + " " + content).casefold() else 0.5
            results.append(
                ExistingKnowledge(
                    id=str(block.get("id") or ""),
                    title=request.title or content[:80],
                    path=hpath,
                    score=score,
                    metadata={"type": block.get("type")},
                )
            )
        return results

    def write(self, request: KnowledgeWriteRequest) -> WriteResult:
        notebook = request.notebook or self.client.config.default_notebook
        notebook_id = self.ensure_notebook(notebook)
        path = doc_path_for(request.title, request.path)
        data = self.client.post(
            "/api/filetree/createDocWithMd",
            {"notebook": notebook_id, "path": path, "markdown": request.content},
        )
        doc_id = ""
        if isinstance(data, dict):
            doc_id = str(data.get("id") or data.get("rootID") or "")
        return WriteResult(
            success=True,
            backend=self.name,
            action="create",
            id=doc_id,
            path=path,
            data={"notebook": notebook, "notebook_id": notebook_id, "data": data},
        )

    def update(self, existing: ExistingKnowledge, request: KnowledgeWriteRequest) -> WriteResult:
        if not existing.id:
            return self.write(request)
        data = self.client.post(
            "/api/block/updateBlock",
            {"id": existing.id, "dataType": "markdown", "data": request.content},
        )
        return WriteResult(
            success=True,
            backend=self.name,
            action="update",
            id=existing.id,
            path=existing.path,
            data={"data": data},
        )


def append_block(client: SiYuanPoster, block_id: str, markdown: str) -> dict[str, Any]:
    data = client.post("/api/block/appendBlock", {"dataType": "markdown", "data": markdown, "parentID": block_id})
    return {"success": True, "data": data}


def export_markdown(client: SiYuanPoster, block_id: str) -> dict[str, Any]:
    data = client.post("/api/export/exportMdContent", {"id": block_id})
    markdown = data.get("content", data) if isinstance(data, dict) else data
    return {"success": True, "markdown": _truncate(str(markdown))}


def update_block(client: SiYuanPoster, block_id: str, markdown: str) -> dict[str, Any]:
    data = client.post("/api/block/updateBlock", {"id": block_id, "dataType": "markdown", "data": markdown})
    return {"success": True, "data": data}


def search_blocks(client: SiYuanPoster, query: str, page: int = 1) -> dict[str, Any]:
    data = client.post(
        "/api/search/fullTextSearchBlock",
        {"query": query[:500], "method": 0, "orderBy": 0, "groupBy": 0, "page": max(int(page or 1), 1), "paths": []},
    )
    blocks = data.get("blocks", []) if isinstance(data, dict) else []
    compact = [
        {"id": b.get("id"), "type": b.get("type"), "content": _truncate(str(b.get("content", "")), 800), "hPath": b.get("hPath")}
        for b in blocks[:20]
    ]
    return {"success": True, "count": len(blocks), "blocks": compact}


def run_sql(client: SiYuanPoster, stmt: str, allow_unsafe: bool = False) -> dict[str, Any]:
    sql = (stmt or "").strip()
    if not sql:
        raise ValueError("stmt is required")
    if len(sql) > MAX_SQL_CHARS:
        raise ValueError(f"stmt exceeds {MAX_SQL_CHARS} characters")
    simple_select = sql.lower().startswith("select") and ";" not in sql[:-1]
    if not simple_select and not allow_unsafe:
        raise ValueError("Only simple SELECT statements are supported unless allow_unsafe=true")
    if simple_select and " limit " not in sql.lower():
        sql = sql.rstrip(";") + " limit 20"
    return {"success": True, "rows": client.post("/api/query/sql", {"stmt": sql})}
