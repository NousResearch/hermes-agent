from __future__ import annotations

import json
from typing import Any

import httpx

from ..config import BrokerSettings
from ..core import IdempotencyStore, ToolExecutionError, result_payload


class NotionProvider:
    def __init__(self, settings: BrokerSettings, store: IdempotencyStore) -> None:
        self.settings = settings
        self.store = store

    def _client(self) -> httpx.Client:
        api_key = self.settings.notion_api_key or "placeholder"
        return httpx.Client(
            base_url="https://api.notion.com/v1",
            timeout=40.0,
            trust_env=True,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Notion-Version": self.settings.notion_version,
                "Content-Type": "application/json",
            },
        )

    def _request(self, method: str, path: str, *, json_body: Any | None = None, params: dict[str, Any] | None = None) -> Any:
        with self._client() as client:
            response = client.request(method, path, json=json_body, params=params)
            if response.status_code >= 400:
                detail = response.text[:1000]
                raise ToolExecutionError(f"Notion API {response.status_code}: {detail}", category="provider")
            if not response.text:
                return {}
            return response.json()

    def _rich_text(self, text: str) -> list[dict[str, Any]]:
        return [{"type": "text", "text": {"content": text}}]

    def _normalize_page(self, page: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": page.get("id"),
            "url": page.get("url"),
            "created_time": page.get("created_time"),
            "last_edited_time": page.get("last_edited_time"),
            "archived": page.get("archived"),
            "properties": page.get("properties", {}),
            "parent": page.get("parent", {}),
        }

    def _normalize_block(self, block: dict[str, Any]) -> dict[str, Any]:
        block_type = block.get("type")
        return {
            "id": block.get("id"),
            "type": block_type,
            "has_children": block.get("has_children", False),
            "archived": block.get("archived", False),
            "data": block.get(block_type, {}) if block_type else {},
        }

    def search(self, *, query: str = "", filter_payload: dict[str, Any] | None = None, page_size: int = 10, debug: bool = False) -> dict[str, Any]:
        body: dict[str, Any] = {"page_size": page_size}
        if query:
            body["query"] = query
        if filter_payload:
            body["filter"] = filter_payload
        data = self._request("POST", "/search", json_body=body)
        results = []
        for item in data.get("results", []):
            if item.get("object") == "page":
                results.append({"object": "page", **self._normalize_page(item)})
            elif item.get("object") in {"database", "data_source"}:
                results.append(
                    {
                        "object": item.get("object"),
                        "id": item.get("id"),
                        "url": item.get("url"),
                        "title": item.get("title", []),
                    }
                )
        return result_payload(data={"results": results, "has_more": data.get("has_more", False)}, debug=data if debug else None)

    def page_get(self, *, page_id: str, include_blocks: bool = False, debug: bool = False) -> dict[str, Any]:
        page = self._request("GET", f"/pages/{page_id}")
        payload = {"page": self._normalize_page(page)}
        if include_blocks:
            payload["blocks"] = self.blocks_list(block_id=page_id, recursive=True, page_size=100)["data"]
        return result_payload(data=payload, debug=page if debug else None)

    def page_create(
        self,
        *,
        parent: dict[str, Any],
        title: str | None = None,
        properties: dict[str, Any] | None = None,
        children: list[dict[str, Any]] | None = None,
        idempotency_key: str | None = None,
        dry_run: bool = False,
        debug: bool = False,
    ) -> dict[str, Any]:
        if idempotency_key:
            cached = self.store.get("notion.page.create", idempotency_key)
            if cached is not None:
                return cached
        payload: dict[str, Any] = {"parent": parent, "properties": properties or {}}
        if title:
            payload["properties"].setdefault("title", {"title": self._rich_text(title)})
        if children:
            payload["children"] = children
        if dry_run:
            result = result_payload(data={"dry_run": True, "payload": payload})
        else:
            page = self._request("POST", "/pages", json_body=payload)
            result = result_payload(data=self._normalize_page(page), debug=page if debug else None)
        if idempotency_key:
            self.store.put("notion.page.create", idempotency_key, result)
        return result

    def page_update_properties(
        self,
        *,
        page_id: str,
        properties: dict[str, Any],
        idempotency_key: str | None = None,
        dry_run: bool = False,
        debug: bool = False,
    ) -> dict[str, Any]:
        if idempotency_key:
            cached = self.store.get("notion.page.update_properties", idempotency_key)
            if cached is not None:
                return cached
        if dry_run:
            result = result_payload(data={"dry_run": True, "page_id": page_id, "properties": properties})
        else:
            page = self._request("PATCH", f"/pages/{page_id}", json_body={"properties": properties})
            result = result_payload(data=self._normalize_page(page), debug=page if debug else None)
        if idempotency_key:
            self.store.put("notion.page.update_properties", idempotency_key, result)
        return result

    def blocks_list(self, *, block_id: str, recursive: bool = False, page_size: int = 100, debug: bool = False) -> dict[str, Any]:
        output: list[dict[str, Any]] = []
        raw_pages: list[dict[str, Any]] = []

        def fetch_children(target_id: str) -> None:
            cursor = None
            while True:
                params = {"page_size": min(page_size, 100)}
                if cursor:
                    params["start_cursor"] = cursor
                data = self._request("GET", f"/blocks/{target_id}/children", params=params)
                raw_pages.append(data)
                for block in data.get("results", []):
                    normalized = self._normalize_block(block)
                    output.append(normalized)
                    if recursive and block.get("has_children"):
                        fetch_children(block["id"])
                if not data.get("has_more"):
                    break
                cursor = data.get("next_cursor")

        fetch_children(block_id)
        return result_payload(data={"results": output}, debug=raw_pages if debug else None)

    def blocks_append(
        self,
        *,
        block_id: str,
        children: list[dict[str, Any]],
        after: str | None = None,
        idempotency_key: str | None = None,
        dry_run: bool = False,
        debug: bool = False,
    ) -> dict[str, Any]:
        if idempotency_key:
            cached = self.store.get("notion.blocks.append", idempotency_key)
            if cached is not None:
                return cached
        body: dict[str, Any] = {"children": children}
        if after:
            body["after"] = after
        if dry_run:
            result = result_payload(data={"dry_run": True, "block_id": block_id, "payload": body})
        else:
            response = self._request("PATCH", f"/blocks/{block_id}/children", json_body=body)
            result = result_payload(data={"results": [self._normalize_block(item) for item in response.get("results", [])]}, debug=response if debug else None)
        if idempotency_key:
            self.store.put("notion.blocks.append", idempotency_key, result)
        return result

    def blocks_replace_range(
        self,
        *,
        parent_block_id: str,
        block_ids: list[str],
        children: list[dict[str, Any]],
        idempotency_key: str | None = None,
        dry_run: bool = False,
        debug: bool = False,
    ) -> dict[str, Any]:
        if idempotency_key:
            cached = self.store.get("notion.blocks.replace_range", idempotency_key)
            if cached is not None:
                return cached
        if dry_run:
            result = result_payload(data={"dry_run": True, "parent_block_id": parent_block_id, "block_ids": block_ids, "children": children})
        else:
            archived = []
            for block_id in block_ids:
                archived.append(self._request("PATCH", f"/blocks/{block_id}", json_body={"archived": True}))
            appended = self._request("PATCH", f"/blocks/{parent_block_id}/children", json_body={"children": children})
            result = result_payload(
                data={
                    "archived_block_ids": block_ids,
                    "inserted": [self._normalize_block(item) for item in appended.get("results", [])],
                },
                debug={"archived": archived, "appended": appended} if debug else None,
            )
        if idempotency_key:
            self.store.put("notion.blocks.replace_range", idempotency_key, result)
        return result

    def blocks_patch_text(
        self,
        *,
        block_id: str,
        text: str,
        idempotency_key: str | None = None,
        dry_run: bool = False,
        debug: bool = False,
    ) -> dict[str, Any]:
        if idempotency_key:
            cached = self.store.get("notion.blocks.patch_text", idempotency_key)
            if cached is not None:
                return cached
        block = self._request("GET", f"/blocks/{block_id}")
        block_type = block.get("type")
        if not block_type or block_type not in block:
            raise ToolExecutionError(f"Unsupported Notion block type for patch_text: {block_type}", category="validation")
        block_body = dict(block[block_type])
        if "rich_text" not in block_body:
            raise ToolExecutionError(f"Block type {block_type} does not expose rich_text", category="validation")
        block_body["rich_text"] = self._rich_text(text)
        if dry_run:
            result = result_payload(data={"dry_run": True, "block_id": block_id, "type": block_type, "text": text})
        else:
            updated = self._request("PATCH", f"/blocks/{block_id}", json_body={block_type: block_body})
            result = result_payload(data=self._normalize_block(updated), debug=updated if debug else None)
        if idempotency_key:
            self.store.put("notion.blocks.patch_text", idempotency_key, result)
        return result

    def database_query(self, *, database_id: str, query: dict[str, Any] | None = None, debug: bool = False) -> dict[str, Any]:
        data = self._request("POST", f"/data_sources/{database_id}/query", json_body=query or {})
        results = [self._normalize_page(item) for item in data.get("results", [])]
        return result_payload(data={"results": results, "has_more": data.get("has_more", False)}, debug=data if debug else None)

    def database_upsert_page(
        self,
        *,
        database_id: str,
        match_property: str,
        match_value: str,
        properties: dict[str, Any],
        children: list[dict[str, Any]] | None = None,
        idempotency_key: str | None = None,
        dry_run: bool = False,
        debug: bool = False,
    ) -> dict[str, Any]:
        if idempotency_key:
            cached = self.store.get("notion.database.upsert_page", idempotency_key)
            if cached is not None:
                return cached
        query_body = {
            "filter": {
                "property": match_property,
                "rich_text": {"equals": match_value},
            },
            "page_size": 1,
        }
        queried = self._request("POST", f"/data_sources/{database_id}/query", json_body=query_body)
        existing = queried.get("results", [])
        if existing:
            page_id = existing[0]["id"]
            result = self.page_update_properties(
                page_id=page_id,
                properties=properties,
                idempotency_key=None,
                dry_run=dry_run,
                debug=debug,
            )
        else:
            result = self.page_create(
                parent={"database_id": database_id},
                properties=properties,
                children=children,
                idempotency_key=None,
                dry_run=dry_run,
                debug=debug,
            )
        if idempotency_key:
            self.store.put("notion.database.upsert_page", idempotency_key, result)
        return result
