"""HTTP client for the bundled Rasputin memory provider."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping
from urllib import request

logger = logging.getLogger(__name__)

_MAX_COMMIT_TEXT_CHARS = 12_000


@dataclass(frozen=True)
class RasputinClientConfig:
    base_url: str
    timeout_seconds: float = 8.0
    commit_timeout_seconds: float = 20.0
    fail_open: bool = True


class RasputinClient:
    def __init__(self, config: RasputinClientConfig):
        self.config = config
        self.base_url = (config.base_url or "").rstrip("/")

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _post_json(self, path: str, payload: Mapping[str, Any], *, timeout: float) -> Dict[str, Any] | None:
        body = json.dumps(dict(payload)).encode("utf-8")
        req = request.Request(
            self._url(path),
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with request.urlopen(req, timeout=timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except Exception:
            if self.config.fail_open:
                logger.debug("Rasputin POST %s failed", path, exc_info=True)
                return None
            raise

        if not raw.strip():
            return {}
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {"results": data}
        except Exception:
            if self.config.fail_open:
                logger.debug("Rasputin POST %s returned non-JSON payload", path, exc_info=True)
                return None
            raise

    def healthcheck(self) -> bool:
        req = request.Request(self._url("/health"))
        try:
            with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                response.read()
            return True
        except Exception:
            if self.config.fail_open:
                logger.debug("Rasputin healthcheck failed", exc_info=True)
                return False
            raise

    def search(self, query: str, *, limit: int = 8) -> List[Dict[str, Any]]:
        clean_query = (query or "").strip()
        if not clean_query:
            return []
        payload = {
            "query": clean_query,
            "limit": max(int(limit or 1), 1),
        }
        data = self._post_json("/search", payload, timeout=self.config.timeout_seconds)
        if not isinstance(data, Mapping):
            return []
        results = data.get("results") or data.get("matches") or data.get("items") or []
        return [dict(item) for item in results if isinstance(item, Mapping)]

    @staticmethod
    def _prepare_commit_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
        prepared = dict(payload)
        metadata = prepared.get("metadata")
        prepared["metadata"] = dict(metadata) if isinstance(metadata, Mapping) else {}

        text = str(prepared.get("text") or "")
        if len(text) > _MAX_COMMIT_TEXT_CHARS:
            prepared["text"] = text[:_MAX_COMMIT_TEXT_CHARS]
            prepared["metadata"]["rasputin_truncated"] = True
            prepared["metadata"]["rasputin_original_text_length"] = len(text)
        else:
            prepared["text"] = text
        return prepared

    def commit(self, payload: Mapping[str, Any]) -> bool:
        prepared = self._prepare_commit_payload(payload)
        data = self._post_json("/commit", prepared, timeout=self.config.commit_timeout_seconds)
        if data is None:
            return False
        ok = data.get("ok")
        return True if ok is None else bool(ok)
