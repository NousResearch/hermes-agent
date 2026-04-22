from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import requests

from market_monitor.models import FetchResult


class HttpFetcher:
    def __init__(self, timeout: int = 20, user_agent: str = "Mozilla/5.0 HermesMarketMonitor/0.1"):
        self.timeout = timeout
        self.user_agent = user_agent

    def fetch(self, *, source_id: str, dataset_id: str | None, url: str) -> FetchResult:
        response = requests.get(url, timeout=self.timeout, headers={"User-Agent": self.user_agent})
        response.raise_for_status()
        fetch_time = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        return FetchResult(
            source_id=source_id,
            dataset_id=dataset_id,
            fetch_time=fetch_time,
            source_url=url,
            content_type=response.headers.get("content-type", "text/html").split(";")[0],
            content_bytes=response.content,
            text=response.text,
            local_path="",
            status_code=response.status_code,
            headers=dict(response.headers),
        )


def content_hash(content: bytes) -> str:
    return sha256(content).hexdigest()


def store_raw_fetch_result(fetch_result: FetchResult, raw_root: Path) -> FetchResult:
    digest = content_hash(fetch_result.content_bytes)
    suffix = _guess_suffix(fetch_result.content_type)
    target_dir = raw_root / fetch_result.source_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{digest}{suffix}"
    if not target_path.exists():
        target_path.write_bytes(fetch_result.content_bytes)
    return replace(fetch_result, local_path=str(target_path))


def _guess_suffix(content_type: str) -> str:
    if "json" in content_type:
        return ".json"
    if "html" in content_type:
        return ".html"
    if "pdf" in content_type:
        return ".pdf"
    return ".bin"
