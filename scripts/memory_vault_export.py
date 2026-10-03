#!/usr/bin/env python3
"""Export mem0/Qdrant memories to human-readable Markdown vault files.

Reads all points from a Qdrant collection that belong to a given user_id
and writes each as a Markdown file to an output directory.

File naming: <qdrant_point_id>.md
Frontmatter: id, agent_id, score, created_at, updated_at
Body: memory text (the 'data' payload field)

Supports both HTTP Qdrant (--qdrant-url) and embedded/local Qdrant
(--qdrant-path).  When neither flag is given the script auto-detects the
mode from mem0.json: a ``path`` key → embedded; a ``url`` key → HTTP.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Qdrant client wrapper — prefers the official SDK when available so that
# embedded (local-path) Qdrant works correctly; falls back to raw HTTP for
# pure HTTP deployments when the SDK is not installed.
# ---------------------------------------------------------------------------

try:
    from qdrant_client import QdrantClient as _QdrantClient  # type: ignore

    _SDK_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SDK_AVAILABLE = False

import urllib.error
import urllib.request
from urllib.parse import quote

SCROLL_LIMIT = 200


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def get_hermes_home() -> Path:
    """Return HERMES_HOME (env var) or the platform-native default.

    Delegates to hermes_constants when available so the resolution is
    identical to every other caller in the codebase.
    """
    try:
        import sys as _sys

        # hermes_constants lives at the project root; add it to the path
        # when running the script directly from scripts/.
        _root = Path(__file__).parent.parent
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        from hermes_constants import get_hermes_home as _ghh

        return _ghh()
    except ImportError:
        # Fallback: honour HERMES_HOME env var, then platform default.
        env = os.environ.get("HERMES_HOME", "")
        if env:
            return Path(env)
        return Path.home() / ".hermes" / "hermes-agent"


def load_mem0_json() -> dict:
    """Load mem0.json from get_hermes_home()/mem0.json.  Returns {} on miss."""
    mem0_path = get_hermes_home() / "mem0.json"
    try:
        with mem0_path.open() as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, AttributeError):
        return {}


def _vs_config(mem0_cfg: dict) -> dict:
    """Extract the oss.vector_store.config block (or {})."""
    return (
        mem0_cfg.get("oss", {})
        .get("vector_store", {})
        .get("config", {})
    )


def load_collection_from_mem0_json() -> str | None:
    """Try to read collection_name from get_hermes_home()/mem0.json."""
    vs = _vs_config(load_mem0_json())
    return vs.get("collection_name") or None


def detect_qdrant_mode(mem0_cfg: dict) -> tuple[str | None, str | None]:
    """Return (path, url) from mem0.json oss.vector_store.config.

    Exactly one of the two will be non-None when the file is present and
    has a vector_store config; both are None when nothing is configured.
    """
    vs = _vs_config(mem0_cfg)
    path = vs.get("path") or None
    url = vs.get("url") or None
    return path, url


# ---------------------------------------------------------------------------
# Qdrant client abstraction
# ---------------------------------------------------------------------------


class _QdrantAdapter:
    """Thin adapter that unifies SDK and raw-HTTP scroll access."""

    def __init__(self, *, qdrant_path: str | None = None, qdrant_url: str | None = None, api_key: str | None = None) -> None:
        if qdrant_path and _SDK_AVAILABLE:
            self._mode = "sdk_path"
            self._client = _QdrantClient(path=qdrant_path)
        elif qdrant_url and _SDK_AVAILABLE:
            self._mode = "sdk_url"
            kw: dict = {"url": qdrant_url}
            if api_key:
                kw["api_key"] = api_key
            self._client = _QdrantClient(**kw)
        elif qdrant_path:
            raise RuntimeError(
                "Embedded Qdrant (--qdrant-path) requires the qdrant-client Python "
                "SDK.  Install it with:  pip install qdrant-client"
            )
        elif qdrant_url:
            self._mode = "http"
            self._url = qdrant_url.rstrip("/")
            self._api_key = api_key
            self._client = None
        else:
            raise ValueError("Either qdrant_path or qdrant_url must be provided.")

    # ---- health check -------------------------------------------------------

    def health_check(self) -> str:
        """Return the Qdrant server version string (raises on failure)."""
        if self._mode == "http":
            info = self._api_get("/")
            return info.get("version", "unknown")
        # SDK path / url
        info = self._client.get_collections()  # lightweight call
        # Version not directly available via SDK collections endpoint;
        # use a best-effort approach.
        return "(sdk)"

    # ---- scroll -------------------------------------------------------------

    def scroll_all_for_user(self, collection: str, user_id: str) -> list:
        """Return all points in *collection* whose user_id matches."""
        if self._mode == "http":
            return self._http_scroll_all(collection, user_id)
        return self._sdk_scroll_all(collection, user_id)

    # ---- SDK paths ----------------------------------------------------------

    def _sdk_scroll_all(self, collection: str, user_id: str) -> list:
        from qdrant_client.models import Filter, FieldCondition, MatchValue  # type: ignore

        points = []
        offset = None
        while True:
            result, next_offset = self._client.scroll(
                collection_name=collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
                ),
                limit=SCROLL_LIMIT,
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )
            for p in result:
                points.append({"id": str(p.id), "payload": p.payload or {}})
            if next_offset is None or not result:
                break
            offset = next_offset
        return points

    # ---- HTTP paths ---------------------------------------------------------

    def _api_post(self, path: str, body: dict, timeout: int = 120) -> dict:
        data = json.dumps(body).encode()
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["api-key"] = self._api_key
        req = urllib.request.Request(
            f"{self._url}{path}",
            data=data,
            method="POST",
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    def _api_get(self, path: str, timeout: int = 30) -> dict:
        headers: dict[str, str] = {}
        if self._api_key:
            headers["api-key"] = self._api_key
        req = urllib.request.Request(f"{self._url}{path}", method="GET", headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    def _http_scroll_all(self, collection: str, user_id: str) -> list:
        points = []
        offset = None
        while True:
            body: dict = {
                "limit": SCROLL_LIMIT,
                "with_vector": False,
                "with_payload": True,
                "filter": {
                    "must": [{"key": "user_id", "match": {"value": user_id}}]
                },
            }
            if offset is not None:
                body["offset"] = offset

            try:
                result = self._api_post(
                    f"/collections/{quote(collection, safe='')}/points/scroll", body
                )
            except urllib.error.URLError as exc:
                print(f"[ERROR] Qdrant scroll failed: {exc}", file=sys.stderr)
                sys.exit(1)

            batch = result.get("result", {}).get("points", [])
            points.extend(batch)

            next_offset = result.get("result", {}).get("next_page_offset")
            if next_offset is None or not batch:
                break
            offset = next_offset

        return points


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def extract_text(payload: dict) -> str:
    """Pull the memory text from a Qdrant payload (mem0 field ordering)."""
    return (
        payload.get("data")
        or payload.get("text")
        or payload.get("memory")
        or payload.get("content")
        or ""
    )


def _yaml_escape(val: object) -> str:
    """Escape double quotes in a value so it is safe inside YAML double-quoted strings."""
    s = str(val)
    if '"' in s:
        return s.replace('"', '\\"')
    return s


def render_markdown(point_id: str, payload: dict) -> str:
    """Render a single Qdrant point as a Markdown file with YAML frontmatter."""
    agent_id = _yaml_escape(payload.get("agent_id", ""))
    created_at = _yaml_escape(payload.get("created_at", ""))
    updated_at = _yaml_escape(payload.get("updated_at", ""))
    safe_id = _yaml_escape(point_id)
    # Score is not stored in the Qdrant payload — it only exists in search results.
    # We leave it blank; the sync script will not overwrite this field.
    score = payload.get("score", "")

    text = extract_text(payload)

    # Build frontmatter — quote string values to prevent YAML injection
    # (values may contain colons, brackets, or other YAML special characters).
    frontmatter_lines = [
        "---",
        f'id: "{safe_id}"',
        f'agent_id: "{agent_id}"',
        f"score: {score}",
        f'created_at: "{created_at}"',
        f'updated_at: "{updated_at}"',
        "---",
    ]
    return "\n".join(frontmatter_lines) + "\n" + text + "\n"


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_vault(adapter: "_QdrantAdapter", collection: str, user_id: str, vault_dir: Path) -> int:
    """Export memories to vault. Returns count of files written."""
    vault_dir.mkdir(parents=True, exist_ok=True)

    # Verify Qdrant is up
    try:
        version = adapter.health_check()
        print(f"Qdrant {version} is reachable.", file=sys.stderr)
    except Exception as exc:
        print(f"[ERROR] Cannot reach Qdrant: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Scrolling collection '{collection}' for user_id='{user_id}'...", file=sys.stderr)
    points = adapter.scroll_all_for_user(collection, user_id)
    print(f"Found {len(points)} memory point(s).", file=sys.stderr)

    written = 0
    skipped = 0

    for point in points:
        point_id = str(point["id"])
        payload = point.get("payload") or {}

        text = extract_text(payload)
        if not text:
            print(f"[SKIP] {point_id}: no text in payload", file=sys.stderr)
            skipped += 1
            continue

        md_content = render_markdown(point_id, payload)
        dest = (vault_dir / f"{point_id}.md").resolve()
        if not str(dest).startswith(str(vault_dir.resolve())):
            print(f"[SKIP] {point_id}: unsafe path (contains path separators)", file=sys.stderr)
            skipped += 1
            continue
        dest.write_text(md_content, encoding="utf-8")
        written += 1

    print(
        f"Vault export complete: {written} written, {skipped} skipped (no text).",
        file=sys.stderr,
    )
    print(f"Vault location: {vault_dir}", file=sys.stderr)
    return written


def parse_args() -> argparse.Namespace:
    mem0_cfg = load_mem0_json()
    default_collection = load_collection_from_mem0_json() or "mem0"
    auto_path, auto_url = detect_qdrant_mode(mem0_cfg)
    # Embedded path from mem0.json takes precedence over the QDRANT_URL env var so
    # that a stale env var cannot silently redirect an embedded deployment to HTTP.
    default_qdrant_path = os.environ.get("QDRANT_PATH", auto_path or "")
    if default_qdrant_path:
        # Embedded mode: ignore QDRANT_URL entirely.
        default_qdrant_url = None
    else:
        default_qdrant_url = os.environ.get("QDRANT_URL", auto_url or "http://localhost:6333")
    default_output_dir = get_hermes_home() / "memories" / "vault"

    parser = argparse.ArgumentParser(
        description="Export mem0/Qdrant memories to human-readable Markdown vault files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--user",
        required=True,
        help="user_id to filter memories by (e.g. 'clark').",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory to write Markdown files into (created if absent).",
    )

    qdrant_group = parser.add_mutually_exclusive_group()
    qdrant_group.add_argument(
        "--qdrant-url",
        default=default_qdrant_url if not default_qdrant_path else None,
        help=(
            "Base URL for the Qdrant HTTP API. "
            "Also reads QDRANT_URL env var. "
            "Mutually exclusive with --qdrant-path."
        ),
    )
    qdrant_group.add_argument(
        "--qdrant-path",
        default=default_qdrant_path or None,
        help=(
            "Local filesystem path to an embedded Qdrant database. "
            "Requires qdrant-client SDK. "
            "Also reads QDRANT_PATH env var. "
            "Mutually exclusive with --qdrant-url."
        ),
    )
    parser.add_argument(
        "--collection",
        default=default_collection,
        help=(
            "Qdrant collection name. Defaults to collection_name from mem0.json "
            "oss.vector_store.config, falling back to 'mem0'."
        ),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("QDRANT_API_KEY"),
        help="Qdrant API key (HTTP mode only). Also reads QDRANT_API_KEY env var.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    adapter = _QdrantAdapter(
        qdrant_path=args.qdrant_path,
        qdrant_url=args.qdrant_url,
        api_key=args.api_key,
    )
    written = export_vault(
        adapter=adapter,
        collection=args.collection,
        user_id=args.user,
        vault_dir=args.output_dir,
    )
    print(f"Exported {written} memories to {args.output_dir}")


if __name__ == "__main__":
    main()
