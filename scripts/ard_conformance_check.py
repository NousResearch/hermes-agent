#!/usr/bin/env python3
"""ARD conformance and leak checker.

Validates Hermes/ARD ai-catalog.json manifests and captured SearchRequest /
SearchResponse payloads against the operational contracts Hermes relies on:
url xor data, public/private visibility boundaries, current root-level
federation/referrals shape, and secret/local-path leak prevention.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

CATALOG_TYPES = {
    "application/ai-skill",
    "application/mcp-server-card+json",
    "application/mcp-server+json",  # accepted legacy input only
    "application/a2a-agent-card+json",
    "application/vnd.huggingface.space+json",
    "application/vnd.hermes.tool-candidate+json",
}
SECRET_FIELD_RE = re.compile(
    r"(?i)^(api[_-]?key|access[_-]?token|auth[_-]?token|bearer|client[_-]?secret|password|private[_-]?key|refresh[_-]?token|session[_-]?token|.*[_-](api[_-]?key|token|secret|password))$"
)
TOKEN_VALUE_RE = re.compile(
    r"(?i)(sk-[A-Za-z0-9_-]{10,}|tk_[A-Za-z0-9_-]{10,}|Bearer\\s+[A-Za-z0-9._-]{10,})"
)
LOCAL_PATH_RE = re.compile(r"(/home/[A-Za-z0-9_.-]+/|/mnt/[a-z]/|C:\\\\Users\\\\|C:/Users/)")


def _issue(code: str, message: str, *, path: str = "") -> dict[str, str]:
    item = {"code": code, "message": message}
    if path:
        item["path"] = path
    return item


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _contains_sensitive_marker(value: Any) -> bool:
    def walk(obj: Any) -> bool:
        if isinstance(obj, dict):
            for key, inner in obj.items():
                if isinstance(key, str) and SECRET_FIELD_RE.match(key):
                    return True
                if walk(inner):
                    return True
            return False
        if isinstance(obj, list):
            return any(walk(item) for item in obj)
        if isinstance(obj, str):
            return bool(TOKEN_VALUE_RE.search(obj))
        return False

    return walk(value)


def _contains_local_path(value: Any) -> bool:
    return bool(LOCAL_PATH_RE.search(_json_text(value)))


def _validate_catalog_shape(catalog: dict[str, Any]) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    if not isinstance(catalog.get("specVersion"), str):
        errors.append(_issue("catalog_missing_spec_version", "Catalog must include string specVersion", path="specVersion"))
    if not isinstance(catalog.get("host"), dict):
        errors.append(_issue("catalog_missing_host", "Catalog must include host object", path="host"))
    entries = catalog.get("entries")
    if not isinstance(entries, list):
        errors.append(_issue("catalog_missing_entries", "Catalog entries must be a list", path="entries"))
        return errors
    for idx, entry in enumerate(entries):
        path = f"entries[{idx}]"
        if not isinstance(entry, dict):
            errors.append(_issue("entry_not_object", "Catalog entry must be an object", path=path))
            continue
        for key in ("identifier", "displayName", "type"):
            if not isinstance(entry.get(key), str) or not entry.get(key):
                errors.append(_issue(f"entry_missing_{key}", f"Entry missing non-empty {key}", path=f"{path}.{key}"))
        if isinstance(entry.get("type"), str) and entry["type"] not in CATALOG_TYPES:
            errors.append(_issue("entry_unknown_type", f"Unknown ARD type {entry['type']}", path=f"{path}.type"))
        has_url = "url" in entry
        has_data = "data" in entry
        if has_url == has_data:
            errors.append(_issue("entry_source_xor_violation", "Entry must expose exactly one of url or data", path=path))
        if has_url and entry.get("url") == "":
            errors.append(_issue("entry_empty_url", "Entry url must not be empty", path=f"{path}.url"))
    return errors


def validate_catalog(catalog: dict[str, Any], *, visibility: str = "public") -> dict[str, Any]:
    """Validate an ARD ai-catalog.json manifest.

    visibility="public" forbids local stdio URLs and local filesystem paths.
    visibility="private" allows stdio/local paths but still forbids secret-like
    material because private catalogs are often copied into reports/logs.
    """
    if visibility not in {"public", "private"}:
        raise ValueError("visibility must be public or private")

    errors = _validate_catalog_shape(catalog)
    raw_entries = catalog.get("entries")
    entries: list[Any] = raw_entries if isinstance(raw_entries, list) else []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        path = f"entries[{idx}]"
        url = entry.get("url")
        if visibility == "public":
            if isinstance(url, str) and url.startswith("stdio:"):
                errors.append(_issue("public_stdio_url", "Public catalog must not expose local stdio MCP URLs", path=f"{path}.url"))
            if _contains_local_path(entry):
                errors.append(_issue("public_local_path", "Public catalog must not expose local filesystem paths", path=path))
            if _contains_sensitive_marker(entry):
                errors.append(_issue("public_sensitive_marker", "Public catalog contains secret-like marker", path=path))
        else:
            if _contains_sensitive_marker(entry):
                errors.append(_issue("private_sensitive_marker", "Private catalog contains secret-like marker", path=path))

    return {
        "ok": not errors,
        "kind": "catalog",
        "visibility": visibility,
        "entry_count": len(entries),
        "errors": errors,
    }


def validate_search_exchange(request: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    """Validate current ARD SearchRequest/SearchResponse shape."""
    errors: list[dict[str, str]] = []
    if not isinstance(request.get("query"), dict):
        errors.append(_issue("request_missing_query", "SearchRequest must include query object", path="query"))
    if request.get("federation") != "referrals":
        errors.append(_issue("request_missing_root_federation", "SearchRequest should set root federation='referrals'", path="federation"))
    if "search" in request.get("query", {}):
        errors.append(_issue("request_legacy_nested_query", "Use query.text, not legacy query.search", path="query"))
    page_size = request.get("pageSize")
    if page_size is not None and (not isinstance(page_size, int) or page_size < 1):
        errors.append(_issue("request_invalid_page_size", "pageSize must be a positive integer", path="pageSize"))

    if not isinstance(response.get("results"), list):
        errors.append(_issue("response_missing_results", "SearchResponse must include results list", path="results"))
    if isinstance(response.get("federation"), dict) and "referrals" in response["federation"]:
        errors.append(_issue("response_legacy_nested_referrals", "SearchResponse referrals must be root-level", path="federation.referrals"))
    referrals = response.get("referrals", [])
    if referrals is not None and not isinstance(referrals, list):
        errors.append(_issue("response_invalid_referrals", "referrals must be a list when present", path="referrals"))

    return {"ok": not errors, "kind": "search_exchange", "errors": errors}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate ARD catalog/search conformance")
    parser.add_argument("--catalog", type=Path, help="Path to ai-catalog.json")
    parser.add_argument("--visibility", choices=("public", "private"), default="public")
    parser.add_argument("--search-request", type=Path, help="Captured SearchRequest JSON")
    parser.add_argument("--search-response", type=Path, help="Captured SearchResponse JSON")
    parser.add_argument("--json", action="store_true", help="Emit JSON report")
    args = parser.parse_args(argv)

    reports: list[dict[str, Any]] = []
    if args.catalog:
        reports.append(validate_catalog(load_json(args.catalog), visibility=args.visibility))
    if args.search_request or args.search_response:
        if not args.search_request or not args.search_response:
            parser.error("--search-request and --search-response must be provided together")
        reports.append(validate_search_exchange(load_json(args.search_request), load_json(args.search_response)))
    if not reports:
        parser.error("provide --catalog or --search-request/--search-response")

    ok = all(r["ok"] for r in reports)
    output = {"ok": ok, "reports": reports}
    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(f"ARD conformance: {'ok' if ok else 'failed'}")
        for report in reports:
            print(f"- {report['kind']}: {'ok' if report['ok'] else 'failed'}")
            for issue in report.get("errors", []):
                loc = f" [{issue.get('path')}]" if issue.get("path") else ""
                print(f"  {issue['code']}{loc}: {issue['message']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
