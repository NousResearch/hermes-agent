#!/usr/bin/env python3
"""Snapshot official Notion API/developer surfaces for drift monitoring.

No credentials are used. The script fetches public docs/spec/package metadata,
prints JSON by default, and can compare against a prior JSON snapshot.

Examples:
  python skills/productivity/notion/scripts/notion_api_surface_snapshot.py
  python skills/productivity/notion/scripts/notion_api_surface_snapshot.py --baseline /tmp/notion-snapshot.json --markdown
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

USER_AGENT = "Hermes-Notion-API-Surface-Snapshot/1.0 (+https://hermes-agent.nousresearch.com)"
URLS = {
    "openapi": "https://developers.notion.com/openapi.json",
    "llms": "https://developers.notion.com/llms.txt",
    "changelog": "https://developers.notion.com/page/changelog.md",
    "changes_by_version": "https://developers.notion.com/reference/changes-by-version.md",
    "sdk_readme": "https://raw.githubusercontent.com/makenotion/notion-sdk-js/main/README.md",
    "mcp_manifest": "https://www.notion.com/.well-known/mcp.json",
    "mcp_oauth_metadata": "https://mcp.notion.com/.well-known/oauth-authorization-server",
}
NPM_PACKAGES = [
    "@notionhq/client",
    "ntn",
    "@notionhq/workers",
    "@notionhq/notion-mcp-server",
]


def fetch(url: str) -> tuple[bytes, dict[str, str]]:
    req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json,text/plain,text/markdown,*/*"})
    with urlopen(req, timeout=45) as response:
        data = response.read()
        headers = {k.lower(): v for k, v in response.headers.items()}
    return data, headers


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def summarize_openapi(data: bytes) -> dict:
    spec = json.loads(data.decode("utf-8"))
    paths = spec.get("paths", {})
    operations = []
    stale_operation_ids = []
    paginated = []
    for path, path_item in sorted(paths.items()):
        for method, op in sorted(path_item.items()):
            if method.lower() not in {"get", "post", "patch", "delete", "put"} or not isinstance(op, dict):
                continue
            params = path_item.get("parameters", []) + op.get("parameters", [])
            param_names = [p.get("name") for p in params if isinstance(p, dict)]
            opid = op.get("operationId", "")
            item = {"method": method.upper(), "path": path, "operationId": opid, "tags": op.get("tags", [])}
            operations.append(item)
            if "data_sources" in path and "database" in opid.lower():
                stale_operation_ids.append(item)
            body = json.dumps(op.get("requestBody", {}), sort_keys=True)
            if any(name in {"start_cursor", "page_size"} for name in param_names) or "start_cursor" in body or "page_size" in body:
                paginated.append(item)
    webhook_events = sorted((spec.get("webhooks") or {}).keys())
    notion_version_parameter = spec.get("components", {}).get("parameters", {}).get("notionVersion", {})
    version_enums = notion_version_parameter.get("schema", {}).get("enum", [])
    for path_item in paths.values():
        for p in path_item.get("parameters", []) if isinstance(path_item, dict) else []:
            if p.get("name") == "Notion-Version":
                version_enums = p.get("schema", {}).get("enum", version_enums)
    return {
        "openapi": spec.get("openapi"),
        "info": spec.get("info", {}),
        "servers": spec.get("servers", []),
        "path_count": len(paths),
        "operation_count": len(operations),
        "schema_count": len(spec.get("components", {}).get("schemas", {})),
        "notion_version_enum": version_enums,
        "operations": operations,
        "stale_operation_ids": stale_operation_ids,
        "paginated_operations": paginated,
        "webhook_events": webhook_events,
    }


def snapshot() -> dict:
    out = {"retrieved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "sources": {}, "npm": {}}
    for name, url in URLS.items():
        try:
            data, headers = fetch(url)
            source = {
                "url": url,
                "sha256": sha256(data),
                "bytes": len(data),
                "etag": headers.get("etag"),
                "last_modified": headers.get("last-modified"),
                "content_type": headers.get("content-type"),
            }
            if name == "openapi":
                source["summary"] = summarize_openapi(data)
            out["sources"][name] = source
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            out["sources"][name] = {"url": url, "error": repr(exc)}
    for package in NPM_PACKAGES:
        url = "https://registry.npmjs.org/" + package.replace("/", "%2f")
        try:
            data, headers = fetch(url)
            meta = json.loads(data.decode("utf-8"))
            latest = meta.get("dist-tags", {}).get("latest")
            latest_meta = meta.get("versions", {}).get(latest, {}) if latest else {}
            out["npm"][package] = {
                "url": url,
                "latest": latest,
                "dist_tags": meta.get("dist-tags", {}),
                "engines": latest_meta.get("engines"),
                "repository": latest_meta.get("repository"),
                "sha256": sha256(data),
                "bytes": len(data),
            }
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            out["npm"][package] = {"url": url, "error": repr(exc)}
    return out


def compare(before: dict, after: dict) -> list[str]:
    alerts: list[str] = []
    for name, src in after.get("sources", {}).items():
        old = before.get("sources", {}).get(name, {})
        if src.get("sha256") and old.get("sha256") and src.get("sha256") != old.get("sha256"):
            alerts.append(f"source {name} hash changed: {old.get('sha256')} -> {src.get('sha256')}")
    old_api = before.get("sources", {}).get("openapi", {}).get("summary", {})
    new_api = after.get("sources", {}).get("openapi", {}).get("summary", {})
    for field in ["path_count", "operation_count", "schema_count", "notion_version_enum", "webhook_events"]:
        if old_api.get(field) != new_api.get(field):
            alerts.append(f"openapi {field} changed: {old_api.get(field)} -> {new_api.get(field)}")
    old_ops = {(o.get("method"), o.get("path"), o.get("operationId")) for o in old_api.get("operations", [])}
    new_ops = {(o.get("method"), o.get("path"), o.get("operationId")) for o in new_api.get("operations", [])}
    for item in sorted(new_ops - old_ops):
        alerts.append(f"openapi operation added: {item[0]} {item[1]} ({item[2]})")
    for item in sorted(old_ops - new_ops):
        alerts.append(f"openapi operation removed/renamed: {item[0]} {item[1]} ({item[2]})")
    for package, meta in after.get("npm", {}).items():
        old = before.get("npm", {}).get(package, {})
        if old.get("latest") and meta.get("latest") and old.get("latest") != meta.get("latest"):
            alerts.append(f"npm {package} latest changed: {old.get('latest')} -> {meta.get('latest')}")
    return alerts


def markdown_report(snap: dict, alerts: list[str] | None = None) -> str:
    lines = ["# Notion API Surface Snapshot", "", f"Retrieved: `{snap['retrieved_at']}`", ""]
    if alerts is not None:
        lines += ["## Alerts", ""]
        lines += [f"- {alert}" for alert in alerts] or ["- No monitored high-level changes."]
        lines.append("")
    api = snap.get("sources", {}).get("openapi", {}).get("summary", {})
    if api:
        lines += ["## OpenAPI", ""]
        for field in ["openapi", "path_count", "operation_count", "schema_count", "notion_version_enum"]:
            lines.append(f"- {field}: `{api.get(field)}`")
        lines.append("")
        if api.get("stale_operation_ids"):
            lines += ["Stale data-source operation IDs:", ""]
            for op in api["stale_operation_ids"]:
                lines.append(f"- `{op['method']} {op['path']}` — `{op['operationId']}`")
            lines.append("")
    lines += ["## Package versions", ""]
    for package, meta in sorted(snap.get("npm", {}).items()):
        lines.append(f"- `{package}`: `{meta.get('latest', 'unknown')}`")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", help="prior JSON snapshot to compare against")
    parser.add_argument("--markdown", action="store_true", help="print concise markdown instead of JSON")
    args = parser.parse_args()
    current = snapshot()
    alerts = None
    if args.baseline:
        with open(args.baseline, "r", encoding="utf-8") as fh:
            baseline = json.load(fh)
        alerts = compare(baseline, current)
    if args.markdown:
        sys.stdout.write(markdown_report(current, alerts))
    else:
        if alerts is not None:
            current["alerts"] = alerts
        json.dump(current, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
