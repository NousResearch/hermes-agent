#!/usr/bin/env python3
"""Small Mealie API CLI for Hermes skills."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_ENV_FILES = [
    Path("mini-mealie/.env.local"),
    Path(".env.local"),
    Path.home() / ".config" / "mealie.env",
]


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def resolve_config(args: argparse.Namespace) -> tuple[str, str]:
    env_values: dict[str, str] = {}
    for env_file in DEFAULT_ENV_FILES:
        env_values.update(load_env_file(env_file))
    if args.env_file:
        env_values.update(load_env_file(Path(args.env_file).expanduser()))

    base_url = (
        args.base_url
        or os.environ.get("MEALIE_BASE_URL")
        or os.environ.get("WXT_MEALIE_SERVER")
        or env_values.get("MEALIE_BASE_URL")
        or env_values.get("WXT_MEALIE_SERVER")
    )
    token = (
        args.token
        or os.environ.get("MEALIE_API_TOKEN")
        or os.environ.get("WXT_MEALIE_API_TOKEN")
        or env_values.get("MEALIE_API_TOKEN")
        or env_values.get("WXT_MEALIE_API_TOKEN")
    )

    if not base_url:
        raise SystemExit("Missing Mealie base URL. Set MEALIE_BASE_URL or WXT_MEALIE_SERVER.")
    if not token:
        raise SystemExit("Missing Mealie API token. Set MEALIE_API_TOKEN or WXT_MEALIE_API_TOKEN.")
    return base_url.strip().rstrip("/"), token.strip()


def request_json(
    base_url: str,
    token: str,
    method: str,
    path: str,
    body: Any | None = None,
    timeout: int = 30,
) -> Any:
    url = urllib.parse.urljoin(base_url + "/", path.lstrip("/"))
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method=method.upper(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            raw = res.read().decode("utf-8")
            if not raw:
                return None
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code} {exc.reason} for {method.upper()} {url}\n{details}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Request failed for {method.upper()} {url}: {exc.reason}") from exc


def print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True))


def cmd_whoami(args: argparse.Namespace) -> None:
    base_url, token = resolve_config(args)
    data = request_json(base_url, token, "GET", "/api/users/self")
    print_json({"baseUrl": base_url, "user": data})


def cmd_search(args: argparse.Namespace) -> None:
    base_url, token = resolve_config(args)
    query = urllib.parse.urlencode({"search": args.query, "perPage": args.limit})
    data = request_json(base_url, token, "GET", f"/api/recipes?{query}")
    print_json(summarize_recipes(data))


def cmd_recent(args: argparse.Namespace) -> None:
    base_url, token = resolve_config(args)
    query = urllib.parse.urlencode(
        {"perPage": args.limit, "orderBy": "dateUpdated", "orderDirection": "desc"}
    )
    data = request_json(base_url, token, "GET", f"/api/recipes?{query}")
    print_json(summarize_recipes(data))


def cmd_get_recipe(args: argparse.Namespace) -> None:
    base_url, token = resolve_config(args)
    data = request_json(base_url, token, "GET", f"/api/recipes/{urllib.parse.quote(args.slug)}")
    print_json(data)


def cmd_import_url(args: argparse.Namespace) -> None:
    base_url, token = resolve_config(args)
    body = {
        "url": args.url,
        "includeTags": args.include_tags,
        "includeCategories": args.include_categories,
    }
    data = request_json(base_url, token, "POST", "/api/recipes/create/url", body, timeout=60)
    slug = data if isinstance(data, str) else data.get("slug") if isinstance(data, dict) else data
    result = {"slug": slug}
    if isinstance(slug, str) and slug:
        result["url"] = f"{base_url}/g/home/r/{slug}"
    print_json(result)


def cmd_test_scrape(args: argparse.Namespace) -> None:
    base_url, token = resolve_config(args)
    data = request_json(
        base_url,
        token,
        "POST",
        "/api/recipes/test-scrape-url",
        {"url": args.url},
        timeout=args.timeout,
    )
    print_json(data)


def cmd_raw(args: argparse.Namespace) -> None:
    base_url, token = resolve_config(args)
    body = json.loads(args.json) if args.json else None
    data = request_json(base_url, token, args.method, args.path, body, timeout=args.timeout)
    print_json(data)


def summarize_recipes(data: Any) -> dict[str, Any]:
    items = data.get("items", []) if isinstance(data, dict) else []
    recipes = []
    for item in items:
        if not isinstance(item, dict):
            continue
        recipes.append(
            {
                "name": item.get("name"),
                "slug": item.get("slug"),
                "id": item.get("id"),
                "orgURL": item.get("orgURL"),
            }
        )
    return {
        "count": len(recipes),
        "total": data.get("total") if isinstance(data, dict) else None,
        "items": recipes,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Call common Mealie API endpoints.")
    parser.add_argument("--base-url", help="Mealie base URL. Defaults to MEALIE_BASE_URL.")
    parser.add_argument("--token", help="Mealie API token. Prefer env vars over this flag.")
    parser.add_argument("--env-file", help="Optional env file with Mealie settings.")
    sub = parser.add_subparsers(required=True)

    whoami = sub.add_parser("whoami", help="Verify token and show current user.")
    whoami.set_defaults(func=cmd_whoami)

    search = sub.add_parser("search", help="Search recipes by name.")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=10)
    search.set_defaults(func=cmd_search)

    recent = sub.add_parser("recent", help="List recently updated recipes.")
    recent.add_argument("--limit", type=int, default=20)
    recent.set_defaults(func=cmd_recent)

    get_recipe = sub.add_parser("get-recipe", help="Fetch one recipe by slug.")
    get_recipe.add_argument("slug")
    get_recipe.set_defaults(func=cmd_get_recipe)

    import_url = sub.add_parser("import-url", help="Import a recipe from a URL.")
    import_url.add_argument("url")
    import_url.add_argument("--include-tags", action="store_true")
    import_url.add_argument("--include-categories", action="store_true")
    import_url.set_defaults(func=cmd_import_url)

    test_scrape = sub.add_parser("test-scrape", help="Test whether Mealie can scrape a URL.")
    test_scrape.add_argument("url")
    test_scrape.add_argument("--timeout", type=int, default=30)
    test_scrape.set_defaults(func=cmd_test_scrape)

    raw = sub.add_parser("raw", help="Call an arbitrary Mealie API path.")
    raw.add_argument("method", choices=["GET", "POST", "PATCH", "PUT", "DELETE"])
    raw.add_argument("path")
    raw.add_argument("--json", help="JSON request body for mutating calls.")
    raw.add_argument("--timeout", type=int, default=30)
    raw.set_defaults(func=cmd_raw)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
