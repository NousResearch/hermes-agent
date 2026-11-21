#!/usr/bin/env python3
"""Search using Querit AI API.

Usage:
    python search_querit.py "your search query"
    python search_querit.py "your search query" --count 20
    python search_querit.py "your search query" --json

Note:
    The API key will be prompted interactively on first use (input hidden).
    The key is stored only in memory and will be lost when the process exits.
"""
import getpass
import json
import sys
import urllib.request
import urllib.error

API_URL = "https://api.querit.ai/v1/search"

# 内存级 API key 缓存（进程内有效，退出即失效）
_API_KEY_CACHE = None


def _get_api_key():
    """获取 Querit API key。

    仅支持交互式输入（隐藏输入），仅存储于内存，退出即失效。

    Returns:
        str: API key，如果用户取消输入或非TTY环境则返回 None
    """
    global _API_KEY_CACHE

    # 1. 检查内存缓存
    if _API_KEY_CACHE is not None:
        return _API_KEY_CACHE

    # 2. 检测是否在交互式终端
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print(
            "Error: This script requires an interactive terminal (TTY) for API key input.",
            file=sys.stderr
        )
        return None

    # 3. 交互式询问（首次使用）
    print("Please enter your Querit API key (input will be hidden):", file=sys.stderr)

    try:
        api_key = getpass.getpass("Querit API Key: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.", file=sys.stderr)
        return None

    if not api_key:
        print("Error: API key is required.", file=sys.stderr)
        return None

    # 仅存储在内存，不写入文件
    _API_KEY_CACHE = api_key
    print("API key accepted for this session only.\n", file=sys.stderr)
    return api_key


def search(query: str, count: int = 10, raw_json: bool = False):
    """Search using Querit AI API."""
    api_key = _get_api_key()
    if not api_key:
        sys.exit(1)

    data = json.dumps({
        "query": query,
        "count": count
    }).encode('utf-8')

    req = urllib.request.Request(
        API_URL,
        data=data,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept-Encoding": "gzip, deflate",
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            result = response.read()
            if response.headers.get('Content-Encoding') == 'gzip':
                import gzip
                result = gzip.decompress(result)
            data = json.loads(result.decode('utf-8'))
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.reason}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if raw_json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print_results(data)


def print_results(data: dict):
    """Print search results in a readable format."""
    if not isinstance(data, dict):
        print("Unexpected response format:", data)
        return

    error_code = data.get("error_code", 200)
    if error_code != 200:
        print(f"Error {error_code}: {data.get('error_msg', 'Unknown error')}")
        return

    results_data = data.get("results", {})
    results = results_data.get("result", []) if isinstance(results_data, dict) else []

    query_context = data.get("query_context", {})
    query = query_context.get("query", "Unknown query") if isinstance(query_context, dict) else "Unknown query"

    took = data.get("took", "unknown")
    search_id = data.get("search_id")

    if not results:
        print(f"No results found for: {query}")
        return

    print(f"Results for: {query}")
    if took:
        print(f"Time: {took}")
    print(f"Found {len(results)} result(s)\n")

    for i, result in enumerate(results, 1):
        if isinstance(result, dict):
            title = result.get("title", "No title")
            url = result.get("url", "")
            snippet = result.get("snippet", "")
            site_name = result.get("site_name", "")
            page_age = result.get("page_age", "")
            site_icon = result.get("site_icon", "")
        else:
            title = str(result)
            url = ""
            snippet = ""
            site_name = ""
            page_age = ""
            site_icon = ""

        print(f"{i}. {title}")
        if site_name:
            print(f"   Site: {site_name}")
        if url:
            print(f"   URL: {url}")
        if page_age:
            print(f"   Age: {page_age}")
        if snippet:
            if len(snippet) > 300:
                snippet = snippet[:300] + "..."
            print(f"   {snippet}")
        if site_icon:
            print(f"   Icon: {site_icon}")
        print()

    if search_id:
        print(f"Search ID: {search_id}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    query_parts = []
    count = 10
    raw_json = False

    i = 0
    while i < len(args):
        if args[i] == "--count" and i + 1 < len(args):
            count = int(args[i + 1])
            i += 2
        elif args[i] == "--json":
            raw_json = True
            i += 1
        else:
            query_parts.append(args[i])
            i += 1

    if not query_parts:
        print("Error: No search query provided.", file=sys.stderr)
        sys.exit(1)

    query = " ".join(query_parts)
    search(query=query, count=count, raw_json=raw_json)
