#!/usr/bin/env python3
"""TokenJuice-style output compression for Hermes Agent.
Compresses tool outputs BEFORE they enter LLM context.
"""

import sys
import re
import json
import argparse
from typing import List, Dict, Optional


def compress_terminal(text: str) -> str:
    """Compress terminal output: keep last 80 lines + errors."""
    lines = text.split("\n")
    if len(lines) <= 80:
        return text

    error_lines = [l for l in lines if re.search(r'(?i)(error|traceback|fail|cannot|denied|timeout)', l)]
    last_lines = lines[-80:]

    combined = error_lines + [f"\n--- [skipped {len(lines) - len(last_lines) - len(error_lines)} lines] ---\n"] + last_lines
    return "\n".join(combined)


def compress_web(text: str) -> str:
    """Compress web content: strip nav/footer/ads, keep main."""
    lines = text.split("\n")
    result = []
    skip = False
    for line in lines:
        stripped = line.strip()
        # Skip obvious boilerplate
        if re.match(r'(?i)^(cookie|advertisement|sponsored|subscribe|newsletter|privacy policy|terms of service)', stripped):
            skip = True
            continue
        if skip and stripped == "":
            skip = False
            continue
        if not skip:
            result.append(line)
    return "\n".join(result)


def compress_search(data) -> str:
    """Dedup + trim search results."""
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return data

    items = data if isinstance(data, list) else data.get("results", data.get("items", []))
    seen_urls = set()
    deduped = []
    for item in items:
        url = item.get("url", item.get("link", ""))
        if url in seen_urls:
            continue
        seen_urls.add(url)
        desc = item.get("description", item.get("snippet", ""))[:150]
        title = item.get("title", "")[:100]
        deduped.append({"title": title, "url": url, "description": desc})

    return json.dumps(deduped[:5], ensure_ascii=False, indent=2)


def compress_file(text: str) -> str:
    """Light file compression: skip blank lines + import blocks >10 lines."""
    lines = text.split("\n")
    result = []
    import_count = 0
    in_imports = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Skip consecutive blank lines
        if stripped == "" and (i > 0 and lines[i - 1].strip() == ""):
            continue

        # Detect import blocks
        if re.match(r'^(import |from \w+ import |use |require\(|#include)', stripped):
            import_count += 1
            in_imports = True
            if import_count <= 5:
                result.append(line)
            elif import_count == 6:
                result.append(f"# ... {import_count} total imports, showing first 5")
            continue

        if in_imports and stripped == "":
            in_imports = False
            result.append(line)
            continue

        if in_imports:
            continue

        result.append(line)

    return "\n".join(result)


def auto_compress(text: str, source: str = "unknown") -> str:
    """Auto-detect compression type from content."""
    if len(text) < 2000:
        return text

    if re.search(r'(?i)(error|traceback|exit code|\$ |# |~ )', text[:500]):
        return compress_terminal(text)

    if re.search(r'(?i)(http|title|description|snippet|url)', text[:500]):
        return compress_search(text)

    return compress_file(text)


def main():
    parser = argparse.ArgumentParser(description="TokenJuice compression")
    parser.add_argument("--type", choices=["terminal", "web", "search", "file", "auto"],
                        default="auto", help="Compression type")
    parser.add_argument("--input", help="Input file path (reads stdin if omitted)")
    parser.add_argument("--stats", action="store_true", help="Show compression stats")
    args = parser.parse_args()

    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    if not text.strip():
        print(text, end="")
        return

    original_len = len(text)
    compressors = {
        "terminal": compress_terminal,
        "web": compress_web,
        "search": compress_search,
        "file": compress_file,
        "auto": auto_compress,
    }

    result = compressors[args.type](text)
    result = result if isinstance(result, str) else result

    if args.stats:
        new_len = len(result)
        saved = original_len - new_len
        pct = (saved / original_len * 100) if original_len > 0 else 0
        print(f"[TokenJuice: {original_len} → {new_len} chars ({pct:.0f}% saved)]\n", file=sys.stderr)

    print(result, end="")


if __name__ == "__main__":
    main()
