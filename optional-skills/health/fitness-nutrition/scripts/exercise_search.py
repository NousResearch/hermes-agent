#!/usr/bin/env python3
"""Search or filter wger exercises and emit compact JSON."""

import argparse
from html.parser import HTMLParser
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request


WGER_SEARCH_URL = "https://wger.de/api/v2/exerciseinfo/"
ENGLISH_LANGUAGE_ID = 2


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        text = data.strip()
        if text:
            self.parts.append(text)


def _plain_text(value):
    parser = _TextExtractor()
    parser.feed(value or "")
    return " ".join(parser.parts)


def _names(items):
    return [
        item.get("name_en") or item.get("name")
        for item in items
        if item.get("name_en") or item.get("name")
    ]


def _compact(exercise):
    translation = next(
        (
            item
            for item in exercise.get("translations", [])
            if item.get("language") == ENGLISH_LANGUAGE_ID
        ),
        {},
    )
    images = exercise.get("images") or []
    return {
        "id": exercise.get("id"),
        "name": translation.get("name", ""),
        "category": (exercise.get("category") or {}).get("name", ""),
        "primary_muscles": _names(exercise.get("muscles") or []),
        "secondary_muscles": _names(exercise.get("muscles_secondary") or []),
        "equipment": _names(exercise.get("equipment") or []),
        "description": _plain_text(translation.get("description", "")),
        "image": images[0].get("image", "") if images else "",
    }


def _name_tokens(value):
    """Return normalized tokens for conservative exercise-name matching."""
    normalized = []
    for token in re.findall(r"[a-z0-9]+", value.lower()):
        if token == "ups":
            token = "up"
        elif token == "downs":
            token = "down"
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        normalized.append(token)
    return tuple(normalized)


def _relevance_key(record, query):
    """Prefer exact/all-token names over wger's broad full-text rank."""
    query_tokens = _name_tokens(query)
    name = record.get("name", "")
    candidate_tokens = _name_tokens(name)
    query_compact = "".join(query_tokens)
    candidate_compact = "".join(candidate_tokens)
    compact_exact = candidate_compact != query_compact
    missing = len(set(query_tokens) - set(candidate_tokens))
    token_exact = candidate_tokens != query_tokens
    extras = len(set(candidate_tokens) - set(query_tokens))
    return (
        compact_exact,
        missing,
        token_exact,
        extras,
        len(candidate_tokens),
        name.lower(),
    )


def _limit(value):
    parsed = int(value)
    if not 1 <= parsed <= 20:
        raise argparse.ArgumentTypeError("limit must be between 1 and 20")
    return parsed


def _positive_id(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("filter IDs must be positive integers")
    return parsed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", help="exercise name or partial name")
    parser.add_argument("--limit", type=_limit, default=5)
    parser.add_argument("--muscle", type=_positive_id, help="wger muscle ID")
    parser.add_argument("--category", type=_positive_id, help="wger category ID")
    parser.add_argument("--equipment", type=_positive_id, help="wger equipment ID")
    args = parser.parse_args(argv)

    query = (args.query or "").strip()
    filters = {
        name: value
        for name, value in (
            ("muscles", args.muscle),
            ("category", args.category),
            ("equipment", args.equipment),
        )
        if value is not None
    }
    if args.query is not None and not _name_tokens(query):
        print("Error: query must contain letters or numbers", file=sys.stderr)
        return 2
    if not query and not filters:
        print("Error: provide a query or at least one filter", file=sys.stderr)
        return 2

    request_params = {
        "language__code": "en",
        "limit": 100,
        "format": "json",
        **filters,
    }
    if query:
        request_params["name__search"] = query
    params = urllib.parse.urlencode(request_params)
    request = urllib.request.Request(
        f"{WGER_SEARCH_URL}?{params}",
        headers={
            "Accept": "application/json",
            "User-Agent": "Hermes-Agent/fitness-nutrition",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"Error: wger request failed: {exc}", file=sys.stderr)
        return 1

    results = [_compact(item) for item in payload.get("results", [])]
    if query:
        results.sort(key=lambda item: _relevance_key(item, query))
    else:
        results.sort(key=lambda item: item.get("name", "").lower())
    result = {
        "query": query,
        "count": payload.get("count", 0),
        "results": results[: args.limit],
    }
    if filters:
        result["filters"] = filters
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
