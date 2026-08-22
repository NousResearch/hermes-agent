"""Fetch an OpenRouter model page and extract endpoint provider slugs + quantization levels.

Usage: python3 or_endpoints.py <model-id>
Example: python3 or_endpoints.py deepseek/deepseek-v4-flash

No API key needed — reads the public model page HTML and pulls the embedded
per-endpoint data (provider_slug + quantization). Variant suffixes (:free,
:nitro, ...) are stripped automatically.
"""
import re
import sys
import urllib.request


def get_endpoints(model: str) -> list[tuple[str, str]]:
    # strip variant suffix
    base = model.split(":", 1)[0]
    url = f"https://openrouter.ai/{base}"
    req = urllib.request.Request(
        url,
        headers={"Accept": "text/html", "User-Agent": "Mozilla/5.0"},
    )
    html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")

    # Page embeds endpoint rows as escaped JSON:
    #   provider_slug\":\"<slug>\",... quantization\":\"<level>\"
    pairs: set[tuple[str, str]] = set()
    pat = re.compile(r'provider_slug\\":\\"([^"\\]+)\\"')
    for m in pat.finditer(html):
        tail = html[m.end() : m.end() + 400]
        qm = re.search(r'quantization\\":\\"([^"\\]+)\\"', tail)
        if qm:
            pairs.add((m.group(1), qm.group(1)))
    return sorted(pairs)


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: or_endpoints.py <model-id>", file=sys.stderr)
        sys.exit(2)
    model = sys.argv[1]
    try:
        rows = get_endpoints(model)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    if not rows:
        print(f"{model}: no endpoint data found (layout may have changed or model is new)")
        sys.exit(0)
    for slug, q in rows:
        print(f"{slug:30s} {q}")
    print("---")
    print("unique quantizations:", ", ".join(sorted({q for _, q in rows})))


if __name__ == "__main__":
    main()
