from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlopen


DISCOVERY_URLS = {
    "youtube": "https://youtube.googleapis.com/$discovery/rest?version=v3",
    "youtubeAnalytics": "https://youtubeanalytics.googleapis.com/$discovery/rest?version=v2",
    "youtubeReporting": "https://youtubereporting.googleapis.com/$discovery/rest?version=v1",
}

SKILL_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = SKILL_ROOT / "references" / "discovery_cache"


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for service, url in DISCOVERY_URLS.items():
        cache_file = CACHE_DIR / f"{service}.json"
        try:
            with urlopen(url, timeout=30) as response:  # nosec: official discovery URL
                data = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            if cache_file.exists():
                print(f"{service}: kept existing cache after fetch failure: {exc}")
                continue
            raise
        cache_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{service}: wrote {cache_file}")

    print("Refresh complete. Regenerate derived references from discovery_cache using the skill build process.")


if __name__ == "__main__":
    main()
