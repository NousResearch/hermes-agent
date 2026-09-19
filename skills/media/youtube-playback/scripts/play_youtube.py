#!/usr/bin/env python3
"""Autonomous YouTube search and playback script for Hermes Agent.

Provides ultra-low-latency YouTube search and instant browser launch
without requiring third-party dependencies or Google API keys.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
import webbrowser
from typing import Any, Dict, List, Optional

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

_YOUTUBE_URL_PATTERNS = [
    re.compile(r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})"),
    re.compile(r"(?:https?://)?youtu\.be/([a-zA-Z0-9_-]{11})"),
    re.compile(r"(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})"),
    re.compile(r"(?:https?://)?(?:www\.)?youtube\.com/shorts/([a-zA-Z0-9_-]{11})"),
]


def extract_video_id(text: str) -> Optional[str]:
    """Return an 11-char YouTube video ID if text contains or is a valid ID/URL."""
    text = text.strip()
    if re.fullmatch(r"[a-zA-Z0-9_-]{11}", text):
        return text
    for pattern in _YOUTUBE_URL_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1)
    return None


def search_youtube(query: str, max_results: int = 5, timeout: int = 6) -> List[Dict[str, Any]]:
    """Search YouTube for a query and return structured video metadata."""
    direct_id = extract_video_id(query)
    if direct_id:
        return [{
            "id": direct_id,
            "title": f"YouTube Video ({direct_id})",
            "url": f"https://www.youtube.com/watch?v={direct_id}",
            "duration": "--:--",
            "channel": "YouTube",
            "thumbnail": f"https://i.ytimg.com/vi/{direct_id}/hqdefault.jpg",
        }]

    encoded_query = urllib.parse.quote_plus(query)
    url = f"https://www.youtube.com/results?search_query={encoded_query}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
    except Exception as exc:
        return [{
            "error": f"Search request failed: {exc}",
            "query": query,
            "fallback_url": f"https://www.youtube.com/results?search_query={encoded_query}",
        }]

    results: List[Dict[str, Any]] = []

    # 1. Primary extraction: parse ytInitialData JSON for rich metadata
    data_match = re.search(r"var ytInitialData\s*=\s*({.*?});</script>", html, re.DOTALL)
    if not data_match:
        data_match = re.search(r"ytInitialData\s*=\s*({.*?});", html, re.DOTALL)

    if data_match:
        try:
            data = json.loads(data_match.group(1))
            section_list = (
                data.get("contents", {})
                .get("twoColumnSearchResultsRenderer", {})
                .get("primaryContents", {})
                .get("sectionListRenderer", {})
                .get("contents", [])
            )
            for section in section_list:
                item_section = section.get("itemSectionRenderer", {}).get("contents", [])
                for item in item_section:
                    v = item.get("videoRenderer")
                    if not v:
                        continue
                    vid_id = v.get("videoId")
                    if not vid_id:
                        continue

                    # Extract title
                    title_runs = v.get("title", {}).get("runs", [])
                    title = title_runs[0].get("text") if title_runs else "Unknown Title"

                    # Extract channel
                    owner_runs = v.get("ownerText", {}).get("runs", [])
                    channel = owner_runs[0].get("text") if owner_runs else ""

                    # Extract duration
                    duration = v.get("lengthText", {}).get("simpleText", "--:--")

                    # Extract thumbnail
                    thumbs = v.get("thumbnail", {}).get("thumbnails", [])
                    thumb_url = thumbs[-1].get("url") if thumbs else f"https://i.ytimg.com/vi/{vid_id}/hqdefault.jpg"

                    results.append({
                        "id": vid_id,
                        "title": title,
                        "url": f"https://www.youtube.com/watch?v={vid_id}",
                        "duration": duration,
                        "channel": channel,
                        "thumbnail": thumb_url,
                    })

                    if len(results) >= max_results:
                        break
                if len(results) >= max_results:
                    break
        except Exception:
            pass

    # 2. Fast regex fallback if JSON structure changed or was truncated
    if not results:
        found_ids = list(dict.fromkeys(re.findall(r"/watch\?v=([a-zA-Z0-9_-]{11})", html)))
        for vid_id in found_ids[:max_results]:
            results.append({
                "id": vid_id,
                "title": f"YouTube Video: {query}",
                "url": f"https://www.youtube.com/watch?v={vid_id}",
                "duration": "--:--",
                "channel": "YouTube",
                "thumbnail": f"https://i.ytimg.com/vi/{vid_id}/hqdefault.jpg",
            })

    if not results:
        # Ultimate fallback: return search URL
        results.append({
            "id": "",
            "title": f"YouTube Search: {query}",
            "url": f"https://www.youtube.com/results?search_query={encoded_query}",
            "duration": "--:--",
            "channel": "YouTube Search",
            "thumbnail": "",
        })

    return results


def open_in_browser(url: str, autoplay: bool = True) -> bool:
    """Open YouTube video directly in default system browser with autoplay."""
    target_url = url
    if autoplay and "watch?v=" in target_url and "autoplay=" not in target_url:
        separator = "&" if "?" in target_url else "?"
        target_url = f"{target_url}{separator}autoplay=1"

    try:
        if webbrowser.open(target_url):
            return True
    except Exception:
        pass

    try:
        # Fallback on Windows if webbrowser.open fails
        if sys.platform == "win32":
            import os
            os.startfile(target_url)
            return True
    except Exception as exc:
        print(f"Warning: could not launch browser: {exc}", file=sys.stderr)

    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Search and play YouTube videos or music with ultra-low latency."
    )
    parser.add_argument("query", help="Search query or YouTube URL (e.g. 'Iron Man music')")
    parser.add_argument(
        "--open",
        dest="open_browser",
        action="store_true",
        default=True,
        help="Open video in default browser (default: True)",
    )
    parser.add_argument(
        "--no-open",
        dest="open_browser",
        action="store_false",
        help="Do not open in browser, only return search results",
    )
    parser.add_argument(
        "--autoplay",
        action="store_true",
        default=True,
        help="Add autoplay parameter to watch URL",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=1,
        help="Maximum results to return (default: 1 for instant play)",
    )

    args = parser.parse_args()
    results = search_youtube(args.query, max_results=args.max_results)

    if not results or "error" in results[0]:
        err_msg = results[0].get("error", "No videos found") if results else "No videos found"
        if args.json:
            print(json.dumps({"success": False, "error": err_msg, "results": []}))
        else:
            print(f"Error: {err_msg}", file=sys.stderr)
        return 1

    top = results[0]
    opened = False
    if args.open_browser and top.get("url"):
        opened = open_in_browser(top["url"], autoplay=args.autoplay)

    if args.json:
        print(json.dumps({
            "success": True,
            "opened": opened,
            "query": args.query,
            "top": top,
            "results": results,
        }, indent=2))
    else:
        print(f"▶ Playing: {top.get('title')}")
        print(f"🔗 URL: {top.get('url')}")
        if top.get("duration") and top.get("duration") != "--:--":
            print(f"⏱ Duration: {top.get('duration')} | Channel: {top.get('channel')}")
        if opened:
            print("🚀 Autonomous playback launched in default browser.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
