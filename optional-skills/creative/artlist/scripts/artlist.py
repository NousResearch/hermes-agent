#!/usr/bin/env python3
"""Artlist connector: search via internal API + download WAV via browser.

Credential-free by design. The user logs in to artlist.io ONCE in a persistent
Playwright Chromium profile; this script reads that profile's state.json session
cookie, exchanges it for a short-lived access token, and uses it for API calls.

Usage:
  python artlist.py search   --profile <dir> --term "elegant" [--category 62] [--vocal INSTRUMENTAL]
  python artlist.py download --profile <dir> --url <song-url> --out <dir>
"""
import argparse
import json
import os
import time
from pathlib import Path

import requests

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"

SONG_LIST_QUERY = """
query SongList($page: Int!, $songSortType: SongSortType!, $take: Int!, $vocalType: VocalType!, $categoryIds: [Int], $searchTerm: String) {
  songList(page: $page, songSortType: $songSortType, take: $take, vocalType: $vocalType, categoryIds: $categoryIds, searchTerm: $searchTerm) {
    songs { songId, songName, artistName, nameForURL, durationTime, genreCategories { id, name } }
  }
}
"""


def get_access_token(profile_dir):
    """Exchange the Playwright profile's session cookie for a short-lived access token."""
    state = json.loads(Path(profile_dir, "state.json").read_text(encoding="utf-8"))
    cookie = "; ".join(
        f"{c['name']}={c['value']}" for c in state.get("cookies", []) if "artlist.io" in c.get("domain", "")
    )
    r = requests.get(
        "https://artlist.io/api/auth/session",
        headers={"User-Agent": UA, "Accept": "application/json", "Cookie": cookie},
        timeout=25,
    )
    r.raise_for_status()
    return r.json().get("accessToken", "")


def _headers(token):
    return {
        "User-Agent": UA,
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Origin": "https://artlist.io",
        "Referer": "https://artlist.io/",
    }


def search(profile_dir, term=None, category_ids=None, vocal="INSTRUMENTAL", take=15):
    token = get_access_token(profile_dir)
    r = requests.post(
        "https://search-api.artlist.io/v2/graphql",
        json={
            "query": SONG_LIST_QUERY,
            "variables": {
                "page": 1,
                "take": take,
                "songSortType": "NEWEST",
                "vocalType": vocal,
                "categoryIds": category_ids,
                "searchTerm": term,
            },
        },
        headers=_headers(token),
        timeout=30,
    )
    r.raise_for_status()
    return (r.json().get("data") or {}).get("songList", {}).get("songs", [])


def song_url(name_for_url, song_id):
    return f"https://artlist.io/royalty-free-music/song/{name_for_url}/{song_id}"


def download(profile_dir, url, out_dir):
    from playwright.sync_api import sync_playwright

    os.makedirs(out_dir, exist_ok=True)
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            str(profile_dir),
            headless=False,
            no_viewport=True,
            accept_downloads=True,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
        )
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=90000)
        time.sleep(7)

        center = page.evaluate(
            "() => { for (const e of document.querySelectorAll('[aria-label=\"direct download\"]'))"
            " { const r = e.getBoundingClientRect(); if (r.width > 5 && r.height > 5)"
            " return { x: r.x + r.width / 2, y: r.y + r.height / 2 }; } return null; }"
        )
        if not center:
            ctx.close()
            raise RuntimeError("download button not found (Cloudflare challenge, or not logged in?)")
        page.mouse.click(center["x"], center["y"])
        time.sleep(2)

        wav = page.evaluate(
            "() => { for (const e of document.querySelectorAll('button,a,li,span,div,[role=\"menuitem\"]'))"
            " { if ((e.innerText || '').trim() === 'WAV') { const r = e.getBoundingClientRect();"
            " if (r.width > 5 && r.height > 5) return { x: r.x + r.width / 2, y: r.y + r.height / 2 }; } } return null; }"
        )
        if not wav:
            ctx.close()
            raise RuntimeError("WAV option not found after clicking download")

        with page.expect_download(timeout=90000) as dli:
            page.mouse.click(wav["x"], wav["y"])
        dl = dli.value
        dest = os.path.join(out_dir, dl.suggested_filename)
        dl.save_as(dest)
        ctx.close()
        return dest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Artlist search + WAV download connector")
    ap.add_argument("--profile", required=True, help="Playwright persistent profile dir containing state.json")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="search songs")
    s.add_argument("--term")
    s.add_argument("--category", type=int, action="append")
    s.add_argument("--vocal", default="INSTRUMENTAL")

    d = sub.add_parser("download", help="download one WAV")
    d.add_argument("--url", required=True)
    d.add_argument("--out", required=True)

    a = ap.parse_args()
    if a.cmd == "search":
        for x in search(a.profile, term=a.term, category_ids=a.category, vocal=a.vocal):
            print(f"{x['songId']} | {x['songName']} | {x['artistName']} | "
                  f"{x.get('durationTime')} | {song_url(x.get('nameForURL', ''), x['songId'])}")
    else:
        print(download(a.profile, a.url, a.out))
