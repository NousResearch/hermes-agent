"""Regression for #125350: pinned BtbN ffmpeg artifacts must still exist upstream.

BtbN prunes old dated autobuild tags, so a rotted lockfile pin 404s on GitHub
and the sha256 mirror has no object for a download that never succeeded —
fresh installs die in the python-deps stage while every other package is fine.
This walks the REAL pinned URLs (no mock): each must still be reachable.
Skips when the host has no network — the assertion is upstream availability,
not local connectivity.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _btbn_ffmpeg_urls():
    lock = json.loads((REPO_ROOT / "pm" / "lock.json").read_text(encoding="utf-8"))
    artifacts = lock["packages"]["ffmpeg"]["artifacts"]
    for target in sorted(artifacts):
        url = artifacts[target]["url"]
        if "BtbN/FFmpeg-Builds" in url:
            yield target, url


def _online() -> bool:
    try:
        urllib.request.urlopen("https://api.github.com", timeout=10).close()
        return True
    except OSError:
        return False


@pytest.mark.parametrize("target,url", sorted(_btbn_ffmpeg_urls()))
def test_pinned_btbn_artifact_is_still_hosted(target, url):
    if not _online():
        pytest.skip("no network; upstream availability unverifiable here")
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        assert response.status == 200, f"{target}: pinned artifact is gone: {url}"
