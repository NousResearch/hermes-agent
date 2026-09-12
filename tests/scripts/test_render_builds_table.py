"""render-builds-table.py: tables from REAL bucket object names, spliced idempotently.

The contract: table rows exist only for objects that are actually in the
R2 bucket for the tag's exact version (missing artifact = missing row,
never a dead link), links point at the R2 public URL, msixbundle / zip /
feed manifests stay out, and re-rendering replaces the previous block
instead of stacking a second one.

Adapted from the restack suite for this branch's artifact shapes: the
Windows per-arch artifact here is .msix (the msixbundle folds both arches
and stays out of the tables), not the NSIS .exe.
"""

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen

import pytest

from tests.scripts.test_release_r2 import r2_server  # noqa: F401

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "render-builds-table.py"
_SPEC = importlib.util.spec_from_file_location("render_builds_table", _SCRIPT)
assert _SPEC and _SPEC.loader
rbt = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rbt)


ASSETS = [
    "releases/tag/v0.28.0/HermesBundled-0.28.0-mac-arm64.dmg",
    "releases/tag/v0.28.0/HermesBundled-0.28.0-mac-arm64.zip",           # updater delta target — no row
    "releases/tag/v0.28.0/HermesBundled-0.28.0-win-x64.msix",
    "releases/tag/v0.28.0/HermesBundled-0.28.0-win-arm64.msix",
    "releases/tag/v0.28.0/HermesBundled-0.28.0-win.msixbundle",          # store/sideload channel — no row
    "releases/tag/v0.28.0/HermesBundled-0.28.0-linux-x64.AppImage",
    "releases/tag/v0.28.0/HermesLight-0.28.0-win-x64.msix",
    "latest.yml",                                   # feed manifest — no row
    "light.yml",
    "releases/tag/v0.28.0/HermesBundled-0.28.0-win-x64.msix.blockmap",   # no row
]

BASE_URL = "https://releases.example.com"


class TestParseAssets:
    def test_only_table_shaped_assets_parse(self):
        parsed = rbt.parse_assets(ASSETS)
        assert ("mac", "arm64") in parsed["HermesBundled"]
        assert ("win", "x64") in parsed["HermesBundled"]
        assert ("win", "arm64") in parsed["HermesBundled"]
        assert ("linux", "x64") in parsed["HermesBundled"]
        assert len(parsed["HermesBundled"]) == 4   # zip/msixbundle/blockmap/yml excluded
        assert parsed["HermesLight"] == {("win", "x64"): ("releases/tag/v0.28.0/HermesLight-0.28.0-win-x64.msix", "msix")}

    def test_canary_versions_parse(self):
        parsed = rbt.parse_assets(["releases/tag/v0.28.0-canary.20260818/HermesBundled-0.28.0-canary.20260818-win-x64.msix"])
        assert ("win", "x64") in parsed["HermesBundled"]

    def test_flat_names_still_parse(self):
        # Names without the releases/tag/ prefix (e.g. from a plain list) are
        # handled too — the shape match runs on the basename either way.
        parsed = rbt.parse_assets(["HermesBundled-0.28.0-win-x64.msix"])
        assert parsed["HermesBundled"][("win", "x64")] == ("HermesBundled-0.28.0-win-x64.msix", "msix")


class TestFilterNamesForVersion:
    def test_stable_tag_does_not_match_canary_objects(self):
        # '0.28.0' is a prefix of '0.28.0-canary...' — the filter must be
        # exact, or a stable table would list canary binaries.
        names = [
            "releases/tag/v0.28.0/HermesBundled-0.28.0-mac-arm64.dmg",
            "releases/tag/v0.28.0-canary.20260818/HermesBundled-0.28.0-canary.20260818-win-x64.msix",
            "releases/tag/v0.29.0/HermesBundled-0.29.0-win-x64.msix",
            "latest.yml",
        ]
        assert rbt.filter_names_for_version(names, "0.28.0") == [
            "releases/tag/v0.28.0/HermesBundled-0.28.0-mac-arm64.dmg",
        ]

    def test_canary_tag_matches_its_objects(self):
        names = [
            "releases/tag/v0.28.0-canary.20260818/HermesBundled-0.28.0-canary.20260818-win-x64.msix",
            "releases/tag/v0.28.0-canary.20260818/HermesBundled-0.28.0-canary.20260818-win-x64.msix.blockmap",
            "releases/tag/v0.28.0-canary.20260817/HermesBundled-0.28.0-canary.20260817-win-arm64.msix",
            "releases/tag/v0.28.0/HermesBundled-0.28.0-mac-arm64.dmg",
        ]
        assert rbt.filter_names_for_version(names, "0.28.0-canary.20260818") == [
            "releases/tag/v0.28.0-canary.20260818/HermesBundled-0.28.0-canary.20260818-win-x64.msix",
        ]


class TestRenderAndSplice:
    def test_rows_only_for_present_assets(self):
        block = rbt.render_tables(rbt.parse_assets(ASSETS), BASE_URL)
        assert f"{BASE_URL}/releases/tag/v0.28.0/HermesBundled-0.28.0-mac-arm64.dmg" in block
        assert "HermesBundled-0.28.0-win.msixbundle" not in block
        assert ".zip" not in block
        assert ".blockmap" not in block
        # A leg that never uploaded leaves no row at all.
        assert "linux-arm64" not in block

    def test_links_point_at_the_r2_base_url(self):
        block = rbt.render_tables(rbt.parse_assets(ASSETS), BASE_URL)
        assert "github.com" not in block
        assert f"{BASE_URL}/releases/tag/v0.28.0/HermesLight-0.28.0-win-x64.msix" in block

    def test_base_url_trailing_slash_is_stripped(self):
        block = rbt.render_tables(rbt.parse_assets(ASSETS), f"{BASE_URL}/")
        assert f"{BASE_URL}/releases/tag/v0.28.0/HermesBundled-0.28.0-mac-arm64.dmg" in block
        assert f"{BASE_URL}//releases" not in block

    def test_splice_replaces_marker_and_is_idempotent(self):
        body = f"# Notes\n\n{rbt.MARKER}\n\n## Changes"
        block = rbt.render_tables(rbt.parse_assets(ASSETS), BASE_URL)
        once = rbt.splice(body, block)
        assert "## Downloads" in once
        assert once.count(rbt.MARKER) == 1
        # Second render (e.g. a re-run with more assets) replaces, not stacks.
        twice = rbt.splice(once, block)
        assert twice.count("## Downloads") == 1
        assert twice.count(rbt.END_MARKER) == 1

    def test_no_marker_leaves_body_alone(self):
        block = rbt.render_tables(rbt.parse_assets(ASSETS), BASE_URL)
        assert rbt.splice("no marker here", block) == "no marker here"


class TestPendingPlaceholder:
    def test_final_render_replaces_the_pending_link(self):
        # The lifecycle: marker → pending link (builds-pending job) →
        # tables (builds-table job). The link must not survive step 3.
        body = f"# Notes\n\n{rbt.MARKER}\n\n## Changes"
        pending = rbt.render_pending("https://github.com/o/r/actions/runs/123")
        with_pending = rbt.splice(body, pending)
        assert "actions/runs/123" in with_pending
        assert with_pending.count(rbt.MARKER) == 1  # wrapper survives for step 3
        tables = rbt.render_tables(rbt.parse_assets(ASSETS), BASE_URL)
        final = rbt.splice(with_pending, tables)
        assert "actions/runs/123" not in final
        assert "## Downloads" in final
        assert final.count(rbt.END_MARKER) == 1

    def test_pending_rerender_replaces_not_stacks(self):
        once = rbt.splice(rbt.MARKER, rbt.render_pending("https://x/runs/1"))
        twice = rbt.splice(once, rbt.render_pending("https://x/runs/2"))
        assert "https://x/runs/1" not in twice
        assert twice.count("https://x/runs/2") == 1
        assert twice.count(rbt.END_MARKER) == 1


class TestBucketPage:
    """The page is the same row set as the release-body table.

    Every download link in the markdown table exists in the page, and the
    artifacts the table deliberately hides (zip/msixbundle/blockmap) stay
    out of the page too — a page that outlives the release body must not
    advertise a delta target as a download.
    """

    def test_page_carries_exactly_the_table_rows(self):
        block = rbt.render_tables(rbt.parse_assets(ASSETS), BASE_URL)
        page = rbt.render_page("v0.28.0", rbt.parse_assets(ASSETS), BASE_URL)
        links = re.findall(r"\]\((https?://[^)]+)\)", block)
        assert links  # the table is non-empty, so the comparison means something
        for url in links:
            assert f'href="{url}"' in page
        # One row per table row, plus one header row per section.
        assert page.count("<tr>") == len(links) + page.count("<table>")
        assert ".zip" not in page and ".blockmap" not in page and ".msixbundle" not in page
        # A leg that never uploaded has no row in either sink, while a leg
        # that did (linux-x64) is listed.
        assert BASE_URL + "/releases/tag/v0.28.0/HermesBundled-0.28.0-linux-x64.AppImage" in page
        assert "linux-arm64" not in page

    def test_page_names_the_build_it_describes(self):
        page = rbt.render_page("v0.28.0", rbt.parse_assets(ASSETS), BASE_URL)
        assert rbt.recorded_build(page) == "v0.28.0"
        canary = rbt.render_page("v0.28.0-canary.20260818101010", rbt.parse_assets(ASSETS), BASE_URL)
        assert rbt.recorded_build(canary) == "v0.28.0-canary.20260818101010"
        assert "canary" in canary

    def test_older_tag_never_regresses_the_channel_page(self):
        """A re-run of an old tag must not overwrite the page a newer release
        published; the recorded tag is compared with the feed's own authority."""
        current = rbt.render_page("v0.29.0", rbt.parse_assets(ASSETS), BASE_URL)
        assert not rbt.supersedes(current, "v0.28.0")
        assert not rbt.supersedes(current, "v0.28.0-canary.20260818101010")
        assert rbt.supersedes(current, "v0.29.0")   # same tag re-run rewrites
        assert rbt.supersedes(current, "v0.30.0")
        newer_canary = rbt.render_page("v0.29.0-canary.20260901090000", rbt.parse_assets(ASSETS), BASE_URL)
        assert not rbt.supersedes(newer_canary, "v0.29.0-canary.20260801090000")
        assert rbt.supersedes(newer_canary, "v0.29.0-canary.20260901090001")
        # Nothing published yet, or an unreadable record: the new page wins.
        assert rbt.supersedes(None, "v0.28.0")
        assert rbt.supersedes("<html>garbage</html>", "v0.28.0")

    def test_channel_page_key_follows_the_tag(self):
        assert rbt.r2.channel_page_key_for(rbt.r2.channel_for_tag("v0.28.0")) == "releases/stable/index.html"
        assert rbt.r2.channel_page_key_for(
            rbt.r2.channel_for_tag("v0.28.0-canary.20260818101010")) == "releases/canary/index.html"


class TestTagRunPublishesThePage:
    """The real CLI path: the page is uploaded for the tag's own channel."""

    TAG = "v0.28.0-canary.20260818101010"
    KEYS = [
        f"releases/tag/{TAG}/HermesBundled-0.28.0-canary.20260818101010-win-x64.msix",
        f"releases/tag/{TAG}/HermesBundled-0.28.0-canary.20260818101010-mac-arm64.dmg",
        f"releases/tag/{TAG}/HermesBundled-0.28.0-canary.20260818101010-win.msixbundle",
        f"releases/tag/{TAG}/HermesLight-0.28.0-canary.20260818101010-win-x64.msix",
        "releases/tag/v0.27.0/HermesBundled-0.27.0-win-x64.msix",  # neighbor release
    ]

    @staticmethod
    def _gh(argv, **kwargs):
        if argv[:3] == ["gh", "release", "view"]:
            body = f"# Notes\n\n{rbt.MARKER}\n\n## Changes\n- x\n"
            return subprocess.CompletedProcess(argv, 0, stdout=json.dumps({"body": body}), stderr="")
        if argv[:3] == ["gh", "release", "edit"]:
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        raise AssertionError(argv)

    def test_page_upload_key_and_bytes(self, monkeypatch, capsys, tmp_path):
        monkeypatch.setenv("RELEASE_NEEDS", json.dumps({
            "validate": {"result": "success"}, "build-win32": {"result": "success"},
            "publish-win32-updater": {"result": "success"},
        }))
        uploads: list[tuple[str, str, bool]] = []
        monkeypatch.setattr(rbt.r2, "list_objects", lambda prefix="": {"keys": self.KEYS})
        monkeypatch.setattr(rbt, "existing_page", lambda key: None)
        monkeypatch.setattr(rbt.r2, "put", lambda tag, key, file, key_is_full=False, immutable=False:
                            uploads.append((key, Path(file).read_text(encoding="utf-8"), key_is_full)))
        monkeypatch.setattr(rbt.subprocess, "run", self._gh)
        monkeypatch.setattr(sys, "argv", [
            "render-builds-table.py", "--tag", self.TAG, "--repo", "o/r", "--r2-base-url", BASE_URL,
        ])
        assert rbt.main() == 0
        assert len(uploads) == 2
        assert uploads[0][0] == f"releases/tag/{self.TAG}/index.html"
        key, page, key_is_full = uploads[1]
        assert uploads[0][1] == page
        # Canary tag → the canary channel page, as a full key (not a tag name).
        assert key == "releases/canary/index.html" and key_is_full
        assert rbt.recorded_build(page) == self.TAG
        for name in ("HermesBundled-0.28.0-canary.20260818101010-win-x64.msix",
                     "HermesBundled-0.28.0-canary.20260818101010-mac-arm64.dmg",
                     "HermesLight-0.28.0-canary.20260818101010-win-x64.msix"):
            assert f"{BASE_URL}/releases/tag/{self.TAG}/{name}" in page
        # The neighbor release and the hidden artifact shapes stay out.
        assert "v0.27.0" not in page and ".msixbundle" not in page
        assert f"✓ Page {BASE_URL}/releases/canary/index.html" in capsys.readouterr().out

    def test_stale_tags_keep_their_own_page_and_dry_runs_write_nothing(self, monkeypatch, tmp_path):
        uploads: list[str] = []
        monkeypatch.setattr(rbt.r2, "list_objects", lambda prefix="": {"keys": self.KEYS})
        monkeypatch.setattr(rbt.r2, "put", lambda **kwargs: uploads.append(kwargs["key"]))
        monkeypatch.setattr(rbt.subprocess, "run", self._gh)
        stale = rbt.render_page("v0.29.0", rbt.parse_assets(ASSETS), BASE_URL)
        monkeypatch.setattr(rbt, "existing_page", lambda key: stale)
        monkeypatch.setattr(sys, "argv", [
            "render-builds-table.py", "--tag", self.TAG, "--repo", "o/r", "--r2-base-url", BASE_URL,
        ])
        assert rbt.main() == 0                      # stale tag: channel untouched
        assert uploads == [f"releases/tag/{self.TAG}/index.html"]
        uploads.clear()
        monkeypatch.setattr(rbt, "existing_page", lambda key: None)
        monkeypatch.setattr(sys, "argv", [
            "render-builds-table.py", "--tag", self.TAG, "--repo", "o/r",
            "--r2-base-url", BASE_URL, "--dry-run",
        ])
        assert rbt.main() == 0                      # dry run: nothing published
        assert uploads == []

    @pytest.mark.parametrize("asset_present", [False, True])
    @pytest.mark.parametrize("result", ["failure", "cancelled", "skipped"])
    def test_incomplete_tag_is_readable_without_replacing_last_good_channel(
        self, monkeypatch, r2_server, asset_present, result,
    ):
        base = f"http://127.0.0.1:{r2_server.server_port}/hermes-releases"
        channel_key = "releases/canary/index.html"
        previous = rbt.render_page("v0.27.0-canary.20260817101010", {}, base).encode()
        r2_server.store[channel_key] = (previous, "text/html")
        if asset_present:
            r2_server.store[self.KEYS[0]] = (b"transport fixture", "application/octet-stream")
        edits = []

        def gh(argv, **kwargs):
            if argv[:3] == ["gh", "release", "edit"]:
                edits.append(kwargs["input"])
            return self._gh(argv, **kwargs)

        monkeypatch.setattr(rbt.subprocess, "run", gh)
        monkeypatch.setenv("RELEASE_NEEDS", json.dumps({
            "validate": {"result": "success"},
            "build-win32": {"result": "success"},
            "publish-win32-updater": {"result": result},
        }))
        monkeypatch.setattr(sys, "argv", [
            "render-builds-table.py", "--tag", self.TAG, "--r2-base-url", base,
        ])
        assert rbt.main() == 0
        tag_key = f"releases/tag/{self.TAG}/index.html"
        assert tag_key in r2_server.store
        with urlopen(f"{base}/{tag_key}", timeout=5) as response:
            page = response.read().decode()
        assert rbt.recorded_build(page) == self.TAG
        assert "Build incomplete" in page and "Build incomplete" in edits[0]
        assert f"publish-win32-updater ({result})" in page
        assert f"publish-win32-updater ({result})" in edits[0]
        links = re.findall(r'href="([^"]+)"', page)
        assert links == ([f"{base}/{self.KEYS[0]}"] if asset_present else [])
        assert r2_server.store[channel_key][0] == previous
        if not asset_present:
            assert "No downloadable artifacts" in page
