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
from pathlib import Path

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
