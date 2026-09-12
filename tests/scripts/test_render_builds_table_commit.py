"""Commit-mode summary renderer: the receipt-bound expected-binary matrix.

The commit summary is a SEPARATE sink from the release-body tables: it
never reads or edits a GitHub release, and every expected binary gets a
row — Built (receipt + object both present, with a link) or Not built —
so a skipped or interrupted leg can never be a silent omission or a
forged success.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "render-builds-table.py"
_SPEC = importlib.util.spec_from_file_location("render_builds_table", _SCRIPT)
assert _SPEC and _SPEC.loader
rbt = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rbt)

COMMIT = "a" * 40
BASE = "https://cdn.example.com"


def _leg_for(basename: str) -> str:
    """The receipt name whose regex matches this artifact basename."""
    import re
    for _label, leg, pattern in rbt._COMMIT_EXPECTED:
        if re.match(pattern, basename):
            return leg
    raise AssertionError(f"no expected row matches {basename!r}")


# Every produced binary of a fully green commit run (Termux's .deb is a
# nested deb/ path — the workflow stages it under deb/pool/).
_ALL_BASENAMES = [
    "HermesBundled-0.28.0-win-x64.msix",
    "HermesBundled-0.28.0-win-arm64.msix",
    "HermesBundled-0.28.0-win.msixbundle",
    "HermesBundled-0.28.0-mac-arm64.dmg",
    "HermesBundled-0.28.0-mac-x64.dmg",
    "HermesBundled-0.28.0-mac-arm64.zip",
    "HermesBundled-0.28.0-mac-x64.zip",
    "deb/pool/hermes-agent_0.28.0_aarch64.deb",
]


def _receipts_all_built() -> dict[str, dict | None]:
    """Validated receipts: each leg lists exactly the artifact basenames
    the matrix expects from it (the real staged shape)."""
    receipts: dict[str, dict | None] = {name: None for name in rbt.COMMIT_RECEIPT_NAMES}
    for basename in _ALL_BASENAMES:
        leg = _leg_for(basename.rsplit("/", 1)[-1])
        receipt = receipts[leg] or {"schema": 2, "commit": COMMIT, "name": leg, "files": []}
        receipt["files"].append({"path": basename, "size": 1, "sha256": "0" * 64})
        receipts[leg] = receipt
    return receipts


def _names(*basenames: str) -> list[str]:
    return [f"releases/commit/{COMMIT}/{name}" for name in basenames]


def _all_built_names() -> list[str]:
    return _names(*_ALL_BASENAMES)


def test_every_expected_binary_gets_a_row_built_or_not():
    receipts = _receipts_all_built()
    rows = rbt.commit_expected_rows(_all_built_names(), receipts)
    assert len(rows) == len(rbt._COMMIT_EXPECTED)
    for row in rows:
        assert row["state"] == "built", row
    summary = rbt.render_commit_summary(_all_built_names(), BASE, COMMIT, receipts)
    assert summary.count("✅ Built") == len(rbt._COMMIT_EXPECTED)
    # zip/Termux rows exist in the COMMIT summary (the release-body
    # table hides zips on purpose; the commit summary shows every binary).
    assert any("ZIP" in row["label"] for row in rows)
    assert any("Termux" in row["label"] for row in rows)
    assert not any("Store" in row["label"] for row in rows)
    assert any("MSIXBUNDLE" in row["label"] for row in rows)
    # The universal sideload bundle links correctly.
    bundle = next(row for row in rows if "MSIXBUNDLE" in row["label"])
    assert bundle["key"].endswith(".msixbundle")


def test_missing_binaries_are_explicit_rows_never_links():
    receipts = _receipts_all_built()
    names = _names("HermesBundled-0.28.0-win-x64.msix")
    summary = rbt.render_commit_summary(names, BASE, COMMIT, receipts)
    built = sum(1 for line in summary.splitlines() if "✅ Built" in line)
    missing = sum(1 for line in summary.splitlines()
                  if "❌ Not built" in line and "release leg disabled" not in line)
    assert built == 1
    assert missing == len(rbt._COMMIT_EXPECTED) - built
    # No link for an object that does not exist.
    assert "win-arm64.msix)" not in summary
    # Failed legs are blamed by name on the RECEIPT-missing rows; without
    # blame the row says incomplete.
    darwin_down = dict(receipts)
    darwin_down["darwin-arm64"] = None
    blamed = rbt.render_commit_summary(names, BASE, COMMIT, darwin_down, failed_legs=["build-darwin"])
    assert "failed: build-darwin" in blamed
    assert "Not built (leg incomplete or upload interrupted)" in rbt.render_commit_summary(names, BASE, COMMIT, darwin_down)


def test_object_without_completion_receipt_never_renders_built():
    """The old draft trusted raw object names: an interrupted upload that
    left artifacts but no receipt looked ✅ Built. Now the receipt is the
    completion marker."""
    receipts = _receipts_all_built()
    names = _all_built_names()  # every object present...
    for leg in ("win32-x64", "termux"):
        receipts[leg] = None  # ...two legs' receipts absent
    summary = rbt.render_commit_summary(names, BASE, COMMIT, receipts)
    for row in rbt.commit_expected_rows(names, receipts):
        if row["leg"] in ("win32-x64", "termux"):
            assert row["state"] != "built"
            assert f"| {row['label']} | ✅ Built |" not in summary
        else:
            assert row["state"] == "built"


def test_orphan_object_with_unrelated_valid_receipt_never_renders_built():
    """An object matching a row's shape, staged under a leg whose VALID
    receipt covers only unrelated files, is NOT evidence that the row's
    binary was built. The old draft accepted ANY receipt object for the
    leg; rows now bind to the receipt's own file paths."""
    receipts = _receipts_all_built()
    # win32-x64's receipt is valid but lists a DIFFERENT file.
    receipts["win32-x64"] = {"schema": 2, "commit": COMMIT, "name": "win32-x64",
                             "files": [{"path": "metadata-windows-x64.json",
                                        "size": 1, "sha256": "0" * 64}]}
    names = _all_built_names()  # the orphan win-x64 msix object exists
    rows = rbt.commit_expected_rows(names, receipts)
    x64 = next(row for row in rows if row["label"] == "Windows x64 (MSIX)")
    assert x64["state"] != "built"
    assert x64["key"] is None
    summary = rbt.render_commit_summary(names, BASE, COMMIT, receipts)
    assert "| Windows x64 (MSIX) | ✅ Built |" not in summary


def test_receipt_without_object_is_reported_not_built():
    receipts = _receipts_all_built()
    summary = rbt.render_commit_summary([], BASE, COMMIT, receipts)
    assert "Not built (receipt present but object missing)" in summary
    assert "✅ Built" not in summary


def test_ambiguous_or_extra_objects_never_render_a_link():
    """Two objects that BOTH match a row's shape AND are listed by the
    leg's receipt are genuinely ambiguous — no link is rendered. An extra
    object the receipt does NOT list cannot poison a row whose own
    receipt-listed object is unique (it is foreign to the binding)."""
    receipts = _receipts_all_built()
    # List both versions in win32-x64's receipt (a torn staging would
    # leave exactly this shape behind two shape-matching objects).
    receipts["win32-x64"]["files"].append(
        {"path": "HermesBundled-0.28.1-win-x64.msix", "size": 1, "sha256": "0" * 64})
    names = _names(
        "HermesBundled-0.28.0-win-x64.msix",
        "HermesBundled-0.28.1-win-x64.msix",
    )
    rows = rbt.commit_expected_rows(names, receipts)
    x64 = next(row for row in rows if row["label"] == "Windows x64 (MSIX)")
    assert x64["state"] == "ambiguous" and x64["key"] is None

    # The extra 0.28.1 object without a receipt listing is NOT ambiguous:
    # the receipt's own 0.28.0 binding stays built.
    plain = _receipts_all_built()
    rows = rbt.commit_expected_rows(
        _names("HermesBundled-0.28.0-win-x64.msix",
               "HermesBundled-0.28.1-win-x64.msix"), plain)
    x64 = next(row for row in rows if row["label"] == "Windows x64 (MSIX)")
    assert x64["state"] == "built"
    assert x64["key"].endswith("HermesBundled-0.28.0-win-x64.msix")


def test_failed_legs_from_release_needs_seam():
    needs = json.dumps({
        "build-win32": {"result": "success"},
        "build-darwin": {"result": "failure"},
        "termux-deb": {"result": "skipped"},
        "build-linux": {"result": "failure"},
    })
    assert rbt.failed_legs_from_release_needs(needs) == ["build-darwin", "build-linux"]
    assert rbt.failed_legs_from_release_needs(None) == []
    assert rbt.failed_legs_from_release_needs("") == []
    assert rbt.failed_legs_from_release_needs("not json") == []


def test_summary_uses_exact_nested_keys_and_lists_the_sideload_bundle():
    from scripts.releases import handoff

    paths = {
        "windows-universal": [
            "HermesBundled-0.28.0.0-win.msixbundle",
        ],
        "termux": ["deb/pool build/hermes-agent_0.28.0_aarch64.deb"],
    }
    receipts = {
        leg: {"schema": 2, "commit": COMMIT, "name": leg,
              "files": [{"path": path, "size": 1, "sha256": "0" * 64} for path in files]}
        for leg, files in paths.items()
    }
    for leg, receipt in receipts.items():
        handoff.validate_commit_receipt(receipt, COMMIT, leg)
    names = _names(*(path for files in paths.values() for path in files))
    summary = rbt.render_commit_summary(names, BASE + "/downloads", COMMIT, receipts)
    from urllib.parse import quote

    for key in names:
        assert f"]({BASE}/downloads/{quote(key, safe='/')})" in summary
    built = [line for line in summary.splitlines() if "✅ Built" in line]
    assert len(built) == len(names)
    assert not any("Store" in line for line in built)
    assert "Linux x64" in summary and "Linux ARM64" in summary


@pytest.mark.parametrize("has_download", [True, False])
def test_incomplete_commit_rows_link_run_not_missing_downloads(has_download):
    run_url = "https://github.example/o/r/actions/runs/12345"
    receipts = _receipts_all_built() if has_download else {}
    names = _names("HermesBundled-0.28.0-win-x64.msix") if has_download else []
    # The uploaded x64 binary stays downloadable even when that job fails
    # after staging, or a sibling fails before staging its own artifact.
    failed = ["build-win32", "build-darwin", "publish-win32-updater", "termux-deb"]
    summary = rbt.render_commit_summary(names, BASE, COMMIT, receipts, failed, run_url=run_url)
    page = rbt.render_commit_page(COMMIT, names, BASE, receipts, failed, run_url=run_url)
    markdown_rows = [line for line in summary.splitlines() if line.startswith("| ")][1:]
    html_rows = re.findall(r"<tr><td>(.*?)</tr>", page)
    assert len(markdown_rows) == len(html_rows) == len(rbt._COMMIT_EXPECTED) + len(rbt._COMMIT_DISABLED)
    for md, html_row in zip(markdown_rows, html_rows):
        if "Linux" in md:
            assert "Disabled" in md and "Disabled" in html_row
            assert "](" not in md and "href=" not in html_row
        elif has_download and "Windows x64" in md:
            url = f"{BASE}/{names[0]}"
            assert "✅ Built" in md and "✅ Built" in html_row
            assert f"]({url})" in md and f'href="{url}"' in html_row
            assert run_url not in md and run_url not in html_row
        else:
            assert "❌ Not built" in md and "❌ Not built" in html_row
            assert f"[View build run]({run_url})" in md
            assert f'<a href="{run_url}">View build run</a>' in html_row
            assert BASE not in md and BASE not in html_row
    assert summary.count("✅ Built") == page.count("✅ Built") == int(has_download)


def test_commit_page_matches_the_summary_rows():
    """Two sinks, ONE row set: every link and every status in the step
    summary appears on the page, and neither sink drops a missing binary."""
    receipts = _receipts_all_built()
    names = _all_built_names()
    summary = rbt.render_commit_summary(names, BASE, COMMIT, receipts)
    page = rbt.render_commit_page(COMMIT, names, BASE, receipts)
    links = re.findall(r"\]\((https?://[^)]+)\)", summary)
    assert len(links) == len(rbt._COMMIT_EXPECTED)
    for url in links:
        assert f'href="{url}"' in page
    assert page.count("✅ Built") == summary.count("✅ Built") == len(rbt._COMMIT_EXPECTED)
    assert page.count("Disabled") == summary.count("Disabled") == len(rbt._COMMIT_DISABLED)
    assert "❌ Not built" not in page and "❌ Not built" not in summary
    assert rbt.recorded_build(page) == COMMIT


def test_commit_page_lists_a_missing_binary_without_a_link():
    receipts = _receipts_all_built()
    receipts["darwin-arm64"] = None
    names = _names("HermesBundled-0.28.0-win-x64.msix")
    page = rbt.render_commit_page(COMMIT, names, BASE, receipts, failed_legs=["build-darwin"])
    assert page.count("<tr>") == len(rbt._COMMIT_EXPECTED) + len(rbt._COMMIT_DISABLED) + 1
    assert "failed: build-darwin" in page
    # Only the one staged object was built, so it is the only anchor.
    assert page.count("<a href=") == 1


def test_commit_run_publishes_the_commit_page(tmp_path, monkeypatch, capsys):
    """The real CLI path: commit mode writes the summary AND the page."""
    summary = tmp_path / "summary.md"
    uploads: list[tuple[str, str, bool]] = []
    receipts = _receipts_all_built()
    monkeypatch.setattr(rbt.r2, "list_objects", lambda prefix="": {"keys": _all_built_names()})
    monkeypatch.setattr(rbt.handoff, "read_commit_receipt", lambda commit, name: receipts[name])
    monkeypatch.setattr(rbt.r2, "put", lambda tag, key, file, key_is_full=False, immutable=False:
                        uploads.append((key, Path(file).read_text(encoding="utf-8"), key_is_full)))
    monkeypatch.setattr(sys, "argv", ["render-builds-table.py", "--summary-commit", COMMIT,
                                      "--summary-out", str(summary), "--r2-base-url", BASE])
    assert rbt.main() == 0
    assert len(uploads) == 1
    key, page, key_is_full = uploads[0]
    assert key == f"releases/commit/{COMMIT}/index.html" and key_is_full
    assert f"✓ Page {BASE}/releases/commit/{COMMIT}/index.html" in capsys.readouterr().out
    written = summary.read_text(encoding="utf-8")
    for url in re.findall(r"\]\((https?://[^)]+)\)", written):
        assert f'href="{url}"' in page
    assert written.count("✅ Built") == page.count("✅ Built") == len(rbt._COMMIT_EXPECTED)
