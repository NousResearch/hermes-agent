"""End-to-end contracts for commit claims on the Stable Release pipeline.

Covers:
- Pre-PR legacy commit compatibility (``commit_supports_claim_tag``, ``stable.verify``,
  ``release_artifacts.record(..., legacy_commit_build=True)``, and ``validate_candidates``).
- Full commit-page assets (universal ``.msixbundle``, per-arch ``.msix``, macOS ``.dmg`` + ``.zip``,
  and Termux ``.deb``) plus commit link and baked ``bundleEnv`` in ``render-builds-table.py``.
- 90-day R2 retention for both ``releases/tag/commit.<stamp>-vX.Y.Z/`` and legacy
  ``releases/commit/<sha>/`` prefixes.
- ``scripts/release.py --build-commit`` bundles+smoke defaults, ``--with termux``, and refusal
  of shared-cache lanes.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from scripts.bundles import release_artifacts
from scripts.releases import r2, stable

ROOT = Path(__file__).resolve().parents[2]
COMMIT = "0123456" + "a" * 33
COMMIT_TAG = "commit.20261026T180000Z-v0.33.0"
PUBLIC_BASE = "https://hermes-assets.nousresearch.com"


def _load_render_module():
    spec = importlib.util.spec_from_file_location("render_builds_table_commit_test",
                                                  ROOT / "scripts/render-builds-table.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_msix(path: Path, *, arch: str, version: str, identity: str, stamp: dict) -> None:
    manifest = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<Package xmlns="http://schemas.microsoft.com/appx/manifest/foundation/windows10">'
        f'<Identity Name="{identity}" Publisher="CN=Nous Research" Version="{version}" '
        f'ProcessorArchitecture="{arch}" />'
        '<Applications><Application Id="HermesBundled" Executable="app\\Hermes.exe" /></Applications>'
        '</Package>'
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("AppxManifest.xml", manifest)
        archive.writestr("app/resources/install-stamp.json", json.dumps(stamp))


def test_verify_detects_pre_pr_commit_and_emits_legacy_flag(tmp_path, monkeypatch):
    claim = {
        "claim_tag": COMMIT_TAG, "claim_object": "b" * 40, "claim_epoch": 1793037600,
        "policy": "commit", "lanes": ("macos-arm64", "smoke", "windows"),
        "bundle_env": {"HERMES_PREVIEW_FULL_CONNECTORS": "1"},
        "require_green": True, "autopublish": False,
    }
    monkeypatch.setattr(stable, "stable_context", lambda _env: (COMMIT_TAG, COMMIT, claim))

    def old_git(argv):
        assert argv[:2] == ["git", "show"]
        if argv[2].endswith("update_channel.py"):
            return "def release_version(tag): ...\n"
        if argv[2].endswith("msix-shared.mjs"):
            return "export const CANARY_TAG_RE = /canary/\n"
        return "CHANNEL = 'stable'\n"

    out_old = tmp_path / "old-output"
    stable.verify({
        "GITHUB_OUTPUT": str(out_old),
        "BUNDLE_ENV_JSON": json.dumps(claim["bundle_env"]),
    }, run=old_git)
    parsed_old = dict(line.split("=", 1) for line in out_old.read_text(encoding="utf-8").splitlines())
    assert parsed_old["legacy-commit-build"] == "true"
    assert parsed_old["core-version"] == "0.33.0"
    assert parsed_old["channel"] == "commit"

    def new_git(argv):
        if argv[2].endswith("update_channel.py"):
            return "def is_commit_tag(tag): return True\n"
        if argv[2].endswith("msix-shared.mjs"):
            return "export const COMMIT_TAG_RE = /commit/\n"
        return 'return "commit"\n'

    out_new = tmp_path / "new-output"
    stable.verify({
        "GITHUB_OUTPUT": str(out_new),
        "BUNDLE_ENV_JSON": json.dumps(claim["bundle_env"]),
    }, run=new_git)
    parsed_new = dict(line.split("=", 1) for line in out_new.read_text(encoding="utf-8").splitlines())
    assert parsed_new["legacy-commit-build"] == "false"
    assert parsed_new["core-version"] == "0.33.0"

    # Mismatched bundle defaults are refused before any build runs.
    with pytest.raises(ValueError, match="Bundle environment differs"):
        stable.verify({"GITHUB_OUTPUT": str(tmp_path / "bad"), "BUNDLE_ENV_JSON": "{}"}, run=new_git)


def test_legacy_commit_record_and_candidate_validation_round_trip(tmp_path):
    stamp = {
        "source": "commit-build", "commit": COMMIT, "tag": None,
        "baseVersion": "0.33.0", "updateMechanism": "external", "branch": None,
    }
    msix = tmp_path / "HermesBundled-0.33.0-win-x64.msix"
    _write_msix(msix, arch="x64", version="0.33.0.0",
                identity=f"NousResearch.HermesBundledCommit{COMMIT[:7]}", stamp=stamp)
    (tmp_path / "version-info-x64.json").write_text(json.dumps({"productVersion": "0.33.0"}), encoding="utf-8")
    out = tmp_path / "metadata-windows-x64.json"

    release_artifacts.record("windows", "x64", tmp_path, COMMIT_TAG, COMMIT, out,
                             legacy_commit_build=True)
    row = json.loads(out.read_text(encoding="utf-8"))
    assert row["legacyCommitBuild"] is True
    assert row["version"] == "0.33.0.0"
    assert row["executableVersion"] == "0.33.0"
    assert row["identity"] == f"NousResearch.HermesBundledCommit{COMMIT[:7]}"

    # Stable or canary tags may never use legacy_commit_build.
    with pytest.raises(ValueError, match="Legacy commit build"):
        release_artifacts.record("windows", "x64", tmp_path, "v0.33.0", COMMIT, out,
                                 legacy_commit_build=True)

    manifest = {
        "schema": 2, "tag": COMMIT_TAG, "commit": COMMIT,
        "releaseEpoch": 1793037600, "archive": COMMIT_TAG,
        "packages": [
            {
                **row,
                "artifact": {
                    "url": f"{PUBLIC_BASE}/releases/tag/{COMMIT_TAG}/HermesBundled-0.33.0.0-win.msixbundle",
                    "sha256": "c" * 64,
                },
            },
            {
                "platform": "macos", "arch": "arm64", "tag": COMMIT_TAG, "commit": COMMIT,
                "identity": f"com.nousresearch.hermes.bundled.commit.{COMMIT[:7]}",
                "teamId": "ABCDE12345", "version": "0.33.0", "legacyCommitBuild": True,
                "artifact": {
                    "url": f"{PUBLIC_BASE}/releases/tag/{COMMIT_TAG}/HermesBundled-0.33.0-mac-arm64.zip",
                    "sha256": "d" * 64,
                },
            },
            {
                "platform": "termux", "arch": "aarch64", "tag": COMMIT_TAG, "commit": COMMIT,
                "identity": "hermes-agent", "version": "0.33.0-1", "legacyCommitBuild": True,
                "artifact": {
                    "url": f"{PUBLIC_BASE}/releases/tag/{COMMIT_TAG}/deb/hermes-agent_0.33.0-1_aarch64.deb",
                    "sha256": "e" * 64,
                },
            },
        ],
    }
    validated = stable._validated_rows(manifest, COMMIT_TAG, COMMIT, PUBLIC_BASE, 1793037600,
                                       archive=COMMIT_TAG)
    assert set(validated) == {"windows/x64", "macos/arm64", "termux/aarch64"}

    # Stable releases reject legacyCommitBuild rows outright.
    stable_manifest = {
        **manifest, "tag": "v0.33.0", "archive": "rc.1-v0.33.0",
        "packages": [{
            **manifest["packages"][0], "tag": "v0.33.0",
            "artifact": {
                "url": f"{PUBLIC_BASE}/releases/tag/rc.1-v0.33.0/HermesBundled-0.33.0.0-win.msixbundle",
                "sha256": "c" * 64,
            },
        }],
    }
    with pytest.raises(ValueError, match="Legacy commit build"):
        stable._validated_rows(stable_manifest, "v0.33.0", COMMIT, PUBLIC_BASE, 1793037600,
                               archive="rc.1-v0.33.0")


def test_commit_claim_page_lists_all_downloadable_assets_commit_link_and_bundle_env():
    render = _load_render_module()
    files = [
        f"releases/tag/{COMMIT_TAG}/HermesBundled-26.1026.18.0000-win.msixbundle",
        f"releases/tag/{COMMIT_TAG}/HermesBundled-0.33.0+commit.20261026T180000Z-win-x64.msix",
        f"releases/tag/{COMMIT_TAG}/HermesBundled-0.33.0+commit.20261026T180000Z-win-arm64.msix",
        f"releases/tag/{COMMIT_TAG}/HermesBundled-0.33.0+commit.20261026T180000Z-mac-arm64.dmg",
        f"releases/tag/{COMMIT_TAG}/HermesBundled-0.33.0+commit.20261026T180000Z-mac-arm64.zip",
        f"releases/tag/{COMMIT_TAG}/HermesBundled-0.33.0+commit.20261026T180000Z-mac-arm64.zip.blockmap",
        f"releases/tag/{COMMIT_TAG}/deb/hermes-agent_0.33.0~commit.20261026T180000Z-1_aarch64.deb",
    ]
    bundle_env = {"HERMES_PREVIEW_FULL_CONNECTORS": "1", "HERMES_HOME": None}
    page = render.render_page(
        COMMIT_TAG, {}, PUBLIC_BASE, repo="NousResearch/hermes-agent",
        commit=COMMIT, bundle_env=bundle_env, commit_files=files,
    )
    assert f'href="https://github.com/NousResearch/hermes-agent/commit/{COMMIT}"' in page
    assert "Bundle environment" in page and "HERMES_PREVIEW_FULL_CONNECTORS" in page and "Unset" in page
    for expected in (".msixbundle", "-win-x64.msix", "-win-arm64.msix",
                     "-mac-arm64.dmg", "-mac-arm64.zip", ".deb"):
        assert expected in page, expected
    assert ".blockmap" not in page

    summary = render.render_commit_claim_tables(files, PUBLIC_BASE)
    for expected in (".msixbundle", "-win-x64.msix", "-win-arm64.msix",
                     "-mac-arm64.dmg", "-mac-arm64.zip", ".deb"):
        assert expected in summary, expected
    assert ".blockmap" not in summary

    # Missing source commit is refused for a commit claim page.
    with pytest.raises(ValueError, match="exact source commit"):
        render.render_page(COMMIT_TAG, {}, PUBLIC_BASE, commit=None, commit_files=files)


def test_commit_r2_retention_ages_out_claim_archives_and_legacy_commit_prefixes():
    now = datetime(2026, 11, 21, 0, 0, tzinfo=UTC)
    cutoff_dt = now - timedelta(days=r2.COMMIT_KEEP_DAYS)
    cutoff = cutoff_dt.strftime("%Y%m%d")
    cutoff_ms = cutoff_dt.timestamp() * 1000
    old_claim = "releases/tag/commit.20260701T000000Z-v0.30.0/HermesBundled-0.30.0-mac-arm64.dmg"
    fresh_claim = "releases/tag/commit.20261101T000000Z-v0.33.0/HermesBundled-0.33.0-mac-arm64.dmg"
    stable_tag = "releases/tag/v0.30.0/HermesBundled-0.30.0-mac-arm64.dmg"
    old_legacy = f"releases/commit/{COMMIT}/HermesBundled-0.30.0-mac-arm64.dmg"
    fresh_legacy = f"releases/commit/{'b' * 40}/HermesBundled-0.33.0-mac-arm64.dmg"

    keys = [old_claim, fresh_claim, stable_tag, old_legacy, fresh_legacy]
    last_modified = {
        old_claim: datetime(2026, 11, 20, tzinfo=UTC).timestamp() * 1000,
        fresh_claim: datetime(2026, 11, 1, tzinfo=UTC).timestamp() * 1000,
        stable_tag: datetime(2025, 1, 1, tzinfo=UTC).timestamp() * 1000,
        old_legacy: datetime(2026, 7, 1, tzinfo=UTC).timestamp() * 1000,
        fresh_legacy: datetime(2026, 11, 1, tzinfo=UTC).timestamp() * 1000,
    }
    doomed = set(r2.commit_doomed_keys(keys, cutoff, last_modified, cutoff_ms))
    assert doomed == {old_claim, old_legacy}


def test_release_cli_build_commit_allows_termux_and_refuses_shared_cache_lanes():
    # Direct CLI check: --with ci is rejected, --build-commit help advertises bundles + smoke only.
    help_run = subprocess.run([sys.executable, str(ROOT / "scripts/release.py"), "--help"],
                              capture_output=True, text=True, check=True)
    assert "bundles + smoke only" in help_run.stdout
    bad = subprocess.run([sys.executable, str(ROOT / "scripts/release.py"),
                          "--build-commit", "HEAD", "--with", "ci"],
                         capture_output=True, text=True, check=False)
    assert bad.returncode != 0
    assert "commit" in (bad.stderr + bad.stdout).lower()
