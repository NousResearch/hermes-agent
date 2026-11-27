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

from hermes_platform.resolver import locate_command
from scripts.bundles import release_artifacts
from scripts.releases import lanes, r2, stable

ROOT = Path(__file__).resolve().parents[2]
COMMIT = "0123456" + "a" * 33
COMMIT_TAG = "commit.20261026T180000Z-v0.33.0"
# What a pre-claim tree's tagless --commit Termux build stamps (build_deb.sh).
LEGACY_DEB_VERSION = f"0.33.0+commit{COMMIT[:12]}"
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


def _tree(files: dict[str, str]):
    """A fake ``git`` over one claimed commit's tree: ls-tree lists, show prints, else it fails."""
    def git(argv):
        if argv[:3] == ["git", "ls-tree", "--name-only"]:
            return argv[-1] if argv[-1] in files else ""
        if argv[:2] == ["git", "show"] and argv[2].split(":", 1)[1] in files:
            return files[argv[2].split(":", 1)[1]]
        raise subprocess.CalledProcessError(128, argv, stderr="fatal: bad object")
    return git


# The claim readers' source text as the PR's first commit (bd507fa031) had it: before its
# packaging could build a claim, so it must still build through the tagless path.
READERS_WITHOUT_PACKAGING = {
    "hermes_cli/update_channel.py": "def is_commit_tag(tag): return True\n",
    "scripts/msix-shared.mjs": "export const COMMIT_TAG_RE = /commit/\n",
    "scripts/termux/deb_version.py": 'r"^(?:(?P<kind>canary|commit)\\."\n',
}


def test_verify_detects_pre_pr_commit_and_emits_legacy_flag(tmp_path, monkeypatch):
    claim = {
        "claim_tag": COMMIT_TAG, "claim_object": "b" * 40, "claim_epoch": 1793037600,
        "policy": "commit", "lanes": ("macos-arm64", "smoke", "windows"),
        "bundle_env": {"HERMES_PREVIEW_FULL_CONNECTORS": "1"},
        "require_green": True, "autopublish": False,
    }
    monkeypatch.setattr(stable, "stable_context", lambda _env: (COMMIT_TAG, COMMIT, claim))

    def verified(name, tree):
        out = tmp_path / name
        stable.verify({"GITHUB_OUTPUT": str(out), "BUNDLE_ENV_JSON": json.dumps(claim["bundle_env"])},
                      run=_tree(tree))
        return dict(line.split("=", 1) for line in out.read_text(encoding="utf-8").splitlines())

    parsed_old = verified("old-output", {"hermes_cli/update_channel.py": "def release_version(tag): ...\n"})
    assert parsed_old["legacy-commit-build"] == "true"
    assert parsed_old["core-version"] == "0.33.0"
    assert parsed_old["channel"] == "commit"
    # Claim readers without the declaration are not a claim build.
    assert verified("readers-output", READERS_WITHOUT_PACKAGING)["legacy-commit-build"] == "true"

    new_tree = {stable.CLAIM_BUILD_FILE: "# the contract\n1\n"}
    parsed_new = verified("new-output", new_tree)
    assert parsed_new["legacy-commit-build"] == "false"
    assert parsed_new["core-version"] == "0.33.0"

    # Mismatched bundle defaults are refused before any build runs.
    with pytest.raises(ValueError, match="Bundle environment differs"):
        stable.verify({"GITHUB_OUTPUT": str(tmp_path / "bad"), "BUNDLE_ENV_JSON": "{}"}, run=_tree(new_tree))


def test_claim_support_refuses_a_failed_read_or_an_unknown_contract():
    """Neither a git failure nor a contract this controller can't build passes for an old commit."""
    def unreadable(argv):
        raise subprocess.CalledProcessError(128, argv, stderr="fatal: bad object")

    with pytest.raises(subprocess.CalledProcessError):
        stable.commit_supports_claim_tag(COMMIT, run=unreadable)
    with pytest.raises(ValueError, match="declares commit-claim build"):
        stable.commit_supports_claim_tag(COMMIT, run=_tree({stable.CLAIM_BUILD_FILE: "2\n"}))


@pytest.mark.platforms("linux")
@pytest.mark.skipif(not locate_command("dpkg-deb").found, reason="builds a real .deb")
def test_legacy_termux_record_accepts_the_tagless_commit_version(tmp_path):
    package = tmp_path / "package"
    stamp = package / "data/data/com.termux/files/usr/opt/hermes-agent/install-stamp.json"
    stamp.parent.mkdir(parents=True)
    stamp.write_text(json.dumps({"source": "commit-build", "commit": COMMIT, "tag": None}), encoding="utf-8")
    (package / "DEBIAN").mkdir()
    (package / "DEBIAN/control").write_text(
        f"Package: hermes-agent\nVersion: {LEGACY_DEB_VERSION}\nArchitecture: aarch64\n"
        "Maintainer: Fixture <fixture@example.invalid>\nDescription: fixture\n", encoding="utf-8")
    root = tmp_path / "termux-build"
    (root / "deb").mkdir(parents=True)
    subprocess.run(["dpkg-deb", "--root-owner-group", "--build", str(package), str(root / "deb/hermes-agent.deb")],
                   check=True, capture_output=True, timeout=60)
    out = root / "metadata-termux-aarch64.json"

    release_artifacts.record("termux", "aarch64", root, COMMIT_TAG, COMMIT, out, legacy_commit_build=True)
    row = json.loads(out.read_text(encoding="utf-8"))
    assert row["version"] == LEGACY_DEB_VERSION and row["legacyCommitBuild"] is True


@pytest.mark.parametrize("selection", [{"only": ["smoke"]}, {"without": ["windows", "macos-arm64"]}])
def test_a_commit_claim_that_builds_no_bundle_is_refused_before_and_at_admission(selection):
    with pytest.raises(ValueError, match="at least one bundle"):
        lanes.select(policy="commit", **selection)
    message = json.dumps({
        "schema": 1, "kind": "commit", "version": "0.33.0", "commit": COMMIT, "lanes": ["smoke"],
        "lanesRevision": lanes.LANES_REVISION, "requireGreen": True, "bundleEnv": {},
    })
    with pytest.raises(ValueError, match="at least one bundle"):
        stable.stamped_claim(message, tag=COMMIT_TAG, commit=COMMIT)
    assert stable.stamped_claim(message.replace('["smoke"]', '["smoke", "termux"]'),
                                tag=COMMIT_TAG, commit=COMMIT)["lanes"] == {"smoke", "termux"}


def test_a_commit_of_this_tree_builds_as_a_claim():
    """Only trees from before commit claims take the legacy path; this one must not."""
    def git(argv):
        return subprocess.check_output(argv, cwd=ROOT, text=True, encoding="utf-8")

    assert stable.commit_supports_claim_tag("HEAD", run=git)


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
                "identity": "hermes-agent", "version": LEGACY_DEB_VERSION, "legacyCommitBuild": True,
                "artifact": {
                    "url": f"{PUBLIC_BASE}/releases/tag/{COMMIT_TAG}/deb/hermes-agent_{LEGACY_DEB_VERSION}_aarch64.deb",
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


def test_a_commit_claim_never_publishes_or_promotes_a_release_channel(tmp_path):
    """publish and promote key the APT prefix and the feeds on release_channel: a commit claim's
    manifest is refused there instead of reading as stable."""
    manifest = {"tag": COMMIT_TAG, "packages": [], "files": []}
    for step in (release_artifacts.publish, release_artifacts.promote):
        with pytest.raises(ValueError, match="Commit builds publish to no release channel"):
            step(manifest, tmp_path, PUBLIC_BASE)


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
