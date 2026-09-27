"""Literal tag boundaries and real Debian ordering (not a second parser)."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from scripts.termux.deb_version import channel_for_tag, deb_version_for_tag

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/termux/deb_version.py"


@pytest.mark.parametrize("bom", ["", "\ufeff"])
def test_apt_readers_accept_bom_and_keep_archive_bytes(tmp_path, bom):
    from scripts.termux import stage_apt_repo as apt
    from tests.termux_fixtures import build_deb

    control = {bom + "Package": "example", "Version": "1.2.3-1", "Architecture": "aarch64",
               "Description": "café 東京"}
    package = tmp_path / "example.deb"
    build_deb(package, control)
    original = package.read_bytes()
    fields, raw = apt.deb_control_fields_and_bytes(package)
    assert fields == {key.removeprefix("\ufeff"): value for key, value in control.items()}
    assert raw == original == package.read_bytes()
    index = tmp_path / "dists/stable" / apt.COMPONENT / f"binary-{apt.ARCH}" / "Packages"
    index.parent.mkdir(parents=True)
    index.write_text(bom + "Package: example\nVersion: 1.2.3-1\n\nPackage: incomplete\n", encoding="utf-8")
    assert apt.existing_published(tmp_path, "stable") == {("example", "1.2.3-1")}


@pytest.mark.parametrize("tag,version,channel", [
    ("v1.2.3", "1.2.3-1", "stable"),
    ("v26.8.31", "26.8.31-1", "stable"),
    ("v126.8.31", "126.8.31-1", "stable"),
    ("v1.234.567", "1.234.567-1", "stable"),
    ("v0.20.6+canary.20260831T120000Z", "0.20.6~canary.20260831T120000Z-1", "canary"),
])
def test_tag_mapping(tag, version, channel):
    assert deb_version_for_tag(tag) == version
    assert channel_for_tag(tag) == channel


def test_canary_shape_matches_the_stable_shape_on_every_component():
    # scripts/releases/semver.py::is_valid_version gates release handoffs;
    # a canary cut over the newest stable must never be rejected when the
    # stable tag itself is accepted.
    from hermes_cli.update_channel import _CANARY_TAG_RE
    from scripts.releases.semver import is_release_version

    assert _CANARY_TAG_RE.fullmatch("v1.9.15+canary.20260916T120000Z")
    assert is_release_version("1.9.15+canary.20260916T120000Z")
    assert not is_release_version("2026.9.15+canary.20260916T120000Z")
    assert not is_release_version("1.9.15-canary.20260916120000")


@pytest.mark.parametrize("tag", [
    "", "1.2.3", "v1.2", "v1.2.3.4", "v1.2.3-", "v1.2.3-canary", "v1.2.3-canary.abc",
    "v1.2.3-beta.1", "v1.2.x", "v-1.2.3", "v2026.9.15",
    "v1.2.3-canary.202608311", "v1.2.3-canary.12345678", "v1.2.3-canary.202608311200001",
])
def test_malformed_tags_rejected_by_both_mappings(tag):
    for mapping in (deb_version_for_tag, channel_for_tag):
        with pytest.raises(ValueError):
            mapping(tag)


@pytest.mark.parametrize("args,status,output", [
    (["v9.8.7"], 0, "9.8.7-1"),
    (["--channel", "v9.8.7"], 0, "stable"),
    (["--channel", "v9.8.7+canary.20260831T120000Z"], 0, "canary"),

    (["v1.2"], 1, ""), (["--channel", "v1.2"], 1, ""),
])
def test_cli_dispatch(args, status, output):
    result = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)
    assert result.returncode == status, result.stderr
    assert result.stdout.strip() == output
    if status:
        assert result.stderr


@pytest.mark.skipif(shutil.which("dpkg") is None, reason="requires native dpkg")
def test_dpkg_orders_canary_and_numeric_versions():
    tags = ["v1.2.3+canary.20260831T000000Z", "v1.2.3+canary.20260831T235959Z", "v1.2.3", "v1.2.10"]
    versions = list(map(deb_version_for_tag, tags))
    for earlier, later in zip(versions, versions[1:]):
        subprocess.run(["dpkg", "--compare-versions", earlier, "lt", later], check=True)
