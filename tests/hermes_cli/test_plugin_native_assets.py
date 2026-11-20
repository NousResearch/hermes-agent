"""Review metadata and match a native release to the backend's host facts."""

import pytest

from hermes_cli.plugin_catalog import entry_from_mapping
from hermes_cli.plugin_native_assets import select_native_asset


@pytest.mark.parametrize(
    "family,arch,version,expected",
    [
        ("linux", "amd64", "2.35", "linux-amd64"),
        ("linux", "arm64", "2.36", "linux-arm64"),
        ("darwin", "amd64", "15.0", "macos-amd64"),
        ("darwin", "arm64", "15.0.1", "macos-arm64"),
        ("win32", "amd64", "", None),
        ("linux", "x86", "2.35", "No reviewed native host"),
        ("linux", "arm64", "", "requires glibc 2.35"),
        ("linux", "amd64", "2.34", "requires glibc 2.35"),
        ("darwin", "arm64", "14.0", "requires macOS 15.0"),
    ],
)
def test_catalog_native_asset_selection(family, arch, version, expected):
    declaration = {
        "tag": "v0.3.0",
        "member": "package/host/native",
        "destination": "host/native",
        "targets": {
            "linux-amd64": {"filename": "linux-x64.tar.gz", "sha256": "1" * 64, "min_glibc": "2.35"},
            "linux-arm64": {"filename": "linux-arm64.tar.gz", "sha256": "2" * 64, "min_glibc": "2.35"},
            "macos-amd64": {"filename": "macos-intel.zip", "sha256": "3" * 64, "min_macos": "15.0"},
            "macos-arm64": {"filename": "macos-arm64.zip", "sha256": "4" * 64, "min_macos": "15.0"},
        },
    }
    entry = entry_from_mapping({
        "name": "plugin", "repo": "https://github.com/example/plugin", "sha": "a" * 40,
        "platforms": ["linux", "macos", "windows"], "native_assets": declaration,
    }, "live catalog")
    assert entry is not None
    assert entry.to_dict()["native_assets"] == declaration
    if expected is None:
        assert select_native_asset(entry.native_assets, family, arch, version) is None
    elif expected in declaration["targets"]:
        key, asset = select_native_asset(entry.native_assets, family, arch, version)
        assert key == expected
        assert asset.to_dict() == declaration["targets"][expected]
    else:
        with pytest.raises(ValueError, match=expected):
            select_native_asset(entry.native_assets, family, arch, version)
