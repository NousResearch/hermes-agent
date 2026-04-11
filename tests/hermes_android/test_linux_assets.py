from hermes_android.linux_assets import (
    ANDROID_TO_TERMUX_ARCH,
    IGNORED_DEPENDENCIES,
    ROOT_PACKAGES,
    parse_depends,
    parse_packages_index,
    resolve_dependency_closure,
    strip_termux_prefix,
)


def test_android_linux_assets_manifest_exposes_supported_arches_and_core_packages():
    assert ANDROID_TO_TERMUX_ARCH == {"arm64-v8a": "aarch64", "x86_64": "x86_64"}
    assert "bash" in ROOT_PACKAGES
    assert "coreutils" in ROOT_PACKAGES
    assert "git" in ROOT_PACKAGES
    assert "termux-tools" in IGNORED_DEPENDENCIES


def test_parse_depends_skips_ignored_termux_companion_packages():
    raw = "libandroid-support, termux-tools, openssl (>= 3.0), foo | bar"
    assert parse_depends(raw) == ("libandroid-support", "openssl", "foo")


def test_parse_packages_index_and_dependency_resolution():
    sample = """
Package: bash
Version: 5.3
Filename: pool/main/b/bash/bash_5.3_aarch64.deb
SHA256: deadbeef
Depends: libandroid-support, termux-tools

Package: libandroid-support
Version: 28
Filename: pool/main/liba/libandroid-support/libandroid-support_28_aarch64.deb
SHA256: feedface

"""
    records = parse_packages_index(sample)
    resolved = resolve_dependency_closure(records, root_packages=["bash"])
    assert [record.name for record in resolved] == ["bash", "libandroid-support"]


def test_strip_termux_prefix_only_accepts_usr_members():
    assert strip_termux_prefix("./data/data/com.termux/files/usr/bin/bash") == "bin/bash"
    assert strip_termux_prefix("data/data/com.termux/files/usr/lib/libreadline.so.8") == "lib/libreadline.so.8"
    assert strip_termux_prefix("./etc/passwd") is None
