"""Tests for hermes_cli/platforms.py — platform registry invariants."""


def test_platforms_is_ordered():
    from hermes_cli.platforms import PLATFORMS
    from collections import OrderedDict
    assert isinstance(PLATFORMS, OrderedDict)


def test_every_platform_has_label_and_toolset():
    from hermes_cli.platforms import PLATFORMS, PlatformInfo
    for key, info in PLATFORMS.items():
        assert isinstance(key, str) and key
        assert isinstance(info, PlatformInfo)
        assert info.label
        assert info.default_toolset


def test_core_platforms_present():
    from hermes_cli.platforms import PLATFORMS
    core = {"telegram", "discord", "slack", "email", "cli"}
    assert core.issubset(set(PLATFORMS.keys()))


def test_no_duplicate_keys():
    from hermes_cli.platforms import PLATFORMS
    assert len(PLATFORMS) == len(set(PLATFORMS.keys()))
