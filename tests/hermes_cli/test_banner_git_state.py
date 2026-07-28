from unittest.mock import MagicMock, patch

from hermes_cli.version_info import VersionInfo


def test_format_banner_version_label_without_git_state():
    from hermes_cli import banner

    with patch.object(
        banner,
        "get_version_info",
        return_value=VersionInfo(banner.VERSION, banner.VERSION, None, None, None, "unknown"),
    ):
        value = banner.format_banner_version_label()

    assert value == f"Hermes Agent v{banner.VERSION}"


def test_format_banner_version_label_includes_derived_version_and_provenance():
    from hermes_cli import banner

    with patch.object(
        banner,
        "get_version_info",
        return_value=VersionInfo("0.19.0", "0.19.0+3", 3, "b" * 40, "feature/version", "git"),
    ):
        value = banner.format_banner_version_label()

    assert "v0.19.0+3" in value
    assert "feature/version" in value
    assert "b" * 12 in value


def test_format_banner_version_label_omits_zero_suffix():
    from hermes_cli import banner

    with patch.object(
        banner,
        "get_version_info",
        return_value=VersionInfo("0.19.0", "0.19.0", 0, "a" * 40, "main", "git"),
    ):
        value = banner.format_banner_version_label()

    assert "v0.19.0" in value
    assert "+0" not in value
    assert "carried" not in value