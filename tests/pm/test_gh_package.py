"""GitHub CLI release URLs cover every supported Linux target."""

from pm import get_package


def test_gh_bionic_arm_uses_generic_linux_archive():
    package = get_package("gh")

    assert package.fetch_url("2.102.0", "linux-arm64-bionic") == package.fetch_url(
        "2.102.0", "linux-arm64"
    )
