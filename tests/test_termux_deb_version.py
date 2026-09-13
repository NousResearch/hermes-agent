"""Unit tests for scripts/termux/deb_version.py (Task 4 of the termux-deb plan)."""

import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parent / "scripts" / "termux" / "deb_version.py"

from scripts.termux.deb_version import channel_for_tag, deb_version_for_tag  # noqa: E402


def test_canary_tag_shape_matches_canonical():
    """Invariant: the deb versioner accepts EXACTLY the canary tags the
    canonical release tooling mints. The canonical shape lives in
    hermes_cli/update_channel.py:_CANARY_TAG_RE (8-or-14-digit, 20-prefixed
    timestamps); scripts/releases/r2.py:channel_for_tag parses the same shape.
    A tag this module accepts but the release flow would never mint (or vice
    versa) is version-drift between the .deb channel and the feed channel.
    """
    from hermes_cli.update_channel import _CANARY_TAG_RE as _NIGHTLY_TAG_RE
    from scripts.termux import deb_version as dv

    samples = [
        "v0.20.6-canary.20260831120000",  # canonical canary (14-digit)
        "v0.20.6-canary.20260831",        # canonical canary (8-digit)
        "v1.2.3",                          # stable
    ]
    for tag in samples:
        assert dv._TAG_RE.match(tag), f"deb versioner rejects canonical tag {tag}"

    never_minted = [
        "v1.2.3-canary.202608311",   # 9 digits -- canonical rejects
        "v1.2.3-canary.12345678",    # non-20 prefix -- canonical rejects
        "v1.2.3-canary.202608311200001",  # 15 digits -- canonical rejects
    ]
    for tag in never_minted:
        assert not _NIGHTLY_TAG_RE.match(tag), f"sample is actually canonical: {tag}"
        assert not dv._TAG_RE.match(tag), f"deb versioner accepts never-minted tag {tag}"


def test_stable_tag_maps_to_revision_1():
    assert deb_version_for_tag("v1.2.3") == "1.2.3-1"


def _dpkg_key(v: str) -> str:
    # Approximate dpkg ordering for these versions: '~' sorts before everything
    # (even the empty string / '-'), so map it low.
    return v.replace("~", "\x00")


def test_stable_tag_multi_digit():
    assert deb_version_for_tag("v26.8.31") == "26.8.31-1"


def test_major_can_be_three_digits():
    assert deb_version_for_tag("v126.8.31") == "126.8.31-1"


def test_canary_tag_ranks_below_stable():
    got = deb_version_for_tag("v1.2.3-canary.20260831120000")
    assert got == "1.2.3~canary.20260831120000-1"
    assert _dpkg_key(got) < _dpkg_key(deb_version_for_tag("v1.2.3"))  # dpkg ordering


def test_canary_canary_ranking_among_nightlies():
    earlier = deb_version_for_tag("v1.2.3-canary.20260831000000")
    later = deb_version_for_tag("v1.2.3-canary.20260831235959")
    assert _dpkg_key(earlier) < _dpkg_key(later) < _dpkg_key(deb_version_for_tag("v1.2.3"))


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "1.2.3",                      # missing v prefix
        "v1.2",                       # not three components
        "v1.2.3.4",                   # four components
        "v1.2.3-",                    # empty suffix
        "v1.2.3-canary",             # canary without timestamp
        "v1.2.3-canary.abc",         # non-numeric timestamp
        "v1.2.3-beta.1",              # unknown suffix channel
        "v1.2.x",
        "v-1.2.3",
    ],
)
def test_malformed_tags_raise(bad):
    with pytest.raises(ValueError):
        deb_version_for_tag(bad)


@pytest.mark.parametrize("bad", ["v1234.1.2", "v99999.0.0"])
def test_major_above_three_digits_rejected(bad):
    with pytest.raises(ValueError):
        deb_version_for_tag(bad)


def test_minor_patch_can_be_three_digits():
    # Cap applies to major only; minor/patch may be wide.
    assert deb_version_for_tag("v1.234.567") == "1.234.567-1"


def test_cli_invocation(capsys):
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "v9.8.7"], capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "9.8.7-1"


def test_channel_matches_canary_shape():
    """--channel derives from the SAME _TAG_RE as the deb version: any tag
    that yields a '~canary' version is canary, everything else stable."""
    assert channel_for_tag("v1.2.3") == "stable"
    assert channel_for_tag("v26.8.31") == "stable"
    assert channel_for_tag("v0.20.6-canary.20260831120000") == "canary"
    assert channel_for_tag("v0.20.6-canary.20260831") == "canary"


def test_channel_agrees_with_deb_version():
    for tag in ("v1.2.3", "v126.8.31", "v1.2.3-canary.20260831120000"):
        assert ("~canary" in deb_version_for_tag(tag)) == (channel_for_tag(tag) == "canary")


def test_channel_malformed_tag_raises():
    with pytest.raises(ValueError):
        channel_for_tag("v1.2")


def test_channel_cli_invocation():
    for tag, expected in [("v9.8.7", "stable"), ("v9.8.7-canary.20260831120000", "canary")]:
        r = subprocess.run(
            [sys.executable, str(SCRIPT), "--channel", tag], capture_output=True, text=True
        )
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == expected
