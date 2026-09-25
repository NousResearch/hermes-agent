"""``checkout_contains`` on a stamped image (no ``.git``): ancestry collapses to stamp equality.

On a Docker/Cloud image ``git merge-base`` has nothing to walk, so the probe was always False and the
pending-restart catch-up printed "every gateway already serves the checkout" and "still off the checkout
code" in the same run. The stamp IS the checkout there. Identity comes from
``hermes_cli.version_info`` (install stamp first, live git second) — the retired ``build_info``
shim always answers "unknown", which left the stamped-image branch dead code.
"""

import pytest

from hermes_cli import update_cmd_fleet_checkout as chk
from hermes_cli import version_info

STAMPED_SOURCES = ["build", "commit-build", "ci", "docker", "fallback", "local", "nix"]


def _identity(monkeypatch, sha, source):
    monkeypatch.setattr(
        version_info, "get_code_identity",
        lambda refresh=False: {"sha": sha, "short_sha": (sha or "")[:8], "version": None, "source": source},
    )


@pytest.mark.parametrize("source", STAMPED_SOURCES)
def test_stamped_image_contains_exactly_its_stamp(monkeypatch, source):
    stamped = "b936546561aa0d2e6d0f7c3d1a9c5e8f2b4d6a70"
    _identity(monkeypatch, stamped, source)
    # no git call may be attempted on a stamped image: make one blow up if it is
    monkeypatch.setattr(chk.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("git probe on a stamped image")))
    assert chk.checkout_contains(stamped) is True
    assert chk.checkout_contains(stamped[:12]) is True  # short form recorded by an older writer
    assert chk.checkout_contains("0000000000aa0d2e6d0f7c3d1a9c5e8f2b4d6a70") is False


def test_unstamped_unknown_identity_stays_fail_closed(monkeypatch):
    """No stamp and no git: unknown ancestry is not evidence the fleet serves the update."""
    _identity(monkeypatch, None, "unknown")
    monkeypatch.setattr(chk.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("git probe without provenance")))
    assert chk.checkout_contains("b936546561aa0d2e6d0f7c3d1a9c5e8f2b4d6a70") is False


def test_git_checkout_still_walks_ancestry(monkeypatch):
    """Control: a source install keeps asking git, so a carried hotfix past the pulled SHA still counts."""
    _identity(monkeypatch, "deadbeef" * 5, "git")
    calls = []

    class _R:
        returncode = 0

    monkeypatch.setattr(chk.subprocess, "run", lambda cmd, **k: calls.append(cmd) or _R())
    assert chk.checkout_contains("cafebabe" * 5) is True
    assert calls and calls[0][:3] == ["git", "merge-base", "--is-ancestor"]
