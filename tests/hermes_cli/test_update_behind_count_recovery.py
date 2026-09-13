"""Behind-count recovery via the GitHub compare API (banner.py).

The class of bug: any code path that knows two tip SHAs but has no local
history to count across (shallow installer clones, ls-remote-only probes)
used to fabricate a count of ``1`` — the UI then rendered "+1" / "1 commit
behind" forever while the real distance grew (#84591: 61 commits behind,
indicator said 1). The fix has two halves:

1. Honesty: never fabricate a number. Uncountable = UPDATE_AVAILABLE_NO_COUNT
   sentinel (CLI) / null (desktop), rendered as a generic "update available".
2. Accuracy: recover the exact count via GitHub's compare API, which knows
   the full graph regardless of local clone depth.
"""

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import hermes_cli.banner as banner

SHA_A = "a" * 40
SHA_B = "b" * 40


def _compare_payload(ahead):
    return io.BytesIO(json.dumps({"ahead_by": ahead, "status": "ahead"}).encode())


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _patch_urlopen(payload):
    banner._compare_payload_cache.clear()
    return patch(
        "urllib.request.urlopen",
        return_value=_FakeResponse(json.dumps(payload).encode()),
    )


@pytest.fixture(autouse=True)
def _fresh_compare_cache():
    banner._compare_payload_cache.clear()
    yield
    banner._compare_payload_cache.clear()


# ---------------------------------------------------------------------------
# _github_compare_behind
# ---------------------------------------------------------------------------


def test_compare_behind_returns_ahead_by():
    with _patch_urlopen({"ahead_by": 61, "status": "ahead"}):
        assert banner._github_compare_behind(SHA_A, SHA_B) == 61


def test_compare_behind_zero_means_local_ahead():
    with _patch_urlopen({"ahead_by": 0, "status": "behind"}):
        assert banner._github_compare_behind(SHA_A, SHA_B) == 0


def test_compare_behind_rejects_short_shas_without_network():
    with patch("urllib.request.urlopen") as mock_open:
        assert banner._github_compare_behind("abc123", SHA_B) is None
        assert banner._github_compare_behind(SHA_A, "") is None
        assert banner._github_compare_behind(None, SHA_B) is None
    mock_open.assert_not_called()


def test_compare_behind_network_failure_returns_none():
    with patch("urllib.request.urlopen", side_effect=OSError("offline")):
        assert banner._github_compare_behind(SHA_A, SHA_B) is None


@pytest.mark.parametrize(
    "payload",
    [
        {"status": "diverged"},  # no ahead_by
        {"ahead_by": -3},  # negative
        {"ahead_by": "12"},  # wrong type
        {"ahead_by": True},  # bool masquerading as int
        [],  # wrong shape
    ],
)
def test_compare_behind_rejects_malformed_payloads(payload):
    with _patch_urlopen(payload):
        assert banner._github_compare_behind(SHA_A, SHA_B) is None


# ---------------------------------------------------------------------------
# _check_via_rev: sentinel replaced by exact count when compare API answers
# ---------------------------------------------------------------------------


def _upstream_tip(sha):
    return patch.object(banner, "_github_branch_tip", return_value=sha)


def test_check_via_rev_recovers_exact_count():
    with _upstream_tip(SHA_B), patch.object(banner, "_github_compare_behind", return_value=61) as compare:
        assert banner._check_via_rev(SHA_A) == 61
    compare.assert_called_once_with(SHA_A, SHA_B)


def test_check_via_rev_falls_back_to_sentinel_offline():
    """FAIL-BEFORE (class): this path returned a fabricated 1 via callers."""
    with _upstream_tip(SHA_B), patch.object(banner, "_github_compare_behind", return_value=None):
        assert banner._check_via_rev(SHA_A) == banner.UPDATE_AVAILABLE_NO_COUNT


def test_check_via_rev_up_to_date_short_circuits_compare():
    with _upstream_tip(SHA_A), patch.object(banner, "_github_compare_behind") as compare:
        assert banner._check_via_rev(SHA_A) == 0
    compare.assert_not_called()


def test_check_via_rev_local_ahead_reports_up_to_date():
    """ahead_by == 0 with differing tips = local commits on top, not behind."""
    with _upstream_tip(SHA_B), patch.object(banner, "_github_compare_behind", return_value=0):
        assert banner._check_via_rev(SHA_A) == 0


# ---------------------------------------------------------------------------
# _check_via_local_git: tips from the API, exact count via compare, no fetch
# ---------------------------------------------------------------------------


def _local_git(head_sha):
    def fake_run(cmd, **kwargs):
        if cmd[:4] == ["git", "remote", "get-url", "origin"]:
            return MagicMock(returncode=0, stdout="https://github.com/NousResearch/hermes-agent.git\n")
        if cmd[:3] == ["git", "rev-parse", "HEAD"]:
            return MagicMock(returncode=0, stdout=f"{head_sha}\n")
        if cmd[:3] == ["git", "merge-base", "--is-ancestor"]:
            return MagicMock(returncode=1, stdout="")
        raise AssertionError(f"unexpected git command: {cmd!r}")

    return fake_run


def test_local_checkout_recovers_exact_count(tmp_path):
    """The #84591 shape: no local history across the tips (shallow clone), tips differ.

    FAIL-BEFORE (class): reported UPDATE_AVAILABLE_NO_COUNT (or, further back,
    a fabricated 1) even though the compare API could count exactly.
    """
    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()

    with patch("hermes_cli.banner.subprocess.run", side_effect=_local_git(SHA_A)), \
            patch.object(banner, "_github_branch_tip", return_value=SHA_B), \
            patch.object(banner, "_github_compare_behind", return_value=61):
        assert banner._check_via_local_git(repo_dir) == 61


def test_local_checkout_offline_compare_keeps_honest_sentinel(tmp_path):
    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()

    with patch("hermes_cli.banner.subprocess.run", side_effect=_local_git(SHA_A)), \
            patch.object(banner, "_github_branch_tip", return_value=SHA_B), \
            patch.object(banner, "_github_compare_behind", return_value=None):
        assert banner._check_via_local_git(repo_dir) == banner.UPDATE_AVAILABLE_NO_COUNT


def test_local_checkout_equal_tips_up_to_date_without_compare(tmp_path):
    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()

    with patch("hermes_cli.banner.subprocess.run", side_effect=_local_git(SHA_A)), \
            patch.object(banner, "_github_branch_tip", return_value=SHA_A), \
            patch.object(banner, "_github_compare_behind") as compare:
        assert banner._check_via_local_git(repo_dir) == 0
    compare.assert_not_called()
