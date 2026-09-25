"""Behind-count recovery via the GitHub compare API (source_check.py).

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

import json
from unittest.mock import patch

import pytest

import hermes_cli.source_check as source_check

SHA_A = "a" * 40
SHA_B = "b" * 40


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self, limit=None):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _patch_urlopen(payload):
    return patch(
        "urllib.request.urlopen",
        return_value=_FakeResponse(json.dumps(payload).encode()),
    )


# ---------------------------------------------------------------------------
# _github_compare_behind
# ---------------------------------------------------------------------------


def test_compare_behind_returns_ahead_by():
    with _patch_urlopen({"ahead_by": 61, "status": "ahead"}):
        assert source_check._github_compare_behind(SHA_A, SHA_B) == 61


def test_compare_behind_zero_means_local_ahead():
    with _patch_urlopen({"ahead_by": 0, "status": "behind"}):
        assert source_check._github_compare_behind(SHA_A, SHA_B) == 0


def test_compare_behind_rejects_short_shas_without_network():
    with patch("urllib.request.urlopen") as mock_open:
        assert source_check._github_compare_behind("abc123", SHA_B) is None
        assert source_check._github_compare_behind(SHA_A, "") is None
        assert source_check._github_compare_behind(None, SHA_B) is None
    mock_open.assert_not_called()


def test_compare_behind_network_failure_returns_none():
    with patch("urllib.request.urlopen", side_effect=OSError("offline")):
        assert source_check._github_compare_behind(SHA_A, SHA_B) is None


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
        assert source_check._github_compare_behind(SHA_A, SHA_B) is None
