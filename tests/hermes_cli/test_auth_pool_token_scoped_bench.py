"""A 401/403 bench judges the token pair it saw; a billing bench is account-wide.

``_merge_pool_row_generation`` runs when a stale writer persists a row another process has since
refreshed: a sign-in rejection of the old bearer must not land on the new one, a spending block must.
"""

from __future__ import annotations

import pytest

from hermes_cli.auth import _merge_pool_row_generation

OLD, NEW = ("ya29.old", "1//rt"), ("ya29.new", "1//rt")


def _row(pair, **status):
    return {"id": "r1", "access_token": pair[0], "refresh_token": pair[1], **status}


@pytest.mark.parametrize(("status", "kept_on_new_pair"), [
    ({"last_status": "exhausted", "last_error_code": 401}, False),
    ({"last_status": "exhausted", "last_error_code": 403}, False),
    ({"last_status": "exhausted", "last_error_code": 403, "failure_reason": "billing"}, True),
    ({"last_status": "exhausted", "last_error_code": 429}, True),
])
def test_a_stale_bench_lands_on_a_peers_new_pair_only_when_account_wide(status, kept_on_new_pair):
    stale = _row(OLD, last_status_at=1e12, **status)
    merged = _merge_pool_row_generation(stale, _row(NEW), "example", base_pair=OLD)
    assert merged["access_token"] == "ya29.new"
    assert (merged.get("last_status") == "exhausted") is kept_on_new_pair
