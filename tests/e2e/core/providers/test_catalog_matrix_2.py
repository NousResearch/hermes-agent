"""Provider-catalog matrix, shard 2/3: one real oneshot turn per discovered provider.

Rows are the providers whose name hashes to this shard (``_catalog_helpers.shard_of``), so a new
plugin joins some shard automatically. Each row runs ``hermes -z`` against its own loopback fake
(redirected via ``model.base_url``; the fake answers only the exact configured path) with every
other provider's key present as a decoy, and checks: the turn completes with a tool round trip
through the provider's dialect; only the provider's own key reaches the wire, in the dialect's auth
header, and only at its configured endpoint (no egress to another provider's host); the session's
usage equals what the fake reported and unknown pricing is not a silent $0. Open bugs are gated on
their own signature in ``_catalog_helpers.MATRIX_KNOWN``.
"""

from __future__ import annotations

import pytest

from tests.e2e.core.providers._catalog_helpers import Row, TurnResult, check_row, drive_shard, shard_params

SHARD = 2


@pytest.fixture(scope="module")
def turns(tmp_path_factory: pytest.TempPathFactory) -> dict[str, TurnResult]:
    return drive_shard(SHARD, tmp_path_factory)


@pytest.mark.parametrize("row", shard_params(SHARD))
def test_provider_row(row: Row, turns: dict[str, TurnResult]) -> None:
    check_row(row, turns)
