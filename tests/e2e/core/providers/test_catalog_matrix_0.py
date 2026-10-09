"""Provider-catalog matrix, shard 0/3: one real oneshot turn per discovered provider.

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

from pathlib import Path

import pytest

from tests.e2e.core.providers._catalog_helpers import (
    SHARDS, Row, TurnResult, check_row, discover_catalog, drive_shard, shard_of, shard_params,
)

SHARD = 0


@pytest.fixture(scope="module")
def turns(tmp_path_factory: pytest.TempPathFactory) -> dict[str, TurnResult]:
    return drive_shard(SHARD, tmp_path_factory)


@pytest.mark.parametrize("row", shard_params(SHARD))
def test_provider_row(row: Row, turns: dict[str, TurnResult]) -> None:
    check_row(row, turns)


def test_catalog_is_discovered_not_listed() -> None:
    """Every bundled plugin dir registers a provider that some shard runs (or skips with a
    reason): the matrix follows discovery, so a new plugin can never silently fall out of it."""
    root = Path(__file__).resolve().parents[4] / "plugins" / "model-providers"
    dirs = {d.name for d in root.iterdir() if (d / "__init__.py").exists()}
    names = {r.name for r in discover_catalog()}
    assert dirs, "no bundled model-provider plugins found"
    assert dirs <= names, f"plugin dirs with no discovered profile: {sorted(dirs - names)}"
    assert {shard_of(n) for n in names} <= set(range(SHARDS))
