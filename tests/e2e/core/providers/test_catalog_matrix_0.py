"""Provider-catalog matrix, shard 0/3: one real oneshot turn per discovered provider.

Rows are the providers whose name hashes to this shard (``_catalog_helpers.shard_of``), so a new
plugin joins some shard automatically. Each row runs ``hermes -z`` against its own loopback fake
(redirected via ``model.base_url``) with every other provider's key present as a decoy, and checks:
the turn completes with a tool round trip through the provider's dialect; only the provider's own
key reaches the wire, in the dialect's auth header, and only at its configured endpoint (no egress
to any provider host); usage lands in state.db and unknown pricing is not a silent $0.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tests.e2e.core.providers._catalog_helpers import (
    SHARDS, Row, TurnResult, discover_catalog, drive_turn, evaluate, shard_of,
)

SHARD = 0
# Cells red on origin/main for a tracked open bug: (provider, cell) -> "#issue one line".
# Strict: the row FAILS as soon as a listed cell turns green (drop the entry with the fix), and
# any cell NOT listed still fails normally, so a known bug never masks a new regression.
XAI_BASE_URL_IGNORED = "#121347 xai ignores model.base_url; request + key go to api.x.ai"
KNOWN: dict[tuple[str, str], str] = {
    ("xai", "turn_completed"): XAI_BASE_URL_IGNORED,
    ("xai", "reached_own_endpoint"): XAI_BASE_URL_IGNORED,
    ("xai", "dialect_matches_transport"): XAI_BASE_URL_IGNORED,
    ("xai", "tool_round_trip"): XAI_BASE_URL_IGNORED,
    ("xai", "own_key_in_auth_header"): XAI_BASE_URL_IGNORED,
    ("xai", "usage_recorded"): XAI_BASE_URL_IGNORED,
}

CATALOG = discover_catalog()
ROWS = [r for r in CATALOG if shard_of(r.name) == SHARD]


@pytest.fixture(scope="module")
def turns(tmp_path_factory: pytest.TempPathFactory) -> dict[str, TurnResult]:
    """Every runnable row of the shard driven concurrently (own home, fake and process each)."""
    runnable = [r for r in ROWS if r.skip_reason() is None]
    roots = {r.name: Path(tmp_path_factory.mktemp(f"cat-{r.name}")) for r in runnable}
    with ThreadPoolExecutor(max_workers=8, thread_name_prefix="catalog") as pool:
        futs = {r.name: pool.submit(drive_turn, r, roots[r.name], CATALOG) for r in runnable}
        return {name: f.result() for name, f in futs.items()}


def _params() -> list:
    out = []
    for r in ROWS:
        marks = [pytest.mark.skip(reason=r.skip_reason())] if r.skip_reason() else []
        out.append(pytest.param(r, id=r.name, marks=marks))
    return out


@pytest.mark.parametrize("row", _params())
def test_provider_row(row: Row, turns: dict[str, TurnResult]) -> None:
    t = turns[row.name]
    cells = evaluate(t, CATALOG)
    known = {c: ref for (p, c), ref in KNOWN.items() if p == row.name}
    fixed = sorted(c for c in known if cells.get(c))
    assert not fixed, f"{row.name}: {fixed} now green — drop their KNOWN entries ({set(known.values())})"
    failed = sorted(c for c, ok in cells.items() if not ok and c not in known)
    assert not failed, f"{row.name} ({row.api_mode}/{row.auth_type}): cells red: {failed}\n{t.detail()}"
    if known:
        pytest.xfail(f"{sorted(known)}: {'; '.join(sorted(set(known.values())))}")


def test_catalog_is_discovered_not_listed() -> None:
    """Every bundled plugin dir registers a provider that some shard runs (or skips with a
    reason): the matrix follows discovery, so a new plugin can never silently fall out of it."""
    root = Path(__file__).resolve().parents[4] / "plugins" / "model-providers"
    dirs = {d.name for d in root.iterdir() if (d / "__init__.py").exists()}
    names = {r.name for r in CATALOG}
    assert dirs, "no bundled model-provider plugins found"
    assert dirs <= names, f"plugin dirs with no discovered profile: {sorted(dirs - names)}"
    assert {shard_of(n) for n in names} <= set(range(SHARDS))
