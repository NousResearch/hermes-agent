"""fallback_providers across the discovered catalog: a dead provider falls through to the next.

Every runnable provider is the PRIMARY of one row, its catalog neighbour the fallback (both at
their own loopback fakes, both configured through the documented ``model.base_url`` /
``fallback_providers[].base_url`` keys). The primary answers HTTP 500 to every inference request.
The oneshot must still deliver the fallback's answer after one tool round trip, the fallback must
be called with ITS OWN key only, and the primary's key must never reach the fallback's endpoint.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tests.e2e.core.providers._catalog_helpers import (
    FINAL, URL_SUFFIX_OF_DIALECT, Row, decoy_keys, discover_catalog, run_hermes, write_home,
)
from tests.fakes.providers.catalog_fake import CatalogFake

CATALOG = discover_catalog()
# Rows whose primary can be redirected; xai cannot (#121347), so it is never a primary/fallback here.
ROWS = [r for r in CATALOG if r.skip_reason() is None and r.name != "xai"]
# Keyed on the FALLBACK provider (pairs follow catalog order, so primaries shift as plugins are
# added). Strict: a listed cell that turns green fails the row until the entry is dropped.
_FB_BASE_URL_IGNORED = "#121359 fallback entry ignores its base_url; goes to the vendor host"
_FB_CELLS = ("fallback_answered", "fallback_tool_round_trip", "fallback_used_own_key")
KNOWN: dict[tuple[str, str], str] = {
    **{("anthropic", c): _FB_BASE_URL_IGNORED for c in _FB_CELLS},
    **{("openrouter", c): _FB_BASE_URL_IGNORED for c in _FB_CELLS},
}


def _url(fake: CatalogFake, row: Row) -> str:
    return f"{fake.origin}/{row.name}{URL_SUFFIX_OF_DIALECT[row.dialect or 'chat']}"


def _drive(primary: Row, fallback: Row, root: Path) -> dict:
    project = root / "project"
    project.mkdir(parents=True, exist_ok=True)
    canary = f"CANARY-FB-{primary.name}"
    (project / "canary.txt").write_text(canary + "\n", encoding="utf-8")
    keys = decoy_keys(CATALOG)
    args = {"path": str(project / "canary.txt")}
    with CatalogFake(fail_status=500) as dead, CatalogFake(tool_args=args, final_text=FINAL) as alive:
        home = write_home(root, {"provider": primary.name, "base_url": _url(dead, primary)}, {
            "fallback_providers": [{"provider": fallback.name, "model": "catalog-model-a",
                                    "base_url": _url(alive, fallback)}]})
        proc = run_hermes(home, project, {**keys, **dead.proxy_env()}, "-z", "Read canary.txt and report.")
        dead_reqs, alive_reqs = dead.inference(), alive.inference()
        egress = dead.egress_hosts()
    pk, fk = keys[primary.key_env or ""], keys[fallback.key_env or ""]
    return {
        "rc": proc.returncode, "stdout": proc.stdout[-400:], "stderr": proc.stderr[-1200:],
        "egress": egress, "dead_paths": [r.path for r in dead_reqs], "alive_paths": [r.path for r in alive_reqs],
        "cells": {
            "primary_tried_first": bool(dead_reqs) and any(pk in v for v in dead_reqs[0].headers.values()),
            "fallback_answered": proc.returncode == 0 and FINAL in proc.stdout,
            "fallback_tool_round_trip": any(canary in json.dumps(r.body) for r in alive_reqs),
            "fallback_used_own_key": bool(alive_reqs) and all(
                any(fk in v for v in r.headers.values()) for r in alive_reqs),
            "primary_key_not_sent_to_fallback": pk == fk or not any(
                pk in v for r in alive_reqs for v in r.headers.values()),
        },
    }


@pytest.fixture(scope="module")
def results(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict]:
    pairs = {r.name: (r, ROWS[(i + 1) % len(ROWS)]) for i, r in enumerate(ROWS)}
    with ThreadPoolExecutor(max_workers=8, thread_name_prefix="fallback") as pool:
        futs = {n: pool.submit(_drive, p, f, Path(tmp_path_factory.mktemp(f"fb-{n}"))) for n, (p, f) in pairs.items()}
        out = {n: f.result() for n, f in futs.items()}
    for n, (_p, f) in pairs.items():
        out[n]["fallback"] = f.name
    return out


@pytest.mark.parametrize("row", [pytest.param(r, id=r.name) for r in ROWS])
def test_dead_primary_falls_through(row: Row, results: dict[str, dict]) -> None:
    res = results[row.name]
    cells = res["cells"]
    known = {c: ref for (p, c), ref in KNOWN.items() if p == res["fallback"]}
    fixed = sorted(c for c in known if cells.get(c))
    assert not fixed, f"{row.name}: {fixed} now green — drop their KNOWN entries ({set(known.values())})"
    failed = sorted(c for c, ok in cells.items() if not ok and c not in known)
    assert not failed, f"{row.name} -> {res['fallback']}: cells red: {failed}\n" + json.dumps(
        {k: v for k, v in res.items() if k != "cells"})[:2500]
    if known:
        pytest.xfail(f"{sorted(known)}: {'; '.join(sorted(set(known.values())))}")
