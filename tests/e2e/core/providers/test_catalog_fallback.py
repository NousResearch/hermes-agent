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
    AUTH_HEADER_OF_DIALECT, FINAL, Known, Row, decoy_keys, discover_catalog, gate, known_gate, run_hermes, write_home,
)
from tests.fakes.providers.catalog_fake import CatalogFake

CATALOG = discover_catalog()
# Rows whose primary can be redirected; xai cannot (#121347), so it is never a primary/fallback
# here. Rows with no loopback dialect FAIL in the matrix shards.
ROWS = [r for r in CATALOG if r.skip_reason() is None and r.dialect and r.name != "xai"]
# Keyed on the FALLBACK provider (pairs follow catalog order, so primaries shift as plugins are
# added). Signature: the fallback's own fake got nothing while its vendor host was CONNECTed.
_FB_CELLS = frozenset({"fallback_answered", "fallback_tool_round_trip", "fallback_used_own_key"})


def _fb_known(name: str) -> Known:
    host = next(r.vendor_host() for r in CATALOG if r.name == name)
    return Known(rf"^alive_inference=0 egress=\[[^\]]*'{host.replace('.', r'[.]')}'",
                 "#121359 fallback entry ignores its base_url; goes to the vendor host", _FB_CELLS)


KNOWN: dict[str, Known] = {n: _fb_known(n) for n in ("anthropic", "openrouter")}


def _drive(primary: Row, fallback: Row, root: Path) -> dict:
    project = root / "project"
    project.mkdir(parents=True, exist_ok=True)
    canary = f"CANARY-FB-{primary.name}"
    (project / "canary.txt").write_text(canary + "\n", encoding="utf-8")
    keys = decoy_keys(CATALOG)
    args = {"path": str(project / "canary.txt")}
    with CatalogFake(fail_status=500, routes=primary.routes()) as dead, \
            CatalogFake(tool_args=args, final_text=FINAL, routes=fallback.routes()) as alive:
        home = write_home(root, {"provider": primary.name, "base_url": dead.origin + primary.base_path}, {
            "fallback_providers": [{"provider": fallback.name, "model": "catalog-model-a",
                                    "base_url": alive.origin + fallback.base_path}]})
        proc = run_hermes(home, project, {**keys, **dead.proxy_env()}, "-z", "Read canary.txt and report.")
        dead_reqs, alive_reqs = dead.inference(), alive.inference()
        egress = dead.egress_hosts()
    pk, fk = keys[primary.key_env or ""], keys[fallback.key_env or ""]
    return {
        "rc": proc.returncode, "stdout": proc.stdout[-400:], "stderr": proc.stderr[-1200:], "egress": egress,
        "alive_inference": len(alive_reqs), "dead_paths": [f"{r.path} [{r.dialect}]" for r in dead_reqs],
        "alive_paths": [f"{r.path} [{r.dialect}]" for r in alive_reqs],
        "cells": {
            "primary_tried_first": bool(dead_reqs) and dead_reqs[0].dialect != "unknown" and pk in dead_reqs[0].headers.get(
                AUTH_HEADER_OF_DIALECT[dead_reqs[0].dialect], ""),
            "fallback_answered": proc.returncode == 0 and FINAL in proc.stdout,
            "fallback_tool_round_trip": any(canary in json.dumps(r.body) for r in alive_reqs),
            # Every fallback call hit its exact configured route with ITS key in the dialect's header.
            "fallback_used_own_key": bool(alive_reqs) and all(
                r.dialect != "unknown" and fk in r.headers.get(AUTH_HEADER_OF_DIALECT[r.dialect], "") for r in alive_reqs),
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
    failed = sorted(c for c, ok in cells.items() if not ok)
    known = KNOWN.get(res["fallback"])
    unlisted = [c for c in failed if not known or c not in known.cells]
    facts = json.dumps({k: v for k, v in res.items() if k != "cells"})[:2500]
    assert not unlisted, f"{row.name} -> {res['fallback']}: cells red: {unlisted}\n{facts}"
    with known_gate(known):
        gate(not failed, f"alive_inference={res['alive_inference']} egress={res['egress']}\n"
                         f"{row.name} -> {res['fallback']}: cells red: {failed}\n{facts}")
