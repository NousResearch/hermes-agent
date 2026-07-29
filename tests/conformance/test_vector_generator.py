"""Conformance vector generator tests (Phase 5 oracle workstream).

Behavior contracts, not change-detectors: determinism, corpus invariants,
oracle-import health (the generator calls format_message UNBOUND — these
tests fail loudly if a refactor gives those renderers instance state), and
vector-file shape. Native outputs themselves are NOT snapshotted here — the
connector's conformance runner is the consumer that asserts them.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate_conformance_vectors import (  # noqa: E402
    ADVERSARIAL,
    GRID,
    SCAR,
    _oracles,
    corpus,
    generate,
)

PLATFORMS = ("discord", "slack", "telegram", "whatsapp")


def test_corpus_ids_unique_and_categorized():
    rows = corpus()
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "vector ids must be unique (runner keys on them)"
    assert {r["category"] for r in rows} == {"grid", "scar", "adversarial"}
    assert len(GRID) >= 20 and len(SCAR) >= 10 and len(ADVERSARIAL) >= 8


def test_oracles_importable_and_self_free():
    """The generator invokes format_message unbound (self=None) — verify every
    oracle actually renders under that calling convention. A refactor that
    adds instance state to a renderer must update the generator too."""
    oracles = _oracles()
    assert set(oracles) == set(PLATFORMS)
    for platform, render in oracles.items():
        out = render("**bold** and `code`")
        assert isinstance(out, str) and out, platform




def test_vector_files_shape(tmp_path):
    generate(tmp_path)
    for platform in PLATFORMS:
        doc = json.loads((tmp_path / f"{platform}.json").read_text(encoding="utf-8"))
        assert doc["platform"] == platform
        assert doc["oracle"]["repo"] == "NousResearch/hermes-agent"
        assert re.match(r"^[0-9a-f]{40}$|^unknown$", doc["oracle"]["commit"])
        assert doc["oracle"]["generator_version"] >= 1
        ids = set()
        for v in doc["vectors"]:
            assert v["expect"] in ("parity", "semantic", "divergent"), v["id"]
            assert isinstance(v["native_output"], str)
            if v["expect"] == "divergent":
                assert v.get("note"), f"{platform}/{v['id']}: divergent needs a note"
            ids.add(v["id"])
        assert len(ids) == len(doc["vectors"])






def test_committed_vectors_match_regeneration(tmp_path):
    """The committed tests/conformance/vectors/ must reproduce from the
    current oracle — the same lockstep discipline as openapi.json (drift
    means someone changed a renderer without regenerating)."""
    committed_dir = REPO_ROOT / "tests" / "conformance" / "vectors"
    if not committed_dir.exists():
        pytest.skip("no committed vectors in this checkout")
    generate(tmp_path)
    for platform in PLATFORMS:
        fresh = json.loads((tmp_path / f"{platform}.json").read_text(encoding="utf-8"))
        committed = json.loads((committed_dir / f"{platform}.json").read_text(encoding="utf-8"))
        # Compare everything except the commit stamp (committed file may be
        # one commit behind HEAD in a dirty working tree).
        fresh["oracle"].pop("commit")
        committed["oracle"].pop("commit")
        assert fresh == committed, (
            f"{platform} vectors drifted — run "
            "`python scripts/generate_conformance_vectors.py` and commit"
        )
