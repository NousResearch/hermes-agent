"""The shared update-marker corpus, run against the Python reader/releaser (A7 rule 7).

``tests/fixtures/update_marker_corpus.json`` is the one table every marker implementation
(Python, Rust ``marker.rs``, Electron, the Bash/PowerShell hand-off scripts) must agree with.
The process table and clock are injected; the bytes, parser and verdict code are production.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import update_lock

CORPUS = json.loads((Path(__file__).resolve().parents[1] / "fixtures/update_marker_corpus.json")
                    .read_text(encoding="utf-8"))


def _world(case: dict, pid_key: str, ct_key: str) -> update_lock._World:
    me = (case.get(pid_key, CORPUS["our_pid"]), case.get(ct_key, CORPUS["our_ct"]))
    table = {int(pid): ct for pid, ct in case["live"].items()}
    table[me[0]] = me[1]
    return update_lock._World(pid=me[0], ct=me[1], now=CORPUS["now"], alive=lambda pid: pid in table,
                              ct_of=lambda pid: table.get(pid))


def test_corpus_constants_are_the_production_ones():
    assert CORPUS["own_ct_epsilon"] == update_lock._OWN_CREATE_TIME_EPSILON
    assert CORPUS["ct_tolerance"] == update_lock.CREATE_TIME_TOLERANCE_SECONDS
    assert CORPUS["v1_max_age"] == update_lock.UPDATE_MARKER_MAX_AGE_SECONDS


@pytest.mark.parametrize("case", CORPUS["judge"], ids=[c["name"] for c in CORPUS["judge"]])
def test_judge(case):
    verdict, owner, run = update_lock.judge_marker(case["text"].encode("utf-8"), _world(case, "our_pid", "our_ct"))
    assert {"verdict": verdict, "owner": owner, "run": run} == case["expect"]


@pytest.mark.parametrize("case", CORPUS["release"], ids=[c["name"] for c in CORPUS["release"]])
def test_release(case):
    action, body = update_lock._release_decision(case["text"].encode("utf-8"),
                                                  _world(case, "releaser_pid", "releaser_ct"))
    expected = case["expect"]
    assert action == expected["action"]
    if action == "rewrite":
        assert body is not None
        assert body.decode("utf-8") == expected["text"]
