"""Tests for the QMD-in-mem0 unified-recall fold-in (spec v0.3).

Pure-function + adversarial coverage that needs no live daemon. Live E2E is a separate
worktree smoke step (Phase-2/3 verify). Run:
  venv/bin/python -m pytest plugins/memory/mem0/test_qmd_recall.py -v -o addopts=""
"""
import gc
import json
import os
import socket
import threading
import time
from pathlib import Path

import pytest

from plugins.memory.mem0 import qmd_recall as qr

FIXTURE = Path(__file__).parent / "testdata" / "qmd_query_response.json"


# ---- Task 1: config -------------------------------------------------------
def test_config_defaults_when_absent():
    cfg = qr.load_qmd_config(None)
    assert cfg["enabled"] is False
    assert cfg["qmd_total_deadline_s"] == 6.0
    assert cfg["mem0_budget_s"] == 6.0
    assert cfg["min_score"] == 0.45
    assert cfg["prefetch_limit"] == 3
    assert cfg["search_limit"] == 5
    assert cfg["collections"] == ["obsidian", "skills", "plans", "projects"]
    assert "sessions" not in cfg["collections"] and "memories" not in cfg["collections"]
    assert cfg["exclude_path_globs"] == []
    assert cfg["intent_min_tokens"] == 1
    assert cfg["prefetch_rerank"] is True
    assert cfg["use_rerank_score_floor"] is False
    assert cfg["rerank_score_min"] == 0.50
    # budgets fit the join ceiling (INV-4a)
    # The runtime clamp (eff_deadline = min(qmd_deadline, join - mem0_elapsed - 0.25)) is what
    # actually guarantees the two legs never exceed the join ceiling — the nominal knobs need not
    # sum under it. Assert each knob is individually bounded by the join ceiling instead (INV-4a).
    assert cfg["mem0_budget_s"] <= 10.0
    assert cfg["qmd_total_deadline_s"] <= 10.0


def test_config_overrides_merge():
    cfg = qr.load_qmd_config({"enabled": True, "min_score": 0.7, "bogus": 1})
    assert cfg["enabled"] is True and cfg["min_score"] == 0.7
    assert "bogus" not in cfg  # unknown keys ignored


# ---- Task 2: intent gate (AC3, D-9/INV-7) ---------------------------------
# The token-count floor defaults to 1 (the leader-word set is the real gate); these tests
# exercise the SHIPPED default so a future floor bump that re-breaks short lookups is caught.
_MT = qr.QMD_DEFAULTS["intent_min_tokens"]


@pytest.mark.parametrize("q", [
    "where did we decide the local dns split",
    "is there a skill that covers reolink doorbell audio",
    "find the spec for the greenhouse nightly seeds build",
    "what did we conclude about the hdmi splitter edid",
    # short, specific lookups the old min_tokens=4 floor wrongly dropped (the blocked.local.ace miss):
    "reolink doorbell voice",
    "local dns split",
    "blocked.local.ace",
    "unifi",
])
def test_intent_true_for_lookups(q):
    assert qr.is_lookup_intent(q, _MT) is True


@pytest.mark.parametrize("q", [
    # affirmations / imperatives — caught by the LEADER set regardless of length:
    "yes", "ok", "thanks", "ship it", "do it", "fix line 12", "go ahead",
    "yes do that", "", "   ", "perfect, ship it", "run it", "no", "stop", "commit",
])
def test_intent_false_for_non_lookups(q):
    assert qr.is_lookup_intent(q, _MT) is False


# ---- Task 3: render + join (m2, INV-7) ------------------------------------
def test_render_empty():
    assert qr.render_qmd_block([]) == ""


def test_render_pointers_only():
    hits = [{"file": "obsidian/DNS-PRD.md", "title": "DNS Block Portal", "score": 0.93, "line": 1,
             "docid": "#ec41f3", "snippet": "SHOULD NOT APPEAR"}]
    out = qr.render_qmd_block(hits)
    assert out.startswith("## Local Docs (QMD)")
    assert "obsidian/DNS-PRD.md — DNS Block Portal (93%) :1" in out
    assert "SHOULD NOT APPEAR" not in out


def test_join_skips_separator_when_empty():
    assert qr.join_blocks("MEM", "") == "MEM"      # byte-identical guard
    assert qr.join_blocks("", "QMD") == "QMD"
    assert qr.join_blocks("MEM", "QMD") == "MEM\n\nQMD"
    assert qr.join_blocks("", "") == ""


# ---- Task 4: parse against the CAPTURED real fixture (M1, INV-5) ----------
def test_parse_real_fixture():
    payload = json.loads(FIXTURE.read_text())
    hits = qr.parse_qmd_results(payload, min_score=0.3)
    assert len(hits) >= 1
    top = hits[0]
    assert set(top) == {"file", "title", "score", "line", "docid"}  # NO snippet/context
    assert "snippet" not in top and "context" not in top
    assert top["score"] >= 0.3


def test_parse_min_score_filters():
    payload = json.loads(FIXTURE.read_text())
    # an impossibly-high floor drops everything
    assert qr.parse_qmd_results(payload, min_score=2.0) == []


def test_parse_semantic_floor_uses_rerank_score_not_blended_score():
    payload = {"result": {"structuredContent": {"results": [
        {"file": "obsidian/semantic.md", "title": "semantic", "score": 0.10, "rerankScore": 0.82, "line": 1, "docid": "#a"},
        {"file": "obsidian/positional.md", "title": "positional", "score": 0.99, "rerankScore": 0.20, "line": 2, "docid": "#b"},
        {"file": "obsidian/missing.md", "title": "missing", "score": 0.99, "line": 3, "docid": "#c"},
        {"file": "obsidian/null.md", "title": "null", "score": 0.99, "rerankScore": None, "line": 4, "docid": "#d"},
    ]}}}
    hits = qr.parse_qmd_results(
        payload,
        min_score=0.95,               # ignored by semantic mode; blended score is display-only
        use_rerank_score_floor=True,
        rerank_score_min=0.50,
    )
    assert [h["file"] for h in hits] == ["obsidian/semantic.md"]
    assert hits[0]["score"] == 0.10   # rendered score remains QMD's blended display score


def test_build_query_args_semantic_floor_requests_explain_and_disables_blended_prefilter():
    default_args = qr.build_qmd_query_args(
        "local dns split", limit=3, min_score=0.45, collections=["obsidian"], rerank=True
    )
    assert default_args["minScore"] == 0.45
    assert "explain" not in default_args

    semantic_args = qr.build_qmd_query_args(
        "local dns split", limit=3, min_score=0.45, collections=["obsidian"], rerank=True,
        use_rerank_score_floor=True,
    )
    assert semantic_args["minScore"] == 0.0
    assert semantic_args["explain"] is True
    assert semantic_args["collections"] == ["obsidian"]


def test_parse_exclude_globs_drops_match():
    payload = {"result": {"structuredContent": {"results": [
        {"file": "sessions/secret.md", "title": "S", "score": 0.9, "line": 1, "docid": "#a"},
        {"file": "obsidian/ok.md", "title": "O", "score": 0.9, "line": 2, "docid": "#b"},
    ]}}}
    hits = qr.parse_qmd_results(payload, 0.3, exclude_globs=["sessions/*"])
    assert [h["file"] for h in hits] == ["obsidian/ok.md"]


def test_parse_garbage_is_empty():
    assert qr.parse_qmd_results(None, 0.3) == []
    assert qr.parse_qmd_results({"result": {}}, 0.3) == []
    assert qr.parse_qmd_results({"nonsense": 1}, 0.3) == []


# ---- Task 4: degrade + trickle (AC5, AC8, INV-3/INV-4) -------------------
def test_query_dead_endpoint_returns_empty_no_raise():
    # nothing listening on this port → connection refused → [] (no raise)
    t0 = time.monotonic()
    out = qr.qmd_query("x", limit=3, min_score=0.5, collections=None, rerank=False,
                       deadline_s=2, url="http://[::1]:59999/mcp")
    assert out == []
    assert time.monotonic() - t0 < 3.0


def _trickle_server():
    """A server that accepts, sends an SSE keepalive comment forever, never completes."""
    srv = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("::1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    stop = threading.Event()

    def serve():
        try:
            conn, _ = srv.accept()
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n")
            while not stop.is_set():
                try:
                    conn.sendall(b": keepalive\n\n")
                except Exception:
                    break
                time.sleep(0.2)
        except Exception:
            pass
    th = threading.Thread(target=serve, daemon=True)
    th.start()
    return srv, port, stop, th


def test_query_sse_trickle_hang_aborts_within_deadline_no_leak():
    srv, port, stop, th = _trickle_server()
    try:
        gc.collect()
        threads_before = threading.active_count()
        t0 = time.monotonic()
        out = qr.qmd_query("x", limit=3, min_score=0.5, collections=None, rerank=False,
                           deadline_s=1.0, url=f"http://[::1]:{port}/mcp")
        elapsed = time.monotonic() - t0
        assert out == []                       # degraded to empty
        assert elapsed < 2.5                   # the watchdog tripped ~1s, not forever
        time.sleep(0.3)
        gc.collect()
        threads_after = threading.active_count()
        # no leaked worker/watchdog thread left running after the call
        assert threads_after <= threads_before + 1
    finally:
        stop.set()
        srv.close()
