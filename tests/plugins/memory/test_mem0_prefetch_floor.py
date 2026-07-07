"""Tests for the mem0 prefetch relevance floor (spec 2026-07-06 prefetch-relevance-floor v0.6).

TWO layers:
  Gate A (specificity, PRIMARY): a pure-acknowledgment turn → inject nothing.
  Gate B (cosine, weak SECONDARY): trim a truly-orthogonal candidate on a substantive turn.
Both fail OPEN to today's behavior on any error; both stamp a greppable outcome.
"""

import json
import math

import pytest

from plugins.memory.mem0 import Mem0MemoryProvider


def _provider(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("MEM0_HOST", "http://mem0.test")
    monkeypatch.setenv("MEM0_ADMIN_API_KEY", "admin-key")
    monkeypatch.setenv("MEM0_USER_ID", "ace")
    monkeypatch.setenv("MEM0_AGENT_ID", "apollo")
    monkeypatch.delenv("MEM0_API_KEY", raising=False)
    p = Mem0MemoryProvider()
    p.initialize("test-session")
    return p


def _write_floor_cfg(tmp_path, block):
    cfg_path = tmp_path / "mem0.json"
    existing = {}
    if cfg_path.exists():
        existing = json.loads(cfg_path.read_text())
    existing["prefetch_relevance_floor"] = block
    cfg_path.write_text(json.dumps(existing))


def _unit(vec):
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


_Q = _unit([1.0, 0.0, 0.0])
_NEAR = _unit([0.9, 0.1, 0.0])      # cosine ~0.99 to query
_ORTHO = _unit([0.0, 1.0, 0.0])     # cosine ~0.0


def _results(*memories):
    return [{"memory": m, "score": 1.0} for m in memories]


# ---------------------------------------------------------------------------
# GATE A — query specificity (the primary fix)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ack", [
    "yes please", "ok do it", "sure go ahead", "yes please proceed",
    "thanks that sounds good", "yep", "ok", "do it",
    "perfect thanks", "sounds great", "yes", "okay cool",
])
def test_specificity_gate_blanks_ack_query(monkeypatch, tmp_path, ack):
    """A pure-acknowledgment turn has 0 content tokens → gated (the fix)."""
    p = _provider(monkeypatch, tmp_path)
    _write_floor_cfg(tmp_path, {"enabled": True})
    assert p._content_token_count(ack) == 0
    assert p._prefetch_specificity_gated(ack) is True


@pytest.mark.parametrize("q", [
    "restart plex container", "reboot HAOS", "fix the DNS on the media server",
    "AC on", "TV off", "IP of the NAS", "reboot", "sleep", "plex",
    "what temperature is the guest room", "bent pin CPU socket LGA1718 NAS",
    "yes please spec out the mem0 prefetch relevance floor",  # ack-prefixed but substantive
    "mute the kitchen",
])
def test_specificity_gate_keeps_real_short_query(monkeypatch, tmp_path, q):
    """INV-7/AC-8: a genuine short query (incl. ≤2-char domain tokens, 1-token commands)
    has ≥1 content token → NOT gated. Zero false positives."""
    p = _provider(monkeypatch, tmp_path)
    _write_floor_cfg(tmp_path, {"enabled": True})
    assert p._content_token_count(q) >= 1, f"{q!r} wrongly stripped to 0"
    assert p._prefetch_specificity_gated(q) is False


def test_specificity_gate_disabled_never_gates(monkeypatch, tmp_path):
    """INV-3: enabled:false → Gate A never fires even on a pure ack."""
    p = _provider(monkeypatch, tmp_path)
    _write_floor_cfg(tmp_path, {"enabled": False})
    assert p._prefetch_specificity_gated("yes please") is False


def test_gate_a_fail_open_on_bad_input(monkeypatch, tmp_path):
    """AC-11: an error in content counting → NOT gated (treat as substantive), never crash."""
    p = _provider(monkeypatch, tmp_path)
    _write_floor_cfg(tmp_path, {"enabled": True})
    # None and odd types must not raise and must not gate.
    assert p._content_token_count(None) >= 1  # fail-open high count
    assert p._prefetch_specificity_gated(None) is False


def test_specificity_min_content_tokens_pinned_default_1(monkeypatch, tmp_path):
    p = _provider(monkeypatch, tmp_path)
    _write_floor_cfg(tmp_path, {"enabled": True})
    assert p._prefetch_min_content_tokens() == 1


# ---------------------------------------------------------------------------
# GATE B — weak cosine secondary
# ---------------------------------------------------------------------------

def test_gate_b_drops_low_cosine(monkeypatch, tmp_path):
    """INV-1: candidates with /search score 1.0 but true cosine below the floor are dropped.
    Uses cosine, NOT the saturated score."""
    p = _provider(monkeypatch, tmp_path)
    _write_floor_cfg(tmp_path, {"enabled": True, "min_cosine": 0.10})
    embeds = [_Q, _NEAR, _NEAR, _ORTHO, _ORTHO]
    monkeypatch.setattr(p, "_dedup_embed", lambda texts, *, timeout=15: embeds)
    results = _results("near1", "near2", "junk1", "junk2")
    kept, outcome = p._apply_gate_b_cosine("substantive query", results, budget_s=3.0)
    assert [r["memory"] for r in kept] == ["near1", "near2"]
    assert outcome == "ran_kept_2_of_4"


def test_gate_b_telemetry_logs_cosine_distribution(monkeypatch, tmp_path, caplog):
    """Telemetry (2026-07-07): the ran_kept line must carry the cosine DISTRIBUTION
    (cos_max/cos_med/cos_min + per-candidate list) so an operator can tell a genuinely
    on-topic recall from junk that merely slipped the low floor — not just a kept/total count."""
    import logging
    p = _provider(monkeypatch, tmp_path)
    _write_floor_cfg(tmp_path, {"enabled": True, "min_cosine": 0.10})
    embeds = [_Q, _NEAR, _ORTHO]
    monkeypatch.setattr(p, "_dedup_embed", lambda texts, *, timeout=15: embeds)
    results = _results("near1", "junk1")
    with caplog.at_level(logging.INFO):
        p._apply_gate_b_cosine("substantive query", results, budget_s=3.0)
    line = next((r.getMessage() for r in caplog.records if "prefetch_floor outcome=ran_kept" in r.getMessage()), "")
    assert line, "no ran_kept telemetry line emitted"
    # distribution fields present
    for field in ("cos_max=", "cos_med=", "cos_min=", "cos=", "floor="):
        assert field in line, f"telemetry missing {field}: {line}"
    # cos_max >= cos_min (sorted), and the per-candidate list has one entry per candidate
    import re as _re
    cmax = float(_re.search(r"cos_max=([0-9.]+)", line).group(1))
    cmin = float(_re.search(r"cos_min=([0-9.]+)", line).group(1))
    assert cmax >= cmin
    cos_list = _re.search(r"cos=([0-9.,]+)", line).group(1).split(",")
    assert len(cos_list) == 2, f"expected 2 per-candidate cosines, got {cos_list}"
    # no memory TEXT leaks into the telemetry (privacy) — only a query hash
    assert "near1" not in line and "junk1" not in line and "substantive query" not in line


def test_gate_b_saturated_score_does_not_save_low_cosine(monkeypatch, tmp_path):
    p = _provider(monkeypatch, tmp_path)
    _write_floor_cfg(tmp_path, {"enabled": True, "min_cosine": 0.10})
    low_cos = _unit([0.02, 0.999, 0.0])  # cosine ~0.02, well below the 0.10 floor
    monkeypatch.setattr(p, "_dedup_embed", lambda texts, *, timeout=15: [_Q, low_cos])
    results = [{"memory": "off-topic but scored 1.0", "score": 1.0}]
    kept, outcome = p._apply_gate_b_cosine("real query", results, budget_s=3.0)
    assert kept == []
    assert outcome == "ran_kept_0_of_1"


def test_gate_b_fail_open_on_embed_none(monkeypatch, tmp_path):
    """INV-2/INV-6: embed failure → full un-floored passthrough + failed_open."""
    p = _provider(monkeypatch, tmp_path)
    _write_floor_cfg(tmp_path, {"enabled": True, "min_cosine": 0.10})
    monkeypatch.setattr(p, "_dedup_embed", lambda texts, *, timeout=15: None)
    results = _results("a", "b", "c")
    kept, outcome = p._apply_gate_b_cosine("real query", results, budget_s=3.0)
    assert kept == results
    assert outcome == "failed_open"


def test_gate_b_fail_open_on_short_vec(monkeypatch, tmp_path):
    p = _provider(monkeypatch, tmp_path)
    _write_floor_cfg(tmp_path, {"enabled": True, "min_cosine": 0.10})
    monkeypatch.setattr(p, "_dedup_embed", lambda texts, *, timeout=15: [_Q, _NEAR])  # need 4
    results = _results("a", "b", "c")
    kept, outcome = p._apply_gate_b_cosine("real query", results, budget_s=3.0)
    assert kept == results
    assert outcome == "failed_open"


def test_gate_b_no_budget_fails_open_without_embed(monkeypatch, tmp_path):
    """INV-5: below 1s budget → fail open fast, never issue a doomed embed call."""
    p = _provider(monkeypatch, tmp_path)
    _write_floor_cfg(tmp_path, {"enabled": True, "min_cosine": 0.10})
    called = {"embed": False}

    def _spy(texts, *, timeout=15):
        called["embed"] = True
        return [_Q, _NEAR]
    monkeypatch.setattr(p, "_dedup_embed", _spy)
    results = _results("a", "b")
    kept, outcome = p._apply_gate_b_cosine("real query", results, budget_s=0.3)
    assert called["embed"] is False
    assert kept == results
    assert outcome == "failed_open"


def test_gate_b_disabled_passthrough_no_embed(monkeypatch, tmp_path):
    p = _provider(monkeypatch, tmp_path)
    _write_floor_cfg(tmp_path, {"enabled": False})
    called = {"embed": False}

    def _spy(texts, *, timeout=15):
        called["embed"] = True
        return [_Q, _ORTHO]
    monkeypatch.setattr(p, "_dedup_embed", _spy)
    results = _results("a", "b")
    kept, outcome = p._apply_gate_b_cosine("real query", results, budget_s=3.0)
    assert called["embed"] is False
    assert kept == results
    assert outcome == "disabled"


def test_gate_b_exact_token_query_bypasses(monkeypatch, tmp_path):
    """D-5: exact-token queries bypass Gate B (cosine unreliable for that class)."""
    p = _provider(monkeypatch, tmp_path)
    _write_floor_cfg(tmp_path, {"enabled": True, "min_cosine": 0.10})
    called = {"embed": False}

    def _spy(texts, *, timeout=15):
        called["embed"] = True
        return [_Q, _ORTHO]
    monkeypatch.setattr(p, "_dedup_embed", _spy)
    results = _results("the box is 192.168.1.208", "unrelated")
    kept, outcome = p._apply_gate_b_cosine("192.168.1.208", results, budget_s=3.0)
    assert called["embed"] is False
    assert kept == results
    assert outcome == "disabled"


def test_gate_b_unforeseen_exception_fails_open(monkeypatch, tmp_path):
    p = _provider(monkeypatch, tmp_path)
    _write_floor_cfg(tmp_path, {"enabled": True, "min_cosine": 0.10})

    def _boom(texts, *, timeout=15):
        raise RuntimeError("kaboom")
    monkeypatch.setattr(p, "_dedup_embed", _boom)
    results = _results("a", "b")
    kept, outcome = p._apply_gate_b_cosine("real query", results, budget_s=3.0)
    assert kept == results
    assert outcome == "failed_open"


def test_gate_b_drain_all_clears_prefetch_result(monkeypatch, tmp_path):
    """Greptile #212 P2: when Gate B drops EVERY candidate on a substantive query, the
    worker must EXPLICITLY set _prefetch_result="" — never leave a prior query's stale
    block behind (the 'inject nothing' contract must be enforced, not assumed pre-cleared)."""
    p = _provider(monkeypatch, tmp_path)
    _write_floor_cfg(tmp_path, {"enabled": True, "min_cosine": 0.10})
    # Simulate a prior query having left a stale block resident.
    p._prefetch_result = "- STALE memory from an earlier query"
    # A substantive query whose candidates all fall below the floor.
    ortho = _unit([0.0, 1.0, 0.0])
    monkeypatch.setattr(p, "_get_client", lambda: type("C", (), {
        "search": lambda self, **kw: {"results": [{"memory": "off-topic", "score": 1.0}]}})())
    monkeypatch.setattr(p, "_drop_forgotten", lambda x: x)
    monkeypatch.setattr(p, "_unwrap_results", lambda x: x.get("results", x))
    monkeypatch.setattr(p, "_read_filters", lambda: {})
    monkeypatch.setattr(p, "_dedup_embed", lambda texts, *, timeout=15: [_Q, ortho])
    # Drive one real prefetch worker cycle.
    p.queue_prefetch("what temperature is the guest room", session_id="s")
    import time as _t
    for _ in range(50):
        fut = getattr(p, "_prefetch_future", None)
        if fut and fut.done():
            break
        _t.sleep(0.05)
    # The stale block MUST have been cleared (drain-all → inject nothing).
    assert p._prefetch_result == "", f"stale block leaked: {p._prefetch_result!r}"


# ---------------------------------------------------------------------------
# Config: thresholds, clamps, fail-safes
# ---------------------------------------------------------------------------

def test_floor_default_on_when_unset(monkeypatch, tmp_path):
    p = _provider(monkeypatch, tmp_path)
    assert p._prefetch_floor_enabled() is True
    assert p._prefetch_floor_cosine() == pytest.approx(0.10)
    assert p._prefetch_min_content_tokens() == 1


def test_floor_cosine_clamped_out_of_range(monkeypatch, tmp_path):
    """D-7: an impossible threshold clamps to 0.95 / 0.0."""
    p = _provider(monkeypatch, tmp_path)
    _write_floor_cfg(tmp_path, {"enabled": True, "min_cosine": 2.0})
    assert p._prefetch_floor_cosine() == pytest.approx(0.95)
    _write_floor_cfg(tmp_path, {"enabled": True, "min_cosine": -1.0})
    assert p._prefetch_floor_cosine() == pytest.approx(0.0)


def test_floor_garbage_cosine_falls_to_default(monkeypatch, tmp_path):
    p = _provider(monkeypatch, tmp_path)
    _write_floor_cfg(tmp_path, {"enabled": True, "min_cosine": "abc"})
    assert p._prefetch_floor_cosine() == pytest.approx(0.10)


def test_floor_partial_json_failsafe(monkeypatch, tmp_path):
    """INV-3 extends to PARSE errors: a truncated mem0.json → floor ENABLED at defaults."""
    p = _provider(monkeypatch, tmp_path)
    (tmp_path / "mem0.json").write_text('{"prefetch_relevance_floor": {"enabled": tr')  # truncated
    assert p._prefetch_floor_enabled() is True
    assert p._prefetch_floor_cosine() == pytest.approx(0.10)
    assert p._prefetch_min_content_tokens() == 1


def test_embed_timeout_derivation(monkeypatch, tmp_path):
    """INV-5: timeout = clamp(budget, 1s, 3s)."""
    p = _provider(monkeypatch, tmp_path)
    assert p._prefetch_embed_timeout(5.0) == pytest.approx(3.0)
    assert p._prefetch_embed_timeout(2.0) == pytest.approx(2.0)
    assert p._prefetch_embed_timeout(0.4) == pytest.approx(1.0)
    assert p._prefetch_embed_timeout("x") == pytest.approx(3.0)  # type: ignore[arg-type]


def test_query_hash_stable_and_short(monkeypatch, tmp_path):
    p = _provider(monkeypatch, tmp_path)
    h1 = p._prefetch_query_hash("yes please")
    h2 = p._prefetch_query_hash("yes please")
    assert h1 == h2 and len(h1) == 10
    assert p._prefetch_query_hash(None) == "-" or len(p._prefetch_query_hash(None)) == 10


# ---------------------------------------------------------------------------
# L2 — rerank-score gate (spec 2026-07-07): the data-proven primary lever.
# The server returns a per-row `rerank_score` (cross-encoder logit) that SEPARATES
# on-topic (+2..+5) from off-topic junk (-6..-11) where score/cosine cannot.
# ---------------------------------------------------------------------------

def _write_cfg_block(tmp_path, key, block):
    cfg_path = tmp_path / "mem0.json"
    existing = {}
    if cfg_path.exists():
        existing = json.loads(cfg_path.read_text())
    existing[key] = block
    cfg_path.write_text(json.dumps(existing))


def _rr_results(*pairs):
    """(memory, rerank_score) pairs → result rows with a saturated score + the rerank_score."""
    return [{"memory": m, "score": 1.0, "rerank_score": rs} for m, rs in pairs]


def test_rerank_gate_default_off_is_inert(monkeypatch, tmp_path):
    """Ships OFF: with no config block, the gate returns everything untouched."""
    p = _provider(monkeypatch, tmp_path)
    results = _rr_results(("on", 5.0), ("junk", -9.0))
    kept, outcome = p._apply_rerank_gate("substantive query", results)
    assert kept == results
    assert outcome == "rr_disabled"


def test_rerank_gate_drops_negative_keeps_positive(monkeypatch, tmp_path):
    """The core behavior: at min_rerank=0.0, keep +score rows, drop the deeply-negative junk
    (the weather→Clanker case). Off-topic turn → 0 injected."""
    p = _provider(monkeypatch, tmp_path)
    _write_cfg_block(tmp_path, "prefetch_rerank_gate", {"enabled": True, "min_rerank": 0.0})
    # on-topic: 2 positive, 3 negative → keep 2
    on = _rr_results(("dns1", 5.36), ("dns2", 2.03), ("t1", -1.2), ("t2", -2.5), ("t3", -2.8))
    kept, outcome = p._apply_rerank_gate("what is my home DNS", on)
    assert [r["memory"] for r in kept] == ["dns1", "dns2"]
    assert outcome == "rr_kept_2_of_5"
    # off-topic: all deeply negative → keep 0 (the junk block is eliminated)
    off = _rr_results(("j1", -7.4), ("j2", -11.2), ("j3", -11.2), ("j4", -11.2), ("j5", -11.3))
    kept2, outcome2 = p._apply_rerank_gate("the weather is nice today", off)
    assert kept2 == []
    assert outcome2 == "rr_kept_0_of_5"


def test_rerank_gate_threshold_configurable(monkeypatch, tmp_path):
    p = _provider(monkeypatch, tmp_path)
    _write_cfg_block(tmp_path, "prefetch_rerank_gate", {"enabled": True, "min_rerank": 3.0})
    results = _rr_results(("a", 5.0), ("b", 2.5), ("c", 3.0))
    kept, _ = p._apply_rerank_gate("q", results)
    assert [r["memory"] for r in kept] == ["a", "c"]  # >= 3.0


def test_rerank_gate_missing_score_fails_open(monkeypatch, tmp_path):
    """If ANY row lacks a numeric rerank_score (rerank off / server variant), keep ALL —
    never drop a candidate to a missing field."""
    p = _provider(monkeypatch, tmp_path)
    _write_cfg_block(tmp_path, "prefetch_rerank_gate", {"enabled": True, "min_rerank": 0.0})
    results = [{"memory": "a", "score": 1.0, "rerank_score": 5.0},
               {"memory": "b", "score": 1.0}]  # no rerank_score
    kept, outcome = p._apply_rerank_gate("q", results)
    assert kept == results
    assert outcome == "rr_failed_open"


def test_rerank_gate_exact_token_query_bypasses(monkeypatch, tmp_path):
    """IP/email/port queries bypass — the cross-encoder is unreliable for exact-identifier
    lookups (same carve-out as Gate B)."""
    p = _provider(monkeypatch, tmp_path)
    _write_cfg_block(tmp_path, "prefetch_rerank_gate", {"enabled": True, "min_rerank": 0.0})
    results = _rr_results(("hit", -5.0))  # would be dropped if the gate ran
    kept, outcome = p._apply_rerank_gate("192.168.1.208", results)
    assert kept == results
    assert outcome == "rr_bypass_exact"


def test_rerank_gate_telemetry_logs_distribution(monkeypatch, tmp_path, caplog):
    import logging
    p = _provider(monkeypatch, tmp_path)
    _write_cfg_block(tmp_path, "prefetch_rerank_gate", {"enabled": True, "min_rerank": 0.0})
    results = _rr_results(("a", 5.36), ("b", -9.0))
    with caplog.at_level(logging.INFO):
        p._apply_rerank_gate("substantive query", results)
    line = next((r.getMessage() for r in caplog.records if "prefetch_rerank outcome=kept" in r.getMessage()), "")
    assert line
    for field in ("rr_max=", "rr_min=", "rr=", "min="):
        assert field in line, f"missing {field}: {line}"
    # no memory text leaks (privacy) — only scores + query hash
    assert "substantive query" not in line and " a " not in line


def test_rerank_gap_default_off_is_inert(monkeypatch, tmp_path):
    p = _provider(monkeypatch, tmp_path)
    results = _rr_results(("a", 5.0), ("b", 2.0), ("c", -1.0))
    kept, outcome = p._apply_rerank_gap("q", results)
    assert kept == results and outcome == "gap_disabled"


def test_rerank_gap_trims_tail_beyond_gap(monkeypatch, tmp_path):
    """1 great + a tail: keep only candidates within max_gap of the top score."""
    p = _provider(monkeypatch, tmp_path)
    _write_cfg_block(tmp_path, "prefetch_rerank_gap", {"enabled": True, "max_gap": 3.0})
    results = _rr_results(("top", 5.0), ("near", 2.5), ("far", 1.0), ("tail", -2.0))
    kept, outcome = p._apply_rerank_gap("q", results)
    # top=5.0; keep >= 2.0 (5.0-3.0): top(5.0), near(2.5). far(1.0) and tail(-2.0) trimmed.
    assert [r["memory"] for r in kept] == ["top", "near"]
    assert outcome == "gap_kept_2_of_4"


def test_rerank_gap_missing_score_fails_open(monkeypatch, tmp_path):
    p = _provider(monkeypatch, tmp_path)
    _write_cfg_block(tmp_path, "prefetch_rerank_gap", {"enabled": True, "max_gap": 3.0})
    results = [{"memory": "a", "score": 1.0, "rerank_score": 5.0}, {"memory": "b", "score": 1.0}]
    kept, outcome = p._apply_rerank_gap("q", results)
    assert kept == results and outcome == "gap_failed_open"


def test_rerank_gap_negative_config_clamped_not_fail_closed(monkeypatch, tmp_path):
    """A config typo like max_gap:-1 must NOT silently clear all L2 survivors.
    Negative gap is clamped to the default (6.0), keeping recall open."""
    p = _provider(monkeypatch, tmp_path)
    _write_cfg_block(tmp_path, "prefetch_rerank_gap", {"enabled": True, "max_gap": -1.0})
    results = _rr_results(("top", 5.0), ("near", 2.5), ("far", 1.0))
    kept, outcome = p._apply_rerank_gap("q", results)
    # clamped to 6.0 → top=5.0, keep >= -1.0: all three survive, nothing wrongly dropped.
    assert [r["memory"] for r in kept] == ["top", "near", "far"]
    assert outcome == "gap_kept_3_of_3"
