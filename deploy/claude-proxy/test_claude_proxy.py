"""Unit tests for the Claude Code proxy fix (poison-cache + empty handling).

Run: python3 -m pytest test_claude_proxy.py -v
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import claude_proxy as cp  # noqa: E402


GOOD = {"type": "result", "subtype": "success", "is_error": False,
        "result": "Hello!", "stop_reason": "end_turn", "num_turns": 1,
        "duration_ms": 1200, "usage": {"input_tokens": 10, "output_tokens": 2}}
EMPTY = {"type": "result", "subtype": "success", "is_error": False,
         "result": "", "stop_reason": "end_turn"}
WHITESPACE = {**EMPTY, "result": "   \n  "}
CLAUDE_ERR = {"type": "result", "subtype": "error_during_execution",
              "is_error": True, "result": "boom"}
MAXTURNS = {"type": "result", "subtype": "error_max_turns", "is_error": False,
            "result": ""}
PROXY_ERR = {"error": True, "message": "Claude returned no output", "code": 0}


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    """Fresh cache/breaker state and small, fast breaker threshold per test."""
    cp._cache.clear()
    cp._neg_cache.clear()
    cp._breaker_bad.clear()
    cp._breaker_open_until = 0.0
    cp._breaker_trips = 0
    monkeypatch.setattr(cp, "CACHE_TTL", 86400)
    monkeypatch.setattr(cp, "NEG_CACHE_TTL", 60)
    monkeypatch.setattr(cp, "EMPTY_RETRIES", 1)
    monkeypatch.setattr(cp, "BREAKER_ENABLED", True)
    monkeypatch.setattr(cp, "BREAKER_THRESHOLD", 3)
    monkeypatch.setattr(cp, "BREAKER_WINDOW", 120)
    monkeypatch.setattr(cp, "BREAKER_COOLDOWN", 90)
    yield


# ── _is_bad_result ──

@pytest.mark.parametrize("res,bad", [
    (GOOD, False),
    (EMPTY, True),
    (WHITESPACE, True),
    (CLAUDE_ERR, True),
    (MAXTURNS, True),
    (PROXY_ERR, True),
    ({}, True),
    (None, True),
    ("notadict", True),
])
def test_is_bad_result(res, bad):
    assert cp._is_bad_result(res) is bad


# ── cache_set never stores bad results (the poison-cache bug) ──

def test_cache_set_refuses_empty():
    cp.cache_set("sys", "p", "sonnet", EMPTY)
    assert cp.cache_get("sys", "p", "sonnet") is None
    assert len(cp._cache) == 0

def test_cache_set_refuses_claude_error():
    cp.cache_set("sys", "p", "sonnet", CLAUDE_ERR)
    assert len(cp._cache) == 0

def test_cache_set_accepts_good():
    cp.cache_set("sys", "p", "sonnet", GOOD)
    assert cp.cache_get("sys", "p", "sonnet") == GOOD


# ── _cache_load drops pre-fix poisoned entries ──

def test_cache_load_skips_bad(tmp_path, monkeypatch):
    import json, time
    f = tmp_path / ".cache.json"
    now = time.time()
    f.write_text(json.dumps({
        "k_good": {"response": GOOD, "ts": now},
        "k_empty": {"response": EMPTY, "ts": now},
        "k_err": {"response": CLAUDE_ERR, "ts": now},
    }))
    monkeypatch.setattr(cp, "CACHE_FILE", str(f))
    cp._cache.clear()
    cp._cache_load()
    assert "k_good" in cp._cache
    assert "k_empty" not in cp._cache
    assert "k_err" not in cp._cache


# ── call_claude: real text → normal path, cached, breaker reset ──

def test_call_good_caches_and_resets_breaker(monkeypatch):
    monkeypatch.setattr(cp, "_run_claude_once", lambda *a, **k: GOOD)
    out = cp.call_claude("sys", "hello", "sonnet")
    assert out == GOOD
    assert cp.cache_get("sys", "hello", "sonnet") == GOOD          # cached
    assert len(cp._neg_cache) == 0
    assert len(cp._breaker_bad) == 0


# ── call_claude: transient empty (empty then good) → recovered, no failover ──

def test_call_transient_empty_recovers(monkeypatch):
    seq = iter([EMPTY, GOOD])
    monkeypatch.setattr(cp, "_run_claude_once", lambda *a, **k: next(seq))
    out = cp.call_claude("sys", "hello", "sonnet")
    assert out == GOOD                                              # recovered via retry
    assert cp.cache_get("sys", "hello", "sonnet") == GOOD
    assert len(cp._breaker_bad) == 0


# ── call_claude: persistent empty → NOT in 24h cache, in neg cache, breaker++ ──

def test_call_persistent_empty(monkeypatch):
    monkeypatch.setattr(cp, "_run_claude_once", lambda *a, **k: EMPTY)
    out = cp.call_claude("sys", "deterministic", "sonnet")
    assert cp._is_bad_result(out)
    assert cp.cache_get("sys", "deterministic", "sonnet") is None  # NOT poisoned
    assert len(cp._neg_cache) == 1                                 # negative-cached
    assert len(cp._breaker_bad) == 1                               # breaker recorded


# ── call_claude: negative cache absorbs burst retries (no respawn) ──

def test_neg_cache_absorbs_burst(monkeypatch):
    calls = {"n": 0}
    def run(*a, **k):
        calls["n"] += 1
        return EMPTY
    monkeypatch.setattr(cp, "_run_claude_once", run)
    cp.call_claude("sys", "same", "sonnet")          # spawns: 1 + 1 retry = 2
    n_after_first = calls["n"]
    cp.call_claude("sys", "same", "sonnet")          # neg-cache hit: 0 spawns
    cp.call_claude("sys", "same", "sonnet")          # neg-cache hit: 0 spawns
    assert calls["n"] == n_after_first               # burst retries did not respawn claude


# ── call_claude: breaker opens after THRESHOLD distinct bad prompts, short-circuits ──

def test_breaker_opens_and_short_circuits(monkeypatch):
    calls = {"n": 0}
    def run(*a, **k):
        calls["n"] += 1
        return EMPTY
    monkeypatch.setattr(cp, "_run_claude_once", run)
    # 3 distinct bad prompts (threshold=3) → breaker opens
    for i in range(3):
        cp.call_claude("sys", f"prompt-{i}", "sonnet")
    assert cp._breaker_is_open()
    spawns_before = calls["n"]
    # Next distinct prompt while open → short-circuited, no claude spawn
    out = cp.call_claude("sys", "prompt-new", "sonnet")
    assert out.get("error") and out.get("code") == 503
    assert calls["n"] == spawns_before               # did not spawn claude while open


def test_breaker_single_prompt_does_not_trip(monkeypatch):
    monkeypatch.setattr(cp, "_run_claude_once", lambda *a, **k: EMPTY)
    # Same prompt many times → neg-cache absorbs, only 1 distinct bad recorded
    for _ in range(10):
        cp.call_claude("sys", "one-bad-prompt", "sonnet")
    assert not cp._breaker_is_open()
    assert len(cp._breaker_bad) == 1


# ── hung subprocess → watchdog timeout → error dict ──

def test_run_once_timeout(monkeypatch):
    def boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="claude", timeout=300)
    monkeypatch.setattr(cp.subprocess, "run", boom)
    out = cp._run_claude_once(["claude"], "hi", {})
    assert out.get("error") and out.get("timeout") is True
    assert cp._is_bad_result(out)


def test_run_once_no_output(monkeypatch):
    class P:
        stdout = ""
        stderr = ""
        returncode = 0
    monkeypatch.setattr(cp.subprocess, "run", lambda *a, **k: P())
    out = cp._run_claude_once(["claude"], "hi", {})
    assert out.get("error") and cp._is_bad_result(out)


# ── claude_to_openai mapping ──

def test_openai_good():
    r = cp.claude_to_openai(GOOD, "claude-sonnet-4-6")
    assert "error" not in r
    assert r["choices"][0]["message"]["content"] == "Hello!"
    assert r["choices"][0]["finish_reason"] == "stop"

def test_openai_claude_error_envelope():
    r = cp.claude_to_openai(CLAUDE_ERR, "claude-sonnet-4-6")
    assert r.get("error") and r["error"]["type"] == "claude_error"

def test_openai_maxturns_envelope():
    r = cp.claude_to_openai(MAXTURNS, "claude-sonnet-4-6")
    assert r.get("error") and r["error"]["type"] == "claude_error"

def test_openai_proxy_error_envelope():
    r = cp.claude_to_openai(PROXY_ERR, "claude-sonnet-4-6")
    assert r.get("error") and r["error"]["type"] == "proxy_error"
