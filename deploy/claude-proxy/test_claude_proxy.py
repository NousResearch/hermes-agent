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


T = "acme"  # default tenant used by most cache tests


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    """Fresh cache/breaker state and small, fast breaker threshold per test."""
    cp._cache.clear()
    cp._neg_cache.clear()
    cp._classify_cache.clear()
    cp._breaker_bad.clear()
    cp._breaker_open_until = 0.0
    cp._breaker_trips = 0
    cp._tenant_meters.clear()
    cp._tenant_budgets.clear()
    cp._keys_map.clear()
    cp._keys_mtime = 0.0
    monkeypatch.setattr(cp, "CACHE_TTL", 86400)
    monkeypatch.setattr(cp, "NEG_CACHE_TTL", 60)
    monkeypatch.setattr(cp, "EMPTY_RETRIES", 1)
    monkeypatch.setattr(cp, "BREAKER_ENABLED", True)
    monkeypatch.setattr(cp, "BREAKER_THRESHOLD", 3)
    monkeypatch.setattr(cp, "BREAKER_WINDOW", 120)
    monkeypatch.setattr(cp, "BREAKER_COOLDOWN", 90)
    monkeypatch.setattr(cp, "ALLOW_ANON", True)
    monkeypatch.setattr(cp, "DAILY_TOKEN_BUDGET", 5_000_000)
    monkeypatch.setattr(cp, "BACKEND", cp.ClaudeCliBackend())
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
    cp.cache_set(T, "sys", "p", "sonnet", EMPTY)
    assert cp.cache_get(T, "sys", "p", "sonnet") is None
    assert len(cp._cache) == 0

def test_cache_set_refuses_claude_error():
    cp.cache_set(T, "sys", "p", "sonnet", CLAUDE_ERR)
    assert len(cp._cache) == 0

def test_cache_set_accepts_good():
    cp.cache_set(T, "sys", "p", "sonnet", GOOD)
    assert cp.cache_get(T, "sys", "p", "sonnet") == GOOD


# ── cache is tenant-scoped: one tenant never serves another's result ──

def test_cache_is_tenant_scoped():
    cp.cache_set("acme", "sys", "p", "sonnet", GOOD)
    assert cp.cache_get("acme", "sys", "p", "sonnet") == GOOD
    assert cp.cache_get("globex", "sys", "p", "sonnet") is None


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
    out = cp.call_claude("sys", "hello", "sonnet", tenant=T)
    assert out == GOOD
    assert cp.cache_get(T, "sys", "hello", "sonnet") == GOOD       # cached
    assert len(cp._neg_cache) == 0
    assert len(cp._breaker_bad) == 0


# ── call_claude: transient empty (empty then good) → recovered, no failover ──

def test_call_transient_empty_recovers(monkeypatch):
    seq = iter([EMPTY, GOOD])
    monkeypatch.setattr(cp, "_run_claude_once", lambda *a, **k: next(seq))
    out = cp.call_claude("sys", "hello", "sonnet", tenant=T)
    assert out == GOOD                                              # recovered via retry
    assert cp.cache_get(T, "sys", "hello", "sonnet") == GOOD
    assert len(cp._breaker_bad) == 0


# ── call_claude: persistent empty → NOT in 24h cache, in neg cache, breaker++ ──

def test_call_persistent_empty(monkeypatch):
    monkeypatch.setattr(cp, "_run_claude_once", lambda *a, **k: EMPTY)
    out = cp.call_claude("sys", "deterministic", "sonnet", tenant=T)
    assert cp._is_bad_result(out)
    assert cp.cache_get(T, "sys", "deterministic", "sonnet") is None  # NOT poisoned
    assert len(cp._neg_cache) == 1                                 # negative-cached
    assert len(cp._breaker_bad) == 1                               # breaker recorded


# ── call_claude: negative cache absorbs burst retries (no respawn) ──

def test_neg_cache_absorbs_burst(monkeypatch):
    calls = {"n": 0}
    def run(*a, **k):
        calls["n"] += 1
        return EMPTY
    monkeypatch.setattr(cp, "_run_claude_once", run)
    cp.call_claude("sys", "same", "sonnet", tenant=T)  # spawns: 1 + 1 retry = 2
    n_after_first = calls["n"]
    cp.call_claude("sys", "same", "sonnet", tenant=T)  # neg-cache hit: 0 spawns
    cp.call_claude("sys", "same", "sonnet", tenant=T)  # neg-cache hit: 0 spawns
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
        cp.call_claude("sys", f"prompt-{i}", "sonnet", tenant=T)
    assert cp._breaker_is_open()
    spawns_before = calls["n"]
    # Next distinct prompt while open → short-circuited, no claude spawn
    out = cp.call_claude("sys", "prompt-new", "sonnet", tenant=T)
    assert out.get("error") and out.get("code") == 503
    assert calls["n"] == spawns_before               # did not spawn claude while open


def test_breaker_single_prompt_does_not_trip(monkeypatch):
    monkeypatch.setattr(cp, "_run_claude_once", lambda *a, **k: EMPTY)
    # Same prompt many times → neg-cache absorbs, only 1 distinct bad recorded
    for _ in range(10):
        cp.call_claude("sys", "one-bad-prompt", "sonnet", tenant=T)
    assert not cp._breaker_is_open()
    assert len(cp._breaker_bad) == 1


# ── breaker is GLOBAL: distinct prompts across tenants still trip it ──

def test_breaker_is_global_across_tenants(monkeypatch):
    monkeypatch.setattr(cp, "_run_claude_once", lambda *a, **k: EMPTY)
    for i in range(3):                                # threshold=3, distinct prompts
        cp.call_claude("sys", f"p-{i}", "sonnet", tenant=f"tenant-{i}")
    assert cp._breaker_is_open()


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


# ── Phase 0: sandbox flags + no bypassPermissions ──

def test_sandbox_flags_strip_tools():
    flags = cp._sandbox_flags()
    assert "--tools" in flags
    # --tools is followed by an empty string (disable all tools)
    assert flags[flags.index("--tools") + 1] == ""
    assert "--disallowedTools" in flags
    for t in ["Bash", "Read", "Write", "Edit", "WebFetch"]:
        assert t in flags


def test_cli_backend_drops_bypass_and_sandboxes(monkeypatch):
    captured = {}
    def fake_run(cmd, prompt, env, timeout=300, stream_mode=False):
        captured["cmd"] = cmd
        return GOOD
    monkeypatch.setattr(cp, "_run_claude_once", fake_run)
    cp.ClaudeCliBackend().complete("sys", "hi", "sonnet", "acme")
    cmd = captured["cmd"]
    assert "bypassPermissions" not in cmd            # bypass removed (default-deny)
    assert "--tools" in cmd and "--disallowedTools" in cmd


# ── Phase 0: atomic cache save ──

def test_cache_save_is_atomic(tmp_path, monkeypatch):
    f = tmp_path / "sub" / ".cache.json"
    monkeypatch.setattr(cp, "CACHE_FILE", str(f))
    cp._cache.clear()
    cp._cache["k"] = {"response": GOOD, "ts": __import__("time").time()}
    cp._cache_save()
    assert f.exists()
    assert not (f.parent / (f.name + ".tmp")).exists()   # temp file replaced
    import json as _j
    assert "k" in _j.loads(f.read_text())


# ── Phase 0: claude bin resolution honors CLAUDE_BIN override ──

def test_resolve_claude_bin_env_override(monkeypatch):
    monkeypatch.setenv("CLAUDE_BIN", "/custom/claude")
    assert cp._resolve_claude_bin() == "/custom/claude"


# ── Phase 1: tenant resolution + auth ──

def _write_keys(tmp_path, monkeypatch, mapping):
    import json as _j
    f = tmp_path / "keys.json"
    f.write_text(_j.dumps(mapping))
    monkeypatch.setattr(cp, "KEYS_FILE", str(f))
    cp._keys_map.clear()
    cp._keys_mtime = 0.0
    return f


def test_resolve_tenant_valid_key(tmp_path, monkeypatch):
    _write_keys(tmp_path, monkeypatch, {"secret-abc": "acme"})
    tenant, ok = cp.resolve_tenant("Bearer secret-abc")
    assert tenant == "acme" and ok is True


def test_resolve_tenant_anon_allowed_by_default(tmp_path, monkeypatch):
    _write_keys(tmp_path, monkeypatch, {})
    monkeypatch.setattr(cp, "ALLOW_ANON", True)
    tenant, ok = cp.resolve_tenant(None)
    assert tenant == "anon" and ok is True            # deploy-safe default


def test_resolve_tenant_anon_denied_when_locked(tmp_path, monkeypatch):
    _write_keys(tmp_path, monkeypatch, {"k": "acme"})
    monkeypatch.setattr(cp, "ALLOW_ANON", False)
    # missing key → denied
    assert cp.resolve_tenant(None) == ("anon", False)
    # invalid key → denied
    assert cp.resolve_tenant("Bearer nope") == ("anon", False)
    # valid key → allowed even when locked
    assert cp.resolve_tenant("Bearer k") == ("acme", True)


def test_keys_reload_on_mtime_change(tmp_path, monkeypatch):
    import json as _j, os as _os, time as _t
    f = _write_keys(tmp_path, monkeypatch, {"k1": "acme"})
    assert cp.resolve_tenant("Bearer k1") == ("acme", True)
    _t.sleep(0.01)
    f.write_text(_j.dumps({"k2": "globex"}))
    _os.utime(f, None)
    assert cp.resolve_tenant("Bearer k2") == ("globex", True)


# ── Phase 4: per-tenant daily budget → 429 ──

def test_budget_exceeded_returns_429(monkeypatch):
    monkeypatch.setattr(cp, "_run_claude_once", lambda *a, **k: GOOD)
    monkeypatch.setattr(cp, "DAILY_TOKEN_BUDGET", 5)   # GOOD uses 12 tokens
    out1 = cp.call_claude("sys", "q1", "sonnet", tenant="acme")
    assert out1 == GOOD                               # first call goes through
    out2 = cp.call_claude("sys", "q2", "sonnet", tenant="acme")
    assert out2.get("error") and out2.get("code") == 429


def test_budget_is_per_tenant(monkeypatch):
    monkeypatch.setattr(cp, "_run_claude_once", lambda *a, **k: GOOD)
    monkeypatch.setattr(cp, "DAILY_TOKEN_BUDGET", 5)
    cp.call_claude("sys", "q1", "sonnet", tenant="acme")     # acme over budget
    assert cp.call_claude("sys", "q2", "sonnet", tenant="acme").get("code") == 429
    # globex is unaffected
    assert cp.call_claude("sys", "q1", "sonnet", tenant="globex") == GOOD


def test_meter_records_tokens(monkeypatch):
    monkeypatch.setattr(cp, "_run_claude_once", lambda *a, **k: GOOD)
    cp.call_claude("sys", "hello", "sonnet", tenant="acme")
    m = cp._tenant_meters["acme"]
    assert m["requests"] == 1
    assert m["input_tokens"] == 10 and m["output_tokens"] == 2


# ── Durable metering: persist + restore meters/budgets across a restart ──

def test_meters_save_load_roundtrip(tmp_path, monkeypatch):
    f = tmp_path / "sub" / "meters.json"
    monkeypatch.setattr(cp, "METERS_FILE", str(f))
    cp.meter_record("acme", 100, 40)
    cp.meter_record("acme", 10, 5)
    cp._meters_save()
    assert f.exists()
    assert not (f.parent / (f.name + ".tmp")).exists()   # atomic temp replaced
    # Simulate a restart: wipe in-memory state, reload from disk.
    cp._tenant_meters.clear()
    cp._tenant_budgets.clear()
    cp._meters_load()
    m = cp._tenant_meters["acme"]
    assert m["requests"] == 2 and m["input_tokens"] == 110 and m["output_tokens"] == 45
    # Budget survives the restart, so spend isn't silently refunded.
    assert cp._tenant_budgets["acme"]["tokens"] == 155


def test_meters_load_drops_stale_budget_day(tmp_path, monkeypatch):
    import json as _j
    f = tmp_path / "meters.json"
    f.write_text(_j.dumps({
        "tenants": {"acme": {"requests": 3, "input_tokens": 9, "output_tokens": 1}},
        "budgets": {"acme": {"day": "1999-01-01", "tokens": 999}},  # yesterday
    }))
    monkeypatch.setattr(cp, "METERS_FILE", str(f))
    cp._tenant_meters.clear()
    cp._tenant_budgets.clear()
    cp._meters_load()
    # Lifetime meters restored, but a stale budget window is NOT carried into today.
    assert cp._tenant_meters["acme"]["requests"] == 3
    assert "acme" not in cp._tenant_budgets
    assert cp.budget_exceeded("acme") is False


# ── no-cache bypass is gated to authenticated tenants ──

def test_effective_no_cache_gated_to_authenticated():
    assert cp._effective_no_cache("no-cache", "acme") is True
    assert cp._effective_no_cache("no-cache", cp.ANON_TENANT) is False   # anon can't bypass
    assert cp._effective_no_cache("", "acme") is False


# ── Request hardening: body cap (413) + anon-locked (401) over real HTTP ──

def _start_proxy(monkeypatch, **attrs):
    import threading
    for k, v in attrs.items():
        monkeypatch.setattr(cp, k, v)
    srv = cp.ThreadingHTTPServer(("127.0.0.1", 0), cp.ProxyHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def test_http_body_cap_returns_413(monkeypatch):
    import http.client
    srv, port = _start_proxy(monkeypatch, MAX_BODY_BYTES=16)
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("POST", "/v1/chat/completions", body="x" * 1000)
        resp = conn.getresponse()
        assert resp.status == 413
    finally:
        srv.shutdown()


def test_http_anon_locked_returns_401(tmp_path, monkeypatch):
    import http.client, json as _j
    cp._keys_map.clear()
    cp._keys_mtime = 0.0
    srv, port = _start_proxy(
        monkeypatch, ALLOW_ANON=False, KEYS_FILE=str(tmp_path / "absent.json"),
    )
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        body = _j.dumps({"messages": [{"role": "user", "content": "hi"}]})
        conn.request("POST", "/v1/chat/completions", body=body)   # no Authorization
        resp = conn.getresponse()
        assert resp.status == 401
    finally:
        srv.shutdown()


def test_worker_semaphore_is_bounded():
    # MAX_WORKERS > 0 installs a BoundedSemaphore guarding completion work.
    assert cp.MAX_WORKERS > 0
    assert cp._worker_sem is not None


# ── Phase 4: classifier keyword-first + caching ──

def test_classify_keyword_first_no_subprocess(monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(cp, "classify_via_haiku",
                        lambda p: called.__setitem__("n", called["n"] + 1) or "opus")
    # "implement" matches SONNET_PATTERNS → confident, no Haiku call
    assert cp.classify_model("implement the login feature") == "sonnet"
    assert called["n"] == 0


def test_classify_caches_result(monkeypatch):
    called = {"n": 0}
    def fake_haiku(p):
        called["n"] += 1
        return "opus"
    monkeypatch.setattr(cp, "classify_via_haiku", fake_haiku)
    # an ambiguous prompt (no keywords) → falls back to haiku ONCE, then cached
    amb = "zxqv frobnicate the wibble"
    assert cp.classify_model(amb) == "opus"
    assert cp.classify_model(amb) == "opus"
    assert called["n"] == 1                            # second call hit the cache


def test_classify_override_tilde(monkeypatch):
    monkeypatch.setattr(cp, "classify_via_haiku", lambda p: pytest.fail("should not call"))
    assert cp.classify_model("~haiku do a thing") == "haiku"


# ── Phase 4: Anthropic backend response → claude result shape ──

def test_anthropic_backend_shape():
    api = {
        "id": "msg_123",
        "content": [{"type": "text", "text": "Hello there"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 7, "output_tokens": 3},
    }
    shaped = cp.AnthropicApiBackend._to_claude_shape(api)
    assert not cp._is_bad_result(shaped)
    r = cp.claude_to_openai(shaped, "claude-sonnet-4-6")
    assert r["choices"][0]["message"]["content"] == "Hello there"
    assert r["usage"]["total_tokens"] == 10


def test_anthropic_backend_missing_key(monkeypatch):
    monkeypatch.setattr(cp, "ANTHROPIC_API_KEY", "")
    out = cp.AnthropicApiBackend().complete("sys", "hi", "sonnet", "acme")
    assert out.get("error") and cp._is_bad_result(out)


# ── /health endpoint ──

def test_health_returns_200_and_valid_json(monkeypatch):
    """GET /health returns 200 with the expected top-level keys."""
    import http.client, json as _j
    srv, port = _start_proxy(monkeypatch)
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/health")
        resp = conn.getresponse()
        assert resp.status == 200
        data = _j.loads(resp.read())
        # Required top-level keys — presence, not values (values are env-dependent)
        for key in ("status", "claude_bin_exists", "backend", "requests",
                     "auth", "tenants", "limits", "cache", "breaker", "deepseek"):
            assert key in data, f"Missing key: {key}"
        assert data["status"] == "ok"
    finally:
        srv.shutdown()


def test_health_deepseek_routing_disabled_when_no_key(monkeypatch):
    """When DEEPSEEK_API_KEY is empty, deepseek.routing reports 'disabled'."""
    import http.client, json as _j
    srv, port = _start_proxy(monkeypatch, DEEPSEEK_API_KEY="")
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/health")
        data = _j.loads(conn.getresponse().read())
        assert data["deepseek"]["routing"] == "disabled"
        assert data["deepseek"]["key_set"] is False
    finally:
        srv.shutdown()


def test_health_deepseek_routing_enabled_with_key_and_threshold(monkeypatch):
    """With a key AND threshold>0, deepseek.routing reports 'enabled'."""
    import http.client, json as _j
    srv, port = _start_proxy(
        monkeypatch, DEEPSEEK_API_KEY="sk-fake", DEEPSEEK_THRESHOLD=20000,
    )
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/health")
        data = _j.loads(conn.getresponse().read())
        assert data["deepseek"]["routing"] == "enabled"
        assert data["deepseek"]["key_set"] is True
    finally:
        srv.shutdown()


def test_health_breaker_open_reflected(monkeypatch):
    """After the breaker trips, /health reports breaker.open=true."""
    import http.client, json as _j
    monkeypatch.setattr(cp, "_run_claude_once", lambda *a, **k: EMPTY)
    # Force breaker open with many distinct bad prompts
    for i in range(cp.BREAKER_THRESHOLD + 2):
        cp.call_claude("sys", f"breaker-test-{i}", "sonnet", tenant="acme")
    assert cp._breaker_is_open()
    srv, port = _start_proxy(monkeypatch)
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/health")
        data = _j.loads(conn.getresponse().read())
        assert data["breaker"]["open"] is True
        assert data["breaker"]["trips"] >= 1
    finally:
        srv.shutdown()


def test_health_root_returns_service_info(monkeypatch):
    """GET / (non-health) returns service info with endpoints list."""
    import http.client, json as _j
    srv, port = _start_proxy(monkeypatch)
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/")
        resp = conn.getresponse()
        assert resp.status == 200
        data = _j.loads(resp.read())
        assert data["service"] == "Claude Code Proxy"
        assert "POST /v1/chat/completions" in data["endpoints"]
        assert "GET /health" in data["endpoints"]
    finally:
        srv.shutdown()
