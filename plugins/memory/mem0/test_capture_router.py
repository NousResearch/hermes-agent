"""Tests for the Arm-B two-pass capture router (Phase 2.5).

Run: PYTHONPATH=<worktree>/plugins/memory/mem0 pytest plugins/memory/mem0/test_capture_router.py -q

Covers:
  - deterministic class dispatch (preference/ops_state -> mem0 dest; world_entity/event -> staged)
  - the dedup step (world-pass candidate that duplicates a prefs-pass fact is dropped — the leak fix)
  - the primary/fallback provider path (codex primary; gemini fallback on error/timeout)
  - flag-OFF inertness in the drain worker: router=None -> byte-identical behavior, mem0 path untouched

All extraction is injected via a fake HTTP/auth fn — no network, no 1Password.
"""
import json
import os

import pytest

import capture_router as cr
from capture_router import (
    CaptureRouter, BridgeExtractor, dedup_world_against_prefs, parse_candidates,
    build_router_from_config, correction_marker_match, facts_contradict,
)
from capture_queue import CaptureQueue, idem_key
from capture_drain import CaptureDrainWorker
import capture_scrub as scrub


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

def _resp(candidates):
    """Build a fake OpenAI-compatible chat-completions JSON string with a candidates payload."""
    content = json.dumps({"candidates": candidates})
    return json.dumps({
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    })


class FakeHTTP:
    """Deterministic fake for BridgeExtractor's http_fn. Routes by URL to a canned response, and can
    be told to FAIL the primary (codex) URL to exercise the fallback path."""
    def __init__(self, *, prefs_cands=None, world_cands=None, fail_primary=False, fail_all=False):
        self.prefs_cands = prefs_cands or []
        self.world_cands = world_cands or []
        self.fail_primary = fail_primary
        self.fail_all = fail_all
        self.calls = []

    def __call__(self, url, body, headers, timeout):
        self.calls.append(url)
        if self.fail_all:
            raise TimeoutError("simulated bridge timeout")
        is_primary = "18812" in url
        if is_primary and self.fail_primary:
            raise TimeoutError("simulated codex-bridge timeout")
        # decide prefs vs world by the system prompt embedded in the body
        payload = json.loads(body)
        sys_prompt = payload["messages"][0]["content"]
        if "preference/ops" in sys_prompt or "preference|ops_state" in sys_prompt:
            return _resp(self.prefs_cands)
        return _resp(self.world_cands)


def make_router(tmp_path, http, *, staging_mode=True, **kw):
    ext = BridgeExtractor(http_fn=http, auth_fn=lambda ref: "fake-secret")
    return CaptureRouter(
        extractor=ext,
        prefs_prompt="preference|ops_state dedicated extraction prompt",
        world_prompt="world_entity|event dedicated extraction prompt",
        staging_dir=str(tmp_path / "staged"),
        brain_inbox_dir=str(tmp_path / "inbox"),
        staging_mode=staging_mode,
        **kw,
    )


# ---------------------------------------------------------------------------
# parse_candidates
# ---------------------------------------------------------------------------

def test_parse_candidates_plain_and_fenced():
    assert parse_candidates('{"candidates": [{"content": "x", "class": "preference"}]}')[0]["content"] == "x"
    fenced = "```json\n{\"candidates\": [{\"content\": \"y\", \"class\": \"event\"}]}\n```"
    assert parse_candidates(fenced)[0]["class"] == "event"
    assert parse_candidates("not json at all") is None
    assert parse_candidates('{"candidates": []}') == []


# ---------------------------------------------------------------------------
# dedup step (the leak fix)
# ---------------------------------------------------------------------------

def test_dedup_drops_world_candidate_matching_a_prefs_fact():
    prefs = [{"content": "User runs QMD on the Mac Studio at 192.168.1.216", "class": "ops_state"}]
    world = [
        {"content": "QMD runs on the Mac Studio at 192.168.1.216", "class": "world_entity"},  # dup of prefs
        {"content": "gbrain uses PGLite by default and supports Postgres pgvector", "class": "world_entity"},
    ]
    kept, dropped = dedup_world_against_prefs(world, prefs)
    assert len(dropped) == 1 and "QMD runs on the Mac Studio" in dropped[0]["content"]
    assert len(kept) == 1 and "gbrain" in kept[0]["content"]


def test_dedup_keeps_all_when_no_prefs_overlap():
    prefs = [{"content": "User prefers concise replies", "class": "preference"}]
    world = [{"content": "Anthropic released a new model", "class": "world_entity"}]
    kept, dropped = dedup_world_against_prefs(world, prefs)
    assert len(kept) == 1 and dropped == []


# ---------------------------------------------------------------------------
# correction signals (RC1): markers + retrieved-fact contradiction
# ---------------------------------------------------------------------------

def test_correction_marker_defaults_and_config_override():
    assert correction_marker_match("Actually, ACE-AI is 192.168.1.177", "ok")
    assert correction_marker_match("No, ACE-AI is 192.168.1.177", "ok")
    assert correction_marker_match("it's ACE-AI not ACE-MEDIA", "ok")
    # Marker list is DATA: a mem0_capture_router.correction_markers override controls detection.
    assert correction_marker_match("FIXFACT: ACE-AI is 192.168.1.177", "ok", [r"\bFIXFACT:"])
    assert correction_marker_match("Actually, ACE-AI is 192.168.1.177", "ok", [r"\bFIXFACT:"]) is None


def test_fact_contradiction_detects_same_subject_different_ip():
    assert facts_contradict(
        "ACE-AI runs at 192.168.1.177",
        "ACE-AI runs at 192.168.1.176",
    ) is True
    assert facts_contradict(
        "ACE-AI runs at 192.168.1.177",
        "ACE-MEDIA runs at 192.168.1.176",
    ) is False


def test_fact_contradiction_boolean_words_do_not_treat_on_off_as_bare_booleans():
    assert facts_contradict(
        "ACE-AI monitoring report was on Tuesday",
        "ACE-AI monitoring report was off Tuesday",
    ) is False
    assert facts_contradict(
        "ACE-AI capture router is enabled",
        "ACE-AI capture router is disabled",
    ) is True


def test_router_flags_contradiction_against_existing_lookup(tmp_path):
    http = FakeHTTP(
        prefs_cands=[],
        world_cands=[{"content": "ACE-AI runs at 192.168.1.177", "class": "world_entity"}],
    )
    router = make_router(
        tmp_path, http,
        existing_fact_lookup_fn=lambda q: ["ACE-AI runs at 192.168.1.176"],
    )
    res = router.route_turn("ACE-AI runs at 192.168.1.177", "ok", turn_id="t-corr", session="s")
    assert res["correction_detected"] is True
    assert any(s["type"] == "contradiction" for s in res["correction_signals"])


def test_route_turn_stage_false_defers_world_write_until_stage_result(tmp_path):
    http = FakeHTTP(
        prefs_cands=[],
        world_cands=[{"content": "gbrain uses PGLite", "class": "world_entity"}],
    )
    router = make_router(tmp_path, http)
    res = router.route_turn("u", "a", turn_id="t-stage", session="s", stage=False)
    assert res["world_facts"]
    assert "staged_path" not in res
    assert list(tmp_path.rglob("*.md")) == []
    router.stage_route_result(res)
    assert os.path.exists(res["staged_path"])


class RaisingExtractor(BridgeExtractor):
    def extract(self, *a, **k):
        raise RuntimeError("extract blew up")


def test_marker_correction_stat_increments_even_when_extraction_fails(tmp_path):
    router = CaptureRouter(
        extractor=RaisingExtractor(),
        prefs_prompt="prefs",
        world_prompt="world",
        staging_dir=str(tmp_path),
    )
    res = router.route_turn("Actually, ACE-AI changed", "ok", turn_id="t-err", session="s")
    assert res["correction_detected"] is True
    assert res["error"]
    assert router.stats["corrections_detected"] == 1


# ---------------------------------------------------------------------------
# deterministic class dispatch
# ---------------------------------------------------------------------------

def test_router_dispatch_prefs_to_mem0_world_to_staged(tmp_path):
    http = FakeHTTP(
        prefs_cands=[{"content": "User prefers dark mode", "class": "preference", "confidence": 0.9},
                     {"content": "User's DNS is AdGuard at 192.168.1.208", "class": "ops_state", "confidence": 0.8}],
        world_cands=[{"content": "gbrain uses PGLite by default", "class": "world_entity", "confidence": 0.7},
                     {"content": "Team decided to migrate to Postgres next quarter", "class": "event", "confidence": 0.6}],
    )
    router = make_router(tmp_path, http)
    res = router.route_turn("u", "a", turn_id="t001", session="sess1", ts="2026-07-08T00:00:00Z")

    # prefs/ops_state facts are surfaced as "mem0" destination (written by the unchanged add path — router does not re-write)
    assert len(res["prefs_facts"]) == 2
    assert {c["class"] for c in res["prefs_facts"]} == {"preference", "ops_state"}
    # world/event facts staged to disk
    assert len(res["world_facts"]) == 2
    assert res["destination"] == "staging"
    staged = res["staged_path"]
    assert os.path.exists(staged)
    body = open(staged, encoding="utf-8").read()
    assert body.startswith("---")
    assert "source_turn: t001" in body
    assert "session: sess1" in body
    assert "world_entity" in body and "event" in body
    # NOT written to the brain inbox while staging
    assert not os.path.exists(str(tmp_path / "inbox"))


def test_router_drops_out_of_domain_class_labels(tmp_path):
    # a prefs pass that (wrongly) emits a world_entity, and a world pass that emits a preference:
    # the deterministic class filter drops the cross-domain leakage on each side.
    http = FakeHTTP(
        prefs_cands=[{"content": "User prefers X", "class": "preference"},
                     {"content": "some world entity leaked in prefs pass", "class": "world_entity"}],
        world_cands=[{"content": "a real world entity", "class": "world_entity"},
                     {"content": "a preference leaked into world pass", "class": "preference"}],
    )
    router = make_router(tmp_path, http)
    res = router.route_turn("u", "a", turn_id="t002", session="s")
    assert [c["class"] for c in res["prefs_facts"]] == ["preference"]
    assert [c["class"] for c in res["world_facts"]] == ["world_entity"]


def test_router_staging_mode_off_targets_brain_inbox(tmp_path):
    http = FakeHTTP(
        prefs_cands=[],
        world_cands=[{"content": "gbrain uses PGLite", "class": "world_entity"}],
    )
    router = make_router(tmp_path, http, staging_mode=False)
    res = router.route_turn("u", "a", turn_id="t003", session="s")
    assert res["destination"] == "brain-inbox"
    assert res["staged_path"].startswith(str(tmp_path / "inbox"))
    assert os.path.exists(res["staged_path"])


def test_confidence_floor_filters_low_confidence(tmp_path):
    http = FakeHTTP(
        prefs_cands=[{"content": "high", "class": "preference", "confidence": 0.9},
                     {"content": "low", "class": "preference", "confidence": 0.1}],
        world_cands=[],
    )
    router = make_router(tmp_path, http, confidence_floor=0.5)
    res = router.route_turn("u", "a", turn_id="t004", session="s")
    assert [c["content"] for c in res["prefs_facts"]] == ["high"]


# ---------------------------------------------------------------------------
# primary / fallback provider path
# ---------------------------------------------------------------------------

def test_extractor_uses_codex_primary():
    http = FakeHTTP(prefs_cands=[{"content": "x", "class": "preference"}])
    ext = BridgeExtractor(http_fn=http, auth_fn=lambda ref: "s")
    out = ext.extract("preference|ops_state prompt", "u", "a")
    assert out["provider"] == "codex-bridge"
    assert "18812" in http.calls[-1]
    assert out["candidates"][0]["content"] == "x"


def test_extractor_falls_back_to_gemini_on_primary_error():
    http = FakeHTTP(prefs_cands=[{"content": "x", "class": "preference"}], fail_primary=True)
    ext = BridgeExtractor(http_fn=http, auth_fn=lambda ref: "s")
    out = ext.extract("preference|ops_state prompt", "u", "a")
    assert out["provider"] == "gemini-bridge"
    assert out["primary_error"]
    # both URLs were attempted, gemini (18813) last
    assert any("18812" in u for u in http.calls) and "18813" in http.calls[-1]


def test_extractor_soft_error_when_both_fail():
    http = FakeHTTP(fail_all=True)
    ext = BridgeExtractor(http_fn=http, auth_fn=lambda ref: "s")
    out = ext.extract("world prompt", "u", "a")
    assert out["provider"] == "none"
    assert out["candidates"] == [] and out["error"]


def test_router_records_fallback_and_soft_errors(tmp_path):
    http = FakeHTTP(prefs_cands=[{"content": "p", "class": "preference"}],
                    world_cands=[{"content": "w", "class": "world_entity"}], fail_primary=True)
    router = make_router(tmp_path, http)
    res = router.route_turn("u", "a", turn_id="t005", session="s")
    # both passes fell back to gemini
    assert res["providers"] == {"prefs": "gemini-bridge", "world": "gemini-bridge"}
    assert router.stats["fallback_passes"] == 2


# ---------------------------------------------------------------------------
# build_router_from_config: flag gating
# ---------------------------------------------------------------------------

def test_build_router_none_when_flag_absent_or_off():
    assert build_router_from_config({}) is None
    assert build_router_from_config({"mem0_capture_router": {"enabled": False}}) is None


def test_build_router_constructs_when_enabled(tmp_path):
    cfg = {"mem0_capture_router": {"enabled": True, "staging_dir": str(tmp_path / "s"),
                                   "staging_mode": True}}
    # will try to load prompt assets from the plugin dir; that's fine — assets ship with the plugin.
    r = build_router_from_config(cfg)
    assert isinstance(r, CaptureRouter)
    assert r._staging_mode is True


def test_build_router_threads_correction_marker_config(tmp_path):
    cfg = {"mem0_capture_router": {
        "enabled": True,
        "staging_dir": str(tmp_path / "s"),
        "correction_markers": [r"\bFIXFACT:"],
    }}
    r = build_router_from_config(cfg)
    assert isinstance(r, CaptureRouter)
    assert r.correction_marker("Actually, ACE-AI changed", "ok") is None
    assert r.correction_marker("FIXFACT: ACE-AI changed", "ok") == r"\bFIXFACT:"


def test_build_router_wires_bridge_knobs(tmp_path):
    """primary/fallback URLs + secret refs are config knobs, not hardcoded (Greptile #250 P1)."""
    cfg = {"mem0_capture_router": {
        "enabled": True, "staging_dir": str(tmp_path / "s"),
        "primary_url": "http://10.0.0.9:1111/v1/chat/completions",
        "fallback_url": "http://10.0.0.10:2222/v1/chat/completions",
        "primary_secret_ref": "op://V/p/secret",
        "fallback_secret_ref": "op://V/f/secret",
    }}
    r = build_router_from_config(cfg)
    ex = r._extractor
    assert ex._primary_url == "http://10.0.0.9:1111/v1/chat/completions"
    assert ex._fallback_url == "http://10.0.0.10:2222/v1/chat/completions"
    assert ex._primary_ref == "op://V/p/secret"
    assert ex._fallback_ref == "op://V/f/secret"


def test_secret_cache_ttl_and_single_mint():
    """TTL'd secret cache + per-ref lock: concurrent first calls mint ONCE; expiry re-mints
    (Greptile #250 P2 x2)."""
    import threading as _t
    mints = []

    def fake_auth(ref):
        mints.append(ref)
        return f"tok-{len(mints)}"

    ex = BridgeExtractor(auth_fn=fake_auth)
    # concurrent first use -> exactly one mint
    out = []
    threads = [_t.Thread(target=lambda: out.append(ex._secret("op://x"))) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(mints) == 1 and set(out) == {"tok-1"}
    # expire the entry -> next call re-mints
    tok, _ts = ex._secret_cache["op://x"]
    ex._secret_cache["op://x"] = (tok, _ts - ex._secret_ttl_s - 1)
    assert ex._secret("op://x") == "tok-2"
    assert len(mints) == 2
    # invalidate_secret -> re-mint on demand (rotation heal path)
    ex.invalidate_secret("op://x")
    assert ex._secret("op://x") == "tok-3"


def test_auth_shaped_error_refreshes_secret_and_retries():
    """A 401 from the bridge invalidates the cached secret and retries once with a fresh one."""
    import urllib.error as _ue
    mints = []
    calls = []

    def fake_auth(ref):
        mints.append(ref)
        return f"tok-{len(mints)}"

    def fake_http(url, body, headers, timeout):
        calls.append(headers["Authorization"])
        if headers["Authorization"] == "Bearer tok-1":
            raise _ue.HTTPError(url, 401, "unauthorized", None, None)
        inner = json.dumps({"candidates": []})
        return json.dumps({"choices": [{"message": {"content": inner}}], "usage": {}})

    ex = BridgeExtractor(auth_fn=fake_auth, http_fn=fake_http)
    res = ex.extract("sys", "u", "a")
    assert res["provider"] == "codex-bridge"      # healed on primary, no fallback needed
    assert res["candidates"] == []
    assert calls == ["Bearer tok-1", "Bearer tok-2"]
    assert len(mints) == 2


# ---------------------------------------------------------------------------
# FLAG-OFF INERTNESS in the drain worker (byte-identical behavior)
# ---------------------------------------------------------------------------

class FakeStore:
    def __init__(self):
        self.rows = []
        self._id = 0
        self.add_calls = 0

    def add(self, messages, kwargs):
        self.add_calls += 1
        idem = (kwargs.get("metadata") or {}).get("capture_idem", "")
        self._id += 1
        self.rows.append({"id": f"m{self._id}", "memory": messages[0]["content"], "capture_idem": idem})
        return 1

    def recall_idem(self, key):
        return sum(1 for r in self.rows if r["capture_idem"] == key)

    def get_written(self, key):
        return [r for r in self.rows if r["capture_idem"] == key]

    def forget(self, mid):
        self.rows = [r for r in self.rows if r["id"] != mid]


class GateAwareFakeStore(FakeStore):
    """Fake mem0 add where the salience gate drops the row, but an ungated correction add writes it."""
    def __init__(self):
        super().__init__()
        self.kwargs_seen = []

    def add(self, messages, kwargs):
        self.add_calls += 1
        self.kwargs_seen.append(dict(kwargs))
        if kwargs.get("prompt"):
            return 0  # below the salience threshold: gate says "drop"
        idem = (kwargs.get("metadata") or {}).get("capture_idem", "")
        self._id += 1
        self.rows.append({"id": f"m{self._id}", "memory": messages[0]["content"], "capture_idem": idem})
        return 1


def _worker(q, store, router=None):
    return CaptureDrainWorker(
        q, add_fn=store.add, recall_idem_fn=store.recall_idem,
        scrub_fn=lambda facts: scrub.filter_facts(facts),
        forget_fn=store.forget, get_written_fn=store.get_written,
        gate="GATE_V3", model="gpt-5.4-mini", write_filters={"user_id": "ace"},
        max_attempts=3, backoff_base_s=1.0, router=router)


def test_drain_router_none_is_inert(tmp_path):
    """router=None (flag OFF): the drain worker completes the turn exactly as today — mem0 add ran,
    row marked done, and NOTHING was staged to disk anywhere."""
    q = CaptureQueue(str(tmp_path / "cq.db"))
    store = FakeStore()
    w = _worker(q, store, router=None)
    k = idem_key("s", 1, "User prefers dark mode.", "ok")
    q.enqueue(k, {"user": "User prefers dark mode.", "assistant": "ok", "session_id": "s"})
    assert w.drain_once() is True
    assert q.counts()["done"] == 1
    assert store.add_calls == 1
    # no staging dir created anywhere under tmp
    staged = list(tmp_path.rglob("*.md"))
    assert staged == []


class SpyRouter:
    def __init__(self):
        self.calls = []
    def route_turn(self, user, assistant, *, turn_id, session, ts=None, stage=True):
        self.calls.append((user, assistant, turn_id, session))
        return {"error": None, "world_facts": [], "prefs_facts": []}


def test_drain_invokes_router_when_wired(tmp_path):
    """router injected (flag ON): the mem0 add still runs unchanged AND the router is invoked once
    with the turn. mem0 path is untouched when no correction signal is present."""
    q = CaptureQueue(str(tmp_path / "cq.db"))
    store = FakeStore()
    spy = SpyRouter()
    w = _worker(q, store, router=spy)
    k = idem_key("s", 1, "Alex met Maria who runs a FinOps startup.", "cool")
    q.enqueue(k, {"user": "Alex met Maria who runs a FinOps startup.", "assistant": "cool",
                  "session_id": "sess-x"})
    assert w.drain_once() is True
    assert q.counts()["done"] == 1
    assert store.add_calls == 1               # unchanged mem0 write path still ran
    assert len(spy.calls) == 1                # router invoked once
    assert spy.calls[0][3] == "sess-x"        # session threaded through


class BoomRouter:
    def route_turn(self, *a, **k):
        raise RuntimeError("router blew up")


def test_drain_router_failure_never_fails_the_turn(tmp_path):
    """A router exception must NOT requeue or fail the turn — the mem0 write is already durable."""
    q = CaptureQueue(str(tmp_path / "cq.db"))
    store = FakeStore()
    w = _worker(q, store, router=BoomRouter())
    k = idem_key("s", 1, "some turn", "ok")
    q.enqueue(k, {"user": "some turn", "assistant": "ok", "session_id": "s"})
    assert w.drain_once() is True
    assert q.counts()["done"] == 1            # turn completed despite router failure
    assert store.add_calls == 1


def test_drain_correction_marker_bypasses_salience_gate(tmp_path):
    """Acceptance C: a below-threshold correction marker omits the salience prompt, so capture fires."""
    q = CaptureQueue(str(tmp_path / "cq.db"))
    store = GateAwareFakeStore()
    router = make_router(tmp_path, FakeHTTP(prefs_cands=[], world_cands=[]))
    w = _worker(q, store, router=router)
    k = idem_key("s", 1, "Actually, ACE-AI runs at 192.168.1.177.", "ok")
    q.enqueue(k, {"user": "Actually, ACE-AI runs at 192.168.1.177.",
                  "assistant": "ok", "session_id": "s"})
    assert w.drain_once() is True
    assert q.counts()["done"] == 1
    assert len(store.rows) == 1
    assert "prompt" not in store.kwargs_seen[0]
    assert store.kwargs_seen[0]["metadata"]["capture_correction"] is True
    assert store.kwargs_seen[0]["metadata"]["capture_correction_signal"] == "marker"


def test_drain_contradiction_signal_bypasses_salience_gate(tmp_path):
    """A world fact contradicting an existing retrieved fact also bypasses the salience prompt."""
    q = CaptureQueue(str(tmp_path / "cq.db"))
    store = GateAwareFakeStore()
    router = make_router(
        tmp_path,
        FakeHTTP(
            prefs_cands=[],
            world_cands=[{"content": "ACE-AI runs at 192.168.1.177", "class": "world_entity"}],
        ),
        existing_fact_lookup_fn=lambda q: ["ACE-AI runs at 192.168.1.176"],
    )
    w = _worker(q, store, router=router)
    k = idem_key("s", 1, "ACE-AI runs at 192.168.1.177.", "ok")
    q.enqueue(k, {"user": "ACE-AI runs at 192.168.1.177.", "assistant": "ok", "session_id": "s"})
    assert w.drain_once() is True
    assert q.counts()["done"] == 1
    assert len(store.rows) == 1
    assert "prompt" not in store.kwargs_seen[0]
    assert store.kwargs_seen[0]["metadata"]["capture_correction_signal"] == "contradiction"


def test_drain_non_correction_keeps_normal_salience_path(tmp_path):
    """No deterministic correction signal => normal salience gate remains in force (false-negative fallback)."""
    q = CaptureQueue(str(tmp_path / "cq.db"))
    store = GateAwareFakeStore()
    router = make_router(tmp_path, FakeHTTP(prefs_cands=[], world_cands=[]))
    w = _worker(q, store, router=router)
    k = idem_key("s", 1, "ACE-AI has a status page.", "ok")
    q.enqueue(k, {"user": "ACE-AI has a status page.", "assistant": "ok", "session_id": "s"})
    assert w.drain_once() is True
    assert q.counts()["done"] == 1
    assert store.rows == []                 # gate dropped the below-threshold turn
    assert store.kwargs_seen[0]["prompt"] == "GATE_V3"
    assert "capture_correction" not in store.kwargs_seen[0]["metadata"]
