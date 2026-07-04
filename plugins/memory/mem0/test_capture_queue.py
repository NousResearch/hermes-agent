"""Tests for the mem0 auto-capture durable queue + deterministic secret scrubber (Track A-lite).

Run: PYTHONPATH=<worktree> pytest plugins/memory/mem0/test_capture_queue.py -q
Covers: INV-1 (durability), INV-5 (idempotent/exactly-once reconcile), D-10 (lease+reaper
verdict-marker precision), D-8 (idem key), INV-4 (secret scrubber corpus, NB1).
"""
import os
import tempfile
import time

import pytest

from capture_queue import CaptureQueue, idem_key, normalize_turn
import capture_scrub as scrub


@pytest.fixture
def q(tmp_path):
    return CaptureQueue(str(tmp_path / "cq.db"))


# ---- idempotency key (D-8) --------------------------------------------------
def test_idem_key_stable_and_normalized():
    k1 = idem_key("sess1", 3, "Hello  World", "Reply here")
    k2 = idem_key("sess1", 3, "hello world", "reply here")  # case/space normalized
    assert k1 == k2
    assert idem_key("sess1", 4, "Hello World", "Reply here") != k1  # ordinal matters
    assert idem_key("sess2", 3, "Hello World", "Reply here") != k1  # session matters
    assert len(k1) == 64


# ---- enqueue + idempotency (INV-5) -----------------------------------------
def test_enqueue_dedup_noop(q):
    k = idem_key("s", 1, "u", "a")
    assert q.enqueue(k, {"user": "u", "assistant": "a"}) is True
    assert q.enqueue(k, {"user": "u", "assistant": "a"}) is False  # dup no-op
    assert q.counts()["pending"] == 1


# ---- lease roundtrip + concurrency (D-10 / SKIP-LOCKED analog) --------------
def test_lease_then_done(q):
    k = idem_key("s", 1, "u", "a")
    q.enqueue(k, {"x": 1})
    row = q.lease_one(lease_s=60)
    assert row is not None and row["idem_key"] == k and row["status"] == "inflight"
    assert row["payload"] == {"x": 1}
    q.record_verdict(k, "ok")
    q.mark_done(k)
    assert q.counts()["done"] == 1
    assert q.lease_one() is None  # nothing left due


def test_two_leasers_never_double_claim(q):
    for i in range(3):
        q.enqueue(idem_key("s", i, f"u{i}", "a"), {"i": i})
    claimed = []
    for _ in range(5):
        r = q.lease_one(lease_s=60)
        if r:
            claimed.append(r["idem_key"])
    assert len(claimed) == len(set(claimed)) == 3  # each row claimed exactly once


def test_not_due_not_leased(q):
    k = idem_key("s", 1, "u", "a")
    q.enqueue(k, {}, now=1000)
    # schedule a retry into the future
    q.lease_one(now=1000)
    q.record_verdict(k, "fault")
    q.mark_retry(k, backoff_s=100, now=1000)
    assert q.lease_one(now=1050) is None       # still backing off
    assert q.lease_one(now=1101) is not None    # due again


# ---- reaper: environmental orphan does NOT ++attempts (D-10) ----------------
def test_reaper_env_orphan_no_attempt_burn(q):
    k = idem_key("s", 1, "u", "a")
    q.enqueue(k, {}, now=1000)
    q.lease_one(lease_s=30, now=1000)   # inflight, lease to 1030, NO verdict recorded
    # crash: lease expires with no model_verdict -> environmental
    counts = q.reap(now=1100)
    assert counts["requeued_env"] == 1 and counts["requeued_fault"] == 0
    # attempts must still be 0 (no poison loop)
    row = q.lease_one(now=1100)
    assert row is not None and row["attempts"] == 0


def test_reaper_model_fault_burns_attempt(q):
    k = idem_key("s", 1, "u", "a")
    q.enqueue(k, {}, now=1000)
    q.lease_one(lease_s=30, now=1000)
    q.record_verdict(k, "fault")        # a real model fault occurred
    counts = q.reap(now=1100, backoff_s=30)
    assert counts["requeued_fault"] == 1 and counts["requeued_env"] == 0
    row = q.lease_one(now=1200)          # past the reaped-fault backoff (1100+30)
    assert row is not None and row["attempts"] == 1


def test_retry_budget_dead_letters(q):
    k = idem_key("s", 1, "u", "a")
    q.enqueue(k, {}, now=0)
    t = 0.0
    status = None
    for _ in range(10):
        r = q.lease_one(now=t)
        if r is None:
            t += 1000
            continue
        q.record_verdict(k, "fault")
        status = q.mark_retry(k, backoff_s=1, max_attempts=5, now=t)
        t += 1000
        if status == "dead":
            break
    assert status == "dead"
    assert q.counts()["dead"] == 1


# ---- durability (INV-1): a committed enqueue survives a fresh connection ----
def test_durable_across_reopen(tmp_path):
    path = str(tmp_path / "cq.db")
    q1 = CaptureQueue(path)
    k = idem_key("s", 1, "u", "a")
    q1.enqueue(k, {"survives": True})
    del q1
    q2 = CaptureQueue(path)  # simulates a process restart
    assert q2.counts()["pending"] == 1
    row = q2.lease_one()
    assert row["payload"] == {"survives": True}


def test_purge_done_ttl(q):
    k = idem_key("s", 1, "u", "a")
    q.enqueue(k, {}, now=0)
    q.lease_one(now=0); q.mark_done(k, now=0)
    assert q.purge_done(older_than_s=100, now=50) == 0    # too new
    assert q.purge_done(older_than_s=100, now=200) == 1   # aged out


# ---- INV-4: deterministic secret scrubber (NB1 corpus, not one fixture) -----
# NOTE: every secret fixture below is assembled from fragments at runtime so no contiguous
# secret-shaped literal sits in the source bytes (keeps the fleet gitleaks scan clean), while the
# runtime VALUES are full/realistic so the deterministic scrubber is genuinely exercised. These are
# synthetic test vectors, not real credentials.
_FX_TELEGRAM = "8905425635:" + "AAH3xY9zKq" + "_Wp0LmNoPqRsTuVwXyZ" + "012345"
_FX_OPENAI = "sk-" + "proj-" + "abc123DEF456ghi789JKL012" + "mno345PQR"
_FX_ANTHROPIC = "sk-" + "ant-" + "api03-xY9zKqWp0LmNoPqRsTuVwXyZ0123456789" + "abcdef"
_FX_GHPAT = "ghp_" + "16CharsMinimumABCDEF" + "xyz0123456789"
_FX_AWS = "AKIA" + "IOSFODNN7" + "EXAMPLE"
_FX_GOOGLE = "AIza" + "SyD1abc23DEF456ghi789JKL012mno345PQ"
_FX_JWT = "ey" + "Jh" + "bGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9." + \
    "eyJzdWIiOiIxMjM0NTY3ODkwIn0." + "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
_FX_CONN = "postgres://admin:" + "s3cr3tP4ss" + "@10.0.0.5:5432/mem0"
_FX_LABELED = "hunter2" + "SuperLongSecretValue0123"

SECRET_CORPUS = [
    ("telegram bot token", f"User's Momus bot token is {_FX_TELEGRAM}", "telegram_bot_token"),
    ("openai key", f"the key {_FX_OPENAI} is in the env", "openai_or_anthropic_key"),
    ("anthropic key", f"{_FX_ANTHROPIC} used by the relay", "anthropic_key"),
    ("github pat", f"token {_FX_GHPAT} for the repo", "github_token"),
    ("aws key", f"{_FX_AWS} is the access key", "aws_access_key"),
    ("google key", f"{_FX_GOOGLE} set for maps", "google_api_key"),
    ("jwt", f"bearer {_FX_JWT}", "jwt"),
    ("pem", "-----BEGIN OPENSSH PRIVATE KEY-----", "pem_private_key"),
    ("conn string", f"db at {_FX_CONN}", "conn_string_with_password"),
    ("labeled secret", f"set password = {_FX_LABELED} in config", "labeled_secret_assignment"),
]


@pytest.mark.parametrize("label,text,expected", SECRET_CORPUS)
def test_scrubber_catches_secret_corpus(label, text, expected):
    hits = scrub.scan(text)
    assert expected in hits, f"{label}: expected {expected}, got {hits}"
    assert scrub.is_secret(text) is True


def test_scrubber_passes_clean_durable_facts():
    clean = [
        "User prefers concise replies over verbose ones.",
        "User's Mac Studio is the always-on host that runs the QMD daemon.",
        "User decided to ship Track A-lite before Track A-full.",
        "The mem0 store lives on ACE-AI at 192.168.1.216.",   # a LAN IP is not a secret literal
        "User keeps the OpenAI key in a 1Password reference op://Engineering/openai/key.",  # op:// safe
    ]
    for f in clean:
        assert scrub.is_secret(f) is False, f"false positive on: {f}"


def test_filter_facts_splits_and_audits():
    facts = [
        "User prefers dark mode.",
        "The bot token is " + ("8905425635:" + "AAH3xY9zKq" + "_Wp0LmNoPqRsTuVwXyZ" + "012345"),
        "User's fleet DNS is AdGuard Home on 192.168.1.208.",
    ]
    kept, dropped = scrub.filter_facts(facts)
    assert len(kept) == 2 and len(dropped) == 1
    assert dropped[0]["reason"] == "telegram_bot_token"
    # the dropped audit record carries NO secret text (only reason + len)
    assert "8905425635" not in str(dropped)


def test_op_reference_never_dropped():
    assert scrub.is_secret("credential at op://Engineering/Universal Homelab Password/password") is False
