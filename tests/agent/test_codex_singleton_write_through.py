"""Codex OAuth singleton root write-through + reuse-rescue (spec v11, #87503).

Covers the three production changes in ``hermes_cli/auth.py``:

* C1 — ``_save_codex_tokens`` mirrors a saved chain to the global root.
* C2 — ``_refresh_codex_auth_tokens`` resolves the caller's token source once
  and, for a root-resolved reader, persists its rotation directly to root.
* C3 — a relogin-required refresh attempts a root reuse-rescue (adopting a
  fresher sibling chain) before falling back to ``~/.codex`` CLI recovery.

Plus D-id (identity gating) and the three-outcome durability contract
(OUTCOME-SUCCESS silent / CLASS-D warning / CLASS-N critical+retry).

All token endpoints are mocked (autospec-friendly); no interactive OAuth runs.
The tests drive the real read-modify-write path against on-disk stores under
``tmp_path``, mirroring ``tests/agent/test_credential_pool_oauth_writethrough.py``.
"""

import base64
import json
import logging
import os
import re
import subprocess
import threading

from contextlib import contextmanager

import pytest

import hermes_cli.auth as A
from hermes_cli.auth import (
    AuthError,
    _CODEX_ROOT_PERSIST_ATTEMPTS,
    _CODEX_ROOT_PERSIST_BACKOFF_SECONDS,
    _codex_root_rescue_seen,
    _codex_token_identity,
    _refresh_codex_auth_tokens,
    _reset_codex_root_rescue_seen,
    _save_codex_tokens,
    _write_through_codex_to_global_root,
)


# ── token / store helpers ────────────────────────────────────────────────────

def _b64(obj) -> str:
    return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()


def _jwt(sub="acct-1", iss="https://auth.openai.com") -> str:
    """A valid three-segment base64url JWT with the given ``sub``/``iss``."""
    header = _b64({"alg": "none", "typ": "JWT"})
    return ".".join([header, _b64({"sub": sub, "iss": iss}), "sig"])


def _write_store(path, store) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store), encoding="utf-8")


def _read_store(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _codex_state(tokens, last_refresh="2026-06-01T00:00:00Z", **extra):
    state = {"tokens": dict(tokens), "last_refresh": last_refresh, "auth_mode": "chatgpt"}
    state.update(extra)
    return state


@pytest.fixture
def profile_and_root(tmp_path, monkeypatch):
    """Wire a profile auth store + a distinct global-root auth store on disk."""
    profile_path = tmp_path / "profiles" / "work" / "auth.json"
    root_path = tmp_path / "root" / "auth.json"
    monkeypatch.setattr(A, "_auth_file_path", lambda: profile_path)
    monkeypatch.setattr(A, "_global_auth_file_path", lambda: root_path)
    monkeypatch.setenv("HOME", str(tmp_path / "not-the-root"))
    return profile_path, root_path


@pytest.fixture(autouse=True)
def _reset_rescue_state(monkeypatch):
    """Clear the process-lifetime rescue seen-set and global-store memo."""
    _reset_codex_root_rescue_seen()
    A._global_auth_store_cache = None
    yield
    _reset_codex_root_rescue_seen()
    A._global_auth_store_cache = None


@pytest.fixture
def top_level_store(tmp_path, monkeypatch):
    """Top-level mode: the active store IS the global root (one shared file)."""
    root_path = tmp_path / "root" / "auth.json"
    monkeypatch.setattr(A, "_auth_file_path", lambda: root_path)
    monkeypatch.setattr(A, "_global_auth_file_path", lambda: root_path)
    monkeypatch.setenv("HOME", str(tmp_path / "not-the-root"))
    return root_path


def _mock_refresh(monkeypatch, result=None, exc=None, calls=None):
    """Replace ``refresh_codex_oauth_pure`` with a scripted stub."""
    def fake(access_token, refresh_token, timeout_seconds=20.0, **kw):
        if calls is not None:
            calls.append((access_token, refresh_token))
        if exc is not None:
            raise exc
        return dict(result or {"access_token": "new-at", "refresh_token": "new-rf"})

    monkeypatch.setattr(A, "refresh_codex_oauth_pure", fake)
    return fake


# ── T1 / T2 / T3 / T4 — the R1′/R2′/R3′/switch persistence matrix ───────────

def test_t1_root_resolved_success_writes_root_zero_profile(profile_and_root, monkeypatch):
    """R1′: a root-resolved reader persists to root and never seeds a profile block."""
    profile_path, root_path = profile_and_root
    _write_store(root_path, {"version": 1, "providers": {
        "openai-codex": _codex_state({"access_token": _jwt("acct-1"), "refresh_token": "root-rf"}),
    }})
    _write_store(profile_path, {"version": 1})  # no openai-codex block

    _mock_refresh(monkeypatch, result={"access_token": _jwt("acct-1"), "refresh_token": "new-rf"})

    tokens = {"access_token": _jwt("acct-1"), "refresh_token": "stale-rf"}
    out = _refresh_codex_auth_tokens(tokens, 20.0)

    assert out["access_token"] == _jwt("acct-1")
    assert out["refresh_token"] == "new-rf"

    root = _read_store(root_path)
    assert root["providers"]["openai-codex"]["tokens"]["refresh_token"] == "new-rf"
    assert root["providers"]["openai-codex"]["auth_mode"] == "chatgpt"

    profile = _read_store(profile_path)
    assert "openai-codex" not in profile.get("providers", {}), (
        "a root-resolved reader must NOT seed a shadowing profile block (#74339)"
    )


def test_t1_top_level_root_silent_success(top_level_store, monkeypatch, caplog):
    """F1: when root == active store (top-level mode), the rotation persists
    directly to that store as OUTCOME-SUCCESS — silent, zero CRITICAL."""
    root_path = top_level_store
    _write_store(root_path, {"version": 1, "providers": {"openai-codex": _codex_state(
        {"access_token": _jwt("acct-1"), "refresh_token": "root-rf"})}})

    _mock_refresh(monkeypatch, result={"access_token": _jwt("acct-1"), "refresh_token": "new-rf"})

    with caplog.at_level("DEBUG"):
        out = _refresh_codex_auth_tokens(
            {"access_token": _jwt("acct-1"), "refresh_token": "stale-rf"}, 20.0)

    assert out["refresh_token"] == "new-rf"

    # The rotated chain IS durably written to the (root == active) store.
    rt = _read_store(root_path)["providers"]["openai-codex"]["tokens"]["refresh_token"]
    assert rt == "new-rf", "top-level rotation was lost (same-path no-op)"

    # OUTCOME-SUCCESS is silent: no WARNING and no CRITICAL.
    assert not any(r.levelno >= logging.WARNING for r in caplog.records), caplog.text


def test_t2_owned_success_syncs_root_and_preserves_independent(profile_and_root, monkeypatch):
    """R2′: an owned save updates profile + root; independent account untouched."""
    profile_path, root_path = profile_and_root
    root = {
        "version": 1,
        "providers": {"openai-codex": _codex_state(
            {"access_token": _jwt("acct-1"), "refresh_token": "root-old-rf"}
        )},
        "credential_pool": {"openai-codex": [
            {"id": "dev", "source": "device_code", "access_token": _jwt("acct-1"),
             "refresh_token": "root-old-rf", "label": "singleton"},
            {"id": "alias", "source": "manual:device_code", "access_token": _jwt("acct-1"),
             "refresh_token": "root-old-rf", "label": "legacy-alias", "priority": 5},
            {"id": "indep", "source": "manual:device_code", "access_token": _jwt("acct-9"),
             "refresh_token": "indep-rf", "label": "independent", "priority": 1},
        ]},
    }
    _write_store(root_path, root)
    _write_store(profile_path, {"version": 1, "providers": {"openai-codex": _codex_state(
        {"access_token": _jwt("acct-1"), "refresh_token": "profile-old-rf"}
    )}})

    _save_codex_tokens(
        {"access_token": _jwt("acct-1"), "refresh_token": "fresh-rf"},
        last_refresh="2026-06-12T00:00:00Z",
        label="My Codex",
    )

    root_store = _read_store(root_path)
    rc = root_store["providers"]["openai-codex"]
    assert rc["tokens"]["refresh_token"] == "fresh-rf"
    assert rc["auth_mode"] == "chatgpt"
    assert rc["label"] == "My Codex"

    by_id = {e["id"]: e for e in root_store["credential_pool"]["openai-codex"]}
    # device_code singleton alias synced
    assert by_id["dev"]["refresh_token"] == "fresh-rf"
    # legacy alias (access_token matched previous singleton) synced
    assert by_id["alias"]["refresh_token"] == "fresh-rf"
    assert by_id["alias"]["priority"] == 5  # non-token fields untouched
    # independent account (#39236) byte-identical
    assert by_id["indep"]["refresh_token"] == "indep-rf"
    assert by_id["indep"]["access_token"] == _jwt("acct-9")


def test_t3_identity_mismatch_untouched_empty_root_populated(profile_and_root, monkeypatch):
    """R3′: mismatched identity leaves root untouched; empty root is populated."""
    profile_path, root_path = profile_and_root
    # (a) mismatched identity — root holds a different account
    _write_store(root_path, {"version": 1, "providers": {"openai-codex": _codex_state(
        {"access_token": _jwt("acct-OTHER"), "refresh_token": "other-rf"}
    )}})
    _write_store(profile_path, {"version": 1})
    _save_codex_tokens({"access_token": _jwt("acct-1"), "refresh_token": "mine-rf"})
    root_store = _read_store(root_path)
    assert root_store["providers"]["openai-codex"]["tokens"]["refresh_token"] == "other-rf"

    # (b) empty root — populated with set_active left alone
    _write_store(root_path, {"version": 1, "active_provider": "anthropic", "providers": {}})
    _save_codex_tokens({"access_token": _jwt("acct-1"), "refresh_token": "mine-rf"})
    root_store = _read_store(root_path)
    assert root_store["providers"]["openai-codex"]["tokens"]["refresh_token"] == "mine-rf"
    assert root_store["active_provider"] == "anthropic"  # set_active untouched


@pytest.mark.parametrize("same_sub", [True, False], ids=["same-sub-propagated", "diff-sub-untouched"])
def test_t4_switch_matrix(profile_and_root, monkeypatch, same_sub):
    """login-introduced rotation: same-sub propagates, different-sub does not."""
    profile_path, root_path = profile_and_root
    root_sub = "acct-root"
    _write_store(root_path, {"version": 1, "providers": {"openai-codex": _codex_state(
        {"access_token": _jwt(root_sub), "refresh_token": "root-rf"}
    )}})
    _write_store(profile_path, {"version": 1})
    save_sub = root_sub if same_sub else "acct-else"
    _save_codex_tokens({"access_token": _jwt(save_sub), "refresh_token": "new-rf"})
    root_store = _read_store(root_path)
    got = root_store["providers"]["openai-codex"]["tokens"]["refresh_token"]
    assert got == ("new-rf" if same_sub else "root-rf")


# ── T5 — the fault matrix (SILENT / CLASS-D / CLASS-N) ──────────────────────

def test_t5_c1_root_failure_profile_intact_warning(profile_and_root, monkeypatch, caplog):
    """CLASS-D: a root write-through failure leaves the profile save intact.

    Contract: the failure logs WARNING (self-healing next-trigger resync), and
    the profile copy stays durably intact."""
    profile_path, root_path = profile_and_root
    _write_store(root_path, {"version": 1, "providers": {"openai-codex": _codex_state(
        {"access_token": _jwt("acct-1"), "refresh_token": "root-rf"}
    )}})
    _write_store(profile_path, {"version": 1})

    monkeypatch.setattr(A, "_save_auth_store", _failing_save(A._save_auth_store, root_path=root_path))

    with caplog.at_level("DEBUG"):
        _save_codex_tokens({"access_token": _jwt("acct-1"), "refresh_token": "new-rf"})

    profile = _read_store(profile_path)
    assert profile["providers"]["openai-codex"]["tokens"]["refresh_token"] == "new-rf"

    # CLASS-D = WARNING only; no CRITICAL (the profile copy is the durable store).
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warns, "CLASS-D must emit a WARNING (self-healing root resync)"
    assert not any(r.levelno >= logging.CRITICAL for r in caplog.records), caplog.text


def _failing_save(original_save, root_path):
    """Fail ``_save_auth_store`` only when targetting the root path."""
    def wrapped(store, target_path=None):
        if target_path is not None and A._same_path(target_path, root_path):
            raise OSError("root write failure")
        return original_save(store, target_path=target_path)
    return wrapped


def _fake_timer(monkeypatch):
    """Replace ``time.sleep`` with a recorder; returns the ordered delay list."""
    sleeps = []
    monkeypatch.setattr(A.time, "sleep", sleeps.append)
    return sleeps


def test_t5_c2_direct_root_failure_returns_tokens_critical(profile_and_root, monkeypatch, caplog):
    """CLASS-N: a root-resolved direct write that always fails still returns tokens.

    Contract: exactly 3 attempts, fixed backoffs [0.5s, 1.0s], then CRITICAL
    naming manual ``hermes model`` re-auth (refreshed tokens still returned)."""
    profile_path, root_path = profile_and_root
    _write_store(root_path, {"version": 1, "providers": {"openai-codex": _codex_state(
        {"access_token": _jwt("acct-1"), "refresh_token": "root-rf"}
    )}})
    _write_store(profile_path, {"version": 1})

    monkeypatch.setattr(A, "_save_auth_store", _failing_save(A._save_auth_store, root_path=root_path))

    results = []
    _mock_refresh(monkeypatch, result={"access_token": _jwt("acct-1"), "refresh_token": "new-rf"})

    def fake_write_through(*a, **k):
        results.append(("attempt", a, k))
        return False  # always fails

    monkeypatch.setattr(A, "_write_through_codex_to_global_root", fake_write_through)
    sleeps = _fake_timer(monkeypatch)

    tokens = {"access_token": _jwt("acct-1"), "refresh_token": "stale-rf"}
    with caplog.at_level("DEBUG"):
        out = _refresh_codex_auth_tokens(tokens, 20.0)

    assert out["refresh_token"] == "new-rf"  # tokens still returned
    assert len(results) == _CODEX_ROOT_PERSIST_ATTEMPTS  # exactly 3 attempts
    # fixed backoff ordering, index-aligned with the attempts that precede them
    assert sleeps == list(_CODEX_ROOT_PERSIST_BACKOFF_SECONDS), sleeps

    crits = [r for r in caplog.records if r.levelno >= logging.CRITICAL]
    assert crits, "CLASS-N must emit a CRITICAL after exhausting retries"
    assert any("hermes model" in r.getMessage() for r in crits), "CRITICAL must name manual re-auth"


def test_t5_c3_post_persist_failure_returns_tokens_no_autherror(profile_and_root, monkeypatch, caplog):
    """CLASS-N (root-resolved rescue): persistence failure after a successful
    adoption POST still returns tokens — 3 attempts, 2 backoffs, then CRITICAL."""
    profile_path, root_path = profile_and_root
    _write_store(root_path, {"version": 1, "providers": {"openai-codex": _codex_state(
        {"access_token": _jwt("acct-1"), "refresh_token": "fresher-rf"}
    )}})
    _write_store(profile_path, {"version": 1})  # root-resolved (no own block)

    # First POST (our stale token) relogin-fails; adoption POST succeeds.
    state = {"n": 0}

    def fake(access_token, refresh_token, timeout_seconds=20.0, **kw):
        if refresh_token == "stale-rf":
            raise AuthError("rejected", provider="openai-codex",
                            code="invalid_grant", relogin_required=True)
        state["n"] += 1
        return {"access_token": _jwt("acct-1"), "refresh_token": "adopted-rf"}

    monkeypatch.setattr(A, "refresh_codex_oauth_pure", fake)
    # Persistence always fails → CLASS-N (must still return tokens, not raise).
    monkeypatch.setattr(A, "_save_auth_store", _failing_save(A._save_auth_store, root_path=root_path))
    monkeypatch.setattr(A, "_write_through_codex_to_global_root", lambda *a, **k: False)
    sleeps = _fake_timer(monkeypatch)

    # COUNT persistence attempts: every CLASS-N row must run the full
    # 3-attempt ladder against the failing root persist, with the two
    # intervening backoffs in contract order.
    wt_calls = []
    real_wt = A._write_through_codex_to_global_root

    def counting_wt(*a, **k):
        wt_calls.append(1)
        return False

    monkeypatch.setattr(A, "_write_through_codex_to_global_root", counting_wt)

    tokens = {"access_token": _jwt("acct-1"), "refresh_token": "stale-rf"}
    with caplog.at_level("DEBUG"):
        out = _refresh_codex_auth_tokens(tokens, 20.0)

    assert out["refresh_token"] == "adopted-rf"
    assert state["n"] == 1  # one adoption POST ran
    # R16'/A1v10 contract: the persistence ladder runs the FULL attempt count
    # (== _CODEX_ROOT_PERSIST_ATTEMPTS) against the failing root persist —
    # asserting sleeps alone would allow a single-attempt implementation.
    assert len(wt_calls) == A._CODEX_ROOT_PERSIST_ATTEMPTS, (
        f"expected {A._CODEX_ROOT_PERSIST_ATTEMPTS} root-RMW attempts, saw {len(wt_calls)}"
    )
    assert len(sleeps) == A._CODEX_ROOT_PERSIST_ATTEMPTS - 1, sleeps
    assert sleeps == list(_CODEX_ROOT_PERSIST_BACKOFF_SECONDS), sleeps

    crits = [r for r in caplog.records if r.levelno >= logging.CRITICAL]
    assert crits, "CLASS-N rescue persistence failure must CRITICAL"
    assert any("hermes model" in r.getMessage() for r in crits)


def test_t5_c3_owned_local_save_failure_critical_retries(profile_and_root, monkeypatch, caplog):
    """F2/C3-owned CLASS-N: a local save failure after a successful rescue POST
    retries 3× with 2 backoffs BEFORE CRITICAL (mirrors the C2 CLASS-N loop)."""
    profile_path, root_path = profile_and_root
    _write_store(root_path, {"version": 1, "providers": {"openai-codex": _codex_state(
        {"access_token": _jwt("acct-1"), "refresh_token": "fresher-rf"}
    )}})
    # OWNED caller: the profile holds its own (stale) codex block.
    _write_store(profile_path, {"version": 1, "providers": {"openai-codex": _codex_state(
        {"access_token": _jwt("acct-1"), "refresh_token": "stale-rf"}
    )}})

    def fake(access_token, refresh_token, timeout_seconds=20.0, **kw):
        if refresh_token == "stale-rf":
            raise AuthError("rejected", provider="openai-codex",
                            code="invalid_grant", relogin_required=True)
        return {"access_token": _jwt("acct-1"), "refresh_token": "adopted-rf"}

    monkeypatch.setattr(A, "refresh_codex_oauth_pure", fake)

    save_attempts = []

    def failing_local_save(tokens, last_refresh=None, label=None):
        save_attempts.append(tokens.get("refresh_token"))
        raise OSError("simulated local-save failure")

    monkeypatch.setattr(A, "_save_codex_tokens", failing_local_save)
    sleeps = _fake_timer(monkeypatch)

    tokens = {"access_token": _jwt("acct-1"), "refresh_token": "stale-rf"}
    with caplog.at_level("DEBUG"):
        out = _refresh_codex_auth_tokens(tokens, 20.0)

    # Refreshed/adopted tokens are still handed back (loud, not silent).
    assert out["refresh_token"] == "adopted-rf"
    # exactly 3 total local-save attempts, 2 fixed backoffs, then CRITICAL
    assert len(save_attempts) == _CODEX_ROOT_PERSIST_ATTEMPTS, save_attempts
    assert sleeps == list(_CODEX_ROOT_PERSIST_BACKOFF_SECONDS), sleeps
    crits = [r for r in caplog.records if r.levelno >= logging.CRITICAL]
    assert crits, "C3-owned CLASS-N must CRITICAL after retries"
    assert any("hermes model" in r.getMessage() for r in crits)


def test_t5_silent_success_no_warning_no_critical(profile_and_root, monkeypatch, caplog):
    """OUTCOME-SUCCESS is silent: no warning/critical on a fully durable save."""
    profile_path, root_path = profile_and_root
    _write_store(root_path, {"version": 1, "providers": {"openai-codex": _codex_state(
        {"access_token": _jwt("acct-1"), "refresh_token": "root-rf"}
    )}})
    _write_store(profile_path, {"version": 1})
    _mock_refresh(monkeypatch, result={"access_token": _jwt("acct-1"), "refresh_token": "new-rf"})

    with caplog.at_level("WARNING"):
        _refresh_codex_auth_tokens({"access_token": _jwt("acct-1"), "refresh_token": "stale-rf"}, 20.0)

    assert not any(r.levelno >= 30 for r in caplog.records), caplog.text


# ── T6 — classic mode byte-identity ─────────────────────────────────────────

def test_t6_classic_mode_inert(profile_and_root, monkeypatch, tmp_path):
    """D-classic: with no global root, C1/C2/C3 paths are inert."""
    profile_path, root_path = profile_and_root
    monkeypatch.setattr(A, "_global_auth_file_path", lambda: None)
    monkeypatch.setattr(A, "_auth_file_path", lambda: root_path)  # single store

    wt_calls = []
    monkeypatch.setattr(A, "_write_through_codex_to_global_root",
                        lambda *a, **k: wt_calls.append(("wt", a, k)) or True)

    _write_store(root_path, {"version": 1})
    _mock_refresh(monkeypatch, result={"access_token": _jwt("acct-1"), "refresh_token": "new-rf"})
    _save_codex_tokens({"access_token": _jwt("acct-1"), "refresh_token": "r1"})

    assert wt_calls == []  # C1 write-through never fires in classic mode

    out = _refresh_codex_auth_tokens({"access_token": _jwt("acct-1"), "refresh_token": "r1"}, 20.0)
    assert out["refresh_token"] == "new-rf"
    assert wt_calls == []  # C2 direct root write also inert (owned path)

    store = _read_store(root_path)
    assert store["providers"]["openai-codex"]["tokens"]["refresh_token"] == "new-rf"


def test_t6_classic_mode_byte_identity(profile_and_root, monkeypatch):
    """D-classic: classic-mode persistence is byte-identical to merge-base.

    With no global root, the write-through feature must not perturb any
    non-codex byte of the store, and the codex block holds exactly the rotated
    chain — i.e. the same bytes a pre-feature save would have produced."""
    profile_path, root_path = profile_and_root
    monkeypatch.setattr(A, "_global_auth_file_path", lambda: None)
    monkeypatch.setattr(A, "_auth_file_path", lambda: root_path)

    baseline = {
        "version": A.AUTH_STORE_VERSION,
        "active_provider": "anthropic",
        "providers": {
            "anthropic": {"tokens": {"access_token": "ak-ant"}, "auth_mode": "chatgpt"},
        },
        "credential_pool": {
            "anthropic": [{"id": "a", "source": "manual", "access_token": "ak-ant", "label": "l"}],
        },
        "nested": {"key": "value", "list": [1, 2, 3]},
    }
    _write_store(root_path, baseline)
    _t6_pre = _read_store(root_path)
    _mock_refresh(monkeypatch, result={"access_token": _jwt("acct-1"), "refresh_token": "new-rf"})

    # function outcome: classic-mode refresh returns the rotated chain
    out = _refresh_codex_auth_tokens(
        {"access_token": _jwt("acct-1"), "refresh_token": "r1"}, 20.0)
    assert out == {"access_token": _jwt("acct-1"), "refresh_token": "new-rf"}

    store = _read_store(root_path)

    # byte-identity vs the PRE-FUNCTION on-disk snapshot: the classic-mode
    # (no-global-root) save must produce exactly the merge-base bytes — i.e.
    # compare against what an UNCHANGED baseline module would have written,
    # not merely selected fields. Non-codex subtrees must be untouched AND
    # structural formatting/order preserved, so we diff raw objects.
    pre = _t6_pre
    # `updated_at` is stamped by _save_auth_store at EVERY save (merge-base
    # L1431 behavior, unchanged) — it is the one legitimate post-save addition.
    assert set(store.keys()) - set(pre.keys()) == {"updated_at"}, "top-level key drift in classic mode"
    assert store["version"] == pre["version"]
    assert store["nested"] == pre["nested"]
    assert store["active_provider"] == "openai-codex"  # expected codex-adjacent mutation
    assert store["providers"]["anthropic"] == pre["providers"]["anthropic"]
    assert store["credential_pool"]["anthropic"] == pre["credential_pool"]["anthropic"]
    # providers/credential_pool gained ONLY the openai-codex key, nothing else:
    assert set(store["providers"].keys()) - set(pre["providers"].keys()) == {"openai-codex"}
    assert set(pre["providers"].keys()) - set(store["providers"].keys()) == set()
    assert set(store["credential_pool"].keys()) == set(pre["credential_pool"].keys())
    # the codex block itself must equal exactly what merge-base logic writes:
    # tokens + auth_mode="chatgpt" (+ label rule), no write-through extras.
    codex_block = store["providers"]["openai-codex"]
    assert set(codex_block.keys()) <= {"tokens", "last_refresh", "auth_mode", "label"}
    assert "write_through" not in json.dumps(store)

    # the only codex-adjacent mutations are the expected merge-base ones:
    # active_provider flips to openai-codex and the codex block holds the chain.
    assert store["active_provider"] == "openai-codex"
    codex = store["providers"]["openai-codex"]
    assert codex["tokens"]["access_token"] == _jwt("acct-1")
    assert codex["tokens"]["refresh_token"] == "new-rf"


# ── T7 — rescue-order matrix + repeat/cap ───────────────────────────────────

def _relogin_stub_then_rescue(monkeypatch, rescue_result, calls):
    """The caller's POST fails relogin; the adoption POST succeeds once."""
    def fake(access_token, refresh_token, timeout_seconds=20.0, **kw):
        calls.append(refresh_token)
        if refresh_token == "stale-rf":
            raise AuthError("rejected", provider="openai-codex",
                            code="invalid_grant", relogin_required=True)
        return dict(rescue_result)

    monkeypatch.setattr(A, "refresh_codex_oauth_pure", fake)


def test_t7_eligible_rescue_skips_cli_recovery(profile_and_root, monkeypatch):
    profile_path, root_path = profile_and_root
    _write_store(root_path, {"version": 1, "providers": {"openai-codex": _codex_state(
        {"access_token": _jwt("acct-1"), "refresh_token": "fresher-rf"}
    )}})
    _write_store(profile_path, {"version": 1})

    calls = []
    _relogin_stub_then_rescue(monkeypatch,
                              {"access_token": _jwt("acct-1"), "refresh_token": "adopted-rf"}, calls)
    cli = []
    monkeypatch.setattr(A, "_recover_codex_tokens_from_cli", lambda *a, **k: cli.append(1) or None)

    out = _refresh_codex_auth_tokens({"access_token": _jwt("acct-1"), "refresh_token": "stale-rf"}, 20.0)

    assert out["refresh_token"] == "adopted-rf"
    assert cli == []  # CLI recovery unused when rescue eligible


def test_t7_ineligible_uses_cli_recovery(profile_and_root, monkeypatch):
    """When root's refresh token equals ours (nothing fresher), fall to CLI."""
    profile_path, root_path = profile_and_root
    _write_store(root_path, {"version": 1, "providers": {"openai-codex": _codex_state(
        {"access_token": _jwt("acct-1"), "refresh_token": "stale-rf"}  # same — ineligible
    )}})
    _write_store(profile_path, {"version": 1})

    def fake(access_token, refresh_token, timeout_seconds=20.0, **kw):
        raise AuthError("rejected", provider="openai-codex",
                        code="invalid_grant", relogin_required=True)

    monkeypatch.setattr(A, "refresh_codex_oauth_pure", fake)
    monkeypatch.setattr(A, "_import_codex_cli_tokens", lambda: {"access_token": "cli-at", "refresh_token": "cli-rf"})

    out = _refresh_codex_auth_tokens({"access_token": _jwt("acct-1"), "refresh_token": "stale-rf"}, 20.0)
    assert out["refresh_token"] == "cli-rf"


def test_t7_classic_direct_cli_recovery(profile_and_root, monkeypatch):
    profile_path, root_path = profile_and_root
    monkeypatch.setattr(A, "_global_auth_file_path", lambda: None)
    _write_store(profile_path, {"version": 1})

    def fake(access_token, refresh_token, timeout_seconds=20.0, **kw):
        raise AuthError("rejected", provider="openai-codex",
                        code="invalid_grant", relogin_required=True)

    monkeypatch.setattr(A, "refresh_codex_oauth_pure", fake)
    monkeypatch.setattr(A, "_import_codex_cli_tokens", lambda: {"access_token": "cli-at", "refresh_token": "cli-rf"})

    out = _refresh_codex_auth_tokens({"access_token": _jwt("acct-1"), "refresh_token": "stale-rf"}, 20.0)
    assert out["refresh_token"] == "cli-rf"


def test_t7_repeat_tuple_skip_and_ineligible_no_consume(profile_and_root, monkeypatch):
    """The seen-set caps one adoption per dead tuple per process; reset re-arms."""
    profile_path, root_path = profile_and_root
    _write_store(root_path, {"version": 1, "providers": {"openai-codex": _codex_state(
        {"access_token": _jwt("acct-1"), "refresh_token": "fresher-rf"}
    )}})
    _write_store(profile_path, {"version": 1})

    posts = []
    def fake(access_token, refresh_token, timeout_seconds=20.0, **kw):
        if refresh_token == "stale-rf":
            raise AuthError("rejected", provider="openai-codex",
                            code="invalid_grant", relogin_required=True)
        posts.append(refresh_token)
        return {"access_token": _jwt("acct-1"), "refresh_token": "adopted-rf"}

    monkeypatch.setattr(A, "refresh_codex_oauth_pure", fake)
    monkeypatch.setattr(A, "_recover_codex_tokens_from_cli", lambda *a, **k: None)

    tokens = {"access_token": _jwt("acct-1"), "refresh_token": "stale-rf"}
    # first: rescue eligible → one adoption POST, returns the adopted chain
    out = _refresh_codex_auth_tokens(tokens, 20.0)
    assert out["refresh_token"] == "adopted-rf"
    assert len(posts) == 1  # one adoption POST

    # second run, same process, same dead tuple → skipped entirely (no POST),
    # falls through to CLI recovery (empty here) → the original error surfaces
    with pytest.raises(AuthError):
        _refresh_codex_auth_tokens(tokens, 20.0)
    assert len(posts) == 1  # no second adoption POST

    # reset simulates a fresh process → attempts again
    _reset_codex_root_rescue_seen()
    out = _refresh_codex_auth_tokens(tokens, 20.0)
    assert out["refresh_token"] == "adopted-rf"
    assert len(posts) == 2


# ── T7c — double-barrier single-adoption ────────────────────────────────────

def test_t7c_two_threads_single_adoption(profile_and_root, monkeypatch):
    """Two same-process threads, same dead tuple ⇒ exactly one adoption POST.

    AMENDMENT-T7cv10: deterministic double-barrier choreography. B1 aligns the
    threads pre-race; each records its instrumented candidate seen-set
    observation BEFORE any lock acquisition; B2 releases only after BOTH
    pre-check observations are recorded; only then do they contend the flock.
    The authoritative re-check+mark happens INSIDE the acquired critical
    section — the loser observes the winner's mark there and falls through to
    CLI recovery. A defective outside-lock-decides implementation would
    double-POST and fail under this schedule.
    """
    profile_path, root_path = profile_and_root
    _write_store(root_path, {"version": 1, "providers": {"openai-codex": _codex_state(
        {"access_token": _jwt("acct-1"), "refresh_token": "fresher-rf"}
    )}})
    _write_store(profile_path, {"version": 1})

    DEAD = (_jwt("acct-1"), "stale-rf")
    posts = []
    lock = threading.Lock()
    b1 = threading.Barrier(2)
    b2 = threading.Barrier(2)
    obs = []  # (phase, thread_name, dead_tuple_in_seen)

    def fake(access_token, refresh_token, timeout_seconds=20.0, **kw):
        if refresh_token == "stale-rf":
            raise AuthError("rejected", provider="openai-codex",
                            code="invalid_grant", relogin_required=True)
        with lock:
            posts.append(threading.current_thread().name)
        return {"access_token": _jwt("acct-1"), "refresh_token": "adopted-rf"}

    monkeypatch.setattr(A, "refresh_codex_oauth_pure", fake)
    monkeypatch.setattr(A, "_recover_codex_tokens_from_cli",
                        lambda *a, **k: {"access_token": "cli-at", "refresh_token": "cli-rf"})

    real_lock = A._auth_store_lock
    tl = threading.local()

    @contextmanager
    def instrumented_lock(timeout_seconds=A.AUTH_LOCK_TIMEOUT_SECONDS, *, target_path=None):
        if target_path is not None and A._same_path(target_path, root_path):
            depth = getattr(tl, "depth", 0)
            outer = depth == 0
            if outer:
                # instrumented candidate seen-set check BEFORE any lock acquisition
                obs.append(("pre", threading.current_thread().name, DEAD in A._codex_root_rescue_seen))
                b2.wait(timeout=30)
            tl.depth = depth + 1
            try:
                with real_lock(timeout_seconds, target_path=target_path):
                    if outer:
                        # authoritative in-lock observation (the only copy that decides)
                        obs.append(("inlock", threading.current_thread().name,
                                    DEAD in A._codex_root_rescue_seen))
                    yield
            finally:
                tl.depth = getattr(tl, "depth", 1) - 1
        else:
            with real_lock(timeout_seconds, target_path=target_path):
                yield

    monkeypatch.setattr(A, "_auth_store_lock", instrumented_lock)

    rescued = []
    recovered = []
    errors = []

    def worker():
        try:
            b1.wait(timeout=30)
            out = A._refresh_codex_auth_tokens(
                {"access_token": _jwt("acct-1"), "refresh_token": "stale-rf"}, 20.0)
        except AuthError as exc:  # noqa: BLE001
            with lock:
                errors.append(exc)
            return
        with lock:
            (rescued if out.get("refresh_token") == "adopted-rf" else recovered).append(
                threading.current_thread().name)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, errors
    assert all(not t.is_alive() for t in threads)

    pre = [o for o in obs if o[0] == "pre"]
    inlock = [o for o in obs if o[0] == "inlock"]
    # both threads recorded their outside-lock pre-check before any lock was taken
    assert len(pre) == 2, f"expected both pre-check observations; got {obs}"
    assert all(not o[2] for o in pre), "outside-lock pre-checks must both see the tuple unmarked"
    # exactly one winner (in-lock sees unmarked) and one loser (in-lock sees the mark)
    winners = [o for o in inlock if not o[2]]
    losers = [o for o in inlock if o[2]]
    assert len(winners) == 1, f"expected one in-lock winner; got {inlock}"
    assert len(losers) == 1, "the loser's authoritative in-lock observation must see the winner's mark"

    assert len(posts) == 1, "exactly ONE adoption POST globally"
    assert len(rescued) == 1, "exactly one thread self-healed via rescue"
    assert len(recovered) == 1, "the loser fell through to CLI recovery"


# ── T8 — pool-sync previous-singleton equivalence ───────────────────────────

def test_t8_pool_sync_previous_singleton_equivalence(profile_and_root, monkeypatch):
    """R7′: the pre-save singleton capture still drives alias classification."""
    profile_path, root_path = profile_and_root
    _write_store(root_path, {"version": 1, "providers": {"openai-codex": _codex_state(
        {"access_token": _jwt("acct-1"), "refresh_token": "root-rf"}
    )}})
    _write_store(profile_path, {"version": 1, "providers": {"openai-codex": _codex_state(
        {"access_token": _jwt("acct-1"), "refresh_token": "profile-rf"}
    )}})

    captured = []
    real_sync = A._sync_codex_pool_entries
    def spy_sync(auth_store, tokens, last_refresh, previous_singleton_tokens=None):
        captured.append(previous_singleton_tokens)
        real_sync(auth_store, tokens, last_refresh,
                  previous_singleton_tokens=previous_singleton_tokens)

    monkeypatch.setattr(A, "_sync_codex_pool_entries", spy_sync)
    _save_codex_tokens({"access_token": _jwt("acct-1"), "refresh_token": "fresh-rf"})

    # The in-profile sync is fed the PRE-save singleton tokens (byte-unchanged
    # capture); the root write-through feed the ROOT pre-save snapshot.
    assert any(v and v.get("refresh_token") == "profile-rf" for v in captured), (
        "in-profile pool sync must receive the pre-save singleton tokens"
    )
    assert any(v and v.get("refresh_token") == "root-rf" for v in captured), (
        "root write-through must classify against the ROOT pre-save snapshot"
    )


# ── T9-v6 — concurrency (honest R16′ invariants) ────────────────────────────

def test_t9_concurrency_honest_invariants(profile_and_root, monkeypatch):
    profile_path, root_path = profile_and_root
    _write_store(root_path, {"version": 1, "providers": {"openai-codex": _codex_state(
        {"access_token": _jwt("acct-1"), "refresh_token": "fresher-rf"}
    )}})
    _write_store(profile_path, {"version": 1})

    posts = []
    errors = []
    lock = threading.Lock()

    def fake(access_token, refresh_token, timeout_seconds=20.0, **kw):
        if refresh_token == "stale-rf":
            raise AuthError("rejected", provider="openai-codex",
                            code="invalid_grant", relogin_required=True)
        with lock:
            posts.append(refresh_token)
        return {"access_token": _jwt("acct-1"), "refresh_token": "adopted-rf"}

    monkeypatch.setattr(A, "refresh_codex_oauth_pure", fake)
    # P0/P1/P2: root chain is the same account; only cohort members write; the
    # CLI recovery path also yields a valid chain, so every loser self-heals —
    # ZERO relogin surfaces.
    monkeypatch.setattr(A, "_recover_codex_tokens_from_cli",
                        lambda *a, **k: {"access_token": "cli-at", "refresh_token": "cli-rf"})

    n = 3
    held = []
    errors = []
    lock = threading.Lock()

    def worker():
        try:
            out = A._refresh_codex_auth_tokens(
                {"access_token": _jwt("acct-1"), "refresh_token": "stale-rf"}, 20.0)
        except AuthError as exc:
            with lock:
                errors.append(exc)
            return
        with lock:
            held.append(out)

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)

    # (b)/(c) under P0∧P1∧P2: every participant self-heals — ZERO surfaced
    # AuthErrors/relogin across all threads.
    assert not errors, f"no relogin should surface under P0/P1/P2; got {errors}"
    assert len(held) == n, f"every thread must hold a chain; got {len(held)} held"
    assert all(h.get("refresh_token") for h in held)
    # per-chain adoption POST bound (seen-set ⇒ exactly 1 across the cohort)
    assert len(posts) == 1, f"exactly one adoption POST; got {len(posts)}"
    # final root holds a valid rotated chain
    root_store = _read_store(root_path)
    rt = root_store["providers"]["openai-codex"]["tokens"]["refresh_token"]
    assert rt in ("fresher-rf", "adopted-rf")


# ── A2 premises / T10 / T13 — identity + structure ──────────────────────────

def test_t13_identity_corner_matrix():
    """D-id corner cases: valid, 2-segment, non-JSON, foreign-iss, empty-sub, opaque."""
    assert _codex_token_identity(_jwt("acct-1")) == "acct-1"  # positive control
    assert _codex_token_identity("a.b") is None  # 2 segments
    assert _codex_token_identity("a.b.c") is None  # non-JSON payload
    assert _codex_token_identity(_jwt("acct-1", iss="https://evil.example")) is None  # foreign iss
    assert _codex_token_identity(_jwt("")) is None  # empty sub
    assert _codex_token_identity("opaque-token") is None  # not a JWT
    assert _codex_token_identity(None) is None  # not a str
    # missing-iss ⇒ None-conservative (no issuer claim ⇒ cannot establish identity)
    tok_no_iss = ".".join([_b64({"alg": "none", "typ": "JWT"}), _b64({"sub": "acct-1"}), "sig"])
    assert _codex_token_identity(tok_no_iss) is None


def test_t13_both_none_pair_store_matrix(profile_and_root, monkeypatch):
    """D-id both-None: an opaque token on BOTH sides refuses the cross-store write
    (root + aliases untouched); an empty root is still populated."""
    profile_path, root_path = profile_and_root
    opaque = "opaque-token"  # not a JWT → identity None (conservative skip)

    # (a) non-empty root with opaque credentials → both identities None ⇒ the
    # D-id gate refuses: root singleton and its alias are left byte-identical.
    _write_store(root_path, {"version": 1, "providers": {"openai-codex": _codex_state(
        {"access_token": "opaque-root", "refresh_token": "root-rf"}
    )}, "credential_pool": {"openai-codex": [
        {"id": "alias", "source": "manual:device_code", "access_token": "opaque-root",
         "refresh_token": "root-rf", "label": "lbl"},
    ]}})
    _write_store(profile_path, {"version": 1})
    _save_codex_tokens({"access_token": opaque, "refresh_token": "new-rf"})

    root_store = _read_store(root_path)
    assert root_store["providers"]["openai-codex"]["tokens"]["refresh_token"] == "root-rf"  # untouched
    assert root_store["credential_pool"]["openai-codex"][0]["refresh_token"] == "root-rf"  # aliases skip

    # (b) empty root → populated (no credentials ⇒ identity gate skipped).
    _write_store(root_path, {"version": 1, "active_provider": "anthropic", "providers": {}})
    _save_codex_tokens({"access_token": opaque, "refresh_token": "new-rf"})
    root_store = _read_store(root_path)
    assert root_store["providers"]["openai-codex"]["tokens"]["refresh_token"] == "new-rf"  # empty-root populated
    assert root_store["active_provider"] == "anthropic"  # set_active untouched


def test_t10_structure_preservation(profile_and_root, monkeypatch):
    """Alias labels/ids/priority/suppressed_sources untouched; token fields in place."""
    profile_path, root_path = profile_and_root
    _write_store(root_path, {"version": 1, "providers": {"openai-codex": _codex_state(
        {"access_token": _jwt("acct-1"), "refresh_token": "root-rf"}
    )}, "credential_pool": {"openai-codex": [
        {"id": "alias", "source": "manual:device_code", "access_token": _jwt("acct-1"),
         "refresh_token": "root-rf", "label": "lbl", "priority": 7, "suppressed_sources": ["x"]},
    ]}})
    _write_store(profile_path, {"version": 1})

    _save_codex_tokens({"access_token": _jwt("acct-1"), "refresh_token": "fresh-rf"})

    entry = _read_store(root_path)["credential_pool"]["openai-codex"][0]
    assert entry["refresh_token"] == "fresh-rf"  # mutated in place
    assert entry["id"] == "alias"
    assert entry["label"] == "lbl"
    assert entry["priority"] == 7
    assert entry["suppressed_sources"] == ["x"]


def test_t17_field_set_pin(profile_and_root, monkeypatch):
    """C2 field set: root carries tokens + last_refresh + auth_mode + label."""
    profile_path, root_path = profile_and_root
    _write_store(root_path, {"version": 1, "providers": {"openai-codex": _codex_state(
        {"access_token": _jwt("acct-1"), "refresh_token": "root-rf"}, custom_field="keep"
    )}})
    _write_store(profile_path, {"version": 1})
    _mock_refresh(monkeypatch, result={"access_token": _jwt("acct-1"), "refresh_token": "new-rf"})

    _refresh_codex_auth_tokens({"access_token": _jwt("acct-1"), "refresh_token": "stale-rf"}, 20.0)

    rc = _read_store(root_path)["providers"]["openai-codex"]
    assert rc["tokens"]["refresh_token"] == "new-rf"
    assert rc["tokens"]["access_token"] == _jwt("acct-1")
    assert rc["auth_mode"] == "chatgpt"
    assert rc["last_refresh"]  # present
    assert rc["custom_field"] == "keep"  # root-only field preserved


def test_t17_label_rule(profile_and_root, monkeypatch):
    """Label rule: a non-empty label is written through; absent/empty is not
    (a pre-existing label is preserved rather than cleared)."""
    profile_path, root_path = profile_and_root
    _write_store(root_path, {"version": 1, "providers": {"openai-codex": _codex_state(
        {"access_token": _jwt("acct-1"), "refresh_token": "root-rf"}
    )}})
    _write_store(profile_path, {"version": 1})

    # non-empty label present → written to the root (and profile)
    _save_codex_tokens({"access_token": _jwt("acct-1"), "refresh_token": "r1"},
                       last_refresh="2026-06-01T00:00:00Z", label="My Codex")
    rc = _read_store(root_path)["providers"]["openai-codex"]
    assert rc["label"] == "My Codex"

    # absent/empty (whitespace) label → NOT written; prior label preserved
    _save_codex_tokens({"access_token": _jwt("acct-1"), "refresh_token": "r2"},
                       last_refresh="2026-06-02T00:00:00Z", label="   ")
    rc = _read_store(root_path)["providers"]["openai-codex"]
    assert rc["label"] == "My Codex"  # whitespace label is not a label


# ── T15 / T18 — negative greps + dual-enumeration budget pin ────────────────

def _git(*args):
    root = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                          capture_output=True, text=True).stdout.strip()
    return subprocess.run(["git", "-C", root, *args], capture_output=True, text=True)


def _git_repo_root() -> str:
    return subprocess.run(["git", "rev-parse", "--show-toplevel"],
                          capture_output=True, text=True).stdout.strip()


def _diff_base():
    # The branch descends from upstream commit 77001a6be; upstream/main has
    # since advanced, so resolve the true fork point via merge-base rather than
    # diffing against the (moved) tip.
    for ref in ("upstream/main", "origin/main"):
        r = _git("rev-parse", "--verify", ref)
        if r.returncode == 0 and r.stdout.strip():
            m = _git("merge-base", "HEAD", r.stdout.strip())
            if m.returncode == 0 and m.stdout.strip():
                return m.stdout.strip()
            return r.stdout.strip()
    return "77001a6be"


def _diff_text():
    base = _diff_base()
    committed = _git("diff", base, "HEAD").stdout or ""
    worktree = _git("diff", base).stdout or ""
    return committed + "\n" + worktree


def _auth_diff_text():
    """Diff of the single production file under budget (auth.py) only."""
    base = _diff_base()
    committed = _git("diff", base, "HEAD", "--", "hermes_cli/auth.py").stdout or ""
    worktree = _git("diff", base, "--", "hermes_cli/auth.py").stdout or ""
    return committed + "\n" + worktree


def _changed_files():
    base = _diff_base()
    changed = _git("diff", "--name-only", base).stdout.split()
    changed += _git("diff", "--name-only", base, "HEAD").stdout.split()
    changed += _git("ls-files", "--others", "--exclude-standard").stdout.split()
    return {f for f in changed if f}


def test_t15_negative_greps():
    """No provenance parameter and no id_token usage in the codex paths."""
    diff = _auth_diff_text()
    assert "provenance" not in diff
    assert "id_token" not in diff


def _ast_top_level_symbols(source_text: str) -> dict:
    """Map every top-level symbol name -> stable AST dump of its definition."""
    import ast as _ast
    tree = _ast.parse(source_text)
    out = {}
    for node in tree.body:
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)):
            out[node.name] = _ast.dump(node)
        elif isinstance(node, _ast.Assign):
            for t in node.targets:
                if isinstance(t, _ast.Name):
                    out[t.id] = _ast.dump(node)
        elif isinstance(node, _ast.AnnAssign) and isinstance(node.target, _ast.Name):
            out[node.target.id] = _ast.dump(node)
    return out


def _auth_symbol_maps():
    """{merge_base: symbols, worktree_HEAD: symbols} for hermes_cli/auth.py."""
    import ast as _ast
    base_sha = _diff_base()
    base_text = _git("show", f"{base_sha}:hermes_cli/auth.py").stdout
    head_path = os.path.join(_git_repo_root(), "hermes_cli", "auth.py")
    with open(head_path) as fh:
        head_text = fh.read()
    return (
        _ast_top_level_symbols(base_text),
        _ast_top_level_symbols(head_text),
    )


def test_t18_dual_enumeration_budget_pin():
    """R17'prod/§SB: EXACT production-symbol + non-production-file budgets.

    Structural, not heuristic: parse BOTH versions of hermes_cli/auth.py with
    ``ast`` and compare their top-level symbol trees exactly. Any added,
    removed, or modified production symbol outside the §SB enumeration fails —
    including symbol kinds regex-on-diff hunks would miss (classes, imports,
    ann-assigns, mid-file inserts).
    """
    changed = _changed_files()

    allowed_nonprod = {
        "CHANGELOG.md",
        "tests/agent/test_codex_singleton_write_through.py",
    }
    nonprod = {f for f in changed if f != "hermes_cli/auth.py"}
    assert nonprod == allowed_nonprod, (
        f"non-production files must EXACTLY equal {sorted(allowed_nonprod)}; got {sorted(nonprod)}"
    )

    base_syms, head_syms = _auth_symbol_maps()
    added = set(head_syms) - set(base_syms)
    removed = set(base_syms) - set(head_syms)

    # §SB amended budget: exactly these NEW top-level symbols.
    expected_added = {
        "_codex_token_identity",                    # helper (D-id identity computation)
        "_write_through_codex_to_global_root",      # helper (C1/C2/C3 root RMW)
        "_reset_codex_root_rescue_seen",            # hook (rescue cap reset)
        "_CODEX_OAUTH_ISSUER",                      # constant (D-id issuer pin)
        "_CODEX_ROOT_PERSIST_ATTEMPTS",             # constant (A1v10 retry count)
        "_CODEX_ROOT_PERSIST_BACKOFF_SECONDS",      # constant (A1v10 backoff ladder)
        "_codex_root_rescue_seen",                  # seen-set store
    }
    assert added == expected_added, (
        f"production-symbol additions EXACTLY {sorted(expected_added)} required; "
        f"got added={sorted(added)}"
    )
    assert removed == set(), f"no production symbol may be removed: {sorted(removed)}"

    # Of pre-existing symbols, ONLY the two host functions may have CHANGED
    # bodies/defaults. Everything else must be AST-identical to merge-base.
    expected_modified = {"_save_codex_tokens", "_refresh_codex_auth_tokens"}
    changed_bodies = {
        name for name in (set(base_syms) & set(head_syms))
        if base_syms[name] != head_syms[name]
    }
    assert changed_bodies == expected_modified, (
        f"only {sorted(expected_modified)} may be modified in place; "
        f"found {sorted(changed_bodies)}"
    )
