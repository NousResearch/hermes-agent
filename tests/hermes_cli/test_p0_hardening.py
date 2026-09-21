"""P0 hardening tests (2026-06-12 kanban/orchestration review).

Covers the confirmed-bug fixes:
  R-1  deterministic review sampling (sha256, not per-process hash())
  I-2  intake never routes children to non-spawnable profiles
  PE-1 profile_edit fails CLOSED when the blast-radius guard can't load
  PE-2 profile_rollback requires a reason and fails CLOSED without the guard
"""

from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_decompose as decompose


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# --------------------------------------------------------------------------
# R-1 — deterministic review sampling
# --------------------------------------------------------------------------

def _sha_bucket(task_id: str, rate: int) -> int:
    return int(hashlib.sha256(task_id.encode("utf-8")).hexdigest(), 16) % rate


def test_should_review_uses_stable_sha256_bucket(kanban_home, monkeypatch):
    """Fast-tier sampling must match the sha256 bucket, not builtin hash()."""
    cfg = {"tiered_review": True, "review_sample_rate": 5}
    # Pick a task_id that sha256 maps to bucket 0 (reviewed) for rate 5.
    sampled_in = next(
        f"task-{i}" for i in range(1000) if _sha_bucket(f"task-{i}", 5) == 0
    )
    sampled_out = next(
        f"task-{i}" for i in range(1000) if _sha_bucket(f"task-{i}", 5) != 0
    )
    with kb.connect() as conn:
        # No failed runs → decision is purely the sampling bucket.
        assert kb._should_review(conn, "fast", sampled_in, kanban_cfg=cfg) is True
        assert kb._should_review(conn, "fast", sampled_out, kanban_cfg=cfg) is False
        # Stable across repeated calls (proxy for across-process stability).
        for _ in range(3):
            assert kb._should_review(conn, "fast", sampled_in, kanban_cfg=cfg) is True


# --------------------------------------------------------------------------
# I-2 — intake routes to spawnable profiles only
# --------------------------------------------------------------------------

def test_build_roster_excludes_nonspawnable_profiles(monkeypatch):
    profiles = [
        types.SimpleNamespace(name="quan-code", description="coder"),
        types.SimpleNamespace(name="quan", description="lead, not spawnable"),
        types.SimpleNamespace(name="octacon", description="lead, not spawnable"),
    ]
    monkeypatch.setattr(
        decompose.profiles_mod, "list_profiles", lambda: profiles
    )
    # Mirror the dispatcher: leads are not spawnable, sub-profiles are.
    monkeypatch.setattr(
        kb, "_is_profile_spawnable",
        lambda name: name not in ("quan", "octacon"),
    )
    roster, valid = decompose._build_roster()
    names = {r["name"] for r in roster}
    assert names == {"quan-code"}
    assert valid == {"quan-code"}
    assert "quan" not in valid and "octacon" not in valid


def test_is_profile_spawnable_fails_closed_on_profiles_import_error(monkeypatch):
    """If hermes_cli.profiles can't be imported, treat the profile as
    non-spawnable rather than assuming it is safe to dispatch to."""
    monkeypatch.setitem(sys.modules, "hermes_cli.profiles", None)
    assert kb._is_profile_spawnable("some-profile") is False


def test_is_profile_spawnable_fails_closed_on_config_load_error(monkeypatch):
    """If profile_exists succeeds but config can't be loaded, treat the
    profile as non-spawnable rather than assuming it isn't blocked."""
    from hermes_cli import profiles as profiles_mod
    monkeypatch.setattr(profiles_mod, "profile_exists", lambda name: True)

    from hermes_cli import config as config_mod

    def _boom():
        raise RuntimeError("config unreadable")

    monkeypatch.setattr(config_mod, "load_config_readonly", _boom)
    assert kb._is_profile_spawnable("some-profile") is False


def test_build_roster_excludes_profile_when_spawnability_unknown(monkeypatch):
    """Fail closed: if the spawnability predicate raises, exclude the profile
    rather than risk offering a non-spawnable one. Excluded profiles fall
    through to default_assignee, so work is never stranded."""
    profiles = [types.SimpleNamespace(name="solo", description="x")]
    monkeypatch.setattr(
        decompose.profiles_mod, "list_profiles", lambda: profiles
    )

    def _boom(name):
        raise RuntimeError("config unavailable")

    monkeypatch.setattr(kb, "_is_profile_spawnable", _boom)
    roster, valid = decompose._build_roster()
    assert valid == set()
    assert roster == []


# --------------------------------------------------------------------------
# PE-1 / PE-2 — profile mutation fails CLOSED without the guard
# --------------------------------------------------------------------------

def _force_blast_radius_import_error(monkeypatch):
    # Setting the module to None makes `from hermes_cli.blast_radius import X`
    # raise ImportError, simulating a broken/absent guard.
    monkeypatch.setitem(sys.modules, "hermes_cli.blast_radius", None)


def test_profile_edit_fails_closed_without_guard(monkeypatch):
    from tools import kanban_tools
    _force_blast_radius_import_error(monkeypatch)
    out = kanban_tools._handle_profile_edit({
        "profile": "denji", "file": "config.yaml",
        "key_path": "x", "new_value": "1", "reason": "test",
    })
    assert "fail-closed" in out.lower()
    assert "guard unavailable" in out.lower()


def test_profile_rollback_requires_reason(monkeypatch):
    from tools import kanban_tools
    out = kanban_tools._handle_profile_rollback({"commit_hash": "abc123"})
    assert "reason is required" in out.lower()


def test_profile_rollback_fails_closed_without_guard(monkeypatch):
    from tools import kanban_tools
    _force_blast_radius_import_error(monkeypatch)
    out = kanban_tools._handle_profile_rollback({
        "commit_hash": "abc123", "reason": "revert bad edit",
    })
    assert "fail-closed" in out.lower()
