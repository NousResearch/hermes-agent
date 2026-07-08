"""Tests for agent/curator.py — orchestrator, idle gating, state transitions.

LLM spawning is never exercised here — `_run_llm_review` is monkeypatched so
tests run fully offline and the curator module doesn't need real credentials.
"""

from __future__ import annotations

import importlib
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


@pytest.fixture
def curator_env(tmp_path, monkeypatch):
    """Isolated HERMES_HOME + freshly reloaded curator + skill_usage modules."""
    home = tmp_path / ".hermes"
    (home / "skills").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    import tools.skill_usage as usage
    importlib.reload(usage)
    import agent.curator as curator
    importlib.reload(curator)

    # Neutralize the real LLM pass by default — tests opt in per-case.
    monkeypatch.setattr(curator, "_run_llm_review", lambda prompt: "llm-stub")

    # Default: no config file → curator defaults. Tests can override.
    monkeypatch.setattr(curator, "_load_config", lambda: {})
    # Pin prune_builtins OFF by default so transition tests don't pick up
    # built-ins unless they explicitly enable it. Both config-reading paths
    # are pinned (curator reads via _load_config; skill_usage reads config
    # directly). Tests opt in with _enable_prune_builtins(...).
    monkeypatch.setattr(usage, "_prune_builtins_enabled", lambda: False)

    yield {"home": home, "curator": curator, "usage": usage}

    # Teardown: a curator review launched with synchronous=False spawns a
    # daemon "curator-review" thread that calls save_state() when it finishes.
    # save_state() resolves the state path from HERMES_HOME at write time, so a
    # straggler thread that outlives this test would write into whatever home
    # the *next* test has configured (or the default ~/.hermes once monkeypatch
    # restores the env) — corrupting an unrelated test's state file. This race
    # is invisible on a fast machine but flakes under CI load. Join any such
    # thread here, while HERMES_HOME is still pinned to this test's tmp home
    # (curator_env depends on monkeypatch, so this teardown runs before the
    # monkeypatch env is restored). See the salvage of #14261 CI flake.
    for t in threading.enumerate():
        if t.name == "curator-review" and t.is_alive():
            t.join(timeout=10.0)


def _write_skill(skills_dir: Path, name: str):
    d = skills_dir / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: x\n---\n", encoding="utf-8",
    )
    return d


# ---------------------------------------------------------------------------
# Config gates
# ---------------------------------------------------------------------------

def test_curator_enabled_default_true(curator_env):
    assert curator_env["curator"].is_enabled() is True


def test_curator_disabled_via_config(curator_env, monkeypatch):
    c = curator_env["curator"]
    monkeypatch.setattr(c, "_load_config", lambda: {"enabled": False})
    assert c.is_enabled() is False
    assert c.should_run_now() is False


def test_curator_defaults(curator_env):
    c = curator_env["curator"]
    assert c.get_interval_hours() == 24 * 7  # 7 days
    assert c.get_min_idle_hours() == 2
    assert c.get_stale_after_days() == 30
    assert c.get_archive_after_days() == 90


def test_curator_config_overrides(curator_env, monkeypatch):
    c = curator_env["curator"]
    monkeypatch.setattr(c, "_load_config", lambda: {
        "interval_hours": 12,
        "min_idle_hours": 0.5,
        "stale_after_days": 7,
        "archive_after_days": 60,
    })
    assert c.get_interval_hours() == 12
    assert c.get_min_idle_hours() == 0.5
    assert c.get_stale_after_days() == 7
    assert c.get_archive_after_days() == 60


# ---------------------------------------------------------------------------
# should_run_now
# ---------------------------------------------------------------------------

def test_first_run_defers(curator_env):
    """The FIRST observation of the curator (fresh install, no state file)
    must NOT trigger an immediate run. The curator is designed to run after
    a full ``interval_hours`` of skill activity, not on the first background
    tick after installation. Fixes #18373.
    """
    c = curator_env["curator"]
    # No state file — should defer and seed last_run_at.
    assert c.should_run_now() is False
    state = c.load_state()
    assert state.get("last_run_at") is not None, (
        "first observation should seed last_run_at so the interval clock "
        "starts ticking instead of firing immediately next tick"
    )
    # A second immediate call still returns False (seeded, not yet stale).
    assert c.should_run_now() is False


def test_recent_run_blocks(curator_env):
    c = curator_env["curator"]
    c.save_state({
        "last_run_at": datetime.now(timezone.utc).isoformat(),
        "paused": False,
    })
    assert c.should_run_now() is False


def test_old_run_eligible(curator_env):
    """A run older than the configured interval should re-trigger. Use a
    2x-interval cushion so the test doesn't become coupled to the exact
    default — bumping DEFAULT_INTERVAL_HOURS shouldn't break it."""
    c = curator_env["curator"]
    long_ago = datetime.now(timezone.utc) - timedelta(
        hours=c.get_interval_hours() * 2
    )
    c.save_state({"last_run_at": long_ago.isoformat(), "paused": False})
    assert c.should_run_now() is True


def test_paused_blocks_even_if_stale(curator_env):
    c = curator_env["curator"]
    long_ago = datetime.now(timezone.utc) - timedelta(days=30)
    c.save_state({"last_run_at": long_ago.isoformat(), "paused": True})
    assert c.should_run_now() is False


def test_set_paused_roundtrip(curator_env):
    c = curator_env["curator"]
    c.set_paused(True)
    assert c.is_paused() is True
    c.set_paused(False)
    assert c.is_paused() is False


# ---------------------------------------------------------------------------
# Automatic state transitions
# ---------------------------------------------------------------------------

def test_unused_skill_transitions_to_stale(curator_env):
    c = curator_env["curator"]
    u = curator_env["usage"]
    skills_dir = curator_env["home"] / "skills"
    _write_skill(skills_dir, "old-skill")

    # Record last-use well past stale_after_days (30 default)
    long_ago = (datetime.now(timezone.utc) - timedelta(days=45)).isoformat()
    data = u.load_usage()
    data["old-skill"] = u._empty_record()
    data["old-skill"]["created_by"] = "agent"
    data["old-skill"]["last_used_at"] = long_ago
    data["old-skill"]["created_at"] = long_ago
    u.save_usage(data)

    counts = c.apply_automatic_transitions()
    assert counts["marked_stale"] == 1
    assert u.get_record("old-skill")["state"] == "stale"


def test_very_old_skill_gets_archived(curator_env):
    c = curator_env["curator"]
    u = curator_env["usage"]
    skills_dir = curator_env["home"] / "skills"
    skill_dir = _write_skill(skills_dir, "ancient")

    super_old = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat()
    data = u.load_usage()
    data["ancient"] = u._empty_record()
    data["ancient"]["created_by"] = "agent"
    data["ancient"]["last_used_at"] = super_old
    data["ancient"]["created_at"] = super_old
    u.save_usage(data)

    counts = c.apply_automatic_transitions()
    assert counts["archived"] == 1
    assert not skill_dir.exists()
    assert (skills_dir / ".archive" / "ancient" / "SKILL.md").exists()
    assert u.get_record("ancient")["state"] == "archived"


def test_pinned_skill_is_never_touched(curator_env):
    c = curator_env["curator"]
    u = curator_env["usage"]
    skills_dir = curator_env["home"] / "skills"
    _write_skill(skills_dir, "precious")

    super_old = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
    data = u.load_usage()
    data["precious"] = u._empty_record()
    data["precious"]["created_by"] = "agent"
    data["precious"]["last_used_at"] = super_old
    data["precious"]["created_at"] = super_old
    data["precious"]["pinned"] = True
    u.save_usage(data)

    counts = c.apply_automatic_transitions()
    assert counts["archived"] == 0
    assert counts["marked_stale"] == 0
    rec = u.get_record("precious")
    assert rec["state"] == "active"  # untouched
    assert rec["pinned"] is True


def test_stale_skill_reactivates_on_recent_use(curator_env):
    c = curator_env["curator"]
    u = curator_env["usage"]
    skills_dir = curator_env["home"] / "skills"
    _write_skill(skills_dir, "revived")

    recent = datetime.now(timezone.utc).isoformat()
    data = u.load_usage()
    data["revived"] = u._empty_record()
    data["revived"]["created_by"] = "agent"
    data["revived"]["state"] = "stale"
    data["revived"]["last_used_at"] = recent
    data["revived"]["created_at"] = recent
    u.save_usage(data)

    counts = c.apply_automatic_transitions()
    assert counts["reactivated"] == 1
    assert u.get_record("revived")["state"] == "active"


def test_new_skill_without_last_used_not_immediately_archived(curator_env):
    """A freshly-created skill with no use history should not get archived
    just because last_used_at is None."""
    c = curator_env["curator"]
    u = curator_env["usage"]
    skills_dir = curator_env["home"] / "skills"
    _write_skill(skills_dir, "fresh")

    # Bump nothing — record doesn't exist yet. Curator should create it
    # and fall back to created_at which is ~now.
    counts = c.apply_automatic_transitions()
    assert counts["archived"] == 0
    assert counts["marked_stale"] == 0
    assert (skills_dir / "fresh").exists()


def _backdate(u, name: str, days: int, *, use_count: int = 1):
    """Write an agent-created usage record whose activity is *days* old."""
    ts = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    data = u.load_usage()
    data[name] = u._empty_record()
    data[name]["created_by"] = "agent"
    data[name]["created_at"] = ts
    data[name]["last_used_at"] = ts if use_count else None
    data[name]["last_activity_at"] = ts if use_count else None
    data[name]["use_count"] = use_count
    u.save_usage(data)


def test_cron_referenced_skill_is_not_archived(curator_env, monkeypatch):
    """A skill referenced by a cron job must survive inactivity archival even
    when its activity is well past archive_after_days. The scheduler only
    bumps usage when a job fires, so paused / infrequent / far-future jobs
    would otherwise have their skills aged out from under them."""
    c = curator_env["curator"]
    u = curator_env["usage"]
    skills_dir = curator_env["home"] / "skills"
    _write_skill(skills_dir, "cron-dep")
    _write_skill(skills_dir, "orphan")
    _backdate(u, "cron-dep", 200)
    _backdate(u, "orphan", 200)

    # Pretend a (paused/infrequent) cron job references "cron-dep" only.
    monkeypatch.setattr(c, "_cron_referenced_skills", lambda: {"cron-dep"})

    counts = c.apply_automatic_transitions()

    assert u.get_record("cron-dep")["state"] == "active"  # protected
    assert (skills_dir / "cron-dep").exists()
    assert u.get_record("orphan")["state"] == "archived"  # control
    assert counts["archived"] == 1


def test_unused_skill_not_archived_before_stale_floor(curator_env):
    """A never-used skill (use_count == 0) younger than stale_after_days must
    not be marked stale or archived — absence of use is not evidence of
    staleness when the skill simply hasn't had its trigger come up yet."""
    c = curator_env["curator"]
    u = curator_env["usage"]
    skills_dir = curator_env["home"] / "skills"
    _write_skill(skills_dir, "young-unused")
    _backdate(u, "young-unused", 10, use_count=0)  # < 30d stale floor

    counts = c.apply_automatic_transitions()

    assert u.get_record("young-unused")["state"] == "active"
    assert counts["archived"] == 0
    assert counts["marked_stale"] == 0


def test_unused_skill_archived_past_archive_window(curator_env):
    """The use=0 floor only protects YOUNG skills — a never-used skill older
    than archive_after_days still archives (no perpetual reprieve)."""
    c = curator_env["curator"]
    u = curator_env["usage"]
    skills_dir = curator_env["home"] / "skills"
    _write_skill(skills_dir, "old-unused")
    _backdate(u, "old-unused", 200, use_count=0)

    counts = c.apply_automatic_transitions()

    assert u.get_record("old-unused")["state"] == "archived"
    assert counts["archived"] == 1


def test_candidate_list_marks_cron_referenced_skills(curator_env, monkeypatch):
    """The LLM review candidate list flags cron-referenced skills so the
    review pass knows not to prune them."""
    c = curator_env["curator"]
    u = curator_env["usage"]
    skills_dir = curator_env["home"] / "skills"
    _write_skill(skills_dir, "cron-dep")
    _write_skill(skills_dir, "plain")
    _backdate(u, "cron-dep", 1)
    _backdate(u, "plain", 1)
    monkeypatch.setattr(c, "_cron_referenced_skills", lambda: {"cron-dep"})

    listing = c._render_candidate_list()
    cron_line = next(l for l in listing.splitlines() if l.startswith("- cron-dep"))
    plain_line = next(l for l in listing.splitlines() if l.startswith("- plain"))
    assert "cron=yes" in cron_line
    assert "cron=no" in plain_line


def test_manual_skill_is_not_auto_archived(curator_env):
    """Manual skills can have usage records, but without the agent-created
    marker they must stay out of curator transitions."""
    c = curator_env["curator"]
    u = curator_env["usage"]
    skills_dir = curator_env["home"] / "skills"
    skill_dir = _write_skill(skills_dir, "manual")

    super_old = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
    data = u.load_usage()
    data["manual"] = u._empty_record()
    data["manual"]["last_used_at"] = super_old
    data["manual"]["created_at"] = super_old
    u.save_usage(data)

    counts = c.apply_automatic_transitions()
    assert counts["checked"] == 0
    assert counts["archived"] == 0
    assert skill_dir.exists()


def test_bundled_skill_not_touched_by_transitions(curator_env):
    c = curator_env["curator"]
    u = curator_env["usage"]
    skills_dir = curator_env["home"] / "skills"
    _write_skill(skills_dir, "bundled")
    (skills_dir / ".bundled_manifest").write_text(
        "bundled:abc\n", encoding="utf-8",
    )

    super_old = (datetime.now(timezone.utc) - timedelta(days=500)).isoformat()
    data = u.load_usage()
    data["bundled"] = u._empty_record()
    data["bundled"]["last_used_at"] = super_old
    u.save_usage(data)

    counts = c.apply_automatic_transitions()
    # bundled skills are excluded from the agent-created list entirely
    assert counts["checked"] == 0
    assert (skills_dir / "bundled").exists()  # never moved


# ---------------------------------------------------------------------------
# prune_builtins: curator may archive bundled built-ins after inactivity
# ---------------------------------------------------------------------------

def _enable_prune_builtins(curator_env, monkeypatch):
    """Flip curator.prune_builtins on for both config-reading paths."""
    c = curator_env["curator"]
    u = curator_env["usage"]
    monkeypatch.setattr(c, "_load_config", lambda: {"prune_builtins": True})
    monkeypatch.setattr(u, "_prune_builtins_enabled", lambda: True)


def _disable_prune_builtins(curator_env, monkeypatch):
    """Flip curator.prune_builtins off for both config-reading paths."""
    c = curator_env["curator"]
    u = curator_env["usage"]
    monkeypatch.setattr(c, "_load_config", lambda: {"prune_builtins": False})
    monkeypatch.setattr(u, "_prune_builtins_enabled", lambda: False)


def test_prune_builtins_default_on(curator_env):
    # Shipped default is ON: with no explicit config, built-ins are eligible.
    c = curator_env["curator"]
    # _load_config returns {} (fixture) → default True surfaces.
    assert c.get_prune_builtins() is True


def test_prune_builtins_off_excludes_bundled(curator_env, monkeypatch):
    c = curator_env["curator"]
    skills_dir = curator_env["home"] / "skills"
    _write_skill(skills_dir, "bundled")
    (skills_dir / ".bundled_manifest").write_text("bundled:abc\n", encoding="utf-8")

    # Explicitly off → bundled is not a candidate (the opt-out path).
    _disable_prune_builtins(curator_env, monkeypatch)
    assert c.get_prune_builtins() is False
    counts = c.apply_automatic_transitions()
    assert counts["checked"] == 0
    assert (skills_dir / "bundled").exists()


def test_prune_builtins_seeds_clock_on_first_sight(curator_env, monkeypatch):
    c = curator_env["curator"]
    u = curator_env["usage"]
    skills_dir = curator_env["home"] / "skills"
    _write_skill(skills_dir, "bundled")
    (skills_dir / ".bundled_manifest").write_text("bundled:abc\n", encoding="utf-8")
    _enable_prune_builtins(curator_env, monkeypatch)

    # First pass: built-in has no record yet → it's seeded, NOT archived,
    # even though it's "old" on disk. The inactivity clock starts now.
    counts = c.apply_automatic_transitions()
    assert counts["checked"] == 1
    assert counts["seeded"] == 1
    assert counts["archived"] == 0
    assert (skills_dir / "bundled").exists()
    # A record now exists with created_at ~ now.
    assert isinstance(u.load_usage().get("bundled"), dict)


def test_prune_builtins_archives_stale_bundled_and_suppresses(curator_env, monkeypatch):
    c = curator_env["curator"]
    u = curator_env["usage"]
    skills_dir = curator_env["home"] / "skills"
    _write_skill(skills_dir, "bundled")
    (skills_dir / ".bundled_manifest").write_text("bundled:abc\n", encoding="utf-8")
    _enable_prune_builtins(curator_env, monkeypatch)

    # Seed a record whose last activity is far past the archive cutoff.
    super_old = (datetime.now(timezone.utc) - timedelta(days=500)).isoformat()
    data = u.load_usage()
    data["bundled"] = u._empty_record()
    data["bundled"]["last_used_at"] = super_old
    u.save_usage(data)

    counts = c.apply_automatic_transitions()
    assert counts["archived"] == 1
    # Directory moved into .archive/, suppression recorded so update won't restore.
    assert not (skills_dir / "bundled").exists()
    assert (skills_dir / ".archive" / "bundled").exists()
    assert "bundled" in u.read_suppressed_names()


def test_prune_builtins_restore_clears_suppression(curator_env, monkeypatch):
    u = curator_env["usage"]
    skills_dir = curator_env["home"] / "skills"
    _write_skill(skills_dir, "bundled")
    (skills_dir / ".bundled_manifest").write_text("bundled:abc\n", encoding="utf-8")
    _enable_prune_builtins(curator_env, monkeypatch)

    ok, _ = u.archive_skill("bundled")
    assert ok
    assert "bundled" in u.read_suppressed_names()

    ok, _ = u.restore_skill("bundled")
    assert ok
    assert (skills_dir / "bundled").exists()
    assert "bundled" not in u.read_suppressed_names()


def test_protected_builtin_never_archived_even_when_stale(curator_env, monkeypatch):
    """A protected built-in (e.g. `plan`) is never archived, even when it is a
    stale bundled skill under prune_builtins — it backs a load-bearing slash
    command and must survive every curator pass."""
    u = curator_env["usage"]
    c = curator_env["curator"]
    skills_dir = curator_env["home"] / "skills"
    name = next(iter(u.PROTECTED_BUILTIN_SKILLS))  # the real protected name(s)
    _write_skill(skills_dir, name)
    (skills_dir / ".bundled_manifest").write_text(f"{name}:abc\n", encoding="utf-8")
    _enable_prune_builtins(curator_env, monkeypatch)

    # Force a record that is far past the archive cutoff.
    super_old = (datetime.now(timezone.utc) - timedelta(days=500)).isoformat()
    data = u.load_usage()
    data[name] = u._empty_record()
    data[name]["last_used_at"] = super_old
    u.save_usage(data)

    counts = c.apply_automatic_transitions()
    assert counts["archived"] == 0
    # Not even enumerated as a candidate → not "checked".
    assert name not in u.list_agent_created_skill_names()
    assert (skills_dir / name).exists()
    assert name not in u.read_suppressed_names()


def test_protected_builtin_is_not_curation_eligible(curator_env, monkeypatch):
    """is_curation_eligible() returns False for protected built-ins regardless
    of prune_builtins, and archive_skill() refuses them directly."""
    u = curator_env["usage"]
    skills_dir = curator_env["home"] / "skills"
    name = next(iter(u.PROTECTED_BUILTIN_SKILLS))
    _write_skill(skills_dir, name)
    (skills_dir / ".bundled_manifest").write_text(f"{name}:abc\n", encoding="utf-8")
    _enable_prune_builtins(curator_env, monkeypatch)

    assert u.is_protected_builtin(name) is True
    assert u.is_curation_eligible(name) is False
    ok, msg = u.archive_skill(name)
    assert ok is False
    assert (skills_dir / name).exists()


def test_prune_builtins_never_touches_hub_skills(curator_env, monkeypatch):
    u = curator_env["usage"]
    skills_dir = curator_env["home"] / "skills"
    _write_skill(skills_dir, "hubskill")
    hub_dir = skills_dir / ".hub"
    hub_dir.mkdir(parents=True, exist_ok=True)
    (hub_dir / "lock.json").write_text(
        '{"version": 1, "installed": {"hubskill": {"install_path": "hubskill"}}}',
        encoding="utf-8",
    )
    _enable_prune_builtins(curator_env, monkeypatch)

    # Even with prune_builtins on, hub-installed skills stay off-limits.
    assert u.is_curation_eligible("hubskill") is False
    ok, msg = u.archive_skill("hubskill")
    assert ok is False
    assert "hub-installed" in msg
    assert (skills_dir / "hubskill").exists()


# ---------------------------------------------------------------------------
# run_curator_review orchestration
# ---------------------------------------------------------------------------

def test_run_review_records_state(curator_env):
    c = curator_env["curator"]
    u = curator_env["usage"]
    skills_dir = curator_env["home"] / "skills"
    _write_skill(skills_dir, "a")
    u.mark_agent_created("a")

    result = c.run_curator_review(synchronous=True)
    assert "started_at" in result
    state = c.load_state()
    assert state["last_run_at"] is not None
    assert state["run_count"] >= 1
    assert state["last_run_summary"] is not None


def test_dry_run_does_not_advance_state(curator_env, monkeypatch):
    """Dry-run previews must not bump last_run_at or run_count. A preview
    shouldn't defer the next scheduled real pass or look like a real run in
    `hermes curator status`. Fixes #18373.
    """
    c = curator_env["curator"]
    u = curator_env["usage"]
    skills_dir = curator_env["home"] / "skills"
    _write_skill(skills_dir, "a")
    u.mark_agent_created("a")

    # Stub the LLM so the test doesn't need a provider.
    monkeypatch.setattr(
        c, "_run_llm_review",
        lambda prompt: {
            "final": "", "summary": "dry preview", "model": "", "provider": "",
            "tool_calls": [], "error": None,
        },
    )

    c.run_curator_review(synchronous=True, dry_run=True)
    state = c.load_state()
    assert state.get("last_run_at") is None, "dry-run must not seed last_run_at"
    assert state.get("run_count", 0) == 0, "dry-run must not bump run_count"
    assert "dry-run" in (state.get("last_run_summary") or ""), (
        "dry-run summary should be labeled so status output is unambiguous"
    )


def test_dry_run_injects_report_only_banner(curator_env, monkeypatch):
    """The dry-run prompt must carry a banner instructing the LLM not to
    call any mutating tool. This is defense in depth — the caller also
    skips automatic transitions — but the LLM prompt is the only guard
    against the model calling skill_manage directly."""
    c = curator_env["curator"]
    u = curator_env["usage"]
    skills_dir = curator_env["home"] / "skills"
    _write_skill(skills_dir, "a")
    u.mark_agent_created("a")

    captured = {}
    def _stub(prompt):
        captured["prompt"] = prompt
        return {"final": "", "summary": "s", "model": "", "provider": "",
                "tool_calls": [], "error": None}
    monkeypatch.setattr(c, "_run_llm_review", _stub)

    c.run_curator_review(synchronous=True, dry_run=True, consolidate=True)
    assert "DRY-RUN" in captured["prompt"]
    assert "DO NOT" in captured["prompt"]


def test_dry_run_skips_automatic_transitions(curator_env, monkeypatch):
    """Dry-run must not call apply_automatic_transitions — the auto pass
    archives skills deterministically, and a preview must not touch the
    filesystem."""
    c = curator_env["curator"]
    u = curator_env["usage"]
    skills_dir = curator_env["home"] / "skills"
    _write_skill(skills_dir, "a")
    u.mark_agent_created("a")

    called = {"n": 0}
    def _explode(*_a, **_kw):
        called["n"] += 1
        return {"checked": 0, "marked_stale": 0, "archived": 0, "reactivated": 0}
    monkeypatch.setattr(c, "apply_automatic_transitions", _explode)
    monkeypatch.setattr(
        c, "_run_llm_review",
        lambda p: {"final": "", "summary": "s", "model": "", "provider": "",
                   "tool_calls": [], "error": None},
    )

    c.run_curator_review(synchronous=True, dry_run=True)
    assert called["n"] == 0, "dry-run must skip apply_automatic_transitions"


def test_run_review_synchronous_invokes_llm_stub(curator_env, monkeypatch):
    c = curator_env["curator"]
    u = curator_env["usage"]
    skills_dir = curator_env["home"] / "skills"
    _write_skill(skills_dir, "a")
    u.mark_agent_created("a")

    calls = []
    def _stub(prompt):
        calls.append(prompt)
        return {
            "final": "stubbed-summary",
            "summary": "stubbed-summary",
            "model": "stub-model",
            "provider": "stub-provider",
            "tool_calls": [],
            "error": None,
        }
    monkeypatch.setattr(c, "_run_llm_review", _stub)

    captured = []
    c.run_curator_review(
        on_summary=lambda s: captured.append(s),
        synchronous=True,
        consolidate=True,
    )

    assert len(calls) == 1
    assert "skill CURATOR" in calls[0] or "CURATOR" in calls[0]
    assert captured  # on_summary was called
    assert any("stubbed-summary" in s for s in captured)


def test_run_review_skips_llm_when_no_candidates(curator_env, monkeypatch):
    c = curator_env["curator"]
    # No skills in the dir → no candidates
    calls = []
    monkeypatch.setattr(
        c, "_run_llm_review",
        lambda prompt: (calls.append(prompt), "never-called")[1],
    )

    captured = []
    c.run_curator_review(on_summary=lambda s: captured.append(s), synchronous=True)

    assert calls == []  # LLM not invoked
    assert any("skipped" in s for s in captured)


def test_consolidate_default_off(curator_env):
    """Consolidation (the LLM umbrella pass) is OFF by default — only the
    deterministic inactivity prune runs unless the user opts in."""
    c = curator_env["curator"]
    assert c.get_consolidate() is False


def test_consolidate_enabled_via_config(curator_env, monkeypatch):
    c = curator_env["curator"]
    monkeypatch.setattr(c, "_load_config", lambda: {"consolidate": True})
    assert c.get_consolidate() is True


def test_run_review_skips_llm_when_consolidate_off(curator_env, monkeypatch):
    """With consolidation off (the default), a run does the deterministic
    prune but never spawns the LLM consolidation fork — even with candidates
    present. The run is still recorded and a 'consolidation off' summary is
    surfaced."""
    c = curator_env["curator"]
    u = curator_env["usage"]
    skills_dir = curator_env["home"] / "skills"
    _write_skill(skills_dir, "a")
    u.mark_agent_created("a")

    calls = []
    monkeypatch.setattr(
        c, "_run_llm_review",
        lambda prompt: (calls.append(prompt), "never-called")[1],
    )

    captured = []
    c.run_curator_review(on_summary=lambda s: captured.append(s), synchronous=True)

    assert calls == []  # LLM consolidation fork not invoked
    assert any("consolidation off" in s for s in captured)
    # The run is still recorded (deterministic prune happened).
    state = c.load_state()
    assert state["last_run_at"] is not None
    assert state["run_count"] >= 1


def test_run_review_consolidate_override_runs_llm(curator_env, monkeypatch):
    """Passing consolidate=True overrides the config default (off) and drives
    the LLM consolidation pass — mirrors `hermes curator run --consolidate`."""
    c = curator_env["curator"]
    u = curator_env["usage"]
    skills_dir = curator_env["home"] / "skills"
    _write_skill(skills_dir, "a")
    u.mark_agent_created("a")

    calls = []
    monkeypatch.setattr(
        c, "_run_llm_review",
        lambda prompt: (calls.append(prompt), {
            "final": "", "summary": "s", "model": "", "provider": "",
            "tool_calls": [], "error": None,
        })[1],
    )

    c.run_curator_review(synchronous=True, consolidate=True)
    assert len(calls) == 1


def test_maybe_run_curator_respects_disabled(curator_env, monkeypatch):
    c = curator_env["curator"]
    monkeypatch.setattr(c, "_load_config", lambda: {"enabled": False})
    result = c.maybe_run_curator()
    assert result is None


def test_maybe_run_curator_enforces_idle_gate(curator_env, monkeypatch):
    c = curator_env["curator"]
    monkeypatch.setattr(c, "_load_config", lambda: {"min_idle_hours": 2})
    # idle less than the threshold
    result = c.maybe_run_curator(idle_for_seconds=60.0)
    assert result is None


def test_maybe_run_curator_runs_when_eligible(curator_env, monkeypatch):
    c = curator_env["curator"]
    u = curator_env["usage"]
    skills_dir = curator_env["home"] / "skills"
    _write_skill(skills_dir, "a")
    u.mark_agent_created("a")
    # Seed last_run_at far in the past so the interval gate opens — the
    # "no state" path intentionally defers the first run now (#18373).
    long_ago = datetime.now(timezone.utc) - timedelta(hours=c.get_interval_hours() * 2)
    c.save_state({"last_run_at": long_ago.isoformat(), "paused": False})
    # Force idle over threshold
    result = c.maybe_run_curator(idle_for_seconds=99999.0)
    assert result is not None
    assert "started_at" in result


def test_maybe_run_curator_defers_on_fresh_install(curator_env):
    """Fresh install (no curator state file) must NOT fire the curator on
    the first gateway tick. The first observation seeds last_run_at and
    returns None. Fixes #18373."""
    c = curator_env["curator"]
    skills_dir = curator_env["home"] / "skills"
    _write_skill(skills_dir, "a")
    # Infinite idle — the only thing that should block the run is the new
    # deferred-first-run gate.
    result = c.maybe_run_curator(idle_for_seconds=99999.0)
    assert result is None
    # And the next tick still defers (we seeded last_run_at to "now").
    result2 = c.maybe_run_curator(idle_for_seconds=99999.0)
    assert result2 is None


def test_maybe_run_curator_swallows_exceptions(curator_env, monkeypatch):
    c = curator_env["curator"]

    def explode():
        raise RuntimeError("boom")

    monkeypatch.setattr(c, "should_run_now", explode)
    # Must not raise
    assert c.maybe_run_curator() is None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_state_file_survives_corrupt_read(curator_env):
    c = curator_env["curator"]
    c._state_file().write_text("not json", encoding="utf-8")
    # Must fall back to default, not raise
    assert c.load_state() == c._default_state()


def test_state_atomic_write_no_tmp_leftovers(curator_env):
    c = curator_env["curator"]
    c.save_state({"paused": True})
    parent = c._state_file().parent
    tmp_files = [p.name for p in parent.iterdir() if p.name.endswith(".tmp")]
    assert tmp_files == []


def test_state_preserves_last_report_path(curator_env):
    c = curator_env["curator"]
    c.save_state({
        "last_run_at": "2026-04-30T12:00:00+00:00",
        "last_run_summary": "ok",
        "last_report_path": "/tmp/curator-report",
        "paused": False,
        "run_count": 1,
    })
    state = c.load_state()
    assert state["last_report_path"] == "/tmp/curator-report"


def test_curator_review_prompt_has_invariants():
    """Core invariants must be in the review prompt text."""
    from agent.curator import CURATOR_REVIEW_PROMPT
    assert "MUST NOT" in CURATOR_REVIEW_PROMPT or "DO NOT" in CURATOR_REVIEW_PROMPT
    assert "bundled" in CURATOR_REVIEW_PROMPT.lower()
    assert "delete" in CURATOR_REVIEW_PROMPT.lower()
    assert "pinned" in CURATOR_REVIEW_PROMPT.lower()
    # Must describe the actions the reviewer can take. The exact vocabulary
    # has tightened over time (the umbrella-first prompt drops 'keep' as a
    # first-class decision verb, since passive keep-everything is the
    # failure mode the prompt is trying to avoid), but the core merge /
    # archive / patch trio must remain callable.
    for verb in ("patch", "archive"):
        assert verb in CURATOR_REVIEW_PROMPT.lower()
    # Must mention consolidation (possibly via "merge" or "consolidat")
    assert "consolidat" in CURATOR_REVIEW_PROMPT.lower() or "merge" in CURATOR_REVIEW_PROMPT.lower()


def test_curator_review_prompt_points_at_existing_tools_only():
    """The review prompt must rely on existing tools (skill_manage + terminal)
    and must NOT reference bespoke curator tools that are not registered
    model tools."""
    from agent.curator import CURATOR_REVIEW_PROMPT
    assert "skill_manage" in CURATOR_REVIEW_PROMPT
    assert "skills_list" in CURATOR_REVIEW_PROMPT
    assert "skill_view" in CURATOR_REVIEW_PROMPT
    assert "terminal" in CURATOR_REVIEW_PROMPT.lower()
    # These would be nice but aren't actually registered as tools — the
    # curator uses skill_manage + terminal mv instead.
    assert "archive_skill" not in CURATOR_REVIEW_PROMPT
    assert "pin_skill" not in CURATOR_REVIEW_PROMPT


def test_curator_does_not_instruct_model_to_pin():
    """Pinning is a user opt-out, not a model decision. The prompt should
    not tell the reviewer to pin skills autonomously."""
    from agent.curator import CURATOR_REVIEW_PROMPT
    # "pinned" appears in the invariant ("skip pinned skills"), but "pin"
    # as a decision verb should not.
    lines = CURATOR_REVIEW_PROMPT.split("\n")
    decision_block = "\n".join(
        l for l in lines
        if l.strip().startswith(("keep", "patch", "archive", "consolidate", "pin "))
    )
    # No standalone "pin" action line
    assert not any(l.strip().startswith("pin ") for l in lines), (
        f"Found a pin action line in:\n{decision_block}"
    )


def test_curator_review_prompt_is_umbrella_first():
    """The curator prompt must push umbrella-building / class-level thinking,
    not pair-level 'are these two the same?' analysis."""
    from agent.curator import CURATOR_REVIEW_PROMPT
    lower = CURATOR_REVIEW_PROMPT.lower()
    # Must frame the task as active umbrella-building, not a passive audit.
    assert "umbrella" in lower, (
        "must use UMBRELLA framing — the class-first abstraction the curator "
        "is designed to produce"
    )
    # Must tell the reviewer not to stop at pair-level distinctness.
    assert "class" in lower, "must reference class-level thinking"
    # Must cover the three consolidation methods explicitly
    assert "references/" in CURATOR_REVIEW_PROMPT, (
        "must name references/ as a demotion target for session-specific content"
    )
    # templates/ and scripts/ make the umbrella a real class-level skill
    assert "templates/" in CURATOR_REVIEW_PROMPT
    assert "scripts/" in CURATOR_REVIEW_PROMPT
    # Must say the counter argument: usage=0 is not a reason to skip
    assert "use_count" in CURATOR_REVIEW_PROMPT or "counter" in lower, (
        "must preempt the 'usage counters are zero, I can't judge' bailout"
    )


def test_curator_review_prompt_preserves_skill_package_integrity():
    """Consolidation must not flatten package skills and break linked files."""
    from agent.curator import CURATOR_REVIEW_PROMPT

    lower = CURATOR_REVIEW_PROMPT.lower()
    assert "complete" in lower and "directory package" in lower
    assert "not a new skill root" in lower
    assert "do not flatten only skill.md" in lower
    assert "rewrite" in lower and "new paths" in lower
    assert "archive the entire original skill package unchanged" in lower
    for dirname in ("references/", "templates/", "scripts/", "assets/"):
        assert dirname in CURATOR_REVIEW_PROMPT



def test_curator_review_prompt_offers_support_file_actions():
    """Support-file demotion (references/templates/scripts) must be one of
    the three consolidation methods, alongside merge-into-existing and
    create-new-umbrella."""
    from agent.curator import CURATOR_REVIEW_PROMPT
    # skill_manage action=write_file is how references/ are added to an
    # existing skill — this is the create-adjacent action the curator needs
    # to demote narrow siblings without touching their SKILL.md.
    assert "write_file" in CURATOR_REVIEW_PROMPT
    # Must offer creating a brand-new umbrella when no existing one fits
    assert "action=create" in CURATOR_REVIEW_PROMPT or "create a new umbrella" in CURATOR_REVIEW_PROMPT.lower()



def test_cli_unpin_refuses_bundled_skill(curator_env, capsys):
    """hermes curator unpin must refuse bundled/hub skills too (matches pin)."""
    from hermes_cli import curator as cli
    skills_dir = curator_env["home"] / "skills"
    _write_skill(skills_dir, "ship-skill")
    (skills_dir / ".bundled_manifest").write_text(
        "ship-skill:abc\n", encoding="utf-8",
    )

    class _A:
        skill = "ship-skill"

    rc = cli._cmd_unpin(_A())
    captured = capsys.readouterr()
    assert rc == 1
    assert "bundled" in captured.out.lower() or "hub" in captured.out.lower()


def test_cli_pin_refuses_bundled_skill(curator_env, capsys):
    from hermes_cli import curator as cli
    skills_dir = curator_env["home"] / "skills"
    _write_skill(skills_dir, "ship-skill")
    (skills_dir / ".bundled_manifest").write_text(
        "ship-skill:abc\n", encoding="utf-8",
    )

    class _A:
        skill = "ship-skill"

    rc = cli._cmd_pin(_A())
    captured = capsys.readouterr()
    assert rc == 1
    assert "bundled" in captured.out.lower() or "hub" in captured.out.lower()


# ---------------------------------------------------------------------------
# curator review-model resolution (canonical auxiliary.curator slot)
#
# Curator was unified with the rest of the aux task system in Apr 2026 so
# `hermes model` → auxiliary picker, the dashboard Models tab, and the full
# per-task config (timeout, base_url, api_key, extra_body) all work for it.
# Voscko report: curator.auxiliary.{provider,model} was advertised but never
# read. Fix wires curator through auxiliary.curator with a legacy fallback.
# ---------------------------------------------------------------------------


def test_review_model_defaults_to_main_when_slot_is_auto(curator_env):
    """auxiliary.curator absent (or auto/empty) → use main model.provider/model."""
    curator = curator_env["curator"]
    cfg = {
        "model": {"provider": "openrouter", "default": "openai/gpt-5.5"},
    }
    assert curator._resolve_review_model(cfg) == ("openrouter", "openai/gpt-5.5")

    # Explicit auto/empty slot — still main model.
    cfg["auxiliary"] = {"curator": {"provider": "auto", "model": ""}}
    assert curator._resolve_review_model(cfg) == ("openrouter", "openai/gpt-5.5")


def test_review_model_honors_auxiliary_curator_slot(curator_env):
    """auxiliary.curator.{provider,model} fully set → that pair wins."""
    curator = curator_env["curator"]
    cfg = {
        "model": {"provider": "openrouter", "default": "openai/gpt-5.5"},
        "auxiliary": {
            "curator": {
                "provider": "openrouter",
                "model": "openai/gpt-5.4-mini",
            },
        },
    }
    assert curator._resolve_review_model(cfg) == (
        "openrouter", "openai/gpt-5.4-mini",
    )


def test_review_runtime_passes_auxiliary_curator_credentials(curator_env):
    """Per-slot api_key/base_url must ride into resolve_runtime_provider (not main-only creds)."""
    curator = curator_env["curator"]
    cfg = {
        "model": {"provider": "openrouter", "default": "openai/gpt-5.5"},
        "auxiliary": {
            "curator": {
                "provider": "custom",
                "model": "local-mini",
                "api_key": "sk-curator-only",
                "base_url": "http://localhost:11434/v1",
            },
        },
    }
    binding = curator._resolve_review_runtime(cfg)
    assert binding.provider == "custom"
    assert binding.model == "local-mini"
    assert binding.explicit_api_key == "sk-curator-only"
    assert binding.explicit_base_url == "http://localhost:11434/v1"


def test_review_runtime_strips_blank_aux_credentials(curator_env):
    curator = curator_env["curator"]
    cfg = {
        "model": {"provider": "openrouter", "default": "openai/gpt-5.5"},
        "auxiliary": {
            "curator": {
                "provider": "openrouter",
                "model": "x/y",
                "api_key": "   ",
                "base_url": "",
            },
        },
    }
    binding = curator._resolve_review_runtime(cfg)
    assert binding.explicit_api_key is None
    assert binding.explicit_base_url is None


def test_review_runtime_ignores_auxiliary_credentials_when_using_main(curator_env):
    """Falling through to main model must not pick up stray auxiliary.curator secrets."""
    curator = curator_env["curator"]
    cfg = {
        "model": {"provider": "openrouter", "default": "openai/gpt-5.5"},
        "auxiliary": {
            "curator": {
                "provider": "auto",
                "model": "",
                "api_key": "must-not-leak",
                "base_url": "http://curator-slot-ignored/",
            },
        },
    }
    binding = curator._resolve_review_runtime(cfg)
    assert (binding.provider, binding.model) == ("openrouter", "openai/gpt-5.5")
    assert binding.explicit_api_key is None
    assert binding.explicit_base_url is None


def test_review_runtime_legacy_auxiliary_carry_credentials(curator_env, caplog):
    curator = curator_env["curator"]
    cfg = {
        "model": {"provider": "openrouter", "default": "openai/gpt-5.5"},
        "curator": {
            "auxiliary": {
                "provider": "custom",
                "model": "m",
                "api_key": "legacy-key",
                "base_url": "http://legacy/v1",
            },
        },
    }
    import logging
    with caplog.at_level(logging.INFO, logger="agent.curator"):
        binding = curator._resolve_review_runtime(cfg)
    assert binding.explicit_api_key == "legacy-key"
    assert binding.explicit_base_url == "http://legacy/v1"
    assert any("deprecated curator.auxiliary" in rec.message for rec in caplog.records)


def test_review_model_auxiliary_curator_partial_override_falls_back(curator_env):
    """Only one of slot provider/model set → fall back to the main pair.

    Prevents half-configured overrides from sending an empty side to
    resolve_runtime_provider.
    """
    curator = curator_env["curator"]
    base_main = {"provider": "openrouter", "default": "openai/gpt-5.5"}

    cfg_provider_only = {
        "model": dict(base_main),
        "auxiliary": {"curator": {"provider": "openrouter", "model": ""}},
    }
    assert curator._resolve_review_model(cfg_provider_only) == (
        "openrouter", "openai/gpt-5.5",
    )

    cfg_model_only = {
        "model": dict(base_main),
        "auxiliary": {"curator": {"provider": "auto", "model": "gpt-5.4-mini"}},
    }
    assert curator._resolve_review_model(cfg_model_only) == (
        "openrouter", "openai/gpt-5.5",
    )


def test_review_model_legacy_curator_auxiliary_still_works(curator_env, caplog):
    """Pre-unification users set curator.auxiliary.{provider,model} — honor it.

    Emits a deprecation log line but keeps their config working.
    """
    curator = curator_env["curator"]
    cfg = {
        "model": {"provider": "openrouter", "default": "openai/gpt-5.5"},
        "curator": {
            "auxiliary": {
                "provider": "openrouter",
                "model": "openai/gpt-5.4-mini",
            },
        },
    }
    import logging
    with caplog.at_level(logging.INFO, logger="agent.curator"):
        result = curator._resolve_review_model(cfg)
    assert result == ("openrouter", "openai/gpt-5.4-mini")
    assert any(
        "deprecated curator.auxiliary" in rec.message for rec in caplog.records
    ), "expected deprecation warning when legacy curator.auxiliary is used"


def test_review_model_new_slot_wins_over_legacy(curator_env):
    """When BOTH new and legacy are set, the canonical slot wins."""
    curator = curator_env["curator"]
    cfg = {
        "model": {"provider": "openrouter", "default": "openai/gpt-5.5"},
        "auxiliary": {
            "curator": {"provider": "nous", "model": "new-winner"},
        },
        "curator": {
            "auxiliary": {"provider": "openrouter", "model": "legacy-loser"},
        },
    }
    assert curator._resolve_review_model(cfg) == ("nous", "new-winner")


def test_review_model_handles_missing_sections(curator_env):
    """Missing auxiliary/curator sections never raise — fall back cleanly."""
    curator = curator_env["curator"]
    cfg = {"model": {"provider": "anthropic", "model": "claude-sonnet-4-6"}}
    assert curator._resolve_review_model(cfg) == (
        "anthropic", "claude-sonnet-4-6",
    )

    # Completely empty config → ("auto", "") — resolve_runtime_provider
    # handles the auto-detection chain from there.
    assert curator._resolve_review_model({}) == ("auto", "")


def test_curator_slot_is_canonical_aux_task():
    """Curator must be a first-class slot in every aux-task registry.

    Four sources of truth, all checked by the shared registry test
    (test_aux_config.py) for the main tasks — this test pins `curator`
    specifically so the unification doesn't silently regress.
    """
    from hermes_cli.config import DEFAULT_CONFIG
    from hermes_cli.main import _AUX_TASKS
    from hermes_cli.web_server import _AUX_TASK_SLOTS

    # 1. DEFAULT_CONFIG.auxiliary — schema source
    assert "curator" in DEFAULT_CONFIG["auxiliary"], \
        "curator missing from DEFAULT_CONFIG['auxiliary']"
    slot = DEFAULT_CONFIG["auxiliary"]["curator"]
    assert slot["provider"] == "auto"
    assert slot["model"] == ""
    assert slot["timeout"] > 0, "curator timeout should be set (reviews run long)"

    # 2. hermes_cli/main.py _AUX_TASKS — CLI picker
    aux_keys = {k for k, _name, _desc in _AUX_TASKS}
    assert "curator" in aux_keys, "curator missing from _AUX_TASKS (CLI picker)"

    # 3. hermes_cli/web_server.py _AUX_TASK_SLOTS — REST API allowlist
    assert "curator" in _AUX_TASK_SLOTS, \
        "curator missing from _AUX_TASK_SLOTS (dashboard REST API)"

    # 4. web/src/pages/ModelsPage.tsx is checked at build time; the tsx
    #    array and this tuple share a ``Must match _AUX_TASK_SLOTS`` comment.


def test_review_fork_runs_under_background_review_origin(curator_env, monkeypatch):
    """The curator LLM fork must tag itself as background_review.

    This is the keystone that makes skill_manager_tool's
    ``_background_review_write_guard`` fire during a curation pass. Without
    ``_memory_write_origin = "background_review"`` on the fork, the agent
    inherits the default ``assistant_tool`` origin, ``is_background_review()``
    stays False, and the external/bundled/hub-installed skill_manage guards
    never trigger — leaving the LLM agent free to mutate skills.external_dirs
    skills (GH-47688). turn_context.py binds this attribute onto the
    write-origin ContextVar at turn start, so asserting it is set is the
    enforceable invariant linking the fork to the guard.
    """
    curator = curator_env["curator"]

    # The curator_env fixture stubs out _run_llm_review wholesale; this test
    # exercises the real implementation, so reload the module to restore it.
    import importlib
    importlib.reload(curator)
    monkeypatch.setattr(curator, "_load_config", lambda: {})

    captured = {}

    class _StubAgent:
        def __init__(self, *args, **kwargs):
            # AIAgent.__init__ normally sets this default; mirror it so the
            # production assignment in _run_llm_review is what flips it.
            self._memory_write_origin = "assistant_tool"
            self._memory_nudge_interval = 10
            self._skill_nudge_interval = 10
            self.platform = kwargs.get("platform")
            self._session_messages = []

        def run_conversation(self, user_message=None, **kwargs):
            # Capture the origin AT RUN TIME — i.e. after _run_llm_review has
            # finished configuring the fork, which is exactly when it matters.
            captured["write_origin"] = self._memory_write_origin
            return {"final_response": "no change"}

    monkeypatch.setattr("run_agent.AIAgent", _StubAgent)

    meta = curator._run_llm_review("review prompt")

    assert meta.get("error") is None, meta.get("error")
    assert captured.get("write_origin") == "background_review", (
        "curator review fork did not set _memory_write_origin to "
        "'background_review' — the skill_manage background-review write "
        "guard would not fire (GH-47688 regression)"
    )
