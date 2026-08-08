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





def test_curator_defaults(curator_env):
    c = curator_env["curator"]
    assert c.get_interval_hours() == 24 * 7  # 7 days
    assert c.get_min_idle_hours() == 2
    assert c.get_stale_after_days() == 30
    assert c.get_archive_after_days() == 90




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








def test_set_paused_roundtrip(curator_env):
    c = curator_env["curator"]
    c.set_paused(True)
    assert c.is_paused() is True
    c.set_paused(False)
    assert c.is_paused() is False


# ---------------------------------------------------------------------------
# Automatic state transitions
# ---------------------------------------------------------------------------





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


def test_prune_builtins_reanchors_preseeded_never_used_builtin(curator_env, monkeypatch):
    """A pre-seeded built-in record must not mark the skill stale on the first
    curator run.

    Telemetry writes a usage record the moment a built-in is seeded — possibly
    weeks before the first curator run with ``curator.prune_builtins`` on. That
    record has use_count 0, no activity, and a created_at older than
    stale_after_days; anchoring the inactivity clock there marks every fresh
    built-in stale on the first pass. The curator re-anchors created_at to now
    on first sight instead, so the clock starts from this run (#79295).
    """
    u = curator_env["usage"]
    c = curator_env["curator"]
    skills_dir = curator_env["home"] / "skills"
    _write_skill(skills_dir, "bundled-helper")
    (skills_dir / ".bundled_manifest").write_text(
        "bundled-helper:abc\n", encoding="utf-8"
    )
    _enable_prune_builtins(curator_env, monkeypatch)

    # Pure-telemetry record batch-created before the first curator run.
    super_old = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
    data = u.load_usage()
    data["bundled-helper"] = u._empty_record()
    data["bundled-helper"]["created_at"] = super_old
    u.save_usage(data)

    counts = c.apply_automatic_transitions()
    assert counts["marked_stale"] == 0
    assert counts["archived"] == 0
    rec = u.get_record("bundled-helper")
    assert rec["state"] == "active"
    # Clock was re-anchored to this run, not left at the telemetry timestamp.
    assert rec["first_seen_at"] is not None
    reanchored = datetime.fromisoformat(rec["created_at"])
    assert reanchored > datetime.fromisoformat(super_old)
    assert reanchored > datetime.now(timezone.utc) - timedelta(minutes=5)


def test_prune_builtins_reanchored_builtin_still_ages_out(curator_env, monkeypatch):
    """Re-anchoring is one-shot: after first sight, the re-anchored clock keeps
    ticking, so a still-unused built-in goes stale after a full stale window
    instead of being deferred forever."""
    u = curator_env["usage"]
    c = curator_env["curator"]
    skills_dir = curator_env["home"] / "skills"
    _write_skill(skills_dir, "bundled-helper")
    (skills_dir / ".bundled_manifest").write_text(
        "bundled-helper:abc\n", encoding="utf-8"
    )
    _enable_prune_builtins(curator_env, monkeypatch)

    super_old = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
    data = u.load_usage()
    data["bundled-helper"] = u._empty_record()
    data["bundled-helper"]["created_at"] = super_old
    u.save_usage(data)

    t0 = datetime.now(timezone.utc)
    assert c.apply_automatic_transitions(now=t0)["marked_stale"] == 0

    # A month later (past stale_after_days=30) with still no use: stale.
    counts = c.apply_automatic_transitions(now=t0 + timedelta(days=31))
    assert counts["marked_stale"] == 1
    assert u.get_record("bundled-helper")["state"] == "stale"


def test_prune_builtins_used_builtin_still_marked_stale(curator_env, monkeypatch):
    """The never-used re-anchor must not shield built-ins that were actually
    used long ago — their activity anchor governs and they age out normally."""
    u = curator_env["usage"]
    c = curator_env["curator"]
    skills_dir = curator_env["home"] / "skills"
    _write_skill(skills_dir, "bundled-used")
    (skills_dir / ".bundled_manifest").write_text(
        "bundled-used:abc\n", encoding="utf-8"
    )
    _enable_prune_builtins(curator_env, monkeypatch)

    # Last used 45 days ago: inside the stale window (>30d) but well before
    # the archive window (90d) — an unequivocal stale candidate.
    old = (datetime.now(timezone.utc) - timedelta(days=45)).isoformat()
    data = u.load_usage()
    data["bundled-used"] = u._empty_record()
    data["bundled-used"]["created_at"] = old
    data["bundled-used"]["last_used_at"] = old
    data["bundled-used"]["use_count"] = 1
    u.save_usage(data)

    counts = c.apply_automatic_transitions()
    assert counts["marked_stale"] == 1
    assert u.get_record("bundled-used")["state"] == "stale"


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






















# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------



def test_state_atomic_write_no_tmp_leftovers(curator_env):
    c = curator_env["curator"]
    c.save_state({"paused": True})
    parent = c._state_file().parent
    tmp_files = [p.name for p in parent.iterdir() if p.name.endswith(".tmp")]
    assert tmp_files == []








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




def test_review_fork_forwards_runtime_pool_and_overrides(curator_env, monkeypatch):
    """Curator must pass credential_pool + request_overrides from resolve_runtime_provider."""
    curator = curator_env["curator"]
    import importlib
    importlib.reload(curator)

    fake_pool = object()
    fake_overrides = {"extra_body": {"store": False}}
    captured = {}

    def _fake_resolve_runtime_provider(**kwargs):
        return {
            "provider": "custom",
            "api_key": "pool-token",
            "base_url": "https://hyper.charm.land/v1",
            "api_mode": "chat_completions",
            "credential_pool": fake_pool,
            "request_overrides": fake_overrides,
        }

    class _StubAgent:
        def __init__(self, *args, **kwargs):
            captured["kwargs"] = kwargs
            self._memory_write_origin = "assistant_tool"
            self._memory_nudge_interval = 0
            self._skill_nudge_interval = 0
            self._session_messages = []

        def run_conversation(self, user_message=None, **kwargs):
            return {"final_response": "ok"}

        def close(self):
            pass

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"model": {"provider": "custom:hyper-charm", "default": "glm-5.2"}},
    )
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"model": {"provider": "custom:hyper-charm", "default": "glm-5.2"}},
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        _fake_resolve_runtime_provider,
    )
    monkeypatch.setattr("run_agent.AIAgent", _StubAgent)

    meta = curator._run_llm_review("review prompt")

    assert meta.get("error") is None, meta.get("error")
    assert captured["kwargs"]["credential_pool"] is fake_pool
    assert captured["kwargs"]["request_overrides"] == fake_overrides


def test_review_fork_uses_runtime_model_and_output_cap(curator_env, monkeypatch):
    curator = curator_env["curator"]
    import importlib
    importlib.reload(curator)
    captured = {}

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"model": {"provider": "custom:gateway", "default": "gateway"}},
    )
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"model": {"provider": "custom:gateway", "default": "gateway"}},
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: {
            "provider": "custom",
            "model": "real-model-id",
            "api_key": "test-key",
            "base_url": "https://gateway.example/v1",
            "api_mode": "chat_completions",
            "max_output_tokens": 1234,
        },
    )

    class _StubAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self._session_messages = []

        def run_conversation(self, **_kwargs):
            return {"final_response": "ok"}

        def close(self):
            pass

    monkeypatch.setattr("run_agent.AIAgent", _StubAgent)
    result = curator._run_llm_review("review")

    assert result["error"] is None
    assert captured["model"] == "real-model-id"
    assert captured["max_tokens"] == 1234




def test_review_fork_restricts_toolsets_to_skills_and_terminal(curator_env, monkeypatch):
    """The curator LLM fork must advertise only the skills + terminal toolsets.

    Without ``enabled_toolsets=["skills", "terminal"]`` on the AIAgent(...) call
    in ``_run_llm_review``, ``enabled_toolsets`` defaults to None and init_agent
    grants the fork the full default catalog (~30 tools) plus the context_engine
    (lcm_*) tools, billing ~7K wasted schema tokens on every one of the fork's
    50-100 API calls per consolidation pass. The prompt (curator.py:509-523)
    confines the model to four tools in natural language, but only this kwarg
    filters the advertised request schema. Capturing the constructor kwarg is
    the sole assertion that distinguishes fixed from unfixed code.
    """
    curator = curator_env["curator"]

    # curator_env stubs _run_llm_review wholesale; exercise the real
    # implementation, so reload the module to restore it.
    import importlib
    importlib.reload(curator)

    captured = {}

    class _StubAgent:
        def __init__(self, *args, **kwargs):
            captured["enabled_toolsets"] = kwargs.get("enabled_toolsets", "UNSET")
            self._memory_write_origin = "assistant_tool"
            self._memory_nudge_interval = 0
            self._skill_nudge_interval = 0
            self._session_messages = []

        def run_conversation(self, user_message=None, **kwargs):
            return {"final_response": "no change"}

        def close(self):
            pass

    monkeypatch.setattr("run_agent.AIAgent", _StubAgent)

    meta = curator._run_llm_review("review prompt")

    # error is None proves the fork was actually constructed (capture ran).
    assert meta.get("error") is None, meta.get("error")
    assert captured.get("enabled_toolsets") == ["skills", "terminal"], (
        "curator review fork did not pass enabled_toolsets=['skills', "
        "'terminal'] to AIAgent; the full default tool catalog (plus lcm_* "
        "context_engine tools) would be advertised; got "
        f"{captured.get('enabled_toolsets')!r}"
    )


def test_review_fork_toolset_surface_is_skills_plus_terminal():
    """Documentary check on the static surface the fork's kwarg resolves to.

    Registry-independent (include_registry=False) so a plugin-registered tool
    tagged into these toolsets cannot flake the membership checks. This
    documents the intended surface (the four prompt-named tools present, dead
    default and lcm_* schema absent) but does not itself guard the call-site
    kwarg. No exact-set pin: intentional additions to either toolset must not
    fail this test.
    """
    from toolsets import resolve_toolset

    surface = set(resolve_toolset("skills", include_registry=False)) | set(
        resolve_toolset("terminal", include_registry=False)
    )

    # The four prompt-named tools are all present.
    assert "skills_list" in surface
    assert "skill_view" in surface
    assert "skill_manage" in surface
    assert "terminal" in surface

    # Representative dropped default + context_engine tools are absent.
    assert "read_file" not in surface
    assert "web_search" not in surface
    assert "lcm_grep" not in surface
