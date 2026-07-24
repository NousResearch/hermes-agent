"""HER-96 — contract tests for mechanical HER/SCA loop isolation."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "factory_lane.py"


def run_lane(registry: Path, *args: str, check: bool = False):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--registry", str(registry), *args],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if check and result.returncode != 0:
        raise AssertionError(result.stderr or result.stdout)
    return result


def load_factory_lane():
    spec = importlib.util.spec_from_file_location("factory_lane_her96", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_team_config(path: Path) -> Path:
    config = {
        "freshness": {
            "canonical_branch": "origin/main",
            "max_age_seconds": 300,
        },
        "teams": {
            "HER": {
                "profiles": ["default"],
                "allowed_teams": ["HER"],
                "job_id": "job-her",
                "gateway_started_at": "2026-07-23T14:00:00Z",
            },
            "SCA": {
                "profiles": ["hermes-immo"],
                "allowed_teams": ["SCA"],
                "job_id": "job-sca",
                "gateway_started_at": "2026-07-23T14:00:00Z",
            },
        },
    }
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def write_freshness_evidence(
    path: Path,
    issue: str,
    *,
    verdict: str = "current",
    checked_at: float | None = None,
    needs_rebase_red: bool = True,
    branch: str = "origin/main",
) -> Path:
    evidence = {
        "issue": issue,
        "checked_at": time.time() if checked_at is None else checked_at,
        "canonical_branch": branch,
        "canonical_head": "a" * 40,
        "sources": {
            "newer_linear_issues": ["HER-97"],
            "newer_prs_commits": ["a" * 40],
            "current_main_behavior": {"checked": True, "summary": "current-main probed"},
        },
        "verdict": verdict,
    }
    if verdict == "needs-rebase" and needs_rebase_red:
        evidence["current_main_red"] = {
            "reproduced": True,
            "evidence": "RED: current main fails",
        }
    path.write_text(json.dumps(evidence), encoding="utf-8")
    return path


def team_admit(registry: Path, config: Path, issue: str, profile: str, worktree: Path,
               evidence: Path, session: str = "session", check: bool = False):
    return run_lane(
        registry,
        "team-admit",
        issue,
        "--team-config",
        str(config),
        "--profile",
        profile,
        "--agent",
        "hermes-code" if profile == "default" else "hermes-immo",
        "--session",
        session,
        "--worktree",
        str(worktree),
        "--freshness-evidence",
        str(evidence),
        check=check,
    )


def run_lane_bounded(registry: Path, *args: str):
    """Run an untrusted-file path with a short bound so a FIFO cannot hang CI."""
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--registry", str(registry), *args],
        capture_output=True,
        text=True,
        timeout=2,
    )


def test_dual_team_slots_allow_her_and_sca_simultaneously(tmp_path):
    registry = tmp_path / "registry"
    config = write_team_config(tmp_path / "teams.json")
    her_worktree = tmp_path / "her"
    sca_worktree = tmp_path / "sca"
    her_worktree.mkdir()
    sca_worktree.mkdir()

    team_admit(
        registry,
        config,
        "HER-96",
        "default",
        her_worktree,
        write_freshness_evidence(tmp_path / "her-freshness.json", "HER-96"),
        check=True,
    )
    team_admit(
        registry,
        config,
        "SCA-616",
        "hermes-immo",
        sca_worktree,
        write_freshness_evidence(tmp_path / "sca-freshness.json", "SCA-616"),
        check=True,
    )

    status = run_lane(registry, "team-status", "--team-config", str(config), "--json", check=True)
    payload = json.loads(status.stdout)
    assert payload["teams"]["HER"]["profile"] == "default"
    assert payload["teams"]["HER"]["lane"] == "HER-96"
    assert payload["teams"]["SCA"]["profile"] == "hermes-immo"
    assert payload["teams"]["SCA"]["lane"] == "SCA-616"


def test_team_admission_rejects_cross_routing_and_unknown_profiles(tmp_path):
    registry = tmp_path / "registry"
    config = write_team_config(tmp_path / "teams.json")
    worktree = tmp_path / "repo"
    worktree.mkdir()

    her_selects_sca = run_lane(
        registry, "team-admit", "SCA-616", "--team-config", str(config),
        "--profile", "default", "--agent", "hermes-code", "--session", "wrong",
        "--worktree", str(worktree),
    )
    assert her_selects_sca.returncode != 0
    assert "profile default cannot own lane SCA-616" in her_selects_sca.stderr

    sca_selects_her = run_lane(
        registry, "team-admit", "HER-96", "--team-config", str(config),
        "--profile", "hermes-immo", "--agent", "hermes-immo", "--session", "wrong",
        "--worktree", str(worktree),
    )
    assert sca_selects_her.returncode != 0
    assert "profile hermes-immo cannot own lane HER-96" in sca_selects_her.stderr

    unknown = run_lane(
        registry, "team-admit", "HER-96", "--team-config", str(config),
        "--profile", "mystery", "--agent", "mystery", "--session", "unknown",
        "--worktree", str(worktree),
    )
    assert unknown.returncode != 0
    assert "not mapped to any team" in unknown.stderr


def test_out_of_scope_team_admission_does_not_create_an_absent_registry_root(tmp_path, monkeypatch):
    """Reject invalid controller routing before making registry state observable."""
    registry = tmp_path / "registry"
    module = load_factory_lane()
    config = {
        "teams": {
            "HER": {"profiles": ["default"], "allowed_teams": ["HER"]},
            "SCA": {"profiles": ["hermes-immo"], "allowed_teams": ["SCA"]},
        },
    }
    worktree = tmp_path / "repo"
    worktree.mkdir()
    monkeypatch.setattr(module, "load_team_config", lambda _path: config)
    monkeypatch.setattr(
        module,
        "_safe_registry_root",
        lambda _path: (_ for _ in ()).throw(AssertionError("registry root was created")),
    )

    assert module.main([
        "--registry", str(registry), "team-admit", "SCA-616",
        "--team-config", str(tmp_path / "teams.json"), "--profile", "default",
        "--agent", "hermes-code", "--session", "wrong", "--worktree", str(worktree),
    ]) == 1
    assert not registry.exists()


def test_explicit_mandate_can_authorize_the_same_profile_for_a_new_team(tmp_path):
    """Team routing is mandate data, never a profile-name conditional in code."""
    registry = tmp_path / "registry"
    config = write_team_config(tmp_path / "new-mandate.json")
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["teams"]["HER"]["allowed_teams"] = ["SCA"]
    config.write_text(json.dumps(payload), encoding="utf-8")
    worktree = tmp_path / "sca-under-new-mandate"
    worktree.mkdir()

    admitted = team_admit(
        registry,
        config,
        "SCA-616",
        "default",
        worktree,
        write_freshness_evidence(tmp_path / "freshness.json", "SCA-616"),
        check=True,
    )

    assert admitted.returncode == 0
    owner = json.loads((registry / "locks" / "SCA-616" / "owner.json").read_text())
    assert owner["profile"] == "default"
    assert owner["team"] == "HER"
    assert owner["allowed_teams"] == ["SCA"]


def test_second_owner_for_same_team_is_rejected_even_on_different_worktree(tmp_path):
    registry = tmp_path / "registry"
    config = write_team_config(tmp_path / "teams.json")
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    team_admit(
        registry, config, "HER-96", "default", first,
        write_freshness_evidence(tmp_path / "first-freshness.json", "HER-96"),
        session="s1", check=True,
    )

    result = team_admit(
        registry, config, "HER-97", "default", second,
        write_freshness_evidence(tmp_path / "second-freshness.json", "HER-97"),
        session="s2",
    )

    assert result.returncode != 0
    assert "team HER already claimed by HER-96" in result.stderr


def test_same_canonical_worktree_rejected_across_teams(tmp_path):
    registry = tmp_path / "registry"
    config = write_team_config(tmp_path / "teams.json")
    worktree = tmp_path / "shared"
    worktree.mkdir()
    team_admit(
        registry, config, "HER-96", "default", worktree,
        write_freshness_evidence(tmp_path / "her-freshness.json", "HER-96"),
        check=True,
    )

    result = team_admit(
        registry, config, "SCA-616", "hermes-immo", worktree,
        write_freshness_evidence(tmp_path / "sca-freshness.json", "SCA-616"),
    )

    assert result.returncode != 0
    assert "worktree already claimed" in result.stderr


def test_unrelated_profile_without_team_controller_keeps_legacy_admission(tmp_path):
    registry = tmp_path / "registry"
    worktree = tmp_path / "imp"
    worktree.mkdir()

    run_lane(
        registry, "admit", "IMP-12", "--mode", "owner",
        "--profile", "impact", "--domain-prefixes", "IMP",
        "--agent", "impact", "--session", "imp", "--worktree", str(worktree),
        check=True,
    )


def test_team_status_is_runtime_derived_and_ignores_direct_execution_as_recurrence(monkeypatch, tmp_path):
    module = load_factory_lane()
    registry = tmp_path / "registry"
    config = write_team_config(tmp_path / "teams.json")
    worktree = tmp_path / "her"
    worktree.mkdir()
    team_admit(
        registry, config, "HER-96", "default", worktree,
        write_freshness_evidence(tmp_path / "freshness.json", "HER-96"),
        check=True,
    )

    def load_profile_status(_profile, job_id):
        jobs = {
            "job-her": {"id": "job-her", "next_run_at": "2026-07-23T15:00:00+02:00"},
            "job-sca": {"id": "job-sca", "next_run_at": "2026-07-23T15:05:00+02:00"},
        }
        latest = {
            "job-her": {"id": "manual", "job_id": "job-her", "source": "direct", "status": "completed", "claimed_at": "2026-07-23T14:10:00Z"},
            "job-sca": {"id": "builtin", "job_id": "job-sca", "source": "builtin", "status": "completed", "claimed_at": "2026-07-23T14:10:00Z"},
        }
        builtin = {job_id: latest[job_id]} if latest[job_id]["source"] == "builtin" else {}
        return jobs, {job_id: latest[job_id]}, builtin

    monkeypatch.setattr(module, "_load_profile_cron_status", load_profile_status)

    status = module.build_team_status(
        module._safe_registry_root(str(registry)),
        module.load_team_config(str(config)),
    )

    assert status["teams"]["HER"]["runtime_status_source"] == "registry"
    assert status["teams"]["HER"]["last_builtin_execution"] is None
    assert status["teams"]["HER"]["latest_execution"]["source"] == "direct"
    assert status["teams"]["SCA"]["last_builtin_execution"]["id"] == "builtin"
    assert status["teams"]["HER"]["next_run_at"] == "2026-07-23T15:00:00+02:00"
    assert status["teams"]["SCA"]["last_builtin_tick_after_gateway_start"]["id"] == "builtin"
    assert status["teams"]["HER"]["worker"]["agent"] == "hermes-code"
    assert status["teams"]["HER"]["heartbeat"] is not None
    assert status["teams"]["HER"]["gate"]["freshness"]["verdict"] == "current"
    assert "conversation" not in status["teams"]["HER"]


def test_team_status_reads_jobs_and_executions_from_each_configured_profile(tmp_path, monkeypatch):
    """HER/SCA status must not read both cron stores from the caller's profile."""
    from cron import executions, jobs
    from cron.executions import use_execution_store
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    module = load_factory_lane()
    registry = tmp_path / "registry"
    config = write_team_config(tmp_path / "teams.json")
    default_home = tmp_path / "hermes"
    sca_home = default_home / "profiles" / "hermes-immo"
    default_home.mkdir(parents=True)
    sca_home.mkdir(parents=True)

    import hermes_constants
    monkeypatch.setattr(hermes_constants, "_get_platform_default_hermes_home", lambda: default_home)
    monkeypatch.setenv("HERMES_HOME", str(default_home))

    for home, job_id in ((default_home, "job-her"), (sca_home, "job-sca")):
        token = set_hermes_home_override(home)
        try:
            with jobs.use_cron_store(home), use_execution_store(home):
                job = jobs.create_job("status probe", "every 1h", deliver="local")
                job["id"] = job_id
                jobs.save_jobs([job])
                execution = executions.create_execution(job_id, source="builtin")
                executions.finish_execution(execution["id"], success=True)
        finally:
            reset_hermes_home_override(token)

    status = module.build_team_status(
        module._safe_registry_root(str(registry)),
        module.load_team_config(str(config)),
    )

    assert status["teams"]["HER"]["next_run_at"] is not None
    assert status["teams"]["HER"]["last_builtin_execution"]["job_id"] == "job-her"
    assert status["teams"]["SCA"]["next_run_at"] is not None
    assert status["teams"]["SCA"]["last_builtin_execution"]["job_id"] == "job-sca"


def test_team_status_projects_latest_execution_without_raw_error(monkeypatch, tmp_path):
    """Status may expose execution state, never a raw provider error or internals."""
    module = load_factory_lane()
    registry = tmp_path / "registry"
    config = write_team_config(tmp_path / "teams.json")
    sentinel = "Bearer HER96_STATUS_SENTINEL_DO_NOT_EMIT"

    monkeypatch.setattr(module, "_load_cron_jobs_by_id", lambda: {})
    monkeypatch.setattr(module, "_load_latest_cron_executions", lambda _job_ids: {
        "job-her": {
            "id": "execution-her",
            "job_id": "job-her",
            "source": "direct",
            "status": "failed",
            "claimed_at": "2026-07-23T14:10:00Z",
            "started_at": "2026-07-23T14:10:01Z",
            "finished_at": "2026-07-23T14:10:02Z",
            "error": sentinel,
            "process_id": "internal-process-id",
            "pid": 12345,
        },
    })

    status = module.build_team_status(
        module._safe_registry_root(str(registry)),
        module.load_team_config(str(config)),
    )
    latest = status["teams"]["HER"]["latest_execution"]

    assert set(latest) == {
        "id", "job_id", "source", "status", "claimed_at", "started_at", "finished_at", "error",
    }
    assert latest["error"] == "redacted"
    assert sentinel not in json.dumps(status)
    assert "internal-process-id" not in json.dumps(status)


def test_team_admission_requires_fresh_current_evidence_before_owner_mutation(tmp_path):
    registry = tmp_path / "registry"
    config = write_team_config(tmp_path / "teams.json")
    worktree = tmp_path / "repo"
    worktree.mkdir()

    missing = run_lane(
        registry, "team-admit", "HER-96", "--team-config", str(config),
        "--profile", "default", "--agent", "hermes-code", "--session", "s1",
        "--worktree", str(worktree),
    )

    assert missing.returncode != 0
    assert "freshness" in missing.stderr.lower()
    assert not (registry / "locks" / "HER-96" / "owner.json").exists()


def test_team_admission_succeeds_without_opath_from_private_directories(tmp_path):
    """macOS must admit ordinary controller files from a private directory."""
    registry = tmp_path / "registry"
    config = write_team_config(tmp_path / "teams.json")
    worktree = tmp_path / "repo"
    worktree.mkdir()

    admitted = team_admit(
        registry,
        config,
        "HER-96",
        "default",
        worktree,
        write_freshness_evidence(tmp_path / "freshness.json", "HER-96"),
        check=True,
    )

    assert admitted.returncode == 0
    assert (registry / "locks" / "HER-96" / "owner.json").exists()


def test_team_admission_rejects_unknown_freshness_field_before_owner_write(tmp_path):
    """Freshness is an allowlist, not a persisted metadata pass-through."""
    registry = tmp_path / "registry"
    config = write_team_config(tmp_path / "teams.json")
    worktree = tmp_path / "repo"
    worktree.mkdir()
    evidence = write_freshness_evidence(tmp_path / "freshness.json", "HER-96")
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    payload["unreviewed_operator_note"] = "attacker-controlled text"
    evidence.write_text(json.dumps(payload), encoding="utf-8")

    rejected = team_admit(registry, config, "HER-96", "default", worktree, evidence)

    assert rejected.returncode != 0
    assert "unknown" in rejected.stderr.lower()
    assert not (registry / "locks" / "HER-96" / "owner.json").exists()


@pytest.mark.parametrize(
    "credential_text",
    [
        "Authorization: Bearer HER96_SENTINEL_DO_NOT_PERSIST",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJIRVI5NiJ9.signature",
        "ghp_HER96SENTINELDO_NOTPERSIST1234567890",
        "-----BEGIN PRIVATE KEY-----\nHER96_SENTINEL\n-----END PRIVATE KEY-----",
    ],
)
def test_team_admission_rejects_credential_shaped_freshness_summary_before_owner_write(
    tmp_path, credential_text,
):
    """Free text credential forms never reach the persisted status record."""
    registry = tmp_path / "registry"
    config = write_team_config(tmp_path / "teams.json")
    worktree = tmp_path / "repo"
    worktree.mkdir()
    evidence = write_freshness_evidence(tmp_path / "freshness.json", "HER-96")
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    payload["sources"]["current_main_behavior"]["summary"] = credential_text
    evidence.write_text(json.dumps(payload), encoding="utf-8")

    rejected = team_admit(registry, config, "HER-96", "default", worktree, evidence)

    assert rejected.returncode != 0
    assert "credential" in rejected.stderr.lower()
    assert not (registry / "locks" / "HER-96" / "owner.json").exists()


def test_team_status_exposes_only_normalized_allowlisted_freshness(tmp_path):
    """The persisted and emitted freshness record contains only schema fields."""
    registry = tmp_path / "registry"
    config = write_team_config(tmp_path / "teams.json")
    worktree = tmp_path / "repo"
    worktree.mkdir()
    evidence = write_freshness_evidence(tmp_path / "freshness.json", "HER-96")

    team_admit(registry, config, "HER-96", "default", worktree, evidence, check=True)
    owner = json.loads((registry / "locks" / "HER-96" / "owner.json").read_text(encoding="utf-8"))
    status = json.loads(
        run_lane(registry, "team-status", "--team-config", str(config), "--json", check=True).stdout
    )

    expected_keys = {
        "issue", "checked_at", "canonical_branch", "canonical_head", "sources", "verdict",
    }
    assert set(owner["freshness"]) == expected_keys
    assert set(status["teams"]["HER"]["gate"]["freshness"]) == expected_keys
    assert "owner" not in status["teams"]["HER"]
    assert "unreviewed_operator_note" not in json.dumps(status)


def test_dual_team_controller_rejects_any_mapping_other_than_her_and_sca(tmp_path):
    config = json.loads(write_team_config(tmp_path / "teams.json").read_text(encoding="utf-8"))
    config["teams"]["IMP"] = {
        "profiles": ["impact"],
        "prefixes": ["IMP"],
        "job_id": "job-imp",
        "gateway_started_at": "2026-07-23T14:00:00Z",
    }
    path = tmp_path / "invalid-teams.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    result = run_lane(tmp_path / "registry", "team-status", "--team-config", str(path), "--json")

    assert result.returncode != 0
    assert "exactly HER and SCA" in result.stderr


def test_team_admission_rejects_stale_superseded_or_duplicate_freshness(tmp_path):
    registry = tmp_path / "registry"
    config = write_team_config(tmp_path / "teams.json")
    worktree = tmp_path / "repo"
    worktree.mkdir()

    for name, verdict, checked_at in (
        ("stale", "current", time.time() - 301),
        ("superseded", "superseded", time.time()),
        ("duplicate", "duplicate", time.time()),
    ):
        evidence = write_freshness_evidence(
            tmp_path / f"{name}.json", "HER-96", verdict=verdict, checked_at=checked_at,
        )
        result = team_admit(registry, config, "HER-96", "default", worktree, evidence, session=name)
        assert result.returncode != 0
        assert not (registry / "locks" / "HER-96" / "owner.json").exists()


def test_needs_rebase_requires_current_main_red_and_persists_validated_evidence(tmp_path):
    registry = tmp_path / "registry"
    config = write_team_config(tmp_path / "teams.json")
    worktree = tmp_path / "repo"
    worktree.mkdir()
    incomplete = write_freshness_evidence(
        tmp_path / "missing-red.json", "HER-96", verdict="needs-rebase", needs_rebase_red=False,
    )

    missing_red = team_admit(registry, config, "HER-96", "default", worktree, incomplete, session="s1")
    assert missing_red.returncode != 0
    assert "current-main red" in missing_red.stderr.lower()

    valid = write_freshness_evidence(tmp_path / "valid-red.json", "HER-96", verdict="needs-rebase")
    admitted = team_admit(
        registry, config, "HER-96", "default", worktree, valid, session="s2", check=True,
    )
    assert admitted.returncode == 0
    owner = json.loads((registry / "locks" / "HER-96" / "owner.json").read_text())
    assert owner["freshness"]["verdict"] == "needs-rebase"
    assert owner["freshness"]["canonical_head"] == "a" * 40


def test_team_config_fails_closed_when_mandate_allows_an_unknown_lane_team(tmp_path):
    registry = tmp_path / "registry"
    config = write_team_config(tmp_path / "teams.json")
    payload = json.loads(config.read_text())
    payload["teams"]["HER"]["allowed_teams"] = ["IMP"]
    config.write_text(json.dumps(payload), encoding="utf-8")
    worktree = tmp_path / "repo"
    worktree.mkdir()
    evidence = write_freshness_evidence(tmp_path / "freshness.json", "HER-96")

    result = team_admit(registry, config, "HER-96", "default", worktree, evidence)

    assert result.returncode != 0
    assert "allowed_teams" in result.stderr


def test_dual_team_controller_requires_profiles_to_be_singleton_lists(tmp_path):
    module = load_factory_lane()
    config = json.loads(write_team_config(tmp_path / "teams.json").read_text(encoding="utf-8"))
    config["teams"]["HER"]["profiles"] = "d"
    path = tmp_path / "invalid-profiles.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(module.RegistryError, match="must map exactly one profile"):
        module.load_team_config(path)


@pytest.mark.parametrize("invalid_age", [float("nan"), float("inf"), float("-inf")])
def test_team_config_rejects_non_finite_freshness_max_age(tmp_path, invalid_age):
    module = load_factory_lane()
    config = json.loads(write_team_config(tmp_path / "teams.json").read_text(encoding="utf-8"))
    config["freshness"]["max_age_seconds"] = invalid_age
    path = tmp_path / "invalid-age.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(module.RegistryError, match="max_age_seconds"):
        module.load_team_config(path)


@pytest.mark.parametrize("invalid_timestamp", [float("nan"), float("inf"), float("-inf")])
def test_handoff_timestamp_rejects_non_finite_numbers(invalid_timestamp):
    """Non-finite JSON numbers must never bypass stale-handoff checks."""
    module = load_factory_lane()

    assert module._parse_handoff_timestamp(invalid_timestamp) is None


@pytest.mark.parametrize("invalid_age", [0.5, 86_401])
def test_team_config_rejects_non_integral_or_overlong_freshness_max_age(tmp_path, invalid_age):
    """Freshness policies are bounded whole-second windows, never eternal proofs."""
    module = load_factory_lane()
    config = json.loads(write_team_config(tmp_path / "teams.json").read_text(encoding="utf-8"))
    config["freshness"]["max_age_seconds"] = invalid_age
    path = tmp_path / "invalid-age.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(module.RegistryError, match="max_age_seconds"):
        module.load_team_config(path)


def test_freshness_epoch_zero_is_never_admitted_even_at_epoch(monkeypatch, tmp_path):
    """An epoch sentinel is invalid evidence, not a fresh check at process epoch."""
    module = load_factory_lane()
    config_path = write_team_config(tmp_path / "teams.json")
    evidence_path = write_freshness_evidence(tmp_path / "freshness.json", "HER-96", checked_at=0)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    policy = module.load_team_config(config_path)["freshness"]
    monkeypatch.setattr(module.time, "time", lambda: 0)

    with pytest.raises(module.RegistryError, match="checked_at"):
        module._validate_freshness_evidence(evidence, "HER-96", policy)


@pytest.mark.parametrize("invalid_checked_at", [float("nan"), float("inf"), float("-inf")])
def test_team_admission_rejects_non_finite_freshness_checked_at(tmp_path, invalid_checked_at):
    registry = tmp_path / "registry"
    config = write_team_config(tmp_path / "teams.json")
    worktree = tmp_path / "repo"
    worktree.mkdir()
    evidence = write_freshness_evidence(
        tmp_path / "freshness.json", "HER-96", checked_at=invalid_checked_at,
    )

    result = team_admit(registry, config, "HER-96", "default", worktree, evidence)

    assert result.returncode != 0
    assert not (registry / "locks" / "HER-96" / "owner.json").exists()


@pytest.mark.parametrize(
    ("source", "invalid_item"),
    [
        ("newer_linear_issues", {"password": "TOPSECRET"}),
        ("newer_prs_commits", {"nested": ["arbitrary metadata"]}),
    ],
)
def test_team_admission_rejects_unbounded_or_secret_bearing_freshness_source_items(
    tmp_path, source, invalid_item,
):
    registry = tmp_path / "registry"
    config = write_team_config(tmp_path / "teams.json")
    worktree = tmp_path / "repo"
    worktree.mkdir()
    evidence = write_freshness_evidence(tmp_path / "freshness.json", "HER-96")
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    payload["sources"][source].append(invalid_item)
    evidence.write_text(json.dumps(payload), encoding="utf-8")

    result = team_admit(registry, config, "HER-96", "default", worktree, evidence)

    assert result.returncode != 0
    owner_file = registry / "locks" / "HER-96" / "owner.json"
    assert not owner_file.exists()
    assert "TOPSECRET" not in str(owner_file)


def test_team_admission_rejects_excess_freshness_source_items(tmp_path):
    registry = tmp_path / "registry"
    config = write_team_config(tmp_path / "teams.json")
    worktree = tmp_path / "repo"
    worktree.mkdir()
    evidence = write_freshness_evidence(tmp_path / "freshness.json", "HER-96")
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    payload["sources"]["newer_linear_issues"] = ["HER-97"] * 101
    evidence.write_text(json.dumps(payload), encoding="utf-8")

    result = team_admit(registry, config, "HER-96", "default", worktree, evidence)

    assert result.returncode != 0
    assert not (registry / "locks" / "HER-96" / "owner.json").exists()


def test_freshness_manifest_ancestor_swap_cannot_redirect_read(tmp_path, monkeypatch):
    """A swap after pathname validation must fail closed, not read attacker JSON."""
    module = load_factory_lane()
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    manifest = write_freshness_evidence(manifest_dir / "freshness.json", "HER-96")
    evil_dir = tmp_path / "evil"
    evil_dir.mkdir()
    evil = write_freshness_evidence(evil_dir / "freshness.json", "HER-999")
    original_open_secure = module._open_secure
    swapped = {"done": False}

    def swap_ancestor_then_open(path, flags, mode=0o600):
        if not swapped["done"]:
            swapped["done"] = True
            shutil.move(str(manifest_dir), str(tmp_path / "manifest-real"))
            os.symlink(str(evil_dir), str(manifest_dir), target_is_directory=True)
        return original_open_secure(path, flags, mode)

    monkeypatch.setattr(module, "_open_secure", swap_ancestor_then_open)

    with pytest.raises(module.RegistryError):
        module._load_freshness_evidence(manifest)
    assert evil.exists()


def test_freshness_path_does_not_remap_system_aliases_on_non_macos_hosts(monkeypatch):
    module = load_factory_lane()
    monkeypatch.setattr(module, "_is_known_system_alias", lambda _path: False)

    assert module._safe_absolute_path("/tmp/freshness.json") == "/tmp/freshness.json"


def test_same_session_continuation_rebinds_owner_to_live_worker_identity(tmp_path):
    module = load_factory_lane()
    registry = module._safe_registry_root(str(tmp_path / "registry"))
    worktree = tmp_path / "repo"
    worktree.mkdir()
    previous_worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        previous_start = module._get_process_start_time(previous_worker.pid)
        module._claim_under_gate(
            registry, "HER-96", "hermes-code", "continuation", str(worktree), False, 72.0,
            owner_pid=previous_worker.pid, owner_start_time=previous_start,
        )
        live_start = module._get_process_start_time(os.getpid())

        module._claim_under_gate(
            registry, "HER-96", "hermes-code", "continuation", str(worktree), False, 72.0,
            owner_pid=os.getpid(), owner_start_time=live_start,
        )

        owner = module._read_owner_via_chain(str(registry), "HER-96")
        assert owner["pid"] == os.getpid()
        assert owner["process_start_time"] == live_start
    finally:
        previous_worker.terminate()
        previous_worker.wait(timeout=10)


def test_same_session_continuation_rejects_forged_identity_without_rebinding(tmp_path):
    module = load_factory_lane()
    registry = module._safe_registry_root(str(tmp_path / "registry"))
    worktree = tmp_path / "repo"
    worktree.mkdir()
    previous_worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        previous_start = module._get_process_start_time(previous_worker.pid)
        module._claim_under_gate(
            registry, "HER-96", "hermes-code", "continuation", str(worktree), False, 72.0,
            owner_pid=previous_worker.pid, owner_start_time=previous_start,
        )

        with pytest.raises(module.RegistryError, match="owner-start-time"):
            module._claim_under_gate(
                registry, "HER-96", "hermes-code", "continuation", str(worktree), False, 72.0,
                owner_pid=os.getpid(), owner_start_time="forged-start",
            )

        owner = module._read_owner_via_chain(str(registry), "HER-96")
        assert owner["pid"] == previous_worker.pid
        assert owner["process_start_time"] == previous_start
    finally:
        previous_worker.terminate()
        previous_worker.wait(timeout=10)


def test_team_config_rejects_symlinked_ancestor_before_reading_attacker_config(tmp_path):
    """A valid-looking config reached through a symlinked parent is untrusted."""
    module = load_factory_lane()
    config_parent = tmp_path / "controller"
    config_parent.mkdir()
    config = write_team_config(config_parent / "teams.json")
    evil_parent = tmp_path / "evil"
    evil_parent.mkdir()
    evil = json.loads(write_team_config(evil_parent / "teams.json").read_text())
    evil["freshness"]["canonical_branch"] = "attacker-main"
    (evil_parent / "teams.json").write_text(json.dumps(evil), encoding="utf-8")

    shutil.move(str(config_parent), str(tmp_path / "controller-real"))
    os.symlink(str(evil_parent), str(config_parent), target_is_directory=True)

    with pytest.raises(module.RegistryError, match="symlink"):
        module.load_team_config(config)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="named pipes are unavailable")
def test_team_config_fifo_is_rejected_without_blocking(tmp_path):
    config = tmp_path / "teams.fifo"
    os.mkfifo(config)

    result = run_lane_bounded(tmp_path / "registry", "team-status", "--team-config", str(config), "--json")

    assert result.returncode != 0
    assert "regular" in result.stderr.lower()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="named pipes are unavailable")
def test_freshness_evidence_fifo_is_rejected_without_blocking(tmp_path):
    registry = tmp_path / "registry"
    config = write_team_config(tmp_path / "teams.json")
    evidence = tmp_path / "freshness.fifo"
    worktree = tmp_path / "repo"
    worktree.mkdir()
    os.mkfifo(evidence)

    result = run_lane_bounded(
        registry, "team-admit", "HER-96", "--team-config", str(config),
        "--profile", "default", "--agent", "hermes-code", "--session", "fifo",
        "--worktree", str(worktree), "--freshness-evidence", str(evidence),
    )

    assert result.returncode != 0
    assert "regular" in result.stderr.lower()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="named pipes are unavailable")
def test_owner_json_fifo_is_rejected_without_blocking(tmp_path):
    registry = tmp_path / "registry"
    config = write_team_config(tmp_path / "teams.json")
    owner_dir = registry / "locks" / "HER-96"
    owner_dir.mkdir(parents=True)
    os.mkfifo(owner_dir / "owner.json")

    result = run_lane_bounded(registry, "team-status", "--team-config", str(config), "--json")

    assert result.returncode != 0
    assert "regular" in result.stderr.lower()


def test_team_config_rejects_unknown_schema_fields(tmp_path):
    """The dual-team controller is a fixed security policy, not an extensible blob."""
    module = load_factory_lane()
    config = write_team_config(tmp_path / "teams.json")
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["unreviewed_override"] = {"profiles": ["attacker"]}
    config.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(module.RegistryError, match="unknown"):
        module.load_team_config(config)


def test_claim_discovery_fails_closed_when_locks_ancestor_is_symlinked(tmp_path):
    """Admission discovery must not downgrade a swapped locks/ ancestor to empty."""
    module = load_factory_lane()
    root = module._safe_registry_root(str(tmp_path / "registry"))
    evil_locks = tmp_path / "evil-locks"
    evil_locks.mkdir()
    os.symlink(str(evil_locks), str(root / "locks"), target_is_directory=True)

    with pytest.raises(module.RegistryError, match="symlink"):
        module._find_worktree_claim(root, os.path.realpath(str(tmp_path / "repo")))
