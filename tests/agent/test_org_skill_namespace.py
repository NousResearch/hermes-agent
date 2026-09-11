"""M2 org-skill namespace: token-gated resolution, provenance, collisions.

Covers the design agreed 2026-07-23 (bare-name first-class org skills):
 1. TOKEN-GATED discovery — only the `.active_org`-marked mirror resolves;
    stale mirrors and marker-less trees never load.
 2. Fail-loud collisions — a personal/org name clash lists BOTH sides flagged;
    skill_view's existing multi-candidate guard refuses the bare name.
 3. Load-time provenance header — org skill content announces org + author.
 4. Org mirrors are read-only (skill_manage guards) and curation-exempt.
"""

import json

import pytest

from agent import skill_utils as sku
from agent.prompt_builder import _build_snapshot_entry


def _mk_skill(root, rel, name=None, body="# body\n"):
    d = root
    for part in rel.split("/"):
        d = d / part
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name or rel.split('/')[-1]}\ndescription: d\n---\n{body}",
        encoding="utf-8",
    )
    return d


def _mark_active(skills, org_id):
    org_root = skills / sku.ORG_MIRROR_DIR_NAME
    org_root.mkdir(parents=True, exist_ok=True)
    (org_root / sku.ORG_ACTIVE_MARKER).write_text(org_id, encoding="utf-8")


class TestTokenGatedDiscovery:
    def test_no_marker_no_org_skills(self, tmp_path):
        skills = tmp_path / "skills"
        _mk_skill(skills, "personal-a")
        _mk_skill(skills, f"{sku.ORG_MIRROR_DIR_NAME}/org-1/shared-x", name="shared-x")
        found = [p.parent.name for p in sku.iter_skill_index_files(skills, "SKILL.md")]
        assert "personal-a" in found
        assert "shared-x" not in found  # unmarked mirror never resolves

    def test_marker_gates_to_active_org_only(self, tmp_path):
        skills = tmp_path / "skills"
        _mk_skill(skills, f"{sku.ORG_MIRROR_DIR_NAME}/org-1/shared-x", name="shared-x")
        _mk_skill(skills, f"{sku.ORG_MIRROR_DIR_NAME}/org-OLD/stale-y", name="stale-y")
        _mark_active(skills, "org-1")
        found = [p.parent.name for p in sku.iter_skill_index_files(skills, "SKILL.md")]
        assert "shared-x" in found
        assert "stale-y" not in found  # stale mirror pruned at resolution

    def test_switching_org_flips_resolution(self, tmp_path):
        skills = tmp_path / "skills"
        _mk_skill(skills, f"{sku.ORG_MIRROR_DIR_NAME}/org-1/shared-x", name="shared-x")
        _mk_skill(skills, f"{sku.ORG_MIRROR_DIR_NAME}/org-2/other-z", name="other-z")
        _mark_active(skills, "org-2")
        found = [p.parent.name for p in sku.iter_skill_index_files(skills, "SKILL.md")]
        assert found and "other-z" in found and "shared-x" not in found

    def test_helpers(self, tmp_path):
        skills = tmp_path / "skills"
        d = _mk_skill(skills, f"{sku.ORG_MIRROR_DIR_NAME}/org-9/cat/sk", name="sk")
        assert sku.is_org_mirror_path(d, skills) is True
        assert sku.org_id_of_path(d, skills) == "org-9"
        p = _mk_skill(skills, "plain")
        assert sku.is_org_mirror_path(p, skills) is False
        assert sku.read_active_org_id(skills) is None
        _mark_active(skills, "org-9")
        assert sku.read_active_org_id(skills) == "org-9"


class TestSnapshotEntryProvenance:
    def test_org_entry_strips_prefix_and_carries_provenance(self, tmp_path):
        skills = tmp_path / "skills"
        d = _mk_skill(
            skills, f"{sku.ORG_MIRROR_DIR_NAME}/org-1/devops/beta", name="beta"
        )
        (skills / sku.ORG_MIRROR_DIR_NAME / "org-1" / sku.ORG_PROVENANCE_FILE).write_text(
            json.dumps(
                {"author_device": "bens-macbook-a1b2c3", "author_user_id": "u1"}
            ),
            encoding="utf-8",
        )
        entry = _build_snapshot_entry(d / "SKILL.md", skills, {"name": "beta"}, "d")
        assert entry["org_id"] == "org-1"
        assert entry["org_author"] == "bens-macbook-a1b2c3"
        # Category derives from the path WITHIN the mirror, not _org/org-1/...
        assert entry["category"] == "devops"
        assert entry["skill_name"] == "beta"

    def test_personal_entry_unchanged(self, tmp_path):
        skills = tmp_path / "skills"
        d = _mk_skill(skills, "devops/beta", name="beta")
        entry = _build_snapshot_entry(d / "SKILL.md", skills, {"name": "beta"}, "d")
        assert "org_id" not in entry
        assert entry["category"] == "devops"


class TestListingCollisionsAndLabels:
    def _render(self, tmp_path, monkeypatch):
        from agent import prompt_builder as pb

        skills = tmp_path / "skills"
        skills.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(pb, "get_skills_dir", lambda: skills, raising=True)
        monkeypatch.setattr(
            pb, "get_all_skills_dirs", lambda: [skills], raising=True
        )
        monkeypatch.setattr(pb, "get_disabled_skill_names", lambda *a, **k: set())
        monkeypatch.setattr(
            pb, "_skills_prompt_snapshot_path", lambda: tmp_path / "snap.json"
        )
        pb.clear_skills_system_prompt_cache()
        return skills, pb

    def test_org_skill_listed_with_provenance_tag(self, tmp_path, monkeypatch):
        skills, pb = self._render(tmp_path, monkeypatch)
        _mk_skill(skills, "personal-a")
        _mk_skill(skills, f"{sku.ORG_MIRROR_DIR_NAME}/org-1/shared-x", name="shared-x")
        (skills / sku.ORG_MIRROR_DIR_NAME / "org-1" / sku.ORG_PROVENANCE_FILE).write_text(
            json.dumps({"author_device": "bens-macbook"}), encoding="utf-8"
        )
        _mark_active(skills, "org-1")
        out = pb.build_skills_system_prompt()
        assert "org:org-1" in out
        assert "[org-shared: by bens-macbook]" in out
        assert "personal-a" in out

    def test_collision_flags_both_sides(self, tmp_path, monkeypatch):
        skills, pb = self._render(tmp_path, monkeypatch)
        _mk_skill(skills, "k8s-debug", body="personal version\n")
        _mk_skill(
            skills, f"{sku.ORG_MIRROR_DIR_NAME}/org-1/k8s-debug", name="k8s-debug"
        )
        _mark_active(skills, "org-1")
        out = pb.build_skills_system_prompt()
        # BOTH entries flagged — neither silently wins.
        assert out.count("[name collision") == 2

    def test_no_collision_flag_when_unique(self, tmp_path, monkeypatch):
        skills, pb = self._render(tmp_path, monkeypatch)
        _mk_skill(skills, "personal-a")
        _mk_skill(skills, f"{sku.ORG_MIRROR_DIR_NAME}/org-1/shared-x", name="shared-x")
        _mark_active(skills, "org-1")
        out = pb.build_skills_system_prompt()
        assert "[name collision" not in out


class TestOrgSkillsAreEditableInPlace:
    """The learning loop must work ON shared skills, not around them.

    Refusing edits to `_org/` froze exactly the skills the most people use:
    the agent is instructed to patch a skill the moment it finds a gap, and
    "fork it to a personal skill first" is not something an agent does
    mid-task. So edits land in place; org updates never clobber them; the
    user (or auto-propose) shares them back.
    """

    def _org_skill(self, tmp_path, monkeypatch):
        from tools import skill_manager_tool as smt
        from agent import skill_utils as _sku

        skills = tmp_path / "skills"
        d = _mk_skill(
            skills, f"{sku.ORG_MIRROR_DIR_NAME}/org-1/shared-x", name="shared-x"
        )
        _mark_active(skills, "org-1")
        monkeypatch.setattr(smt, "_skills_dir", lambda: skills)
        monkeypatch.setattr(
            _sku, "get_all_skills_dirs", lambda: [skills], raising=True
        )
        return smt, skills, d

    def test_patch_is_allowed_and_applied(self, tmp_path, monkeypatch):
        smt, _skills, d = self._org_skill(tmp_path, monkeypatch)
        result = smt._patch_skill("shared-x", "body", "improved")
        assert result["success"] is True, result.get("error")
        assert "improved" in (d / "SKILL.md").read_text(encoding="utf-8")

    def test_edit_tells_the_user_how_to_share_it_back(self, tmp_path, monkeypatch):
        smt, _skills, _d = self._org_skill(tmp_path, monkeypatch)
        result = smt._patch_skill("shared-x", "body", "improved")
        # Without auto-propose the edit stays local, and the tool result must
        # say so AND name the command — otherwise the improvement is stranded.
        assert "propose" in (result.get("org_sharing") or "")

    def test_delete_is_still_refused(self, tmp_path, monkeypatch):
        smt, _skills, d = self._org_skill(tmp_path, monkeypatch)
        guard = smt._org_mirror_write_guard("shared-x", d, "delete")
        assert guard is not None and guard["success"] is False
        assert "admin" in guard["error"]

    def test_curation_is_allowed(self, tmp_path, monkeypatch):
        from tools import skill_usage as su

        skills = tmp_path / "skills"
        d = _mk_skill(
            skills, f"{sku.ORG_MIRROR_DIR_NAME}/org-1/shared-x", name="shared-x"
        )
        monkeypatch.setattr(su, "_skills_dir", lambda: skills)
        # The curator must be able to improve shared skills — they are the
        # highest-leverage ones in the system.
        assert su.is_curation_eligible("shared-x", d) is True


class TestOrgPullIsWiredIn:
    """Guards the integration gap that unit tests structurally cannot catch.

    The org pull functions were fully implemented and unit-tested while having
    ZERO runtime callers, so org skills never loaded for anyone. Testing the
    functions directly could never surface that. These tests assert the CALL
    SITES exist, so the feature can't silently become dead code again.
    """

    def test_session_startup_calls_maybe_pull_org_skills(self):
        import pathlib

        cli_src = (
            pathlib.Path(__file__).resolve().parents[2] / "cli.py"
        ).read_text(encoding="utf-8")
        assert "maybe_pull_org_skills" in cli_src, (
            "cli.py session startup must call maybe_pull_org_skills() — "
            "without a call site the org mirror is never populated and org "
            "skills never load (the function being importable is not enough)."
        )
        # It must sit alongside the personal pull, not replace it.
        assert "maybe_pull_skills" in cli_src

    def test_sync_pull_command_refreshes_org_mirror(self):
        import pathlib

        main_src = (
            pathlib.Path(__file__).resolve().parents[2]
            / "hermes_cli"
            / "main_platform_setup.py"
        ).read_text(encoding="utf-8")
        assert "maybe_pull_org_skills" in main_src, (
            "`hermes sync pull` must also refresh the org mirror."
        )

    def test_sync_status_exposes_org_state(self):
        from tools import skills_sync_client as ssc

        status = ssc.sync_status()
        # These keys must always be present so a user can tell whether the org
        # workflow applies to them, rather than it being invisible.
        for key in ("org_available", "org_id", "org_role", "org_skills"):
            assert key in status, f"sync status must expose {key!r}"

    def test_no_internal_jargon_in_user_facing_strings(self):
        """User-visible help/errors must not leak internal design coordinates."""
        import pathlib
        import re

        root = pathlib.Path(__file__).resolve().parents[2]
        targets = [
            root / "hermes_cli" / "subcommands" / "sync.py",
            root / "hermes_cli" / "subcommands" / "skills.py",
        ]
        banned = re.compile(
            r"\(M[12]\)|\bHSP\b|HSP/1|§[0-9]|DEV-PHASE|hsp-1-contract"
        )
        for path in targets:
            for i, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
                if "help=" in line or "description=" in line:
                    assert not banned.search(line), (
                        f"{path.name}:{i} leaks internal jargon to users: {line.strip()}"
                    )


class TestSkillSyncIsOneCommand:
    """Every Skill Sync verb lives under `hermes sync` for launch.

    The surface is deliberately encapsulated: one command to learn, one to
    document, and top-level `sync` stays free of skill-management verbs that
    belong elsewhere. `propose` in particular used to sit under `hermes
    skills`, which split one feature across two commands.
    """

    def _src(self, *parts):
        import pathlib

        return (
            pathlib.Path(__file__).resolve().parents[2].joinpath(*parts)
        ).read_text(encoding="utf-8")

    def test_propose_is_a_sync_subcommand(self):
        sync_src = self._src("hermes_cli", "subcommands", "sync.py")
        assert '"propose"' in sync_src, (
            "`propose` must be a `hermes sync` subcommand."
        )

    def test_propose_is_not_under_skills(self):
        skills_src = self._src("hermes_cli", "subcommands", "skills.py")
        assert '"propose"' not in skills_src, (
            "`propose` must NOT remain under `hermes skills` — Skill Sync is "
            "one command for launch."
        )

    def test_sync_usage_lists_propose(self):
        main_src = self._src("hermes_cli", "main_platform_setup.py")
        usage_start = main_src.index("usage: hermes sync ")
        usage_block = main_src[usage_start : usage_start + 1400]
        assert "propose" in usage_block, (
            "`hermes sync` usage must list the propose verb."
        )


class TestLocalEditsSurviveOrgUpdates:
    """Ben's requirement: local edits are never silently overwritten.

    An org pull materializes the shared set. Before this, it `rmtree`'d each
    skill dir and re-wrote it, so any local improvement vanished on the next
    session start with no warning. Now a locally-modified skill is skipped
    and reported as a conflict for the user to resolve deliberately.
    """

    def _mirror(self, tmp_path, monkeypatch, body="original\n"):
        from tools import skills_sync_client as ssc

        skills = tmp_path / "skills"
        d = _mk_skill(
            skills,
            f"{sku.ORG_MIRROR_DIR_NAME}/org-1/shared-x",
            name="shared-x",
            body=body,
        )
        _mark_active(skills, "org-1")
        monkeypatch.setattr(ssc, "_skills_dir", lambda: skills)
        monkeypatch.setattr(
            ssc, "_org_dir", lambda: skills / sku.ORG_MIRROR_DIR_NAME
        )
        from tools import skills_sync_client_org as org
        return org, skills, d

    def test_unmodified_skill_is_not_flagged(self, tmp_path, monkeypatch):
        org, _skills, d = self._mirror(tmp_path, monkeypatch)
        org._write_org_baseline(
            "org-1",
            {"shared-x": {"fingerprint": org._skill_dir_fingerprint(d), "tree": "t1"}},
        )
        assert org.org_skill_is_locally_modified("shared-x", "org-1") is False
        assert org.list_locally_modified_org_skills("org-1") == []

    def test_edited_skill_is_detected(self, tmp_path, monkeypatch):
        org, _skills, d = self._mirror(tmp_path, monkeypatch)
        org._write_org_baseline(
            "org-1",
            {"shared-x": {"fingerprint": org._skill_dir_fingerprint(d), "tree": "t1"}},
        )
        (d / "SKILL.md").write_text("---\nname: shared-x\n---\nEDITED\n", encoding="utf-8")
        assert org.org_skill_is_locally_modified("shared-x", "org-1") is True
        assert org.list_locally_modified_org_skills("org-1") == ["shared-x"]

    def test_missing_baseline_does_not_cry_wolf(self, tmp_path, monkeypatch):
        org, _skills, _d = self._mirror(tmp_path, monkeypatch)
        # Mirror pulled before baselines existed — must not be reported as
        # modified (that would block every update with a phantom conflict).
        assert org.org_skill_is_locally_modified("shared-x", "org-1") is False

    def test_fingerprint_is_content_based_not_mtime(self, tmp_path, monkeypatch):
        import os
        import time

        org, _skills, d = self._mirror(tmp_path, monkeypatch)
        before = org._skill_dir_fingerprint(d)
        time.sleep(0.01)
        os.utime(d / "SKILL.md", None)  # touch: mtime changes, content doesn't
        assert org._skill_dir_fingerprint(d) == before

    def test_auto_propose_defaults_off(self, monkeypatch):
        from tools import skills_sync_client as ssc

        monkeypatch.delenv("HERMES_SYNC_ORG_AUTO_PROPOSE", raising=False)
        monkeypatch.setattr(
            "hermes_cli.config.load_config", lambda: {}, raising=False
        )
        # Default must be OFF: silently pushing every agent edit to the whole
        # organisation is not a safe default.
        assert ssc.sync_org_auto_propose() is False

    def test_auto_propose_can_be_enabled_by_env(self, monkeypatch):
        from tools import skills_sync_client as ssc

        monkeypatch.setenv("HERMES_SYNC_ORG_AUTO_PROPOSE", "1")
        assert ssc.sync_org_auto_propose() is True
