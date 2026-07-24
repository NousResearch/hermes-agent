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


class TestOrgMirrorReadOnly:
    def test_skill_manage_patch_refuses_org_mirror(self, tmp_path, monkeypatch):
        from tools import skill_manager_tool as smt

        skills = tmp_path / "skills"
        _mk_skill(skills, f"{sku.ORG_MIRROR_DIR_NAME}/org-1/shared-x", name="shared-x")
        _mark_active(skills, "org-1")
        monkeypatch.setattr(smt, "_skills_dir", lambda: skills)
        from agent import skill_utils as _sku
        monkeypatch.setattr(
            _sku, "get_all_skills_dirs", lambda: [skills], raising=True
        )
        result = smt._patch_skill("shared-x", "body", "hacked")
        assert result["success"] is False
        assert "ORG-SHARED" in result["error"]
        assert "propose" in result["error"]

    def test_curation_exempt(self, tmp_path, monkeypatch):
        from tools import skill_usage as su

        skills = tmp_path / "skills"
        d = _mk_skill(skills, f"{sku.ORG_MIRROR_DIR_NAME}/org-1/shared-x", name="shared-x")
        monkeypatch.setattr(su, "_skills_dir", lambda: skills)
        assert su.is_curation_eligible("shared-x", d) is False
