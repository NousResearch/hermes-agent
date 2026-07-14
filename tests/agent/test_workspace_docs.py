from __future__ import annotations

from pathlib import Path

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def _with_hermes_home(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return set_hermes_home_override(path)


VALID_DOC = """---
type: hermes-workspace-document
doc_type: skill-template
title: Draft import workflow
workspace_id: /tmp/project
created_at: '2026-07-14T00:00:00Z'
updated_at: '2026-07-14T00:00:00Z'
status: draft
apply_state: unapplied
description: Reusable draft
tags:
  - skills
---
# Body
"""


class TestWorkspaceDocFrontmatter:
    def test_parse_valid_document(self):
        from agent.workspace_docs import (
            WorkspaceDocApplyState,
            WorkspaceDocStatus,
            WorkspaceDocType,
            parse_workspace_doc,
        )

        frontmatter, body = parse_workspace_doc(VALID_DOC)

        assert frontmatter.doc_type is WorkspaceDocType.SKILL_TEMPLATE
        assert frontmatter.title == "Draft import workflow"
        assert frontmatter.workspace_id == "/tmp/project"
        assert frontmatter.status is WorkspaceDocStatus.DRAFT
        assert frontmatter.apply_state is WorkspaceDocApplyState.UNAPPLIED
        assert frontmatter.tags == ("skills",)
        assert body.strip() == "# Body"

    @pytest.mark.parametrize(
        "doc_type",
        [
            "skill-template",
            "memory-note",
            "workspace-instructions",
            "prompt-template",
            "runbook",
            "generic-md",
        ],
    )
    def test_all_initial_doc_types_are_valid(self, doc_type):
        from agent.workspace_docs import validate_workspace_doc_frontmatter

        parsed = validate_workspace_doc_frontmatter(
            {
                "type": "hermes-workspace-document",
                "doc_type": doc_type,
                "title": "Title",
            }
        )

        assert parsed.doc_type.value == doc_type
        assert parsed.status.value == "draft"
        assert parsed.apply_state.value == "unapplied"

    @pytest.mark.parametrize(
        "frontmatter,error",
        [
            ({"doc_type": "runbook", "title": "T"}, "type must be"),
            ({"type": "hermes-workspace-document", "title": "T"}, "doc_type"),
            ({"type": "hermes-workspace-document", "doc_type": "bogus", "title": "T"}, "invalid doc_type"),
            ({"type": "hermes-workspace-document", "doc_type": "runbook", "title": ""}, "title"),
            ({"type": "hermes-workspace-document", "doc_type": "runbook", "title": "T", "tags": "x"}, "tags"),
            ({"type": "hermes-workspace-document", "doc_type": "runbook", "title": "T", "status": "live"}, "status"),
            ({"type": "hermes-workspace-document", "doc_type": "runbook", "title": "T", "apply_state": "active"}, "apply_state"),
        ],
    )
    def test_invalid_frontmatter_rejected(self, frontmatter, error):
        from agent.workspace_docs import WorkspaceDocValidationError, validate_workspace_doc_frontmatter

        with pytest.raises(WorkspaceDocValidationError, match=error):
            validate_workspace_doc_frontmatter(frontmatter)

    def test_missing_frontmatter_rejected(self):
        from agent.workspace_docs import WorkspaceDocValidationError, parse_workspace_doc

        with pytest.raises(WorkspaceDocValidationError, match="missing YAML frontmatter"):
            parse_workspace_doc("# plain markdown")


class TestWorkspaceDocPaths:
    def test_workspace_identity_uses_resolved_root(self, tmp_path):
        from agent.workspace_docs import resolve_workspace_identity

        workspace = tmp_path / "project"
        workspace.mkdir()

        identity = resolve_workspace_identity(workspace)

        assert identity.kind == "workspace"
        assert identity.identity == str(workspace.resolve())
        assert identity.root == workspace.resolve()

    def test_profile_fallback_identity_uses_active_hermes_home(self, tmp_path):
        from agent.workspace_docs import resolve_workspace_identity

        hermes_home = tmp_path / ".hermes"
        token = _with_hermes_home(hermes_home)
        try:
            identity = resolve_workspace_identity()
        finally:
            reset_hermes_home_override(token)

        assert identity.kind == "profile-fallback"
        assert identity.identity == str(hermes_home.resolve())

    def test_defaults_include_workspace_identity_and_inert_state(self, tmp_path):
        from agent.workspace_docs import workspace_doc_frontmatter_defaults

        workspace = tmp_path / "repo"
        workspace.mkdir()

        defaults = workspace_doc_frontmatter_defaults(workspace)

        assert defaults == {
            "type": "hermes-workspace-document",
            "workspace_id": str(workspace.resolve()),
            "status": "draft",
            "apply_state": "unapplied",
        }

    def test_resolves_safe_relative_path_under_workspace_docs(self, tmp_path):
        from agent.workspace_docs import resolve_workspace_doc_path

        workspace = tmp_path / "repo"
        (workspace / ".hermes" / "docs").mkdir(parents=True)

        resolved = resolve_workspace_doc_path(workspace, "runbooks/oncall.md")

        assert resolved == (workspace / ".hermes" / "docs" / "runbooks" / "oncall.md").resolve()

    @pytest.mark.parametrize("relative_path", ["../skills/SKILL.md", "../../outside.md", "/abs/path.md"])
    def test_path_traversal_and_absolute_paths_rejected(self, tmp_path, relative_path):
        from agent.workspace_docs import WorkspaceDocPathError, resolve_workspace_doc_path

        workspace = tmp_path / "repo"
        (workspace / ".hermes" / "docs").mkdir(parents=True)

        with pytest.raises(WorkspaceDocPathError):
            resolve_workspace_doc_path(workspace, relative_path)

    def test_docs_root_symlink_escape_rejected(self, tmp_path):
        from agent.workspace_docs import WorkspaceDocPathError, get_workspace_docs_root

        workspace = tmp_path / "repo"
        outside = tmp_path / "outside"
        (workspace / ".hermes").mkdir(parents=True)
        outside.mkdir()
        (workspace / ".hermes" / "docs").symlink_to(outside, target_is_directory=True)

        with pytest.raises(WorkspaceDocPathError, match="escapes workspace root"):
            get_workspace_docs_root(workspace)

    def test_workspace_inside_hermes_home_is_not_misclassified_as_profile_state(self, tmp_path):
        from agent.workspace_docs import resolve_workspace_doc_path

        hermes_home = tmp_path / ".hermes"
        workspace = hermes_home / "hermes-agent"
        (workspace / ".hermes" / "docs").mkdir(parents=True)
        token = _with_hermes_home(hermes_home)
        try:
            resolved = resolve_workspace_doc_path(workspace, "notes/plan.md")
        finally:
            reset_hermes_home_override(token)

        assert resolved == (workspace / ".hermes" / "docs" / "notes" / "plan.md").resolve()

    def test_workspace_root_inside_other_profile_scoped_area_is_rejected(self, tmp_path, monkeypatch):
        from agent.workspace_docs import WorkspaceDocSafetyError, resolve_workspace_doc_path

        hermes_home = tmp_path / ".hermes"
        other_profile_workspace = hermes_home / "profiles" / "other" / "skills" / "draft-repo"
        (other_profile_workspace / ".hermes" / "docs").mkdir(parents=True)
        # file_safety's root resolver follows HERMES_HOME, while get_hermes_home
        # can also be context-overridden. Set both so this mirrors a real profile
        # process without relying on the developer's ambient ~/.hermes.
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        token = _with_hermes_home(hermes_home)
        try:
            with pytest.raises(WorkspaceDocSafetyError) as exc:
                resolve_workspace_doc_path(other_profile_workspace, "x.md")
        finally:
            reset_hermes_home_override(token)

        assert exc.value.classifier == "cross_profile"

    def test_sandbox_mirror_workspace_root_is_rejected(self, tmp_path):
        from agent.workspace_docs import WorkspaceDocSafetyError, resolve_workspace_doc_path

        workspace = tmp_path / "profiles" / "group1" / "sandboxes" / "docker" / "default" / "home"
        (workspace / ".hermes" / "docs").mkdir(parents=True)

        with pytest.raises(WorkspaceDocSafetyError) as exc:
            resolve_workspace_doc_path(workspace, "x.md")

        assert exc.value.classifier == "sandbox_mirror"

    def test_container_mirror_workspace_root_is_rejected_when_prefix_supplied(self, tmp_path):
        from agent.workspace_docs import WorkspaceDocSafetyError, resolve_workspace_doc_path

        mirror_root = tmp_path / "container-home" / ".hermes"
        workspace = mirror_root.parent
        (workspace / ".hermes" / "docs").mkdir(parents=True)

        with pytest.raises(WorkspaceDocSafetyError) as exc:
            resolve_workspace_doc_path(workspace, "x.md", mirror_prefix=str(mirror_root))

        assert exc.value.classifier == "container_mirror"


class TestWorkspaceDocsInertness:
    def test_workspace_docs_path_shape_detected(self, tmp_path):
        from agent.workspace_docs import is_workspace_docs_path

        assert is_workspace_docs_path(tmp_path / "repo" / ".hermes" / "docs" / "runbook.md")
        assert not is_workspace_docs_path(tmp_path / "repo" / ".hermes" / "skills" / "SKILL.md")

    def test_workspace_docs_are_not_a_skill_discovery_root(self, tmp_path):
        from agent.skill_utils import get_all_skills_dirs
        from agent.workspace_docs import workspace_docs_are_inert

        hermes_home = tmp_path / ".hermes"
        workspace = tmp_path / "repo"
        docs_skill = workspace / ".hermes" / "docs" / "skill-template" / "SKILL.md"
        docs_skill.parent.mkdir(parents=True)
        docs_skill.write_text("---\nname: draft\ndescription: inert\n---\n# Draft\n")

        token = _with_hermes_home(hermes_home)
        try:
            skill_roots = [path.resolve() for path in get_all_skills_dirs()]
        finally:
            reset_hermes_home_override(token)

        assert workspace_docs_are_inert() is True
        assert (workspace / ".hermes" / "docs").resolve() not in skill_roots
        assert hermes_home.joinpath("skills").resolve() in skill_roots
