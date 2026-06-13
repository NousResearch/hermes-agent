"""Tests for agent.coding_context — RuntimeMode seam, resolver, toolset, git probe."""

import json
import os
import subprocess
import shutil
import tempfile
from pathlib import Path

import pytest

from agent import coding_context as cc


def test_coding_guidance_advertises_persistent_terminal_state():
    assert "Terminal state persists across calls" in cc.CODING_AGENT_GUIDANCE
    assert "Activate a virtualenv" in cc.CODING_AGENT_GUIDANCE
    assert "instead of re-sourcing it before every test command" in cc.CODING_AGENT_GUIDANCE


def _git_init(path):
    env = {
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
        "HOME": str(path),
    }
    for args in (
        ["init", "-q", "-b", "main"],
        ["commit", "-q", "--allow-empty", "-m", "init commit"],
    ):
        subprocess.run([shutil.which("git"), "-C", str(path), *args], check=True, env=env)


# ── resolver ──────────────────────────────────────────────────────────────

class TestIsCodingContext:
    def test_off_never_activates(self, tmp_path):
        _git_init(tmp_path)
        cfg = {"agent": {"coding_context": "off"}}
        assert cc.is_coding_context(platform="cli", cwd=tmp_path, config=cfg) is False

    def test_on_forces_even_without_git(self, tmp_path):
        cfg = {"agent": {"coding_context": "on"}}
        assert cc.is_coding_context(platform="telegram", cwd=tmp_path, config=cfg) is True

    def test_auto_requires_git_repo(self, tmp_path):
        cfg = {"agent": {"coding_context": "auto"}}
        assert cc.is_coding_context(platform="cli", cwd=tmp_path, config=cfg) is False
        _git_init(tmp_path)
        assert cc.is_coding_context(platform="cli", cwd=tmp_path, config=cfg) is True

    def test_auto_skips_messaging_surfaces(self, tmp_path):
        _git_init(tmp_path)
        cfg = {"agent": {"coding_context": "auto"}}
        assert cc.is_coding_context(platform="discord", cwd=tmp_path, config=cfg) is False
        assert cc.is_coding_context(platform="tui", cwd=tmp_path, config=cfg) is True

    def test_default_mode_is_auto(self, tmp_path):
        # Unknown/missing value normalizes to auto.
        _git_init(tmp_path)
        assert cc.is_coding_context(platform="cli", cwd=tmp_path, config={}) is True


# ── toolset substitution ────────────────────────────────────────────────────

class TestCodingSelection:
    def test_selects_coding_under_focus(self, tmp_path):
        _git_init(tmp_path)
        cfg = {"agent": {"coding_context": "focus"}}
        out = cc.coding_selection(platform="cli", cwd=tmp_path, config=cfg)
        assert out is not None
        assert out[0] == cc.CODING_TOOLSET

    def test_auto_is_prompt_only(self, tmp_path):
        # Default posture must never override the user's configured toolsets —
        # off-by-default toolsets are already off, and explicit opt-ins
        # (image-gen, spotify, …) survive entering a code workspace.
        _git_init(tmp_path)
        cfg = {"agent": {"coding_context": "auto"}}
        assert cc.coding_selection(platform="cli", cwd=tmp_path, config=cfg) is None
        # …while the prompt posture is still active.
        assert cc.is_coding_context(platform="cli", cwd=tmp_path, config=cfg) is True

    def test_on_is_prompt_only(self, tmp_path):
        cfg = {"agent": {"coding_context": "on"}}
        assert cc.coding_selection(platform="cli", cwd=tmp_path, config=cfg) is None
        assert cc.is_coding_context(platform="cli", cwd=tmp_path, config=cfg) is True

    def test_focus_requires_workspace(self, tmp_path):
        # focus inherits auto's detection gate — bare dir stays general.
        cfg = {"agent": {"coding_context": "focus"}}
        assert cc.coding_selection(platform="cli", cwd=tmp_path, config=cfg) is None

    def test_none_when_inactive(self, tmp_path):
        cfg = {"agent": {"coding_context": "off"}}
        assert cc.coding_selection(platform="cli", cwd=tmp_path, config=cfg) is None

    def test_coding_toolset_is_registered(self):
        from toolsets import resolve_toolset

        tools = resolve_toolset(cc.CODING_TOOLSET)
        # Coding essentials present…
        for t in ("read_file", "write_file", "patch", "search_files", "terminal", "todo"):
            assert t in tools
        # …and the noise is gone.
        for t in ("send_message", "text_to_speech", "image_generate", "computer_use"):
            assert t not in tools


# ── git/workspace probe ─────────────────────────────────────────────────────

class TestWorkspaceBlock:
    def test_empty_outside_repo(self, tmp_path):
        assert cc.build_coding_workspace_block(tmp_path) == ""

    def test_reports_branch_and_clean_status(self, tmp_path):
        _git_init(tmp_path)
        block = cc.build_coding_workspace_block(tmp_path)
        assert "Workspace" in block
        assert f"Root: {tmp_path.resolve()}" in block or "Root:" in block
        assert "Branch: main" in block
        assert "Status: clean" in block
        assert "init commit" in block

    def test_reports_dirty_counts(self, tmp_path):
        _git_init(tmp_path)
        (tmp_path / "untracked.txt").write_text("hi")
        block = cc.build_coding_workspace_block(tmp_path)
        assert "untracked" in block
        assert "clean" not in block.split("Status:")[1].splitlines()[0]


# ── project facts (verify-loop detection) ───────────────────────────────────

class TestProjectFacts:
    def test_package_json_scripts_surface_verify_commands(self, tmp_path):
        _git_init(tmp_path)
        (tmp_path / "package.json").write_text(
            json.dumps({"scripts": {"test": "vitest", "lint": "eslint .", "dev": "vite"}})
        )
        (tmp_path / "pnpm-lock.yaml").write_text("")
        block = cc.build_coding_workspace_block(tmp_path)
        assert "Project: package.json (pnpm)" in block
        assert "pnpm run test" in block and "pnpm run lint" in block
        # Non-verify scripts (dev servers, …) stay out of the snapshot.
        assert "run dev" not in block

    def test_pytest_config_and_run_tests_script(self, tmp_path):
        _git_init(tmp_path)
        (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        (scripts / "run_tests.sh").write_text("#!/bin/sh\n")
        block = cc.build_coding_workspace_block(tmp_path)
        assert "scripts/run_tests.sh" in block
        assert "pytest" in block.split("Verify:")[1]

    def test_makefile_verify_targets_only(self, tmp_path):
        _git_init(tmp_path)
        (tmp_path / "Makefile").write_text("test:\n\tgo test ./...\n\ndeploy:\n\t./deploy.sh\n")
        block = cc.build_coding_workspace_block(tmp_path)
        assert "make test" in block
        assert "make deploy" not in block

    def test_context_files_listed(self, tmp_path):
        _git_init(tmp_path)
        (tmp_path / "AGENTS.md").write_text("# rules")
        block = cc.build_coding_workspace_block(tmp_path)
        assert "Context files: AGENTS.md" in block

    def test_worktree_detected_without_primary_path(self, tmp_path):
        # A linked worktree should be detected, but the output must NOT contain
        # the absolute path to the primary tree — exposing that path causes the
        # model to sometimes run commands in the wrong directory.
        main_tree = tmp_path / "main"
        main_tree.mkdir()
        _git_init(main_tree)
        worktree = tmp_path / "worktree"
        subprocess.run(
            ["git", "-C", str(main_tree), "worktree", "add", "-b", "wt-branch", str(worktree)],
            check=True,
            env={"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path),
                 "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                 "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"},
        )
        block = cc.build_coding_workspace_block(worktree)
        assert "Worktree: linked" in block
        # The primary tree path must NOT appear anywhere in the output.
        assert str(main_tree.resolve()) not in block
        assert str(main_tree) not in block
        # The worktree root IS the reported root.
        assert f"Root: {worktree.resolve()}" in block or "Root:" in block

    def test_marker_only_project_gets_snapshot_without_git(self, tmp_path):
        # A non-git project (manifest only) still gets a workspace snapshot —
        # just without the git lines.
        (tmp_path / "package.json").write_text("{}")
        block = cc.build_coding_workspace_block(tmp_path)
        assert f"Root: {tmp_path.resolve()}" in block
        assert "package.json" in block
        assert "Branch:" not in block and "Status:" not in block

    def test_malformed_package_json_is_ignored(self, tmp_path):
        _git_init(tmp_path)
        (tmp_path / "package.json").write_text("{not json")
        block = cc.build_coding_workspace_block(tmp_path)
        assert "Project: package.json" in block
        assert "Verify:" not in block


# ── $HOME dotfiles guard ────────────────────────────────────────────────────

class TestHomeDotfilesGuard:
    def test_dotfiles_repo_at_home_is_not_coding(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        home.mkdir()
        _git_init(home)
        monkeypatch.setattr(Path, "home", lambda: home)
        cfg = {"agent": {"coding_context": "auto"}}
        assert cc.is_coding_context(platform="cli", cwd=home, config=cfg) is False
        # …and a plain subdirectory of the dotfiles repo stays general too.
        docs = home / "Documents"
        docs.mkdir()
        assert cc.is_coding_context(platform="cli", cwd=docs, config=cfg) is False

    def test_marker_at_home_is_not_a_project_signal(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        home.mkdir()
        (home / "Makefile").write_text("all:\n")
        monkeypatch.setattr(Path, "home", lambda: home)
        cfg = {"agent": {"coding_context": "auto"}}
        assert cc.is_coding_context(platform="cli", cwd=home, config=cfg) is False

    def test_real_project_under_dotfiles_home_still_detects(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        home.mkdir()
        _git_init(home)
        monkeypatch.setattr(Path, "home", lambda: home)
        proj = home / "www" / "app"
        proj.mkdir(parents=True)
        (proj / "package.json").write_text("{}")
        cfg = {"agent": {"coding_context": "auto"}}
        assert cc.is_coding_context(platform="cli", cwd=proj, config=cfg) is True

    def test_on_mode_bypasses_the_guard(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: home)
        cfg = {"agent": {"coding_context": "on"}}
        assert cc.is_coding_context(platform="cli", cwd=home, config=cfg) is True


# ── prompt assembly integration ─────────────────────────────────────────────

class TestStatusParsing:
    def test_parse_status_counts_and_branch(self):
        porcelain = (
            "# branch.head feature\n"
            "# branch.upstream origin/feature\n"
            "# branch.ab +2 -1\n"
            "1 M. N... 100644 100644 100644 aaa bbb staged.py\n"
            "1 .M N... 100644 100644 100644 ccc ddd modified.py\n"
            "? new.py\n"
            "u UU N... 1 2 3 abc def conflict.py\n"
        )
        branch, counts = cc._parse_status(porcelain)
        assert branch["head"] == "feature"
        assert branch["upstream"] == "origin/feature"
        assert branch["ahead"] == "2" and branch["behind"] == "1"
        assert counts["staged"] == 1
        assert counts["modified"] == 1
        assert counts["untracked"] == 1
        assert counts["conflicts"] == 1


# ── RuntimeMode seam ────────────────────────────────────────────────────────

class TestRuntimeMode:
    def test_resolves_coding_in_repo(self, tmp_path):
        _git_init(tmp_path)
        mode = cc.resolve_runtime_mode(platform="cli", cwd=tmp_path, config={})
        assert mode.is_coding is True
        assert mode.kind == "coding"
        assert mode.profile is cc.CODING_PROFILE

    def test_resolves_general_outside_workspace(self, tmp_path):
        mode = cc.resolve_runtime_mode(platform="cli", cwd=tmp_path, config={})
        assert mode.is_coding is False
        assert mode.kind == "general"
        # General posture pins no toolset and injects no blocks.
        assert mode.toolset_selection() is None
        assert mode.system_blocks() == []

    def test_is_frozen(self, tmp_path):
        mode = cc.resolve_runtime_mode(platform="cli", cwd=tmp_path, config={})
        with pytest.raises(Exception):
            mode.profile = cc.CODING_PROFILE  # type: ignore[misc]

    def test_system_blocks_include_brief_and_workspace(self, tmp_path):
        _git_init(tmp_path)
        mode = cc.resolve_runtime_mode(platform="cli", cwd=tmp_path, config={"agent": {"coding_context": "on"}})
        blocks = mode.system_blocks()
        assert any("coding agent" in b for b in blocks)
        assert any("Workspace" in b for b in blocks)

    def test_toolset_selection_gated_on_focus(self, tmp_path):
        _git_init(tmp_path)
        focus = cc.resolve_runtime_mode(platform="cli", cwd=tmp_path, config={"agent": {"coding_context": "focus"}})
        sel = focus.toolset_selection()
        assert sel and sel[0] == cc.CODING_TOOLSET
        # auto/on resolve the coding profile but stay prompt-only.
        for raw in ("auto", "on"):
            mode = cc.resolve_runtime_mode(platform="cli", cwd=tmp_path, config={"agent": {"coding_context": raw}})
            assert mode.is_coding is True
            assert mode.toolset_selection() is None


# ── edit-format steering (per-model harness tuning) ──────────────────────────

class TestEditFormatSteering:
    def test_family_detection(self):
        assert cc._model_family("openai/gpt-5.4") == "patch"
        assert cc._model_family("openai/codex-mini") == "patch"
        assert cc._model_family("anthropic/claude-opus-4.8") == "replace"
        assert cc._model_family("anthropic/claude-sonnet-4") == "replace"
        # Gemini + open-weight coding models (RL'd on str_replace-style
        # editors) steer to replace, not neutral.
        for m in (
            "google/gemini-3-pro", "deepseek-v3.2", "qwen3-coder",
            "moonshot/kimi-k2", "zai/glm-4.6", "nousresearch/hermes-4-405b",
        ):
            assert cc._model_family(m) == "replace"
        # Unknown family and no model both fall through to neutral wording.
        assert cc._model_family("acme/foo-1") is None
        assert cc._model_family(None) is None
        assert cc._model_family("") is None

    def test_openai_family_gets_v4a_nudge(self, tmp_path):
        _git_init(tmp_path)
        mode = cc.resolve_runtime_mode(
            platform="cli", cwd=tmp_path,
            config={"agent": {"coding_context": "on"}}, model="openai/gpt-5.4",
        )
        brief = mode.system_blocks()[0]
        assert "mode='patch'" in brief
        assert "V4A" in brief
        assert "write_file" in brief  # new files authored, not patched
        # Codex-family harnesses ship apply_patch (V4A) as the ONLY editor and
        # instruct it even for single-file edits — never nudge replace mode.
        assert "single-file" in brief
        assert "mode='replace'" not in brief

    def test_anthropic_family_gets_replace_nudge(self, tmp_path):
        _git_init(tmp_path)
        mode = cc.resolve_runtime_mode(
            platform="cli", cwd=tmp_path,
            config={"agent": {"coding_context": "on"}},
            model="anthropic/claude-opus-4.8",
        )
        brief = mode.system_blocks()[0]
        assert "mode='replace'" in brief
        assert "write_file" in brief  # new files authored, not patched

    def test_unknown_model_keeps_neutral_brief(self, tmp_path):
        # No edit-format line appended — brief equals the bare profile guidance.
        _git_init(tmp_path)
        mode = cc.resolve_runtime_mode(
            platform="cli", cwd=tmp_path,
            config={"agent": {"coding_context": "on"}}, model="acme/foo-1",
        )
        assert mode.system_blocks()[0] == cc.CODING_AGENT_GUIDANCE

    def test_no_model_keeps_neutral_brief(self, tmp_path):
        _git_init(tmp_path)
        mode = cc.resolve_runtime_mode(
            platform="cli", cwd=tmp_path,
            config={"agent": {"coding_context": "on"}},
        )
        assert mode.system_blocks()[0] == cc.CODING_AGENT_GUIDANCE

    def test_general_posture_emits_nothing_regardless_of_model(self, tmp_path):
        # Edit steering only fires inside the coding posture.
        mode = cc.resolve_runtime_mode(
            platform="telegram", cwd=tmp_path, config={}, model="openai/gpt-5.4",
        )
        assert mode.system_blocks() == []


# ── profile registry ────────────────────────────────────────────────────────

class TestProfiles:
    def test_registered_profiles(self):
        assert cc.get_profile("coding") is cc.CODING_PROFILE
        assert cc.get_profile("general") is cc.GENERAL_PROFILE

    def test_unknown_profile_falls_back_to_general(self):
        assert cc.get_profile("nonsense") is cc.GENERAL_PROFILE

    def test_coding_profile_shape(self):
        # The coding profile declares the seams other domains read.
        assert cc.CODING_PROFILE.toolset == cc.CODING_TOOLSET
        assert cc.CODING_PROFILE.guidance
        assert cc.CODING_PROFILE.model_hint == "coding"
        # General is inert.
        assert cc.GENERAL_PROFILE.toolset is None
        assert cc.GENERAL_PROFILE.guidance == ""

    def test_skill_demotion_gated_on_focus(self, tmp_path):
        # Names-only demotion is opt-in via focus mode — the default (auto)
        # and forced (on) postures leave the skill index untouched. Under
        # focus, clearly-non-coding categories are demoted (never hidden) and
        # coding-adjacent ones keep full entries (deny-list semantics).
        _git_init(tmp_path)
        for raw in ("auto", "on"):
            mode = cc.resolve_runtime_mode(
                platform="cli", cwd=tmp_path, config={"agent": {"coding_context": raw}}
            )
            assert mode.is_coding is True
            assert mode.compact_skill_categories() == frozenset()
        focus = cc.resolve_runtime_mode(
            platform="cli", cwd=tmp_path,
            config={"agent": {"coding_context": "focus"}},
        )
        assert focus.is_coding is True
        compact = focus.compact_skill_categories()
        assert "social-media" in compact and "smart-home" in compact
        for kept in ("github", "devops", "software-development", "data-science"):
            assert kept not in compact
        # General posture demotes nothing.
        general = cc.resolve_runtime_mode(platform="telegram", cwd=tmp_path, config={})
        assert general.compact_skill_categories() == frozenset()


# ── detection signals ───────────────────────────────────────────────────────

class TestDetection:
    @pytest.mark.parametrize("marker", ["pyproject.toml", "package.json", "go.mod", "AGENTS.md"])
    def test_project_manifest_triggers_without_git(self, tmp_path, marker):
        (tmp_path / marker).write_text("x")
        cfg = {"agent": {"coding_context": "auto"}}
        assert cc.is_coding_context(platform="cli", cwd=tmp_path, config=cfg) is True

    def test_marker_in_parent_counts_from_subdir(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("x")
        sub = tmp_path / "src" / "pkg"
        sub.mkdir(parents=True)
        cfg = {"agent": {"coding_context": "auto"}}
        assert cc.is_coding_context(platform="cli", cwd=sub, config=cfg) is True

    def test_bare_dir_is_not_coding(self, tmp_path):
        cfg = {"agent": {"coding_context": "auto"}}
        assert cc.is_coding_context(platform="cli", cwd=tmp_path, config=cfg) is False


# ── deleted-CWD tolerance (caller-side getcwd guard) ─────────────────────────

class TestDeletedCwdTolerance:
    """resolve_agent_cwd() propagates OSError when the process CWD is deleted
    (no TERMINAL_CWD / session override) — by contract coding_context owns the
    guard. The fallback must NOT re-call os.getcwd(): that re-raises the same
    error and blows up posture resolution / workspace-block building.
    """

    @staticmethod
    def _kill_getcwd(monkeypatch):
        def _raise(*args, **kwargs):
            raise FileNotFoundError("cwd gone")

        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        # os is the same module object across coding_context and runtime_cwd,
        # so this kills os.getcwd() everywhere — exactly what a deleted CWD does.
        monkeypatch.setattr(os, "getcwd", _raise)

    def test_resolve_runtime_mode_survives_getcwd_failure(self, monkeypatch, tmp_path):
        # Home is a bare dir (no git/markers) so detection deterministically
        # settles on general — and, critically, never raises.
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: home)
        self._kill_getcwd(monkeypatch)
        mode = cc.resolve_runtime_mode(
            platform="cli", config={"agent": {"coding_context": "auto"}}
        )
        assert mode.is_coding is False
        assert mode.kind == "general"

    def test_resolve_runtime_mode_off_survives_getcwd_failure(self, monkeypatch):
        # The candidate's exact repro: off-mode posture resolution must not raise.
        self._kill_getcwd(monkeypatch)
        mode = cc.resolve_runtime_mode(
            platform="cli", config={"agent": {"coding_context": "off"}}
        )
        assert mode.is_coding is False

    def test_build_workspace_block_survives_getcwd_failure(self, monkeypatch, tmp_path):
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: home)
        self._kill_getcwd(monkeypatch)
        # No workspace can be detected for a vanished CWD → empty block, no raise.
        assert cc.build_coding_workspace_block() == ""

    def test_explicit_cwd_unaffected_by_getcwd_failure(self, monkeypatch, tmp_path):
        # An explicit cwd never touches os.getcwd(), so a dead getcwd is irrelevant.
        _git_init(tmp_path)
        self._kill_getcwd(monkeypatch)
        mode = cc.resolve_runtime_mode(
            platform="cli", cwd=tmp_path, config={"agent": {"coding_context": "auto"}}
        )
        assert mode.is_coding is True
        assert "Branch: main" in cc.build_coding_workspace_block(tmp_path)

    def test_resolve_cwd_degrades_to_home(self, monkeypatch, tmp_path):
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: home)
        self._kill_getcwd(monkeypatch)
        # Degrades to home (not the temp dir) — compared via the same getcwd-free
        # resolve the production path uses, so the assertion holds on Windows too
        # (where even Path.resolve() of an absolute path would call os.getcwd()).
        result = cc._resolve_cwd(None)
        assert result == cc._home()
        assert result.name == "home"

    def test_resolve_cwd_degrades_to_tempdir_when_home_also_dead(self, monkeypatch):
        def _no_home():
            raise RuntimeError("no home")

        monkeypatch.setattr(Path, "home", _no_home)
        self._kill_getcwd(monkeypatch)
        result = cc._resolve_cwd(None)
        assert result.is_absolute()
        assert result == Path(tempfile.gettempdir())
