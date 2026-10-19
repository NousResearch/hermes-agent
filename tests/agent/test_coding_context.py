"""Tests for agent.coding_context — RuntimeMode seam, resolver, toolset, git probe."""

import json
import os
import subprocess
import shutil
from pathlib import Path

import pytest

from agent import coding_context as cc




def _git_init(path):
    env = {
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
        "HOME": str(path),
    }
    # Commit a source file so the fixture is a real *code* workspace: a bare git
    # repo with no code no longer flips into the coding posture (see
    # _detect_profile_name / _has_code_files), so "a code repo" needs code.
    (Path(path) / "main.py").write_text("print('hi')\n")
    for args in (
        ["init", "-q", "-b", "main"],
        # Pin line-ending handling to the repo itself: with a host-global
        # core.autocrlf=true, a just-committed tree can report "1 modified"
        # immediately after init (CRLF round-trip), which poisons the
        # clean-status contract this suite asserts.
        ["config", "core.autocrlf", "false"],
        ["add", "-A"],
        ["commit", "-q", "-m", "init commit"],
    ):
        subprocess.run([shutil.which("git"), "-C", str(path), *args], check=True, env=env)


# ── resolver ──────────────────────────────────────────────────────────────

class TestIsCodingContext:



    def test_auto_bare_git_repo_without_code_stays_general(self, tmp_path):
        # A git repo of only prose (notes/writing/research — a big non-coding use
        # case) is NOT a code workspace: .git alone must not flip the posture.
        cfg = {"agent": {"coding_context": "auto"}}
        env = {
            "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t", "HOME": str(tmp_path),
        }
        (tmp_path / "notes.md").write_text("# my novel\n")
        for args in (["init", "-q", "-b", "main"], ["add", "-A"], ["commit", "-q", "-m", "notes"]):
            subprocess.run([shutil.which("git"), "-C", str(tmp_path), *args], check=True, env=env)

        assert cc.is_coding_context(platform="cli", cwd=tmp_path, config=cfg) is False
        # …but adding a manifest or source file makes it a code workspace.
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        assert cc.is_coding_context(platform="cli", cwd=tmp_path, config=cfg) is True


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
        # (image-gen, homeassistant, …) survive entering a code workspace.
        _git_init(tmp_path)
        cfg = {"agent": {"coding_context": "auto"}}
        assert cc.coding_selection(platform="cli", cwd=tmp_path, config=cfg) is None
        # …while the prompt posture is still active.
        assert cc.is_coding_context(platform="cli", cwd=tmp_path, config=cfg) is True


# ── warning on unrecognized values ──────────────────────────────────────────

class TestCodingContextWarnings:
    """Unrecognized agent.coding_context values should warn, not silently fall back."""

    def test_unknown_value_warns_and_falls_back(self, caplog):
        """Bogus values like 'focu' or 'truee' should warn and resolve to 'auto'."""
        import logging
        cc._warned_unknown_values.clear()  # Reset dedupe set

        with caplog.at_level(logging.WARNING, logger="hermes.coding_context"):
            result = cc._coding_mode({"agent": {"coding_context": "focu"}})

        assert result == "auto", "Should fall back to auto"
        assert len(caplog.records) == 1, "Should emit exactly one warning"
        assert "focu" in caplog.text, "Warning should mention the bogus value"
        assert "auto" in caplog.text, "Warning should mention the fallback"

    def test_empty_value_is_silent(self, caplog):
        """Empty string should resolve to 'auto' without warning."""
        import logging
        cc._warned_unknown_values.clear()

        with caplog.at_level(logging.WARNING, logger="hermes.coding_context"):
            result = cc._coding_mode({"agent": {"coding_context": ""}})

        assert result == "auto"
        assert len(caplog.records) == 0, "Empty value should NOT warn"

    def test_missing_key_is_silent(self, caplog):
        """Missing key should resolve to 'auto' without warning."""
        import logging
        cc._warned_unknown_values.clear()

        with caplog.at_level(logging.WARNING, logger="hermes.coding_context"):
            result = cc._coding_mode({"agent": {}})

        assert result == "auto"
        assert len(caplog.records) == 0, "Missing key should NOT warn"

    def test_boolean_aliases_silent_and_resolve(self, caplog):
        """Legacy boolean aliases (true/false/yes/1) should resolve without warning."""
        import logging
        cc._warned_unknown_values.clear()

        with caplog.at_level(logging.WARNING, logger="hermes.coding_context"):
            assert cc._coding_mode({"agent": {"coding_context": "true"}}) == "on"
            assert cc._coding_mode({"agent": {"coding_context": "false"}}) == "off"
            assert cc._coding_mode({"agent": {"coding_context": "yes"}}) == "on"
            assert cc._coding_mode({"agent": {"coding_context": "1"}}) == "on"

        assert len(caplog.records) == 0, "Legacy aliases should NOT warn"

    def test_canonical_modes_silent(self, caplog):
        """Canonical modes (auto/focus/on/off) should resolve without warning."""
        import logging
        cc._warned_unknown_values.clear()

        with caplog.at_level(logging.WARNING, logger="hermes.coding_context"):
            assert cc._coding_mode({"agent": {"coding_context": "auto"}}) == "auto"
            assert cc._coding_mode({"agent": {"coding_context": "focus"}}) == "focus"
            assert cc._coding_mode({"agent": {"coding_context": "on"}}) == "on"
            assert cc._coding_mode({"agent": {"coding_context": "off"}}) == "off"

        assert len(caplog.records) == 0, "Canonical modes should NOT warn"

    def test_dedupe_same_bogus_value_warns_once(self, caplog):
        """Same bogus value should warn only once per process."""
        import logging
        cc._warned_unknown_values.clear()

        with caplog.at_level(logging.WARNING, logger="hermes.coding_context"):
            cc._coding_mode({"agent": {"coding_context": "typo"}})
            cc._coding_mode({"agent": {"coding_context": "typo"}})
            cc._coding_mode({"agent": {"coding_context": "typo"}})

        warning_count = sum(1 for r in caplog.records if "typo" in r.message)
        assert warning_count == 1, "Should warn exactly once for the same value"

    def test_different_bogus_values_each_warn(self, caplog):
        """Different bogus values should each warn once."""
        import logging
        cc._warned_unknown_values.clear()

        with caplog.at_level(logging.WARNING, logger="hermes.coding_context"):
            cc._coding_mode({"agent": {"coding_context": "typo1"}})
            cc._coding_mode({"agent": {"coding_context": "typo2"}})

        warning_count = len(caplog.records)
        assert warning_count == 2, "Different bogus values should each warn"




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



    def test_detect_project_facts_structured(self, tmp_path):
        (tmp_path / "package.json").write_text(
            json.dumps({"scripts": {"test": "vitest", "dev": "vite"}})
        )
        (tmp_path / "pnpm-lock.yaml").write_text("")
        facts = cc.detect_project_facts(tmp_path)
        assert facts.manifests == ["package.json"]
        assert facts.package_managers == ["pnpm"]
        assert facts.verify_commands == ["pnpm run test"]  # dev excluded
        assert facts.context_files == []

    def test_project_facts_for_matches_prompt_block(self, tmp_path):
        # Invariant: the structured facts the UI consumes must not drift from the
        # commands the prompt snapshot renders — one detector feeds both.
        _git_init(tmp_path)
        (tmp_path / "package.json").write_text(
            json.dumps({"scripts": {"test": "vitest", "lint": "eslint ."}})
        )
        (tmp_path / "pnpm-lock.yaml").write_text("")
        facts = cc.project_facts_for(tmp_path)
        assert facts is not None
        verify_line = cc.build_coding_workspace_block(tmp_path).split("Verify:")[1].splitlines()[0]
        assert facts["verifyCommands"]
        for cmd in facts["verifyCommands"]:
            assert cmd in verify_line



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

    def test_resolved_focus_ignores_an_on_disk_auto(self, tmp_path, monkeypatch):
        # The focus decision is the mode's own config_mode, not a re-read of the user's config.
        _git_init(tmp_path)
        home = tmp_path / "home"
        home.mkdir()
        (home / "config.yaml").write_text(json.dumps({"agent": {"coding_context": "auto"}}))
        monkeypatch.setenv("HERMES_HOME", str(home))
        mode = cc.resolve_runtime_mode(platform="cli", cwd=tmp_path, config={"agent": {"coding_context": "focus"}})
        assert mode.toolset_selection() == [cc.CODING_TOOLSET]





    def test_no_instructions_block_when_unset(self, tmp_path):
        _git_init(tmp_path)
        mode = cc.resolve_runtime_mode(platform="cli", cwd=tmp_path, config={"agent": {"coding_context": "on"}})
        assert not any("Operator instructions" in b for b in mode.system_blocks())



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







# ── profile registry ────────────────────────────────────────────────────────

class TestProfiles:



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
