"""Tests for agent.skill_preprocessing — template vars, inline shell expansion."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from agent.skill_preprocessing import (
    _INLINE_SHELL_RE,
    _SKILL_TEMPLATE_RE,
    expand_inline_shell,
    preprocess_skill_content,
    run_inline_shell,
    substitute_template_vars,
)


# ============================================================================
# substitute_template_vars
# ============================================================================
class TestSubstituteTemplateVars:
    def test_empty_content(self):
        assert substitute_template_vars("", Path("/skills"), "sess-1") == ""

    def test_no_tokens_returns_unchanged(self):
        assert substitute_template_vars("plain text", None, None) == "plain text"

    def test_skill_dir_token(self):
        result = substitute_template_vars(
            "Path: ${HERMES_SKILL_DIR}/refs",
            Path("/opt/skills/my-skill"),
            None,
        )
        assert result == "Path: /opt/skills/my-skill/refs"

    def test_session_id_token(self):
        result = substitute_template_vars(
            "Session: ${HERMES_SESSION_ID}",
            None,
            "abc-123",
        )
        assert result == "Session: abc-123"

    def test_both_tokens(self):
        result = substitute_template_vars(
            "${HERMES_SKILL_DIR}/data session=${HERMES_SESSION_ID}",
            Path("/skills/x"),
            "s1",
        )
        assert result == "/skills/x/data session=s1"

    def test_skill_dir_none_leaves_token(self):
        """When skill_dir is None, token is left as-is."""
        result = substitute_template_vars(
            "Dir: ${HERMES_SKILL_DIR}",
            None,
            None,
        )
        assert result == "Dir: ${HERMES_SKILL_DIR}"

    def test_session_id_none_leaves_token(self):
        """When session_id is None, token is left as-is."""
        result = substitute_template_vars(
            "ID: ${HERMES_SESSION_ID}",
            None,
            None,
        )
        assert result == "ID: ${HERMES_SESSION_ID}"

    def test_skill_dir_empty_string_leaves_token(self):
        """Falsy skill_dir_str (empty Path → '' → falsy) leaves token."""
        result = substitute_template_vars(
            "${HERMES_SKILL_DIR}",
            None,
            None,
        )
        assert result == "${HERMES_SKILL_DIR}"

    def test_session_id_truthy_string(self):
        """Non-empty session_id is substituted."""
        result = substitute_template_vars(
            "Session: ${HERMES_SESSION_ID}",
            None,
            "42",
        )
        assert result == "Session: 42"

    def test_unknown_token_left_as_is(self):
        """Tokens not in the regex are left alone."""
        result = substitute_template_vars(
            "${UNKNOWN_VAR} and ${HERMES_SKILL_DIR}",
            None,
            None,
        )
        assert result == "${UNKNOWN_VAR} and ${HERMES_SKILL_DIR}"

    def test_multiple_occurrences(self):
        content = "${HERMES_SKILL_DIR} ${HERMES_SKILL_DIR}"
        result = substitute_template_vars(content, Path("/a"), None)
        assert result == "/a /a"

    def test_token_in_middle_of_text(self):
        result = substitute_template_vars(
            "prefix${HERMES_SESSION_ID}suffix",
            None,
            "mid",
        )
        assert result == "prefixmidsuffix"


# ============================================================================
# _SKILL_TEMPLATE_RE regex
# ============================================================================
class TestTemplateRegex:
    def test_matches_skill_dir(self):
        assert _SKILL_TEMPLATE_RE.search("${HERMES_SKILL_DIR}") is not None

    def test_matches_session_id(self):
        assert _SKILL_TEMPLATE_RE.search("${HERMES_SESSION_ID}") is not None

    def test_does_not_match_unknown(self):
        assert _SKILL_TEMPLATE_RE.search("${OTHER}") is None

    def test_findall_multiple(self):
        matches = _SKILL_TEMPLATE_RE.findall(
            "${HERMES_SKILL_DIR} ${HERMES_SESSION_ID}"
        )
        assert matches == ["HERMES_SKILL_DIR", "HERMES_SESSION_ID"]


# ============================================================================
# expand_inline_shell
# ============================================================================
class TestExpandInlineShell:
    def test_no_shell_markers_returns_unchanged(self):
        assert expand_inline_shell("plain text", None, 10) == "plain text"

    def test_single_shell_snippet(self):
        with patch(
            "agent.skill_preprocessing.run_inline_shell",
            return_value="2026-05-25",
        ):
            result = expand_inline_shell("Today: !`date +%F`", None, 10)
            assert result == "Today: 2026-05-25"

    def test_multiple_snippets(self):
        outputs = {"hostname": "devbox", "whoami": "hermes"}
        def fake_run(cmd, cwd, timeout):
            return outputs.get(cmd.strip(), "")

        with patch("agent.skill_preprocessing.run_inline_shell", fake_run):
            result = expand_inline_shell(
                "User: !`whoami` on !`hostname`", None, 10
            )
            assert result == "User: hermes on devbox"

    def test_empty_command_returns_empty(self):
        """Regex requires 1+ chars between backticks — !`` is not matched."""
        result = expand_inline_shell("before!``after", None, 10)
        # !`` doesn't match the regex (needs 1+ non-newline chars)
        assert result == "before!``after"

    def test_whitespace_only_command(self):
        """Command with only whitespace after strip → empty."""
        with patch("agent.skill_preprocessing.run_inline_shell") as mock_run:
            result = expand_inline_shell("!`   `", None, 10)
            assert result == ""
            mock_run.assert_not_called()

    def test_guard_none_cwd_passed_as_none(self):
        """When skill_dir is None, it passes None to run_inline_shell."""
        with patch(
            "agent.skill_preprocessing.run_inline_shell",
            return_value="ok",
        ) as mock_run:
            expand_inline_shell("!`cmd`", None, 10)
            mock_run.assert_called_once_with("cmd", None, 10)

    def test_guard_path_cwd_passed_through(self):
        with patch(
            "agent.skill_preprocessing.run_inline_shell",
            return_value="ok",
        ) as mock_run:
            expand_inline_shell("!`cmd`", Path("/tmp"), 30)
            mock_run.assert_called_once_with("cmd", Path("/tmp"), 30)

    def test_no_exclamation_before_backtick_not_expanded(self):
        """Only !`cmd` pattern is expanded, not bare `cmd`."""
        result = expand_inline_shell("`not expanded`", None, 10)
        assert result == "`not expanded`"


# ============================================================================
# _INLINE_SHELL_RE regex
# ============================================================================
class TestInlineShellRegex:
    def test_matches_basic(self):
        assert _INLINE_SHELL_RE.search("!`echo hi`") is not None

    def test_does_not_match_without_exclamation(self):
        assert _INLINE_SHELL_RE.search("`echo hi`") is None

    def test_does_not_match_newlines(self):
        """Regex is single-line — no newlines inside backticks."""
        assert _INLINE_SHELL_RE.search("!`echo\nhi`") is None

    def test_non_greedy_multiple(self):
        matches = _INLINE_SHELL_RE.findall("!`a` and !`b`")
        assert matches == ["a", "b"]


# ============================================================================
# preprocess_skill_content
# ============================================================================
class TestPreprocessSkillContent:
    def test_empty_content(self):
        assert preprocess_skill_content("", None) == ""

    def test_no_preprocessing_enabled_returns_unchanged(self):
        """With inline_shell=False (default) and template_vars=True,
        content without tokens returns unchanged."""
        cfg = {"template_vars": False, "inline_shell": False}
        result = preprocess_skill_content(
            "${HERMES_SKILL_DIR} !`date`", None, None, cfg
        )
        assert result == "${HERMES_SKILL_DIR} !`date`"

    def test_template_vars_enabled(self):
        cfg = {"template_vars": True}
        result = preprocess_skill_content(
            "Dir: ${HERMES_SKILL_DIR}", Path("/s"), None, cfg
        )
        assert result == "Dir: /s"

    def test_inline_shell_enabled(self):
        cfg = {"inline_shell": True, "inline_shell_timeout": 5}
        with patch(
            "agent.skill_preprocessing.run_inline_shell",
            return_value="output",
        ):
            result = preprocess_skill_content(
                "!`cmd`", None, None, cfg
            )
            assert result == "output"

    def test_both_enabled(self):
        cfg = {"template_vars": True, "inline_shell": True, "inline_shell_timeout": 15}
        with patch(
            "agent.skill_preprocessing.run_inline_shell",
            return_value="done",
        ):
            result = preprocess_skill_content(
                "${HERMES_SESSION_ID}: !`cmd`",
                None,
                "sess-1",
                cfg,
            )
            assert result == "sess-1: done"

    def test_inline_shell_timeout_default(self):
        """When inline_shell_timeout is not in config, defaults to 10."""
        cfg = {"inline_shell": True}
        with patch(
            "agent.skill_preprocessing.run_inline_shell",
            return_value="ok",
        ) as mock_run:
            preprocess_skill_content("!`cmd`", None, None, cfg)
            mock_run.assert_called_once_with("cmd", None, 10)

    def test_inline_shell_timeout_falsy_defaults_to_10(self):
        cfg = {"inline_shell": True, "inline_shell_timeout": 0}
        with patch(
            "agent.skill_preprocessing.run_inline_shell",
            return_value="ok",
        ) as mock_run:
            preprocess_skill_content("!`cmd`", None, None, cfg)
            mock_run.assert_called_once_with("cmd", None, 10)

    def test_skills_cfg_not_dict_loads_from_config(self):
        """When skills_cfg is not a dict, calls load_skills_config()."""
        with patch(
            "agent.skill_preprocessing.load_skills_config",
            return_value={"template_vars": True},
        ):
            result = preprocess_skill_content(
                "Dir: ${HERMES_SKILL_DIR}", Path("/x"), None, skills_cfg=None
            )
            assert result == "Dir: /x"

    def test_template_vars_defaults_true(self):
        """When template_vars key is missing, defaults to True."""
        cfg = {}  # no template_vars key
        result = preprocess_skill_content(
            "${HERMES_SKILL_DIR}", Path("/a"), None, cfg
        )
        assert result == "/a"
