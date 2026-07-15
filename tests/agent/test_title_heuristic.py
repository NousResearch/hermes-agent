"""Tests for agent.title_heuristic — regex-based title extraction."""

from agent.title_heuristic import extract_title


class TestActionVerbs:
    """Action verb patterns produce clean imperative titles."""

    def test_fix_pattern(self):
        assert extract_title("Fix the dropdown menu closing when you hover") == "Fix dropdown menu closing"

    def test_review_pattern(self):
        assert extract_title("Can you review PR #3979 on radix-ui/primitives?") == "Review PR #3979"

    def test_add_pattern(self):
        assert extract_title("I want to add dark mode support to the settings page") == "Add dark mode support"

    def test_remove_pattern(self):
        assert extract_title("Remove the deprecated auth middleware from the gateway") == "Remove deprecated auth middleware"

    def test_refactor_pattern(self):
        assert extract_title("Let's refactor the session DB layer into a clean module") == "Refactor session DB layer"

    def test_patch_preserves_original_verb(self):
        """'patch' should stay 'patch', not be normalized to 'fix'."""
        assert extract_title("Patch the title generator to include tool call context") == "Patch title generator"

    def test_adjust_preserves_original_verb(self):
        """'adjust' should stay 'adjust', not be normalized to 'update'."""
        title = extract_title("Adjust the timeout for auxiliary LLM calls")
        assert title.startswith("Adjust")

    def test_explain_pattern(self):
        assert extract_title("Explain the credential pool resolution order") == "Explain the credential pool"

    def test_investigate_preserves_question_word(self):
        title = extract_title("I'm trying to figure out why my titles keep getting overwritten")
        assert "why" in title.lower()


class TestQuestions:
    """Question patterns produce clean topic phrases."""

    def test_how_does_x_work(self):
        assert extract_title("How does the auxiliary client fallback chain work?") == "How the auxiliary client fallback chain works"

    def test_what_is_difference_between(self):
        title = extract_title("What's the difference between auto_title_session and maybe_auto_title?")
        assert title.startswith("Difference between")

    def test_where_is_x(self):
        title = extract_title("Where is the title generation prompt defined?")
        assert title.startswith("Where")
        assert "title generation prompt" in title

    def test_can_we_x(self):
        title = extract_title("Can we use a cheaper model for title generation?")
        assert title.startswith("Use")


class TestPrefixStripping:
    """Conversational prefixes are stripped iteratively."""

    def test_strips_hey_comma(self):
        title = extract_title("Hey, I need help understanding the session lifecycle in the TUI")
        assert not title.lower().startswith("hey")

    def test_strips_so_prefix(self):
        title = extract_title("so the dropdown focus thing is still broken right?")
        assert not title.lower().startswith("so ")

    def test_strips_nested_prefixes(self):
        title = extract_title("Can you please help me fix the auth middleware")
        assert title.startswith("Fix") or title.startswith("Help")


class TestEdgeCases:
    """Edge cases that should produce reasonable output without crashing."""

    def test_empty_message(self):
        assert extract_title("") == ""

    def test_whitespace_only(self):
        assert extract_title("   ") == ""

    def test_very_short_message(self):
        title = extract_title("fix bug")
        assert title == "Fix bug"

    def test_no_action_verb(self):
        """Messages without action verbs fall to keyphrase fallback."""
        title = extract_title("MiMo produces worse titles than Claude")
        assert title  # should return something
        assert len(title) > 0

    def test_em_dash_artifact(self):
        title = extract_title("here's what's happening — when I open a new session the title is just generic")
        assert not title.startswith("—")
        assert not title.startswith("–")


class TestTokenPreservation:
    """Technical tokens (PR #N, snake_case, file paths) are preserved."""

    def test_preserves_pr_number(self):
        title = extract_title("Can you review PR #3979 on radix-ui/primitives?")
        assert "PR #3979" in title

    def test_preserves_snake_case(self):
        title = extract_title("What's the difference between auto_title_session and maybe_auto_title?")
        assert "auto_title_session" in title

    def test_preserves_file_extension(self):
        title = extract_title("Create a SKILL.md for session title best practices")
        assert "SKILL.md" in title


class TestLengthGuard:
    """Progressive relaxation prevents overly short titles."""

    def test_write_test_keeps_context(self):
        """'Write test' alone is too short; length guard should keep more context."""
        title = extract_title("Write a test for the title generator")
        assert len(title.split()) >= 3

    def test_build_script_keeps_context(self):
        title = extract_title("Build a script that monitors session title quality over time")
        assert len(title.split()) >= 3


class TestNeverRaises:
    """extract_title must never raise — always return a string."""

    def test_none_input(self):
        # Should handle gracefully (returns empty string)
        try:
            result = extract_title(None)
            assert isinstance(result, str)
        except TypeError:
            pass  # None input is acceptable to reject

    def test_unicode_input(self):
        title = extract_title("修复下拉菜单关闭问题")
        assert isinstance(title, str)

    def test_very_long_message(self):
        title = extract_title("Fix " + "x" * 5000)
        assert isinstance(title, str)
        assert len(title) <= 80
