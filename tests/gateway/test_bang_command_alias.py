"""Shared '!cmd' -> '/cmd' alias helper used by platform adapters whose clients
block or reserve '/'-prefixed posts (Mattermost #25184; Matrix and Slack threads
carry the same restriction with private copies)."""

from unittest.mock import patch

from gateway.platforms.helpers import normalize_bang_command


class TestNormalizeBangCommand:
    def test_known_gateway_commands_rewrite(self):
        assert normalize_bang_command("!model") == "/model"
        assert normalize_bang_command("!queue continue the plan") == "/queue continue the plan"
        assert normalize_bang_command("!tasks") == "/tasks"
        # /status is a Mattermost BUILT-IN slash name — the whole point of the alias.
        assert normalize_bang_command("!status") == "/status"

    def test_skill_commands_rewrite(self):
        import agent.skill_commands as skill_commands_mod

        fake_skills = {"/arxiv": {}, "/obsidian": {}}
        with patch.object(skill_commands_mod, "get_skill_commands", return_value=fake_skills):
            assert normalize_bang_command("!arxiv") == "/arxiv"
            assert normalize_bang_command("!obsidian search foo") == "/obsidian search foo"

    def test_registered_underscore_alias_passes_through(self):
        # '/reload-skills' ships a '/reload_skills' alias; the resolver returns the typed
        # token unchanged and the dispatcher resolves it — never a forced canonical form.
        assert normalize_bang_command("!reload_skills") == "/reload_skills"

    def test_unknown_and_non_bang_text_passes_through(self):
        assert normalize_bang_command("!definitelynotacommand") == "!definitelynotacommand"
        assert normalize_bang_command("!nice work") == "!nice work"
        assert normalize_bang_command("!") == "!"
        assert normalize_bang_command("!123") == "!123"
        assert normalize_bang_command("hello !model") == "hello !model"
        assert normalize_bang_command("/model") == "/model"
        assert normalize_bang_command("") == ""

    def test_multiline_arguments_preserved(self):
        assert normalize_bang_command("!bg do this\nand that") == "/bg do this\nand that"
