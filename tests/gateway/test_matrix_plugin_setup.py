"""Tests for the Matrix plugin's interactive_setup wizard home-channel flow.

The interactive_setup wizard lazy-imports its CLI helpers from
``hermes_cli.config`` (get_env_value / save_env_value / remove_env_value),
``hermes_cli.cli_output`` (prompt / prompt_yes_no / print_*), and
``tools.lazy_deps`` (mautrix ensure). We patch each at its source module so
the wizard runs without touching pip or the network. Covers the home-channel
clear-on-blank behavior added in the follow-up to PR #58421.
"""
import hermes_cli.config as config_mod
import hermes_cli.cli_output as cli_output_mod
import tools.lazy_deps as lazy_deps_mod
from plugins.platforms.matrix.adapter import interactive_setup


def _patch_setup_io(monkeypatch, prompts, yes_no_responses, saved, removed, existing):
    prompt_iter = iter(prompts)
    yes_no_iter = iter(yes_no_responses)
    monkeypatch.setattr(config_mod, "get_env_value", lambda key: existing.get(key, ""))
    monkeypatch.setattr(config_mod, "save_env_value", lambda k, v: saved.update({k: v}))

    def _remove(key):
        removed.append(key)
        return existing.pop(key, None) is not None

    monkeypatch.setattr(config_mod, "remove_env_value", _remove)
    monkeypatch.setattr(cli_output_mod, "prompt", lambda *_a, **_kw: next(prompt_iter))
    monkeypatch.setattr(
        cli_output_mod, "prompt_yes_no", lambda *_a, **_kw: next(yes_no_iter)
    )
    for name in ("print_header", "print_info", "print_success", "print_warning"):
        monkeypatch.setattr(cli_output_mod, name, lambda *_a, **_kw: None)
    # Block the auto-install path so the test never invokes pip.
    monkeypatch.setattr(lazy_deps_mod, "feature_missing", lambda feature: ())
    monkeypatch.setattr(lazy_deps_mod, "ensure", lambda *a, **kw: None)


# Matrix prompts (after the E2EE yes_no): allowed_users, home_channel.
# Homeserver, token are still text prompts before E2EE.
_PROMPTS_NONEMPTY = [
    "https://matrix.example.org",  # homeserver
    "syt_test_token_value",        # access token (password)
    "@bot:matrix.example.org",     # user_id (optional)
    "",                            # allowed_users
    "!AbCdEfGhIjKlMn:matrix.example.org",  # home room
]
_PROMPTS_BLANK = [
    "https://matrix.example.org",
    "syt_test_token_value",
    "@bot:matrix.example.org",
    "",
    "",
]
_PROMPTS_WHITESPACE = [
    "https://matrix.example.org",
    "syt_test_token_value",
    "@bot:matrix.example.org",
    "",
    "   ",
]
# E2EE? = False so we don't pull the [encryption] extras.
_YES_NO = [False]


class TestMatrixHomeChannelClear:
    """Blank home-room answer must clear MATRIX_HOME_ROOM (#12423)."""

    def test_blank_removes_existing_home_room(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        saved, removed = {}, []
        _patch_setup_io(
            monkeypatch,
            _PROMPTS_BLANK,
            _YES_NO,
            saved,
            removed,
            existing={"MATRIX_HOME_ROOM": "!oldRoomId:matrix.example.org"},
        )
        interactive_setup()
        assert "MATRIX_HOME_ROOM" in removed
        assert "MATRIX_HOME_ROOM" not in saved

    def test_blank_without_prior_home_still_attempts_remove(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        saved, removed = {}, []
        _patch_setup_io(
            monkeypatch, _PROMPTS_BLANK, _YES_NO, saved, removed, existing={}
        )
        interactive_setup()
        assert removed.count("MATRIX_HOME_ROOM") == 1

    def test_nonempty_saves_home_room(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        saved, removed = {}, []
        _patch_setup_io(
            monkeypatch, _PROMPTS_NONEMPTY, _YES_NO, saved, removed, existing={}
        )
        interactive_setup()
        assert saved["MATRIX_HOME_ROOM"] == "!AbCdEfGhIjKlMn:matrix.example.org"
        assert "MATRIX_HOME_ROOM" not in removed

    def test_whitespace_only_clears_home_room(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        saved, removed = {}, []
        _patch_setup_io(
            monkeypatch,
            _PROMPTS_WHITESPACE,
            _YES_NO,
            saved,
            removed,
            existing={"MATRIX_HOME_ROOM": "!oldRoomId:matrix.example.org"},
        )
        interactive_setup()
        assert "MATRIX_HOME_ROOM" in removed
        assert "MATRIX_HOME_ROOM" not in saved