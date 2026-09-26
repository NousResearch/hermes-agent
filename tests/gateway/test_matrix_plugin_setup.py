"""Tests for the Matrix plugin's interactive_setup wizard home-channel flow.

The interactive_setup wizard lazy-imports its CLI helpers from
``hermes_cli.config`` (get_env_value / save_env_value / remove_env_value),
``hermes_cli.cli_output`` (prompt / prompt_yes_no / print_*), and
``pm`` (Matrix sync_venv). We patch each at its source module so
the wizard runs without provisioning dependencies. Covers the home-channel
clear-on-blank behavior added in the follow-up to PR #58421.
"""
import pytest

import hermes_cli.config as config_mod
import hermes_cli.cli_output as cli_output_mod
import pm as pm_mod
import pm.extras as extras_mod
from plugins.platforms.matrix.adapter import interactive_setup


def _patch_setup_io(monkeypatch, prompts, yes_no_responses, saved, removed, existing, synced=None):
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

    # Setup explicitly syncs dependencies; lazy-import patches do not intercept it.
    def _sync(extras=None, **kw):
        if synced is not None:
            synced.append((list(extras or []), kw))

    monkeypatch.setattr(pm_mod, "sync_venv", _sync)


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


_E2EE_KEYS = ("MATRIX_ENCRYPTION", "MATRIX_E2EE_MODE")
_OLM_GATED_OFF = "sys_platform == 'never'"


@pytest.mark.parametrize(("e2ee_gate", "existing", "answers", "mode"), [
    # python-olm installs: E2EE is offered, and a yes on a fresh setup turns it on.
    (None, {}, [True], "required"),
    # The saved mode is the default, and a no turns E2EE off instead of leaving it required.
    (None, {"MATRIX_ENCRYPTION": "true"}, [False], "off"),
    # A yes also beats an explicit MATRIX_E2EE_MODE=off, which outranks MATRIX_ENCRYPTION.
    (None, {"MATRIX_E2EE_MODE": "off"}, [True], "required"),
    # python-olm can't install: never offered, even though olm imports in this venv (hand-installed).
    (_OLM_GATED_OFF, {}, [], "off"),
    # A required mode saved before this gate existed (the #62401 state) is offered for removal.
    (_OLM_GATED_OFF, {"MATRIX_ENCRYPTION": "true"}, [True], "off"),
])
def test_e2ee_setup_follows_the_python_olm_gate(monkeypatch, tmp_path, e2ee_gate, existing, answers, mode):
    """#62401: a yes on macOS stored MATRIX_ENCRYPTION=true, then the adapter refused to start because
    E2EE was required and python-olm cannot install there. The wizard asks pm's ``matrix-e2ee`` gate,
    not the running venv, and the mode the adapter resolves from the .env it leaves behind is the one
    its answers chose, whatever was saved before. It prepares the plaintext deps either way."""
    from plugins.platforms.matrix.adapter import _resolve_e2ee_mode

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(extras_mod, "_PLATFORM_GATES", {"matrix-e2ee": e2ee_gate} if e2ee_gate else {})
    monkeypatch.setattr(extras_mod, "_importable", lambda _anchor: True)
    env, saved, synced = dict(existing), {}, []
    answers = iter(answers)
    _patch_setup_io(monkeypatch, _PROMPTS_BLANK, answers, saved, [], existing=env, synced=synced)
    interactive_setup()
    assert list(answers) == []  # every yes/no the case expects was asked, and no other
    env.update(saved)  # the .env the wizard leaves behind (its removals already popped from env)
    for key in _E2EE_KEYS:
        if key in env:
            monkeypatch.setenv(key, env[key])
        else:
            monkeypatch.delenv(key, raising=False)
    assert _resolve_e2ee_mode() == mode
    assert synced == [(["matrix"], {"explicit": True})]
