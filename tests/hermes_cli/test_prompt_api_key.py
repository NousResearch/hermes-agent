"""Tests for ``_prompt_api_key`` — the shared Keep/Replace/Clear menu used by
``hermes setup`` / ``hermes model`` when an API key already exists in ``.env``.

Regression coverage for #16394: the wizard used to silently skip the key prompt
when any value was present (even malformed junk), leaving users stuck.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def profile_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / ".env").write_text("")
    return home


def _pconfig(name="deepseek"):
    from hermes_cli.auth import PROVIDER_REGISTRY
    return PROVIDER_REGISTRY[name]


def _run_prompt(existing_key, choice, new_key="", provider_id="", pconfig_name="deepseek"):
    """Invoke _prompt_api_key with mocked input()/getpass() responses."""
    from hermes_cli import main as m

    pconfig = _pconfig(pconfig_name)
    with patch("builtins.input", return_value=choice), \
         patch("getpass.getpass", return_value=new_key):
        return m._prompt_api_key(pconfig, existing_key, provider_id=provider_id)


# First-time entry ────────────────────────────────────────────────────────────

def test_first_time_save_new_key(profile_env):
    from hermes_cli.config import get_env_value

    key, abort = _run_prompt(existing_key="", choice="", new_key="sk-abcdef")
    assert key == "sk-abcdef"
    assert abort is False
    assert get_env_value("DEEPSEEK_API_KEY") == "sk-abcdef"


def test_first_time_cancelled(profile_env):
    key, abort = _run_prompt(existing_key="", choice="", new_key="")
    assert key == ""
    assert abort is True


# Already configured — K / R / C ───────────────────────────────────────────────

def test_keep_default_empty_input(profile_env):
    from hermes_cli.config import save_env_value
    save_env_value("DEEPSEEK_API_KEY", "sk-existing")

    key, abort = _run_prompt(existing_key="sk-existing", choice="")
    assert key == "sk-existing"
    assert abort is False


def test_keep_letter_k(profile_env):
    key, abort = _run_prompt(existing_key="sk-existing", choice="k")
    assert key == "sk-existing"
    assert abort is False


def test_keep_on_unrecognised_input(profile_env):
    """Garbage input falls through to keep — never destroys the user's key."""
    key, abort = _run_prompt(existing_key="sk-existing", choice="xyz")
    assert key == "sk-existing"
    assert abort is False


def test_replace_saves_new_key(profile_env):
    from hermes_cli.config import get_env_value, save_env_value
    save_env_value("DEEPSEEK_API_KEY", "sk-malformed-junk")

    key, abort = _run_prompt(
        existing_key="sk-malformed-junk", choice="r", new_key="sk-fresh"
    )
    assert key == "sk-fresh"
    assert abort is False
    assert get_env_value("DEEPSEEK_API_KEY") == "sk-fresh"


def test_replace_cancelled_preserves_key(profile_env):
    """Empty entry to the Replace prompt means cancel — keeps the old key intact."""
    from hermes_cli.config import get_env_value, save_env_value
    save_env_value("DEEPSEEK_API_KEY", "sk-existing")

    key, abort = _run_prompt(
        existing_key="sk-existing", choice="r", new_key=""
    )
    assert key == "sk-existing"
    assert abort is False
    assert get_env_value("DEEPSEEK_API_KEY") == "sk-existing"


def test_clear_wipes_env_and_aborts(profile_env):
    from hermes_cli.config import get_env_value, save_env_value
    save_env_value("DEEPSEEK_API_KEY", "sk-existing")
    save_env_value("OTHER_VAR", "keep-me")

    key, abort = _run_prompt(existing_key="sk-existing", choice="c")
    assert key == ""
    assert abort is True
    # Cleared, but sibling entries untouched.
    assert not get_env_value("DEEPSEEK_API_KEY")
    assert get_env_value("OTHER_VAR") == "keep-me"


def test_ctrl_c_at_choice_prompt_keeps(profile_env):
    from hermes_cli import main as m

    pconfig = _pconfig("deepseek")
    with patch("builtins.input", side_effect=KeyboardInterrupt):
        key, abort = m._prompt_api_key(pconfig, "sk-existing")
    assert key == "sk-existing"
    assert abort is False


# LM Studio no-auth placeholder ────────────────────────────────────────────────

def test_lmstudio_first_time_empty_uses_placeholder(profile_env):
    from hermes_cli.auth import LMSTUDIO_NOAUTH_PLACEHOLDER
    from hermes_cli.config import get_env_value

    key, abort = _run_prompt(
        existing_key="", choice="", new_key="",
        provider_id="lmstudio", pconfig_name="lmstudio",
    )
    assert key == LMSTUDIO_NOAUTH_PLACEHOLDER
    assert abort is False
    assert get_env_value("LM_API_KEY") == LMSTUDIO_NOAUTH_PLACEHOLDER


def test_lmstudio_replace_empty_does_not_overwrite_with_placeholder(profile_env):
    """On REPLACE with empty input, preserve the user's existing key — do NOT
    silently substitute the placeholder.  The placeholder path only fires for
    first-time configuration where the user has made no explicit choice yet."""
    from hermes_cli.config import get_env_value, save_env_value
    save_env_value("LM_API_KEY", "my-real-lmstudio-key")

    key, abort = _run_prompt(
        existing_key="my-real-lmstudio-key", choice="r", new_key="",
        provider_id="lmstudio", pconfig_name="lmstudio",
    )
    assert key == "my-real-lmstudio-key"
    assert abort is False
    assert get_env_value("LM_API_KEY") == "my-real-lmstudio-key"
