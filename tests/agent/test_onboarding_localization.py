"""First-touch hints follow the active profile language, not import-time state."""

import re

import pytest

from agent import i18n, onboarding, secret_scope
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.mark.parametrize("helper,args", [
    (onboarding.busy_input_hint_gateway, (mode,))
    for mode in ("queue", "steer", "redirect", "interrupt", "unknown")
] + [
    (onboarding.busy_input_hint_cli, (mode,))
    for mode in ("queue", "steer", "redirect", "interrupt", "unknown")
] + [
    (onboarding.tool_progress_hint_gateway, ()),
    (onboarding.tool_progress_hint_cli, ()),
    (onboarding.openclaw_residue_hint_cli, ()),
])
def test_hint_language_follows_profile_without_changing_commands(helper, args, tmp_path, monkeypatch):
    homes = {}
    for lang in ("en", "ru"):
        home = tmp_path / lang
        home.mkdir()
        (home / "config.yaml").write_text(f"display:\n  language: {lang}\n", encoding="utf-8")
        homes[lang] = home
    monkeypatch.setenv("HERMES_HOME", str(homes["en"]))
    monkeypatch.delenv("HERMES_LANGUAGE", raising=False)
    i18n.reset_language_cache()
    secret_scope.set_multiplex_active(True)
    scope = secret_scope.set_secret_scope({})
    try:
        english = helper(*args)
        token = set_hermes_home_override(homes["ru"])
        try:
            russian = helper(*args)
            assert russian != english
            assert re.search("[А-Яа-я]", russian)
            assert re.findall(r"/(?:busy\s+\w+|verbose|stop)\b", russian) == re.findall(
                r"/(?:busy\s+\w+|verbose|stop)\b", english)
            if helper is onboarding.openclaw_residue_hint_cli:
                for literal in ("hermes claw migrate", "hermes claw cleanup", "~/.openclaw/", "~/.openclaw.pre-migration"):
                    assert literal in russian
        finally:
            reset_hermes_home_override(token)
        assert helper(*args) == english
    finally:
        secret_scope.reset_secret_scope(scope)
        secret_scope.set_multiplex_active(False)
        i18n.reset_language_cache()
