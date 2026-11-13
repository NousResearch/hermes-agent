"""A profile secret scope reads a ``.env`` the way ``load_hermes_dotenv`` publishes it (#137080).

``TELEGRAM_HOME_CHANNEL=${TELEGRAM_USER_ID}`` expanded in ``os.environ`` but reached the secret scope,
and so cron home-channel delivery, as the literal template. Real files, real scope, real resolver.
"""

from __future__ import annotations

import os

from agent.secret_scope import build_profile_secret_scope, reset_secret_scope, set_secret_scope


def test_cron_home_target_resolves_a_dotenv_reference(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    home.mkdir()
    (home / ".env").write_text(
        "TELEGRAM_USER_ID=399611999\nTELEGRAM_HOME_CHANNEL=${TELEGRAM_USER_ID}\n"
        "DEFAULTED=${UNSET_IN_TEST:-fallback}\nPLAIN=lit$eral\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("UNSET_IN_TEST", raising=False)
    from hermes_cli.plugins import discover_plugins
    discover_plugins()
    from hermes_agent.cron.scheduler_delivery import _get_home_target_chat_id

    scope = build_profile_secret_scope(home)
    token = set_secret_scope(scope, profile_home=str(home))
    try:
        assert _get_home_target_chat_id("telegram") == "399611999"
    finally:
        reset_secret_scope(token)
    assert scope["DEFAULTED"] == "fallback" and scope["PLAIN"] == "lit$eral"


def test_satellite_reference_never_resolves_against_the_launch_profiles_secret(tmp_path, monkeypatch):
    launch, satellite = tmp_path / "launch", tmp_path / "launch" / "profiles" / "sat"
    satellite.mkdir(parents=True)
    (satellite / ".env").write_text(
        "OPENAI_API_KEY=${OPENAI_API_KEY:-satellite-default}\nOWN_HOME=${HOME}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("OPENAI_API_KEY", "launch-profile-key")  # the gateway's own shell export

    scope = build_profile_secret_scope(satellite)

    assert scope["OPENAI_API_KEY"] == "satellite-default"
    assert scope["OWN_HOME"] == os.environ["HOME"]  # process-global vars still resolve
