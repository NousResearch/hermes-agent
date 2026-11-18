"""The YAML → env bridge must refresh its own leftovers on the next start (#react‑reactions).

A gateway started from the CLI hands its whole environment to the restart child, and
``DAEMON_STRIPPED_ENV`` carries no ``TELEGRAM_*`` names. So a value the bridge wrote from YAML once
— say the stock ``reactions: false`` — survives into the next process as a plain env var, where the
bridge used to skip ("env is set") and the reader saw env before YAML. Editing ``reactions: true``
then did nothing until the operator restarted from a shell outside the gateway.

The bridge now records what it wrote: a value equal to the marker is its own leftover and is
refreshed from YAML. Any other inherited process value loses to an explicit YAML key too (#110295);
only a ``.env`` or external-secret-source value still wins (#109032).
"""

from __future__ import annotations

import pytest

import gateway.platforms._shared as shared
from gateway.platforms._shared import bridged_env_digest, bridged_env_marker, yaml_env_setter


def test_first_bridge_writes_the_value_and_its_marker(monkeypatch):
    monkeypatch.delenv("TELEGRAM_REACTIONS", raising=False)
    monkeypatch.delenv(bridged_env_marker("TELEGRAM_REACTIONS"), raising=False)

    yaml_env_setter()("TELEGRAM_REACTIONS", "true")

    assert shared.os.environ["TELEGRAM_REACTIONS"] == "true"
    assert shared.os.environ[bridged_env_marker("TELEGRAM_REACTIONS")] == bridged_env_digest("true")


def test_an_inherited_process_value_loses_to_explicit_yaml(monkeypatch):
    """#110295: ``TELEGRAM_REACTIONS=true`` inherited from a launcher, YAML says false: YAML wins."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    monkeypatch.delenv(bridged_env_marker("TELEGRAM_REACTIONS"), raising=False)

    yaml_env_setter()("TELEGRAM_REACTIONS", "false")

    assert shared.os.environ["TELEGRAM_REACTIONS"] == "false"
    assert shared.os.environ[bridged_env_marker("TELEGRAM_REACTIONS")] == bridged_env_digest("false")


def test_a_dotenv_value_still_beats_yaml(tmp_path, monkeypatch):
    """#109032: ``TELEGRAM_REACTIONS=true`` in ``.env`` beats the stock ``reactions: false``."""
    from hermes_cli import env_loader

    for attr in ("_LOADED_DOTENV_KEYS", "_MANAGED_DOTENV_KEYS"):
        monkeypatch.setattr(env_loader, attr, set())
    monkeypatch.setattr(env_loader, "_DOTENV_PUBLISHED", {})
    monkeypatch.setenv("TELEGRAM_REACTIONS", "")  # teardown removes what .env publishes
    monkeypatch.delenv("TELEGRAM_REACTIONS")
    monkeypatch.delenv(bridged_env_marker("TELEGRAM_REACTIONS"), raising=False)
    (tmp_path / ".env").write_text("TELEGRAM_REACTIONS=true\n", encoding="utf-8")
    env_loader.load_hermes_dotenv(hermes_home=tmp_path, load_external_secrets=False)

    yaml_env_setter()("TELEGRAM_REACTIONS", "false")

    assert shared.os.environ["TELEGRAM_REACTIONS"] == "true"


def test_an_inherited_bridge_value_is_refreshed_from_yaml(monkeypatch):
    """The regression: a restarted gateway inherits the bridged ``false``; YAML now says true."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "false")
    monkeypatch.setenv(bridged_env_marker("TELEGRAM_REACTIONS"), bridged_env_digest("false"))

    yaml_env_setter()("TELEGRAM_REACTIONS", "true")

    assert shared.os.environ["TELEGRAM_REACTIONS"] == "true"
    assert shared.os.environ[bridged_env_marker("TELEGRAM_REACTIONS")] == bridged_env_digest("true")


def test_secondary_profile_scope_still_skips_the_write(monkeypatch):
    monkeypatch.delenv("TELEGRAM_REACTIONS", raising=False)
    monkeypatch.delenv(bridged_env_marker("TELEGRAM_REACTIONS"), raising=False)
    monkeypatch.setattr(shared, "profile_scoped", lambda: True)

    yaml_env_setter()("TELEGRAM_REACTIONS", "true")

    assert "TELEGRAM_REACTIONS" not in shared.os.environ
    assert bridged_env_marker("TELEGRAM_REACTIONS") not in shared.os.environ


def test_none_is_skipped(monkeypatch):
    monkeypatch.delenv("TELEGRAM_REACTIONS", raising=False)

    yaml_env_setter()("TELEGRAM_REACTIONS", None)

    assert "TELEGRAM_REACTIONS" not in shared.os.environ


def test_a_blank_env_counts_as_unset(monkeypatch):
    """``TELEGRAM_REACTIONS=""`` is unset for every other reader, so the bridge must fill it in."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "")
    monkeypatch.delenv(bridged_env_marker("TELEGRAM_REACTIONS"), raising=False)

    yaml_env_setter()("TELEGRAM_REACTIONS", "true")

    assert shared.os.environ["TELEGRAM_REACTIONS"] == "true"



# The 13 bridged names that grant access rather than change a preference (#144353).
AUTHORIZATION_ENV_NAMES = [
    "TELEGRAM_ALLOWED_USERS",
    "TELEGRAM_ALLOWED_CHATS",
    "TELEGRAM_GROUP_ALLOWED_USERS",
    "TELEGRAM_GROUP_ALLOWED_CHATS",
    "TELEGRAM_ALLOWED_TOPICS",
    "TELEGRAM_ALLOW_BOTS",
    "DISCORD_ALLOWED_USERS",
    "DISCORD_ALLOWED_ROLES",
    "DISCORD_ALLOW_ALL_USERS",
    "DISCORD_ALLOW_BOTS",
    "DISCORD_ALLOWED_CHANNELS",
    "DISCORD_IGNORED_CHANNELS",
    "DISCORD_NO_THREAD_CHANNELS",
]


@pytest.mark.parametrize("name", AUTHORIZATION_ENV_NAMES)
def test_a_tightened_authorization_value_reaches_the_process(monkeypatch, name):
    """Each allowlist must lose to the YAML that narrowed it, not to the value the bridge wrote earlier."""
    monkeypatch.setenv(name, "111,222")
    monkeypatch.setenv(bridged_env_marker(name), bridged_env_digest("111,222"))

    yaml_env_setter()(name, ["111"])  # the adapters pass the list form

    assert shared.os.environ[name] == "111"
    assert shared.os.environ[bridged_env_marker(name)] == bridged_env_digest("111")


def test_every_authorization_name_is_refreshed_by_the_real_adapter_hooks(monkeypatch):
    """Execute both hooks rather than infer their writes from source text."""
    from plugins.platforms.discord.adapter import _apply_yaml_config as discord_hook
    from plugins.platforms.telegram.adapter import _apply_yaml_config as telegram_hook

    for name in AUTHORIZATION_ENV_NAMES:
        monkeypatch.setenv(name, "111,222")
        monkeypatch.setenv(bridged_env_marker(name), bridged_env_digest("111,222"))
    telegram_hook({}, {
        "allow_from": ["111"], "allowed_chats": ["111"], "allowed_topics": ["111"],
        "group_allow_from": ["111"], "group_allowed_chats": ["111"], "allow_bots": False,
    })
    discord_hook({}, {
        "allow_from": ["111"], "allowed_roles": ["111"], "allow_all_users": False,
        "allow_bots": False, "allowed_channels": ["111"], "ignored_channels": ["111"],
        "no_thread_channels": ["111"],
    })

    for name in AUTHORIZATION_ENV_NAMES:
        expected = "false" if name.endswith(("ALLOW_BOTS", "ALLOW_ALL_USERS")) else "111"
        assert shared.os.environ[name] == expected
        assert shared.os.environ[bridged_env_marker(name)] == bridged_env_digest(expected)


def test_the_restart_child_env_keeps_the_value_and_its_marker(monkeypatch, tmp_path):
    """End-to-end over the real restart builder: the child must inherit both, or nothing refreshes.

    ``hermes_cli/gateway_runtime_start.py`` builds the replacement daemon's environment with
    ``served_profile_child_env``; if that step dropped or rewrote ``HERMES_BRIDGED_*``, the marker
    scheme would be correct in unit tests and dead in production.
    """
    from tools.environments.local import served_profile_child_env

    monkeypatch.setenv("TELEGRAM_REACTIONS", "false")
    monkeypatch.setenv(bridged_env_marker("TELEGRAM_REACTIONS"), bridged_env_digest("false"))

    from hermes_constants import get_hermes_home

    child = served_profile_child_env(  # the launch profile's own restart: same home
        base=dict(shared.os.environ), target_home=get_hermes_home(), inherit_credentials=False
    )

    assert child["TELEGRAM_REACTIONS"] == "false"
    assert child[bridged_env_marker("TELEGRAM_REACTIONS")] == bridged_env_digest("false")

    # Now be that child: it boots with the inherited pair and reads its own edited YAML.
    for key in ("TELEGRAM_REACTIONS", bridged_env_marker("TELEGRAM_REACTIONS")):
        monkeypatch.setenv(key, child[key])
    yaml_env_setter()("TELEGRAM_REACTIONS", "true")

    assert shared.os.environ["TELEGRAM_REACTIONS"] == "true"


def test_a_dotenv_value_survives_even_when_it_matches_the_bridge(monkeypatch):
    """#109032 holds when .env happens to hold the same string the bridge wrote earlier."""
    from hermes_cli import env_loader

    monkeypatch.setattr(env_loader, "_LOADED_DOTENV_KEYS", {"TELEGRAM_REACTIONS"}, raising=False)
    monkeypatch.setenv("TELEGRAM_REACTIONS", "false")
    monkeypatch.setenv(bridged_env_marker("TELEGRAM_REACTIONS"), bridged_env_digest("false"))

    yaml_env_setter()("TELEGRAM_REACTIONS", "true")

    assert shared.os.environ["TELEGRAM_REACTIONS"] == "false"


def test_the_telegram_adapter_applies_an_edited_reactions_setting(monkeypatch):
    """Whole chain: the adapter's own YAML hook, over a process that still holds the bridged false."""
    telegram = pytest.importorskip("plugins.platforms.telegram.adapter")

    monkeypatch.setenv("TELEGRAM_REACTIONS", "false")
    monkeypatch.setenv(bridged_env_marker("TELEGRAM_REACTIONS"), bridged_env_digest("false"))

    telegram._apply_yaml_config({"platforms": {"telegram": {"reactions": True}}}, {"reactions": True})

    assert shared.os.environ["TELEGRAM_REACTIONS"] == "true"
    assert shared.os.environ[bridged_env_marker("TELEGRAM_REACTIONS")] == bridged_env_digest("true")
