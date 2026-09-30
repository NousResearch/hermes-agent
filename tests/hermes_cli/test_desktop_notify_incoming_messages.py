"""`desktop.notify_incoming_messages` — the config gate (#56187) for the native OS
notification the Desktop fires when an inbound messaging-platform message arrives while
its window is unfocused. Default-off contract: never any unsolicited notification."""
import textwrap


def test_desktop_notify_incoming_messages_defaults_off():
    from hermes_cli.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["desktop"]["notify_incoming_messages"] is False


def test_desktop_notify_incoming_messages_key_is_in_generated_schema():
    from hermes_cli.web_server_config import CONFIG_SCHEMA

    assert CONFIG_SCHEMA["desktop.notify_incoming_messages"]["type"] == "boolean"


def test_desktop_notify_incoming_messages_reads_through_the_desktop_loader(tmp_path, monkeypatch):
    """The Desktop reads config through `/api/config`, which serves `load_config()` —
    the loader merging DEFAULT_CONFIG. A user opting in must see `True`; an absent key
    must fall back to the registered default, not to a permissive truthy read."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    (home / "config.yaml").write_text(
        textwrap.dedent(
            """
            desktop:
              notify_incoming_messages: true
            """
        ),
        encoding="utf-8",
    )

    from hermes_cli.config import _LOAD_CONFIG_CACHE, load_config

    _LOAD_CONFIG_CACHE.clear()
    try:
        enabled = load_config()
    finally:
        _LOAD_CONFIG_CACHE.clear()

    assert enabled["desktop"]["notify_incoming_messages"] is True

    (home / "config.yaml").unlink()
    _LOAD_CONFIG_CACHE.clear()
    try:
        default = load_config()
    finally:
        _LOAD_CONFIG_CACHE.clear()

    assert default["desktop"]["notify_incoming_messages"] is False
