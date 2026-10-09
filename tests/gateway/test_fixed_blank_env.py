"""Regression tests for #135509: a blank env var behind a ``_Cred.fixed`` mapping
must not clobber the yaml value ``PlatformConfig.from_dict`` promoted into ``extra``.

Before the fix, ``_Cred.__call__`` wrote every ``fixed`` entry unconditionally, so an
unset ``MATRIX_HOMESERVER`` executed ``extra["homeserver"] = ""`` and silently
discarded ``platforms.matrix.homeserver`` from config.yaml.  These tests drive the
real ``load_gateway_config`` against a temp HERMES_HOME — real YAML I/O, no mocks
of the code under test — and pin both directions (yaml kept when env blank, env
still wins when set) plus the always-written defaults that share the same loop.
"""

from gateway.config import Platform, load_gateway_config

_PLATFORM_ENV_PREFIXES = (
    "MATRIX_",
    "MATTERMOST_",
    "BLUEBUBBLES_",
    "WECOM_CALLBACK_",
)


def _isolate(monkeypatch, tmp_path, yaml_text):
    import os

    for key in list(os.environ):
        if key.startswith(_PLATFORM_ENV_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(yaml_text, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    return hermes_home


def test_blank_matrix_env_keeps_yaml_homeserver(tmp_path, monkeypatch):
    """#135509: MATRIX_HOMESERVER unset + yaml homeserver set -> yaml value survives."""
    _isolate(
        monkeypatch,
        tmp_path,
        "platforms:\n  matrix:\n    homeserver: https://matrix.example.org\n",
    )
    monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "matrix-token")

    cfg = load_gateway_config().platforms[Platform.MATRIX]

    assert cfg.enabled is True
    assert cfg.extra.get("homeserver") == "https://matrix.example.org", (
        "a blank MATRIX_HOMESERVER clobbered the yaml homeserver promoted into extra (#135509)"
    )


def test_set_matrix_env_still_wins_over_yaml(tmp_path, monkeypatch):
    """Env precedence is unchanged: a set MATRIX_HOMESERVER still overrides yaml."""
    _isolate(
        monkeypatch,
        tmp_path,
        "platforms:\n  matrix:\n    homeserver: https://yaml.example.org\n",
    )
    monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "matrix-token")
    monkeypatch.setenv("MATRIX_HOMESERVER", "https://env.example.org")

    cfg = load_gateway_config().platforms[Platform.MATRIX]

    assert cfg.extra.get("homeserver") == "https://env.example.org"


def test_blank_mattermost_env_keeps_yaml_url(tmp_path, monkeypatch):
    """Same fixed-loop path for mattermost's url (also named in #135509)."""
    _isolate(
        monkeypatch,
        tmp_path,
        "platforms:\n  mattermost:\n    url: https://mm.example.org\n",
    )
    monkeypatch.setenv("MATTERMOST_TOKEN", "mattermost-token")

    cfg = load_gateway_config().platforms[Platform.MATTERMOST]

    assert cfg.extra.get("url") == "https://mm.example.org"


def test_fixed_defaults_and_bool_flags_still_written(tmp_path, monkeypatch):
    """The skip only drops empty strings: defaults (webhook host/port/path) and
    bool-False flags (``require_mention`` — "always written" per its comment)
    must keep landing in ``extra`` when their env is unset."""
    _isolate(monkeypatch, tmp_path, "platforms: {}\n")
    monkeypatch.setenv("BLUEBUBBLES_SERVER_URL", "http://127.0.0.1:1234")
    monkeypatch.setenv("BLUEBUBBLES_PASSWORD", "bb-password")

    cfg = load_gateway_config().platforms[Platform.BLUEBUBBLES]

    assert cfg.extra.get("webhook_host") == "127.0.0.1"
    assert cfg.extra.get("webhook_port") == 8645
    assert cfg.extra.get("webhook_path") == "/bluebubbles-webhook"
    assert cfg.extra.get("require_mention") is False


def test_blank_wecom_callback_optional_keys_absent_not_blank(tmp_path, monkeypatch):
    """Optional callback keys (agent_id/token/...) with no env and no yaml are simply
    absent; the adapter reads them via ``extra.get(...) or default`` either way."""
    _isolate(monkeypatch, tmp_path, "platforms: {}\n")
    monkeypatch.setenv("WECOM_CALLBACK_CORP_ID", "corp-id")
    monkeypatch.setenv("WECOM_CALLBACK_CORP_SECRET", "corp-secret")

    cfg = load_gateway_config().platforms[Platform.WECOM_CALLBACK]

    assert cfg.extra.get("port") == 8645  # ``_int_or`` default still written
    assert "host" not in cfg.extra  # blank stays absent instead of a falsy "" entry
    assert "agent_id" not in cfg.extra
