"""Regression tests for #109443: `hermes config get` must notice unknown keys.

A hand-edited config.yaml can park a value at a plausible-but-wrong dotted path
(``agent.tool_search.enabled`` where the runtime reads ``tools.tool_search.enabled``).
``hermes config get`` is the natural way to verify such an edit, and previously
echoed the value back with no hint that the runtime would never read it.
``set_config_value`` already prints a post-write unknown-key notice (#34067);
the read path now mirrors it on stderr, leaving stdout (and ``--json``) intact.
"""

from pathlib import Path

import pytest
import yaml


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Temp HERMES_HOME whose config.yaml carries the wrong-path edit from #109443."""
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(yaml.dump({
        "agent": {"tool_search": {"enabled": False}},
        "tools": {"tool_search": {"enabled": "off"}},
    }))
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _get(key: str, *, as_json: bool = False):
    from hermes_cli.config import get_config_value
    get_config_value(key, as_json=as_json)


class TestGetUnknownKeyNotice:
    def test_wrong_nested_path_prints_notice_but_echoes_value(self, hermes_home, capsys):
        """agent.tool_search.enabled prints its value AND warns the runtime never reads it."""
        _get("agent.tool_search.enabled")

        out = capsys.readouterr()
        assert 'false' in out.out
        assert "not a recognized config key" in out.err
        assert "agent.tool_search.enabled" in out.err

    def test_known_nested_path_no_notice(self, hermes_home, capsys):
        """The real path tools.tool_search.enabled stays notice-free."""
        _get("tools.tool_search.enabled")

        out = capsys.readouterr()
        assert "not a recognized config key" not in out.err

    def test_json_stdout_stays_parseable(self, hermes_home, capsys):
        """--json output must remain machine-parseable; the notice goes to stderr."""
        _get("agent.tool_search.enabled", as_json=True)

        out = capsys.readouterr()
        assert yaml.safe_load(out.out) is False
        assert "not a recognized config key" in out.err

    def test_env_key_immune_to_schema_notice(self, hermes_home, monkeypatch, capsys):
        """Env-routed keys live outside the YAML schema and must not be flagged."""
        monkeypatch.setenv("HERMES_TEST_TOKEN", "tok")
        _get("hermes_test_token")

        out = capsys.readouterr()
        assert "tok" in out.out
        assert "not a recognized config key" not in out.err

    def test_missing_key_still_errors(self, hermes_home, capsys):
        """The pre-existing MISSING behavior (exit 1) is unchanged."""
        with pytest.raises(SystemExit) as exc:
            _get("agent.tool_search.does_not_exist")
        assert exc.value.code == 1

        out = capsys.readouterr()
        assert "Config key not set" in out.err

    def test_notice_suggests_typo_sibling_when_close(self, tmp_path, monkeypatch, capsys):
        """A near-miss sub-key keeps the set-path did-you-mean behavior (gateway.discord.*)."""
        home = tmp_path / ".hermes"
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text(yaml.dump({
            "gateway": {"strict": True, "trust_recent_file": False},
        }))
        monkeypatch.setenv("HERMES_HOME", str(home))

        _get("gateway.trust_recent_file")

        out = capsys.readouterr()
        assert 'false' in out.out
        assert "not a recognized config key" in out.err
        assert "trust_recent_files" in out.err
