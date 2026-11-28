"""A launch re-parses a profile's config.yaml only when the file changed since a previous parse."""

import pytest

from pm import plugins_state


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_CONFIG_BACKEND", raising=False)
    monkeypatch.setattr(plugins_state, "_selection_cache_path", lambda: tmp_path / "cache" / "selection.json")
    home = tmp_path / "home"
    home.mkdir()
    return home


@pytest.fixture
def parses(monkeypatch):
    """Count the full YAML reads while leaving them real."""
    calls = []
    real = plugins_state.read_home_selection
    monkeypatch.setattr(plugins_state, "read_home_selection", lambda h: calls.append(h) or real(h))
    return calls


def _write(home, text):
    (home / "config.yaml").write_text(text, encoding="utf-8")


def test_an_unchanged_config_is_parsed_once_across_launches(home, parses):
    _write(home, "plugins:\n  enabled: [alpha, beta]\n  disabled: [beta]\nmemory:\n  provider: mem\nmodel: big\n")
    first = plugins_state._read_selection_cached(home)
    second = plugins_state._read_selection_cached(home)
    assert len(parses) == 1
    assert second == {"plugins": {"enabled": ["alpha", "beta"], "disabled": ["beta"]}, "memory": {"provider": "mem"}}
    assert first["plugins"] == second["plugins"] and first["memory"] == second["memory"]


def test_an_edited_config_is_parsed_again_and_the_new_selection_wins(home, parses):
    _write(home, "plugins:\n  enabled: [alpha]\n")
    assert plugins_state._read_selection_cached(home)["plugins"]["enabled"] == ["alpha"]
    _write(home, "plugins:\n  enabled: [alpha, gamma]\n")
    assert plugins_state._read_selection_cached(home)["plugins"]["enabled"] == ["alpha", "gamma"]
    assert len(parses) == 2


def test_a_same_size_rewrite_is_still_noticed(home, parses):
    _write(home, "plugins:\n  enabled: [alpha]\n")
    plugins_state._read_selection_cached(home)
    _write(home, "plugins:\n  enabled: [omega]\n")  # same length, so size alone would miss it
    assert plugins_state._read_selection_cached(home)["plugins"]["enabled"] == ["omega"]


def test_a_broken_config_still_refuses_and_is_never_cached(home, parses):
    _write(home, "plugins: [not, a, mapping]\n")
    for _ in range(2):
        with pytest.raises(ValueError):
            plugins_state._read_selection_cached(home)
    assert len(parses) == 2


def test_a_home_without_config_reads_as_no_selection(home):
    assert plugins_state._read_selection_cached(home) is None


def test_a_corrupt_cache_file_falls_back_to_the_parse(home, tmp_path, parses):
    _write(home, "plugins:\n  enabled: [alpha]\n")
    cache = tmp_path / "cache" / "selection.json"
    cache.parent.mkdir()
    cache.write_text("{not json", encoding="utf-8")
    assert plugins_state._read_selection_cached(home)["plugins"]["enabled"] == ["alpha"]
    assert len(parses) == 1


def test_a_remote_config_backend_never_uses_the_file_cache(home, parses, monkeypatch):
    from hermes_cli import config_backend

    monkeypatch.setenv(config_backend.BACKEND_ENV, "remote")
    _write(home, "plugins:\n  enabled: [alpha]\n")
    monkeypatch.setattr(plugins_state, "read_home_selection", lambda h: parses.append(h) or {"plugins": {}})
    plugins_state._read_selection_cached(home)
    plugins_state._read_selection_cached(home)
    assert len(parses) == 2
