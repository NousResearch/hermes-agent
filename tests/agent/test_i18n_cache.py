"""Bundled locale catalogs are served from a disk cache under HERMES_HOME while the YAML is unchanged."""

from __future__ import annotations

import os

import pytest

from agent import i18n, i18n_layers


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.fixture
def locales(tmp_path, monkeypatch, home):
    locales_dir = tmp_path / "locales"
    locales_dir.mkdir()
    (locales_dir / "en.yaml").write_text("greet:\n  hello: Hello\n  bye: Bye {name}\n", encoding="utf-8")
    monkeypatch.setattr(i18n, "_locales_dir", lambda: locales_dir)
    return locales_dir / "en.yaml"


@pytest.fixture
def parses(monkeypatch):
    calls = []
    real = i18n_layers.parse_locale_file

    def spy(path):
        calls.append(path)
        return real(path)

    monkeypatch.setattr(i18n_layers, "parse_locale_file", spy)
    return calls


def test_second_load_hits_cache_with_identical_catalog(locales, parses):
    first = i18n._load_bundled("en")
    second = i18n._load_bundled("en")

    assert first == {"greet.hello": "Hello", "greet.bye": "Bye {name}"}
    assert second == first
    assert len(parses) == 1


def test_edited_yaml_is_reparsed(locales, parses):
    i18n._load_bundled("en")
    locales.write_text("greet:\n  hello: Hallo there\n", encoding="utf-8")

    assert i18n._load_bundled("en") == {"greet.hello": "Hallo there"}
    assert len(parses) == 2


def test_mtime_only_change_is_reparsed(locales, parses):
    i18n._load_bundled("en")
    st = locales.stat()
    os.utime(locales, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))

    i18n._load_bundled("en")
    assert len(parses) == 2


def test_corrupt_cache_falls_back_to_parsing(locales, parses, home):
    i18n._load_bundled("en")
    cache_files = list((home / "cache" / "i18n").glob("en.yaml.*.json"))
    assert len(cache_files) == 1
    cache_files[0].write_text("{not json", encoding="utf-8")

    assert i18n._load_bundled("en") == {"greet.hello": "Hello", "greet.bye": "Bye {name}"}
    assert len(parses) == 2
    # The rewritten cache serves the next load again.
    i18n._load_bundled("en")
    assert len(parses) == 2

