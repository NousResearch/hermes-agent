"""Config language caching follows successful reads and reset generations."""

import contextvars
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

from agent import i18n, secret_scope
from gateway.run import _profile_runtime_scope
from hermes_cli import config
from hermes_constants import get_hermes_home


@pytest.fixture
def language_homes(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    homes = [tmp_path / ".hermes" / "profiles" / str(index) for index in range(10)]
    for index, home in enumerate(homes):
        home.mkdir(parents=True)
        (home / "config.yaml").write_text(
            f"display:\n  language: {'fr' if index == 0 else 'de'}\n", encoding="utf-8",
        )
    monkeypatch.setenv("HERMES_HOME", str(homes[0]))
    monkeypatch.delenv("HERMES_LANGUAGE", raising=False)
    secret_scope.set_multiplex_active(True)
    i18n.reset_language_cache()
    try:
        yield homes
    finally:
        i18n.reset_language_cache()
        secret_scope.set_multiplex_active(False)


@pytest.mark.parametrize("scenario", ["failed-read", "eviction"])
def test_config_language_cache_admits_only_successful_profile_reads(
    language_homes, monkeypatch, scenario,
):
    homes = language_homes
    reads = []
    original = config.load_config_readonly

    def read_config():
        reads.append(get_hermes_home())
        return original()

    monkeypatch.setattr(config, "load_config_readonly", read_config)

    def language(home):
        with _profile_runtime_scope(home):
            return i18n.get_language()

    if scenario == "failed-read":
        (homes[0] / "config.yaml").write_text("display: [\n", encoding="utf-8")
        before = [language(homes[0]), language(homes[1])]
        (homes[0] / "config.yaml").write_text("display:\n  language: fr\n", encoding="utf-8")
        after = [language(home) for home in (homes[0], homes[1], homes[0])]
        assert (before, after, reads) == (
            ["en", "de"], ["fr", "de", "fr"], [homes[0], homes[1], homes[0]],
        )
        return

    assert [language(home) for home in (homes[0], homes[1], homes[0])] == ["fr", "de", "fr"]
    for home in homes[2:8]:
        language(home)
    language(homes[0])
    language(homes[8])
    language(homes[0])
    language(homes[1])
    assert reads == [*homes[:8], homes[8], homes[1]]


def test_reset_during_profile_config_read_does_not_cache_the_old_language(
    language_homes, monkeypatch,
):
    homes = language_homes
    captured = Event()
    resume = Event()
    original = config.load_config_readonly

    def read_config():
        result = original()
        if get_hermes_home() == homes[0] and not captured.is_set():
            captured.set()
            assert resume.wait(10), "The delayed language read was not released"
        return result

    monkeypatch.setattr(config, "load_config_readonly", read_config)
    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            with _profile_runtime_scope(homes[0]):
                context = contextvars.copy_context()
                future = pool.submit(context.run, i18n.get_language)
            assert captured.wait(10), "The original profile config was not read"
            (homes[0] / "config.yaml").write_text("display:\n  language: ja\n", encoding="utf-8")
            i18n.reset_language_cache()
            with _profile_runtime_scope(homes[1]):
                middle = i18n.get_language()
            resume.set()
            first = future.result(timeout=10)
        finally:
            resume.set()
    subsequent = []
    for home in (homes[0], homes[1], homes[0]):
        with _profile_runtime_scope(home):
            subsequent.append(i18n.get_language())
    assert (first, middle, subsequent) == ("fr", "de", ["ja", "de", "ja"])
