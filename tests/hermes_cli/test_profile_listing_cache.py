"""``list_profiles`` re-reads a profile's YAML only when that file has changed.

The listing is the shared body of ``GET /api/profiles`` and JSON-RPC ``profiles.list``, which the
Bots roster polls every 5s per connection, and it parsed three YAML files per profile every time —
including an installer-seeded ``config.yaml`` (the annotated template, ~119KB) for two strings.

The memo must be invisible: every edit has to show up on the very next read, or `hermes profile
list` and the roster paint stale names and models. Only derived values are cached — the raw
readers keep their uncached contract, because callers write those documents back.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import hermes_cli.profiles as profiles


@pytest.fixture(autouse=True)
def _clear_memo():
    profiles._PROFILE_FILE_CACHE.clear()
    yield
    profiles._PROFILE_FILE_CACHE.clear()


@pytest.fixture
def profile_dir(tmp_path) -> Path:
    p = tmp_path / "profiles" / "bob"
    p.mkdir(parents=True)
    (p / "config.yaml").write_text("model:\n  provider: openai\n  default: gpt-4o\n", encoding="utf-8")
    (p / "profile.yaml").write_text("display_name: Bob\ndescription: first\n", encoding="utf-8")
    (p / "distribution.yaml").write_text("name: starter\nversion: 1.0.0\nsource: registry\n", encoding="utf-8")
    return p


def test_an_edited_config_model_is_seen_on_the_next_read(profile_dir):
    assert profiles._read_config_model(profile_dir) == ("gpt-4o", "openai")

    (profile_dir / "config.yaml").write_text(
        "model:\n  provider: anthropic\n  default: claude-opus-4.6\n", encoding="utf-8")

    assert profiles._read_config_model(profile_dir) == ("claude-opus-4.6", "anthropic")


def test_a_missing_file_is_not_cached_and_is_picked_up_when_created(tmp_path):
    bare = tmp_path / "profiles" / "bare"
    bare.mkdir(parents=True)

    assert profiles._read_distribution_meta(bare) == (None, None, None)
    assert not any(key[1] == "distribution" for key in profiles._PROFILE_FILE_CACHE)

    (bare / "distribution.yaml").write_text("name: late\nversion: 9\nsource: registry\n", encoding="utf-8")

    assert profiles._read_distribution_meta(bare) == ("late", 9, "registry")


def test_an_unchanged_file_is_parsed_once_across_repeated_reads(profile_dir, monkeypatch):
    """The saving itself — the roster poll stops re-parsing files that have not moved."""
    parsed: list = []
    real = profiles._load_yaml_dict

    def _counting(path):
        parsed.append(Path(path).name)
        return real(path)

    monkeypatch.setattr(profiles, "_load_yaml_dict", _counting)

    for _ in range(3):
        profiles.read_profile_meta(profile_dir)
        profiles._read_distribution_meta(profile_dir)

    assert parsed.count("profile.yaml") == 1
    assert parsed.count("distribution.yaml") == 1


def test_the_raw_config_reader_keeps_its_uncached_contract(profile_dir):
    """``read_user_config_raw`` feeds write-back round-trips, so it must NOT be memoised here."""
    from hermes_cli.config import read_user_config_raw

    config_path = profile_dir / "config.yaml"
    assert read_user_config_raw(config_path)["model"]["provider"] == "openai"

    config_path.write_text("model:\n  provider: xai\n  default: grok-4.6\n", encoding="utf-8")

    assert read_user_config_raw(config_path)["model"]["provider"] == "xai"
