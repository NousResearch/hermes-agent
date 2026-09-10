"""Parsing of the host-declared channel ownership stamps."""
import logging

import pytest

from hermes_cli import managed_platforms
from hermes_cli.managed_platforms import (
    DEFAULT_LABEL,
    LABEL_ENV,
    PLATFORMS_ENV,
    URL_ENV,
    load_managed_platforms,
)


def test_unset_means_nothing_managed():
    managed = load_managed_platforms({})
    assert not managed
    assert managed.platforms == {}
    assert managed.record_for("telegram") is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("telegram:native,discord:relay", {"telegram": "native", "discord": "relay"}),
        ("telegram", {"telegram": "native"}),
        (" Telegram:Native , DISCORD:relay ", {"telegram": "native", "discord": "relay"}),
        ("telegram:relay,telegram:native", {"telegram": "relay"}),
        ("telegram:bogus", {"telegram": "native"}),
        (",, :relay ,", {}),
    ],
)
def test_platform_entries(raw, expected):
    assert load_managed_platforms({PLATFORMS_ENV: raw}).platforms == expected


def test_record_carries_kind_label_and_url():
    managed = load_managed_platforms(
        {
            PLATFORMS_ENV: "telegram:native,discord:relay",
            LABEL_ENV: "Nous Portal",
            URL_ENV: "https://portal.example.com",
        }
    )
    assert managed.record_for("discord") == {
        "kind": "relay",
        "label": "Nous Portal",
        "url": "https://portal.example.com",
    }
    assert managed.manages_relay() is True
    assert managed.kind_of("slack") is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", DEFAULT_LABEL),
        ("   ", DEFAULT_LABEL),
        ("  Nous   Portal ", "Nous Portal"),
        ("x" * 100, "x" * 64),
    ],
)
def test_label_fallback_and_cap(raw, expected):
    managed = load_managed_platforms({PLATFORMS_ENV: "telegram", LABEL_ENV: raw})
    assert managed.label == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://portal.example.com/agents", "https://portal.example.com/agents"),
        ("http://localhost:3000", "http://localhost:3000"),
        ("ftp://portal.example.com", None),
        ("http://[", None),
        ("javascript:alert(1)", None),
        ("portal.example.com", None),
        ("", None),
    ],
)
def test_url_accepts_only_http_and_https(raw, expected):
    managed = load_managed_platforms({PLATFORMS_ENV: "telegram", URL_ENV: raw})
    assert managed.url == expected


def test_label_and_url_do_not_unlock_when_no_platforms_listed():
    managed = load_managed_platforms(
        {LABEL_ENV: "Nous Portal", URL_ENV: "https://portal.example.com"}
    )
    assert not managed
    assert managed.label == DEFAULT_LABEL
    assert managed.url is None


def test_unknown_kind_is_logged_once_per_stamp_value(caplog):
    managed_platforms._parse.cache_clear()
    with caplog.at_level(logging.WARNING, logger="hermes_cli.managed_platforms"):
        for _ in range(3):
            load_managed_platforms({PLATFORMS_ENV: "telegram:weird"})
    assert sum("unknown kind" in r.getMessage() for r in caplog.records) == 1
