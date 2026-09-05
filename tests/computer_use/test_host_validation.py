"""Unit tests for CUA host/origin allowlist validation and normalization."""

from __future__ import annotations

import pytest

from tools.computer_use.host_validation import (
    _valid_hostname,
    _valid_host_entry,
    _valid_origin_entry,
    validate_security_allowlists,
)


def test_valid_hosts_pass_through_and_inputs_unchanged():
    hosts = ["localhost:8765", "192.168.222.110:8765", "[::1]:8765", "localhost"]
    normalized_hosts, normalized_origins = validate_security_allowlists(hosts, [])
    assert normalized_hosts == hosts
    assert normalized_origins == []
    # Validation must copy, never mutate the caller's lists.
    assert hosts == ["localhost:8765", "192.168.222.110:8765", "[::1]:8765", "localhost"]


@pytest.mark.parametrize(
    "bad_host",
    [
        "*",
        "localhost *",
        "",
        "host with space",
        "café.example.com",  # non-ASCII labels are rejected
        "example.com/path",
        "user:pass@host",
        123,  # non-str entry
    ],
)
def test_invalid_host_entries_reject_the_allowlist(bad_host):
    with pytest.raises(ValueError):
        validate_security_allowlists([bad_host], [])


def test_valid_origins_normalized_to_lowercase_without_trailing_slash():
    origins = ["http://localhost:8765", "http://localhost:8765/", "https://example.com"]
    _, normalized_origins = validate_security_allowlists(["localhost:8765"], origins)
    assert normalized_origins == [
        "http://localhost:8765",
        "http://localhost:8765",  # trailing "/" stripped
        "https://example.com",
    ]


@pytest.mark.parametrize(
    "bad_origin",
    [
        "https://*",
        "ftp://example.com",
        "example.com",  # no scheme
        "",
    ],
)
def test_invalid_origin_entries_reject_the_allowlist(bad_origin):
    with pytest.raises(ValueError):
        validate_security_allowlists(["localhost:8765"], [bad_origin])


def test_case_insensitive_entries_normalize_to_lowercase():
    hosts, origins = validate_security_allowlists(
        ["Localhost:8443"], ["HTTP://LocalHost:8443"]
    )
    assert hosts == ["localhost:8443"]
    assert origins == ["http://localhost:8443"]


def test_string_inputs_raise_type_error():
    with pytest.raises(TypeError):
        validate_security_allowlists("localhost:8765", [])
    with pytest.raises(TypeError):
        validate_security_allowlists(["localhost:8765"], "http://localhost:8765")


def test_empty_host_list_rejected():
    with pytest.raises(ValueError, match="at least one host"):
        validate_security_allowlists([], [])


def test_one_invalid_host_rejects_the_whole_list():
    with pytest.raises(ValueError):
        validate_security_allowlists(["localhost:8765", "host with space"], [])


@pytest.mark.parametrize(
    "hostname",
    ["localhost", "example.com", "192.168.222.110", "::1", "a-b.example.com"],
)
def test_valid_hostnames(hostname):
    assert _valid_hostname(hostname)


@pytest.mark.parametrize(
    "hostname",
    ["", "-lead.example.com", "trail-.example.com", "café.example.com", "example.com/path"],
)
def test_invalid_hostnames(hostname):
    assert not _valid_hostname(hostname)


@pytest.mark.parametrize(
    "entry, valid",
    [
        ("localhost:8765", True),
        ("[::1]:8765", True),
        ("example.com/path", False),
        ("user:pass@host", False),
    ],
)
def test_valid_host_entry(entry, valid):
    assert _valid_host_entry(entry) is valid


@pytest.mark.parametrize(
    "entry, valid",
    [
        ("http://localhost:8765", True),
        ("http://localhost:8765/", True),
        ("ftp://example.com", False),
        ("example.com", False),
        ("https://example.com/../path", False),
    ],
)
def test_valid_origin_entry(entry, valid):
    assert _valid_origin_entry(entry) is valid