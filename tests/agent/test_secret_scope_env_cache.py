"""The per-path ``load_env_file()`` memo in ``agent.secret_scope``.

``build_profile_secret_scope()`` sits on the gateway's hot paths (every turn, cron fire, MCP/browser
adoption, housekeeping drain) and used to re-read and re-parse the profile ``.env`` on each call.
A stale credential map is a worse failure than a slow one, so the contract under test is: parse once
while the file is unchanged, hand every caller its own dict, and see every real change on disk.

Files are written as bytes: ``Path.write_text`` translates ``\\n`` to ``\\r\\n`` on Windows, which
would break the same-size assertion.
"""

from __future__ import annotations

import os

import pytest

import agent.secret_scope as ss

# Comfortably past the coarsest mtime resolution in common use (FAT32 truncates to two seconds).
_MTIME_STEP_NS = 10_000_000_000


@pytest.fixture(autouse=True)
def _clear_cache():
    ss.invalidate_env_file_cache()
    yield
    ss.invalidate_env_file_cache()


def _count_parses(monkeypatch) -> list:
    calls: list = []
    real = ss._parse_env_text
    monkeypatch.setattr(ss, "_parse_env_text", lambda t: (calls.append(t), real(t))[1])
    return calls


def _rewrite_same_size(path, text: str) -> None:
    """Same length, different content: only the mtime distinguishes the versions."""
    previous = path.stat()
    path.write_bytes(text.encode("utf-8"))
    assert path.stat().st_size == previous.st_size
    bumped = previous.st_mtime_ns + _MTIME_STEP_NS
    os.utime(path, ns=(bumped, bumped))


def test_unchanged_file_is_parsed_once_and_each_caller_gets_its_own_dict(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_bytes(b"A=1\nB=2\n")
    calls = _count_parses(monkeypatch)
    monkeypatch.setattr("hermes_cli.env_loader.get_secret_source_values", lambda home: {"EXT": "vault"})

    first = ss.load_env_file(env)
    first["INJECTED"] = "must-not-persist"  # callers mutate what they get back
    for _ in range(5):
        assert ss.load_env_file(env) == {"A": "1", "B": "2"}
    # The motivating path layers external secrets onto the result; that must not poison the memo.
    scope = ss.build_profile_secret_scope(tmp_path)
    assert scope["EXT"] == "vault"
    assert ss.load_env_file(env) == {"A": "1", "B": "2"}

    assert len(calls) == 1, f"expected one parse, got {len(calls)}"


@pytest.mark.parametrize("prefix", [b"", b"\xef\xbb\xbf"])
def test_strict_read_rejects_invalid_utf8_after_nonstrict_cache_warmup(tmp_path, prefix):
    env = tmp_path / ".env"
    env.write_bytes(prefix + b"KEY=synthetic-\xff\n")
    expected = {"KEY": "synthetic-\u00ff"}
    assert ss.load_env_file(env) == expected
    # A tolerant profile load must not relax a subsequent MCP strict read.
    for _ in range(2):
        with pytest.raises(UnicodeDecodeError):
            ss.load_env_file(env, strict=True)
        assert ss.load_env_file(env) == expected


def test_edits_and_deletion_are_seen(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_bytes(b"KEY=aaa\n")
    calls = _count_parses(monkeypatch)
    assert ss.load_env_file(env) == {"KEY": "aaa"}

    _rewrite_same_size(env, "KEY=bbb\n")
    assert ss.load_env_file(env) == {"KEY": "bbb"}
    assert len(calls) == 2

    env.unlink()
    assert ss.load_env_file(env) == {}
    env.write_bytes(b"KEY=ccc\n")
    assert ss.load_env_file(env) == {"KEY": "ccc"}
    assert len(calls) == 3
