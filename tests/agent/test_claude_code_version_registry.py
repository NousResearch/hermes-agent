"""Claude Code version detection must not drift behind the version Anthropic gates against.

The user-agent carries a detected version; when no CLI is installed Hermes falls back to a
constant that only moves on a Hermes release. The registry probe replaces it with npm's current
``latest`` tag, cached for a day, and every failure degrades back to the constant.
"""

from __future__ import annotations

import json
import subprocess
import time

import pytest

import agent.anthropic_adapter as adapter
from agent.anthropic_adapter import (
    _CLAUDE_CODE_VERSION_FALLBACK,
    _detect_claude_code_version,
)


@pytest.fixture
def no_cli(monkeypatch):
    """No installed CLI: detection reaches the registry/fallback branch."""

    monkeypatch.setattr(adapter, "_claude_code_candidates", list)
    return monkeypatch


class _Resp:
    """Minimal ``urlopen`` response: a JSON body usable as a context manager."""

    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *exc) -> bool:
        return False


def _registry(monkeypatch, version: str) -> None:
    """A registry that answers with ``version`` for any request."""

    def urlopen(req, timeout=None):
        assert "dist-tags" in req.full_url
        return _Resp(json.dumps({"latest": version}).encode())

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)


def test_registry_version_replaces_stale_fallback(no_cli, monkeypatch, tmp_path):
    """No CLI installed → npm's current version, not the shipped constant."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _registry(monkeypatch, "2.1.299")

    assert _detect_claude_code_version() == "2.1.299" != _CLAUDE_CODE_VERSION_FALLBACK


def test_fresh_cache_avoids_the_network(no_cli, monkeypatch, tmp_path):
    """A cache younger than the TTL is used without any HTTP request."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cache = tmp_path / "cache" / "claude_code_registry_version.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps({"version": "2.1.250", "ts": time.time()}))

    def explode(*a, **k):
        raise AssertionError("a fresh cache must not hit the network")

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", explode)
    assert _detect_claude_code_version() == "2.1.250"


def test_stale_cache_is_refreshed_from_the_registry(no_cli, monkeypatch, tmp_path):
    """A cache older than the TTL triggers a live probe, not a stale answer."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cache = tmp_path / "cache" / "claude_code_registry_version.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps({"version": "2.1.100", "ts": time.time() - 3 * 86400}))
    _registry(monkeypatch, "2.1.299")

    assert _detect_claude_code_version() == "2.1.299"


def test_registry_outage_falls_back_to_constant(no_cli, monkeypatch, tmp_path):
    """An unreachable registry degrades to the shipped fallback, never an error."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def dead(req, timeout=None):
        raise OSError("network is down")

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", dead)

    assert _detect_claude_code_version() == _CLAUDE_CODE_VERSION_FALLBACK


def test_corrupt_cache_does_not_break_detection(no_cli, monkeypatch, tmp_path):
    """Unparseable cache content falls through to a live probe."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cache = tmp_path / "cache" / "claude_code_registry_version.json"
    cache.parent.mkdir(parents=True)
    cache.write_text("{{{not json")
    _registry(monkeypatch, "2.1.299")

    assert _detect_claude_code_version() == "2.1.299"


def test_installed_cli_still_wins(monkeypatch, tmp_path):
    """An installed CLI outranks the registry, which must not be consulted at all."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def boom(*a, **k):
        raise AssertionError("an installed CLI must not trigger a registry probe")

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", boom)

    def fake_candidates():
        return ["/usr/bin/fake-claude"]

    monkeypatch.setattr(adapter, "_claude_code_candidates", fake_candidates)

    def fake_run(cmd, *a, **k):
        assert cmd[0] == "/usr/bin/fake-claude"
        return subprocess.CompletedProcess(
            cmd, 0, stdout="9.9.9 (Claude Code)\n", stderr=""
        )

    monkeypatch.setattr(adapter.subprocess, "run", fake_run)

    assert _detect_claude_code_version() == "9.9.9"


def test_npm_mirror_override_is_honored(no_cli, monkeypatch, tmp_path):
    """A user's npm mirror answers the probe; the probe must not hard-code registry.npmjs.org."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    probed: list[str] = []

    def urlopen(req, timeout=None):
        probed.append(req.full_url)
        return _Resp(json.dumps({"latest": "2.1.299"}).encode())

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setenv("npm_config_registry", "https://npm.corp.example/")

    assert _detect_claude_code_version() == "2.1.299"
    assert probed and all(url.startswith("https://npm.corp.example/") for url in probed)


def test_successful_probe_is_cached_to_disk(no_cli, monkeypatch, tmp_path):
    """A good registry answer is persisted so the next cold start skips the network."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _registry(monkeypatch, "2.1.299")

    assert _detect_claude_code_version() == "2.1.299"
    cache = tmp_path / "cache" / "claude_code_registry_version.json"
    assert json.loads(cache.read_text())["version"] == "2.1.299"

    def boom(*a, **k):
        raise AssertionError("a warm cache must not hit the network")

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    adapter._claude_code_version_cache = None
    assert _detect_claude_code_version() == "2.1.299"
