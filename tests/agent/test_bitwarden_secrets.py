"""Hermetic tests for the Bitwarden Secrets Manager integration.

Secret/cache behavior stays offline. PM acquisition and executable invocation
are exercised by tests/pm/test_security_consumers.py.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path
from unittest import mock

import pytest


# Make the worktree importable without depending on the installed wheel.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.secret_sources import bitwarden as bw


@pytest.fixture
def bitwarden_source(monkeypatch):
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.t")
    monkeypatch.setattr(bw, "find_bws", lambda **_kw: Path("/fake/bws"))
    return bw.BitwardenSource()


def test_source_merges_projects_first_name_wins(bitwarden_source, monkeypatch, tmp_path):
    calls = []
    projects = {
        "own": ({"OWN_KEY": "own-only", "SHARED_KEY": "own-secret"}, ["own warning"]),
        "shared": ({"TEAM_KEY": "team-only", "SHARED_KEY": "shared-secret"}, ["shared warning"]),
        "third": ({"SHARED_KEY": "third-secret"}, ["third warning"]),
    }

    def fake_fetch(**kwargs):
        calls.append(kwargs)
        return projects[kwargs["project_id"]]

    monkeypatch.setattr(bw, "fetch_bitwarden_secrets", fake_fetch)
    result = bitwarden_source.fetch({
        "project_id": "own", "extra_project_ids": ["shared", "third"],
        "cache_ttl_seconds": 42, "server_url": "https://vault.bitwarden.eu",
        "encrypted_cache": {"enabled": True, "max_stale_seconds": 123},
    }, tmp_path)

    assert result.ok
    assert result.secrets == {
        "OWN_KEY": "own-only", "TEAM_KEY": "team-only", "SHARED_KEY": "own-secret",
    }
    assert calls == [{
        "access_token": "0.t", "project_id": pid, "binary": Path("/fake/bws"),
        "cache_ttl_seconds": 42.0, "server_url": "https://vault.bitwarden.eu",
        "home_path": tmp_path, "encrypted_cache_enabled": True,
        "encrypted_cache_max_stale_seconds": 123.0,
    } for pid in projects]
    assert result.warnings == [
        "own warning", "shared warning",
        "bitwarden: SHARED_KEY is in project own and shared; using own", "third warning",
    ]
    for secret in ("own-secret", "shared-secret", "third-secret"):
        assert secret not in " ".join(result.warnings)


@pytest.mark.parametrize("error,kind", [
    ("network unavailable", bw.ErrorKind.NETWORK),
    ("unauthorized (401)", bw.ErrorKind.AUTH_FAILED),
])
def test_source_extra_project_failure_is_fatal(
    bitwarden_source, monkeypatch, tmp_path, error, kind,
):
    calls = []

    def fake_fetch(**kwargs):
        calls.append(kwargs["project_id"])
        if kwargs["project_id"] == "shared-project":
            raise RuntimeError(error)
        return {"OWN_KEY": "own-secret"}, ["own warning"]

    monkeypatch.setattr(bw, "fetch_bitwarden_secrets", fake_fetch)
    result = bitwarden_source.fetch({
        "project_id": "own", "extra_project_ids": ["shared-project", "unused"],
    }, tmp_path)

    assert not result.ok
    assert result.secrets == {}
    assert result.error_kind == kind
    assert error in result.error
    assert "shared-project" in result.error
    assert result.warnings == ["own warning"]
    assert calls == ["own", "shared-project"]
    if kind == bw.ErrorKind.AUTH_FAILED:
        assert result.error.startswith("Bitwarden rejected the machine-account access token (BWS_ACCESS_TOKEN)")


@pytest.mark.parametrize("extras", [{}, {"extra_project_ids": []}])
def test_source_without_extra_projects_fetches_once(bitwarden_source, monkeypatch, tmp_path, extras):
    calls = []

    def fake_fetch(**kwargs):
        calls.append(kwargs["project_id"])
        return {"KEY": "value"}, ["original warning"]

    monkeypatch.setattr(bw, "fetch_bitwarden_secrets", fake_fetch)
    result = bitwarden_source.fetch({"project_id": "own", **extras}, tmp_path)

    assert calls == ["own"]
    assert result.ok
    assert result.secrets == {"KEY": "value"}
    assert result.warnings == ["original warning"]


@pytest.mark.parametrize("extras,expected", [
    (["own", "shared", "shared", "own", "third", "third", "", None, 7, {}, []],
     ["own", "shared", "third"]),
    ("shared", ["own", "shared"]),
    ("", ["own"]),
    (None, ["own"]),
    (17, ["own"]),
    ({"project": "shared"}, ["own"]),
])
def test_source_filters_extra_projects(bitwarden_source, monkeypatch, tmp_path, extras, expected):
    calls = []

    def fake_fetch(**kwargs):
        calls.append(kwargs["project_id"])
        return {}, []

    monkeypatch.setattr(bw, "fetch_bitwarden_secrets", fake_fetch)
    result = bitwarden_source.fetch({"project_id": "own", "extra_project_ids": extras}, tmp_path)

    assert result.ok
    assert calls == expected


@pytest.fixture(autouse=True)
def _reset_caches():
    bw._reset_cache_for_tests()
    yield
    bw._reset_cache_for_tests()


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Point Hermes at an isolated home directory."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Some modules cache get_hermes_home; clear if needed.
    import hermes_constants
    if hasattr(hermes_constants, "_HERMES_HOME_CACHE"):
        hermes_constants._HERMES_HOME_CACHE = None  # type: ignore[attr-defined]
    return home


# ---------------------------------------------------------------------------
# fetch_bitwarden_secrets
# ---------------------------------------------------------------------------


def _fake_bws_payload(items):
    return json.dumps(items)














def test_fetch_server_url_sets_env(monkeypatch, tmp_path):
    """server_url must be plumbed into the subprocess as BWS_SERVER_URL."""
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    payload = _fake_bws_payload([{"key": "K", "value": "v"}])

    captured_env = {}

    def fake_run(cmd, **kwargs):
        captured_env.update(kwargs["env"])
        return mock.Mock(returncode=0, stdout=payload, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    bw.fetch_bitwarden_secrets(
        access_token="0.t",
        project_id="p",
        binary=fake_binary,
        use_cache=False,
        server_url="https://vault.bitwarden.eu",
    )
    assert captured_env.get("BWS_SERVER_URL") == "https://vault.bitwarden.eu"








# ---------------------------------------------------------------------------
# apply_bitwarden_secrets — the public entry point used by env_loader
# ---------------------------------------------------------------------------
















# ---------------------------------------------------------------------------
# env_loader integration
# ---------------------------------------------------------------------------




def test_env_loader_calls_bsm_when_enabled(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "secrets:\n"
        "  bitwarden:\n"
        "    enabled: true\n"
        "    project_id: 'proj-1'\n"
        "    access_token_env: 'BWS_ACCESS_TOKEN'\n"
        "    cache_ttl_seconds: 0\n"
        "    override_existing: false\n"
        "    auto_install: false\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.t")
    monkeypatch.delenv("MY_BSM_KEY", raising=False)

    called = {"n": 0}

    def fake_fetch(**kwargs):
        called["n"] += 1
        assert kwargs["project_id"] == "proj-1"
        return {"MY_BSM_KEY": "from-bsm"}, []

    monkeypatch.setattr(
        "agent.secret_sources.bitwarden.find_bws",
        lambda **_kw: Path("/fake/bws"),
    )
    monkeypatch.setattr(
        "agent.secret_sources.bitwarden.fetch_bitwarden_secrets",
        fake_fetch,
    )
    from agent.secret_sources import registry as reg_module

    reg_module._reset_registry_for_tests()

    from hermes_cli.env_loader import _apply_external_secret_sources
    _apply_external_secret_sources(home)

    assert called["n"] == 1
    assert os.environ.get("MY_BSM_KEY") == "from-bsm"


# ---------------------------------------------------------------------------
# Disk-persisted cache (cross-process — speeds up back-to-back CLI invocations)
# ---------------------------------------------------------------------------








def test_disk_cache_key_mismatch_triggers_refetch(monkeypatch, tmp_path):
    """Disk cache entry written by a different token/project is ignored."""
    home = tmp_path / ".hermes"
    home.mkdir()
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    payload = _fake_bws_payload([{"key": "K1", "value": "v1"}])

    call_count = {"n": 0}
    def fake_run(*a, **kw):
        call_count["n"] += 1
        return mock.Mock(returncode=0, stdout=payload, stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)
    bw._reset_cache_for_tests(home)

    # Write a cache entry for a DIFFERENT token/project pair
    cache_path = bw._disk_cache_path(home)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({
        "key": "deadbeef00000000|other-project|",
        "secrets": {"OTHER": "should-not-leak"},
        "fetched_at": time.time(),
    }))

    secrets, _ = bw.fetch_bitwarden_secrets(
        access_token="0.t", project_id="proj-1", binary=fake_binary,
        cache_ttl_seconds=300, home_path=home,
    )
    # We must NOT have used the foreign cache entry
    assert secrets == {"K1": "v1"}
    assert "OTHER" not in secrets
    assert call_count["n"] == 1






@pytest.mark.platforms("linux")
def test_encrypted_cache_writes_without_plaintext(monkeypatch, tmp_path):
    """Encrypted cache stores last-good secrets without raw values on disk."""
    home = tmp_path / ".hermes"
    home.mkdir()
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    payload = _fake_bws_payload([{"key": "K1", "value": "secret-value"}])

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: mock.Mock(returncode=0, stdout=payload, stderr=""),
    )
    bw._reset_cache_for_tests(home)
    # A successful encrypted write must remove a pre-existing legacy plaintext
    # cache from the migration path.
    legacy_key = (bw._token_fingerprint("0.t"), "proj-1", "")
    bw._DISK_CACHE.write(
        legacy_key,
        bw._CachedFetch(secrets={"K1": "legacy"}, fetched_at=time.time()),
        300,
        home,
    )
    assert bw._disk_cache_path(home).exists()

    secrets, warnings = bw.fetch_bitwarden_secrets(
        access_token="0.t", project_id="proj-1", binary=fake_binary,
        cache_ttl_seconds=0, encrypted_cache_enabled=True,
        encrypted_cache_max_stale_seconds=604800, home_path=home,
    )

    assert secrets == {"K1": "secret-value"}
    assert warnings == []
    assert not bw._disk_cache_path(home).exists()
    cache_path = bw._encrypted_disk_cache_path(home)
    assert cache_path.exists()
    mode = stat.S_IMODE(os.stat(cache_path).st_mode)
    assert mode == 0o600, f"expected 0o600, got 0o{mode:o}"
    text = cache_path.read_text()
    assert "secret-value" not in text
    assert "0.t" not in text
    payload_disk = json.loads(text)
    assert set(payload_disk.keys()) == {
        "version", "key", "salt", "nonce", "ciphertext",
    }
    assert not bw._disk_cache_path(home).exists()




def test_encrypted_cache_falls_back_on_network_error(monkeypatch, tmp_path):
    """A fresh-enough encrypted cache is used when BWS is unreachable."""
    home = tmp_path / ".hermes"
    home.mkdir()
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    calls = {"n": 0}

    def fake_run(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return mock.Mock(
                returncode=0,
                stdout=_fake_bws_payload([{"key": "K1", "value": "cached"}]),
                stderr="",
            )
        return mock.Mock(
            returncode=1,
            stdout="",
            stderr="Error: network is unreachable",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    bw._reset_cache_for_tests(home)

    first, _ = bw.fetch_bitwarden_secrets(
        access_token="0.t", project_id="proj-1", binary=fake_binary,
        cache_ttl_seconds=0, encrypted_cache_enabled=True,
        encrypted_cache_max_stale_seconds=604800, home_path=home,
    )
    assert first == {"K1": "cached"}
    bw._CACHE.clear()

    second, warnings = bw.fetch_bitwarden_secrets(
        access_token="0.t", project_id="proj-1", binary=fake_binary,
        cache_ttl_seconds=0, encrypted_cache_enabled=True,
        encrypted_cache_max_stale_seconds=604800, home_path=home,
    )
    assert second == {"K1": "cached"}
    assert calls["n"] == 2
    assert len(warnings) == 1
    assert "stale ENCRYPTED disk cache" in warnings[0]
    assert "bws live fetch failed" in warnings[0]






# ---------------------------------------------------------------------------
# Stale disk cache fallback when live bws fetch fails
# ---------------------------------------------------------------------------


def _seed_stale_disk_cache(home, *, secrets, age_seconds, project_id="proj-1",
                           access_token="0.t", server_url=""):
    """Populate the disk cache as if a successful fetch happened `age_seconds`
    ago. Writes the JSON payload directly (same shape the shared DiskCache
    reads/writes) rather than going through DiskCache.write, since that
    would honor cache_ttl_seconds and refuse to persist an already-"stale"
    entry — this needs to land on disk regardless of TTL."""
    cache_key = (
        bw._token_fingerprint(access_token), project_id, server_url,
    )
    cache_path = bw._disk_cache_path(home)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({
        "key": bw._cache_key_str(cache_key),
        "secrets": secrets,
        "fetched_at": time.time() - age_seconds,
    }))


def test_stale_disk_cache_returned_when_bws_fails(monkeypatch, tmp_path):
    """When bws fails and the disk cache is stale, return the stale secrets
    with a warning rather than raising."""
    home = tmp_path / ".hermes"
    home.mkdir()
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    bw._reset_cache_for_tests(home)

    # Seed a stale (older than TTL) disk cache from a previous successful fetch
    _seed_stale_disk_cache(home, secrets={"OPENAI_API_KEY": "sk-old"},
                           age_seconds=3600)

    # Now simulate a BWS network failure
    def fail_run(*a, **kw):
        return mock.Mock(returncode=1, stdout="",
                         stderr="Error: dns resolution failed")
    monkeypatch.setattr(subprocess, "run", fail_run)

    secrets, warnings = bw.fetch_bitwarden_secrets(
        access_token="0.t", project_id="proj-1", binary=fake_binary,
        cache_ttl_seconds=300, home_path=home,
    )
    assert secrets == {"OPENAI_API_KEY": "sk-old"}
    assert len(warnings) == 1
    assert "stale disk cache" in warnings[0]
    assert "dns resolution failed" in warnings[0]










def test_stale_fallback_skipped_on_auth_failure(monkeypatch, tmp_path):
    """An AUTH_FAILED bws error must raise, not serve stale secrets — a bad
    access token indicates a real credential problem the caller needs to
    see, not a transient outage worth papering over."""
    home = tmp_path / ".hermes"
    home.mkdir()
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    bw._reset_cache_for_tests(home)

    _seed_stale_disk_cache(home, secrets={"K1": "v1"}, age_seconds=3600)

    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **kw: mock.Mock(returncode=1, stdout="",
                                   stderr="Error: unauthorized (401)"),
    )

    with pytest.raises(RuntimeError, match="unauthorized"):
        bw.fetch_bitwarden_secrets(
            access_token="0.t", project_id="proj-1", binary=fake_binary,
            cache_ttl_seconds=300, home_path=home,
        )



