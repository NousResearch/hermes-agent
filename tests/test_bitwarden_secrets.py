"""Hermetic tests for the Bitwarden Secrets Manager integration.

We never hit GitHub or Bitwarden in tests — subprocess + urllib are
mocked so the suite stays fast and offline-safe.  The "live" pull and
binary download are exercised manually by `hermes secrets bitwarden
setup` outside of pytest.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from unittest import mock

import pytest


# Make the worktree importable without depending on the installed wheel.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.secret_sources import bitwarden as bw  # noqa: E402


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
# _platform_asset_name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "system,machine,libc_text,expected",
    [
        ("Darwin", "x86_64", "",
         f"bws-macos-universal-{bw._BWS_VERSION}.zip"),
        ("Darwin", "arm64", "",
         f"bws-macos-universal-{bw._BWS_VERSION}.zip"),
        ("Linux", "x86_64", "glibc",
         f"bws-x86_64-unknown-linux-gnu-{bw._BWS_VERSION}.zip"),
        ("Linux", "x86_64", "musl libc",
         f"bws-x86_64-unknown-linux-musl-{bw._BWS_VERSION}.zip"),
        ("Linux", "aarch64", "",
         f"bws-aarch64-unknown-linux-gnu-{bw._BWS_VERSION}.zip"),
        ("Windows", "AMD64", "",
         f"bws-x86_64-pc-windows-msvc-{bw._BWS_VERSION}.zip"),
        ("Windows", "ARM64", "",
         f"bws-aarch64-pc-windows-msvc-{bw._BWS_VERSION}.zip"),
    ],
)
def test_platform_asset_name(system, machine, libc_text, expected):
    with mock.patch.object(bw.platform, "system", return_value=system), \
         mock.patch.object(bw.platform, "machine", return_value=machine), \
         mock.patch.object(
             bw.subprocess,
             "run",
             return_value=mock.Mock(stdout=libc_text, stderr=libc_text),
         ):
        assert bw._platform_asset_name() == expected


# ---------------------------------------------------------------------------
# install_bws — fully mocked HTTP
# ---------------------------------------------------------------------------


def _make_fake_zip(binary_bytes: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        # Mirror the real release archive: the Windows zip
        # (bws-x86_64-pc-windows-msvc-*.zip) ships bws.exe, while every
        # other platform ships a bare bws. Use the same platform switch
        # install_bws() uses so the mock member matches what it looks for.
        zf.writestr(bw._platform_binary_name(), binary_bytes)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# _safe_extract_member — zip-slip containment
# ---------------------------------------------------------------------------




@pytest.mark.parametrize(
    "evil_name",
    [
        "../escape",
        "../../escape",
        "sub/../../escape",
    ],
)
def test_safe_extract_member_rejects_traversal(tmp_path, evil_name):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(evil_name, b"pwned")
    buf.seek(0)

    dest = tmp_path / "extract"
    dest.mkdir()
    outside = tmp_path / "escape"

    with zipfile.ZipFile(buf) as zf:
        with pytest.raises(RuntimeError, match="unsafe archive member"):
            bw._safe_extract_member(zf, evil_name, dest)

    # The traversal target must not have been written.
    assert not outside.exists()






def test_install_bws_happy_path(hermes_home, monkeypatch):
    fake_binary = b"#!/bin/sh\necho 'bws fake 2.0.0'\n"
    zip_bytes = _make_fake_zip(fake_binary)
    asset_name = bw._platform_asset_name()
    checksum_text = (
        f"{hashlib.sha256(zip_bytes).hexdigest()}  {asset_name}\n"
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff  other-file\n"
    )

    def fake_download(url, dest):
        if url.endswith(".zip"):
            Path(dest).write_bytes(zip_bytes)
        elif url.endswith(".txt"):
            Path(dest).write_text(checksum_text)
        else:
            raise AssertionError(f"unexpected download url: {url}")

    monkeypatch.setattr(bw, "_http_download", fake_download)

    path = bw.install_bws()
    assert path.exists()
    assert path.read_bytes() == fake_binary
    # Executable bit set. POSIX-only: Windows os.chmod only toggles the
    # read-only bit, so S_IXUSR is never reported there.
    if os.name != "nt":
        assert path.stat().st_mode & stat.S_IXUSR






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

    monkeypatch.setattr(bw.subprocess, "run", fake_run)

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
    monkeypatch.setattr(bw.subprocess, "run", fake_run)
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






def test_encrypted_cache_writes_without_plaintext(monkeypatch, tmp_path):
    """Encrypted cache stores last-good secrets without raw values on disk."""
    home = tmp_path / ".hermes"
    home.mkdir()
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    payload = _fake_bws_payload([{"key": "K1", "value": "secret-value"}])

    monkeypatch.setattr(
        bw.subprocess,
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
    # POSIX-only: on Windows os.chmod only toggles the read-only bit, so a
    # writable file always reports 0o666 and an exact 0o600 check cannot hold.
    if os.name != "nt":
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

    monkeypatch.setattr(bw.subprocess, "run", fake_run)
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
    monkeypatch.setattr(bw.subprocess, "run", fail_run)

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
        bw.subprocess, "run",
        lambda *a, **kw: mock.Mock(returncode=1, stdout="",
                                   stderr="Error: unauthorized (401)"),
    )

    with pytest.raises(RuntimeError, match="unauthorized"):
        bw.fetch_bitwarden_secrets(
            access_token="0.t", project_id="proj-1", binary=fake_binary,
            cache_ttl_seconds=300, home_path=home,
        )




