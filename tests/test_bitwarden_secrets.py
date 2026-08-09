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
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt


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
        zf.writestr("bws", binary_bytes)
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
    # Executable bit set
    assert path.stat().st_mode & stat.S_IXUSR






# ---------------------------------------------------------------------------
# fetch_bitwarden_secrets
# ---------------------------------------------------------------------------


def _fake_bws_payload(items):
    return json.dumps(items)


def _seed_legacy_encrypted_cache(
    home,
    *,
    access_token="short-token",
    cache_key=None,
    inner=None,
    raw_plaintext=None,
):
    """Write an authenticated v1 cache fixture through its real crypto format."""
    if cache_key is None:
        cache_key = (bw._token_fingerprint(access_token), "proj-1", "")
    serialized_key = bw._cache_key_str(cache_key)
    salt = bytes(range(16))
    nonce = bytes(range(12))
    if raw_plaintext is None:
        if inner is None:
            inner = {"secrets": {"K1": "cached"}, "fetched_at": time.time()}
        raw_plaintext = json.dumps(inner, separators=(",", ":")).encode("utf-8")
    legacy_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        info=b"hermes-bws-encrypted-cache-v1",
    ).derive(access_token.encode("utf-8"))
    ciphertext = AESGCM(legacy_key).encrypt(
        nonce,
        raw_plaintext,
        serialized_key.encode("utf-8"),
    )
    path = bw._encrypted_disk_cache_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": bw._ENCRYPTED_CACHE_LEGACY_VERSION,
                "key": serialized_key,
                "salt": bw._b64e(salt),
                "nonce": bw._b64e(nonce),
                "ciphertext": bw._b64e(ciphertext),
            }
        )
    )
    return path














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
        server_url="  https://vault.bitwarden.eu  ",
    )
    assert captured_env.get("BWS_SERVER_URL") == "https://vault.bitwarden.eu"


def test_fetch_inherited_server_url_is_bound_to_cache_identity(
    monkeypatch,
    tmp_path,
):
    """An inherited endpoint must bind both bws and encrypted cache routing."""
    home = tmp_path / ".hermes"
    home.mkdir()
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    source_env = {"BWS_SERVER_URL": "https://vault.bitwarden.eu"}
    payloads = iter(
        [
            _fake_bws_payload([{"key": "REGION", "value": "eu"}]),
            _fake_bws_payload([{"key": "REGION", "value": "us"}]),
        ]
    )
    seen_urls = []

    def fake_run(cmd, **kwargs):
        seen_urls.append(kwargs["env"].get("BWS_SERVER_URL"))
        return mock.Mock(returncode=0, stdout=next(payloads), stderr="")

    monkeypatch.setattr(bw, "get_source_environment", lambda: source_env)
    monkeypatch.setattr(bw.subprocess, "run", fake_run)
    first, _ = bw.fetch_bitwarden_secrets(
        access_token="0.t",
        project_id="proj-1",
        binary=fake_binary,
        server_url="",
        encrypted_cache_enabled=True,
        cache_ttl_seconds=300,
        home_path=home,
    )
    bw._CACHE.clear()
    bw._CACHE_POLICY.clear()
    source_env["BWS_SERVER_URL"] = "https://vault.bitwarden.com"
    second, _ = bw.fetch_bitwarden_secrets(
        access_token="0.t",
        project_id="proj-1",
        binary=fake_binary,
        server_url="",
        encrypted_cache_enabled=True,
        cache_ttl_seconds=300,
        home_path=home,
    )

    assert first == {"REGION": "eu"}
    assert second == {"REGION": "us"}
    assert seen_urls == [
        "https://vault.bitwarden.eu",
        "https://vault.bitwarden.com",
    ]


def test_invalidation_marker_direct_fallback_survives_replace_failure(
    monkeypatch,
    tmp_path,
):
    """A failed atomic marker install still leaves a restart-visible veto."""
    home = tmp_path / ".hermes"
    home.mkdir()
    cache_key = (bw._token_fingerprint("0.t"), "proj-1", "")
    marker_path = bw._encrypted_cache_invalidation_marker_path(home)
    encrypted_path = bw._encrypted_disk_cache_path(home)
    original_unlink = Path.unlink

    def fail_encrypted_unlink(path, *args, **kwargs):
        if path == encrypted_path:
            raise PermissionError("forced encrypted cleanup failure")
        return original_unlink(path, *args, **kwargs)

    def fail_replace(*args, **kwargs):
        raise PermissionError("forced atomic replacement failure")

    monkeypatch.setattr(Path, "unlink", fail_encrypted_unlink)
    monkeypatch.setattr(bw.os, "replace", fail_replace)
    bw._mark_encrypted_cache_invalidated(cache_key, home)

    assert marker_path.exists()
    bw._ENCRYPTED_CACHE_INVALIDATIONS.clear()
    warnings = []
    assert (
        bw._read_encrypted_disk_cache(
            cache_key=cache_key,
            access_token="0.t",
            max_age_seconds=300,
            home_path=home,
            transition_warnings=warnings,
        )
        is None
    )
    assert warnings == [bw._encrypted_cache_invalidation_warning(home)]


def test_clear_caches_preserves_veto_when_encrypted_unlink_fails(
    monkeypatch,
    tmp_path,
):
    """Cache clearing must not erase a veto while old ciphertext remains."""
    home = tmp_path / ".hermes"
    home.mkdir()
    access_token = "0.t"
    cache_key = (bw._token_fingerprint(access_token), "proj-1", "")
    encrypted_path = bw._encrypted_disk_cache_path(home)
    assert bw._write_encrypted_disk_cache(
        cache_key=cache_key,
        access_token=access_token,
        entry=bw._CachedFetch(secrets={"K1": "old"}, fetched_at=time.time()),
        home_path=home,
    )
    original_unlink = Path.unlink

    def fail_encrypted_unlink(path, *args, **kwargs):
        if path == encrypted_path:
            raise PermissionError("forced encrypted cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_encrypted_unlink)
    bw.clear_caches(home)
    marker_path = bw._encrypted_cache_invalidation_marker_path(home)
    assert marker_path.exists() or bw._encrypted_cache_invalidation_marker_paths(home)[-1].exists()
    bw._ENCRYPTED_CACHE_INVALIDATIONS.clear()
    monkeypatch.setattr(Path, "unlink", original_unlink)
    warnings = []
    assert (
        bw._read_encrypted_disk_cache(
            cache_key=cache_key,
            access_token=access_token,
            max_age_seconds=300,
            home_path=home,
            transition_warnings=warnings,
        )
        is None
    )
    assert warnings == [bw._encrypted_cache_invalidation_warning(home)]


def _exercise_cross_home_process_only_veto(monkeypatch, tmp_path, *, adapter):
    """An unrelated home clear must preserve another home's only L2 veto."""
    home_a = tmp_path / "home-a"
    home_b = tmp_path / "home-b"
    home_a.mkdir()
    home_b.mkdir()
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    access_token = "0.t"
    encrypted_path_b = bw._encrypted_disk_cache_path(home_b)
    marker_paths_b = set(bw._encrypted_cache_invalidation_marker_paths(home_b))
    identity_a = str(bw._encrypted_disk_cache_path(home_a))
    identity_b = str(encrypted_path_b)
    payloads = iter(
        [
            _fake_bws_payload([{"key": "K", "value": "old-b"}]),
            _fake_bws_payload([{"key": "K", "value": "plaintext-new-b"}]),
            _fake_bws_payload([{"key": "K", "value": "live-latest-b"}]),
        ]
    )
    calls = {"n": 0}

    def fake_run(*args, **kwargs):
        calls["n"] += 1
        return mock.Mock(returncode=0, stdout=next(payloads), stderr="")

    original_unlink = Path.unlink
    original_replace = bw.os.replace
    original_mkstemp = bw.tempfile.mkstemp
    original_open = Path.open

    def fail_encrypted_unlink(path, *args, **kwargs):
        if path == encrypted_path_b:
            raise PermissionError("forced encrypted cleanup failure")
        return original_unlink(path, *args, **kwargs)

    def fail_encrypted_replace(source, destination, *args, **kwargs):
        if Path(destination) == encrypted_path_b:
            raise PermissionError("forced encrypted tombstone failure")
        return original_replace(source, destination, *args, **kwargs)

    def fail_marker_mkstemp(*args, **kwargs):
        if kwargs.get("prefix") == ".bws_cache_enc_marker_":
            raise PermissionError("forced atomic marker failure")
        return original_mkstemp(*args, **kwargs)

    def fail_marker_open(path, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if path in marker_paths_b and "w" in mode:
            raise PermissionError("forced direct marker failure")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(bw.subprocess, "run", fake_run)
    source = bw.BitwardenSource()
    encrypted_cfg = {
        "project_id": "proj-1",
        "cache_ttl_seconds": 300,
        "encrypted_cache": {"enabled": True},
    }
    plaintext_cfg = {
        **encrypted_cfg,
        "cache_ttl_seconds": 0,
        "encrypted_cache": {"enabled": False},
    }
    if adapter:
        monkeypatch.setattr(bw, "find_bws", lambda **kwargs: fake_binary)
        monkeypatch.setattr(
            bw,
            "get_source_environment",
            lambda: {"BWS_ACCESS_TOKEN": access_token},
        )
        old_result = source.fetch(encrypted_cfg, home_b)
        old = old_result.secrets
        assert old_result.error is None
    else:
        old, _ = bw.fetch_bitwarden_secrets(
            access_token=access_token,
            project_id="proj-1",
            binary=fake_binary,
            cache_ttl_seconds=300,
            encrypted_cache_enabled=True,
            home_path=home_b,
            source_env={},
        )

    monkeypatch.setattr(Path, "unlink", fail_encrypted_unlink)
    monkeypatch.setattr(bw.os, "replace", fail_encrypted_replace)
    monkeypatch.setattr(bw.tempfile, "mkstemp", fail_marker_mkstemp)
    monkeypatch.setattr(Path, "open", fail_marker_open)
    if adapter:
        newer_result = source.fetch(plaintext_cfg, home_b)
        newer = newer_result.secrets
        warnings = newer_result.warnings
        assert newer_result.error is None
    else:
        newer, warnings = bw.fetch_bitwarden_secrets(
            access_token=access_token,
            project_id="proj-1",
            binary=fake_binary,
            cache_ttl_seconds=0,
            home_path=home_b,
            source_env={},
        )
    monkeypatch.setattr(Path, "unlink", original_unlink)
    monkeypatch.setattr(bw.os, "replace", original_replace)
    monkeypatch.setattr(bw.tempfile, "mkstemp", original_mkstemp)
    monkeypatch.setattr(Path, "open", original_open)

    assert old == {"K": "old-b"}
    assert newer == {"K": "plaintext-new-b"}
    assert warnings == [bw._encrypted_cache_invalidation_warning(home_b)]
    assert encrypted_path_b.exists()
    assert not any(path.exists() for path in marker_paths_b)
    assert identity_b in bw._ENCRYPTED_CACHE_INVALIDATIONS
    bw._ENCRYPTED_CACHE_INVALIDATIONS.add(identity_a)

    bw.clear_caches(home_a)

    assert identity_a not in bw._ENCRYPTED_CACHE_INVALIDATIONS
    assert identity_b in bw._ENCRYPTED_CACHE_INVALIDATIONS
    if adapter:
        latest_result = source.fetch(encrypted_cfg, home_b)
        latest = latest_result.secrets
        assert latest_result.error is None
    else:
        latest, _ = bw.fetch_bitwarden_secrets(
            access_token=access_token,
            project_id="proj-1",
            binary=fake_binary,
            cache_ttl_seconds=300,
            encrypted_cache_enabled=True,
            home_path=home_b,
            source_env={},
        )

    assert latest == {"K": "live-latest-b"}
    assert calls["n"] == 3
    assert identity_b not in bw._ENCRYPTED_CACHE_INVALIDATIONS


def test_clear_caches_preserves_other_home_process_only_veto_direct(
    monkeypatch,
    tmp_path,
):
    _exercise_cross_home_process_only_veto(
        monkeypatch,
        tmp_path,
        adapter=False,
    )


def test_clear_caches_preserves_other_home_process_only_veto_adapter(
    monkeypatch,
    tmp_path,
):
    _exercise_cross_home_process_only_veto(
        monkeypatch,
        tmp_path,
        adapter=True,
    )


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root bypasses directory permission bits",
)
def test_invalidation_marker_falls_back_outside_read_only_cache_dir(
    tmp_path,
):
    """A read-only cache directory still gets a home-level durable veto."""
    home = tmp_path / ".hermes"
    cache_dir = home / "cache"
    cache_dir.mkdir(parents=True)
    cache_key = (bw._token_fingerprint("0.t"), "proj-1", "")
    primary = bw._encrypted_cache_invalidation_marker_path(home)
    fallback = bw._encrypted_cache_invalidation_marker_paths(home)[-1]
    cache_dir.chmod(0o500)
    try:
        bw._mark_encrypted_cache_invalidated(cache_key, home)
        assert not primary.exists()
        assert fallback.exists()
        bw._ENCRYPTED_CACHE_INVALIDATIONS.clear()
        assert bw._encrypted_cache_was_invalidated(cache_key, home)
    finally:
        cache_dir.chmod(0o700)
        bw._remove_encrypted_cache_invalidation_markers(home)








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
    mode = stat.S_IMODE(os.stat(cache_path).st_mode)
    assert mode == 0o600, f"expected 0o600, got 0o{mode:o}"
    text = cache_path.read_text()
    assert "secret-value" not in text
    assert "0.t" not in text
    payload_disk = json.loads(text)
    assert set(payload_disk.keys()) == {
        "version", "context", "salt", "nonce", "ciphertext",
    }
    assert payload_disk["version"] == bw._ENCRYPTED_CACHE_VERSION
    assert payload_disk["context"] == {
        "project_id": "proj-1",
        "server_url": "",
    }
    fast_fingerprint = hashlib.sha256(b"0.t").hexdigest()[:16]
    assert fast_fingerprint not in json.dumps(payload_disk, sort_keys=True)
    assert not bw._disk_cache_path(home).exists()


def test_encrypted_cache_writer_reports_plaintext_cleanup_failure(
    monkeypatch,
    tmp_path,
):
    """Encrypted replacement alone does not complete the storage transition."""
    home = tmp_path / ".hermes"
    home.mkdir()
    access_token = "0.t"
    cache_key = (bw._token_fingerprint(access_token), "proj-1", "")
    entry = bw._CachedFetch(secrets={"K1": "fresh"}, fetched_at=time.time())
    plaintext_path = bw._disk_cache_path(home)
    bw._DISK_CACHE.write(
        cache_key,
        bw._CachedFetch(secrets={"K1": "legacy"}, fetched_at=time.time()),
        300,
        home,
    )
    original_unlink = Path.unlink

    def fail_plaintext_unlink(path, *args, **kwargs):
        if path == plaintext_path:
            raise PermissionError("forced plaintext cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_plaintext_unlink)

    written = bw._write_encrypted_disk_cache(
        cache_key=cache_key,
        access_token=access_token,
        entry=entry,
        home_path=home,
    )

    assert written is False
    assert bw._encrypted_disk_cache_path(home).exists()
    assert plaintext_path.exists()
    assert "legacy" in plaintext_path.read_text()


def test_encrypted_cache_preserves_known_unrelated_plaintext_entry(tmp_path):
    """Encrypted cleanup must not delete a plaintext entry for another route."""
    home = tmp_path / ".hermes"
    home.mkdir()
    access_token = "0.t"
    cache_key = (bw._token_fingerprint(access_token), "proj-1", "")
    unrelated_key = (bw._token_fingerprint("other-token"), "proj-2", "")
    plaintext_path = bw._disk_cache_path(home)
    bw._DISK_CACHE.write(
        unrelated_key,
        bw._CachedFetch(secrets={"OTHER": "preserved"}, fetched_at=time.time()),
        300,
        home,
    )

    assert bw._write_encrypted_disk_cache(
        cache_key=cache_key,
        access_token=access_token,
        entry=bw._CachedFetch(secrets={"K1": "cached"}, fetched_at=time.time()),
        home_path=home,
    )
    cached = bw._read_encrypted_disk_cache(
        cache_key=cache_key,
        access_token=access_token,
        max_age_seconds=300,
        home_path=home,
    )

    assert cached is not None
    assert cached.secrets == {"K1": "cached"}
    assert plaintext_path.exists()
    assert json.loads(plaintext_path.read_text())["key"] == bw._cache_key_str(
        unrelated_key
    )


def test_encrypted_cache_fresh_fetch_warns_when_plaintext_cleanup_fails(
    monkeypatch,
    tmp_path,
):
    """A fresh fetch must disclose that the plaintext predecessor remains."""
    home = tmp_path / ".hermes"
    home.mkdir()
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    access_token = "0.t"
    cache_key = (bw._token_fingerprint(access_token), "proj-1", "")
    plaintext_path = bw._disk_cache_path(home)
    bw._DISK_CACHE.write(
        cache_key,
        bw._CachedFetch(secrets={"K1": "legacy"}, fetched_at=time.time()),
        300,
        home,
    )
    payload = _fake_bws_payload([{"key": "K1", "value": "fresh"}])
    monkeypatch.setattr(
        bw.subprocess,
        "run",
        lambda *a, **kw: mock.Mock(returncode=0, stdout=payload, stderr=""),
    )
    original_unlink = Path.unlink

    def fail_plaintext_unlink(path, *args, **kwargs):
        if path == plaintext_path:
            raise PermissionError("forced plaintext cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_plaintext_unlink)

    secrets, warnings = bw.fetch_bitwarden_secrets(
        access_token=access_token,
        project_id="proj-1",
        binary=fake_binary,
        cache_ttl_seconds=0,
        encrypted_cache_enabled=True,
        home_path=home,
    )

    assert secrets == {"K1": "fresh"}
    assert bw._encrypted_disk_cache_path(home).exists()
    assert plaintext_path.exists()
    assert len(warnings) == 1
    assert "plaintext" in warnings[0].lower()
    assert str(plaintext_path) in warnings[0]


def test_encrypted_cache_cleanup_failure_does_not_return_pending_l1(
    monkeypatch,
    tmp_path,
):
    """A pending encrypted transition must not be served from L1 again."""
    home = tmp_path / ".hermes"
    home.mkdir()
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    access_token = "0.t"
    cache_key = (bw._token_fingerprint(access_token), "proj-1", "")
    plaintext_path = bw._disk_cache_path(home)
    bw._DISK_CACHE.write(
        cache_key,
        bw._CachedFetch(secrets={"K1": "legacy"}, fetched_at=time.time()),
        300,
        home,
    )
    payloads = iter(
        [
            _fake_bws_payload([{"key": "K1", "value": "first"}]),
            _fake_bws_payload([{"key": "K1", "value": "second"}]),
        ]
    )
    calls = {"n": 0}

    def fake_run(*args, **kwargs):
        calls["n"] += 1
        return mock.Mock(returncode=0, stdout=next(payloads), stderr="")

    monkeypatch.setattr(bw.subprocess, "run", fake_run)
    original_unlink = Path.unlink

    def fail_plaintext_unlink(path, *args, **kwargs):
        if path == plaintext_path:
            raise PermissionError("forced plaintext cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_plaintext_unlink)

    first, first_warnings = bw.fetch_bitwarden_secrets(
        access_token=access_token,
        project_id="proj-1",
        binary=fake_binary,
        cache_ttl_seconds=300,
        encrypted_cache_enabled=True,
        encrypted_cache_max_stale_seconds=300,
        home_path=home,
    )
    second, second_warnings = bw.fetch_bitwarden_secrets(
        access_token=access_token,
        project_id="proj-1",
        binary=fake_binary,
        cache_ttl_seconds=300,
        encrypted_cache_enabled=True,
        encrypted_cache_max_stale_seconds=300,
        home_path=home,
    )

    assert first == {"K1": "first"}
    assert first_warnings and str(plaintext_path) in first_warnings[0]
    assert second == {"K1": "second"}
    assert second_warnings and str(plaintext_path) in second_warnings[0]
    assert calls["n"] == 2
    assert plaintext_path.exists()


def test_plaintext_to_encrypted_mode_switch_bypasses_plaintext_l1(
    monkeypatch,
    tmp_path,
):
    """Encrypted mode cannot reuse a same-process plaintext-era L1 entry."""
    home = tmp_path / ".hermes"
    home.mkdir()
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    access_token = "0.t"
    plaintext_path = bw._disk_cache_path(home)
    encrypted_path = bw._encrypted_disk_cache_path(home)
    payloads = iter(
        [
            _fake_bws_payload([{"key": "K1", "value": "plaintext-era"}]),
            _fake_bws_payload([{"key": "K1", "value": "encrypted-era"}]),
        ]
    )
    calls = {"n": 0}

    def fake_run(*args, **kwargs):
        calls["n"] += 1
        return mock.Mock(returncode=0, stdout=next(payloads), stderr="")

    monkeypatch.setattr(bw.subprocess, "run", fake_run)
    first, first_warnings = bw.fetch_bitwarden_secrets(
        access_token=access_token,
        project_id="proj-1",
        binary=fake_binary,
        cache_ttl_seconds=300,
        home_path=home,
    )
    assert first == {"K1": "plaintext-era"}
    assert first_warnings == []
    assert plaintext_path.exists()

    original_unlink = Path.unlink

    def fail_plaintext_unlink(path, *args, **kwargs):
        if path == plaintext_path:
            raise PermissionError("forced plaintext cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_plaintext_unlink)
    second, second_warnings = bw.fetch_bitwarden_secrets(
        access_token=access_token,
        project_id="proj-1",
        binary=fake_binary,
        cache_ttl_seconds=300,
        encrypted_cache_enabled=True,
        encrypted_cache_max_stale_seconds=300,
        home_path=home,
    )

    assert second == {"K1": "encrypted-era"}
    assert second_warnings and str(plaintext_path) in second_warnings[0]
    assert calls["n"] == 2
    assert encrypted_path.exists()
    assert json.loads(encrypted_path.read_text())["version"] == (
        bw._ENCRYPTED_CACHE_VERSION
    )
    assert plaintext_path.exists()


def test_encrypted_plaintext_encrypted_does_not_resurrect_old_l2(
    monkeypatch,
    tmp_path,
):
    """A newer plaintext fetch invalidates an older encrypted L2 entry."""
    home = tmp_path / ".hermes"
    home.mkdir()
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    access_token = "0.t"
    cache_key = (bw._token_fingerprint(access_token), "proj-1", "")
    encrypted_path = bw._encrypted_disk_cache_path(home)
    plaintext_path = bw._disk_cache_path(home)
    payloads = iter(
        [
            _fake_bws_payload([{"key": "K1", "value": "encrypted-old"}]),
            _fake_bws_payload([{"key": "K1", "value": "plaintext-new"}]),
            _fake_bws_payload([{"key": "K1", "value": "encrypted-new"}]),
        ]
    )
    calls = {"n": 0}

    def fake_run(*args, **kwargs):
        calls["n"] += 1
        return mock.Mock(returncode=0, stdout=next(payloads), stderr="")

    monkeypatch.setattr(bw.subprocess, "run", fake_run)
    first, _ = bw.fetch_bitwarden_secrets(
        access_token=access_token,
        project_id="proj-1",
        binary=fake_binary,
        cache_ttl_seconds=300,
        encrypted_cache_enabled=True,
        encrypted_cache_max_stale_seconds=300,
        home_path=home,
    )
    assert first == {"K1": "encrypted-old"}

    assert bw._write_encrypted_disk_cache(
        cache_key=cache_key,
        access_token=access_token,
        entry=bw._CachedFetch(
            secrets={"K1": "encrypted-old"},
            fetched_at=time.time() - 10,
        ),
        home_path=home,
    )
    bw._CACHE.clear()
    bw._CACHE_POLICY.clear()
    second, _ = bw.fetch_bitwarden_secrets(
        access_token=access_token,
        project_id="proj-1",
        binary=fake_binary,
        cache_ttl_seconds=300,
        home_path=home,
    )
    assert second == {"K1": "plaintext-new"}
    assert plaintext_path.exists()

    third, _ = bw.fetch_bitwarden_secrets(
        access_token=access_token,
        project_id="proj-1",
        binary=fake_binary,
        cache_ttl_seconds=300,
        encrypted_cache_enabled=True,
        encrypted_cache_max_stale_seconds=300,
        home_path=home,
    )

    assert third == {"K1": "encrypted-new"}
    assert calls["n"] == 3
    assert encrypted_path.exists()
    assert not plaintext_path.exists()


def test_encrypted_plaintext_encrypted_does_not_resurrect_old_l2_after_reload(
    monkeypatch,
    tmp_path,
):
    """L2 invalidation remains effective without process-local cache state."""
    home = tmp_path / ".hermes"
    home.mkdir()
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    access_token = "0.t"
    cache_key = (bw._token_fingerprint(access_token), "proj-1", "")
    encrypted_path = bw._encrypted_disk_cache_path(home)
    plaintext_path = bw._disk_cache_path(home)
    payloads = iter(
        [
            _fake_bws_payload([{"key": "K1", "value": "encrypted-old"}]),
            _fake_bws_payload([{"key": "K1", "value": "plaintext-new"}]),
            _fake_bws_payload([{"key": "K1", "value": "encrypted-new"}]),
        ]
    )
    calls = {"n": 0}

    def fake_run(*args, **kwargs):
        calls["n"] += 1
        return mock.Mock(returncode=0, stdout=next(payloads), stderr="")

    monkeypatch.setattr(bw.subprocess, "run", fake_run)
    first, _ = bw.fetch_bitwarden_secrets(
        access_token=access_token,
        project_id="proj-1",
        binary=fake_binary,
        cache_ttl_seconds=300,
        encrypted_cache_enabled=True,
        encrypted_cache_max_stale_seconds=300,
        home_path=home,
    )
    assert first == {"K1": "encrypted-old"}
    assert bw._write_encrypted_disk_cache(
        cache_key=cache_key,
        access_token=access_token,
        entry=bw._CachedFetch(
            secrets={"K1": "encrypted-old"},
            fetched_at=time.time() - 10,
        ),
        home_path=home,
    )
    # Exercise the reader's timestamp guard independently of direct L2
    # invalidation, as a fresh process may race with an older writer.
    monkeypatch.setattr(
        bw,
        "_invalidate_encrypted_disk_cache",
        lambda *args, **kwargs: None,
    )
    second, _ = bw.fetch_bitwarden_secrets(
        access_token=access_token,
        project_id="proj-1",
        binary=fake_binary,
        cache_ttl_seconds=300,
        home_path=home,
    )
    assert second == {"K1": "plaintext-new"}
    assert plaintext_path.exists()

    # A fresh process/reload has no policy-aware L1 state to consult.
    bw._CACHE.clear()
    bw._CACHE_POLICY.clear()
    third, _ = bw.fetch_bitwarden_secrets(
        access_token=access_token,
        project_id="proj-1",
        binary=fake_binary,
        cache_ttl_seconds=300,
        encrypted_cache_enabled=True,
        encrypted_cache_max_stale_seconds=300,
        home_path=home,
    )

    assert third == {"K1": "encrypted-new"}
    assert calls["n"] == 3
    assert encrypted_path.exists()
    assert not plaintext_path.exists()


def test_plaintext_fetch_without_ttl_invalidates_encrypted_l2(
    monkeypatch,
    tmp_path,
):
    """Disabling plaintext caching must still discard an old encrypted L2."""
    home = tmp_path / ".hermes"
    home.mkdir()
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    access_token = "0.t"
    cache_key = (bw._token_fingerprint(access_token), "proj-1", "")
    encrypted_path = bw._encrypted_disk_cache_path(home)
    assert bw._write_encrypted_disk_cache(
        cache_key=cache_key,
        access_token=access_token,
        entry=bw._CachedFetch(
            secrets={"K1": "encrypted-old"}, fetched_at=time.time()
        ),
        home_path=home,
    )
    payloads = iter(
        [
            _fake_bws_payload([{"key": "K1", "value": "plaintext-new"}]),
            _fake_bws_payload([{"key": "K1", "value": "encrypted-new"}]),
        ]
    )
    calls = {"n": 0}

    def fake_run(*args, **kwargs):
        calls["n"] += 1
        return mock.Mock(returncode=0, stdout=next(payloads), stderr="")

    monkeypatch.setattr(bw.subprocess, "run", fake_run)
    first, _ = bw.fetch_bitwarden_secrets(
        access_token=access_token,
        project_id="proj-1",
        binary=fake_binary,
        cache_ttl_seconds=300,
        encrypted_cache_enabled=True,
        encrypted_cache_max_stale_seconds=300,
        home_path=home,
    )
    plaintext, _ = bw.fetch_bitwarden_secrets(
        access_token=access_token,
        project_id="proj-1",
        binary=fake_binary,
        cache_ttl_seconds=0,
        home_path=home,
    )
    encrypted, _ = bw.fetch_bitwarden_secrets(
        access_token=access_token,
        project_id="proj-1",
        binary=fake_binary,
        cache_ttl_seconds=300,
        encrypted_cache_enabled=True,
        encrypted_cache_max_stale_seconds=300,
        home_path=home,
    )

    assert first == {"K1": "encrypted-old"}
    assert plaintext == {"K1": "plaintext-new"}
    assert encrypted == {"K1": "encrypted-new"}
    assert calls["n"] == 2
    assert encrypted_path.exists()


def test_plaintext_ttl_zero_invalidation_tombstones_on_unlink_failure(
    monkeypatch,
    tmp_path,
):
    """A failed encrypted unlink cannot resurrect an old cache later."""
    home = tmp_path / ".hermes"
    home.mkdir()
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    access_token = "0.t"
    cache_key = (bw._token_fingerprint(access_token), "proj-1", "")
    encrypted_path = bw._encrypted_disk_cache_path(home)
    assert bw._write_encrypted_disk_cache(
        cache_key=cache_key,
        access_token=access_token,
        entry=bw._CachedFetch(
            secrets={"K1": "encrypted-old"}, fetched_at=time.time()
        ),
        home_path=home,
    )
    payloads = iter(
        [
            _fake_bws_payload([{"key": "K1", "value": "plaintext-new"}]),
            _fake_bws_payload([{"key": "K1", "value": "encrypted-new"}]),
        ]
    )
    calls = {"n": 0}

    def fake_run(*args, **kwargs):
        calls["n"] += 1
        return mock.Mock(returncode=0, stdout=next(payloads), stderr="")

    original_unlink = Path.unlink

    def fail_encrypted_unlink(path, *args, **kwargs):
        if path == encrypted_path:
            raise PermissionError("forced encrypted cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(bw.subprocess, "run", fake_run)
    monkeypatch.setattr(Path, "unlink", fail_encrypted_unlink)
    first, _ = bw.fetch_bitwarden_secrets(
        access_token=access_token,
        project_id="proj-1",
        binary=fake_binary,
        cache_ttl_seconds=300,
        encrypted_cache_enabled=True,
        encrypted_cache_max_stale_seconds=300,
        home_path=home,
    )
    plaintext, _ = bw.fetch_bitwarden_secrets(
        access_token=access_token,
        project_id="proj-1",
        binary=fake_binary,
        cache_ttl_seconds=0,
        home_path=home,
    )
    bw._CACHE.clear()
    bw._CACHE_POLICY.clear()
    encrypted, _ = bw.fetch_bitwarden_secrets(
        access_token=access_token,
        project_id="proj-1",
        binary=fake_binary,
        cache_ttl_seconds=300,
        encrypted_cache_enabled=True,
        encrypted_cache_max_stale_seconds=300,
        home_path=home,
    )

    assert first == {"K1": "encrypted-old"}
    assert plaintext == {"K1": "plaintext-new"}
    assert encrypted == {"K1": "encrypted-new"}
    assert calls["n"] == 2
    assert json.loads(encrypted_path.read_text())["version"] == (
        bw._ENCRYPTED_CACHE_VERSION
    )


def test_plaintext_ttl_zero_invalidation_failure_blocks_l1_l2_rollback(
    monkeypatch,
    tmp_path,
):
    """Failed unlink and tombstone writes must not restore old L2 in-process."""
    home = tmp_path / ".hermes"
    home.mkdir()
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    access_token = "0.t"
    cache_key = (bw._token_fingerprint(access_token), "proj-1", "")
    encrypted_path = bw._encrypted_disk_cache_path(home)
    assert bw._write_encrypted_disk_cache(
        cache_key=cache_key,
        access_token=access_token,
        entry=bw._CachedFetch(
            secrets={"K1": "encrypted-old"}, fetched_at=time.time()
        ),
        home_path=home,
    )
    payloads = iter(
        [
            _fake_bws_payload([{"key": "K1", "value": "plaintext-new"}]),
            _fake_bws_payload([{"key": "K1", "value": "encrypted-new"}]),
        ]
    )
    calls = {"n": 0}

    def fake_run(*args, **kwargs):
        calls["n"] += 1
        return mock.Mock(returncode=0, stdout=next(payloads), stderr="")

    original_unlink = Path.unlink
    original_replace = bw.os.replace

    def fail_encrypted_unlink(path, *args, **kwargs):
        if path == encrypted_path:
            raise PermissionError("forced encrypted cleanup failure")
        return original_unlink(path, *args, **kwargs)

    def fail_encrypted_replace(source, destination, *args, **kwargs):
        if Path(destination) == encrypted_path:
            raise PermissionError("forced encrypted tombstone failure")
        return original_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(bw.subprocess, "run", fake_run)
    monkeypatch.setattr(Path, "unlink", fail_encrypted_unlink)
    monkeypatch.setattr(bw.os, "replace", fail_encrypted_replace)
    first, _ = bw.fetch_bitwarden_secrets(
        access_token=access_token,
        project_id="proj-1",
        binary=fake_binary,
        cache_ttl_seconds=300,
        encrypted_cache_enabled=True,
        encrypted_cache_max_stale_seconds=300,
        home_path=home,
    )
    plaintext, plaintext_warnings = bw.fetch_bitwarden_secrets(
        access_token=access_token,
        project_id="proj-1",
        binary=fake_binary,
        cache_ttl_seconds=0,
        home_path=home,
    )
    marker_path = bw._encrypted_cache_invalidation_marker_path(home)
    assert marker_path.exists()
    # Model a fresh process: durable marker state must still block the old L2.
    bw._CACHE.clear()
    bw._CACHE_POLICY.clear()
    bw._ENCRYPTED_CACHE_INVALIDATIONS.clear()
    monkeypatch.setattr(bw.os, "replace", original_replace)
    encrypted, _ = bw.fetch_bitwarden_secrets(
        access_token=access_token,
        project_id="proj-1",
        binary=fake_binary,
        cache_ttl_seconds=300,
        encrypted_cache_enabled=True,
        encrypted_cache_max_stale_seconds=300,
        home_path=home,
    )

    assert first == {"K1": "encrypted-old"}
    assert plaintext == {"K1": "plaintext-new"}
    assert any("invalidation did not complete" in warning for warning in plaintext_warnings)
    assert encrypted == {"K1": "encrypted-new"}
    assert calls["n"] == 2
    assert encrypted_path.exists()
    assert not marker_path.exists()


def test_encrypted_ttl_zero_fetch_evicts_older_l1(
    monkeypatch,
    tmp_path,
):
    """A TTL-zero encrypted fetch must supersede an older process-local value."""
    home = tmp_path / ".hermes"
    home.mkdir()
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    payloads = iter(
        [
            _fake_bws_payload([{"key": "K1", "value": "encrypted-old"}]),
            _fake_bws_payload([{"key": "K1", "value": "encrypted-new"}]),
        ]
    )
    calls = {"n": 0}

    def fake_run(*args, **kwargs):
        calls["n"] += 1
        return mock.Mock(returncode=0, stdout=next(payloads), stderr="")

    monkeypatch.setattr(bw.subprocess, "run", fake_run)
    first, _ = bw.fetch_bitwarden_secrets(
        access_token="0.t",
        project_id="proj-1",
        binary=fake_binary,
        cache_ttl_seconds=300,
        encrypted_cache_enabled=True,
        home_path=home,
    )
    second, _ = bw.fetch_bitwarden_secrets(
        access_token="0.t",
        project_id="proj-1",
        binary=fake_binary,
        cache_ttl_seconds=0,
        encrypted_cache_enabled=True,
        home_path=home,
    )
    third, _ = bw.fetch_bitwarden_secrets(
        access_token="0.t",
        project_id="proj-1",
        binary=fake_binary,
        cache_ttl_seconds=300,
        encrypted_cache_enabled=True,
        home_path=home,
    )

    assert first == {"K1": "encrypted-old"}
    assert second == {"K1": "encrypted-new"}
    assert third == {"K1": "encrypted-new"}
    assert calls["n"] == 2


def test_failed_plaintext_invalidation_vetoes_shared_encrypted_home(
    monkeypatch,
    tmp_path,
):
    """A failed global invalidation guards every encrypted route in one home."""
    home = tmp_path / ".hermes"
    home.mkdir()
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    encrypted_path = bw._encrypted_disk_cache_path(home)
    payloads = iter(
        [
            _fake_bws_payload([{"key": "K1", "value": "project-old"}]),
            _fake_bws_payload([{"key": "K1", "value": "other-plaintext"}]),
            _fake_bws_payload([{"key": "K1", "value": "project-new"}]),
        ]
    )
    calls = {"n": 0}

    def fake_run(*args, **kwargs):
        calls["n"] += 1
        return mock.Mock(returncode=0, stdout=next(payloads), stderr="")

    original_unlink = Path.unlink
    original_replace = bw.os.replace

    def fail_encrypted_unlink(path, *args, **kwargs):
        if path == encrypted_path:
            raise PermissionError("forced encrypted cleanup failure")
        return original_unlink(path, *args, **kwargs)

    def fail_encrypted_replace(source, destination, *args, **kwargs):
        if Path(destination) == encrypted_path:
            raise PermissionError("forced encrypted tombstone failure")
        return original_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(bw.subprocess, "run", fake_run)
    first, _ = bw.fetch_bitwarden_secrets(
        access_token="0.t",
        project_id="proj-1",
        binary=fake_binary,
        cache_ttl_seconds=300,
        encrypted_cache_enabled=True,
        home_path=home,
    )
    monkeypatch.setattr(Path, "unlink", fail_encrypted_unlink)
    monkeypatch.setattr(bw.os, "replace", fail_encrypted_replace)
    other, warnings = bw.fetch_bitwarden_secrets(
        access_token="0.t",
        project_id="proj-2",
        binary=fake_binary,
        cache_ttl_seconds=0,
        home_path=home,
    )
    marker_path = bw._encrypted_cache_invalidation_marker_path(home)
    assert marker_path.exists()
    assert warnings and "invalidation did not complete" in warnings[0]
    bw._CACHE.clear()
    bw._CACHE_POLICY.clear()
    bw._ENCRYPTED_CACHE_INVALIDATIONS.clear()
    monkeypatch.setattr(bw.os, "replace", original_replace)
    recovered, _ = bw.fetch_bitwarden_secrets(
        access_token="0.t",
        project_id="proj-1",
        binary=fake_binary,
        cache_ttl_seconds=300,
        encrypted_cache_enabled=True,
        home_path=home,
    )

    assert first == {"K1": "project-old"}
    assert other == {"K1": "other-plaintext"}
    assert recovered == {"K1": "project-new"}
    assert calls["n"] == 3
    assert not marker_path.exists()


def test_encrypted_ttl_zero_failed_write_blocks_older_l2(
    monkeypatch,
    tmp_path,
):
    """A failed encrypted replacement must not expose the previous L2 value."""
    home = tmp_path / ".hermes"
    home.mkdir()
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    encrypted_path = bw._encrypted_disk_cache_path(home)
    payloads = iter(
        [
            _fake_bws_payload([{"key": "K1", "value": "encrypted-old"}]),
            _fake_bws_payload([{"key": "K1", "value": "encrypted-new"}]),
            _fake_bws_payload([{"key": "K1", "value": "encrypted-latest"}]),
        ]
    )
    calls = {"n": 0}

    def fake_run(*args, **kwargs):
        calls["n"] += 1
        return mock.Mock(returncode=0, stdout=next(payloads), stderr="")

    original_replace = bw.os.replace

    def fail_encrypted_replace(source, destination, *args, **kwargs):
        if Path(destination) == encrypted_path:
            raise PermissionError("forced encrypted replacement failure")
        return original_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(bw.subprocess, "run", fake_run)
    first, _ = bw.fetch_bitwarden_secrets(
        access_token="0.t",
        project_id="proj-1",
        binary=fake_binary,
        cache_ttl_seconds=300,
        encrypted_cache_enabled=True,
        home_path=home,
    )
    monkeypatch.setattr(bw.os, "replace", fail_encrypted_replace)
    second, second_warnings = bw.fetch_bitwarden_secrets(
        access_token="0.t",
        project_id="proj-1",
        binary=fake_binary,
        cache_ttl_seconds=0,
        encrypted_cache_enabled=True,
        home_path=home,
    )
    monkeypatch.setattr(bw.os, "replace", original_replace)
    third, _ = bw.fetch_bitwarden_secrets(
        access_token="0.t",
        project_id="proj-1",
        binary=fake_binary,
        cache_ttl_seconds=300,
        encrypted_cache_enabled=True,
        home_path=home,
    )

    assert first == {"K1": "encrypted-old"}
    assert second == {"K1": "encrypted-new"}
    assert second_warnings == [bw._encrypted_cache_invalidation_warning(home)]
    assert third == {"K1": "encrypted-latest"}
    assert calls["n"] == 3


def test_encrypted_prereplacement_failure_vetoes_intact_l2(
    monkeypatch,
    tmp_path,
):
    """Setup failure before replacement must veto a readable old cache."""
    home = tmp_path / ".hermes"
    home.mkdir()
    access_token = "0.t"
    cache_key = (bw._token_fingerprint(access_token), "proj-1", "")
    fetched_at = time.time()
    assert bw._write_encrypted_disk_cache(
        cache_key=cache_key,
        access_token=access_token,
        entry=bw._CachedFetch(
            secrets={"K1": "encrypted-old"},
            fetched_at=fetched_at,
        ),
        home_path=home,
    )
    original_chmod = bw.os.chmod

    def fail_temp_chmod(path, mode):
        if str(path).endswith(".tmp"):
            raise PermissionError("forced pre-replacement failure")
        return original_chmod(path, mode)

    monkeypatch.setattr(bw.os, "chmod", fail_temp_chmod)
    assert not bw._write_encrypted_disk_cache(
        cache_key=cache_key,
        access_token=access_token,
        entry=bw._CachedFetch(
            secrets={"K1": "encrypted-new"},
            fetched_at=fetched_at + 1,
        ),
        home_path=home,
    )
    assert bw._encrypted_cache_invalidation_marker_path(home).exists()
    bw._CACHE.clear()
    bw._CACHE_POLICY.clear()
    bw._ENCRYPTED_CACHE_INVALIDATIONS.clear()
    warnings = []
    cached = bw._read_encrypted_disk_cache(
        cache_key=cache_key,
        access_token=access_token,
        max_age_seconds=300,
        home_path=home,
        transition_warnings=warnings,
    )
    assert cached is None
    assert warnings == [bw._encrypted_cache_invalidation_warning(home)]


def test_encrypted_write_failure_without_predecessor_keeps_l1(
    monkeypatch,
    tmp_path,
):
    """A first disk-write failure must not force a needless live refetch."""
    home = tmp_path / ".hermes"
    home.mkdir()
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    calls = {"n": 0}

    def fake_run(*args, **kwargs):
        calls["n"] += 1
        return mock.Mock(
            returncode=0,
            stdout=_fake_bws_payload([{"key": "K1", "value": "live"}]),
            stderr="",
        )

    def fail_mkstemp(*args, **kwargs):
        raise PermissionError("forced cache write failure")

    monkeypatch.setattr(bw.subprocess, "run", fake_run)
    monkeypatch.setattr(bw.tempfile, "mkstemp", fail_mkstemp)
    first, first_warnings = bw.fetch_bitwarden_secrets(
        access_token="0.t",
        project_id="proj-1",
        binary=fake_binary,
        cache_ttl_seconds=300,
        encrypted_cache_enabled=True,
        home_path=home,
    )
    second, second_warnings = bw.fetch_bitwarden_secrets(
        access_token="0.t",
        project_id="proj-1",
        binary=fake_binary,
        cache_ttl_seconds=300,
        encrypted_cache_enabled=True,
        home_path=home,
    )

    assert first == {"K1": "live"}
    assert second == first
    assert first_warnings == [bw._encrypted_cache_write_warning(home)]
    assert second_warnings == []
    assert calls["n"] == 1


def test_encrypted_write_keeps_veto_when_marker_cleanup_fails(
    monkeypatch,
    tmp_path,
):
    """A new ciphertext must remain vetoed if an old marker cannot be removed."""
    home = tmp_path / ".hermes"
    home.mkdir()
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    access_token = "0.t"
    cache_key = (bw._token_fingerprint(access_token), "proj-1", "")
    assert bw._write_encrypted_disk_cache(
        cache_key=cache_key,
        access_token=access_token,
        entry=bw._CachedFetch(secrets={"K1": "old"}, fetched_at=time.time()),
        home_path=home,
    )
    bw._mark_encrypted_cache_invalidated(cache_key, home)
    marker_paths = set(bw._encrypted_cache_invalidation_marker_paths(home))
    original_unlink = Path.unlink

    def fail_marker_unlink(path, *args, **kwargs):
        if path in marker_paths:
            raise PermissionError("forced marker cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_marker_unlink)
    payload = _fake_bws_payload([{"key": "K1", "value": "new"}])
    monkeypatch.setattr(
        bw.subprocess,
        "run",
        lambda *args, **kwargs: mock.Mock(
            returncode=0,
            stdout=payload,
            stderr="",
        ),
    )
    fresh, warnings = bw.fetch_bitwarden_secrets(
        access_token=access_token,
        project_id="proj-1",
        binary=fake_binary,
        cache_ttl_seconds=300,
        encrypted_cache_enabled=True,
        home_path=home,
    )

    assert fresh == {"K1": "new"}
    assert warnings == [bw._encrypted_cache_invalidation_warning(home)]
    assert any(path.exists() for path in marker_paths)
    bw._CACHE.clear()
    bw._CACHE_POLICY.clear()
    bw._ENCRYPTED_CACHE_INVALIDATIONS.clear()
    monkeypatch.setattr(Path, "unlink", original_unlink)
    assert (
        bw._read_encrypted_disk_cache(
            cache_key=cache_key,
            access_token=access_token,
            max_age_seconds=300,
            home_path=home,
            transition_warnings=[],
        )
        is None
    )


def test_encrypted_write_removes_plaintext_when_marker_cleanup_fails(
    monkeypatch,
    tmp_path,
):
    """A marker failure must not leave same-route plaintext secrets on disk."""
    home = tmp_path / ".hermes"
    home.mkdir()
    access_token = "0.t"
    cache_key = (bw._token_fingerprint(access_token), "proj-1", "")
    fetched_at = time.time()
    assert bw._write_encrypted_disk_cache(
        cache_key=cache_key,
        access_token=access_token,
        entry=bw._CachedFetch(
            secrets={"K1": "encrypted-predecessor"},
            fetched_at=fetched_at,
        ),
        home_path=home,
    )
    bw._mark_encrypted_cache_invalidated(cache_key, home)
    marker_paths = set(bw._encrypted_cache_invalidation_marker_paths(home))
    plaintext_path = bw._disk_cache_path(home)
    bw._DISK_CACHE.write(
        cache_key,
        bw._CachedFetch(
            secrets={"PLAINTEXT_SECRET": "must-be-removed"},
            fetched_at=fetched_at + 1,
        ),
        300,
        home,
    )
    original_unlink = Path.unlink

    def fail_marker_unlink(path, *args, **kwargs):
        if path in marker_paths:
            raise PermissionError("forced marker cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_marker_unlink)
    warnings = []
    transition_state = []

    assert not bw._write_encrypted_disk_cache(
        cache_key=cache_key,
        access_token=access_token,
        entry=bw._CachedFetch(
            secrets={"K1": "encrypted-new"},
            fetched_at=fetched_at + 2,
        ),
        home_path=home,
        transition_warnings=warnings,
        transition_pending_out=transition_state,
    )
    assert not plaintext_path.exists()
    assert any(path.exists() for path in marker_paths)
    assert transition_state == [True]
    assert warnings == [bw._encrypted_cache_invalidation_warning(home)]
    bw._CACHE.clear()
    bw._CACHE_POLICY.clear()
    bw._ENCRYPTED_CACHE_INVALIDATIONS.clear()
    read_warnings = []
    assert (
        bw._read_encrypted_disk_cache(
            cache_key=cache_key,
            access_token=access_token,
            max_age_seconds=300,
            home_path=home,
            transition_warnings=read_warnings,
        )
        is None
    )
    assert read_warnings == [bw._encrypted_cache_invalidation_warning(home)]


def test_encrypted_replacement_success_keeps_l1_fresh(
    monkeypatch,
    tmp_path,
):
    """Replacing an existing ciphertext must not mark the new L1 pending."""
    home = tmp_path / ".hermes"
    home.mkdir()
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    payloads = iter(
        [
            _fake_bws_payload([{"key": "K1", "value": "old"}]),
            _fake_bws_payload([{"key": "K1", "value": "new"}]),
            _fake_bws_payload([{"key": "K1", "value": "unexpected"}]),
        ]
    )
    calls = {"n": 0}

    def fake_run(*args, **kwargs):
        calls["n"] += 1
        return mock.Mock(returncode=0, stdout=next(payloads), stderr="")

    monkeypatch.setattr(bw.subprocess, "run", fake_run)
    first, _ = bw.fetch_bitwarden_secrets(
        access_token="0.t",
        project_id="proj-1",
        binary=fake_binary,
        cache_ttl_seconds=0,
        encrypted_cache_enabled=True,
        home_path=home,
    )
    bw._CACHE.clear()
    bw._CACHE_POLICY.clear()
    assert bw._write_encrypted_disk_cache(
        cache_key=(bw._token_fingerprint("0.t"), "proj-1", ""),
        access_token="0.t",
        entry=bw._CachedFetch(
            secrets={"K1": "old"},
            fetched_at=time.time() - 3600,
        ),
        home_path=home,
    )
    second, _ = bw.fetch_bitwarden_secrets(
        access_token="0.t",
        project_id="proj-1",
        binary=fake_binary,
        cache_ttl_seconds=300,
        encrypted_cache_enabled=True,
        home_path=home,
    )
    third, _ = bw.fetch_bitwarden_secrets(
        access_token="0.t",
        project_id="proj-1",
        binary=fake_binary,
        cache_ttl_seconds=300,
        encrypted_cache_enabled=True,
        home_path=home,
    )

    assert first == {"K1": "old"}
    assert second == {"K1": "new"}
    assert third == second
    assert calls["n"] == 2


def test_encrypted_l1_is_bound_to_hermes_home(monkeypatch, tmp_path):
    """An encrypted L1 entry from another home must not skip storage there."""
    home_a = tmp_path / "home-a"
    home_b = tmp_path / "home-b"
    home_a.mkdir()
    home_b.mkdir()
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    access_token = "0.t"
    cache_key = (bw._token_fingerprint(access_token), "proj-1", "")
    plaintext_b = bw._disk_cache_path(home_b)
    bw._DISK_CACHE.write(
        cache_key,
        bw._CachedFetch(secrets={"K1": "home-b-old"}, fetched_at=time.time()),
        300,
        home_b,
    )
    payloads = iter(
        [
            _fake_bws_payload([{"key": "K1", "value": "home-a"}]),
            _fake_bws_payload([{"key": "K1", "value": "home-b"}]),
        ]
    )
    calls = {"n": 0}

    def fake_run(*args, **kwargs):
        calls["n"] += 1
        return mock.Mock(returncode=0, stdout=next(payloads), stderr="")

    monkeypatch.setattr(bw.subprocess, "run", fake_run)
    first, _ = bw.fetch_bitwarden_secrets(
        access_token=access_token,
        project_id="proj-1",
        binary=fake_binary,
        cache_ttl_seconds=300,
        encrypted_cache_enabled=True,
        home_path=home_a,
    )
    second, _ = bw.fetch_bitwarden_secrets(
        access_token=access_token,
        project_id="proj-1",
        binary=fake_binary,
        cache_ttl_seconds=300,
        encrypted_cache_enabled=True,
        home_path=home_b,
    )

    assert first == {"K1": "home-a"}
    assert second == {"K1": "home-b"}
    assert calls["n"] == 2
    assert bw._encrypted_disk_cache_path(home_b).exists()
    assert not plaintext_b.exists()


def test_bitwarden_source_l1_is_bound_to_hermes_home(monkeypatch, tmp_path):
    """The adapter must not reuse encrypted L1 data across Hermes homes."""
    home_a = tmp_path / "home-a"
    home_b = tmp_path / "home-b"
    home_a.mkdir()
    home_b.mkdir()
    access_token = "0.t"
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    payloads = iter(
        [
            _fake_bws_payload([{"key": "K1", "value": "home-a"}]),
            _fake_bws_payload([{"key": "K1", "value": "home-b"}]),
        ]
    )
    calls = {"n": 0}

    def fake_run(*args, **kwargs):
        calls["n"] += 1
        return mock.Mock(returncode=0, stdout=next(payloads), stderr="")

    monkeypatch.setattr(bw.subprocess, "run", fake_run)
    monkeypatch.setattr(bw, "find_bws", lambda **kwargs: fake_binary)
    monkeypatch.setenv("BWS_ACCESS_TOKEN", access_token)
    cfg = {
        "project_id": "proj-1",
        "cache_ttl_seconds": 300,
        "encrypted_cache": {"enabled": True},
    }
    first = bw.BitwardenSource().fetch(cfg, home_a)
    second = bw.BitwardenSource().fetch(cfg, home_b)

    assert first.secrets == {"K1": "home-a"}
    assert second.secrets == {"K1": "home-b"}
    assert calls["n"] == 2
    assert bw._encrypted_disk_cache_path(home_b).exists()


def test_bitwarden_source_inherited_server_url_is_bound_to_cache_identity(
    monkeypatch,
    tmp_path,
):
    """The registry adapter must resolve inherited endpoint changes too."""
    home = tmp_path / ".hermes"
    home.mkdir()
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    access_token = "0.t"
    source_env = {
        "BWS_ACCESS_TOKEN": access_token,
        "BWS_SERVER_URL": "https://vault.bitwarden.eu",
    }
    payloads = iter(
        [
            _fake_bws_payload([{"key": "REGION", "value": "eu"}]),
            _fake_bws_payload([{"key": "REGION", "value": "us"}]),
        ]
    )
    seen_urls = []

    def fake_run(cmd, **kwargs):
        seen_urls.append(kwargs["env"].get("BWS_SERVER_URL"))
        return mock.Mock(returncode=0, stdout=next(payloads), stderr="")

    monkeypatch.setattr(bw, "get_source_environment", lambda: source_env)
    monkeypatch.setattr(bw, "find_bws", lambda **kwargs: fake_binary)
    monkeypatch.setattr(bw.subprocess, "run", fake_run)
    cfg = {
        "project_id": "proj-1",
        "cache_ttl_seconds": 300,
        "encrypted_cache": {"enabled": True},
    }
    first = bw.BitwardenSource().fetch(cfg, home)
    bw._CACHE.clear()
    bw._CACHE_POLICY.clear()
    source_env["BWS_SERVER_URL"] = "https://vault.bitwarden.com"
    second = bw.BitwardenSource().fetch(cfg, home)

    assert first.secrets == {"REGION": "eu"}
    assert second.secrets == {"REGION": "us"}
    assert seen_urls == [
        "https://vault.bitwarden.eu",
        "https://vault.bitwarden.com",
    ]


def test_bitwarden_source_encrypted_ttl_zero_evicts_older_l1(
    monkeypatch,
    tmp_path,
):
    """The registry adapter must not restore a pre-TTL-zero L1 value."""
    home = tmp_path / ".hermes"
    home.mkdir()
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    access_token = "0.t"
    payloads = iter(
        [
            _fake_bws_payload([{"key": "K1", "value": "encrypted-old"}]),
            _fake_bws_payload([{"key": "K1", "value": "encrypted-new"}]),
        ]
    )
    calls = {"n": 0}

    def fake_run(*args, **kwargs):
        calls["n"] += 1
        return mock.Mock(returncode=0, stdout=next(payloads), stderr="")

    monkeypatch.setattr(bw.subprocess, "run", fake_run)
    monkeypatch.setattr(bw, "find_bws", lambda **kwargs: fake_binary)
    monkeypatch.setenv("BWS_ACCESS_TOKEN", access_token)
    cfg = {
        "project_id": "proj-1",
        "cache_ttl_seconds": 300,
        "encrypted_cache": {"enabled": True},
    }
    first = bw.BitwardenSource().fetch(cfg, home)
    zero_ttl_cfg = {
        **cfg,
        "cache_ttl_seconds": 0,
    }
    second = bw.BitwardenSource().fetch(zero_ttl_cfg, home)
    third = bw.BitwardenSource().fetch(cfg, home)

    assert first.secrets == {"K1": "encrypted-old"}
    assert second.secrets == {"K1": "encrypted-new"}
    assert third.secrets == {"K1": "encrypted-new"}
    assert first.error is None
    assert second.error is None
    assert third.error is None
    assert calls["n"] == 2


def test_bitwarden_source_encrypted_ttl_zero_failed_write_blocks_older_l2(
    monkeypatch,
    tmp_path,
):
    """The adapter must not restore old L2 after an encrypted write failure."""
    home = tmp_path / ".hermes"
    home.mkdir()
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    access_token = "0.t"
    encrypted_path = bw._encrypted_disk_cache_path(home)
    payloads = iter(
        [
            _fake_bws_payload([{"key": "K1", "value": "encrypted-old"}]),
            _fake_bws_payload([{"key": "K1", "value": "encrypted-new"}]),
            _fake_bws_payload([{"key": "K1", "value": "encrypted-latest"}]),
        ]
    )
    calls = {"n": 0}

    def fake_run(*args, **kwargs):
        calls["n"] += 1
        return mock.Mock(returncode=0, stdout=next(payloads), stderr="")

    original_replace = bw.os.replace

    def fail_encrypted_replace(source, destination, *args, **kwargs):
        if Path(destination) == encrypted_path:
            raise PermissionError("forced encrypted replacement failure")
        return original_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(bw.subprocess, "run", fake_run)
    monkeypatch.setattr(bw, "find_bws", lambda **kwargs: fake_binary)
    monkeypatch.setenv("BWS_ACCESS_TOKEN", access_token)
    cfg = {
        "project_id": "proj-1",
        "cache_ttl_seconds": 300,
        "encrypted_cache": {"enabled": True},
    }
    first = bw.BitwardenSource().fetch(cfg, home)
    monkeypatch.setattr(bw.os, "replace", fail_encrypted_replace)
    second = bw.BitwardenSource().fetch(
        {**cfg, "cache_ttl_seconds": 0},
        home,
    )
    monkeypatch.setattr(bw.os, "replace", original_replace)
    third = bw.BitwardenSource().fetch(cfg, home)

    assert first.secrets == {"K1": "encrypted-old"}
    assert second.secrets == {"K1": "encrypted-new"}
    assert second.warnings
    assert third.secrets == {"K1": "encrypted-latest"}
    assert first.error is None
    assert second.error is None
    assert third.error is None
    assert calls["n"] == 3


def test_encrypted_cache_key_uses_memory_hard_scrypt():
    """Low-entropy bootstrap tokens must not use a single-pass fast KDF."""
    token = "short-token"
    salt = bytes(range(16))
    expected = Scrypt(
        salt=salt,
        length=32,
        n=bw._ENCRYPTED_CACHE_SCRYPT_N,
        r=bw._ENCRYPTED_CACHE_SCRYPT_R,
        p=bw._ENCRYPTED_CACHE_SCRYPT_P,
    ).derive(token.encode("utf-8"))

    assert bw._derive_encrypted_cache_key(token, salt) == expected


def test_encrypted_cache_wrong_token_rejected_after_scrypt(tmp_path):
    """V2 has no token verifier that can reject guesses before the KDF."""
    home = tmp_path / ".hermes"
    home.mkdir()
    entry = bw._CachedFetch(secrets={"K1": "cached"}, fetched_at=time.time())
    good_token = "short-token"
    assert bw._write_encrypted_disk_cache(
        cache_key=(bw._token_fingerprint(good_token), "proj-1", ""),
        access_token=good_token,
        entry=entry,
        home_path=home,
    )

    with mock.patch.object(
        bw,
        "_derive_encrypted_cache_key",
        wraps=bw._derive_encrypted_cache_key,
    ) as derive:
        cached = bw._read_encrypted_disk_cache(
            cache_key=(bw._token_fingerprint("wrong-token"), "proj-1", ""),
            access_token="wrong-token",
            max_age_seconds=300,
            home_path=home,
        )

    assert cached is None
    derive.assert_called_once_with("wrong-token", mock.ANY)


def test_encrypted_cache_migrates_v1_hkdf_payload(tmp_path):
    """A valid v1 cache is re-encrypted with scrypt before it is served."""
    home = tmp_path / ".hermes"
    home.mkdir()
    access_token = "short-token"
    cache_key = (bw._token_fingerprint(access_token), "proj-1", "")
    serialized_key = bw._cache_key_str(cache_key)
    salt = bytes(range(16))
    nonce = bytes(range(12))
    plaintext = json.dumps(
        {"secrets": {"K1": "cached"}, "fetched_at": time.time()},
        separators=(",", ":"),
    ).encode("utf-8")
    legacy_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        info=b"hermes-bws-encrypted-cache-v1",
    ).derive(access_token.encode("utf-8"))
    ciphertext = AESGCM(legacy_key).encrypt(
        nonce, plaintext, serialized_key.encode("utf-8")
    )
    path = bw._encrypted_disk_cache_path(home)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "version": bw._ENCRYPTED_CACHE_LEGACY_VERSION,
        "key": serialized_key,
        "salt": bw._b64e(salt),
        "nonce": bw._b64e(nonce),
        "ciphertext": bw._b64e(ciphertext),
    }))

    entry = bw._read_encrypted_disk_cache(
        cache_key=cache_key,
        access_token=access_token,
        max_age_seconds=300,
        home_path=home,
    )

    assert entry is not None
    assert entry.secrets == {"K1": "cached"}
    assert json.loads(path.read_text())["version"] == bw._ENCRYPTED_CACHE_VERSION
    reread = bw._read_encrypted_disk_cache(
        cache_key=cache_key,
        access_token=access_token,
        max_age_seconds=300,
        home_path=home,
    )
    assert reread is not None
    assert reread.secrets == {"K1": "cached"}


def test_encrypted_cache_stale_v1_is_migrated_without_serving(
    tmp_path,
):
    """A valid stale v1 entry is upgraded but never returned as stale data."""
    home = tmp_path / ".hermes"
    home.mkdir()
    access_token = "short-token"
    cache_key = (bw._token_fingerprint(access_token), "proj-1", "")
    serialized_key = bw._cache_key_str(cache_key)
    salt = bytes(range(16))
    nonce = bytes(range(12))
    plaintext = json.dumps(
        {
            "secrets": {"K1": "stale-secret"},
            "fetched_at": time.time() - 3600,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    legacy_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        info=b"hermes-bws-encrypted-cache-v1",
    ).derive(access_token.encode("utf-8"))
    ciphertext = AESGCM(legacy_key).encrypt(
        nonce, plaintext, serialized_key.encode("utf-8")
    )
    path = bw._encrypted_disk_cache_path(home)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "version": bw._ENCRYPTED_CACHE_LEGACY_VERSION,
        "key": serialized_key,
        "salt": bw._b64e(salt),
        "nonce": bw._b64e(nonce),
        "ciphertext": bw._b64e(ciphertext),
    }))

    entry = bw._read_encrypted_disk_cache(
        cache_key=cache_key,
        access_token=access_token,
        max_age_seconds=300,
        home_path=home,
    )

    assert entry is None
    assert json.loads(path.read_text())["version"] == bw._ENCRYPTED_CACHE_VERSION


def test_encrypted_cache_future_v1_is_migrated_without_serving(
    tmp_path,
):
    """A future-dated valid v1 entry is upgraded but never returned."""
    home = tmp_path / ".hermes"
    home.mkdir()
    access_token = "short-token"
    cache_key = (bw._token_fingerprint(access_token), "proj-1", "")
    serialized_key = bw._cache_key_str(cache_key)
    salt = bytes(range(16))
    nonce = bytes(range(12))
    plaintext = json.dumps(
        {
            "secrets": {"K1": "future-secret"},
            "fetched_at": time.time() + 3600,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    legacy_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        info=b"hermes-bws-encrypted-cache-v1",
    ).derive(access_token.encode("utf-8"))
    ciphertext = AESGCM(legacy_key).encrypt(
        nonce, plaintext, serialized_key.encode("utf-8")
    )
    path = bw._encrypted_disk_cache_path(home)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "version": bw._ENCRYPTED_CACHE_LEGACY_VERSION,
        "key": serialized_key,
        "salt": bw._b64e(salt),
        "nonce": bw._b64e(nonce),
        "ciphertext": bw._b64e(ciphertext),
    }))

    entry = bw._read_encrypted_disk_cache(
        cache_key=cache_key,
        access_token=access_token,
        max_age_seconds=300,
        home_path=home,
    )

    assert entry is None
    assert json.loads(path.read_text())["version"] == bw._ENCRYPTED_CACHE_VERSION


def test_encrypted_cache_stale_v2_cleans_plaintext_predecessor(
    tmp_path,
):
    """A stale v2 entry still completes safe plaintext cleanup."""
    home = tmp_path / ".hermes"
    home.mkdir()
    access_token = "short-token"
    cache_key = (bw._token_fingerprint(access_token), "proj-1", "")
    fetched_at = time.time() - 3600
    encrypted_path = bw._encrypted_disk_cache_path(home)
    plaintext_path = bw._disk_cache_path(home)
    assert bw._write_encrypted_disk_cache(
        cache_key=cache_key,
        access_token=access_token,
        entry=bw._CachedFetch(
            secrets={"K1": "stale-encrypted"},
            fetched_at=fetched_at,
        ),
        home_path=home,
    )
    bw._DISK_CACHE.write(
        cache_key,
        bw._CachedFetch(secrets={"K1": "older-plaintext"}, fetched_at=fetched_at),
        300,
        home,
    )
    warnings = []

    entry = bw._read_encrypted_disk_cache(
        cache_key=cache_key,
        access_token=access_token,
        max_age_seconds=300,
        home_path=home,
        transition_warnings=warnings,
    )

    assert entry is None
    assert encrypted_path.exists()
    assert not plaintext_path.exists()
    assert warnings == []


def test_encrypted_cache_stale_v2_cleanup_failure_warns(
    monkeypatch,
    tmp_path,
):
    """A stale v2 cleanup failure remains visible to the caller."""
    home = tmp_path / ".hermes"
    home.mkdir()
    access_token = "short-token"
    cache_key = (bw._token_fingerprint(access_token), "proj-1", "")
    fetched_at = time.time() - 3600
    plaintext_path = bw._disk_cache_path(home)
    assert bw._write_encrypted_disk_cache(
        cache_key=cache_key,
        access_token=access_token,
        entry=bw._CachedFetch(
            secrets={"K1": "stale-encrypted"},
            fetched_at=fetched_at,
        ),
        home_path=home,
    )
    bw._DISK_CACHE.write(
        cache_key,
        bw._CachedFetch(secrets={"K1": "older-plaintext"}, fetched_at=fetched_at),
        300,
        home,
    )
    original_unlink = Path.unlink

    def fail_plaintext_unlink(path, *args, **kwargs):
        if path == plaintext_path:
            raise PermissionError("forced plaintext cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_plaintext_unlink)
    warnings = []

    entry = bw._read_encrypted_disk_cache(
        cache_key=cache_key,
        access_token=access_token,
        max_age_seconds=300,
        home_path=home,
        transition_warnings=warnings,
    )

    assert entry is None
    assert plaintext_path.exists()
    assert warnings
    assert str(plaintext_path) in warnings[0]


def test_encrypted_cache_newer_plaintext_wins_without_cleanup_warning(
    tmp_path,
):
    """A newer plaintext predecessor is preserved without a false failure warning."""
    home = tmp_path / ".hermes"
    home.mkdir()
    access_token = "short-token"
    cache_key = (bw._token_fingerprint(access_token), "proj-1", "")
    encrypted_fetched_at = time.time() - 60
    plaintext_path = bw._disk_cache_path(home)
    assert bw._write_encrypted_disk_cache(
        cache_key=cache_key,
        access_token=access_token,
        entry=bw._CachedFetch(
            secrets={"K1": "encrypted"},
            fetched_at=encrypted_fetched_at,
        ),
        home_path=home,
    )
    bw._DISK_CACHE.write(
        cache_key,
        bw._CachedFetch(
            secrets={"K1": "newer-plaintext"},
            fetched_at=encrypted_fetched_at + 30,
        ),
        300,
        home,
    )
    warnings = []

    entry = bw._read_encrypted_disk_cache(
        cache_key=cache_key,
        access_token=access_token,
        max_age_seconds=300,
        home_path=home,
        transition_warnings=warnings,
    )

    assert entry is None
    assert plaintext_path.exists()
    assert warnings == [bw._encrypted_cache_newer_plaintext_warning(home)]


def test_apply_bitwarden_secrets_missing_binary_reports_cleanup_warning(
    monkeypatch,
    tmp_path,
):
    """The public apply path preserves discard-only transition diagnostics."""
    home = tmp_path / ".hermes"
    home.mkdir()
    access_token = "short-token"
    project_id = "proj-1"
    cache_key = (
        bw._token_fingerprint(access_token),
        project_id,
        "https://vault.bitwarden.eu",
    )
    encrypted_path = bw._encrypted_disk_cache_path(home)
    plaintext_path = bw._disk_cache_path(home)
    assert bw._write_encrypted_disk_cache(
        cache_key=cache_key,
        access_token=access_token,
        entry=bw._CachedFetch(
            secrets={"K1": "encrypted"},
            fetched_at=time.time(),
        ),
        home_path=home,
    )
    bw._DISK_CACHE.write(
        cache_key,
        bw._CachedFetch(secrets={"K1": "plaintext"}, fetched_at=time.time()),
        300,
        home,
    )
    original_unlink = Path.unlink

    def fail_plaintext_unlink(path, *args, **kwargs):
        if path == plaintext_path:
            raise PermissionError("forced plaintext cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_plaintext_unlink)
    monkeypatch.setattr(bw, "find_bws", lambda **kwargs: None)
    monkeypatch.setenv("BWS_ACCESS_TOKEN", access_token)

    result = bw.apply_bitwarden_secrets(
        enabled=True,
        project_id=project_id,
        auto_install=False,
        cache_ttl_seconds=0,
        server_url="  https://vault.bitwarden.eu  ",
        encrypted_cache_enabled=True,
        encrypted_cache_max_stale_seconds=300,
        home_path=home,
    )

    assert result.secrets == {}
    assert result.error is not None
    assert "bws binary not available" in result.error
    assert str(plaintext_path) in result.error
    assert result.binary_path is None
    assert encrypted_path.exists()
    assert plaintext_path.exists()


def test_encrypted_cache_v1_migration_write_failure_is_cache_miss(
    monkeypatch,
    tmp_path,
):
    """Never serve a weak v1 entry unless its atomic v2 rewrite succeeds."""
    home = tmp_path / ".hermes"
    home.mkdir()
    access_token = "short-token"
    cache_key = (bw._token_fingerprint(access_token), "proj-1", "")
    serialized_key = bw._cache_key_str(cache_key)
    salt = bytes(range(16))
    nonce = bytes(range(12))
    plaintext = json.dumps(
        {"secrets": {"K1": "cached"}, "fetched_at": time.time()},
        separators=(",", ":"),
    ).encode("utf-8")
    legacy_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        info=b"hermes-bws-encrypted-cache-v1",
    ).derive(access_token.encode("utf-8"))
    ciphertext = AESGCM(legacy_key).encrypt(
        nonce, plaintext, serialized_key.encode("utf-8")
    )
    path = bw._encrypted_disk_cache_path(home)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "version": bw._ENCRYPTED_CACHE_LEGACY_VERSION,
        "key": serialized_key,
        "salt": bw._b64e(salt),
        "nonce": bw._b64e(nonce),
        "ciphertext": bw._b64e(ciphertext),
    }))
    monkeypatch.setattr(bw, "_write_encrypted_disk_cache", lambda **_: False)

    entry = bw._read_encrypted_disk_cache(
        cache_key=cache_key,
        access_token=access_token,
        max_age_seconds=300,
        home_path=home,
    )

    assert entry is None
    assert not path.exists()


def test_mismatched_v1_cache_is_removed_without_migration(
    tmp_path,
):
    """A v1 artifact for another route must not remain as an offline oracle."""
    home = tmp_path / ".hermes"
    home.mkdir()
    access_token = "short-token"
    cache_key = (bw._token_fingerprint(access_token), "proj-1", "")
    path = bw._encrypted_disk_cache_path(home)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "version": bw._ENCRYPTED_CACHE_LEGACY_VERSION,
                "key": "deadbeefdeadbeef|other-project|https://vault.bitwarden.eu",
                "salt": "",
                "nonce": "",
                "ciphertext": "",
            }
        )
    )

    entry = bw._read_encrypted_disk_cache(
        cache_key=cache_key,
        access_token=access_token,
        max_age_seconds=300,
        home_path=home,
    )

    assert entry is None
    assert not path.exists()


def test_encrypted_cache_incomplete_v1_cleanup_is_not_stale_fallback(
    monkeypatch,
    tmp_path,
):
    """A retained v2 stays unavailable until its plaintext predecessor is gone."""
    home = tmp_path / ".hermes"
    home.mkdir()
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    access_token = "short-token"
    cache_key = (bw._token_fingerprint(access_token), "proj-1", "")
    serialized_key = bw._cache_key_str(cache_key)
    fetched_at = time.time()
    salt = bytes(range(16))
    nonce = bytes(range(12))
    plaintext = json.dumps(
        {"secrets": {"K1": "legacy-v1"}, "fetched_at": fetched_at},
        separators=(",", ":"),
    ).encode("utf-8")
    legacy_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        info=b"hermes-bws-encrypted-cache-v1",
    ).derive(access_token.encode("utf-8"))
    ciphertext = AESGCM(legacy_key).encrypt(
        nonce, plaintext, serialized_key.encode("utf-8")
    )
    encrypted_path = bw._encrypted_disk_cache_path(home)
    encrypted_path.parent.mkdir(parents=True)
    encrypted_path.write_text(json.dumps({
        "version": bw._ENCRYPTED_CACHE_LEGACY_VERSION,
        "key": serialized_key,
        "salt": bw._b64e(salt),
        "nonce": bw._b64e(nonce),
        "ciphertext": bw._b64e(ciphertext),
    }))
    plaintext_path = bw._disk_cache_path(home)
    bw._DISK_CACHE.write(
        cache_key,
        bw._CachedFetch(secrets={"K1": "plaintext"}, fetched_at=fetched_at),
        300,
        home,
    )
    cleanup_allowed = {"value": False}
    original_unlink = Path.unlink

    def fail_plaintext_unlink(path, *args, **kwargs):
        if path == plaintext_path and not cleanup_allowed["value"]:
            raise PermissionError("forced plaintext cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_plaintext_unlink)
    calls = {"n": 0}

    def fail_run(*args, **kwargs):
        calls["n"] += 1
        return mock.Mock(
            returncode=1,
            stdout="",
            stderr="Error: network is unreachable",
        )

    monkeypatch.setattr(bw.subprocess, "run", fail_run)
    bw._CACHE.clear()

    with pytest.raises(RuntimeError) as exc_info:
        bw.fetch_bitwarden_secrets(
            access_token=access_token,
            project_id="proj-1",
            binary=fake_binary,
            cache_ttl_seconds=300,
            encrypted_cache_enabled=True,
            encrypted_cache_max_stale_seconds=300,
            home_path=home,
        )

    assert json.loads(encrypted_path.read_text())["version"] == (
        bw._ENCRYPTED_CACHE_VERSION
    )
    assert plaintext_path.exists()
    assert str(plaintext_path) in str(exc_info.value)
    assert "falling back to stale ENCRYPTED disk cache" not in str(exc_info.value)
    assert calls["n"] == 1

    cleanup_allowed["value"] = True
    recovered, warnings = bw.fetch_bitwarden_secrets(
        access_token=access_token,
        project_id="proj-1",
        binary=fake_binary,
        cache_ttl_seconds=300,
        encrypted_cache_enabled=True,
        encrypted_cache_max_stale_seconds=300,
        home_path=home,
    )

    assert recovered == {"K1": "legacy-v1"}
    assert warnings == []
    assert encrypted_path.exists()
    assert not plaintext_path.exists()
    assert calls["n"] == 1




def test_encrypted_cache_missing_binary_reports_blocked_plaintext_cleanup(
    monkeypatch,
    tmp_path,
):
    """Missing-binary errors retain the exact encrypted-transition warning."""
    home = tmp_path / ".hermes"
    home.mkdir()
    access_token = "short-token"
    cache_key = (bw._token_fingerprint(access_token), "proj-1", "")
    fetched_at = time.time()
    encrypted_path = bw._encrypted_disk_cache_path(home)
    plaintext_path = bw._disk_cache_path(home)
    assert bw._write_encrypted_disk_cache(
        cache_key=cache_key,
        access_token=access_token,
        entry=bw._CachedFetch(
            secrets={"K1": "encrypted"},
            fetched_at=fetched_at,
        ),
        home_path=home,
    )
    bw._DISK_CACHE.write(
        cache_key,
        bw._CachedFetch(secrets={"K1": "plaintext"}, fetched_at=fetched_at),
        300,
        home,
    )
    original_unlink = Path.unlink

    def fail_plaintext_unlink(path, *args, **kwargs):
        if path == plaintext_path:
            raise PermissionError("forced plaintext cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_plaintext_unlink)
    find_calls = []

    def missing_bws(*, install_if_missing=False):
        find_calls.append(install_if_missing)
        return None

    monkeypatch.setattr(bw, "find_bws", missing_bws)
    monkeypatch.setattr(
        bw,
        "_run_bws_list",
        lambda *args, **kwargs: pytest.fail("missing binary must prevent live fetch"),
    )

    with pytest.raises(RuntimeError) as exc_info:
        bw.fetch_bitwarden_secrets(
            access_token=access_token,
            project_id="proj-1",
            cache_ttl_seconds=300,
            encrypted_cache_enabled=True,
            home_path=home,
        )

    error = str(exc_info.value)
    assert "bws binary not available" in error
    assert str(plaintext_path) in error
    assert encrypted_path.exists()
    assert plaintext_path.exists()
    assert find_calls == [True]


def test_bitwarden_source_missing_binary_reports_blocked_plaintext_cleanup(
    monkeypatch,
    tmp_path,
):
    """The source adapter preserves cleanup detail and BINARY_MISSING."""
    home = tmp_path / ".hermes"
    home.mkdir()
    access_token = "short-token"
    project_id = "proj-1"
    cache_key = (bw._token_fingerprint(access_token), project_id, "")
    fetched_at = time.time()
    encrypted_path = bw._encrypted_disk_cache_path(home)
    plaintext_path = bw._disk_cache_path(home)
    assert bw._write_encrypted_disk_cache(
        cache_key=cache_key,
        access_token=access_token,
        entry=bw._CachedFetch(
            secrets={"K1": "encrypted"},
            fetched_at=fetched_at,
        ),
        home_path=home,
    )
    bw._DISK_CACHE.write(
        cache_key,
        bw._CachedFetch(secrets={"K1": "plaintext"}, fetched_at=fetched_at),
        300,
        home,
    )
    original_unlink = Path.unlink

    def fail_plaintext_unlink(path, *args, **kwargs):
        if path == plaintext_path:
            raise PermissionError("forced plaintext cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_plaintext_unlink)
    find_calls = []

    def missing_bws(*, install_if_missing=False):
        find_calls.append(install_if_missing)
        return None

    monkeypatch.setattr(bw, "find_bws", missing_bws)
    monkeypatch.setenv("BWS_ACCESS_TOKEN", access_token)

    result = bw.BitwardenSource().fetch(
        {
            "project_id": project_id,
            "auto_install": False,
            "cache_ttl_seconds": 300,
            "encrypted_cache": {"enabled": True},
        },
        home,
    )

    assert result.secrets == {}
    assert result.error_kind is bw.ErrorKind.BINARY_MISSING
    assert result.error is not None
    assert "bws binary not available" in result.error
    assert str(plaintext_path) in result.error
    assert result.binary_path is None
    assert encrypted_path.exists()
    assert plaintext_path.exists()
    assert find_calls == [False]


@pytest.mark.parametrize(
    "cache_ttl_seconds,max_stale_seconds,entry_age_seconds",
    [
        (0, 300, 30),
        (60, 300, 120),
    ],
)
def test_encrypted_cache_missing_binary_reports_stale_transition_cleanup(
    monkeypatch,
    tmp_path,
    cache_ttl_seconds,
    max_stale_seconds,
    entry_age_seconds,
):
    """Missing-binary inspection includes the encrypted stale window."""
    home = tmp_path / ".hermes"
    home.mkdir()
    access_token = "short-token"
    project_id = "proj-1"
    cache_key = (bw._token_fingerprint(access_token), project_id, "")
    fetched_at = time.time() - entry_age_seconds
    plaintext_path = bw._disk_cache_path(home)
    assert bw._write_encrypted_disk_cache(
        cache_key=cache_key,
        access_token=access_token,
        entry=bw._CachedFetch(
            secrets={"K1": "encrypted"},
            fetched_at=fetched_at,
        ),
        home_path=home,
    )
    bw._DISK_CACHE.write(
        cache_key,
        bw._CachedFetch(secrets={"K1": "plaintext"}, fetched_at=fetched_at),
        300,
        home,
    )
    original_unlink = Path.unlink

    def fail_plaintext_unlink(path, *args, **kwargs):
        if path == plaintext_path:
            raise PermissionError("forced plaintext cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_plaintext_unlink)
    monkeypatch.setattr(bw, "find_bws", lambda **kwargs: None)
    monkeypatch.setattr(
        bw,
        "_run_bws_list",
        lambda *args, **kwargs: pytest.fail("missing binary must prevent live fetch"),
    )

    with pytest.raises(RuntimeError) as exc_info:
        bw.fetch_bitwarden_secrets(
            access_token=access_token,
            project_id=project_id,
            cache_ttl_seconds=cache_ttl_seconds,
            encrypted_cache_enabled=True,
            encrypted_cache_max_stale_seconds=max_stale_seconds,
            home_path=home,
        )

    error = str(exc_info.value)
    assert "bws binary not available" in error
    assert str(plaintext_path) in error
    assert plaintext_path.exists()


@pytest.mark.parametrize(
    "cache_ttl_seconds,max_stale_seconds,entry_age_seconds",
    [
        (0, 300, 30),
        (60, 300, 120),
    ],
)
def test_bitwarden_source_missing_binary_reports_stale_transition_cleanup(
    monkeypatch,
    tmp_path,
    cache_ttl_seconds,
    max_stale_seconds,
    entry_age_seconds,
):
    """The adapter reports stale-window cleanup without serving secrets."""
    home = tmp_path / ".hermes"
    home.mkdir()
    access_token = "short-token"
    project_id = "proj-1"
    cache_key = (bw._token_fingerprint(access_token), project_id, "")
    fetched_at = time.time() - entry_age_seconds
    plaintext_path = bw._disk_cache_path(home)
    assert bw._write_encrypted_disk_cache(
        cache_key=cache_key,
        access_token=access_token,
        entry=bw._CachedFetch(
            secrets={"K1": "encrypted"},
            fetched_at=fetched_at,
        ),
        home_path=home,
    )
    bw._DISK_CACHE.write(
        cache_key,
        bw._CachedFetch(secrets={"K1": "plaintext"}, fetched_at=fetched_at),
        300,
        home,
    )
    original_unlink = Path.unlink

    def fail_plaintext_unlink(path, *args, **kwargs):
        if path == plaintext_path:
            raise PermissionError("forced plaintext cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_plaintext_unlink)
    monkeypatch.setattr(bw, "find_bws", lambda **kwargs: None)
    monkeypatch.setenv("BWS_ACCESS_TOKEN", access_token)

    result = bw.BitwardenSource().fetch(
        {
            "project_id": project_id,
            "auto_install": False,
            "cache_ttl_seconds": cache_ttl_seconds,
            "encrypted_cache": {
                "enabled": True,
                "max_stale_seconds": max_stale_seconds,
            },
        },
        home,
    )

    assert result.secrets == {}
    assert result.error_kind is bw.ErrorKind.BINARY_MISSING
    assert result.error is not None
    assert "bws binary not available" in result.error
    assert str(plaintext_path) in result.error
    assert result.binary_path is None
    assert plaintext_path.exists()


@pytest.mark.parametrize(
    "home_keyword,bws_error,expected_kind",
    [
        ("unauthorized-home", "network is unreachable", bw.ErrorKind.NETWORK),
        ("timed out-home", "unauthorized access token", bw.ErrorKind.AUTH_FAILED),
    ],
)
def test_bitwarden_source_classifies_live_error_without_transition_path(
    monkeypatch,
    tmp_path,
    home_keyword,
    bws_error,
    expected_kind,
):
    """Operator-controlled transition paths cannot override error taxonomy."""
    home = tmp_path / home_keyword / ".hermes"
    home.mkdir(parents=True)
    access_token = "short-token"
    project_id = "proj-1"
    cache_key = (bw._token_fingerprint(access_token), project_id, "")
    fetched_at = time.time()
    plaintext_path = bw._disk_cache_path(home)
    assert bw._write_encrypted_disk_cache(
        cache_key=cache_key,
        access_token=access_token,
        entry=bw._CachedFetch(
            secrets={"K1": "encrypted"},
            fetched_at=fetched_at,
        ),
        home_path=home,
    )
    bw._DISK_CACHE.write(
        cache_key,
        bw._CachedFetch(secrets={"K1": "plaintext"}, fetched_at=fetched_at),
        300,
        home,
    )
    original_unlink = Path.unlink

    def fail_plaintext_unlink(path, *args, **kwargs):
        if path == plaintext_path:
            raise PermissionError("forced plaintext cleanup failure")
        return original_unlink(path, *args, **kwargs)

    def fail_live_fetch(*args, **kwargs):
        raise RuntimeError(bws_error)

    monkeypatch.setattr(Path, "unlink", fail_plaintext_unlink)
    monkeypatch.setattr(bw, "find_bws", lambda **kwargs: tmp_path / "bws")
    monkeypatch.setattr(bw, "_run_bws_list", fail_live_fetch)
    monkeypatch.setenv("BWS_ACCESS_TOKEN", access_token)

    result = bw.BitwardenSource().fetch(
        {
            "project_id": project_id,
            "cache_ttl_seconds": 300,
            "encrypted_cache": {
                "enabled": True,
                "max_stale_seconds": 300,
            },
        },
        home,
    )

    assert result.secrets == {}
    assert result.error_kind is expected_kind
    assert result.error is not None
    assert bws_error in result.error
    assert str(plaintext_path) in result.error


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


def test_auth_failure_inspects_ttl_zero_encrypted_transition(
    monkeypatch,
    tmp_path,
):
    """Fatal auth errors still disclose blocked encrypted cleanup state."""
    home = tmp_path / ".hermes"
    home.mkdir()
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    access_token = "short-token"
    cache_key = (bw._token_fingerprint(access_token), "proj-1", "")
    plaintext_path = bw._disk_cache_path(home)
    assert bw._write_encrypted_disk_cache(
        cache_key=cache_key,
        access_token=access_token,
        entry=bw._CachedFetch(secrets={"K1": "encrypted"}, fetched_at=time.time()),
        home_path=home,
    )
    bw._DISK_CACHE.write(
        cache_key,
        bw._CachedFetch(secrets={"K1": "plaintext"}, fetched_at=time.time()),
        300,
        home,
    )
    original_unlink = Path.unlink

    def fail_plaintext_unlink(path, *args, **kwargs):
        if path == plaintext_path:
            raise PermissionError("forced plaintext cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_plaintext_unlink)
    monkeypatch.setattr(
        bw,
        "_run_bws_list",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("unauthorized access token")
        ),
    )

    with pytest.raises(RuntimeError) as exc_info:
        bw.fetch_bitwarden_secrets(
            access_token=access_token,
            project_id="proj-1",
            binary=fake_binary,
            cache_ttl_seconds=0,
            encrypted_cache_enabled=True,
            encrypted_cache_max_stale_seconds=300,
            home_path=home,
        )

    assert "unauthorized access token" in str(exc_info.value)
    assert str(plaintext_path) in str(exc_info.value)
    assert plaintext_path.exists()


def test_bitwarden_source_auth_failure_inspects_ttl_zero_transition(
    monkeypatch,
    tmp_path,
):
    """The adapter preserves AUTH_FAILED while exposing transition warnings."""
    home = tmp_path / ".hermes"
    home.mkdir()
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    access_token = "short-token"
    cache_key = (bw._token_fingerprint(access_token), "proj-1", "")
    plaintext_path = bw._disk_cache_path(home)
    assert bw._write_encrypted_disk_cache(
        cache_key=cache_key,
        access_token=access_token,
        entry=bw._CachedFetch(secrets={"K1": "encrypted"}, fetched_at=time.time()),
        home_path=home,
    )
    bw._DISK_CACHE.write(
        cache_key,
        bw._CachedFetch(secrets={"K1": "plaintext"}, fetched_at=time.time()),
        300,
        home,
    )
    original_unlink = Path.unlink

    def fail_plaintext_unlink(path, *args, **kwargs):
        if path == plaintext_path:
            raise PermissionError("forced plaintext cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_plaintext_unlink)
    monkeypatch.setattr(bw, "find_bws", lambda **kwargs: fake_binary)
    monkeypatch.setattr(
        bw,
        "_run_bws_list",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("unauthorized access token")
        ),
    )
    monkeypatch.setenv("BWS_ACCESS_TOKEN", access_token)

    result = bw.BitwardenSource().fetch(
        {
            "project_id": "proj-1",
            "cache_ttl_seconds": 0,
            "encrypted_cache": {"enabled": True, "max_stale_seconds": 300},
        },
        home,
    )

    assert result.secrets == {}
    assert result.error_kind is bw.ErrorKind.AUTH_FAILED
    assert result.error is not None
    assert str(plaintext_path) in result.error
    assert plaintext_path.exists()


# ---------------------------------------------------------------------------
# Final encrypted-cache state-machine regressions
# ---------------------------------------------------------------------------


def test_fetch_freezes_default_endpoint_for_child_and_cache(
    monkeypatch,
    tmp_path,
):
    """One fetch cannot inherit a different endpoint after key resolution."""
    home = tmp_path / ".hermes"
    home.mkdir()
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    environments = iter(
        [
            {},
            {"BWS_SERVER_URL": "https://vault.bitwarden.eu"},
        ]
    )
    captured_env = {}

    monkeypatch.setattr(bw, "get_source_environment", lambda: next(environments))

    def fake_run(*args, **kwargs):
        captured_env.update(kwargs["env"])
        return mock.Mock(
            returncode=0,
            stdout=_fake_bws_payload([{"key": "REGION", "value": "default"}]),
            stderr="",
        )

    monkeypatch.setattr(bw.subprocess, "run", fake_run)
    secrets, _ = bw.fetch_bitwarden_secrets(
        access_token="0.t",
        project_id="proj-1",
        binary=fake_binary,
        server_url="",
        encrypted_cache_enabled=True,
        cache_ttl_seconds=0,
        home_path=home,
    )
    payload = json.loads(bw._encrypted_disk_cache_path(home).read_text())

    assert secrets == {"REGION": "default"}
    assert "BWS_SERVER_URL" not in captured_env
    assert payload["context"] == {"project_id": "proj-1", "server_url": ""}


def test_bitwarden_source_freezes_source_environment_once(
    monkeypatch,
    tmp_path,
):
    """The adapter binds token, endpoint, child env, and cache in one snapshot."""
    home = tmp_path / ".hermes"
    home.mkdir()
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    environments = iter(
        [
            {"BWS_ACCESS_TOKEN": "0.t"},
            {"BWS_ACCESS_TOKEN": "0.t"},
            {
                "BWS_ACCESS_TOKEN": "0.t",
                "BWS_SERVER_URL": "https://vault.bitwarden.eu",
            },
        ]
    )
    source_calls = {"n": 0}
    captured_env = {}

    def source_environment():
        source_calls["n"] += 1
        return next(environments)

    def fake_run(*args, **kwargs):
        captured_env.update(kwargs["env"])
        return mock.Mock(
            returncode=0,
            stdout=_fake_bws_payload([{"key": "REGION", "value": "default"}]),
            stderr="",
        )

    monkeypatch.setattr(bw, "get_source_environment", source_environment)
    monkeypatch.setattr(bw, "find_bws", lambda **kwargs: fake_binary)
    monkeypatch.setattr(bw.subprocess, "run", fake_run)
    result = bw.BitwardenSource().fetch(
        {
            "project_id": "proj-1",
            "cache_ttl_seconds": 0,
            "encrypted_cache": {"enabled": True},
        },
        home,
    )
    payload = json.loads(bw._encrypted_disk_cache_path(home).read_text())

    assert result.secrets == {"REGION": "default"}
    assert source_calls["n"] == 1
    assert "BWS_SERVER_URL" not in captured_env
    assert payload["context"] == {"project_id": "proj-1", "server_url": ""}


def test_apply_bitwarden_secrets_freezes_source_environment_once(
    monkeypatch,
    tmp_path,
):
    """The legacy apply wrapper uses one endpoint snapshot through the child."""
    home = tmp_path / ".hermes"
    home.mkdir()
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    source_calls = {"n": 0}
    captured_env = {}

    def source_environment():
        source_calls["n"] += 1
        if source_calls["n"] == 1:
            return {"BWS_ACCESS_TOKEN": "0.t"}
        return {
            "BWS_ACCESS_TOKEN": "0.t",
            "BWS_SERVER_URL": "https://vault.bitwarden.eu",
        }

    def fake_run(*args, **kwargs):
        captured_env.update(kwargs["env"])
        return mock.Mock(
            returncode=0,
            stdout=_fake_bws_payload([{"key": "REGION", "value": "default"}]),
            stderr="",
        )

    monkeypatch.setattr(bw, "get_source_environment", source_environment)
    monkeypatch.setattr(bw, "find_bws", lambda **kwargs: fake_binary)
    monkeypatch.setattr(bw.subprocess, "run", fake_run)
    result = bw.apply_bitwarden_secrets(
        enabled=True,
        project_id="proj-1",
        cache_ttl_seconds=0,
        encrypted_cache_enabled=True,
        home_path=home,
    )
    payload = json.loads(bw._encrypted_disk_cache_path(home).read_text())

    assert result.secrets == {"REGION": "default"}
    assert source_calls["n"] == 1
    assert "BWS_SERVER_URL" not in captured_env
    assert payload["context"] == {"project_id": "proj-1", "server_url": ""}


def test_encrypted_prereplacement_failure_blocks_restart_rollback(
    monkeypatch,
    tmp_path,
):
    """A failed newer write cannot make the older ciphertext fresh after restart."""
    home = tmp_path / ".hermes"
    home.mkdir()
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    encrypted_path = bw._encrypted_disk_cache_path(home)
    payloads = iter(
        [
            _fake_bws_payload([{"key": "K1", "value": "old"}]),
            _fake_bws_payload([{"key": "K1", "value": "new"}]),
            _fake_bws_payload([{"key": "K1", "value": "latest"}]),
        ]
    )
    calls = {"n": 0}

    def fake_run(*args, **kwargs):
        calls["n"] += 1
        return mock.Mock(returncode=0, stdout=next(payloads), stderr="")

    monkeypatch.setattr(bw.subprocess, "run", fake_run)
    old, _ = bw.fetch_bitwarden_secrets(
        access_token="0.t",
        project_id="proj-1",
        binary=fake_binary,
        cache_ttl_seconds=300,
        encrypted_cache_enabled=True,
        home_path=home,
    )
    original_mkstemp = bw.tempfile.mkstemp

    def fail_encrypted_mkstemp(*args, **kwargs):
        if kwargs.get("prefix") == ".bws_cache_enc_":
            raise PermissionError("forced pre-replacement failure")
        return original_mkstemp(*args, **kwargs)

    monkeypatch.setattr(bw.tempfile, "mkstemp", fail_encrypted_mkstemp)
    new, warnings = bw.fetch_bitwarden_secrets(
        access_token="0.t",
        project_id="proj-1",
        binary=fake_binary,
        cache_ttl_seconds=0,
        encrypted_cache_enabled=True,
        home_path=home,
    )
    bw._CACHE.clear()
    bw._CACHE_POLICY.clear()
    bw._ENCRYPTED_CACHE_INVALIDATIONS.clear()
    monkeypatch.setattr(bw.tempfile, "mkstemp", original_mkstemp)
    latest, _ = bw.fetch_bitwarden_secrets(
        access_token="0.t",
        project_id="proj-1",
        binary=fake_binary,
        cache_ttl_seconds=300,
        encrypted_cache_enabled=True,
        home_path=home,
    )

    assert old == {"K1": "old"}
    assert new == {"K1": "new"}
    assert warnings == [bw._encrypted_cache_invalidation_warning(home)]
    assert latest == {"K1": "latest"}
    assert calls["n"] == 3
    assert encrypted_path.exists()
    assert not bw._encrypted_cache_invalidation_marker_path(home).exists()


def test_marker_recovery_keeps_veto_until_live_replacement(
    monkeypatch,
    tmp_path,
):
    """Supported recovery leaves the marker until a live write replaces old data."""
    home = tmp_path / ".hermes"
    home.mkdir()
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    access_token = "0.t"
    cache_key = (bw._token_fingerprint(access_token), "proj-1", "")
    assert bw._write_encrypted_disk_cache(
        cache_key=cache_key,
        access_token=access_token,
        entry=bw._CachedFetch(secrets={"K1": "old"}, fetched_at=time.time()),
        home_path=home,
    )
    bw._mark_encrypted_cache_invalidated(cache_key, home)
    marker = bw._encrypted_cache_invalidation_marker_path(home)
    bw._CACHE.clear()
    bw._CACHE_POLICY.clear()
    bw._ENCRYPTED_CACHE_INVALIDATIONS.clear()
    marker_seen_during_live = []

    def fake_run(*args, **kwargs):
        marker_seen_during_live.append(marker.exists())
        return mock.Mock(
            returncode=0,
            stdout=_fake_bws_payload([{"key": "K1", "value": "new"}]),
            stderr="",
        )

    monkeypatch.setattr(bw.subprocess, "run", fake_run)
    recovered, warnings = bw.fetch_bitwarden_secrets(
        access_token=access_token,
        project_id="proj-1",
        binary=fake_binary,
        cache_ttl_seconds=300,
        encrypted_cache_enabled=True,
        home_path=home,
    )

    assert recovered == {"K1": "new"}
    assert warnings == []
    assert marker_seen_during_live == [True]
    assert not marker.exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "invalid_base64",
        "invalid_nonce_length",
        "tampered_ciphertext",
        "invalid_json",
        "invalid_secrets_schema",
        "invalid_secret_value_schema",
        "invalid_fetched_at_schema",
        "mismatched_route_invalid_base64",
    ],
)
def test_malformed_v1_artifacts_are_retired(mutation, tmp_path):
    """Every recognized v1 decode, decrypt, and schema failure retires v1."""
    home = tmp_path / mutation / ".hermes"
    home.mkdir(parents=True)
    access_token = "short-token"
    cache_key = (bw._token_fingerprint(access_token), "proj-1", "")
    if mutation == "invalid_json":
        path = _seed_legacy_encrypted_cache(
            home,
            access_token=access_token,
            cache_key=cache_key,
            raw_plaintext=b"not-json",
        )
    elif mutation == "invalid_secrets_schema":
        path = _seed_legacy_encrypted_cache(
            home,
            access_token=access_token,
            cache_key=cache_key,
            inner={"secrets": ["not", "a", "mapping"], "fetched_at": time.time()},
        )
    elif mutation == "invalid_secret_value_schema":
        path = _seed_legacy_encrypted_cache(
            home,
            access_token=access_token,
            cache_key=cache_key,
            inner={"secrets": {"K1": 123}, "fetched_at": time.time()},
        )
    elif mutation == "invalid_fetched_at_schema":
        path = _seed_legacy_encrypted_cache(
            home,
            access_token=access_token,
            cache_key=cache_key,
            inner={"secrets": {"K1": "cached"}, "fetched_at": "now"},
        )
    else:
        path = _seed_legacy_encrypted_cache(
            home,
            access_token=access_token,
            cache_key=cache_key,
        )
        payload = json.loads(path.read_text())
        if mutation in ("invalid_base64", "mismatched_route_invalid_base64"):
            payload["salt"] = "%%%"
        elif mutation == "invalid_nonce_length":
            payload["nonce"] = bw._b64e(b"short")
        elif mutation == "tampered_ciphertext":
            ciphertext = bytearray(bw._b64d(payload["ciphertext"]))
            ciphertext[-1] ^= 1
            payload["ciphertext"] = bw._b64e(bytes(ciphertext))
        if mutation == "mismatched_route_invalid_base64":
            payload["key"] = "deadbeefdeadbeef|other-project|"
        path.write_text(json.dumps(payload))

    warnings = []
    entry = bw._read_encrypted_disk_cache(
        cache_key=cache_key,
        access_token=access_token,
        max_age_seconds=300,
        home_path=home,
        transition_warnings=warnings,
    )

    assert entry is None
    assert not path.exists() or json.loads(path.read_text()).get("version") != 1
    assert warnings == []


def test_malformed_v1_failed_retirement_sets_durable_veto(
    monkeypatch,
    tmp_path,
):
    """A v1 that cannot be removed or tombstoned remains durably vetoed."""
    home = tmp_path / ".hermes"
    home.mkdir()
    access_token = "short-token"
    cache_key = (bw._token_fingerprint(access_token), "proj-1", "")
    path = _seed_legacy_encrypted_cache(
        home,
        access_token=access_token,
        cache_key=cache_key,
    )
    payload = json.loads(path.read_text())
    payload["salt"] = "%%%"
    path.write_text(json.dumps(payload))
    original_unlink = Path.unlink
    original_replace = bw.os.replace

    def fail_encrypted_unlink(candidate, *args, **kwargs):
        if candidate == path:
            raise PermissionError("forced v1 unlink failure")
        return original_unlink(candidate, *args, **kwargs)

    def fail_encrypted_replace(source, destination, *args, **kwargs):
        if Path(destination) == path:
            raise PermissionError("forced v1 tombstone failure")
        return original_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_encrypted_unlink)
    monkeypatch.setattr(bw.os, "replace", fail_encrypted_replace)
    warnings = []
    entry = bw._read_encrypted_disk_cache(
        cache_key=cache_key,
        access_token=access_token,
        max_age_seconds=300,
        home_path=home,
        transition_warnings=warnings,
    )

    assert entry is None
    assert json.loads(path.read_text())["version"] == 1
    assert bw._encrypted_cache_was_invalidated(cache_key, home)
    assert warnings == [bw._encrypted_cache_invalidation_warning(home)]


@pytest.mark.parametrize("wrapper", ["apply", "source"])
def test_public_wrappers_retire_malformed_v1_before_missing_binary(
    monkeypatch,
    tmp_path,
    wrapper,
):
    """Both non-raising public wrappers reach fail-closed v1 retirement."""
    home = tmp_path / wrapper / ".hermes"
    home.mkdir(parents=True)
    access_token = "short-token"
    cache_key = (bw._token_fingerprint(access_token), "proj-1", "")
    path = _seed_legacy_encrypted_cache(
        home,
        access_token=access_token,
        cache_key=cache_key,
    )
    payload = json.loads(path.read_text())
    payload["salt"] = "%%%"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(
        bw,
        "get_source_environment",
        lambda: {"BWS_ACCESS_TOKEN": access_token},
    )
    monkeypatch.setattr(bw, "find_bws", lambda **kwargs: None)

    if wrapper == "apply":
        result = bw.apply_bitwarden_secrets(
            enabled=True,
            project_id="proj-1",
            auto_install=False,
            cache_ttl_seconds=300,
            encrypted_cache_enabled=True,
            home_path=home,
        )
    else:
        result = bw.BitwardenSource().fetch(
            {
                "project_id": "proj-1",
                "auto_install": False,
                "cache_ttl_seconds": 300,
                "encrypted_cache": {"enabled": True},
            },
            home,
        )

    assert result.secrets == {}
    assert result.error is not None
    assert "bws binary not available" in result.error
    assert not path.exists() or json.loads(path.read_text()).get("version") != 1


@pytest.mark.parametrize(
    "inherited_endpoint",
    [
        "https://vault.bitwarden.eu",
        "https://vault.example.test",
    ],
)
def test_inherited_endpoint_v1_is_hardened_but_not_served_without_live_binding(
    monkeypatch,
    tmp_path,
    inherited_endpoint,
):
    """An origin-ambiguous base-v1 cache is migrated, quarantined, and diagnosed."""
    home = tmp_path / ".hermes"
    home.mkdir()
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    access_token = "short-token"
    legacy_key = (bw._token_fingerprint(access_token), "proj-1", "")
    path = _seed_legacy_encrypted_cache(
        home,
        access_token=access_token,
        cache_key=legacy_key,
    )
    monkeypatch.setattr(
        bw,
        "get_source_environment",
        lambda: {"BWS_SERVER_URL": inherited_endpoint},
    )
    monkeypatch.setattr(
        bw,
        "_run_bws_list",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("network is unreachable")
        ),
    )

    with pytest.raises(RuntimeError) as exc_info:
        bw.fetch_bitwarden_secrets(
            access_token=access_token,
            project_id="proj-1",
            binary=fake_binary,
            cache_ttl_seconds=300,
            encrypted_cache_enabled=True,
            encrypted_cache_max_stale_seconds=300,
            home_path=home,
        )

    payload = json.loads(path.read_text())
    assert payload["version"] == bw._ENCRYPTED_CACHE_VERSION
    assert payload["context"]["server_url"] == (
        bw._ENCRYPTED_CACHE_LEGACY_UNBOUND_SERVER
    )
    assert "inherited endpoint" in str(exc_info.value)
    assert "falling back to stale ENCRYPTED" not in str(exc_info.value)
    warnings = []
    assert (
        bw._read_encrypted_disk_cache(
            cache_key=legacy_key,
            access_token=access_token,
            max_age_seconds=300,
            home_path=home,
            transition_warnings=warnings,
        )
        is None
    )
    assert warnings == [bw._encrypted_cache_legacy_endpoint_warning(home)]


def test_inherited_endpoint_v1_live_success_rebinds_current_route(
    monkeypatch,
    tmp_path,
):
    """A live success replaces quarantined v1 data under the actual endpoint."""
    home = tmp_path / ".hermes"
    home.mkdir()
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    access_token = "short-token"
    inherited_endpoint = "https://vault.bitwarden.eu"
    _seed_legacy_encrypted_cache(
        home,
        access_token=access_token,
        cache_key=(bw._token_fingerprint(access_token), "proj-1", ""),
    )
    monkeypatch.setattr(
        bw,
        "get_source_environment",
        lambda: {"BWS_SERVER_URL": inherited_endpoint},
    )
    monkeypatch.setattr(
        bw.subprocess,
        "run",
        lambda *args, **kwargs: mock.Mock(
            returncode=0,
            stdout=_fake_bws_payload([{"key": "K1", "value": "live"}]),
            stderr="",
        ),
    )

    secrets, warnings = bw.fetch_bitwarden_secrets(
        access_token=access_token,
        project_id="proj-1",
        binary=fake_binary,
        cache_ttl_seconds=300,
        encrypted_cache_enabled=True,
        home_path=home,
    )
    payload = json.loads(bw._encrypted_disk_cache_path(home).read_text())

    assert secrets == {"K1": "live"}
    assert warnings == []
    assert payload["context"] == {
        "project_id": "proj-1",
        "server_url": inherited_endpoint,
    }


def test_bitwarden_source_reports_inherited_endpoint_v1_live_refetch(
    monkeypatch,
    tmp_path,
):
    """The registry adapter preserves NETWORK and the v1 refetch diagnostic."""
    home = tmp_path / ".hermes"
    home.mkdir()
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    access_token = "short-token"
    inherited_endpoint = "https://vault.bitwarden.eu"
    path = _seed_legacy_encrypted_cache(
        home,
        access_token=access_token,
        cache_key=(bw._token_fingerprint(access_token), "proj-1", ""),
    )
    monkeypatch.setattr(
        bw,
        "get_source_environment",
        lambda: {
            "BWS_ACCESS_TOKEN": access_token,
            "BWS_SERVER_URL": inherited_endpoint,
        },
    )
    monkeypatch.setattr(bw, "find_bws", lambda **kwargs: fake_binary)
    monkeypatch.setattr(
        bw,
        "_run_bws_list",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("network is unreachable")
        ),
    )

    result = bw.BitwardenSource().fetch(
        {
            "project_id": "proj-1",
            "cache_ttl_seconds": 300,
            "encrypted_cache": {"enabled": True, "max_stale_seconds": 300},
        },
        home,
    )

    assert result.secrets == {}
    assert result.error_kind is bw.ErrorKind.NETWORK
    assert result.error is not None
    assert "inherited endpoint" in result.error
    assert json.loads(path.read_text())["version"] == bw._ENCRYPTED_CACHE_VERSION


@pytest.mark.parametrize(
    ("failure_kind", "failure_message"),
    [
        (bw.ErrorKind.BINARY_MISSING, "bws binary not available"),
        (bw.ErrorKind.AUTH_FAILED, "unauthorized access token"),
        (bw.ErrorKind.INTERNAL, "malformed bws response"),
    ],
)
@pytest.mark.parametrize("surface", ["direct", "apply", "source"])
def test_zero_window_failures_retire_v1_without_changing_taxonomy(
    monkeypatch,
    tmp_path,
    surface,
    failure_kind,
    failure_message,
):
    """Discard-only encrypted audits retire v1 even with no serving window."""
    home = tmp_path / surface / failure_kind.value / ".hermes"
    home.mkdir(parents=True)
    fake_binary = tmp_path / surface / failure_kind.value / "bws"
    fake_binary.write_text("")
    access_token = "short-token"
    path = _seed_legacy_encrypted_cache(home, access_token=access_token)
    monkeypatch.setattr(
        bw,
        "get_source_environment",
        lambda: {"BWS_ACCESS_TOKEN": access_token},
    )
    monkeypatch.setattr(
        bw,
        "find_bws",
        lambda **kwargs: (
            None if failure_kind is bw.ErrorKind.BINARY_MISSING else fake_binary
        ),
    )
    if failure_kind is not bw.ErrorKind.BINARY_MISSING:
        monkeypatch.setattr(
            bw,
            "_run_bws_list",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError(failure_message)
            ),
        )

    config = {
        "project_id": "proj-1",
        "auto_install": False,
        "cache_ttl_seconds": 0,
        "encrypted_cache": {"enabled": True, "max_stale_seconds": 0},
    }
    if surface == "direct":
        kwargs = {
            "access_token": access_token,
            "project_id": "proj-1",
            "cache_ttl_seconds": 0,
            "encrypted_cache_enabled": True,
            "encrypted_cache_max_stale_seconds": 0,
            "home_path": home,
        }
        if failure_kind is not bw.ErrorKind.BINARY_MISSING:
            kwargs["binary"] = fake_binary
        with pytest.raises(RuntimeError) as exc_info:
            bw.fetch_bitwarden_secrets(**kwargs)
        error = str(exc_info.value)
        observed_kind = (
            exc_info.value.error_kind
            if isinstance(exc_info.value, bw._BwsFetchError)
            else bw._classify_bws_error(error)
        )
    elif surface == "apply":
        result = bw.apply_bitwarden_secrets(
            enabled=True,
            project_id="proj-1",
            auto_install=False,
            cache_ttl_seconds=0,
            encrypted_cache_enabled=True,
            encrypted_cache_max_stale_seconds=0,
            home_path=home,
        )
        error = result.error or ""
        observed_kind = bw._classify_bws_error(error)
    else:
        result = bw.BitwardenSource().fetch(config, home)
        error = result.error or ""
        observed_kind = result.error_kind

    assert failure_message in error
    assert observed_kind is failure_kind
    assert not path.exists() or json.loads(path.read_text()).get("version") != 1


@pytest.mark.parametrize(
    "max_stale_seconds,should_serve",
    [
        pytest.param(float("nan"), False, id="nan"),
        pytest.param(-1, False, id="negative"),
        pytest.param(float("-inf"), False, id="negative-infinity"),
        pytest.param(float("inf"), False, id="positive-infinity"),
        pytest.param(172800, True, id="large-finite-control"),
    ],
)
@pytest.mark.parametrize("cache_version", ["v1", "v2"])
@pytest.mark.parametrize("surface", ["direct", "apply", "source"])
def test_encrypted_stale_windows_are_finite_positive_and_bounded(
    monkeypatch,
    tmp_path,
    surface,
    cache_version,
    max_stale_seconds,
    should_serve,
):
    """Only finite positive windows serve v1/v2 after NETWORK failures."""
    home = tmp_path / surface / cache_version / ".hermes"
    home.mkdir(parents=True)
    fake_binary = home.parent / "bws"
    fake_binary.write_text("")
    access_token = "short-token"
    cache_key = (bw._token_fingerprint(access_token), "proj-1", "")
    cached_secret = f"{cache_version}-secret"
    entry = bw._CachedFetch(
        secrets={"STALE": cached_secret},
        fetched_at=time.time() - 86400,
    )
    if cache_version == "v1":
        path = _seed_legacy_encrypted_cache(
            home,
            access_token=access_token,
            inner={
                "secrets": entry.secrets,
                "fetched_at": entry.fetched_at,
            },
        )
    else:
        path = bw._encrypted_disk_cache_path(home)
        assert bw._write_encrypted_disk_cache(
            cache_key=cache_key,
            access_token=access_token,
            entry=entry,
            home_path=home,
        )
    monkeypatch.setattr(
        bw,
        "get_source_environment",
        lambda: {"BWS_ACCESS_TOKEN": access_token},
    )
    monkeypatch.setattr(bw, "find_bws", lambda **kwargs: fake_binary)
    monkeypatch.setattr(
        bw,
        "_run_bws_list",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("network is unreachable")
        ),
    )

    if surface == "direct":
        try:
            secrets, warnings = bw.fetch_bitwarden_secrets(
                access_token=access_token,
                project_id="proj-1",
                binary=fake_binary,
                cache_ttl_seconds=0,
                encrypted_cache_enabled=True,
                encrypted_cache_max_stale_seconds=max_stale_seconds,
                home_path=home,
            )
        except RuntimeError as exc:
            secrets = {}
            warnings = []
            error = str(exc)
            observed_kind = bw._classify_bws_error(error)
        else:
            error = ""
            observed_kind = None
    elif surface == "apply":
        result = bw.apply_bitwarden_secrets(
            enabled=True,
            project_id="proj-1",
            auto_install=False,
            cache_ttl_seconds=0,
            encrypted_cache_enabled=True,
            encrypted_cache_max_stale_seconds=max_stale_seconds,
            home_path=home,
        )
        secrets = result.secrets
        warnings = result.warnings
        error = result.error or ""
        observed_kind = bw._classify_bws_error(error) if error else None
    else:
        result = bw.BitwardenSource().fetch(
            {
                "project_id": "proj-1",
                "auto_install": False,
                "cache_ttl_seconds": 0,
                "encrypted_cache": {
                    "enabled": True,
                    "max_stale_seconds": max_stale_seconds,
                },
            },
            home,
        )
        secrets = result.secrets
        warnings = result.warnings
        error = result.error or ""
        observed_kind = result.error_kind

    if should_serve:
        assert secrets == {"STALE": cached_secret}
        assert len(warnings) == 1
        assert "stale ENCRYPTED disk cache" in warnings[0]
        assert error == ""
        assert observed_kind is None
    else:
        assert secrets == {}
        assert warnings == []
        assert "network is unreachable" in error
        assert observed_kind is bw.ErrorKind.NETWORK
    assert json.loads(path.read_text())["version"] == bw._ENCRYPTED_CACHE_VERSION
    reread = bw._read_encrypted_disk_cache(
        cache_key=cache_key,
        access_token=access_token,
        max_age_seconds=max_stale_seconds,
        home_path=home,
    )
    assert (reread is not None) is should_serve


@pytest.mark.parametrize(
    "endpoint",
    ["https://vault.bitwarden.eu", "https://vault.example.test"],
)
@pytest.mark.parametrize("surface", ["direct", "apply", "source"])
@pytest.mark.parametrize("with_v1", [False, True], ids=["plain-only", "v1-rebind"])
def test_inherited_endpoint_live_success_removes_empty_route_plaintext_alias(
    monkeypatch,
    tmp_path,
    endpoint,
    surface,
    with_v1,
):
    """Every public surface removes the legacy inherited-route plaintext alias."""
    home = tmp_path / endpoint.rsplit("/", 1)[-1] / surface / str(with_v1) / ".hermes"
    home.mkdir(parents=True)
    fake_binary = home.parent / "bws"
    fake_binary.write_text("")
    access_token = "short-token"
    legacy_key = (bw._token_fingerprint(access_token), "proj-1", "")
    fetched_at = time.time() - 60
    if with_v1:
        _seed_legacy_encrypted_cache(
            home,
            access_token=access_token,
            cache_key=legacy_key,
            inner={"secrets": {"V1": "legacy"}, "fetched_at": fetched_at},
        )
    bw._DISK_CACHE.write(
        legacy_key,
        bw._CachedFetch(
            secrets={"PLAIN": "legacy-plaintext"},
            fetched_at=fetched_at,
        ),
        300,
        home,
    )
    plaintext_path = bw._disk_cache_path(home)
    source_env = {
        "BWS_ACCESS_TOKEN": access_token,
        "BWS_SERVER_URL": endpoint,
    }
    monkeypatch.setattr(bw, "get_source_environment", lambda: source_env)
    monkeypatch.setattr(bw, "find_bws", lambda **kwargs: fake_binary)
    monkeypatch.setattr(
        bw.subprocess,
        "run",
        lambda *args, **kwargs: mock.Mock(
            returncode=0,
            stdout=_fake_bws_payload([{"key": "K1", "value": "live"}]),
            stderr="",
        ),
    )

    if surface == "direct":
        secrets, warnings = bw.fetch_bitwarden_secrets(
            access_token=access_token,
            project_id="proj-1",
            binary=fake_binary,
            cache_ttl_seconds=300,
            encrypted_cache_enabled=True,
            home_path=home,
            source_env=source_env,
        )
        error = None
    elif surface == "apply":
        result = bw.apply_bitwarden_secrets(
            enabled=True,
            project_id="proj-1",
            auto_install=False,
            cache_ttl_seconds=300,
            encrypted_cache_enabled=True,
            home_path=home,
        )
        secrets, warnings, error = result.secrets, result.warnings, result.error
    else:
        result = bw.BitwardenSource().fetch(
            {
                "project_id": "proj-1",
                "auto_install": False,
                "cache_ttl_seconds": 300,
                "encrypted_cache": {"enabled": True},
            },
            home,
        )
        secrets, warnings, error = result.secrets, result.warnings, result.error

    payload = json.loads(bw._encrypted_disk_cache_path(home).read_text())
    assert secrets == {"K1": "live"}
    assert warnings == []
    assert error is None
    assert payload["version"] == bw._ENCRYPTED_CACHE_VERSION
    assert payload["context"] == {"project_id": "proj-1", "server_url": endpoint}
    assert not plaintext_path.exists()


@pytest.mark.parametrize(
    "encrypted_route",
    [
        "https://vault.bitwarden.eu",
        bw._ENCRYPTED_CACHE_LEGACY_UNBOUND_SERVER,
    ],
)
def test_inherited_empty_route_plaintext_alias_preserves_newer_write(
    tmp_path,
    encrypted_route,
):
    """A newer legacy-alias plaintext write wins and emits the exact warning."""
    home = tmp_path / ".hermes"
    home.mkdir()
    access_token = "short-token"
    current_key = (bw._token_fingerprint(access_token), "proj-1", encrypted_route)
    legacy_key = (current_key[0], current_key[1], "")
    encrypted_fetched_at = time.time() - 60
    bw._DISK_CACHE.write(
        legacy_key,
        bw._CachedFetch(
            secrets={"PLAIN": "newer"},
            fetched_at=encrypted_fetched_at + 30,
        ),
        300,
        home,
    )
    warnings = []

    transition_complete = bw._write_encrypted_disk_cache(
        cache_key=current_key,
        access_token=access_token,
        entry=bw._CachedFetch(
            secrets={"K1": "older-encrypted"},
            fetched_at=encrypted_fetched_at,
        ),
        home_path=home,
        transition_warnings=warnings,
    )

    assert transition_complete is False
    assert bw._disk_cache_path(home).exists()
    assert warnings == [bw._encrypted_cache_newer_plaintext_warning(home)]


@pytest.mark.parametrize(
    "unrelated_key",
    [
        (bw._token_fingerprint("other-token"), "proj-1", ""),
        (bw._token_fingerprint("short-token"), "other-project", ""),
    ],
    ids=["different-token", "different-project"],
)
def test_inherited_alias_cleanup_preserves_unrelated_plaintext(
    tmp_path,
    unrelated_key,
):
    """Empty-route aliases never cross token or project boundaries."""
    home = tmp_path / ".hermes"
    home.mkdir()
    access_token = "short-token"
    current_key = (
        bw._token_fingerprint(access_token),
        "proj-1",
        "https://vault.bitwarden.eu",
    )
    plaintext_path = bw._disk_cache_path(home)
    bw._DISK_CACHE.write(
        unrelated_key,
        bw._CachedFetch(
            secrets={"OTHER": "preserved"},
            fetched_at=time.time() - 60,
        ),
        300,
        home,
    )

    assert bw._write_encrypted_disk_cache(
        cache_key=current_key,
        access_token=access_token,
        entry=bw._CachedFetch(secrets={"K1": "encrypted"}, fetched_at=time.time()),
        home_path=home,
    )
    assert plaintext_path.exists()
    assert json.loads(plaintext_path.read_text())["key"] == bw._cache_key_str(
        unrelated_key
    )
