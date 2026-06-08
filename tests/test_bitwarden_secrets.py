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
        zf.writestr("bws", binary_bytes)
    return buf.getvalue()


def _force_supported_bws_platform(monkeypatch):
    """Keep installer-path tests hermetic on unsupported hosts (Android/Termux)."""
    monkeypatch.setattr(bw, "_platform_asset_name", lambda: f"bws-aarch64-unknown-linux-gnu-{bw._BWS_VERSION}.zip")
    monkeypatch.setattr(bw, "_platform_binary_name", lambda: "bws")



# ---------------------------------------------------------------------------
# _safe_extract_member — zip-slip containment
# ---------------------------------------------------------------------------


def test_safe_extract_member_extracts_normal_member(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("bws", b"hello")
    buf.seek(0)

    dest = tmp_path / "extract"
    dest.mkdir()
    with zipfile.ZipFile(buf) as zf:
        out = bw._safe_extract_member(zf, "bws", dest)

    assert out == (dest / "bws").resolve() or out == dest / "bws"
    assert Path(out).read_bytes() == b"hello"
    # Nothing escaped the destination directory.
    assert Path(out).resolve().parent == dest.resolve()


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


def test_safe_extract_member_rejects_absolute_path(tmp_path):
    # An absolute member name should never resolve inside dest.
    abs_member = "/etc/cron.d/evil" if os.name != "nt" else "C:/Windows/evil"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(abs_member, b"pwned")
    buf.seek(0)

    dest = tmp_path / "extract"
    dest.mkdir()
    with zipfile.ZipFile(buf) as zf:
        # Absolute paths are reduced to a relative member by zipfile, but
        # we exercise the guard directly with the raw escaping name too.
        with pytest.raises(RuntimeError, match="unsafe archive member"):
            bw._safe_extract_member(zf, "../../../etc/cron.d/evil", dest)


def test_install_bws_rejects_malicious_member(hermes_home, monkeypatch):
    _force_supported_bws_platform(monkeypatch)
    # Build an archive whose only matching member escapes the temp dir.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"../../{bw._platform_binary_name()}", b"pwned")
    zip_bytes = buf.getvalue()
    asset_name = bw._platform_asset_name()
    checksum_text = f"{hashlib.sha256(zip_bytes).hexdigest()}  {asset_name}\n"

    def fake_download(url, dest):
        if url.endswith(".zip"):
            Path(dest).write_bytes(zip_bytes)
        elif url.endswith(".txt"):
            Path(dest).write_text(checksum_text)
        else:
            raise AssertionError(f"unexpected download url: {url}")

    monkeypatch.setattr(bw, "_http_download", fake_download)

    with pytest.raises(RuntimeError, match="unsafe archive member"):
        bw.install_bws()


def test_install_bws_happy_path(hermes_home, monkeypatch):
    _force_supported_bws_platform(monkeypatch)
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


def test_install_bws_checksum_mismatch(hermes_home, monkeypatch):
    _force_supported_bws_platform(monkeypatch)
    zip_bytes = _make_fake_zip(b"contents")
    asset_name = bw._platform_asset_name()
    wrong_checksum = "0" * 64
    checksum_text = f"{wrong_checksum}  {asset_name}\n"

    def fake_download(url, dest):
        if url.endswith(".zip"):
            Path(dest).write_bytes(zip_bytes)
        else:
            Path(dest).write_text(checksum_text)

    monkeypatch.setattr(bw, "_http_download", fake_download)

    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        bw.install_bws()


def test_install_bws_missing_checksum_entry(hermes_home, monkeypatch):
    _force_supported_bws_platform(monkeypatch)
    zip_bytes = _make_fake_zip(b"x")

    def fake_download(url, dest):
        if url.endswith(".zip"):
            Path(dest).write_bytes(zip_bytes)
        else:
            Path(dest).write_text("ffffffff  some-other-file.zip\n")

    monkeypatch.setattr(bw, "_http_download", fake_download)

    with pytest.raises(RuntimeError, match="No checksum entry"):
        bw.install_bws()


# ---------------------------------------------------------------------------
# fetch_bitwarden_secrets
# ---------------------------------------------------------------------------


def _fake_bws_payload(items):
    return json.dumps(items)


def test_fetch_happy_path(monkeypatch, tmp_path):
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    payload = _fake_bws_payload([
        {"key": "OPENAI_API_KEY", "value": "sk-abc"},
        {"key": "ANTHROPIC_API_KEY", "value": "sk-ant-xyz"},
    ])

    def fake_run(cmd, **kwargs):
        assert cmd[0] == str(fake_binary)
        assert "secret" in cmd and "list" in cmd
        assert kwargs["env"]["BWS_ACCESS_TOKEN"] == "0.fake.token"
        return mock.Mock(returncode=0, stdout=payload, stderr="")

    monkeypatch.setattr(bw.subprocess, "run", fake_run)

    secrets, warnings = bw.fetch_bitwarden_secrets(
        access_token="0.fake.token",
        project_id="proj-uuid",
        binary=fake_binary,
        use_cache=False,
    )
    assert secrets == {
        "OPENAI_API_KEY": "sk-abc",
        "ANTHROPIC_API_KEY": "sk-ant-xyz",
    }
    assert warnings == []


def test_fetch_skips_invalid_env_names(monkeypatch, tmp_path):
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    payload = _fake_bws_payload([
        {"key": "VALID_KEY", "value": "v1"},
        {"key": "1BAD_START", "value": "v2"},
        {"key": "has spaces", "value": "v3"},
        {"key": "DASH-KEY", "value": "v4"},
    ])

    monkeypatch.setattr(
        bw.subprocess,
        "run",
        lambda *a, **kw: mock.Mock(returncode=0, stdout=payload, stderr=""),
    )

    secrets, warnings = bw.fetch_bitwarden_secrets(
        access_token="0.t",
        project_id="p",
        binary=fake_binary,
        use_cache=False,
    )
    assert secrets == {"VALID_KEY": "v1"}
    assert len(warnings) == 3


def test_fetch_auth_failure(monkeypatch, tmp_path):
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")

    monkeypatch.setattr(
        bw.subprocess,
        "run",
        lambda *a, **kw: mock.Mock(
            returncode=1, stdout="", stderr="Error: invalid access token"
        ),
    )

    with pytest.raises(RuntimeError, match="invalid access token"):
        bw.fetch_bitwarden_secrets(
            access_token="0.bad",
            project_id="p",
            binary=fake_binary,
            use_cache=False,
        )


def test_fetch_timeout(monkeypatch, tmp_path):
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")

    def fake_run(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="bws", timeout=30)

    monkeypatch.setattr(bw.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="timed out"):
        bw.fetch_bitwarden_secrets(
            access_token="0.t",
            project_id="p",
            binary=fake_binary,
            use_cache=False,
        )


def test_fetch_non_json(monkeypatch, tmp_path):
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")

    monkeypatch.setattr(
        bw.subprocess,
        "run",
        lambda *a, **kw: mock.Mock(
            returncode=0, stdout="not json at all", stderr=""
        ),
    )

    with pytest.raises(RuntimeError, match="non-JSON"):
        bw.fetch_bitwarden_secrets(
            access_token="0.t",
            project_id="p",
            binary=fake_binary,
            use_cache=False,
        )


def test_fetch_cache_hits(monkeypatch, tmp_path):
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    payload = _fake_bws_payload([{"key": "K", "value": "v"}])

    call_count = {"n": 0}
    def fake_run(*a, **kw):
        call_count["n"] += 1
        return mock.Mock(returncode=0, stdout=payload, stderr="")

    monkeypatch.setattr(bw.subprocess, "run", fake_run)

    bw.fetch_bitwarden_secrets(access_token="0.t", project_id="p",
                                binary=fake_binary, cache_ttl_seconds=60)
    bw.fetch_bitwarden_secrets(access_token="0.t", project_id="p",
                                binary=fake_binary, cache_ttl_seconds=60)
    assert call_count["n"] == 1  # cached on second call


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


def test_fetch_no_server_url_does_not_set_env(monkeypatch, tmp_path):
    """When server_url is empty, BWS_SERVER_URL must not be injected."""
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    payload = _fake_bws_payload([])
    # Make sure the inherited env doesn't already have BWS_SERVER_URL set.
    monkeypatch.delenv("BWS_SERVER_URL", raising=False)

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
    )
    assert "BWS_SERVER_URL" not in captured_env


def test_fetch_server_url_keyed_in_cache(monkeypatch, tmp_path):
    """Different server_url values must produce separate cache entries."""
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    payload = _fake_bws_payload([{"key": "K", "value": "v"}])

    call_count = {"n": 0}

    def fake_run(*a, **kw):
        call_count["n"] += 1
        return mock.Mock(returncode=0, stdout=payload, stderr="")

    monkeypatch.setattr(bw.subprocess, "run", fake_run)

    # US (default empty) — fresh fetch.
    bw.fetch_bitwarden_secrets(
        access_token="0.t", project_id="p",
        binary=fake_binary, cache_ttl_seconds=60,
    )
    # EU — different server_url, must NOT hit the US cache entry.
    bw.fetch_bitwarden_secrets(
        access_token="0.t", project_id="p",
        binary=fake_binary, cache_ttl_seconds=60,
        server_url="https://vault.bitwarden.eu",
    )
    # Second EU call hits cache.
    bw.fetch_bitwarden_secrets(
        access_token="0.t", project_id="p",
        binary=fake_binary, cache_ttl_seconds=60,
        server_url="https://vault.bitwarden.eu",
    )
    assert call_count["n"] == 2


def test_fetch_cache_disabled(monkeypatch, tmp_path):
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    payload = _fake_bws_payload([])
    call_count = {"n": 0}
    def fake_run(*a, **kw):
        call_count["n"] += 1
        return mock.Mock(returncode=0, stdout=payload, stderr="")
    monkeypatch.setattr(bw.subprocess, "run", fake_run)

    bw.fetch_bitwarden_secrets(access_token="0.t", project_id="p",
                                binary=fake_binary, use_cache=False)
    bw.fetch_bitwarden_secrets(access_token="0.t", project_id="p",
                                binary=fake_binary, use_cache=False)
    assert call_count["n"] == 2


# ---------------------------------------------------------------------------
# apply_bitwarden_secrets — the public entry point used by env_loader
# ---------------------------------------------------------------------------


def test_apply_disabled_returns_empty():
    result = bw.apply_bitwarden_secrets(enabled=False, project_id="p")
    assert result.ok
    assert not result.applied
    assert not result.error


def test_apply_missing_token(monkeypatch):
    monkeypatch.delenv("BWS_ACCESS_TOKEN", raising=False)
    result = bw.apply_bitwarden_secrets(
        enabled=True, project_id="p", auto_install=False
    )
    assert not result.ok
    assert "BWS_ACCESS_TOKEN" in result.error


def test_apply_missing_project_id(monkeypatch):
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.t")
    result = bw.apply_bitwarden_secrets(
        enabled=True, project_id="", auto_install=False
    )
    assert not result.ok
    assert "project_id" in result.error


def test_apply_does_not_override_existing(monkeypatch, tmp_path):
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.t")
    monkeypatch.setenv("OPENAI_API_KEY", "existing-value")
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    payload = _fake_bws_payload([
        {"key": "OPENAI_API_KEY", "value": "bsm-value"},
        {"key": "NEW_KEY", "value": "new-value"},
    ])
    monkeypatch.setattr(
        bw.subprocess, "run",
        lambda *a, **kw: mock.Mock(returncode=0, stdout=payload, stderr=""),
    )
    monkeypatch.setattr(bw, "find_bws", lambda **kw: fake_binary)

    result = bw.apply_bitwarden_secrets(
        enabled=True, project_id="p",
        override_existing=False, auto_install=False,
    )
    assert result.ok
    assert "NEW_KEY" in result.applied
    assert "OPENAI_API_KEY" in result.skipped
    assert os.environ["OPENAI_API_KEY"] == "existing-value"
    assert os.environ["NEW_KEY"] == "new-value"


def test_apply_override_existing(monkeypatch, tmp_path):
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.t")
    monkeypatch.setenv("OPENAI_API_KEY", "stale")
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    payload = _fake_bws_payload([{"key": "OPENAI_API_KEY", "value": "fresh"}])
    monkeypatch.setattr(
        bw.subprocess, "run",
        lambda *a, **kw: mock.Mock(returncode=0, stdout=payload, stderr=""),
    )
    monkeypatch.setattr(bw, "find_bws", lambda **kw: fake_binary)

    result = bw.apply_bitwarden_secrets(
        enabled=True, project_id="p",
        override_existing=True, auto_install=False,
    )
    assert result.ok
    assert os.environ["OPENAI_API_KEY"] == "fresh"


def test_apply_never_overrides_bootstrap_token(monkeypatch, tmp_path):
    """Even with override_existing=True, the access-token var is preserved."""
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.original")
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    payload = _fake_bws_payload([
        {"key": "BWS_ACCESS_TOKEN", "value": "0.malicious-replacement"},
    ])
    monkeypatch.setattr(
        bw.subprocess, "run",
        lambda *a, **kw: mock.Mock(returncode=0, stdout=payload, stderr=""),
    )
    monkeypatch.setattr(bw, "find_bws", lambda **kw: fake_binary)

    result = bw.apply_bitwarden_secrets(
        enabled=True, project_id="p",
        override_existing=True, auto_install=False,
    )
    assert os.environ["BWS_ACCESS_TOKEN"] == "0.original"
    assert "BWS_ACCESS_TOKEN" in result.skipped


def test_apply_swallows_fetch_errors(monkeypatch, tmp_path):
    """A fetch failure produces an error, NOT an exception."""
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.t")
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    monkeypatch.setattr(
        bw.subprocess, "run",
        lambda *a, **kw: mock.Mock(returncode=1, stdout="", stderr="bad token"),
    )
    monkeypatch.setattr(bw, "find_bws", lambda **kw: fake_binary)

    result = bw.apply_bitwarden_secrets(
        enabled=True, project_id="p", auto_install=False,
    )
    assert not result.ok
    assert "bad token" in result.error


# ---------------------------------------------------------------------------
# env_loader integration
# ---------------------------------------------------------------------------


def test_env_loader_skips_when_disabled(tmp_path, monkeypatch):
    """No config.yaml present → no BSM call, no crash."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    from hermes_cli.env_loader import _apply_external_secret_sources
    # Should be a no-op (returns None).
    assert _apply_external_secret_sources(home) is None


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


def test_disk_cache_written_after_first_fetch(monkeypatch, tmp_path):
    """First fetch hits bws AND writes a 0600 file under hermes_home/cache/."""
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

    secrets, _ = bw.fetch_bitwarden_secrets(
        access_token="0.t", project_id="proj-1", binary=fake_binary,
        cache_ttl_seconds=300, home_path=home,
    )
    assert secrets == {"K1": "v1"}
    assert call_count["n"] == 1

    cache_path = bw._disk_cache_path(home)
    assert cache_path.exists()
    # Mode must be 0600 — disk cache contains plaintext secret values
    mode = os.stat(cache_path).st_mode & 0o777
    assert mode == 0o600, f"expected 0o600, got 0o{mode:o}"

    # File contents: key (fingerprint not raw token), secrets dict, fetched_at
    payload_disk = json.loads(cache_path.read_text())
    assert set(payload_disk.keys()) == {"key", "secrets", "fetched_at"}
    assert payload_disk["secrets"] == {"K1": "v1"}
    # Critically, the raw access token must NOT appear anywhere in the file
    assert "0.t" not in cache_path.read_text()


def test_disk_cache_short_circuits_bws_when_fresh(monkeypatch, tmp_path):
    """Second fetch (different process simulation) skips bws entirely."""
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

    # First call: hits bws, populates disk cache
    bw.fetch_bitwarden_secrets(
        access_token="0.t", project_id="proj-1", binary=fake_binary,
        cache_ttl_seconds=300, home_path=home,
    )
    assert call_count["n"] == 1

    # Clear ONLY the in-process cache to simulate a fresh subprocess.
    bw._CACHE.clear()

    secrets2, _ = bw.fetch_bitwarden_secrets(
        access_token="0.t", project_id="proj-1", binary=fake_binary,
        cache_ttl_seconds=300, home_path=home,
    )
    assert secrets2 == {"K1": "v1"}
    # Critical: bws was NOT invoked the second time
    assert call_count["n"] == 1


def test_disk_cache_expires_with_ttl(monkeypatch, tmp_path):
    """Stale disk cache (older than ttl) triggers a refetch."""
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

    # First call
    bw.fetch_bitwarden_secrets(
        access_token="0.t", project_id="proj-1", binary=fake_binary,
        cache_ttl_seconds=300, home_path=home,
    )
    assert call_count["n"] == 1

    # Backdate the disk cache so the TTL window has passed
    cache_path = bw._disk_cache_path(home)
    payload_disk = json.loads(cache_path.read_text())
    payload_disk["fetched_at"] = time.time() - 10_000
    cache_path.write_text(json.dumps(payload_disk))
    bw._CACHE.clear()

    # Second call: stale disk → refetch
    bw.fetch_bitwarden_secrets(
        access_token="0.t", project_id="proj-1", binary=fake_binary,
        cache_ttl_seconds=300, home_path=home,
    )
    assert call_count["n"] == 2


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


def test_disk_cache_use_cache_false_skips_disk(monkeypatch, tmp_path):
    """use_cache=False must skip BOTH in-process and disk caches."""
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

    # First call WITH cache populates disk
    bw.fetch_bitwarden_secrets(
        access_token="0.t", project_id="proj-1", binary=fake_binary,
        cache_ttl_seconds=300, use_cache=True, home_path=home,
    )
    assert call_count["n"] == 1
    bw._CACHE.clear()

    # Second call with use_cache=False MUST hit bws again even though disk is fresh
    bw.fetch_bitwarden_secrets(
        access_token="0.t", project_id="proj-1", binary=fake_binary,
        cache_ttl_seconds=300, use_cache=False, home_path=home,
    )
    assert call_count["n"] == 2


def test_disk_cache_corrupt_file_falls_through(monkeypatch, tmp_path):
    """A garbage cache file must NOT crash startup — we refetch."""
    home = tmp_path / ".hermes"
    home.mkdir()
    fake_binary = tmp_path / "bws"
    fake_binary.write_text("")
    payload = _fake_bws_payload([{"key": "K1", "value": "v1"}])

    monkeypatch.setattr(
        bw.subprocess, "run",
        lambda *a, **kw: mock.Mock(returncode=0, stdout=payload, stderr=""),
    )
    bw._reset_cache_for_tests(home)

    # Write a corrupt cache file
    cache_path = bw._disk_cache_path(home)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("not json {{{")

    secrets, _ = bw.fetch_bitwarden_secrets(
        access_token="0.t", project_id="proj-1", binary=fake_binary,
        cache_ttl_seconds=300, home_path=home,
    )
    # Refetched cleanly
    assert secrets == {"K1": "v1"}
    # And the corrupt file was replaced with a valid one
    assert json.loads(cache_path.read_text())["secrets"] == {"K1": "v1"}


def test_reset_cache_for_tests_deletes_disk_file(tmp_path):
    """_reset_cache_for_tests(home_path) must also clean disk."""
    home = tmp_path / ".hermes"
    home.mkdir()
    cache_path = bw._disk_cache_path(home)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("{}")
    assert cache_path.exists()

    bw._reset_cache_for_tests(home)
    assert not cache_path.exists()
    # Idempotent
    bw._reset_cache_for_tests(home)
