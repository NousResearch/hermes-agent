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
import tempfile
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


def test_install_bws_checksum_mismatch(hermes_home, monkeypatch):
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
    def fake_apply(**kwargs):
        called["n"] += 1
        assert kwargs["enabled"] is True
        assert kwargs["project_id"] == "proj-1"
        os.environ["MY_BSM_KEY"] = "from-bsm"
        return bw.FetchResult(
            secrets={"MY_BSM_KEY": "from-bsm"},
            applied=["MY_BSM_KEY"],
        )

    monkeypatch.setattr(
        "agent.secret_sources.bitwarden.apply_bitwarden_secrets",
        fake_apply,
    )

    from hermes_cli.env_loader import _apply_external_secret_sources
    _apply_external_secret_sources(home)

    assert called["n"] == 1
    assert os.environ.get("MY_BSM_KEY") == "from-bsm"
