"""Hermetic tests for the iron-proxy egress integration.

Covers the pure-function surface (token mint, mapping discovery, config build,
config + mappings I/O), the binary install path (HTTP downloads + tar
extraction + checksum verification fully mocked), the subprocess lifecycle
(spawn / PID / pid_alive / stop, with subprocess.Popen mocked), and the
docker backend's egress arg builder.

Live network and the real ``iron-proxy`` binary are NEVER touched.  See
``tests/test_iron_proxy_e2e.py`` (gated behind a marker) for the real-binary
smoke test.
"""

from __future__ import annotations

import io
import os
import sys
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.proxy_sources import iron_proxy as ip


# ---------------------------------------------------------------------------
# Per-test isolation
# ---------------------------------------------------------------------------


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Point HERMES_HOME at a temp dir so install paths don't touch the real $HOME."""

    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Make sure no stale provider keys influence discovery.
    for key in list(os.environ):
        if key.endswith("_API_KEY"):
            monkeypatch.delenv(key, raising=False)
    return home


# ---------------------------------------------------------------------------
# Token mint + mapping discovery
# ---------------------------------------------------------------------------


def test_mint_proxy_token_has_prefix_and_length():
    t = ip.mint_proxy_token("alpha")
    assert t.startswith("alpha-")
    assert len(t) >= len("alpha-") + 32


def test_mint_proxy_token_is_random():
    a = ip.mint_proxy_token("x")
    b = ip.mint_proxy_token("x")
    assert a != b


def test_discover_provider_mappings_from_env(hermes_home, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-real-1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-real-2")
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    ms = ip.discover_provider_mappings()
    names = [m.real_env_name for m in ms]
    assert "OPENROUTER_API_KEY" in names
    assert "OPENAI_API_KEY" in names
    assert "MISTRAL_API_KEY" not in names


def test_discover_provider_mappings_explicit_names(hermes_home):
    ms = ip.discover_provider_mappings(
        available_env_names=["OPENROUTER_API_KEY", "GROQ_API_KEY", "UNKNOWN_KEY"]
    )
    names = {m.real_env_name for m in ms}
    assert names == {"OPENROUTER_API_KEY", "GROQ_API_KEY"}
    # Unknown providers (no entry in _BEARER_PROVIDERS) are skipped, not warned.


def test_discover_provider_mappings_empty(hermes_home):
    ms = ip.discover_provider_mappings(available_env_names=[])
    assert ms == []


# ---------------------------------------------------------------------------
# Config / mapping serialization
# ---------------------------------------------------------------------------


def _sample_mapping(env_name: str = "OPENROUTER_API_KEY") -> ip.TokenMapping:
    return ip.TokenMapping(
        proxy_token=ip.mint_proxy_token("test"),
        real_env_name=env_name,
        upstream_hosts=("openrouter.ai", "*.openrouter.ai"),
    )


def test_build_proxy_config_shape(tmp_path):
    m = _sample_mapping()
    ca_crt = tmp_path / "ca.crt"
    ca_key = tmp_path / "ca.key"
    cfg = ip.build_proxy_config(
        mappings=[m],
        ca_cert=ca_crt,
        ca_key=ca_key,
    )
    # Top-level sections — note `dns` is required by iron-proxy even when
    # we only use the CONNECT tunnel.
    assert set(cfg.keys()) >= {"dns", "proxy", "tls", "transforms", "log"}
    # Transforms in expected order
    assert [t["name"] for t in cfg["transforms"]] == ["allowlist", "secrets"]
    # Allowlist uses `domains:` (iron-proxy schema), not `hosts:`
    domains = cfg["transforms"][0]["config"]["domains"]
    assert "openrouter.ai" in domains
    # Secrets transform encodes our mapping
    rules = cfg["transforms"][1]["config"]["secrets"]
    assert len(rules) == 1
    rule = rules[0]
    # Real secret value is sourced from env at egress time, NOT inlined.
    assert rule["source"] == {"type": "env", "var": "OPENROUTER_API_KEY"}
    # The proxy token is the replacement target.
    assert rule["replace"]["proxy_value"] == m.proxy_token
    assert "Authorization" in rule["replace"]["match_headers"]
    # Rules list contains one entry per upstream host.
    rule_hosts = {r["host"] for r in rule["rules"]}
    assert rule_hosts == set(m.upstream_hosts)
    # TLS section names the CA paths
    assert cfg["tls"]["ca_cert"] == str(ca_crt)


def test_build_proxy_config_custom_allowed_hosts(tmp_path):
    m = _sample_mapping("OPENAI_API_KEY")
    cfg = ip.build_proxy_config(
        mappings=[m],
        ca_cert=tmp_path / "ca.crt",
        ca_key=tmp_path / "ca.key",
        allowed_hosts=["custom-host.test"],
    )
    domains = cfg["transforms"][0]["config"]["domains"]
    # Custom allowed_hosts wins as the base; mapping's hosts get appended.
    assert "custom-host.test" in domains
    assert "openrouter.ai" in domains  # comes from the mapping


# ---------------------------------------------------------------------------
# Default SSRF deny list (regression: docs promise cloud metadata is denied)
# ---------------------------------------------------------------------------


def test_default_deny_cidrs_present_when_unspecified(tmp_path):
    """build_proxy_config must emit the default deny list when the caller
    passes nothing.  The IMDS subnet (169.254.0.0/16) MUST be in the result
    or the docs claim that ``upstream_deny_cidrs`` refuses cloud metadata
    is a lie."""

    cfg = ip.build_proxy_config(
        mappings=[_sample_mapping()],
        ca_cert=tmp_path / "ca.crt",
        ca_key=tmp_path / "ca.key",
    )
    deny = cfg["proxy"]["upstream_deny_cidrs"]
    assert "169.254.0.0/16" in deny  # IMDS
    assert "127.0.0.0/8" in deny      # loopback v4
    assert "::1/128" in deny           # loopback v6
    assert "10.0.0.0/8" in deny        # RFC1918
    assert "172.16.0.0/12" in deny     # RFC1918
    assert "192.168.0.0/16" in deny    # RFC1918


def test_explicit_empty_deny_cidrs_disables_default(tmp_path):
    """Explicit ``[]`` opts out of the default deny list — needed by
    hermetic tests that want to talk to a loopback upstream."""

    cfg = ip.build_proxy_config(
        mappings=[_sample_mapping()],
        ca_cert=tmp_path / "ca.crt",
        ca_key=tmp_path / "ca.key",
        upstream_deny_cidrs=[],
    )
    assert cfg["proxy"]["upstream_deny_cidrs"] == []


def test_wizard_rendered_yaml_contains_deny_list(hermes_home, tmp_path):
    """End-to-end: cmd_setup writes proxy.yaml; the rendered file must
    contain the deny list because the wizard now passes the operator's
    config-level setting (None → default) through to build_proxy_config."""

    # Simulate the wizard's call shape (matches proxy_cli.cmd_setup).
    state = ip._proxy_state_dir()
    (state / "ca.crt").write_text("fake-ca")
    (state / "ca.key").write_text("fake-key")
    proxy_yaml = ip.build_proxy_config(
        mappings=[_sample_mapping()],
        ca_cert=state / "ca.crt",
        ca_key=state / "ca.key",
        # The wizard passes ``upstream_deny_cidrs`` from the config; when
        # the operator hasn't set anything, that's None and we get the
        # safe default below.
        upstream_deny_cidrs=None,
    )
    out = ip.write_proxy_config(proxy_yaml)
    text = out.read_text(encoding="utf-8")
    assert "169.254.0.0/16" in text


# ---------------------------------------------------------------------------
# Bind policy (regression: must not bind 0.0.0.0)
# ---------------------------------------------------------------------------


def test_default_bind_is_loopback_not_zero_zero(tmp_path):
    """The sandbox-facing listeners must NOT be ``0.0.0.0:PORT`` or
    ``:PORT`` (latter is INADDR_ANY)."""

    cfg = ip.build_proxy_config(
        mappings=[_sample_mapping()],
        ca_cert=tmp_path / "ca.crt",
        ca_key=tmp_path / "ca.key",
        tunnel_port=12345,
        http_listen=["127.0.0.1:12345"],  # explicit so test is deterministic
    )
    # tunnel_listen is the CONNECT/MITM listener sandboxes hit via
    # HTTPS_PROXY; http_listen is the plain-HTTP forward on port+1.
    assert cfg["proxy"]["tunnel_listen"] == "127.0.0.1:12345"
    assert cfg["proxy"]["http_listen"] == "127.0.0.1:12346"
    for key in ("tunnel_listen", "http_listen", "https_listen"):
        val = cfg["proxy"][key]
        # Sentinel: confirm we didn't accidentally serialize a bare-port
        # form like ":12345" (that's INADDR_ANY).
        assert not val.startswith(":")
        assert "0.0.0.0" not in val
    # iron-proxy v0.39 doesn't support http_listens (plural).  We
    # deliberately do NOT emit that key — re-emitting it would cause
    # the daemon to fail YAML unmarshal at start time.
    assert "http_listens" not in cfg["proxy"], (
        "iron-proxy v0.39 rejects http_listens (plural); only the "
        "singular http_listen string is accepted by the binary"
    )


def test_default_bind_uses_docker_bridge_on_linux(tmp_path, monkeypatch):
    """When http_listen isn't passed AND we're on Linux, the singular
    http_listen field is the DOCKER BRIDGE bind — not loopback.
    iron-proxy v0.39 only supports one bind per daemon process, and on
    Linux ``host.docker.internal:host-gateway`` resolves to the bridge
    gateway (172.17.0.1 by default), which a loopback-only daemon never
    answers.  Sandboxes must be able to reach the proxy from the
    container's vantage point."""

    monkeypatch.setattr(ip.platform, "system", lambda: "Linux")
    monkeypatch.setattr(ip, "_detect_docker_bridge_ip", lambda: "172.17.0.1")
    cfg = ip.build_proxy_config(
        mappings=[_sample_mapping()],
        ca_cert=tmp_path / "ca.crt",
        ca_key=tmp_path / "ca.key",
        tunnel_port=9090,
    )
    # The sandbox-facing CONNECT/MITM listener binds the bridge gateway —
    # reachable from containers via host.docker.internal.  Plain-HTTP
    # forward listener rides on port+1, same host.
    assert cfg["proxy"]["tunnel_listen"] == "172.17.0.1:9090"
    assert cfg["proxy"]["http_listen"] == "172.17.0.1:9091"
    # No http_listens (plural) — v0.39 rejects that key.
    assert "http_listens" not in cfg["proxy"]


def test_default_bind_falls_back_to_loopback_without_bridge(tmp_path, monkeypatch):
    """On Linux without a detectable docker0 bridge (docker not
    installed / not running), fall back to loopback rather than
    refusing to bind."""

    monkeypatch.setattr(ip.platform, "system", lambda: "Linux")
    monkeypatch.setattr(ip, "_detect_docker_bridge_ip", lambda: None)
    cfg = ip.build_proxy_config(
        mappings=[_sample_mapping()],
        ca_cert=tmp_path / "ca.crt",
        ca_key=tmp_path / "ca.key",
        tunnel_port=9090,
    )
    assert cfg["proxy"]["tunnel_listen"] == "127.0.0.1:9090"
    assert cfg["proxy"]["http_listen"] == "127.0.0.1:9091"


def test_default_bind_is_loopback_on_macos(tmp_path, monkeypatch):
    """On Docker Desktop platforms host.docker.internal routes to the
    host via VPNkit, so loopback is reachable from containers and is
    the least-exposed bind."""

    monkeypatch.setattr(ip.platform, "system", lambda: "Darwin")
    cfg = ip.build_proxy_config(
        mappings=[_sample_mapping()],
        ca_cert=tmp_path / "ca.crt",
        ca_key=tmp_path / "ca.key",
        tunnel_port=9090,
    )
    assert cfg["proxy"]["tunnel_listen"] == "127.0.0.1:9090"
    assert cfg["proxy"]["http_listen"] == "127.0.0.1:9091"


def test_metrics_listener_pinned_to_loopback_ephemeral(tmp_path):
    """iron-proxy v0.39's default metrics_listen is ``:9090``, which
    collides with our default tunnel_port=9090.  build_proxy_config MUST
    explicitly pin metrics.listen to ``127.0.0.1:0`` so the bind
    collision can never happen at start time."""

    cfg = ip.build_proxy_config(
        mappings=[_sample_mapping()],
        ca_cert=tmp_path / "ca.crt",
        ca_key=tmp_path / "ca.key",
        tunnel_port=9090,
    )
    assert cfg["metrics"]["listen"] == "127.0.0.1:0"


# ---------------------------------------------------------------------------
# audit_log file pre-creation (parameter still accepted; v0.39 doesn't
# wire it into the binary config but ensure_audit_log() still creates
# the file at 0o600 as a logrotate / monitoring sentinel)
# ---------------------------------------------------------------------------


def test_audit_log_kwarg_does_not_inject_audit_path_v039(tmp_path):
    """v0.39 of iron-proxy rejects ``log.audit_path`` (not a struct
    field).  build_proxy_config still accepts the audit_log kwarg for
    forward compatibility but MUST NOT emit it into the rendered yaml
    until the upstream binary supports it.  See the kwarg's docstring
    for the upgrade path."""

    cfg = ip.build_proxy_config(
        mappings=[_sample_mapping()],
        ca_cert=tmp_path / "ca.crt",
        ca_key=tmp_path / "ca.key",
        audit_log=tmp_path / "audit.log",
    )
    assert "audit_path" not in cfg["log"], (
        "iron-proxy v0.39 has no log.audit_path field; emitting it "
        "causes 'field audit_path not found in type config.Log' at "
        "daemon start.  ensure_audit_log() still creates the file as "
        "an operator-facing logrotate target."
    )


def test_audit_log_omitted_when_caller_passes_none(tmp_path):
    cfg = ip.build_proxy_config(
        mappings=[_sample_mapping()],
        ca_cert=tmp_path / "ca.crt",
        ca_key=tmp_path / "ca.key",
        audit_log=None,
    )
    assert "audit_path" not in cfg["log"]


def test_write_and_load_mappings_roundtrip(hermes_home):
    ms = [_sample_mapping("OPENROUTER_API_KEY"), _sample_mapping("OPENAI_API_KEY")]
    path = ip.write_mappings(ms)
    assert path.exists()
    loaded = ip.load_mappings()
    assert len(loaded) == 2
    assert {m.real_env_name for m in loaded} == {"OPENROUTER_API_KEY", "OPENAI_API_KEY"}
    # Tokens preserved
    assert loaded[0].proxy_token == ms[0].proxy_token


def test_load_mappings_handles_missing_file(hermes_home):
    assert ip.load_mappings() == []


def test_load_mappings_handles_corrupt_json(hermes_home):
    state = ip._proxy_state_dir()
    (state / "mappings.json").write_text("{not json", encoding="utf-8")
    assert ip.load_mappings() == []


def test_write_proxy_config_serializes_yaml(hermes_home, tmp_path):
    ca_crt = tmp_path / "ca.crt"
    ca_key = tmp_path / "ca.key"
    cfg = ip.build_proxy_config(
        mappings=[_sample_mapping()],
        ca_cert=ca_crt,
        ca_key=ca_key,
    )
    out = ip.write_proxy_config(cfg)
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "tunnel_listen" in text
    assert f"ca_cert: {ca_crt}" in text


# ---------------------------------------------------------------------------
# Token-preservation on re-setup (regression: clobbered live sandboxes)
# ---------------------------------------------------------------------------


def test_merge_mappings_preserves_existing_tokens():
    """Re-running setup must not invalidate tokens baked into already-
    running sandboxes.  ``merge_mappings`` keeps the prior token for any
    provider that's in both lists."""

    existing = [
        ip.TokenMapping(
            proxy_token="hermes-proxy-original-12345",
            real_env_name="OPENROUTER_API_KEY",
            upstream_hosts=("openrouter.ai",),
        ),
    ]
    discovered = ip.discover_provider_mappings(
        available_env_names=["OPENROUTER_API_KEY", "OPENAI_API_KEY"]
    )
    merged = ip.merge_mappings(existing=existing, discovered=discovered)
    by_name = {m.real_env_name: m for m in merged}
    # Original token preserved.
    assert by_name["OPENROUTER_API_KEY"].proxy_token == "hermes-proxy-original-12345"
    # New provider got a fresh token.
    assert by_name["OPENAI_API_KEY"].proxy_token != "hermes-proxy-original-12345"
    # Both providers in the result.
    assert set(by_name) == {"OPENROUTER_API_KEY", "OPENAI_API_KEY"}


def test_merge_mappings_drops_providers_removed_from_env():
    """When a provider is in `existing` but not in `discovered`, it must
    be dropped from the result — the operator removed the env var."""

    existing = [
        ip.TokenMapping(
            proxy_token="stale", real_env_name="OPENROUTER_API_KEY",
            upstream_hosts=("openrouter.ai",),
        ),
    ]
    discovered = ip.discover_provider_mappings(
        available_env_names=["OPENAI_API_KEY"]
    )
    merged = ip.merge_mappings(existing=existing, discovered=discovered)
    names = {m.real_env_name for m in merged}
    assert names == {"OPENAI_API_KEY"}


def test_merge_mappings_rotate_mints_fresh_tokens():
    """``rotate=True`` rolls every token regardless of overlap.  The
    --rotate-tokens flag uses this."""

    existing = [
        ip.TokenMapping(
            proxy_token="hermes-proxy-original-12345",
            real_env_name="OPENROUTER_API_KEY",
            upstream_hosts=("openrouter.ai",),
        ),
    ]
    discovered = ip.discover_provider_mappings(
        available_env_names=["OPENROUTER_API_KEY"]
    )
    merged = ip.merge_mappings(existing=existing, discovered=discovered, rotate=True)
    assert merged[0].proxy_token != "hermes-proxy-original-12345"


# ---------------------------------------------------------------------------
# Uncovered provider detection (regression: non-bearer providers bypass)
# ---------------------------------------------------------------------------


def test_uncovered_providers_detects_anthropic_aws(hermes_home, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
    uncovered = ip.discover_uncovered_providers()
    assert "ANTHROPIC_API_KEY" in uncovered
    assert "AWS_ACCESS_KEY_ID" in uncovered


def test_uncovered_providers_explicit_names_empty():
    assert ip.discover_uncovered_providers(available_env_names=[]) == []


def test_uncovered_providers_skips_bearer_providers(hermes_home, monkeypatch):
    """OPENROUTER_API_KEY etc. are bearer providers — they should NOT
    appear in the uncovered list."""

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    uncovered = ip.discover_uncovered_providers()
    assert "OPENROUTER_API_KEY" not in uncovered


# ---------------------------------------------------------------------------
# Binary discovery + lazy install
# ---------------------------------------------------------------------------


def test_find_iron_proxy_returns_none_when_missing(hermes_home, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert ip.find_iron_proxy(install_if_missing=False) is None


def test_find_iron_proxy_returns_managed_first(hermes_home, monkeypatch):
    managed = ip._hermes_bin_dir() / ip._platform_binary_name()
    managed.parent.mkdir(parents=True, exist_ok=True)
    managed.write_bytes(b"#!/bin/sh\necho ok\n")
    managed.chmod(0o755)
    # Even with a system one on PATH, the managed copy should win.
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/iron-proxy")
    assert ip.find_iron_proxy() == managed


def _make_fake_tar(binary_name: str, payload: bytes = b"#!/bin/sh\necho ok\n") -> bytes:
    """Build a tar.gz with one file at the root, named ``binary_name``."""

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name=binary_name)
        info.size = len(payload)
        info.mode = 0o755
        tf.addfile(info, io.BytesIO(payload))
    return buf.getvalue()


def test_install_iron_proxy_verifies_checksum_and_extracts(hermes_home, monkeypatch):
    fake_payload = _make_fake_tar(ip._platform_binary_name())
    import hashlib

    expected_sha = hashlib.sha256(fake_payload).hexdigest()
    asset_name = ip._platform_asset_name()
    checksum_text = f"{expected_sha}  {asset_name}\nffff  other-asset.tar.gz\n"

    def fake_download(url: str, dest: Path) -> None:
        if url.endswith(ip._IRON_PROXY_CHECKSUM_NAME):
            dest.write_text(checksum_text)
        else:
            dest.write_bytes(fake_payload)

    monkeypatch.setattr(ip, "_http_download", fake_download)
    target = ip.install_iron_proxy()
    assert target.exists()
    assert target.read_bytes() == b"#!/bin/sh\necho ok\n"
    # Executable bit is set
    assert os.access(target, os.X_OK)


def test_install_iron_proxy_rejects_bad_checksum(hermes_home, monkeypatch):
    fake_payload = _make_fake_tar(ip._platform_binary_name())
    asset_name = ip._platform_asset_name()
    bad_text = f"deadbeef  {asset_name}\n"

    def fake_download(url: str, dest: Path) -> None:
        if url.endswith(ip._IRON_PROXY_CHECKSUM_NAME):
            dest.write_text(bad_text)
        else:
            dest.write_bytes(fake_payload)

    monkeypatch.setattr(ip, "_http_download", fake_download)
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        ip.install_iron_proxy()


def test_install_iron_proxy_rejects_missing_checksum_entry(hermes_home, monkeypatch):
    fake_payload = _make_fake_tar(ip._platform_binary_name())

    def fake_download(url: str, dest: Path) -> None:
        if url.endswith(ip._IRON_PROXY_CHECKSUM_NAME):
            dest.write_text("aaaa  some-other-file.tar.gz\n")
        else:
            dest.write_bytes(fake_payload)

    monkeypatch.setattr(ip, "_http_download", fake_download)
    with pytest.raises(RuntimeError, match="No checksum entry"):
        ip.install_iron_proxy()


def test_pick_tar_member_rejects_path_traversal():
    """A malicious tar that escapes via '..' must be refused."""

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name="../iron-proxy")
        info.size = 1
        info.mode = 0o755
        tf.addfile(info, io.BytesIO(b"x"))
    buf.seek(0)
    with tarfile.open(fileobj=buf, mode="r:gz") as tf:
        with pytest.raises(RuntimeError, match="Could not find iron-proxy"):
            ip._pick_tar_member(tf, "iron-proxy")


# ---------------------------------------------------------------------------
# Subprocess lifecycle
# ---------------------------------------------------------------------------


def test_get_status_when_nothing_configured(hermes_home):
    status = ip.get_status()
    assert status.binary_path is None
    assert status.config_path is None
    assert status.ca_cert_path is None
    assert status.pid is None
    assert status.listening is False
    assert not status.installed
    assert not status.configured


def test_get_status_with_config_present(hermes_home, monkeypatch):
    # Materialize binary, config, and ca cert.
    bin_path = ip._hermes_bin_dir() / ip._platform_binary_name()
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    bin_path.write_bytes(b"")
    bin_path.chmod(0o755)
    state = ip._proxy_state_dir()
    (state / "ca.crt").write_text("fake")
    cfg = ip.build_proxy_config(
        mappings=[_sample_mapping()],
        ca_cert=state / "ca.crt",
        ca_key=state / "ca.key",
        tunnel_port=9999,
    )
    ip.write_proxy_config(cfg)
    monkeypatch.setattr(ip, "iron_proxy_version", lambda b: "iron-proxy v0.0.0-test")

    status = ip.get_status()
    assert status.installed
    assert status.configured
    assert status.tunnel_port == 9999
    assert "test" in (status.binary_version or "")


def test_stop_proxy_handles_missing_pidfile(hermes_home):
    # No pidfile → stop returns False, doesn't raise.
    assert ip.stop_proxy() is False


def test_stop_proxy_cleans_stale_pidfile(hermes_home, monkeypatch):
    pid_file = ip._proxy_state_dir() / "iron-proxy.pid"
    pid_file.write_text("999999999")
    monkeypatch.setattr(ip, "_pid_alive", lambda pid: False)
    assert ip.stop_proxy() is False
    assert not pid_file.exists()


def test_start_proxy_refuses_without_binary(hermes_home, monkeypatch):
    # No binary, auto_install fails → RuntimeError surfaces.
    monkeypatch.setattr(ip, "find_iron_proxy", lambda **kwargs: None)
    state = ip._proxy_state_dir()
    (state / "proxy.yaml").write_text("proxy: {}")
    with pytest.raises(RuntimeError, match="binary not available"):
        ip.start_proxy()


def test_start_proxy_refuses_without_config(hermes_home, monkeypatch):
    binary = ip._hermes_bin_dir() / ip._platform_binary_name()
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(b"")
    binary.chmod(0o755)
    monkeypatch.setattr(ip, "find_iron_proxy", lambda **kwargs: binary)
    with pytest.raises(RuntimeError, match="config not found"):
        ip.start_proxy()


def test_start_proxy_writes_pidfile_when_alive(hermes_home, monkeypatch):
    binary = ip._hermes_bin_dir() / ip._platform_binary_name()
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(b"")
    binary.chmod(0o755)
    state = ip._proxy_state_dir()
    (state / "proxy.yaml").write_text("proxy: {}")

    monkeypatch.setattr(ip, "find_iron_proxy", lambda **kwargs: binary)
    monkeypatch.setattr(ip, "_STARTUP_GRACE_SECONDS", 0)

    # Pre-stub everything start_proxy's get_status() call will touch — it
    # runs INSIDE start_proxy, so by the time Popen is mocked these have
    # to already be hermetic.
    monkeypatch.setattr(ip, "_pid_alive", lambda pid: True)
    # v3: start_proxy now REQUIRES _port_listening to return True before
    # writing the pidfile.  The previous version of this test mocked it
    # to False and relied on the loop falling through; the new code
    # treats fall-through as failure and kills the child.
    monkeypatch.setattr(ip, "_port_listening", lambda h, p: True)
    monkeypatch.setattr(ip, "iron_proxy_version", lambda b: "iron-proxy test")

    fake_proc = MagicMock()
    fake_proc.pid = 4242
    fake_proc.poll.return_value = None  # still alive

    with patch("subprocess.Popen", lambda *a, **k: fake_proc):
        status = ip.start_proxy()
    assert (state / "iron-proxy.pid").read_text() == "4242"
    assert status.pid == 4242


def test_start_proxy_raises_when_immediate_exit(hermes_home, monkeypatch):
    binary = ip._hermes_bin_dir() / ip._platform_binary_name()
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(b"")
    binary.chmod(0o755)
    state = ip._proxy_state_dir()
    (state / "proxy.yaml").write_text("proxy: {}")
    (state / "iron-proxy.log").write_text("bind: address already in use\n")

    monkeypatch.setattr(ip, "find_iron_proxy", lambda **kwargs: binary)
    monkeypatch.setattr(ip, "_STARTUP_GRACE_SECONDS", 0)

    fake_proc = MagicMock()
    fake_proc.pid = 5151
    fake_proc.poll.return_value = 1  # exited immediately
    fake_proc.returncode = 1
    with patch("subprocess.Popen", lambda *a, **k: fake_proc):
        with pytest.raises(RuntimeError, match="exited immediately"):
            ip.start_proxy()


def test_start_proxy_idempotent_when_already_running(hermes_home, monkeypatch):
    state = ip._proxy_state_dir()
    pid_file = state / "iron-proxy.pid"
    pid_file.write_text("12345")
    monkeypatch.setattr(ip, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(ip, "_port_listening", lambda h, p: True)
    monkeypatch.setattr(ip, "iron_proxy_version", lambda b: "test")
    # Materialize config so we get past that check (we shouldn't reach it,
    # but if the idempotent path regresses we want a clean failure mode).
    (state / "proxy.yaml").write_text("proxy: {}")
    # Sentinel: subprocess.Popen must NOT be called.
    with patch("subprocess.Popen", lambda *a, **k: pytest.fail("should not spawn")):
        status = ip.start_proxy()
    # Should return without spawning anything.
    assert status is not None


# ---------------------------------------------------------------------------
# Docker integration
# ---------------------------------------------------------------------------


def test_docker_egress_args_empty_when_disabled(hermes_home, monkeypatch):
    from tools.environments.docker import _egress_proxy_args_for_docker

    # Default config has proxy.enabled=False; helper should return all empties.
    vol, env, host = _egress_proxy_args_for_docker()
    assert vol == []
    assert env == {}
    assert host == []


def test_docker_egress_args_when_enabled_but_unconfigured_raises(hermes_home, monkeypatch):
    from tools.environments.docker import _egress_proxy_args_for_docker
    from hermes_cli.config import load_config, save_config

    cfg = load_config()
    cfg.setdefault("proxy", {})["enabled"] = True
    cfg["proxy"]["enforce_on_docker"] = True
    save_config(cfg)

    # No proxy.yaml exists → enforce_on_docker should raise.
    with pytest.raises(RuntimeError, match="not configured"):
        _egress_proxy_args_for_docker()


def test_docker_egress_args_when_unconfigured_no_enforce(hermes_home, monkeypatch):
    from tools.environments.docker import _egress_proxy_args_for_docker
    from hermes_cli.config import load_config, save_config

    cfg = load_config()
    cfg.setdefault("proxy", {})["enabled"] = True
    cfg["proxy"]["enforce_on_docker"] = False
    save_config(cfg)

    # Without enforcement, missing config returns empties (warning only).
    vol, env, host = _egress_proxy_args_for_docker()
    assert vol == []
    assert env == {}
    assert host == []


def test_docker_egress_args_full_path(hermes_home, monkeypatch):
    """Wire up everything (config, CA, mappings, fake running proxy) and
    verify the docker helper emits the right mounts and env."""

    from tools.environments.docker import _egress_proxy_args_for_docker
    from hermes_cli.config import load_config, save_config

    # Materialize config, CA, mappings.
    state = ip._proxy_state_dir()
    ca = state / "ca.crt"
    ca.write_text("fake-ca")
    (state / "ca.key").write_text("fake-key")
    mapping = _sample_mapping("OPENROUTER_API_KEY")
    proxy_cfg = ip.build_proxy_config(
        mappings=[mapping], ca_cert=ca, ca_key=state / "ca.key", tunnel_port=9090,
    )
    ip.write_proxy_config(proxy_cfg)
    ip.write_mappings([mapping])

    cfg = load_config()
    cfg.setdefault("proxy", {})["enabled"] = True
    cfg["proxy"]["enforce_on_docker"] = True
    save_config(cfg)

    # Fake running proxy.
    (state / "iron-proxy.pid").write_text("99999")
    monkeypatch.setattr(ip, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(ip, "_port_listening", lambda h, p: True)

    vol, env, host = _egress_proxy_args_for_docker()
    # CA mount present and in -v form
    assert "-v" in vol
    assert any("hermes-egress-ca.crt" in arg for arg in vol)
    # Env contains both casings of HTTPS_PROXY and the CA env vars
    assert env["HTTPS_PROXY"].endswith(":9090")
    assert env["https_proxy"] == env["HTTPS_PROXY"]
    assert env["REQUESTS_CA_BUNDLE"].endswith("hermes-egress-ca.crt")
    assert env["NODE_EXTRA_CA_CERTS"] == env["REQUESTS_CA_BUNDLE"]
    # NO_PROXY excludes loopback
    assert "127.0.0.1" in env["NO_PROXY"]
    # Per-mapping proxy token is surfaced under both the standard provider env
    # name (so existing SDKs work without egress-specific code) and the
    # introspection name.
    assert env["OPENROUTER_API_KEY"] == mapping.proxy_token
    assert env["HERMES_PROXY_TOKEN_OPENROUTER_API_KEY"] == mapping.proxy_token
    # Linux host-gateway mapping
    assert host == ["--add-host", "host.docker.internal:host-gateway"]


def test_docker_egress_fingerprint_changes_with_tokens(hermes_home, monkeypatch):
    """Persistent Docker container reuse must not attach to a container that
    was created before egress, before a token rotation, or with a different CA
    mount.  The label hash is what forces a fresh container in those cases."""

    from tools.environments.docker import _egress_reuse_fingerprint

    first = _egress_reuse_fingerprint(
        ["-v", "/tmp/ca:/etc/ssl/certs/hermes-egress-ca.crt:ro"],
        {"OPENROUTER_API_KEY": "token-a", "HTTPS_PROXY": "http://h:9090"},
        ["--add-host", "host.docker.internal:host-gateway"],
    )
    second = _egress_reuse_fingerprint(
        ["-v", "/tmp/ca:/etc/ssl/certs/hermes-egress-ca.crt:ro"],
        {"OPENROUTER_API_KEY": "token-b", "HTTPS_PROXY": "http://h:9090"},
        ["--add-host", "host.docker.internal:host-gateway"],
    )

    assert first
    assert second
    assert first != second


# ---------------------------------------------------------------------------
# Platform asset name resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "system,machine,expected_substring",
    [
        ("Linux", "x86_64", "linux_amd64"),
        ("Linux", "aarch64", "linux_arm64"),
        ("Darwin", "arm64", "darwin_arm64"),
        ("Darwin", "x86_64", "darwin_amd64"),
    ],
)
def test_platform_asset_name(monkeypatch, system, machine, expected_substring):
    monkeypatch.setattr("platform.system", lambda: system)
    monkeypatch.setattr("platform.machine", lambda: machine)
    assert expected_substring in ip._platform_asset_name()


def test_platform_asset_name_rejects_windows(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.setattr("platform.machine", lambda: "AMD64")
    with pytest.raises(RuntimeError, match="does not ship native Windows"):
        ip._platform_asset_name()


# ---------------------------------------------------------------------------
# Subprocess env minimization (regression: host secrets leaked to proxy)
# ---------------------------------------------------------------------------


def test_subprocess_env_strips_unrelated_secrets(hermes_home, monkeypatch):
    """``_build_proxy_subprocess_env`` must NOT carry every host secret
    over to the proxy.  /proc/<pid>/environ on the proxy would otherwise
    expose all of them to same-uid local processes."""

    # Unrelated env vars that should NOT propagate.
    monkeypatch.setenv("MY_PRIVATE_TOKEN", "should-not-leak")
    monkeypatch.setenv("DATABASE_URL", "postgres://very-private")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-very-secret")
    # Provider keys that ARE in load_mappings should propagate.
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-real")
    ip.write_mappings([_sample_mapping("OPENROUTER_API_KEY")])

    env = ip._build_proxy_subprocess_env()
    assert "MY_PRIVATE_TOKEN" not in env
    assert "DATABASE_URL" not in env
    assert "SLACK_BOT_TOKEN" not in env
    assert env.get("OPENROUTER_API_KEY") == "sk-or-real"


def test_subprocess_env_strips_proxy_recursion_vars(hermes_home, monkeypatch):
    """HTTPS_PROXY etc. in the parent env would otherwise recurse iron-proxy
    through itself (or send its traffic through a corporate proxy)."""

    monkeypatch.setenv("HTTPS_PROXY", "http://corporate:3128")
    monkeypatch.setenv("HTTP_PROXY", "http://corporate:3128")
    monkeypatch.setenv("ALL_PROXY", "socks5://corporate:1080")
    env = ip._build_proxy_subprocess_env()
    assert "HTTPS_PROXY" not in env
    assert "https_proxy" not in env
    assert "HTTP_PROXY" not in env
    assert "ALL_PROXY" not in env


def test_subprocess_env_keeps_infrastructure_vars(hermes_home, monkeypatch):
    """PATH / HOME / locale must propagate or the child can't even find
    its libs."""

    monkeypatch.setenv("PATH", "/usr/local/bin:/usr/bin")
    monkeypatch.setenv("HOME", "/home/test")
    monkeypatch.setenv("LANG", "C.UTF-8")
    env = ip._build_proxy_subprocess_env()
    assert env.get("PATH") == "/usr/local/bin:/usr/bin"
    assert env.get("HOME") == "/home/test"
    assert env.get("LANG") == "C.UTF-8"


# ---------------------------------------------------------------------------
# CA generation TOCTOU (regression: 0o600 only set AFTER copy)
# ---------------------------------------------------------------------------


def test_ca_key_created_with_0o600(hermes_home, monkeypatch):
    """The CA private key must NEVER exist on disk with default umask
    permissions, even transiently.  Fix: open with explicit mode=0o600
    so the very first byte is written under tight perms."""

    # ensure_ca_cert shells out to openssl; mock the subprocess.run calls
    # so we don't need openssl on the test host AND don't depend on its
    # output format.
    def fake_run(args, **kwargs):
        # First call: genrsa → -out is at args[-2]
        if args[1] == "genrsa":
            out = args[-2]
            Path(out).write_bytes(b"-----BEGIN RSA PRIVATE KEY-----\nfake\n-----END RSA PRIVATE KEY-----\n")
        elif args[1] == "req":
            # Find -out path
            i = args.index("-out")
            Path(args[i + 1]).write_bytes(b"-----BEGIN CERTIFICATE-----\nfake\n-----END CERTIFICATE-----\n")
        result = MagicMock()
        result.returncode = 0
        return result

    monkeypatch.setattr(ip.shutil, "which", lambda name: "/usr/bin/openssl" if name == "openssl" else None)
    monkeypatch.setattr(ip.subprocess, "run", fake_run)

    ca_crt, ca_key = ip.ensure_ca_cert()
    assert ca_key.exists()
    mode = ca_key.stat().st_mode & 0o777
    assert mode == 0o600, f"CA key has perms {oct(mode)}, expected 0o600"


# ---------------------------------------------------------------------------
# Audit log permissions (regression: depended on umask)
# ---------------------------------------------------------------------------


def test_ensure_audit_log_creates_with_0o600(hermes_home, tmp_path):
    audit = tmp_path / "audit.log"
    ip.ensure_audit_log(audit)
    assert audit.exists()
    mode = audit.stat().st_mode & 0o777
    assert mode == 0o600


def test_ensure_audit_log_tightens_existing_perms(hermes_home, tmp_path):
    audit = tmp_path / "audit.log"
    audit.write_text("preexisting content\n")
    os.chmod(audit, 0o644)
    ip.ensure_audit_log(audit)
    mode = audit.stat().st_mode & 0o777
    assert mode == 0o600


# ---------------------------------------------------------------------------
# State dir hardening (regression: world-traversable on multi-user hosts)
# ---------------------------------------------------------------------------


def test_proxy_state_dir_is_0o700(hermes_home):
    state = ip._proxy_state_dir()
    mode = state.stat().st_mode & 0o777
    assert mode == 0o700


def test_proxy_state_dir_ro_does_not_create(hermes_home):
    """_proxy_state_dir_ro is for read-only callers — it must NOT
    materialize the dir.  Pure-status code paths shouldn't have the
    side-effect of creating ~/.hermes/proxy/."""

    # Sanity: rw path creates it.
    rw = ip._proxy_state_dir()
    assert rw.exists()
    # Remove it and confirm the ro path doesn't recreate.
    import shutil as _shutil
    _shutil.rmtree(str(rw))
    assert not rw.exists()
    ro = ip._proxy_state_dir_ro()
    assert not ro.exists()
    # The path string is the same as the rw one.
    assert ro == rw


# ---------------------------------------------------------------------------
# Mappings clobber refused when corrupt (regression: silent 403s)
# ---------------------------------------------------------------------------


def test_docker_egress_args_raises_on_empty_mappings(hermes_home, monkeypatch):
    """If mappings.json is missing / corrupt / empty AND
    enforce_on_docker is true, refuse to start the sandbox rather than
    silently mounting an unusable proxy config."""

    from tools.environments.docker import _egress_proxy_args_for_docker
    from hermes_cli.config import load_config, save_config

    state = ip._proxy_state_dir()
    (state / "ca.crt").write_text("fake-ca")
    (state / "ca.key").write_text("fake-key")
    proxy_cfg = ip.build_proxy_config(
        mappings=[_sample_mapping()],
        ca_cert=state / "ca.crt", ca_key=state / "ca.key", tunnel_port=9090,
    )
    ip.write_proxy_config(proxy_cfg)
    # Note: we deliberately do NOT write mappings.json — that's the
    # bug-class this test guards against.

    cfg = load_config()
    cfg.setdefault("proxy", {})["enabled"] = True
    cfg["proxy"]["enforce_on_docker"] = True
    save_config(cfg)

    (state / "iron-proxy.pid").write_text("99999")
    monkeypatch.setattr(ip, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(ip, "_port_listening", lambda h, p: True)

    with pytest.raises(RuntimeError, match="mappings.json is empty or"):
        _egress_proxy_args_for_docker()


# ---------------------------------------------------------------------------
# CA missing → enforce_on_docker semantics (regression: silent fail-open)
# ---------------------------------------------------------------------------


def test_docker_egress_args_raises_when_ca_vanishes(hermes_home, monkeypatch):
    """status.configured was True at check time but the CA file
    disappeared between then and now (e.g. operator manually deleted
    ~/.hermes/proxy/ca.crt).  enforce_on_docker=True must refuse."""

    from tools.environments.docker import _egress_proxy_args_for_docker
    from hermes_cli.config import load_config, save_config

    state = ip._proxy_state_dir()
    ca = state / "ca.crt"
    ca.write_text("fake-ca")
    (state / "ca.key").write_text("fake-key")
    proxy_cfg = ip.build_proxy_config(
        mappings=[_sample_mapping()],
        ca_cert=ca, ca_key=state / "ca.key", tunnel_port=9090,
    )
    ip.write_proxy_config(proxy_cfg)
    ip.write_mappings([_sample_mapping()])

    cfg = load_config()
    cfg.setdefault("proxy", {})["enabled"] = True
    cfg["proxy"]["enforce_on_docker"] = True
    save_config(cfg)

    (state / "iron-proxy.pid").write_text("99999")
    monkeypatch.setattr(ip, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(ip, "_port_listening", lambda h, p: True)

    # Build a fake status: configured=True (because both path fields are
    # set), but ca_cert_path.exists() is False — simulating the race
    # where the CA file vanished between get_status() and the
    # exists() recheck inside _egress_proxy_args_for_docker.
    fake_status = ip.ProxyStatus(
        binary_path=state / "fake-bin",  # truthy
        config_path=state / "proxy.yaml",
        ca_cert_path=state / "missing-ca.crt",  # points at nonexistent path
        pid=99999,
        listening=True,
        tunnel_port=9090,
    )
    # ProxyStatus.configured returns True iff config_path AND ca_cert_path
    # both exist.  We need configured=True but the second exists() check
    # in docker.py to return False — force that by writing a placeholder
    # config_path that exists and pointing ca_cert_path at a missing file.
    (state / "proxy.yaml").write_text("# fake")
    # ProxyStatus.configured: config_path.exists() and ca_cert_path.exists().
    # Make ca_cert_path .exists() True for the configured check but the
    # explicit .exists() recheck path in docker.py reads the same Path,
    # which is missing — so we wrap.
    class _CAStub:
        """Path-like that toggles .exists() so configured=True but the
        defensive recheck in docker.py returns False."""
        _calls = 0
        def __init__(self, real: Path):
            self._real = real
        def __str__(self):
            return str(self._real)
        @property
        def parent(self):
            return self._real.parent
        def exists(self):
            type(self)._calls += 1
            # First call: configured property check → say yes.
            # Second call: docker.py defensive recheck → say no.
            return type(self)._calls == 1
    fake_status.ca_cert_path = _CAStub(state / "missing-ca.crt")  # type: ignore[assignment]
    monkeypatch.setattr(ip, "get_status", lambda: fake_status)

    with pytest.raises(RuntimeError, match="CA cert vanished"):
        _egress_proxy_args_for_docker()


# ---------------------------------------------------------------------------
# Docker env collision detection (regression: docker_env silently bypassed proxy)
# ---------------------------------------------------------------------------


def test_docker_env_collision_with_proxy_raises_when_enforce(hermes_home, monkeypatch):
    """Setting docker_env: {HTTPS_PROXY: ''} in config.yaml with
    enforce_on_docker=true must fail-loud rather than silently inverting
    the egress isolation."""

    from tools.environments.docker import DockerEnvironment
    from hermes_cli.config import load_config, save_config

    # Set up a fully-running proxy.
    state = ip._proxy_state_dir()
    ca = state / "ca.crt"
    ca.write_text("fake-ca")
    (state / "ca.key").write_text("fake-key")
    proxy_cfg = ip.build_proxy_config(
        mappings=[_sample_mapping()],
        ca_cert=ca, ca_key=state / "ca.key", tunnel_port=9090,
    )
    ip.write_proxy_config(proxy_cfg)
    ip.write_mappings([_sample_mapping()])
    cfg = load_config()
    cfg.setdefault("proxy", {})["enabled"] = True
    cfg["proxy"]["enforce_on_docker"] = True
    save_config(cfg)
    (state / "iron-proxy.pid").write_text("99999")
    monkeypatch.setattr(ip, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(ip, "_port_listening", lambda h, p: True)

    # Mock the docker availability check so we never shell out.
    monkeypatch.setattr(
        "tools.environments.docker._ensure_docker_available", lambda: None,
    )
    # Mock find_docker so the resolved docker exe isn't probed.
    monkeypatch.setattr(
        "tools.environments.docker.find_docker", lambda: "/bin/true",
    )
    # Mock subprocess.run so we don't actually run `docker run`.  We
    # only need the constructor to get past the env merge logic.
    monkeypatch.setattr(
        "tools.environments.docker.subprocess.run",
        lambda *a, **k: MagicMock(stdout="abc123\n", returncode=0),
    )
    # init_session is the second outbound subprocess we don't care about.
    monkeypatch.setattr(
        "tools.environments.docker.DockerEnvironment.init_session",
        lambda self: None,
    )

    # The collision: user sets HTTPS_PROXY to empty string in docker_env.
    with pytest.raises(RuntimeError, match="overrides egress-proxy variables"):
        DockerEnvironment(
            image="busybox",
            env={"HTTPS_PROXY": ""},  # the collision
        )


# ---------------------------------------------------------------------------
# v3 round: bridge-IP parser hardening (P1 #1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bogus_ip",
    [
        "0.0.0.0",            # INADDR_ANY — must NEVER bind here
        "127.0.0.1",          # loopback — reject (we bind loopback explicitly)
        "224.0.0.1",          # multicast
        "240.0.0.0",          # reserved
        "169.254.0.1",        # link-local / IMDS — never a real bridge
        "8.8.8.8",            # public — never a docker bridge
        "999.999.999.999",    # garbage that count(.)==3 used to accept
        "aa.bb.cc.dd",        # alpha garbage
    ],
)
def test_detect_docker_bridge_ip_rejects_dangerous(monkeypatch, bogus_ip):
    """The parser must reject anything that isn't plausibly a docker
    bridge IP.  Previously ``ip.count('.') == 3`` would accept all of
    these and re-open the LAN exposure the bind-policy fix closed."""

    fake_stdout = (
        f"3: docker0    inet {bogus_ip}/16 brd 172.17.255.255 scope global docker0\\\n"
        "       valid_lft forever preferred_lft forever"
    )

    def fake_run(cmd, **kw):
        from unittest.mock import MagicMock
        m = MagicMock()
        m.returncode = 0
        m.stdout = fake_stdout
        return m

    monkeypatch.setattr(ip.subprocess, "run", fake_run)
    assert ip._detect_docker_bridge_ip() is None


def test_detect_docker_bridge_ip_accepts_typical(monkeypatch):
    fake_stdout = (
        "3: docker0    inet 172.17.0.1/16 brd 172.17.255.255 scope global docker0\\\n"
        "       valid_lft forever preferred_lft forever"
    )

    def fake_run(cmd, **kw):
        from unittest.mock import MagicMock
        m = MagicMock()
        m.returncode = 0
        m.stdout = fake_stdout
        return m

    monkeypatch.setattr(ip.subprocess, "run", fake_run)
    assert ip._detect_docker_bridge_ip() == "172.17.0.1"


def test_detect_docker_bridge_ip_handles_missing_ip_command(monkeypatch):
    """No ``ip`` on PATH (or other OSError) returns None cleanly."""

    def boom(*a, **k):
        raise OSError("no such file")

    monkeypatch.setattr(ip.subprocess, "run", boom)
    assert ip._detect_docker_bridge_ip() is None


# ---------------------------------------------------------------------------
# v3: default deny-list adjacency (P2 IPv4-mapped-v6 + CGNAT)
# ---------------------------------------------------------------------------


def test_default_deny_includes_ipv4_mapped_v6(tmp_path):
    cfg = ip.build_proxy_config(
        mappings=[_sample_mapping()],
        ca_cert=tmp_path / "ca.crt",
        ca_key=tmp_path / "ca.key",
    )
    deny = cfg["proxy"]["upstream_deny_cidrs"]
    assert "::ffff:0:0/96" in deny
    assert "100.64.0.0/10" in deny    # CGNAT
    assert "198.18.0.0/15" in deny    # RFC2544 benchmark


# ---------------------------------------------------------------------------
# v3: split LLM-specific blocked tier (P1 #3 + P2 non_bearer tiers)
# ---------------------------------------------------------------------------


def test_blocked_providers_subset_of_uncovered(hermes_home, monkeypatch):
    """The strict tier that BLOCKS start must be a subset of the
    uncovered-but-warn tier; the wizard surfaces ALL uncovered but only
    blocks the LLM-specific ones."""

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA-test")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/etc/gcp.json")
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")

    uncovered = set(ip.discover_uncovered_providers())
    blocked = set(ip.discover_blocked_providers())

    # Strict subset: every blocked is also uncovered, but the reverse
    # doesn't hold.
    assert blocked.issubset(uncovered)
    # AWS / GCP appdefault present but NOT blocked (those are present on
    # most dev laptops for unrelated cloud tooling).
    assert "AWS_ACCESS_KEY_ID" in uncovered
    assert "AWS_ACCESS_KEY_ID" not in blocked
    assert "GOOGLE_APPLICATION_CREDENTIALS" in uncovered
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in blocked
    # LLM-specific providers ARE blocked.
    assert "ANTHROPIC_API_KEY" in blocked
    assert "GEMINI_API_KEY" in blocked


# ---------------------------------------------------------------------------
# v3: _pid_proc_starttime parser (handles comm with parens, brackets)
# ---------------------------------------------------------------------------


def test_pid_proc_starttime_parses_comm_with_parens(tmp_path, monkeypatch):
    """``/proc/<pid>/stat`` 'comm' field can contain spaces and parens
    (e.g. a process literally named ``weird) tail``, with the comm
    wrapped in the outer parens producing ``(weird) tail)``).  The
    starttime parser must split from the LAST ')' to skip the comm
    correctly, otherwise the field-index math drifts.

    Layout reminder: ``<pid> (<comm>) <state> <ppid> ... <starttime> ...``
    where starttime is field 22 (1-indexed).  Past the LAST ')' we have
    fields 3..N → tail index 0..N-3, so starttime is at tail index 19.
    """

    # Comm contains a ')' character inside the outer parens.  The outer
    # parens are stripped by the kernel's stat format; we test that
    # rfind(')') correctly finds the OUTER closing one.
    # Format: pid (comm) state ppid pgrp session tty_nr tpgid flags
    #         minflt cminflt majflt cmajflt utime stime cutime cstime
    #         priority nice num_threads itrealvalue starttime ...
    fake_stat = (
        "12345 (weird) tail) "          # pid + comm-with-paren-and-space
        "S 1 1 1 0 -1 4194304 "         # state ppid pgrp sess tty tpgid flags
        "100 0 0 0 10 5 0 0 "           # minflt cminflt majflt cmajflt utime stime cutime cstime
        "20 0 1 0 99887766 "            # priority nice nthreads itreal STARTTIME
        "1234 5678 ...rest..."          # vsize rss ...
    )

    real_read_text = Path.read_text

    def fake_read_text(self, *a, **k):
        if str(self).startswith("/proc/12345/stat"):
            return fake_stat
        return real_read_text(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", fake_read_text)
    assert ip._pid_proc_starttime(12345) == "99887766"


def test_pid_proc_starttime_returns_none_on_missing_proc(monkeypatch):
    """Non-Linux hosts or pid not running."""

    def raise_oserror(self, *a, **k):
        raise OSError("no such file")

    monkeypatch.setattr(Path, "read_text", raise_oserror)
    assert ip._pid_proc_starttime(99999) is None


# ---------------------------------------------------------------------------
# v3: stop_proxy SIGKILL suppression on pid recycle (P3 #5 coverage gap)
# ---------------------------------------------------------------------------


def test_stop_proxy_suppresses_sigkill_on_pid_recycle(hermes_home, monkeypatch):
    """When _pid_proc_starttime returns different values before and
    after the SIGTERM grace window, stop_proxy must NOT issue SIGKILL —
    the original pid was recycled to an unrelated process."""

    state = ip._proxy_state_dir()
    (state / "iron-proxy.pid").write_text("12345")

    # _pid_alive: always True (process at this pid keeps appearing alive)
    monkeypatch.setattr(ip, "_pid_alive", lambda pid: True)

    # Starttime changes — first call returns "AAA", second returns "BBB"
    # (simulating PID recycle to an unrelated process during the grace).
    starttime_responses = iter(["111", "222"])
    monkeypatch.setattr(
        ip, "_pid_proc_starttime",
        lambda pid: next(starttime_responses, "222"),
    )

    # Sentinel: kill list — if SIGKILL is sent, this gets populated.
    kills: list = []

    def fake_os_kill(pid, sig):
        kills.append((pid, sig))

    monkeypatch.setattr(ip.os, "kill", fake_os_kill)
    # Speed up the wait loop.
    monkeypatch.setattr(ip.time, "sleep", lambda _: None)
    # Make time advance fast.
    orig_time = ip.time.time
    counter = {"n": 0.0}

    def fake_time():
        counter["n"] += 1.0  # 1 second per call — past deadline immediately
        return counter["n"]

    monkeypatch.setattr(ip.time, "time", fake_time)

    result = ip.stop_proxy()
    assert result is True
    # SIGTERM should fire, SIGKILL should NOT (recycled detection).
    sigterm_count = sum(1 for _, s in kills if s == ip.signal.SIGTERM)
    sigkill_count = sum(1 for _, s in kills if s == ip._KILL_SIGNAL)
    assert sigterm_count == 1
    assert sigkill_count == 0, f"SIGKILL was issued despite pid recycle: {kills}"


# ---------------------------------------------------------------------------
# v3: _reset_for_tests actually clears module state (P3 #1)
# ---------------------------------------------------------------------------


def test_reset_for_tests_clears_version_cache_and_nonce():
    """_reset_for_tests must clear _VERSION_CACHE and _proxy_nonce so
    in-process callers don't see leakage between tests."""

    ip._VERSION_CACHE["dummy"] = "v0.0.0-fake"
    ip._proxy_nonce = "fake-nonce-12345"
    ip._reset_for_tests()
    assert ip._VERSION_CACHE == {}
    assert ip._proxy_nonce is None


# ---------------------------------------------------------------------------
# v3: version cache doesn't poison on empty stdout (P2 _VERSION_CACHE bug B)
# ---------------------------------------------------------------------------


def test_iron_proxy_version_does_not_cache_empty_output(monkeypatch, tmp_path):
    """If --version returns 0 with empty stdout (corrupt binary, flag
    rename upstream), don't poison the cache — re-probe next call."""

    binary = tmp_path / "iron-proxy"
    binary.write_text("")

    # First call: returns empty output.
    def fake_run_empty(cmd, **kw):
        from unittest.mock import MagicMock
        m = MagicMock()
        m.returncode = 0
        m.stdout = ""
        m.stderr = ""
        return m

    ip._VERSION_CACHE.clear()
    monkeypatch.setattr(ip.subprocess, "run", fake_run_empty)
    assert ip.iron_proxy_version(binary) == ""
    # Should NOT have cached the empty string.
    assert str(binary) not in ip._VERSION_CACHE


# ---------------------------------------------------------------------------
# v3: NODE_OPTIONS append-merge in docker env (arshkumarsingh #1)
# ---------------------------------------------------------------------------


def test_docker_egress_node_options_uses_sentinel(hermes_home, monkeypatch):
    """``_egress_proxy_args_for_docker`` should NOT put NODE_OPTIONS in
    env_overrides directly; it uses a sentinel key
    ``_HERMES_EGRESS_NODE_OPTIONS_APPEND`` so DockerEnvironment can
    append-merge with the operator's existing NODE_OPTIONS."""

    from tools.environments.docker import _egress_proxy_args_for_docker
    from hermes_cli.config import load_config, save_config

    state = ip._proxy_state_dir()
    ca = state / "ca.crt"
    ca.write_text("fake-ca")
    (state / "ca.key").write_text("fake-key")
    mapping = _sample_mapping("OPENROUTER_API_KEY")
    proxy_cfg = ip.build_proxy_config(
        mappings=[mapping], ca_cert=ca, ca_key=state / "ca.key", tunnel_port=9090,
    )
    ip.write_proxy_config(proxy_cfg)
    ip.write_mappings([mapping])

    cfg = load_config()
    cfg.setdefault("proxy", {})["enabled"] = True
    cfg["proxy"]["enforce_on_docker"] = True
    save_config(cfg)

    (state / "iron-proxy.pid").write_text("99999")
    monkeypatch.setattr(ip, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(ip, "_port_listening", lambda h, p: True)

    _, env, _ = _egress_proxy_args_for_docker()
    # The egress dict should contain the sentinel, NOT a raw NODE_OPTIONS.
    assert env.get("_HERMES_EGRESS_NODE_OPTIONS_APPEND") == "--use-openssl-ca"
    assert "NODE_OPTIONS" not in env, (
        "NODE_OPTIONS in egress env_overrides would clobber the operator's "
        "docker_env NODE_OPTIONS — that's exactly the bug arshkumarsingh "
        "flagged."
    )


# ---------------------------------------------------------------------------
# v3: ensure_audit_log fails loud on OSError (P2 promise mismatch)
# ---------------------------------------------------------------------------


def test_ensure_audit_log_raises_on_immutable_parent(tmp_path, monkeypatch):
    """When the audit log can't be created with 0o600 (planted symlink,
    immutable dir, disk full), raise — silently logging a warning would
    let the daemon create the file under default umask, breaking the
    privacy promise."""

    # Aim at a path whose parent doesn't exist — open() will OSError.
    audit = tmp_path / "definitely-does-not-exist" / "audit.log"
    with pytest.raises(RuntimeError, match="audit log"):
        ip.ensure_audit_log(audit)


# ---------------------------------------------------------------------------
# v3: persisted nonce roundtrip (stephenschoettler #3 cross-CLI defense)
# ---------------------------------------------------------------------------


def test_persisted_nonce_roundtrip(hermes_home, monkeypatch):
    """Write the nonce next to the pidfile (simulating one CLI invocation
    finishing start_proxy), then verify a fresh _read_persisted_nonce
    can pick it up — that's what cross-process _pid_alive uses."""

    nonce_path = ip._persisted_nonce_path()
    nonce_path.parent.mkdir(parents=True, exist_ok=True)
    nonce_path.write_text("test-nonce-abc123")
    assert ip._read_persisted_nonce() == "test-nonce-abc123"


def test_persisted_nonce_returns_none_when_missing(hermes_home):
    """No nonce file → None, callers fall back to argv0 basename."""
    assert ip._read_persisted_nonce() is None


# ---------------------------------------------------------------------------
# v4 round (GodsBoy follow-up): bind-host-aware liveness probes +
# allow_env_fallback on the partial-secret path
# ---------------------------------------------------------------------------


def test_read_http_listen_from_config_returns_host_and_port(hermes_home):
    """_read_http_listen_from_config must surface the BIND HOST, not just
    the port — on Linux the daemon binds the docker bridge gateway and a
    hardcoded loopback probe would report a healthy daemon as dead."""

    state = ip._proxy_state_dir()
    (state / "proxy.yaml").write_text(
        "proxy:\n  http_listen: 172.17.0.1:9090\n", encoding="utf-8"
    )
    assert ip._read_http_listen_from_config() == ("172.17.0.1", 9090)
    assert ip._read_tunnel_port_from_config() == 9090


def test_read_http_listen_from_config_missing_file(hermes_home):
    assert ip._read_http_listen_from_config() is None


def test_get_status_probes_configured_bind_host(hermes_home, monkeypatch):
    """get_status must probe the configured bind host (e.g. the docker
    bridge IP), not loopback unconditionally."""

    state = ip._proxy_state_dir()
    (state / "proxy.yaml").write_text(
        "proxy:\n  http_listen: 172.17.0.1:9123\n", encoding="utf-8"
    )
    (state / "ca.crt").write_text("cert")
    ip._write_pidfile_safely(ip._pidfile(), 99999)
    monkeypatch.setattr(ip, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(ip, "find_iron_proxy", lambda **kw: None)

    probed = {}

    def fake_probe(host, port):
        probed["host"] = host
        probed["port"] = port
        return True

    monkeypatch.setattr(ip, "_port_listening", fake_probe)
    status = ip.get_status()
    assert probed == {"host": "172.17.0.1", "port": 9123}
    assert status.listening is True
    assert status.tunnel_port == 9123


def test_partial_bitwarden_secrets_honor_allow_env_fallback(
    hermes_home, monkeypatch,
):
    """The missing-secret branch's own error message tells operators to
    set proxy.allow_env_fallback — so the flag must actually work there
    (previously only the empty-token branch honored it)."""

    ip.write_mappings([_sample_mapping("OPENROUTER_API_KEY")])
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-host-fallback")

    import agent.secret_sources.bitwarden as bw
    monkeypatch.setattr(
        bw, "fetch_bitwarden_secrets", lambda **kw: ({}, []),
    )
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "tok")
    bw_cfg = {
        "project_id": "proj",
        "access_token_env": "BWS_ACCESS_TOKEN",
        "allow_env_fallback": True,
    }

    env = ip._build_proxy_subprocess_env(
        refresh_from_bitwarden=True, bitwarden_config=bw_cfg,
    )
    # Falls back to the host env value instead of raising.
    assert env.get("OPENROUTER_API_KEY") == "sk-host-fallback"


def test_partial_bitwarden_secrets_raise_without_fallback(
    hermes_home, monkeypatch,
):
    """Strict default: missing BWS secrets raise."""

    ip.write_mappings([_sample_mapping("OPENROUTER_API_KEY")])
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-host")

    import agent.secret_sources.bitwarden as bw
    monkeypatch.setattr(
        bw, "fetch_bitwarden_secrets", lambda **kw: ({}, []),
    )
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "tok")
    bw_cfg = {"project_id": "proj", "access_token_env": "BWS_ACCESS_TOKEN"}

    with pytest.raises(RuntimeError, match="did not return secrets"):
        ip._build_proxy_subprocess_env(
            refresh_from_bitwarden=True, bitwarden_config=bw_cfg,
        )


def test_bitwarden_importerror_raise_without_fallback(
    hermes_home, monkeypatch,
):
    """Strict default: ImportError on BWS module raises when
    allow_env_fallback is unset, matching the sibling branches."""

    ip.write_mappings([_sample_mapping("OPENROUTER_API_KEY")])
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-host")

    # Simulate the BWS SDK not being installed.  The lazy import
    # ``from agent.secret_sources import bitwarden`` inside
    # _build_proxy_subprocess_env resolves through the parent package's
    # cached attribute; deleting both the sys.modules entry AND the
    # parent-package attribute forces a real import that we intercept.
    #
    # In addition, block importlib.reload in case the test infra used it.
    import agent.secret_sources as ss
    monkeypatch.delitem(sys.modules, "agent.secret_sources.bitwarden", raising=False)
    monkeypatch.delitem(sys.modules, "agent.secret_sources.bitwarden.bws", raising=False)
    monkeypatch.delattr(ss, "bitwarden", raising=False)

    # Now block the re-import.  ``from agent.secret_sources import
    # bitwarden`` resolves to a submodule attribute; setting it to a
    # sentinel that raises on attribute access is more reliable than
    # trying to intercept __import__ at the C level.
    class _MissingBWS:
        """Sentinel: accessing any attribute raises ImportError."""
        def __getattr__(self, _name):
            raise ImportError("bws SDK not installed")
        def __call__(self, *a, **kw):
            raise ImportError("bws SDK not installed")
    monkeypatch.setattr(ss, "bitwarden", _MissingBWS(), raising=False)

    monkeypatch.setenv("BWS_ACCESS_TOKEN", "tok")
    bw_cfg = {"project_id": "proj", "access_token_env": "BWS_ACCESS_TOKEN"}

    with pytest.raises(RuntimeError, match="Bitwarden refresh module unavailable"):
        ip._build_proxy_subprocess_env(
            refresh_from_bitwarden=True, bitwarden_config=bw_cfg,
        )


def test_bitwarden_importerror_honor_allow_env_fallback(
    hermes_home, monkeypatch,
):
    """With allow_env_fallback, an ImportError falls through to host env
    instead of raising."""

    ip.write_mappings([_sample_mapping("OPENROUTER_API_KEY")])
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-host-fallback")

    import agent.secret_sources as ss
    monkeypatch.delitem(sys.modules, "agent.secret_sources.bitwarden", raising=False)
    monkeypatch.delitem(sys.modules, "agent.secret_sources.bitwarden.bws", raising=False)
    monkeypatch.delattr(ss, "bitwarden", raising=False)

    class _MissingBWS:
        def __getattr__(self, _name):
            raise ImportError("bws SDK not installed")
        def __call__(self, *a, **kw):
            raise ImportError("bws SDK not installed")
    monkeypatch.setattr(ss, "bitwarden", _MissingBWS(), raising=False)

    monkeypatch.setenv("BWS_ACCESS_TOKEN", "tok")
    bw_cfg = {
        "project_id": "proj",
        "access_token_env": "BWS_ACCESS_TOKEN",
        "allow_env_fallback": True,
    }

    env = ip._build_proxy_subprocess_env(
        refresh_from_bitwarden=True, bitwarden_config=bw_cfg,
    )
    assert env.get("OPENROUTER_API_KEY") == "sk-host-fallback"
