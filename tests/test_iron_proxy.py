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






    # Unknown providers (no entry in _BEARER_PROVIDERS) are skipped, not warned.




# ---------------------------------------------------------------------------
# Config / mapping serialization
# ---------------------------------------------------------------------------


def _sample_mapping(env_name: str = "OPENROUTER_API_KEY") -> ip.TokenMapping:
    return ip.TokenMapping(
        proxy_token=ip.mint_proxy_token("test"),
        real_env_name=env_name,
        upstream_hosts=("openrouter.ai", "*.openrouter.ai"),
    )




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








# ---------------------------------------------------------------------------
# Bind policy (regression: must not bind 0.0.0.0)
# ---------------------------------------------------------------------------












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








def test_load_mappings_handles_corrupt_json(hermes_home):
    state = ip._proxy_state_dir()
    (state / "mappings.json").write_text("{not json", encoding="utf-8")
    assert ip.load_mappings() == []




# ---------------------------------------------------------------------------
# Token-preservation on re-setup (regression: clobbered live sandboxes)
# ---------------------------------------------------------------------------








# ---------------------------------------------------------------------------
# Uncovered provider detection (regression: signature-auth providers bypass)
# ---------------------------------------------------------------------------










# ---------------------------------------------------------------------------
# Binary discovery + lazy install
# ---------------------------------------------------------------------------






def _make_fake_tar(binary_name: str, payload: bytes = b"#!/bin/sh\necho ok\n") -> bytes:
    """Build a tar.gz with one file at the root, named ``binary_name``."""

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name=binary_name)
        info.size = len(payload)
        info.mode = 0o755
        tf.addfile(info, io.BytesIO(payload))
    return buf.getvalue()








# ── GPG release-signature verification (maxpetrusenko P1) ────────────────────

def test_verify_checksums_signature_skips_without_gpg(hermes_home, monkeypatch, tmp_path):
    """No gpg on PATH → degrade gracefully (return False), do not raise."""
    monkeypatch.setattr(ip.shutil, "which", lambda name: None)
    cks = tmp_path / "checksums.txt"
    cks.write_text("abc  iron-proxy.tar.gz\n")
    assert ip._verify_checksums_signature(tmp_path, cks) is False










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












# ---------------------------------------------------------------------------
# Platform asset name resolution
# ---------------------------------------------------------------------------






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




# ---------------------------------------------------------------------------
# Mappings clobber refused when corrupt (regression: silent 403s)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# CA missing → enforce_on_docker semantics (regression: silent fail-open)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Docker env collision detection (regression: docker_env silently bypassed proxy)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# v3 round: bridge-IP parser hardening (P1 #1)
# ---------------------------------------------------------------------------








# ---------------------------------------------------------------------------
# v3: default deny-list adjacency (P2 IPv4-mapped-v6 + CGNAT)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Header-auth providers (x-api-key family) — match_headers + aliases
# ---------------------------------------------------------------------------










def test_mappings_roundtrip_preserves_headers_and_aliases(hermes_home):
    m = ip.TokenMapping(
        proxy_token=ip.mint_proxy_token("gemini"),
        real_env_name="GEMINI_API_KEY",
        upstream_hosts=("generativelanguage.googleapis.com",),
        match_headers=("x-goog-api-key",),
        alias_env_names=("GOOGLE_API_KEY",),
    )
    ip.write_mappings([m])
    loaded = ip.load_mappings()
    assert loaded[0].match_headers == ("x-goog-api-key",)
    assert loaded[0].alias_env_names == ("GOOGLE_API_KEY",)








# ---------------------------------------------------------------------------
# Management API (hot reload)
# ---------------------------------------------------------------------------




def test_ensure_management_token_persists_and_is_stable(hermes_home):
    t1 = ip.ensure_management_token()
    t2 = ip.ensure_management_token()
    assert t1 == t2
    assert t1.startswith("hermes-mgmt-")
    p = ip._proxy_state_dir() / "management.token"
    assert p.exists()
    assert (p.stat().st_mode & 0o777) == 0o600




def test_reload_proxy_refuses_when_not_running(hermes_home, monkeypatch):
    monkeypatch.setattr(ip, "_read_pid", lambda: None)
    with pytest.raises(RuntimeError, match="not running"):
        ip.reload_proxy()




def test_reload_proxy_posts_bearer_to_management_endpoint(hermes_home, monkeypatch):
    monkeypatch.setattr(ip, "_read_pid", lambda: 4242)
    monkeypatch.setattr(ip, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(
        ip, "_read_management_listen_from_config",
        lambda config_path=None: ("127.0.0.1", 9092),
    )
    ip.ensure_management_token()

    captured = {}

    class _FakeResp:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["auth"] = req.get_header("Authorization")
        return _FakeResp()

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    assert ip.reload_proxy() is True
    assert captured["url"] == "http://127.0.0.1:9092/v1/reload"
    assert captured["method"] == "POST"
    token = (ip._proxy_state_dir() / "management.token").read_text().strip()
    assert captured["auth"] == f"Bearer {token}"


def test_start_proxy_injects_management_key_env(hermes_home, monkeypatch):
    """When the generated config has a management listener, start_proxy
    must inject the bearer key env var — v0.39 refuses to start when
    api_key_env is empty."""

    cfg_path = ip._proxy_state_dir() / "proxy.yaml"
    cfg = ip.build_proxy_config(
        mappings=[_sample_mapping()],
        ca_cert=hermes_home / "ca.crt",
        ca_key=hermes_home / "ca.key",
        http_listen=["127.0.0.1:9090"],
    )
    ip.write_proxy_config(cfg)
    (hermes_home / "bin").mkdir(parents=True, exist_ok=True)
    fake_bin = hermes_home / "bin" / "iron-proxy"
    fake_bin.write_text("#!/bin/sh\nsleep 60\n")
    fake_bin.chmod(0o755)

    captured_env = {}

    class _FakeProc:
        pid = 99999
        def poll(self):
            return None

    def fake_popen(cmd, **kw):
        captured_env.update(kw.get("env") or {})
        return _FakeProc()

    monkeypatch.setattr(ip.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(ip, "_write_pidfile_safely", lambda pf, pid: None)
    monkeypatch.setattr(ip, "_port_listening", lambda h, p: True)
    monkeypatch.setattr(ip, "get_status", lambda: ip.ProxyStatus(pid=99999, listening=True))

    ip.start_proxy(binary=fake_bin, config_path=cfg_path, install_if_missing=False)
    assert captured_env.get(ip._MGMT_API_KEY_ENV)
    assert captured_env[ip._MGMT_API_KEY_ENV].startswith("hermes-mgmt-")


# ---------------------------------------------------------------------------
# v3: _pid_proc_starttime parser (handles comm with parens, brackets)
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# v3: stop_proxy SIGKILL suppression on pid recycle (P3 #5 coverage gap)
# ---------------------------------------------------------------------------




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




# ---------------------------------------------------------------------------
# v4 round (GodsBoy follow-up): bind-host-aware liveness probes +
# allow_env_fallback on the partial-secret path
# ---------------------------------------------------------------------------






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


