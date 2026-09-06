"""Tests for the secret-source contract + orchestrator.

Covers: registration gating (API version, name/scheme uniqueness, shape),
apply_all precedence (mapped beats bulk, first-wins, override_existing,
protected vars), conflict surfacing, timeout enforcement, provenance,
and Bitwarden's SecretSource adapter — plus the conformance kit run
against the bundled Bitwarden source.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.secret_sources.base import (  # noqa: E402
    SECRET_SOURCE_API_VERSION,
    ErrorKind,
    FetchResult,
    SecretSource,
    build_minimal_provider_env,
    is_valid_env_name,
    run_secret_cli,
    sanitize_provider_version,
    scrub_ansi,
)
from agent.secret_sources import registry as reg  # noqa: E402
from agent.secret_sources.bitwarden import BitwardenSource  # noqa: E402
from tests.secret_sources.conformance import SecretSourceConformance  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_registry(monkeypatch):
    """Each test starts with an empty registry and no builtin auto-load."""
    reg._reset_registry_for_tests()
    monkeypatch.setattr(reg, "_ensure_builtin_sources", lambda: None)
    yield
    reg._reset_registry_for_tests()


def _make_source(
    name="dummy",
    shape="mapped",
    secrets=None,
    error=None,
    error_kind=None,
    scheme=None,
    override=False,
    protected=(),
    api_version=SECRET_SOURCE_API_VERSION,
    fetch_fn=None,
):
    """Build a minimal conforming source for orchestrator tests."""

    class _Src(SecretSource):
        def fetch(self, cfg, home_path):
            if fetch_fn is not None:
                return fetch_fn(cfg, home_path)
            res = FetchResult()
            if error:
                res.error = error
                res.error_kind = error_kind or ErrorKind.INTERNAL
            else:
                res.secrets = dict(secrets or {})
            return res

        def override_existing(self, cfg):
            return override

        def protected_env_vars(self, cfg):
            return frozenset(protected)

    _Src.name = name
    _Src.label = name.title()
    _Src.shape = shape
    _Src.scheme = scheme
    _Src.api_version = api_version
    return _Src()


# ---------------------------------------------------------------------------
# Registration gating
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_registers_conforming_source(self):
        assert reg.register_source(_make_source()) is True
        assert reg.get_source("dummy") is not None

    def test_rejects_non_secretsource_instance(self):
        assert reg.register_source(object()) is False

    def test_same_name_is_isolated_by_profile(self, tmp_path):
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        home_a = str((tmp_path / "secrets-a").resolve())
        home_b = str((tmp_path / "secrets-b").resolve())
        source_a = _make_source(name="profile_secret", secrets={"A": "a"})
        source_b = _make_source(name="profile_secret", secrets={"B": "b"})
        assert reg.register_source(source_a, scope=home_a)
        assert reg.register_source(source_b, scope=home_b)

        token = set_hermes_home_override(home_a)
        try:
            assert reg.get_source("profile_secret") is source_a
            explicit_b_env = {}
            report = reg.apply_all(
                {"profile_secret": {"enabled": True}},
                Path(home_b),
                environ=explicit_b_env,
            )
            assert report.sources[0].result.secrets == {"B": "b"}
            assert explicit_b_env == {"B": "b"}
        finally:
            reset_hermes_home_override(token)
        token = set_hermes_home_override(home_b)
        try:
            assert reg.get_source("profile_secret") is source_b
        finally:
            reset_hermes_home_override(token)








# ---------------------------------------------------------------------------
# apply_all: precedence, conflicts, protection
# ---------------------------------------------------------------------------


class TestApplyAll:
    def test_disabled_sources_do_not_run(self, tmp_path):
        called = []

        def _fetch(cfg, home):
            called.append(True)
            return FetchResult(secrets={"A": "1"})

        reg.register_source(_make_source(fetch_fn=_fetch))
        env: dict = {}
        report = reg.apply_all({"dummy": {"enabled": False}}, tmp_path, environ=env)
        assert not called
        assert not report.sources
        assert env == {}

    def test_applies_secrets_and_records_provenance(self, tmp_path):
        reg.register_source(_make_source(secrets={"API_KEY": "v1"}))
        env: dict = {}
        report = reg.apply_all({"dummy": {"enabled": True}}, tmp_path, environ=env)
        assert env["API_KEY"] == "v1"
        assert report.provenance["API_KEY"].source == "dummy"
        assert report.provenance["API_KEY"].shape == "mapped"
        assert report.provenance["API_KEY"].overrode_env is False








    def test_failed_source_does_not_block_others(self, tmp_path):
        reg.register_source(
            _make_source(name="broken", error="boom", error_kind=ErrorKind.NETWORK)
        )
        reg.register_source(_make_source(name="works", secrets={"K": "v"}))
        env: dict = {}
        report = reg.apply_all(
            {"broken": {"enabled": True}, "works": {"enabled": True}},
            tmp_path, environ=env,
        )
        assert env["K"] == "v"
        broken = [s for s in report.sources if s.name == "broken"][0]
        assert broken.result.error_kind is ErrorKind.NETWORK




    def test_malformed_secrets_cfg_shapes_are_safe(self, tmp_path):
        reg.register_source(_make_source(secrets={"K": "v"}))
        for cfg in (None, [], "junk", {"dummy": "not-a-dict"}, {"sources": "junk"}):
            report = reg.apply_all(cfg, tmp_path, environ={})
            assert isinstance(report, reg.ApplyReport)



# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_is_valid_env_name(self):
        assert is_valid_env_name("GOOD_NAME")
        assert is_valid_env_name("_LEADING")
        assert not is_valid_env_name("")
        assert not is_valid_env_name("1BAD")
        assert not is_valid_env_name("bad-name")
        assert not is_valid_env_name("has space")


    def test_run_secret_cli_minimal_env(self):
        proc = run_secret_cli(
            [sys.executable, "-c",
             "import os, json; print(json.dumps(sorted(os.environ)))"],
        )
        import json

        child_env = json.loads(proc.stdout)
        # No credential-bearing vars from the parent env leak through.
        assert not any(k.endswith(("_API_KEY", "_TOKEN", "_SECRET"))
                       for k in child_env)
        assert "NO_COLOR" in child_env

    @pytest.mark.parametrize(
        "control",
        [
            "\x9b31m",                         # 8-bit CSI
            "\x9d0;title\x07",                 # 8-bit OSC
            "\x90provider-data\x9c",           # 8-bit DCS
            "\x98provider-data\x9c",           # 8-bit SOS
            "\x9eprovider-data\x9c",           # 8-bit PM
            "\x9fprovider-data\x9c",           # 8-bit APC
            "\x1bPprovider-data\x1b\\",       # 7-bit DCS
        ],
    )
    def test_scrub_ansi_removes_c1_and_ecma48_strings(self, control):
        token = "synthetic-token-77468"
        split = f"provider {token[:9]}{control}{token[9:]}"

        scrubbed = scrub_ansi(split)

        assert scrubbed == "provider " + token
        assert "\x9b" not in scrubbed

    def test_scrub_ansi_consumes_unterminated_osc(self):
        token = "synthetic-token-77468"

        scrubbed = scrub_ansi(f"provider\x1b]0;{token}")

        assert scrubbed == "provider"
        assert token not in scrubbed

    def test_build_minimal_provider_env_keeps_only_explicit_provider_values(
        self, monkeypatch
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "sentinel-openai")
        monkeypatch.setenv("GH_TOKEN", "sentinel-github")
        monkeypatch.setenv("BWS_SERVER_URL", "https://vault.example")

        env = build_minimal_provider_env(
            allow_env=("BWS_SERVER_URL",),
            extra_env={"BWS_ACCESS_TOKEN": "synthetic-token"},
        )

        assert env["BWS_ACCESS_TOKEN"] == "synthetic-token"
        assert env["BWS_SERVER_URL"] == "https://vault.example"
        assert "OPENAI_API_KEY" not in env
        assert "GH_TOKEN" not in env

    def test_build_minimal_provider_env_preserves_network_tls_settings(
        self, monkeypatch
    ):
        network_values = {
            "HTTPS_PROXY": "http://proxy.upper:8080",
            "HTTP_PROXY": "http://http.upper:8080",
            "ALL_PROXY": "socks5://all.upper:1080",
            "NO_PROXY": "localhost,.internal",
            "https_proxy": "http://proxy.lower:8080",
            "http_proxy": "http://http.lower:8080",
            "all_proxy": "socks5://all.lower:1080",
            "no_proxy": "127.0.0.1,.corp",
            "HERMES_CA_BUNDLE": "/etc/hermes/ca.pem",
            "SSL_CERT_FILE": "/etc/ssl/custom.pem",
            "REQUESTS_CA_BUNDLE": "/etc/requests/custom.pem",
            "CURL_CA_BUNDLE": "/etc/curl/custom.pem",
        }
        for key, value in network_values.items():
            monkeypatch.setenv(key, value)
        monkeypatch.setenv("OPENAI_API_KEY", "sentinel-openai")
        monkeypatch.setenv("GH_TOKEN", "sentinel-github")

        env = build_minimal_provider_env()

        for key, value in network_values.items():
            assert env[key] == value
        assert "OPENAI_API_KEY" not in env
        assert "GH_TOKEN" not in env

    def test_sanitize_provider_version_drops_untrusted_text(self):
        assert sanitize_provider_version("bws v2.0.0 (secret-value)") == "v2.0.0"
        assert sanitize_provider_version("version synthetic-token") == "version unknown"





# ---------------------------------------------------------------------------
# Bitwarden adapter
# ---------------------------------------------------------------------------


class TestBitwardenSource:





    def test_fetch_delegates_to_fetch_bitwarden_secrets(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.token")
        import agent.secret_sources.bitwarden as bw

        monkeypatch.setattr(bw, "find_bws", lambda **kw: Path("/fake/bws"))
        captured = {}

        def _fake_fetch(**kwargs):
            captured.update(kwargs)
            return {"MY_KEY": "val"}, ["a warning"]

        monkeypatch.setattr(bw, "fetch_bitwarden_secrets", _fake_fetch)
        result = BitwardenSource().fetch(
            {"enabled": True, "project_id": "proj",
             "server_url": " https://vault.bitwarden.eu "},
            tmp_path,
        )
        assert result.ok
        assert result.secrets == {"MY_KEY": "val"}
        assert result.warnings == ["a warning"]
        assert captured["project_id"] == "proj"
        assert captured["server_url"] == "https://vault.bitwarden.eu"
        assert captured["home_path"] == tmp_path


    def test_e2e_through_orchestrator(self, tmp_path, monkeypatch):
        """Full path: registry → BitwardenSource → env, with fetch mocked."""
        monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.token")
        import agent.secret_sources.bitwarden as bw

        monkeypatch.setattr(bw, "find_bws", lambda **kw: Path("/fake/bws"))
        monkeypatch.setattr(
            bw, "fetch_bitwarden_secrets",
            lambda **kw: ({"ANTHROPIC_API_KEY": "sk-ant", "BWS_ACCESS_TOKEN": "steal"}, []),
        )
        reg.register_source(BitwardenSource())
        env = {"BWS_ACCESS_TOKEN": "0.token"}
        report = reg.apply_all(
            {"bitwarden": {"enabled": True, "project_id": "proj"}},
            tmp_path, environ=env,
        )
        assert env["ANTHROPIC_API_KEY"] == "sk-ant"
        # The bootstrap token is protected even though BSM carried it.
        assert env["BWS_ACCESS_TOKEN"] == "0.token"
        assert report.provenance["ANTHROPIC_API_KEY"].source == "bitwarden"


# ---------------------------------------------------------------------------
# Conformance kit applied to the bundled source
# ---------------------------------------------------------------------------


class TestBitwardenConformance(SecretSourceConformance):
    @pytest.fixture
    def source(self, monkeypatch):
        # Never hit the network / auto-install path in conformance runs.
        import agent.secret_sources.bitwarden as bw

        monkeypatch.setattr(bw, "find_bws", lambda **kw: None)
        monkeypatch.delenv("BWS_ACCESS_TOKEN", raising=False)
        return BitwardenSource()


# ---------------------------------------------------------------------------
# 1Password adapter
# ---------------------------------------------------------------------------


class TestOnePasswordSource:







    def test_mapped_op_beats_bulk_bitwarden_through_orchestrator(
        self, tmp_path, monkeypatch
    ):
        """The headline multi-source scenario: both vaults claim the same var."""
        import agent.secret_sources.bitwarden as bw
        import agent.secret_sources.onepassword as op

        monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.token")
        monkeypatch.setattr(bw, "find_bws", lambda **kw: Path("/fake/bws"))
        monkeypatch.setattr(
            bw, "fetch_bitwarden_secrets",
            lambda **kw: ({"SHARED_KEY": "from-bitwarden",
                           "BW_ONLY": "bw-val"}, []),
        )
        monkeypatch.setattr(op, "find_op", lambda *_a, **_kw: Path("/fake/op"))
        monkeypatch.setattr(
            op, "fetch_onepassword_secrets",
            lambda **kw: ({"SHARED_KEY": "from-1password"}, []),
        )
        reg.register_source(bw.BitwardenSource())
        reg.register_source(op.OnePasswordSource())
        env = {"BWS_ACCESS_TOKEN": "0.token"}
        report = reg.apply_all(
            {
                # bitwarden listed FIRST — mapped 1Password must still win.
                "sources": ["bitwarden", "onepassword"],
                "bitwarden": {"enabled": True, "project_id": "proj"},
                "onepassword": {"enabled": True,
                                "env": {"SHARED_KEY": "op://V/I/F"}},
            },
            tmp_path, environ=env,
        )
        assert env["SHARED_KEY"] == "from-1password"
        assert env["BW_ONLY"] == "bw-val"
        assert report.provenance["SHARED_KEY"].source == "onepassword"
        assert report.provenance["BW_ONLY"].source == "bitwarden"
        assert report.conflicts  # the shadowed bitwarden claim is surfaced


class TestOnePasswordConformance(SecretSourceConformance):
    @pytest.fixture
    def source(self, monkeypatch):
        import agent.secret_sources.onepassword as op

        monkeypatch.setattr(op, "find_op", lambda *_a, **_kw: None)
        monkeypatch.delenv("OP_SERVICE_ACCOUNT_TOKEN", raising=False)
        return op.OnePasswordSource()
