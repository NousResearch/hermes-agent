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
    is_valid_env_name,
    run_secret_cli,
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

    def test_rejects_wrong_api_version(self):
        src = _make_source(api_version=SECRET_SOURCE_API_VERSION + 1)
        assert reg.register_source(src) is False

    def test_rejects_invalid_name(self):
        assert reg.register_source(_make_source(name="Bad Name")) is False
        assert reg.register_source(_make_source(name="")) is False
        assert reg.register_source(_make_source(name="UPPER")) is False

    def test_rejects_invalid_shape(self):
        assert reg.register_source(_make_source(shape="sideways")) is False

    def test_rejects_duplicate_name_without_replace(self):
        assert reg.register_source(_make_source(name="dup")) is True
        assert reg.register_source(_make_source(name="dup")) is False
        assert reg.register_source(_make_source(name="dup"), replace=True) is True

    def test_rejects_scheme_collision_across_names(self):
        assert reg.register_source(_make_source(name="one", scheme="op")) is True
        assert reg.register_source(_make_source(name="two", scheme="op")) is False

    def test_same_name_replace_keeps_scheme(self):
        assert reg.register_source(_make_source(name="one", scheme="op")) is True
        assert reg.register_source(
            _make_source(name="one", scheme="op"), replace=True
        ) is True


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

    def test_existing_env_wins_without_override(self, tmp_path):
        reg.register_source(_make_source(secrets={"API_KEY": "vault"}))
        env = {"API_KEY": "dotenv"}
        report = reg.apply_all({"dummy": {"enabled": True}}, tmp_path, environ=env)
        assert env["API_KEY"] == "dotenv"
        assert "API_KEY" in report.sources[0].skipped_existing

    def test_override_existing_beats_env_and_is_attributed(self, tmp_path):
        reg.register_source(_make_source(secrets={"API_KEY": "vault"}, override=True))
        env = {"API_KEY": "dotenv"}
        report = reg.apply_all({"dummy": {"enabled": True}}, tmp_path, environ=env)
        assert env["API_KEY"] == "vault"
        assert report.provenance["API_KEY"].overrode_env is True

    def test_mapped_beats_bulk_regardless_of_order(self, tmp_path):
        reg.register_source(
            _make_source(name="bulky", shape="bulk", secrets={"K": "bulk"})
        )
        reg.register_source(
            _make_source(name="mappy", shape="mapped", secrets={"K": "mapped"})
        )
        env: dict = {}
        # bulk listed first in sources order — mapped must still win.
        report = reg.apply_all(
            {"sources": ["bulky", "mappy"],
             "bulky": {"enabled": True}, "mappy": {"enabled": True}},
            tmp_path, environ=env,
        )
        assert env["K"] == "mapped"
        assert report.provenance["K"].source == "mappy"
        assert report.conflicts, "shadowed bulk claim must surface a warning"

    def test_first_source_wins_within_shape(self, tmp_path):
        reg.register_source(_make_source(name="alpha", secrets={"K": "a"}))
        reg.register_source(_make_source(name="beta", secrets={"K": "b"}))
        env: dict = {}
        report = reg.apply_all(
            {"sources": ["beta", "alpha"],
             "alpha": {"enabled": True}, "beta": {"enabled": True}},
            tmp_path, environ=env,
        )
        assert env["K"] == "b"  # beta listed first
        assert report.provenance["K"].source == "beta"
        beta_first = [s for s in report.sources if s.name == "alpha"][0]
        assert "K" in beta_first.skipped_claimed

    def test_cross_source_override_never_clobbers_prior_claim(self, tmp_path):
        """override_existing beats .env, NEVER another source's claim."""
        reg.register_source(_make_source(name="alpha", secrets={"K": "a"}))
        reg.register_source(
            _make_source(name="beta", secrets={"K": "b"}, override=True)
        )
        env: dict = {}
        report = reg.apply_all(
            {"sources": ["alpha", "beta"],
             "alpha": {"enabled": True}, "beta": {"enabled": True}},
            tmp_path, environ=env,
        )
        assert env["K"] == "a"
        assert report.conflicts

    def test_protected_vars_never_overwritten_by_any_source(self, tmp_path):
        reg.register_source(
            _make_source(name="alpha", secrets={"BOOT_TOKEN": "evil"},
                         override=True, protected=("BOOT_TOKEN",))
        )
        env = {"BOOT_TOKEN": "real"}
        report = reg.apply_all({"alpha": {"enabled": True}}, tmp_path, environ=env)
        assert env["BOOT_TOKEN"] == "real"
        assert "BOOT_TOKEN" in report.sources[0].skipped_protected

    def test_invalid_env_names_skipped(self, tmp_path):
        reg.register_source(
            _make_source(secrets={"GOOD_NAME": "v", "bad-name": "v", "1BAD": "v"})
        )
        env: dict = {}
        report = reg.apply_all({"dummy": {"enabled": True}}, tmp_path, environ=env)
        assert "GOOD_NAME" in env and "bad-name" not in env and "1BAD" not in env
        assert set(report.sources[0].skipped_invalid) == {"bad-name", "1BAD"}

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

    def test_raising_fetch_contained_as_internal_error(self, tmp_path):
        def _explode(cfg, home):
            raise ValueError("plugin bug")

        reg.register_source(_make_source(name="buggy", fetch_fn=_explode))
        env: dict = {}
        report = reg.apply_all({"buggy": {"enabled": True}}, tmp_path, environ=env)
        assert report.sources[0].result.error_kind is ErrorKind.INTERNAL
        assert "plugin bug" in report.sources[0].result.error

    def test_wrong_return_type_contained(self, tmp_path):
        reg.register_source(
            _make_source(name="liar", fetch_fn=lambda cfg, home: {"not": "a result"})
        )
        report = reg.apply_all({"liar": {"enabled": True}}, tmp_path, environ={})
        assert report.sources[0].result.error_kind is ErrorKind.INTERNAL

    def test_timeout_enforced(self, tmp_path):
        def _slow(cfg, home):
            time.sleep(5)
            return FetchResult(secrets={"K": "late"})

        src = _make_source(name="slow", fetch_fn=_slow)
        src.fetch_timeout_seconds = lambda cfg: 0.2
        reg.register_source(src)
        env: dict = {}
        start = time.monotonic()
        report = reg.apply_all({"slow": {"enabled": True}}, tmp_path, environ=env)
        assert time.monotonic() - start < 3
        assert report.sources[0].result.error_kind is ErrorKind.TIMEOUT
        assert "K" not in env

    def test_malformed_secrets_cfg_shapes_are_safe(self, tmp_path):
        reg.register_source(_make_source(secrets={"K": "v"}))
        for cfg in (None, [], "junk", {"dummy": "not-a-dict"}, {"sources": "junk"}):
            report = reg.apply_all(cfg, tmp_path, environ={})
            assert isinstance(report, reg.ApplyReport)

    def test_unknown_sources_entry_warns_but_continues(self, tmp_path, caplog):
        reg.register_source(_make_source(secrets={"K": "v"}))
        env: dict = {}
        reg.apply_all(
            {"sources": ["ghost", "dummy"], "dummy": {"enabled": True}},
            tmp_path, environ=env,
        )
        assert env["K"] == "v"


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

    def test_scrub_ansi_removes_whole_sequences(self):
        assert scrub_ansi("\x1b[31mred\x1b[0m plain") == "red plain"
        assert scrub_ansi("\x1b]0;title\x07text") == "text"
        assert scrub_ansi("") == ""

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

    def test_run_secret_cli_allowlist_passes_named_vars(self, monkeypatch):
        monkeypatch.setenv("MY_AUTH_TOKEN", "tok")
        monkeypatch.setenv("OTHER_API_KEY", "leak")
        proc = run_secret_cli(
            [sys.executable, "-c",
             "import os; print(os.environ.get('MY_AUTH_TOKEN', '')); "
             "print(os.environ.get('OTHER_API_KEY', ''))"],
            allow_env=["MY_AUTH_TOKEN"],
        )
        lines = proc.stdout.splitlines()
        assert lines[0] == "tok"
        assert lines[1] == ""

    def test_run_secret_cli_timeout_raises_runtime_error(self):
        with pytest.raises(RuntimeError, match="timed out"):
            run_secret_cli(
                [sys.executable, "-c", "import time; time.sleep(10)"],
                timeout=0.3,
            )

    def test_run_secret_cli_stdin_devnull(self):
        # A helper that tries to prompt reads EOF immediately.
        proc = run_secret_cli(
            [sys.executable, "-c",
             "import sys; print(repr(sys.stdin.read()))"],
        )
        assert proc.stdout.strip() == "''"


# ---------------------------------------------------------------------------
# Bitwarden adapter
# ---------------------------------------------------------------------------


class TestBitwardenSource:
    def test_identity(self):
        src = BitwardenSource()
        assert src.name == "bitwarden"
        assert src.shape == "bulk"
        assert src.scheme == "bws"

    def test_override_existing_defaults_true(self):
        src = BitwardenSource()
        assert src.override_existing({}) is True
        assert src.override_existing({"override_existing": False}) is False

    def test_protected_vars_track_token_env(self):
        src = BitwardenSource()
        assert src.protected_env_vars({}) == frozenset({"BWS_ACCESS_TOKEN"})
        assert src.protected_env_vars(
            {"access_token_env": "CUSTOM_TOKEN"}
        ) == frozenset({"CUSTOM_TOKEN"})

    def test_fetch_missing_token_not_configured(self, tmp_path, monkeypatch):
        monkeypatch.delenv("BWS_ACCESS_TOKEN", raising=False)
        result = BitwardenSource().fetch({"enabled": True}, tmp_path)
        assert result.error_kind is ErrorKind.NOT_CONFIGURED
        assert "BWS_ACCESS_TOKEN" in result.error

    def test_fetch_missing_project_not_configured(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.token")
        result = BitwardenSource().fetch({"enabled": True}, tmp_path)
        assert result.error_kind is ErrorKind.NOT_CONFIGURED
        assert "project_id" in result.error

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

    def test_fetch_runtime_error_classified(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.token")
        import agent.secret_sources.bitwarden as bw

        monkeypatch.setattr(bw, "find_bws", lambda **kw: Path("/fake/bws"))

        def _fail(**kwargs):
            raise RuntimeError("bws exited 1: 401 unauthorized")

        monkeypatch.setattr(bw, "fetch_bitwarden_secrets", _fail)
        result = BitwardenSource().fetch(
            {"enabled": True, "project_id": "proj"}, tmp_path
        )
        assert result.error_kind is ErrorKind.AUTH_FAILED

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
    def test_identity(self):
        from agent.secret_sources.onepassword import OnePasswordSource

        src = OnePasswordSource()
        assert src.name == "onepassword"
        assert src.shape == "mapped"
        assert src.scheme == "op"

    def test_override_existing_defaults_true(self):
        from agent.secret_sources.onepassword import OnePasswordSource

        src = OnePasswordSource()
        assert src.override_existing({}) is True
        assert src.override_existing({"override_existing": False}) is False

    def test_protected_vars_track_token_env(self):
        from agent.secret_sources.onepassword import OnePasswordSource

        src = OnePasswordSource()
        assert src.protected_env_vars({}) == frozenset(
            {"OP_SERVICE_ACCOUNT_TOKEN"}
        )
        assert src.protected_env_vars(
            {"service_account_token_env": "MY_OP_TOKEN"}
        ) == frozenset({"MY_OP_TOKEN"})

    def test_fetch_empty_map_not_configured(self, tmp_path):
        from agent.secret_sources.onepassword import OnePasswordSource

        result = OnePasswordSource().fetch({"enabled": True}, tmp_path)
        assert result.error_kind is ErrorKind.NOT_CONFIGURED

    def test_fetch_missing_binary(self, tmp_path, monkeypatch):
        import agent.secret_sources.onepassword as op

        monkeypatch.setattr(op, "find_op", lambda *_a, **_kw: None)
        result = op.OnePasswordSource().fetch(
            {"enabled": True, "env": {"K": "op://V/I/F"}}, tmp_path
        )
        assert result.error_kind is ErrorKind.BINARY_MISSING

    def test_fetch_delegates_and_passes_config(self, tmp_path, monkeypatch):
        import agent.secret_sources.onepassword as op

        monkeypatch.setattr(op, "find_op", lambda *_a, **_kw: Path("/fake/op"))
        captured = {}

        def _fake_fetch(**kwargs):
            captured.update(kwargs)
            return {"K": "v"}, ["warn"]

        monkeypatch.setattr(op, "fetch_onepassword_secrets", _fake_fetch)
        result = op.OnePasswordSource().fetch(
            {"enabled": True, "env": {"K": "op://V/I/F"},
             "account": "team", "service_account_token_env": "MY_TOK"},
            tmp_path,
        )
        assert result.ok and result.secrets == {"K": "v"}
        assert captured["references"] == {"K": "op://V/I/F"}
        assert captured["account"] == "team"
        assert captured["token_env"] == "MY_TOK"

    def test_invalid_refs_warned_not_fatal(self, tmp_path, monkeypatch):
        import agent.secret_sources.onepassword as op

        monkeypatch.setattr(op, "find_op", lambda *_a, **_kw: Path("/fake/op"))
        monkeypatch.setattr(op, "fetch_onepassword_secrets",
                            lambda **kw: ({"GOOD": "v"}, []))
        result = op.OnePasswordSource().fetch(
            {"enabled": True,
             "env": {"GOOD": "op://V/I/F", "BAD": "not-a-ref",
                     "bad name": "op://V/I/F"}},
            tmp_path,
        )
        assert result.ok
        assert len(result.warnings) == 2

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
