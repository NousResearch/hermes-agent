"""Error remediation for secret sources.

Covers the ErrorKind classification of Bitwarden's `invalid_client`
identity reject, the bws stderr summarizer, the per-source
``remediation()`` hook, and the env_loader startup hint printer.
"""
from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from agent.secret_sources import bitwarden as bw
from agent.secret_sources import onepassword as op
from agent.secret_sources.base import ErrorKind, SecretSource
from agent.secret_sources.bitwarden import (
    BitwardenSource,
    _classify_bws_error,
    _summarize_bws_stderr,
)
from agent.secret_sources.onepassword import OnePasswordSource


_BWS_INVALID_CLIENT_DUMP = """\
Error:
   0: Received error message from server: [400 Bad Request] {"error":"invalid_client"}

Location:
   crates/bws/src/main.rs:108

Backtrace omitted. Run with RUST_BACKTRACE=1 environment variable to display it.
Run with RUST_BACKTRACE=full to include source snippets.
"""


# ---------------------------------------------------------------------------
# _summarize_bws_stderr
# ---------------------------------------------------------------------------


def test_summarize_strips_rust_report_noise():
    summary = _summarize_bws_stderr(_BWS_INVALID_CLIENT_DUMP)
    assert "invalid_client" in summary
    assert "Location:" not in summary
    assert "main.rs" not in summary
    assert "Backtrace" not in summary
    assert "Error:" not in summary


def test_summarize_joins_multiple_cause_lines():
    raw = "Error:\n   0: outer cause\n   1: inner cause\n\nLocation:\n   x.rs:1"
    assert _summarize_bws_stderr(raw) == "outer cause; inner cause"


def test_summarize_falls_back_to_raw_on_unknown_shape():
    assert _summarize_bws_stderr("plain failure text") == "plain failure text"
    assert _summarize_bws_stderr("") == ""


# ---------------------------------------------------------------------------
# _classify_bws_error — the invalid_client identity reject is an auth failure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("message", [
    'bws exited 1: Received error message from server: [400 Bad Request] {"error":"invalid_client"}',
    "invalid_grant returned by identity",
    "server said 401 unauthorized",
])
def test_classify_auth_failures(message):
    assert _classify_bws_error(message) == ErrorKind.AUTH_FAILED


def test_classify_unknown_stays_internal():
    assert _classify_bws_error("some novel explosion") == ErrorKind.INTERNAL


# ---------------------------------------------------------------------------
# BitwardenSource.fetch — auth failures get a human explanation
# ---------------------------------------------------------------------------


def test_fetch_auth_failure_gets_friendly_error(monkeypatch, tmp_path):
    src = BitwardenSource()
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.dead")
    monkeypatch.setattr(bw, "find_bws", lambda install_if_missing=True: tmp_path / "bws")

    def boom(**kwargs):
        raise RuntimeError(
            'bws exited 1: Received error message from server: '
            '[400 Bad Request] {"error":"invalid_client"}'
        )

    monkeypatch.setattr(bw, "fetch_bitwarden_secrets", boom)
    result = src.fetch({"enabled": True, "project_id": "p"}, tmp_path)
    assert result.error_kind == ErrorKind.AUTH_FAILED
    assert "revoked, expired" in result.error
    assert "BWS_ACCESS_TOKEN" in result.error
    assert "invalid_client" in result.error  # mechanics preserved


# ---------------------------------------------------------------------------
# remediation() hook
# ---------------------------------------------------------------------------


def test_bitwarden_auth_remediation_points_at_token_command():
    hint = BitwardenSource().remediation(ErrorKind.AUTH_FAILED, {})
    assert "hermes secrets bitwarden token" in hint


def test_onepassword_auth_remediation_points_at_token_command():
    hint = OnePasswordSource().remediation(ErrorKind.AUTH_FAILED, {})
    assert "hermes secrets onepassword token" in hint
    assert "OP_SERVICE_ACCOUNT_TOKEN" in hint


def test_onepassword_remediation_uses_configured_token_env():
    hint = OnePasswordSource().remediation(
        ErrorKind.AUTH_FAILED, {"service_account_token_env": "MY_OP_TOKEN"}
    )
    assert "MY_OP_TOKEN" in hint


def test_base_remediation_covers_common_kinds():
    class _Src(SecretSource):
        name = "dummy"
        label = "Dummy"

        def fetch(self, cfg, home_path):  # pragma: no cover
            raise NotImplementedError

    src = _Src()
    for kind in (ErrorKind.NOT_CONFIGURED, ErrorKind.BINARY_MISSING,
                 ErrorKind.AUTH_FAILED, ErrorKind.AUTH_EXPIRED,
                 ErrorKind.NETWORK, ErrorKind.TIMEOUT):
        hint = src.remediation(kind, {})
        assert hint, f"no default hint for {kind}"
        if kind in (ErrorKind.NOT_CONFIGURED, ErrorKind.BINARY_MISSING,
                    ErrorKind.AUTH_FAILED, ErrorKind.AUTH_EXPIRED):
            assert "hermes secrets dummy" in hint
    # Kinds without a sensible generic action stay silent.
    assert _Src().remediation(ErrorKind.INTERNAL, {}) == ""
    assert _Src().remediation(None, {}) == ""


def test_remediation_never_raises_on_junk_cfg():
    for cfg in (None, [], "nope", 42):
        assert isinstance(BitwardenSource().remediation(ErrorKind.AUTH_FAILED, cfg), str)
        assert isinstance(OnePasswordSource().remediation(ErrorKind.AUTH_FAILED, cfg), str)


# ---------------------------------------------------------------------------
# env_loader startup hint
# ---------------------------------------------------------------------------


def test_env_loader_prints_remediation_hint(tmp_path, monkeypatch, capsys):
    from hermes_cli import env_loader
    from agent.secret_sources import registry

    registry._reset_registry_for_tests()
    env_loader.reset_secret_source_cache()

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "secrets:\n"
        "  bitwarden:\n"
        "    enabled: true\n"
        "    project_id: proj\n"
    )
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.dead")
    monkeypatch.setattr(bw, "find_bws", lambda install_if_missing=True: tmp_path / "bws")

    def boom(**kwargs):
        raise RuntimeError(
            'bws exited 1: Received error message from server: '
            '[400 Bad Request] {"error":"invalid_client"}'
        )

    monkeypatch.setattr(bw, "fetch_bitwarden_secrets", boom)
    try:
        env_loader._apply_external_secret_sources(home)
    finally:
        registry._reset_registry_for_tests()
        env_loader.reset_secret_source_cache()

    err = capsys.readouterr().err
    assert "rejected the machine-account access token" in err
    assert "hermes secrets bitwarden token" in err


def test_env_loader_hint_survives_broken_remediation(tmp_path, monkeypatch, capsys):
    """A plugin source whose remediation() raises must not break startup."""
    from hermes_cli import env_loader
    from agent.secret_sources import registry

    class _Broken(SecretSource):
        name = "brokensrc"
        label = "Broken"
        shape = "bulk"

        def fetch(self, cfg, home_path):
            from agent.secret_sources.base import FetchResult
            res = FetchResult()
            res.error = "kaput"
            res.error_kind = ErrorKind.AUTH_FAILED
            return res

        def remediation(self, kind, cfg):
            raise RuntimeError("hint machine broke")

    registry._reset_registry_for_tests()
    registry._BUILTINS_LOADED = True  # keep real builtins out of this test
    registry.register_source(_Broken())
    env_loader.reset_secret_source_cache()

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "secrets:\n  brokensrc:\n    enabled: true\n"
    )
    try:
        env_loader._apply_external_secret_sources(home)
    finally:
        registry._reset_registry_for_tests()
        env_loader.reset_secret_source_cache()

    err = capsys.readouterr().err
    assert "kaput" in err  # error still surfaced, no crash
