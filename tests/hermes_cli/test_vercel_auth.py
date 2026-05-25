"""Tests for hermes_cli.vercel_auth — describe_vercel_auth and VercelAuthStatus."""

from __future__ import annotations

from hermes_cli.vercel_auth import VercelAuthStatus, _TOKEN_TUPLE_VARS, describe_vercel_auth


class TestDescribeVercelAuth:
    def test_oidc_token_present(self, monkeypatch):
        monkeypatch.setenv("VERCEL_OIDC_TOKEN", "oidc-token-value")
        monkeypatch.delenv("VERCEL_TOKEN", raising=False)
        monkeypatch.delenv("VERCEL_PROJECT_ID", raising=False)
        monkeypatch.delenv("VERCEL_TEAM_ID", raising=False)

        status = describe_vercel_auth()
        assert status.ok is True
        assert "OIDC" in status.label
        assert "mode: OIDC" in status.detail_lines

    def test_oidc_with_access_token_vars(self, monkeypatch):
        monkeypatch.setenv("VERCEL_OIDC_TOKEN", "oidc-xxx")
        monkeypatch.setenv("VERCEL_TOKEN", "token-xxx")
        monkeypatch.delenv("VERCEL_PROJECT_ID", raising=False)
        monkeypatch.delenv("VERCEL_TEAM_ID", raising=False)

        status = describe_vercel_auth()
        assert status.ok is True
        assert "OIDC" in status.label
        assert "also present: VERCEL_TOKEN" in "\n".join(status.detail_lines)

    def test_full_access_token_auth(self, monkeypatch):
        monkeypatch.delenv("VERCEL_OIDC_TOKEN", raising=False)
        monkeypatch.setenv("VERCEL_TOKEN", "token")
        monkeypatch.setenv("VERCEL_PROJECT_ID", "proj")
        monkeypatch.setenv("VERCEL_TEAM_ID", "team")

        status = describe_vercel_auth()
        assert status.ok is True
        assert "access token" in status.label
        assert "mode: access token" in status.detail_lines

    def test_partial_access_token(self, monkeypatch):
        monkeypatch.delenv("VERCEL_OIDC_TOKEN", raising=False)
        monkeypatch.setenv("VERCEL_TOKEN", "token")
        monkeypatch.delenv("VERCEL_PROJECT_ID", raising=False)
        monkeypatch.delenv("VERCEL_TEAM_ID", raising=False)

        status = describe_vercel_auth()
        assert status.ok is False
        assert "partial" in status.label
        assert "incomplete" in "\n".join(status.detail_lines)

    def test_not_configured(self, monkeypatch):
        for var in ("VERCEL_OIDC_TOKEN", "VERCEL_TOKEN", "VERCEL_PROJECT_ID", "VERCEL_TEAM_ID"):
            monkeypatch.delenv(var, raising=False)

        status = describe_vercel_auth()
        assert status.ok is False
        assert status.label == "not configured"
        assert "recommended" in status.detail_lines[0]

    def test_token_only_with_project_missing(self, monkeypatch):
        monkeypatch.delenv("VERCEL_OIDC_TOKEN", raising=False)
        monkeypatch.setenv("VERCEL_TOKEN", "t")
        monkeypatch.setenv("VERCEL_PROJECT_ID", "p")
        monkeypatch.delenv("VERCEL_TEAM_ID", raising=False)

        status = describe_vercel_auth()
        assert status.ok is False
        assert "missing" in status.label
        assert "VERCEL_TEAM_ID" in status.label

    def test_two_of_three_access_vars(self, monkeypatch):
        monkeypatch.delenv("VERCEL_OIDC_TOKEN", raising=False)
        monkeypatch.delenv("VERCEL_TOKEN", raising=False)
        monkeypatch.setenv("VERCEL_PROJECT_ID", "p")
        monkeypatch.setenv("VERCEL_TEAM_ID", "t")

        status = describe_vercel_auth()
        assert status.ok is False
        assert "missing VERCEL_TOKEN" in status.label


class TestVercelAuthStatus:
    def test_fields(self):
        s = VercelAuthStatus(ok=False, label="L", detail_lines=("d1", "d2"))
        assert s.ok is False
        assert s.label == "L"
        assert s.detail_lines == ("d1", "d2")


class TestTokenTupleVars:
    def test_contains_three_vars(self):
        assert len(_TOKEN_TUPLE_VARS) == 3

    def test_includes_vercel_token(self):
        assert "VERCEL_TOKEN" in _TOKEN_TUPLE_VARS

    def test_includes_project_id(self):
        assert "VERCEL_PROJECT_ID" in _TOKEN_TUPLE_VARS

    def test_includes_team_id(self):
        assert "VERCEL_TEAM_ID" in _TOKEN_TUPLE_VARS
