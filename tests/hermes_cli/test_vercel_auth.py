"""Tests for hermes_cli/vercel_auth.py — describe_vercel_auth and VercelAuthStatus."""

import os
import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True)
def clean_vercel_env(monkeypatch):
    for v in ("VERCEL_TOKEN", "VERCEL_PROJECT_ID", "VERCEL_TEAM_ID", "VERCEL_OIDC_TOKEN"):
        monkeypatch.delenv(v, raising=False)


def test_dataclass_frozen():
    from hermes_cli.vercel_auth import VercelAuthStatus
    s = VercelAuthStatus(True, "label", ("a", "b"))
    assert s.ok is True
    assert s.label == "label"
    assert s.detail_lines == ("a", "b")
    with pytest.raises(Exception):
        s.ok = False


def test_no_vars_set():
    from hermes_cli.vercel_auth import describe_vercel_auth
    status = describe_vercel_auth()
    assert status.ok is False
    assert "missing" in status.label.lower()


def test_oidc_alone():
    from hermes_cli.vercel_auth import describe_vercel_auth
    with patch.dict(os.environ, {"VERCEL_OIDC_TOKEN": "tok"}):
        status = describe_vercel_auth()
    assert status.ok is True
    assert "OIDC" in status.label


def test_vercel_token_set_but_project_missing():
    from hermes_cli.vercel_auth import describe_vercel_auth
    with patch.dict(os.environ, {"VERCEL_TOKEN": "tok", "VERCEL_PROJECT_ID": ""}):
        status = describe_vercel_auth()
    assert status.ok is False


def test_all_token_vars_present():
    from hermes_cli.vercel_auth import describe_vercel_auth
    env = {"VERCEL_TOKEN": "tok", "VERCEL_PROJECT_ID": "proj", "VERCEL_TEAM_ID": "team"}
    with patch.dict(os.environ, env):
        status = describe_vercel_auth()
    assert status.ok is True
    assert "access token" in status.label.lower()
