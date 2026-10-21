"""The AWS credential probe the provider ladder runs: one walk of boto3's chain per question, and no
instance-metadata lookup from the desktop's own backend (a laptop, where it only costs seconds)."""

from __future__ import annotations

import pytest

from agent import bedrock_adapter
from tests.hermes_cli.anon_portal import as_local_desktop_backend

_AWS_VARS = ("AWS_BEARER_TOKEN_BEDROCK", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_PROFILE",
             "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "AWS_WEB_IDENTITY_TOKEN_FILE")


@pytest.fixture
def chain(monkeypatch):
    for var in _AWS_VARS:
        monkeypatch.delenv(var, raising=False)
    walks: list[dict] = []
    monkeypatch.setattr(bedrock_adapter, "_boto3_chain_has_credentials",
                        lambda **kw: (walks.append(kw), False)[1])
    return walks


def test_no_credentials_walks_the_chain_once(chain):
    assert bedrock_adapter.has_aws_credentials() is False
    assert len(chain) == 1


def test_the_desktop_backends_auto_detect_skips_instance_metadata(chain, monkeypatch):
    from hermes_cli.auth import resolve_provider
    from hermes_cli.auth_constants import AuthError

    as_local_desktop_backend(monkeypatch)
    with pytest.raises(AuthError):
        resolve_provider("auto", skip_free_tier=True)

    assert chain and all(walk.get("instance_metadata") is False for walk in chain)
