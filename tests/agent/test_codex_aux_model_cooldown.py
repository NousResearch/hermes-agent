"""A model bench must not hide the Codex login from an auxiliary call for another model.

``model_cooldown_until`` blocks an unscoped lookup on any active model cooldown.
Compression and the raw Codex path know the model they are about to call, and
they pass it through. A caller that omits the model stays unscoped.
"""
import json
import socket
import time

import pytest

BENCHED = "gpt-5.4-mini"
OTHER = "gpt-5.4"
TOKEN = "codex-aux-scope-token"
YEAR = 365 * 24 * 60 * 60


def _refuse(*_args, **_kwargs):
    """Fail the test if credential resolution opens a socket."""
    raise AssertionError("Codex auxiliary lookup made a network call")


@pytest.fixture
def login(tmp_path, monkeypatch):
    """A disposable pool-only Codex login with a year-long bench on one model."""
    root = tmp_path / "hermes-root"
    root.mkdir()
    (tmp_path / "fakehome").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "fakehome"))
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.delenv("HERMES_CODEX_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import hermes_constants
    hermes_constants._default_hermes_root_memo = None  # type: ignore[attr-defined]
    until = time.time() + YEAR
    (root / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {},
        "credential_pool": {"openai-codex": [{
            "id": "cred-0",
            "label": "acct",
            "auth_type": "oauth",
            "priority": 0,
            "source": "manual",
            "access_token": TOKEN,
            "refresh_token": "rt-0",
            "expires_at_ms": 4_000_000_000_000,
            "model_cooldowns": {BENCHED: until},
        }]},
    }))
    monkeypatch.setattr(socket, "create_connection", _refuse)
    monkeypatch.setattr(socket, "getaddrinfo", _refuse)
    return root / "auth.json"


def _saved(path):
    """The pool row's token and model benches, which resolution must not rewrite."""
    row = json.loads(path.read_text())["credential_pool"]["openai-codex"][0]
    return row["access_token"], row["model_cooldowns"]


def test_scoped_lookup_serves_the_other_model_and_refuses_the_bench(login):
    """Building a client for the other model works. Building one for the bench does not."""
    from agent.auxiliary_client import (
        CodexAuxiliaryClient,
        _build_codex_client,
        _resolve_codex_credential_and_base,
        resolve_provider_client,
    )

    before = _saved(login)
    token, _base = _resolve_codex_credential_and_base(OTHER)
    assert token == TOKEN
    missing, _missing_base = _resolve_codex_credential_and_base(BENCHED)
    assert missing is None

    client, model = _build_codex_client(OTHER)
    assert isinstance(client, CodexAuxiliaryClient)
    assert model == OTHER
    assert client.api_key == TOKEN
    try:
        refused, refused_model = _build_codex_client(BENCHED)
        assert (refused, refused_model) == (None, None)

        raw, raw_model = resolve_provider_client("openai-codex", model=OTHER, raw_codex=True)
        assert raw is not None
        assert raw_model == OTHER
        assert raw.api_key == TOKEN
        try:
            raw_refused, raw_refused_model = resolve_provider_client(
                "openai-codex", model=BENCHED, raw_codex=True,
            )
            assert (raw_refused, raw_refused_model) == (None, None)
        finally:
            raw.close()
    finally:
        client.close()

    assert _saved(login)[0] == before[0]
    assert set(_saved(login)[1]) == {BENCHED}
    assert _saved(login)[1][BENCHED] > time.time() + YEAR / 2


def test_omitted_model_stays_unscoped(login):
    """A caller that does not name a model still sees the bench as no credential."""
    from agent.auxiliary_client import _resolve_codex_credential_and_base

    token, _base = _resolve_codex_credential_and_base()
    assert token is None
    assert set(_saved(login)[1]) == {BENCHED}
