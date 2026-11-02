"""A named profile served by a multiplexing gateway reads provider keys only from its own ``.env``.

When the endpoint's config declares ``key_env`` and that variable is missing from the profile's
scope (it may exist in the server's process environment, which the profile must not read), the
turn fails before any request with the provider, the variable, the profile and the command that
stores the key, instead of sending the ``no-key-required`` placeholder and reporting the provider's
401 as "rejected your API key". A custom block with no ``key_env`` is a keyless local server and
keeps the placeholder.
"""

import json

import pytest

from agent import secret_scope


def _named_profile(tmp_path, monkeypatch, *, key_env: bool, dotenv: str = ""):
    home = tmp_path / ".hermes" / "profiles" / "beta"
    home.mkdir(parents=True)
    model = {"default": "fake-model", "provider": "custom", "base_url": "http://127.0.0.1:18991/v1"}
    entry = {"name": "FakeUp", "base_url": "http://127.0.0.1:18991/v1"}
    if key_env:
        entry["key_env"] = "CANARY_KEY"
    (home / "config.yaml").write_text(json.dumps({"model": model, "custom_providers": [entry]}))
    (home / ".env").write_text(dotenv)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    # The key lives only in the server's process environment: invisible to a routed profile.
    monkeypatch.setenv("CANARY_KEY", "canary-process-env")
    return home


@pytest.fixture
def multiplexed():
    tokens = []

    def bind(home):
        tokens.append(secret_scope.set_secret_scope(secret_scope.build_profile_secret_scope(home)))

    secret_scope.set_multiplex_active(True)
    try:
        yield bind
    finally:
        for token in reversed(tokens):
            secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(False)


def test_declared_key_missing_from_profile_scope_fails_before_the_request(tmp_path, monkeypatch, multiplexed):
    from hermes_cli.auth import AuthError
    from hermes_cli.runtime_provider import resolve_runtime_provider

    multiplexed(_named_profile(tmp_path, monkeypatch, key_env=True))
    with pytest.raises(AuthError) as excinfo:
        resolve_runtime_provider()
    message = str(excinfo.value)
    assert excinfo.value.code == "missing_api_key"
    for needle in ("FakeUp", "CANARY_KEY", "'beta'", "hermes -p beta config set CANARY_KEY"):
        assert needle in message
    assert "server's environment" in message  # the key the user exported is named as unreadable here
    assert "canary-process-env" not in message


@pytest.mark.parametrize("key_env,dotenv,expected_key", [
    (False, "", "no-key-required"),  # keyless local server: placeholder, no error
    (True, "CANARY_KEY=canary-profile-dotenv\n", "canary-profile-dotenv"),  # key in the profile's .env
], ids=["keyless-local-endpoint", "key-in-profile-dotenv"])
def test_keyless_endpoint_and_profile_key_still_resolve(tmp_path, monkeypatch, multiplexed, key_env, dotenv,
                                                        expected_key):
    from hermes_cli.runtime_provider import resolve_runtime_provider

    multiplexed(_named_profile(tmp_path, monkeypatch, key_env=key_env, dotenv=dotenv))
    runtime = resolve_runtime_provider()
    assert runtime["provider"] == "custom"
    assert runtime["api_key"] == expected_key
