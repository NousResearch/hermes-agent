"""Regression tests for the bare-``custom`` credential branch of
``_creds_for_switched_provider`` (``hermes_cli/model_switch.py``) when the
session has never recorded an api_key.

The gateway's first ``/model`` switch after boot builds its
``_ModelSwitchContext`` from ``read_config()`` alone (no session override
exists yet, so ``current_api_key`` stays ``""``). The picker always passes
``explicit_provider="custom"`` for a bare-custom row, which routes through
this branch even though the provider did not change. Before the fix, an
empty ``current_api_key`` was carried straight through as the switch
result's ``api_key`` with no attempt to re-resolve it from config, unlike
the "arriving from another provider" branch a few lines above, which does
re-resolve.
"""

from unittest.mock import patch

from hermes_cli.model_switch import switch_model

_MOCK_VALIDATION = {
    "accepted": True,
    "persist": True,
    "recognized": True,
    "message": None,
}


def _patch_switch_collaborators():
    return (
        patch("hermes_cli.models_validate.validate_requested_model", lambda *a, **k: _MOCK_VALIDATION),
        patch("hermes_cli.model_switch.get_model_info", lambda *a, **k: None),
        patch("hermes_cli.model_switch.get_model_capabilities", lambda *a, **k: None),
    )


def test_bare_custom_empty_session_key_reresolves_on_openrouter_host(monkeypatch, tmp_path):
    """The reproduction shape: a bare-custom endpoint that happens to be openrouter.ai, with the
    key available in the environment. current_api_key=="" (the first switch of a fresh session)
    must not stay empty: the resolver lands back on this exact endpoint, so the key it resolved
    for that endpoint is adopted."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "model:\n"
        "  default: openai/gpt-4o-mini\n"
        "  provider: custom\n"
        "  base_url: https://openrouter.ai/api/v1\n"
        "  api_mode: chat_completions\n"
        "  api_key: ${OPENAI_API_KEY}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-real-openrouter-key")

    with _patch_switch_collaborators()[0], _patch_switch_collaborators()[1], _patch_switch_collaborators()[2]:
        result = switch_model(
            raw_input="openai/gpt-4.1-mini",
            current_provider="custom",
            current_model="openai/gpt-4o-mini",
            current_base_url="https://openrouter.ai/api/v1",
            current_api_key="",
            explicit_provider="custom",
            user_providers={},
            custom_providers=[],
        )

    assert result.success is True
    assert result.target_provider == "custom"
    assert result.base_url == "https://openrouter.ai/api/v1"
    assert result.api_key == "sk-test-real-openrouter-key"


def test_bare_custom_empty_session_key_reresolves_on_local_host(monkeypatch, tmp_path):
    """Same shape, a local (loopback) base_url with a literal configured key."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "model:\n"
        "  default: qwen3:8b\n"
        "  provider: custom\n"
        "  base_url: http://127.0.0.1:11434/v1\n"
        "  api_key: sk-local-endpoint-secret\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with _patch_switch_collaborators()[0], _patch_switch_collaborators()[1], _patch_switch_collaborators()[2]:
        result = switch_model(
            raw_input="qwen3:9b",
            current_provider="custom",
            current_model="qwen3:8b",
            current_base_url="http://127.0.0.1:11434/v1",
            current_api_key="",
            explicit_provider="custom",
            user_providers={},
            custom_providers=[],
        )

    assert result.success is True
    assert result.target_provider == "custom"
    assert result.base_url == "http://127.0.0.1:11434/v1"
    assert result.api_key == "sk-local-endpoint-secret"


def test_bare_custom_empty_session_key_keeps_it_empty_when_resolver_lands_elsewhere(monkeypatch, tmp_path):
    """No key must be adopted when the re-resolve does not land back on THIS session's own
    endpoint (e.g. config now points bare custom somewhere else): the empty key stays empty
    rather than silently borrowing an unrelated endpoint's credential."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "model:\n"
        "  default: some-model\n"
        "  provider: custom\n"
        "  base_url: https://other.example.test/v1\n"
        "  api_key: sk-other-endpoint-secret\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with _patch_switch_collaborators()[0], _patch_switch_collaborators()[1], _patch_switch_collaborators()[2]:
        result = switch_model(
            raw_input="some-model-2",
            current_provider="custom",
            current_model="some-model",
            current_base_url="https://session.example.test/v1",
            current_api_key="",
            explicit_provider="custom",
            user_providers={},
            custom_providers=[],
        )

    assert result.success is True
    assert result.target_provider == "custom"
    assert result.base_url == "https://session.example.test/v1"
    assert result.api_key == ""


def test_bare_custom_present_session_key_is_never_touched(monkeypatch, tmp_path):
    """Never change behaviour when a current key is present: no re-resolve call is even made."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "model:\n  default: m\n  provider: custom\n  base_url: https://openrouter.ai/api/v1\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    with _patch_switch_collaborators()[0], _patch_switch_collaborators()[1], _patch_switch_collaborators()[2], \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider") as mock_resolve:
        result = switch_model(
            raw_input="m2",
            current_provider="custom",
            current_model="m",
            current_base_url="https://openrouter.ai/api/v1",
            current_api_key="sk-existing-session-key",
            explicit_provider="custom",
            user_providers={},
            custom_providers=[],
        )

    mock_resolve.assert_not_called()
    assert result.success is True
    assert result.api_key == "sk-existing-session-key"
    assert result.base_url == "https://openrouter.ai/api/v1"


def test_bare_custom_empty_session_key_not_adopted_from_the_openrouter_mirror(monkeypatch, tmp_path):
    """With no custom endpoint configured, the resolver reaches the OpenRouter rung; when
    ``OPENROUTER_BASE_URL`` equals the session's endpoint the URLs match, but the key is the
    OpenRouter credential, not one configured for this custom endpoint: it stays empty."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "config.yaml").write_text("model:\n  default: m\n  provider: custom\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "CUSTOM_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://mirror.example.test/api/v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-mirror-key")

    with _patch_switch_collaborators()[0], _patch_switch_collaborators()[1], _patch_switch_collaborators()[2]:
        result = switch_model(
            raw_input="m2",
            current_provider="custom",
            current_model="m",
            current_base_url="https://mirror.example.test/api/v1",
            current_api_key="",
            explicit_provider="custom",
            user_providers={},
            custom_providers=[],
        )

    assert result.success is True
    assert result.base_url == "https://mirror.example.test/api/v1"
    assert result.api_key == ""


def test_bare_custom_empty_session_key_drops_headers_of_an_endpoint_it_did_not_adopt(monkeypatch, tmp_path):
    """When the re-resolve lands elsewhere, the resolver's extra headers belong to that other
    endpoint and must not be sent while validating against the session's endpoint."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "config.yaml").write_text("model:\n  default: m\n  provider: custom\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    seen = {}

    def _validate(*a, **k):
        seen["headers"] = k.get("headers")
        seen["base_url"] = k.get("base_url")
        return _MOCK_VALIDATION

    foreign = {"api_key": "sk-foreign", "base_url": "https://other.example.test/v1",
               "api_mode": "chat_completions", "extra_headers": {"X-Foreign-Token": "secret"}}
    with patch("hermes_cli.models_validate.validate_requested_model", _validate), \
         _patch_switch_collaborators()[1], _patch_switch_collaborators()[2], \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=foreign):
        result = switch_model(
            raw_input="m2",
            current_provider="custom",
            current_model="m",
            current_base_url="https://session.example.test/v1",
            current_api_key="",
            explicit_provider="custom",
            user_providers={},
            custom_providers=[],
        )

    assert result.success is True
    assert result.api_key == ""
    assert seen["base_url"] == "https://session.example.test/v1"
    assert "X-Foreign-Token" not in (seen["headers"] or {})
