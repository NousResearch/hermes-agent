import json
from unittest.mock import patch

from agent import anthropic_adapter


def _config(tmp_path):
    token_file = tmp_path / "identity.jwt"
    token_file.write_text("identity-assertion", encoding="utf-8")
    return {
        "federation_rule_id": "fdrl_test",
        "organization_id": "org_test",
        "service_account_id": "svac_test",
        "identity_token_file": str(token_file),
        "api_base_url": "https://api.anthropic.com",
    }


def test_wif_token_provider_rechecks_exchange_for_every_request(tmp_path):
    config = _config(tmp_path)
    with patch.object(
        anthropic_adapter,
        "exchange_anthropic_wif_for_access_token",
        side_effect=[{"access_token": "first"}, {"access_token": "refreshed"}],
    ) as exchange:
        provider = anthropic_adapter.build_anthropic_wif_token_provider(config)

        assert provider() == "first"
        assert provider() == "refreshed"

    assert exchange.call_count == 2
    assert exchange.call_args_list[0].args[0] == config


def test_wif_token_provider_rejects_empty_exchange_result(tmp_path):
    with patch.object(
        anthropic_adapter,
        "exchange_anthropic_wif_for_access_token",
        return_value={},
    ):
        provider = anthropic_adapter.build_anthropic_wif_token_provider(_config(tmp_path))
        try:
            provider()
        except ValueError as exc:
            assert "no access token" in str(exc).lower()
        else:  # pragma: no cover
            raise AssertionError("empty WIF exchange must fail")


def test_clear_wif_token_cache_purges_memory_and_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(anthropic_adapter, "get_hermes_home", lambda: tmp_path)
    config = _config(tmp_path)
    provider = anthropic_adapter.build_anthropic_wif_token_provider(config)
    cache_file = tmp_path / anthropic_adapter._WIF_CACHE_FILE_NAME
    cache_file.write_text('{"access_token":"secret"}', encoding="utf-8")
    anthropic_adapter._WIF_ACCESS_TOKEN_CACHE["key"] = {"access_token": "secret"}

    anthropic_adapter.clear_anthropic_wif_token_cache()

    assert anthropic_adapter._WIF_ACCESS_TOKEN_CACHE == {}
    assert anthropic_adapter._WIF_TOKEN_PROVIDERS == {}
    assert not cache_file.exists()
    try:
        provider()
    except RuntimeError as exc:
        assert "removed" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("an issued provider must be invalidated on logout")


def test_wif_provider_is_stable_for_unchanged_config(tmp_path):
    config = _config(tmp_path)
    first = anthropic_adapter.build_anthropic_wif_token_provider(config)
    second = anthropic_adapter.build_anthropic_wif_token_provider(dict(config))
    assert first is second


def test_wif_stale_config_cannot_register_after_invalidation(tmp_path):
    config = _config(tmp_path)
    generation = anthropic_adapter.get_anthropic_wif_provider_generation()

    anthropic_adapter.clear_anthropic_wif_token_cache()

    try:
        anthropic_adapter.build_anthropic_wif_token_provider(
            config, expected_generation=generation
        )
    except anthropic_adapter.AnthropicWIFGenerationChanged:
        pass
    else:  # pragma: no cover
        raise AssertionError("stale WIF config must not survive invalidation")
    assert anthropic_adapter._WIF_TOKEN_PROVIDERS == {}


def test_managed_wif_provider_rechecks_persisted_identity_on_each_call(
    tmp_path, monkeypatch
):
    config = _config(tmp_path)
    current = {"value": dict(config)}
    monkeypatch.setattr(
        anthropic_adapter,
        "read_anthropic_wif_config",
        lambda: current["value"],
    )
    monkeypatch.setattr(
        anthropic_adapter,
        "exchange_anthropic_wif_for_access_token",
        lambda _config: {"access_token": "managed-token"},
    )
    provider = anthropic_adapter.build_anthropic_wif_token_provider(
        config,
        expected_generation=anthropic_adapter.get_anthropic_wif_provider_generation(),
    )

    assert provider() == "managed-token"
    current["value"] = None
    try:
        provider()
    except RuntimeError as exc:
        assert "removed" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("managed provider must reject removed auth state")


def test_wif_cache_read_refuses_destination_symlink(tmp_path, monkeypatch):
    monkeypatch.setattr(anthropic_adapter, "get_hermes_home", lambda: tmp_path)
    target = tmp_path / "target.json"
    target.write_text(
        json.dumps({"cache_key": "key", "access_token": "stolen"}),
        encoding="utf-8",
    )
    cache_file = tmp_path / anthropic_adapter._WIF_CACHE_FILE_NAME
    cache_file.symlink_to(target)

    assert anthropic_adapter._read_cached_wif_exchange("key") is None


def test_wif_cache_write_replaces_symlink_not_its_target(tmp_path, monkeypatch):
    monkeypatch.setattr(anthropic_adapter, "get_hermes_home", lambda: tmp_path)
    target = tmp_path / "valuable.txt"
    target.write_text("must-survive", encoding="utf-8")
    cache_file = tmp_path / anthropic_adapter._WIF_CACHE_FILE_NAME
    cache_file.symlink_to(target)

    anthropic_adapter._write_cached_wif_exchange(
        "key",
        {"access_token": "bearer", "expires_at_ms": 9_999_999_999_999},
    )

    assert target.read_text(encoding="utf-8") == "must-survive"
    assert not cache_file.is_symlink()
    assert json.loads(cache_file.read_text(encoding="utf-8"))["cache_key"] == "key"
