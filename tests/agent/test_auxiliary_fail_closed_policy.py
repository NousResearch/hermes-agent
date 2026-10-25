"""Pinned auxiliary routes must not escape to an alternate provider."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import openai  # noqa: F401 - prime SDK before the suite's real-home I/O guard activates

from agent import auxiliary_client as ac


def pinned(monkeypatch, provider="openai-codex", model="gpt-5.5"):
    monkeypatch.setattr(
        ac,
        "_get_auxiliary_task_config",
        lambda task: {
            "provider": provider,
            "model": model,
            "fail_closed": True,
        },
    )


def request(fail_closed=True):
    return SimpleNamespace(
        client=MagicMock(),
        kwargs={"messages": []},
        request_provider="openai-codex",
        resolved_api_mode=None,
        base_info=None,
        resolved_base_url=None,
        fail_closed=fail_closed,
    )


def test_sync_failure_never_enters_provider_fallback(monkeypatch):
    pinned(monkeypatch)
    failure = RuntimeError("local endpoint failed")
    with (
        patch.object(ac, "_plan_aux_call", return_value=(request(), {}, {})),
        patch.object(ac, "_relay_sync_completion", side_effect=failure),
        patch.object(ac, "_should_retry_same_provider", return_value=False),
        patch.object(ac, "_start_recovery_ladder") as fallback,
    ):
        with pytest.raises(RuntimeError, match="local endpoint failed"):
            ac.call_llm(
                task="title_generation",
                messages=[{"role": "user", "content": "private"}],
            )
    fallback.assert_not_called()


def test_async_failure_never_enters_provider_fallback(monkeypatch):
    pinned(monkeypatch)
    failure = RuntimeError("local endpoint failed")
    with (
        patch.object(ac, "_plan_aux_call", return_value=(request(), {}, {})),
        patch.object(
            ac, "_relay_async_completion", new_callable=AsyncMock, side_effect=failure
        ),
        patch.object(ac, "_should_retry_same_provider", return_value=False),
        patch.object(ac, "_start_recovery_ladder") as fallback,
    ):
        with pytest.raises(RuntimeError, match="local endpoint failed"):
            asyncio.run(
                ac._async_call_llm_impl(
                    task="title_generation",
                    messages=[{"role": "user", "content": "private"}],
                )
            )
    fallback.assert_not_called()


def test_unavailable_pinned_client_does_not_try_auto_or_configured_fallback(
    monkeypatch,
):
    pinned(monkeypatch)
    with (
        patch.object(ac, "_get_cached_client", return_value=(None, None)) as lookup,
        patch.object(
            ac, "_try_configured_fallback_for_unavailable_client"
        ) as configured,
    ):
        with pytest.raises(ac.AuxiliaryClientUnavailable):
            ac._resolve_call_client(
                "title_generation",
                provider=None,
                model=None,
                base_url=None,
                api_key=None,
                resolved_provider="openai-codex",
                resolved_model="gpt-5.5",
                resolved_base_url=None,
                resolved_api_key=None,
                resolved_api_mode=None,
                main_runtime=None,
                async_mode=False,
                fail_closed=True,
            )
    assert lookup.call_count == 1
    configured.assert_not_called()


def test_substituted_model_is_rejected_before_request(monkeypatch):
    pinned(monkeypatch)
    client = MagicMock()
    with (
        patch.object(ac, "_get_cached_client", return_value=(client, "cloud-model")),
        patch.object(ac, "_effective_provider_for_client", return_value="openai-codex"),
    ):
        with pytest.raises(ac.AuxiliaryClientUnavailable, match="substituted"):
            ac._resolve_call_client(
                "title_generation",
                provider=None,
                model=None,
                base_url=None,
                api_key=None,
                resolved_provider="openai-codex",
                resolved_model="gpt-5.5",
                resolved_base_url=None,
                resolved_api_key=None,
                resolved_api_mode=None,
                main_runtime=None,
                async_mode=True,
                fail_closed=True,
            )
    client.chat.completions.create.assert_not_called()


def test_pinned_route_rejects_caller_provider_override(monkeypatch):
    pinned(monkeypatch)
    with patch.object(ac, "_resolve_task_provider_model") as resolve:
        with pytest.raises(ValueError, match="overrides"):
            ac.call_llm(task="title_generation", provider="openai", messages=[])
    resolve.assert_not_called()


def test_pinned_route_rejects_resolver_substitution(monkeypatch):
    pinned(monkeypatch)
    with (
        patch.object(
            ac,
            "_resolve_task_provider_model",
            return_value=("openai", "cloud-model", None, None, None),
        ),
        patch.object(ac, "_get_cached_client") as client,
    ):
        with pytest.raises(ac.AuxiliaryClientUnavailable, match="pinned route"):
            ac.call_llm(task="title_generation", messages=[])
    client.assert_not_called()


@pytest.mark.parametrize("value", ["true", 1, "false", None])
def test_fail_closed_rejects_non_boolean_configuration(monkeypatch, value):
    monkeypatch.setattr(
        ac,
        "_get_auxiliary_task_config",
        lambda task: {
            "provider": "openai-codex",
            "model": "gpt-5.5",
            "fail_closed": value,
        },
    )
    with pytest.raises(ValueError, match="boolean"):
        ac.call_llm(task="title_generation", messages=[])


@pytest.mark.parametrize(
    "override", [{"base_url": "https://cloud.example/v1"}, {"api_key": "other"}]
)
def test_pinned_route_rejects_endpoint_or_key_override(monkeypatch, override):
    pinned(monkeypatch)
    with patch.object(ac, "_resolve_task_provider_model") as resolver:
        with pytest.raises(ValueError, match="overrides"):
            ac.call_llm(task="title_generation", messages=[], **override)
    resolver.assert_not_called()


def test_policy_snapshot_survives_config_change_during_request(monkeypatch):
    state = {"fail_closed": True}
    monkeypatch.setattr(
        ac,
        "_get_auxiliary_task_config",
        lambda task: {
            "provider": "openai-codex",
            "model": "gpt-5.5",
            **state,
        },
    )
    client = MagicMock()
    with (
        patch.object(
            ac,
            "_resolve_task_provider_model",
            return_value=("openai-codex", "gpt-5.5", None, None, None),
        ),
        patch.object(ac, "_get_cached_client", return_value=(client, "gpt-5.5")),
        patch.object(ac, "_effective_provider_for_client", return_value="openai-codex"),
        patch.object(ac, "_get_task_extra_body", return_value={}),
        patch.object(ac, "_effective_aux_timeout", return_value=30),
    ):
        req = ac._prepare_aux_request(
            "title_generation",
            provider=None,
            model=None,
            base_url=None,
            api_key=None,
            main_runtime=None,
            messages=[],
            temperature=None,
            max_tokens=None,
            tools=None,
            timeout=None,
            extra_body=None,
            reasoning_config=None,
            extra_headers=None,
            api_mode=None,
            async_mode=False,
            route_info=None,
        )
    state["fail_closed"] = False
    assert req.fail_closed is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("base_url", "https://cloud.invalid/v1"),
        ("api_key", "secret"),
        ("key_env", "CLOUD_KEY"),
        ("api_key_env", "CLOUD_KEY"),
        ("api_mode", "codex_responses"),
    ],
)
def test_pinned_task_rejects_configured_route_overrides(monkeypatch, field, value):
    monkeypatch.setattr(
        ac,
        "_get_auxiliary_task_config",
        lambda task: {
            "provider": "privacy-router",
            "model": "aux-local",
            "fail_closed": True,
            field: value,
        },
    )
    with patch.object(ac, "_resolve_task_provider_model") as resolver:
        with pytest.raises(ValueError, match="overrides"):
            ac.call_llm(task="title_generation", messages=[])
    resolver.assert_not_called()


def test_config_race_cannot_inject_cloud_endpoint(monkeypatch):
    pinned(monkeypatch, provider="privacy-router", model="aux-local")
    with (
        patch.object(
            ac,
            "_resolve_task_provider_model",
            return_value=(
                "privacy-router",
                "aux-local",
                "https://cloud.invalid/v1",
                None,
                None,
            ),
        ),
        patch.object(ac, "_get_cached_client") as client,
    ):
        with pytest.raises(ac.AuxiliaryClientUnavailable, match="pinned route"):
            ac.call_llm(task="title_generation", messages=[])
    client.assert_not_called()


def test_named_custom_cloud_endpoint_is_denied_even_with_matching_name(monkeypatch):
    pinned(monkeypatch, provider="privacy-router", model="aux-local")
    client = MagicMock()
    client.base_url = "https://cloud.invalid/v1"
    with (
        patch.object(ac, "_get_cached_client", return_value=(client, "aux-local")),
        patch.object(
            ac, "_effective_provider_for_client", return_value="privacy-router"
        ),
        patch(
            "providers.get_provider_profile",
            return_value=SimpleNamespace(base_url="http://127.0.0.1:54321/v1"),
        ),
    ):
        with pytest.raises(
            ac.AuxiliaryClientUnavailable, match="endpoint is not pinned"
        ):
            ac._resolve_call_client(
                "title_generation",
                provider=None,
                model=None,
                base_url=None,
                api_key=None,
                resolved_provider="privacy-router",
                resolved_model="aux-local",
                resolved_base_url=None,
                resolved_api_key=None,
                resolved_api_mode=None,
                main_runtime=None,
                async_mode=False,
                fail_closed=True,
            )
    client.chat.completions.create.assert_not_called()


def test_vision_pin_is_rejected_before_client_resolution(monkeypatch):
    pinned(monkeypatch)
    with patch.object(ac, "_resolve_task_provider_model") as resolver:
        with pytest.raises(ValueError, match="vision"):
            ac.call_llm(task="vision", messages=[])
    resolver.assert_not_called()


def test_caller_api_mode_override_is_rejected(monkeypatch):
    pinned(monkeypatch)
    with patch.object(ac, "_resolve_task_provider_model") as resolver:
        with pytest.raises(ValueError, match="overrides"):
            ac.call_llm(
                task="title_generation", messages=[], api_mode="codex_responses"
            )
    resolver.assert_not_called()


def test_pinned_sync_completion_bypasses_relay_and_aux_hooks():
    payload = {
        "messages": [{"role": "user", "content": "private"}],
        "model": "aux-local",
    }
    with patch.object(ac, "_relay_auxiliary_metadata") as relay:
        result = ac._relay_sync_completion(
            MagicMock(),
            payload,
            fail_closed=True,
            create=lambda request: request,
        )
    assert result["model"] == "aux-local"
    relay.assert_not_called()


def test_pinned_async_completion_bypasses_relay_and_aux_hooks():
    payload = {
        "messages": [{"role": "user", "content": "private"}],
        "model": "aux-local",
    }
    with patch.object(ac, "_relay_auxiliary_metadata") as relay:
        result = asyncio.run(
            ac._relay_async_completion(
                MagicMock(),
                payload,
                fail_closed=True,
                create=AsyncMock(side_effect=lambda request: request),
            )
        )
    assert result["model"] == "aux-local"
    relay.assert_not_called()


def test_mixed_case_router_name_cannot_skip_endpoint_pin(monkeypatch):
    pinned(monkeypatch, provider="Privacy-Router", model="aux-local")
    client = MagicMock()
    client.base_url = "https://cloud.invalid/v1"
    with (
        patch.object(ac, "_get_cached_client", return_value=(client, "aux-local")),
        patch.object(
            ac, "_effective_provider_for_client", return_value="Privacy-Router"
        ),
        patch(
            "providers.get_provider_profile",
            return_value=SimpleNamespace(base_url="http://127.0.0.1:54321/v1"),
        ),
    ):
        with pytest.raises(
            ac.AuxiliaryClientUnavailable, match="endpoint is not pinned"
        ):
            ac._resolve_call_client(
                "title_generation",
                provider=None,
                model=None,
                base_url=None,
                api_key=None,
                resolved_provider="Privacy-Router",
                resolved_model="aux-local",
                resolved_base_url=None,
                resolved_api_key=None,
                resolved_api_mode=None,
                main_runtime=None,
                async_mode=False,
                fail_closed=True,
            )


def test_unreadable_config_cannot_silently_drop_pin(monkeypatch):
    from hermes_cli.config import FailedConfigRead

    with (
        patch(
            "hermes_cli.config.load_config_readonly",
            return_value=FailedConfigRead(error=ValueError("broken YAML")),
        ),
        patch.object(ac, "_get_cached_client") as client,
    ):
        task_config = ac._get_auxiliary_task_config("title_generation")
        assert isinstance(task_config, FailedConfigRead)
        assert task_config.get("provider") is None
        with pytest.raises(
            ac.AuxiliaryClientUnavailable, match="config.yaml could not be read"
        ):
            ac.call_llm(task="title_generation", messages=[])
    client.assert_not_called()


def test_empty_inherited_endpoint_and_key_defaults_are_not_overrides(monkeypatch):
    monkeypatch.setattr(
        ac,
        "_get_auxiliary_task_config",
        lambda task: {
            "provider": "privacy-router",
            "model": "aux-local",
            "fail_closed": True,
            "base_url": "",
            "api_key": "",
        },
    )
    pinned_config = ac._pinned_aux_config("title_generation")
    assert pinned_config is not None
    assert pinned_config["model"] == "aux-local"


def test_fail_closed_requires_a_pinned_provider_and_model(monkeypatch):
    monkeypatch.setattr(
        ac,
        "_get_auxiliary_task_config",
        lambda task: {"fail_closed": True, "provider": "auto"},
    )
    with pytest.raises(ValueError, match="fail_closed"):
        ac.call_llm(
            task="title_generation", messages=[{"role": "user", "content": "private"}]
        )
