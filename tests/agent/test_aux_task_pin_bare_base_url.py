"""Contract tests for provider-less ``auxiliary.<task>`` pins (model + base_url, no provider).

The aux-block schema documents that ``base_url`` overrides ``provider``, and
``_resolve_task_provider_model``'s own docstring states "A bare base_url means
custom". Before the fix, a task block configured with ``model`` + ``base_url``
and no ``provider`` (and no ``api_key``) fell through to ``"auto"``: the
configured endpoint was DROPPED from the resolution tuple while the pinned
model name was kept. The downstream auto route then built a client against the
LIVE session's provider, so a local compression pin (``model: Qwen3.8-27B``,
``base_url: http://127.0.0.1:8000/v1``) was POSTed to whatever aggregator the
chat session happened to use, and billed there.

Each test pins one edge of the resolution matrix; the "survives" tests assert
the endpoint the wire client is actually built against.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

_PIN_CFG = {
    "model": "Qwen3.8-27B",
    "base_url": "http://127.0.0.1:8000/v1",
}

# A live session on a paid aggregator (the shape that triggered the incident).
_MAIN_RT_PAID = {
    "model": "deepseek/deepseek-v4.1-flash-1017",
    "provider": "openrouter",
    "base_url": "https://openrouter.ai/api/v1",
    "api_key": "sk-or-test",
    "api_mode": None,
}


def _resolve_task(task="compression", cfg=None):
    from agent.auxiliary_client import _resolve_task_provider_model

    with patch(
        "agent.auxiliary_client._get_auxiliary_task_config",
        return_value=dict(cfg if cfg is not None else _PIN_CFG),
    ):
        return _resolve_task_provider_model(task)


def _build_client_for_resolution(res, main_runtime):
    """Drive resolve_provider_client the way _prepare_aux_request/_resolve_call_client
    does, and capture the base_url the OpenAI client is actually constructed with."""
    from agent.auxiliary_client import resolve_provider_client

    captured: dict = {}

    def _fake_create(*, api_key, base_url, **kwargs):
        captured["api_key"] = api_key
        captured["base_url"] = base_url
        client = MagicMock()
        client.base_url = base_url
        return client

    with patch("agent.auxiliary_client._create_openai_client", side_effect=_fake_create):
        client, model = resolve_provider_client(
            res[0], res[1],
            explicit_base_url=res[2], explicit_api_key=res[3], api_mode=res[4],
            main_runtime=main_runtime, task="compression",
        )
    return client, model, captured


class TestBareBaseUrlPinIsCustom:
    def test_bare_base_url_pin_resolves_custom_not_auto(self):
        """model + base_url with no provider must resolve as a custom endpoint."""
        provider, model, base_url, api_key, api_mode = _resolve_task()
        assert provider == "custom"
        assert model == "Qwen3.8-27B"
        assert base_url == "http://127.0.0.1:8000/v1"
        assert api_key is None
        assert api_mode is None

    def test_bare_base_url_pin_endpoint_survives_paid_session(self):
        """The client must be built against the pinned local endpoint even while
        the chat session rides a paid aggregator (the incident scenario)."""
        res = _resolve_task()
        _client, model, captured = _build_client_for_resolution(res, _MAIN_RT_PAID)
        assert captured["base_url"] == "http://127.0.0.1:8000/v1"
        assert "openrouter.ai" not in str(captured["base_url"])
        assert model == "Qwen3.8-27B"

    def test_bare_base_url_with_auto_provider_string_also_custom(self):
        """An explicit ``provider: auto`` alongside base_url is still a bare
        endpoint: the aux-block schema says base_url overrides provider."""
        res = _resolve_task(cfg={**_PIN_CFG, "provider": "auto"})
        assert res[0] == "custom"
        assert res[2] == "http://127.0.0.1:8000/v1"


class TestExistingPrecedencePreserved:
    def test_model_only_pin_still_falls_through_to_auto(self):
        """Without base_url there is no endpoint to honor; auto behavior is unchanged."""
        provider, model, base_url, api_key, api_mode = _resolve_task(
            cfg={"model": "some-pinned-model"})
        assert (provider, model, base_url, api_key, api_mode) == (
            "auto", "some-pinned-model", None, None, None)

    def test_provider_plus_base_url_keeps_provider_identity(self):
        """A named provider alongside base_url keeps its identity (auth/transport
        shaping) - the branch above the new one must not regress."""
        res = _resolve_task(cfg={"provider": "deepinfra", "model": "m",
                                 "base_url": "https://deepinfra.example/v1"})
        assert res[0] == "deepinfra"
        assert res[2] == "https://deepinfra.example/v1"

    def test_base_url_with_api_key_branch_unchanged(self):
        """base_url + api_key (no provider) resolved as custom before the fix and
        still does."""
        res = _resolve_task(cfg={"model": "m", "base_url": "http://127.0.0.1:9/v1",
                                 "api_key": "k"})
        assert res[0] == "custom"
        assert res[2] == "http://127.0.0.1:9/v1"
        assert res[3] == "k"

    def test_empty_task_config_still_auto(self):
        """No pin at all: unchanged auto behavior (compression rides the session)."""
        provider, model, base_url, api_key, api_mode = _resolve_task(cfg={})
        assert (provider, model, base_url, api_key, api_mode) == (
            "auto", None, None, None, None)
