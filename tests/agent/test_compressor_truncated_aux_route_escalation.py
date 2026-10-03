"""#113322: an aux-routed truncated summary must escalate to the main route.

A length-stopped (``finish_reason == "length"``) summary is deterministic for a
given (input, model, output budget): re-issuing the identical aux request can
never succeed. When the failed route differs from the main model, the one-shot
retry must be pinned to the main route instead of re-resolving to the same aux
route. When the failed route IS the main model (or unknown), compression still
ABORTS and preserves the session.

The pin is *authoritative*: a main runtime that leaves ``base_url``/``api_mode``
implicit must not inherit them from ``auxiliary.compression``, or the retry would
mix the main model + key with the failed auxiliary endpoint and wire mode.
"""

from unittest.mock import MagicMock, patch

from agent.auxiliary_client import (
    _get_auxiliary_task_config,
    _prepare_aux_request,
    _resolve_task_provider_model,
)
from agent.context_compressor import ContextCompressor

# The failed auxiliary route: a separate endpoint on a different wire mode. No provider, so the
# resolver's "explicit provider adopts the task endpoint" rule is exactly what leaks it.
_CONFLICTING_AUX_CONFIG = {
    "model": "aux-model",
    "base_url": "https://aux.invalid/v1",
    "api_key": "aux-key",
    "api_mode": "responses",
}


def _mock_response(content="x", finish_reason="stop"):
    resp = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    choice.finish_reason = finish_reason
    resp.choices = [choice]
    return resp


def _msgs(n=2):
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i} " + "x" * 50}
        for i in range(n)
    ]


class TestAuxRoutedTruncationEscalation:
    def test_aux_routed_truncation_retries_pinned_to_main(self):
        """The retry after an aux-route truncation must leave the aux route."""
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="main-model", provider="openrouter", api_key="main-key", quiet_mode=True,
            )
        truncated = _mock_response("partial...", "length")
        ok = _mock_response("full summary via main model", "stop")
        calls = []

        def fake_call_llm(**kw):
            calls.append(dict(kw))
            if kw.get("model") == "main-model":
                return ok
            # Simulate task routing resolving to a distinct aux model.
            route_info = kw.get("route_info")
            if isinstance(route_info, dict):
                route_info["provider"] = "google"
                route_info["model"] = "google/gemini-2.5-flash-lite"
            return truncated

        with patch("agent.context_compressor.call_llm", side_effect=fake_call_llm):
            result = c._generate_summary(_msgs())

        assert result is not None
        assert "full summary via main model" in result
        assert len(calls) == 2
        retry = calls[1]
        # The complete main route, not just the model: a pin naming only the model would still
        # resolve the auxiliary endpoint/mode for the fields the main runtime left unset.
        assert retry.get("model") == "main-model"
        assert retry.get("provider") == "openrouter"
        assert retry.get("api_key") == "main-key"
        assert retry.get("route_authoritative") is True
        # Unset on the main runtime → must stay unset (no auxiliary inheritance).
        assert "base_url" not in retry
        assert "api_mode" not in retry
        assert c._last_summary_truncated_failure is False

    def test_truncation_on_main_route_still_aborts(self):
        """No escalation when the failed route is already the main model."""
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="main-model", quiet_mode=True)
        calls = []

        def fake_call_llm(**kw):
            calls.append(dict(kw))
            route_info = kw.get("route_info")
            if isinstance(route_info, dict):
                route_info["provider"] = "auto"
                route_info["model"] = "main-model"
            return _mock_response("partial...", "length")

        with patch("agent.context_compressor.call_llm", side_effect=fake_call_llm):
            result = c._generate_summary(_msgs())

        assert result is None
        assert len(calls) == 1
        assert c._last_summary_truncated_failure is True


class TestAuthoritativeRoutePin:
    """Regression for the mixed route: main model/key + auxiliary endpoint/wire mode."""

    def test_authoritative_pin_ignores_a_conflicting_aux_task_config(self):
        with patch("agent.auxiliary_client._get_auxiliary_task_config",
                   return_value=dict(_CONFLICTING_AUX_CONFIG)):
            # Without the flag the resolver fills the unset fields from the task route — the bug.
            inherited = _resolve_task_provider_model(
                "compression", provider="openrouter", model="main-model", api_key="main-key",
            )
            assert inherited[2] == "https://aux.invalid/v1"
            assert inherited[4]

            resolved = _resolve_task_provider_model(
                "compression", provider="openrouter", model="main-model", api_key="main-key",
                route_authoritative=True,
            )
        assert resolved == ("openrouter", "main-model", None, "main-key", None)

    def test_authoritative_pin_reaches_the_client_selector_unmixed(self):
        """The client-selection boundary sees the main route only."""
        seen = {}

        def _fake_resolve_call_client(task, **kwargs):
            seen.update(kwargs)
            return MagicMock(), "main-model", "openrouter", "openrouter"

        with (
            patch("agent.auxiliary_client._get_auxiliary_task_config",
                  return_value=dict(_CONFLICTING_AUX_CONFIG)),
            patch("agent.auxiliary_client._resolve_call_client", side_effect=_fake_resolve_call_client),
        ):
            _prepare_aux_request(
                "compression", provider="openrouter", model="main-model", base_url=None,
                api_key="main-key", main_runtime={"provider": "openrouter", "model": "main-model"},
                messages=[{"role": "user", "content": "hi"}], temperature=None, max_tokens=None,
                tools=None, timeout=None, extra_body=None, reasoning_config=None,
                extra_headers=None, api_mode=None, route_info={}, async_mode=False,
                route_authoritative=True,
            )

        assert seen["resolved_base_url"] is None  # no auxiliary URL
        assert seen["resolved_api_mode"] is None  # no auxiliary wire mode
        assert seen["resolved_provider"] == "openrouter"
        assert seen["resolved_model"] == "main-model"

    def test_aux_task_config_still_applies_without_the_pin(self):
        """Sanity: ordinary auxiliary calls keep task-config routing."""
        with patch("agent.auxiliary_client._get_auxiliary_task_config",
                   return_value=dict(_CONFLICTING_AUX_CONFIG)):
            resolved = _resolve_task_provider_model("compression")
        assert resolved[1] == "aux-model"
        assert resolved[2] == "https://aux.invalid/v1"

    def test_public_call_llm_forwards_the_authoritative_pin(self, monkeypatch):
        """The pin must survive ``call_llm``'s own signature: the compressor hands it over there,
        and a missing parameter is a TypeError on the real retry path, not a routing bug."""
        from agent import auxiliary_client as ac

        seen = {}

        def _fake_impl(**kwargs):
            seen.update(kwargs)
            return _mock_response("ok")

        monkeypatch.setattr(ac, "_call_llm_impl", _fake_impl)
        monkeypatch.setattr(ac, "_acquire_sync_aux_semaphore", lambda task: None)

        ac.call_llm(
            task="compression", provider="openrouter", model="main-model", api_key="main-key",
            messages=[{"role": "user", "content": "hi"}], route_authoritative=True,
        )
        assert seen.get("route_authoritative") is True
