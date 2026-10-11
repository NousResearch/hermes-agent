"""A retired model id is a dead MODEL, not a dead provider.

Split out of ``tests/agent/test_auxiliary_client.py``: that file is past its line ratchet and may
only shrink, and new tests belong in a ``test_<stem>_<topic>`` sibling.

Covers the 404 a versioned ``:free`` SKU produces when the catalog retires it
(``meituan/longcat-2.0:free``): the body left every predicate False, so the request was admitted
to NO fallback reason and the aux task (title generation in production) failed on every call for
days. The classifier half of the same regression lives in
``test_error_classifier_retired_models.py``.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.auxiliary_client import (
    _FALLBACK_REASONS,
    _is_model_not_found_error,
    _is_payment_error,
    call_llm,
)
from tests.agent.test_auxiliary_client import _DummyResponse


def test_retired_free_sku_404_is_model_not_found():
    """A retired versioned SKU (:free previews) is a dead MODEL ID, not credit depletion.

    Regression: the exact production body left every predicate False, so the request was
    admitted to no fallback reason and the aux task failed on every call for days.
    """
    exc = Exception(
        "Error code: 404 - {'status': 404, 'message': \"This model is no longer free. "
        "To continue using the paid variant, switch to 'meituan/longcat-2.0'.\"}"
    )
    exc.status_code = 404
    assert _is_model_not_found_error(exc) is True
    assert _is_payment_error(exc) is False
    assert next(
        (label for predicate, label in _FALLBACK_REASONS if predicate(exc)), None
    ) == "model not found"


class _RetiredModel404(Exception):
    """The exact production body: a ``:free`` SKU the Nous catalog retired."""

    status_code = 404

    def __init__(self, model="meituan/longcat-2.0:free"):
        super().__init__(
            "Error code: 404 - {'status': 404, 'message': \"This model is no longer free. "
            f"To continue using the paid variant, switch to '{model}'.\"}}"
        )


class TestRetiredModelFallback:
    """A retired model id enters the fallback chain, and only through the task's own chain."""

    def _call(self, task_config, chain_result, main_result=(None, None, "")):
        primary_client = MagicMock()
        primary_client.base_url = "https://inference-api.nousresearch.com/v1"
        primary_client.chat.completions.create.side_effect = _RetiredModel404()
        fallback_client = MagicMock()
        fallback_client.chat.completions.create.return_value = _DummyResponse("fallback title")

        with patch("agent.auxiliary_client._get_cached_client",
                   return_value=(primary_client, "meituan/longcat-2.0:free")), \
             patch("agent.auxiliary_client._resolve_task_provider_model",
                   return_value=("nous", "meituan/longcat-2.0:free", None, None, None)), \
             patch("agent.auxiliary_client._get_auxiliary_task_config",
                   return_value=task_config), \
             patch("agent.auxiliary_client._refresh_nous_recommended_model",
                   return_value=None), \
             patch("agent.auxiliary_client._try_configured_fallback_chain",
                   return_value=chain_result) as mock_chain, \
             patch("agent.auxiliary_client._try_main_agent_model_fallback",
                   return_value=main_result) as mock_main:
            out = None
            error = None
            try:
                out = call_llm(task="title_generation",
                               messages=[{"role": "user", "content": "title this"}])
            except Exception as exc:  # noqa: BLE001
                error = exc
        return SimpleNamespace(out=out, error=error, chain=mock_chain, main=mock_main)

    def test_retired_pin_walks_the_configured_task_chain(self):
        """The pin is dead but the task asked for a chain: walk it, do not surface the 404."""
        result = self._call(
            task_config={"fallback_chain": [{"provider": "openai-codex", "model": "gpt-6-luna"}]},
            chain_result=(MagicMock(**{"chat.completions.create.return_value":
                                       _DummyResponse("fallback title")}),
                          "gpt-6-luna", "fallback_chain[0](openai-codex)"),
        )
        assert result.error is None
        assert result.chain.call_args.kwargs["reason"] == "model not found"
        # Only the dead MODEL is skipped; the provider stays usable for its other models.
        assert result.chain.call_args.kwargs["failed_model"] == "meituan/longcat-2.0:free"
        assert not result.main.called

    def test_retired_pin_without_a_configured_chain_does_not_re_route(self):
        """No chain on an explicit pin: the failure surfaces instead of a silent model swap."""
        result = self._call(task_config={}, chain_result=(None, None, ""),
                            main_result=(MagicMock(), "main-model", "main-agent(nous)"))
        assert result.error is not None
        assert "no longer free" in str(result.error)
        assert not result.main.called