"""Regression: background review agent must inherit the parent agent's
full provider bundle, not just ``model`` and ``provider``.

Failure mode this guards against: if the parent's turn resolved via
``model.routes`` (e.g. owner → slate-3 at a dedicated integration
endpoint), spawning a review agent with only ``model=self.model`` and
``provider=self.provider`` triggers ``AIAgent.__init__`` to re-resolve
``base_url`` / ``api_key`` from ``resolve_provider_client``, which
reads the *config-default* ``model.base_url`` / ``model.api_key``.
That pairs the routed ``model`` with the default ``base_url`` — slate-3
lands on litellm-1 — and the review agent 401s with
``key_model_access_denied`` because the default integration's key is
scoped to a different model.

This test pins the construction shape so that regression stays caught.
"""

import threading
from unittest.mock import MagicMock, patch


def _parent_stub():
    """Build the minimum AIAgent-like stub ``_spawn_background_review``
    reads from ``self``. Avoids building a real AIAgent (expensive,
    pulls in providers, skills, etc.)."""
    from run_agent import AIAgent

    parent = object.__new__(AIAgent)
    parent.model = "slate-3"
    parent.base_url = "https://litellm-3.int.exe.xyz/v1"
    parent.api_key = "slate-3-key"
    parent.api_mode = "chat_completions"
    parent.provider = "custom"
    parent.platform = "telegram"
    parent.acp_command = None
    parent.acp_args = []
    parent._credential_pool = None
    parent._memory_store = MagicMock()
    parent._memory_enabled = True
    parent._user_profile_enabled = True
    return parent


def test_background_review_forwards_full_runtime_bundle():
    parent = _parent_stub()

    captured = {}

    def _capture(*args, **kwargs):
        captured.update(kwargs)
        agent = MagicMock()
        agent._session_messages = []
        return agent

    # Replace threading.Thread so _run_review executes inline and we
    # can assert on the construction kwargs deterministically.
    class _InlineThread:
        def __init__(self, *args, target=None, daemon=None, name=None, **kwargs):
            self._target = target

        def start(self):
            self._target()

    with patch("run_agent.AIAgent", side_effect=_capture), \
         patch.object(threading, "Thread", _InlineThread):
        parent._spawn_background_review(
            messages_snapshot=[{"role": "user", "content": "hi"}],
            review_memory=True,
            review_skills=False,
        )

    # Routed fields that previously got dropped. Missing any of these
    # triggers the ``slate-3 @ litellm-1`` mismatch.
    assert captured.get("model") == "slate-3"
    assert captured.get("base_url") == "https://litellm-3.int.exe.xyz/v1"
    assert captured.get("api_key") == "slate-3-key"
    assert captured.get("api_mode") == "chat_completions"
    assert captured.get("provider") == "custom"


def test_background_review_tolerates_missing_optional_attrs():
    """A parent without ``api_key``/``api_mode`` attributes (legacy
    construction paths, incomplete stubs) must still spawn a review
    agent without AttributeError — ``getattr`` fallbacks cover it."""
    from run_agent import AIAgent

    parent = object.__new__(AIAgent)
    parent.model = "slate-1"
    parent.base_url = "https://litellm-1.int.exe.xyz/v1"
    parent.provider = "custom"
    parent.platform = "cli"
    parent._memory_store = MagicMock()
    parent._memory_enabled = True
    parent._user_profile_enabled = True
    # Intentionally omit: api_key, api_mode, acp_command, acp_args,
    # _credential_pool

    captured = {}

    def _capture(*args, **kwargs):
        captured.update(kwargs)
        agent = MagicMock()
        agent._session_messages = []
        return agent

    class _InlineThread:
        def __init__(self, *args, target=None, daemon=None, name=None, **kwargs):
            self._target = target

        def start(self):
            self._target()

    with patch("run_agent.AIAgent", side_effect=_capture), \
         patch.object(threading, "Thread", _InlineThread):
        parent._spawn_background_review(
            messages_snapshot=[{"role": "user", "content": "hi"}],
            review_memory=True,
            review_skills=False,
        )

    assert captured.get("model") == "slate-1"
    assert captured.get("base_url") == "https://litellm-1.int.exe.xyz/v1"
    assert captured.get("api_key") == ""
    assert captured.get("api_mode") == ""
