"""Completion-script hook tests (Task 13, #80531).

A route's ``completion_script`` runs after the agent run finishes (the true
end of the run), using the same sandboxed script runner as transform scripts.
Its return value is ignored — it is a fire-and-forget hook, never a second
delivery path.
"""

from __future__ import annotations

import pytest

from unittest.mock import MagicMock

from gateway.config import PlatformConfig
from gateway.platforms.webhook import WebhookAdapter


def _adapter(routes, **extra):
    return WebhookAdapter(
        PlatformConfig(
            enabled=True,
            extra={"host": "127.0.0.1", "port": 0, "routes": routes, **extra},
        )
    )


class TestCompletionScript:
    def test_no_script_is_noop(self):
        adapter = _adapter({"r": {"secret": "s", "prompt": "p"}})
        adapter._delivery_info["webhook:r:d1"] = {}
        # Should not raise and should not invoke the script runner.
        adapter._route_processor = MagicMock()
        import asyncio
        asyncio.run(adapter._run_completion_script("webhook:r:d1", "done"))
        adapter._route_processor.run_route_script.assert_not_called()

    def test_script_is_invoked_with_envelope(self):
        adapter = _adapter({"r": {"secret": "s", "prompt": "p"}})
        adapter._delivery_info["webhook:r:d1"] = {"completion_script": "hook.py"}
        adapter._route_processor = MagicMock()
        adapter._route_processor.run_route_script = MagicMock(return_value=(True, {}))
        import asyncio
        asyncio.run(adapter._run_completion_script("webhook:r:d1", "done"))
        adapter._route_processor.run_route_script.assert_called_once()
        args = adapter._route_processor.run_route_script.call_args[0]
        assert args[0] == "hook.py"
        assert args[1]["chat_id"] == "webhook:r:d1"

    def test_script_failure_is_best_effort(self):
        adapter = _adapter({"r": {"secret": "s", "prompt": "p"}})
        adapter._delivery_info["webhook:r:d1"] = {"completion_script": "hook.py"}
        adapter._route_processor = MagicMock()
        adapter._route_processor.run_route_script = MagicMock(
            side_effect=RuntimeError("boom")
        )
        import asyncio
        # Must not raise.
        asyncio.run(adapter._run_completion_script("webhook:r:d1", "done"))
