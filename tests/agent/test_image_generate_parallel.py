"""Regression tests for parallel image-generation tool batches."""

import json
from types import SimpleNamespace
from unittest.mock import patch

from agent import tool_executor


def _tool_call(name: str, args: dict, call_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(args),
        ),
    )






def test_image_generate_parallel_worker_cap_can_be_configured_lower():
    runnable_calls = [
        (
            0,
            _tool_call("image_generate", {"prompt": "one"}, "img_1"),
            "image_generate",
            {},
        ),
        (
            1,
            _tool_call("image_generate", {"prompt": "two"}, "img_2"),
            "image_generate",
            {},
        ),
    ]

    with patch(
        "hermes_cli.config.load_config",
        return_value={"image_gen": {"max_parallel_requests": 1}},
    ):
        assert tool_executor._max_workers_for_tool_batch(runnable_calls) == 1
