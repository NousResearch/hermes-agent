"""P10 read-only ComfyUI REST contract — no generation submission capability."""
from __future__ import annotations

from collections.abc import Mapping

import pytest

from comfyui_rest_contract import (
    ComfyUIContractError,
    ComfyUIReadOnlyProbe,
    REQUIRED_NODES,
)


class FakeReader:
    def __init__(self, responses: dict[str, object]) -> None:
        self.responses = responses
        self.paths: list[str] = []

    def __call__(self, url: str) -> object:
        self.paths.append(url)
        path = url.removeprefix("http://127.0.0.1:8188")
        return self.responses[path]


def _valid_responses() -> dict[str, object]:
    return {
        "/system_stats": {"system": {"comfyui_version": "0.27.0"}, "devices": [{"name": "cuda:0"}]},
        "/object_info": {node: {} for node in REQUIRED_NODES},
        "/queue": {"queue_running": [], "queue_pending": []},
    }


def test_probe_accepts_loopback_read_only_schema() -> None:
    reader = FakeReader(_valid_responses())

    status = ComfyUIReadOnlyProbe("http://127.0.0.1:8188", reader).probe()

    assert status.comfyui_version == "0.27.0"
    assert status.device_count == 1
    assert status.queue_idle is True
    assert reader.paths == [
        "http://127.0.0.1:8188/system_stats",
        "http://127.0.0.1:8188/object_info",
        "http://127.0.0.1:8188/queue",
    ]


def test_probe_rejects_non_loopback_endpoint() -> None:
    with pytest.raises(ComfyUIContractError, match="loopback"):
        ComfyUIReadOnlyProbe("http://0.0.0.0:8188", FakeReader(_valid_responses()))


def test_probe_rejects_missing_required_node() -> None:
    responses = _valid_responses()
    object_info = responses["/object_info"]
    assert isinstance(object_info, dict)
    object_info.pop("SaveImage")

    with pytest.raises(ComfyUIContractError, match="SaveImage"):
        ComfyUIReadOnlyProbe("http://127.0.0.1:8188", FakeReader(responses)).probe()
