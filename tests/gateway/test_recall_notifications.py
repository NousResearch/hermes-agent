"""Gateway chat filter for the memory-provider recall indicator.

``MemoryManager.describe_recall()`` emits a lifecycle status
(``🧠 Holographic — recalled 1 memory``) so the user can see that prefetch
injected facts. That line is not a tool step and not a background-review
notice, so ``display.tool_progress`` and ``display.memory_notifications`` do
not gate it. On Slack it becomes a permanent thread message.

``display.recall_notifications`` (default ``on``) and the per-platform override
``display.platforms.<platform>.recall_notifications`` are the knob. The
platform override must win, including over a global ``on``.
"""

from typing import Optional

import pytest

import gateway.run as gateway_run
from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider, RecallStatus
from gateway.run import _prepare_gateway_status_message


class _FakeProvider(MemoryProvider):
    def __init__(self, name: str, status: Optional[RecallStatus]):
        self._name = name
        self._status = status

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str = "", **kwargs) -> None:
        pass

    def get_tool_schemas(self):
        return []

    def handle_tool_call(self, tool_name, args, **kwargs) -> str:
        return ""

    def recall_status(self) -> Optional[RecallStatus]:
        return self._status


def _line(label: str, count: int) -> str:
    mgr = MemoryManager()
    mgr.add_provider(_FakeProvider(label.lower(), RecallStatus(label, count)))
    return mgr.describe_recall()


HOLOGRAPHIC = _line("Holographic", 1)
PLURAL = _line("Hindsight", 4)
GENERIC = _line("Honcho", 0)


def _config(monkeypatch, config: dict) -> None:
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: config)


def test_default_keeps_describe_recall_lines(monkeypatch):
    """Empty config is today's behavior: the line still posts."""
    _config(monkeypatch, {})
    assert _prepare_gateway_status_message("slack", "lifecycle", HOLOGRAPHIC) == HOLOGRAPHIC


def test_global_off_drops_every_describe_recall_shape(monkeypatch):
    _config(monkeypatch, {"display": {"recall_notifications": "off"}})
    for line in (HOLOGRAPHIC, PLURAL, GENERIC):
        assert "recalled" in line
        assert _prepare_gateway_status_message("slack", "lifecycle", line) is None
        assert _prepare_gateway_status_message("discord", "lifecycle", line) is None


def test_slack_override_off_does_not_mute_other_platforms(monkeypatch):
    """The platform leaf wins over a global on. This is the shared-channel case."""
    _config(monkeypatch, {
        "display": {
            "recall_notifications": "on",
            "platforms": {"slack": {"recall_notifications": False}},  # YAML 1.1 bare off
        },
    })
    assert _prepare_gateway_status_message("slack", "lifecycle", HOLOGRAPHIC) is None
    assert _prepare_gateway_status_message("discord", "lifecycle", HOLOGRAPHIC) == HOLOGRAPHIC
    assert _prepare_gateway_status_message("telegram", "lifecycle", PLURAL) == PLURAL


def test_off_does_not_swallow_other_status_or_review_lines(monkeypatch):
    _config(monkeypatch, {"display": {"recall_notifications": "off"}})
    for message in (
        "still on it",
        "💾 Memory updated",
        "💾 Skill 'hermes-gateway-display' patched",
        "I recalled the incident from last week",
    ):
        assert _prepare_gateway_status_message("slack", "lifecycle", message) == message


@pytest.mark.parametrize("platform", ["local", "api_server", "webhook", "msgraph_webhook"])
def test_raw_surfaces_keep_the_line_even_when_off(monkeypatch, platform):
    _config(monkeypatch, {"display": {"recall_notifications": "off"}})
    assert _prepare_gateway_status_message(platform, "lifecycle", HOLOGRAPHIC) == HOLOGRAPHIC


def test_config_read_error_fails_open(monkeypatch):
    def _boom():
        raise OSError("config unreadable")

    monkeypatch.setattr(gateway_run, "_load_gateway_config", _boom)
    assert _prepare_gateway_status_message("slack", "lifecycle", HOLOGRAPHIC) == HOLOGRAPHIC
