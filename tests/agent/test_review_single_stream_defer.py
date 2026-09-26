"""Deferral of background reviews bound for operator-marked single-stream local endpoints.

Companion contracts to tests/agent/test_review_idle_queue.py (managed-local deferral).
New behavior: a review whose fork would decode on a loopback endpoint with
``providers.<id>.max_in_flight_requests == 1`` is deferred to the idle queue —
otherwise it contends with the next live turn for the only slot, loses via hard
interrupt, and burns a full re-prefill per kill (2026-09-22 Tabby :8290 incident:
12 kills of healthy 60-72 tok/s streams).

- loopback + key == 1        -> defer (True)
- loopback, key absent/other -> immediate (False; pre-existing default)
- hosted endpoint, even +1   -> immediate (False; server-side queueing)
- config unreadable          -> immediate (False; fail open)
- defer: never               -> immediate (False; explicit opt-out respected)
"""

from __future__ import annotations

from unittest.mock import patch

from agent.review_idle_queue import review_targets_local_single_stream
from run_agent import _review_should_defer


class _FakeAgent:
    def __init__(self, provider="qwen27b-exl3", model="Qwen3.8-27B-EXL3-3.5bpw",
                 base_url="http://127.0.0.1:8290/v1"):
        self.provider = provider
        self.model = model
        self._base_url = base_url
        self._credential_pool = None
        self.request_overrides = {}
        self.max_tokens = None
        self.acp_command = None
        self.acp_args = []

    def _current_main_runtime(self):
        return {"api_key": None, "base_url": self._base_url, "api_mode": None}


def _cfg(providers):
    return {"providers": providers}


def test_loopback_single_stream_defers():
    agent = _FakeAgent()
    config = _cfg({"qwen27b-exl3": {"base_url": "http://127.0.0.1:8290/v1",
                                    "max_in_flight_requests": 1}})
    with patch("hermes_cli.config.load_config_readonly", return_value=config):
        assert review_targets_local_single_stream(agent, {}) is True
        assert _review_should_defer(agent, {}) is True


def test_loopback_without_key_spawns_immediately():
    agent = _FakeAgent()
    config = _cfg({"qwen27b-exl3": {"base_url": "http://127.0.0.1:8290/v1"}})
    with patch("hermes_cli.config.load_config_readonly", return_value=config):
        assert review_targets_local_single_stream(agent, {}) is False
        assert _review_should_defer(agent, {}) is False


def test_loopback_matched_by_netloc_when_id_differs():
    agent = _FakeAgent(provider="other-id")
    config = _cfg({"tabby-local": {"base_url": "http://127.0.0.1:8290/v1",
                                   "max_in_flight_requests": 1}})
    with patch("hermes_cli.config.load_config_readonly", return_value=config):
        assert review_targets_local_single_stream(agent, {}) is True


def test_hosted_endpoint_never_defers():
    agent = _FakeAgent(provider="hosted", base_url="https://api.example.com/v1")
    config = _cfg({"hosted": {"base_url": "https://api.example.com/v1",
                              "max_in_flight_requests": 1}})
    with patch("hermes_cli.config.load_config_readonly", return_value=config):
        assert review_targets_local_single_stream(agent, {}) is False


def test_unreadable_config_fails_open():
    agent = _FakeAgent()
    with patch("hermes_cli.config.load_config_readonly", side_effect=OSError("nope")):
        assert review_targets_local_single_stream(agent, {}) is False


def test_defer_never_respected():
    agent = _FakeAgent()
    config = _cfg({"qwen27b-exl3": {"base_url": "http://127.0.0.1:8290/v1",
                                    "max_in_flight_requests": 1}})
    with patch("hermes_cli.config.load_config_readonly", return_value=config):
        assert _review_should_defer(agent, {"defer": "never"}) is False
