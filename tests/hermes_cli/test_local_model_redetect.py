"""Unit tests for CLIAgentSetupMixin._maybe_refresh_local_model (#54454).

An auto-detected local endpoint (e.g. LM Studio with no ``model.default``)
should re-detect the loaded model each turn so a mid-session model swap is
reflected — instead of forever reporting the model that was loaded when the
session started.
"""

from __future__ import annotations

import time

import pytest

from hermes_cli import cli_agent_setup_mixin as mod
from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin


class _Shell(CLIAgentSetupMixin):
    """Minimal carrier for the mixin method — only the attributes it reads."""

    def __init__(self, *, model="", base_url="", is_default=True):
        self.model = model
        self.base_url = base_url
        self._model_is_default = is_default
        self._local_model_last_probe = 0.0


@pytest.fixture
def patch_detect(monkeypatch):
    """Patch _auto_detect_local_model and count calls."""
    state = {"detected": "", "calls": 0}

    def _fake(base_url):
        state["calls"] += 1
        return state["detected"]

    monkeypatch.setattr(
        "hermes_cli.runtime_provider._auto_detect_local_model", _fake
    )
    return state


def test_refreshes_when_local_and_auto_detected(patch_detect):
    shell = _Shell(model="gemma-4", base_url="http://localhost:1234/v1")
    patch_detect["detected"] = "qwen-3.6"

    changed = shell._maybe_refresh_local_model()

    assert changed is True
    assert shell.model == "qwen-3.6"
    assert patch_detect["calls"] == 1


def test_no_change_when_same_model(patch_detect):
    shell = _Shell(model="gemma-4", base_url="http://127.0.0.1:1234/v1")
    patch_detect["detected"] = "gemma-4"

    changed = shell._maybe_refresh_local_model()

    assert changed is False
    assert shell.model == "gemma-4"


def test_skips_when_model_explicitly_chosen(patch_detect):
    shell = _Shell(
        model="gpt-5.3", base_url="http://localhost:1234/v1", is_default=False
    )
    patch_detect["detected"] = "qwen-3.6"

    changed = shell._maybe_refresh_local_model()

    assert changed is False
    assert shell.model == "gpt-5.3"
    assert patch_detect["calls"] == 0  # no probe for an explicit choice


def test_skips_when_endpoint_is_not_local(patch_detect):
    shell = _Shell(model="gemma-4", base_url="https://openrouter.ai/api/v1")
    patch_detect["detected"] = "qwen-3.6"

    changed = shell._maybe_refresh_local_model()

    assert changed is False
    assert shell.model == "gemma-4"
    assert patch_detect["calls"] == 0  # no probe for a remote endpoint


def test_ttl_skips_reprobe_within_window(patch_detect):
    shell = _Shell(model="gemma-4", base_url="http://localhost:1234/v1")
    patch_detect["detected"] = "qwen-3.6"

    # A very recent probe means the next call is inside the TTL window.
    shell._local_model_last_probe = time.monotonic()

    changed = shell._maybe_refresh_local_model()

    assert changed is False
    assert shell.model == "gemma-4"
    assert patch_detect["calls"] == 0


def test_reprobe_after_ttl_expires(patch_detect, monkeypatch):
    shell = _Shell(model="gemma-4", base_url="http://localhost:1234/v1")
    patch_detect["detected"] = "qwen-3.6"

    base = time.monotonic()
    shell._local_model_last_probe = base
    # Advance the clock past the TTL.
    monkeypatch.setattr(
        mod.time, "monotonic", lambda: base + mod.LOCAL_MODEL_REDETECT_TTL_SECONDS + 1
    )

    changed = shell._maybe_refresh_local_model()

    assert changed is True
    assert shell.model == "qwen-3.6"
    assert patch_detect["calls"] == 1


def test_detection_failure_is_safe(monkeypatch):
    shell = _Shell(model="gemma-4", base_url="http://localhost:1234/v1")

    def _boom(base_url):
        raise RuntimeError("server down")

    monkeypatch.setattr(
        "hermes_cli.runtime_provider._auto_detect_local_model", _boom
    )

    changed = shell._maybe_refresh_local_model()

    assert changed is False
    assert shell.model == "gemma-4"  # unchanged, no exception


def test_empty_detection_leaves_model_untouched(patch_detect):
    shell = _Shell(model="gemma-4", base_url="http://localhost:1234/v1")
    patch_detect["detected"] = ""  # server returned 0 or >1 models

    changed = shell._maybe_refresh_local_model()

    assert changed is False
    assert shell.model == "gemma-4"
