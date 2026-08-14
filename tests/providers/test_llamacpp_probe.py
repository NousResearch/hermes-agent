"""Tests for the llamacpp plugin's server probe module.

probe.py detects the server kind (llama-swap vs bare llama-server), reads
/props for the session model, and parses the served chat template for the
effort levels it accepts and its thinking toggle. Canned payloads: the real
qwen38-27b-mtp-q8 template (restricted effort set, high->xhigh remap,
enable_thinking) and a minimal no-effort template.

Safety invariant under test: against llama-swap the probe never issues a
model-dispatched request for a non-resident model - llama-swap starts the
backend for those routes (/props included, see llama-swap
internal/server/server.go modelGetRoutes -> localPeerHandler).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from tests.providers.test_llamacpp_profile import _installed_plugin_dir

DATA_DIR = Path(__file__).parent / "fixtures" / "llamacpp"

pytestmark = pytest.mark.skipif(
    _installed_plugin_dir() is None,
    reason="real llamacpp user plugin not installed on this machine",
)


@pytest.fixture()
def probe():
    import sys

    plugin_dir = _installed_plugin_dir()
    spec = importlib.util.spec_from_file_location(
        "_llamacpp_probe_under_test", plugin_dir / "probe.py"
    )
    mod = importlib.util.module_from_spec(spec)
    # dataclass creation resolves the module through sys.modules
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.modules.pop(spec.name, None)


def _qwen38_props() -> dict:
    return json.loads((DATA_DIR / "qwen38-props.json").read_text())


NO_EFFORT_TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}"
    "{{ '<start_of_turn>' + message['role'] + '\n' + message['content'] "
    "+ '<end_of_turn>\n' }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<start_of_turn>model\n' }}{% endif %}"
)


# ── template parsing ────────────────────────────────────────────────────


def test_parse_qwen38_template_caps(probe):
    caps = probe.parse_template_caps(_qwen38_props()["chat_template"])
    assert caps.has_reasoning_effort is True
    assert set(caps.accepted_efforts) == {"low", "medium", "xhigh"}
    assert caps.remapped_efforts == {"high": "xhigh"}
    assert caps.default_effort == "xhigh"
    assert caps.supports_thinking_toggle is True
    # tolerated on the wire = accepted literals plus remapped inputs
    assert set(caps.tolerated_efforts) == {"low", "medium", "high", "xhigh"}


def test_parse_no_effort_template_caps(probe):
    caps = probe.parse_template_caps(NO_EFFORT_TEMPLATE)
    assert caps.has_reasoning_effort is False
    assert caps.accepted_efforts == ()
    assert caps.remapped_efforts == {}
    assert caps.default_effort is None
    assert caps.supports_thinking_toggle is False
    assert caps.tolerated_efforts == ()


def test_parse_empty_template(probe):
    caps = probe.parse_template_caps("")
    assert caps.has_reasoning_effort is False
    assert caps.tolerated_efforts == ()


# ── server detection ────────────────────────────────────────────────────


def test_detect_llama_swap(probe, monkeypatch):
    def fake_get(url, timeout):
        assert url.endswith("/running")
        return 200, {"running": [{"model": "qwen38-27b-mtp-q8", "state": "ready"}]}

    monkeypatch.setattr(probe, "_http_get_json", fake_get)
    server = probe.detect_server("http://rig:8080/v1")
    assert server.kind == "llama-swap"
    assert server.running == ("qwen38-27b-mtp-q8",)


def test_detect_bare_llama_server(probe, monkeypatch):
    monkeypatch.setattr(probe, "_http_get_json", lambda url, timeout: (404, None))
    server = probe.detect_server("http://rig:10035/v1")
    assert server.kind == "llama-server"
    assert server.running == ()


def test_detect_unreachable(probe, monkeypatch):
    def fake_get(url, timeout):
        raise OSError("connection refused")

    monkeypatch.setattr(probe, "_http_get_json", fake_get)
    server = probe.detect_server("http://down:9999/v1")
    assert server.kind == "unknown"


# ── props fetch safety ──────────────────────────────────────────────────


def test_fetch_props_bare_server(probe, monkeypatch):
    calls = []

    def fake_get(url, timeout):
        calls.append(url)
        return 200, _qwen38_props()

    monkeypatch.setattr(probe, "_http_get_json", fake_get)
    server = probe.ServerInfo(kind="llama-server", running=())
    props = probe.fetch_props("http://rig:10035/v1", "whatever", server=server)
    assert props is not None
    assert calls == ["http://rig:10035/props"]


def test_fetch_props_swap_resident_model(probe, monkeypatch):
    calls = []

    def fake_get(url, timeout):
        calls.append(url)
        return 200, _qwen38_props()

    monkeypatch.setattr(probe, "_http_get_json", fake_get)
    server = probe.ServerInfo(kind="llama-swap", running=("qwen38-27b-mtp-q8",))
    props = probe.fetch_props(
        "http://rig:8080/v1", "qwen38-27b-mtp-q8", server=server
    )
    assert props is not None
    assert calls == ["http://rig:8080/props?model=qwen38-27b-mtp-q8"]


def test_fetch_props_swap_never_dispatches_non_resident(probe, monkeypatch):
    """The core safety rule: a non-resident id must produce NO model-dispatched
    request, because llama-swap would start that model."""
    calls = []

    def fake_get(url, timeout):
        calls.append(url)
        return 200, {}

    monkeypatch.setattr(probe, "_http_get_json", fake_get)
    server = probe.ServerInfo(kind="llama-swap", running=("qwen38-27b-mtp-q8",))
    props = probe.fetch_props("http://rig:8080/v1", "gemma-4-e2b-q4", server=server)
    assert props is None
    assert calls == [], "no HTTP request may be issued for a non-resident model"


# ── end-to-end probe with logging ───────────────────────────────────────


def test_probe_model_logs_kind_and_efforts(probe, monkeypatch, caplog):
    import logging

    def fake_get(url, timeout):
        if url.endswith("/running"):
            return 200, {"running": [{"model": "qwen38-27b-mtp-q8"}]}
        return 200, _qwen38_props()

    monkeypatch.setattr(probe, "_http_get_json", fake_get)
    with caplog.at_level(logging.INFO, logger="providers.llamacpp"):
        result = probe.probe_model("http://rig:8080/v1", "qwen38-27b-mtp-q8")

    assert result.server.kind == "llama-swap"
    assert set(result.caps.tolerated_efforts) == {"low", "medium", "high", "xhigh"}
    joined = " ".join(rec.getMessage() for rec in caplog.records)
    assert "llama-swap" in joined
    assert "xhigh" in joined


def test_probe_model_non_resident_reports_unknown_caps(probe, monkeypatch):
    def fake_get(url, timeout):
        if url.endswith("/running"):
            return 200, {"running": []}
        raise AssertionError(f"unexpected request: {url}")

    monkeypatch.setattr(probe, "_http_get_json", fake_get)
    result = probe.probe_model("http://rig:8080/v1", "gemma-4-e2b-q4")
    assert result.server.kind == "llama-swap"
    assert result.props is None
    assert result.caps is None
