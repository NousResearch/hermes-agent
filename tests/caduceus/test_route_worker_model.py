"""Tests for the Caduceus -> Auto Router worker-selection bridge.

``route_worker_model`` is the only place the router touches the delegate path.
These tests pin its guards (off / router-off / orchestrator-never-routed /
no-candidates) and its happy path, with the classifier stubbed so nothing hits
the network.
"""
import json
from types import SimpleNamespace

import pytest

import agent.caduceus as cad


def _agent(router_cfg, enabled=True):
    st = cad.state_from_config({"caduceus": {"enabled": enabled, "router": router_cfg}})
    return SimpleNamespace(caduceus=st, model="session-model", provider="sess")


_TWO = {
    "enabled": True,
    "candidates": [
        {"model": "cheap-flash", "provider": "openrouter", "cost": 0.3},
        {"model": "strong-model", "provider": "xai", "cost": 5.0},
    ],
}


def _stub_classifier(scores):
    def _factory(_classifier_model):
        def classify(system_prompt, user_content):
            return json.dumps({"scores": scores})
        return classify
    return _factory


# ---- guards (no classifier needed) ---------------------------------------

def test_off_returns_none():
    assert cad.route_worker_model(_agent(_TWO, enabled=False), "task", role="leaf") is None


def test_router_disabled_returns_none():
    assert cad.route_worker_model(_agent({"enabled": False, "candidates": [{"model": "x"}]}), "t", role="leaf") is None


def test_orchestrator_is_never_routed():
    assert cad.route_worker_model(_agent(_TWO), "t", role="orchestrator") is None


def test_no_candidates_returns_none():
    assert cad.route_worker_model(_agent({"enabled": True, "candidates": []}), "t", role="leaf") is None


def test_no_state_returns_none():
    assert cad.route_worker_model(SimpleNamespace(caduceus=None), "t", role="leaf") is None


# ---- happy path (classifier stubbed) -------------------------------------

def test_routes_easy_to_cheap(monkeypatch):
    monkeypatch.setattr(cad, "_build_router_classifier",
                        _stub_classifier({"cheap-flash": 0.9, "strong-model": 0.9}))
    cad_router_reset()
    got = cad.route_worker_model(_agent(_TWO), "add a docstring", role="leaf")
    assert got == {"provider": "openrouter", "model": "cheap-flash"}


def test_routes_hard_to_strong(monkeypatch):
    monkeypatch.setattr(cad, "_build_router_classifier",
                        _stub_classifier({"cheap-flash": 0.3, "strong-model": 0.95}))
    cad_router_reset()
    got = cad.route_worker_model(_agent(_TWO), "refactor auth across 9 files with a subtle race", role="leaf")
    assert got == {"provider": "xai", "model": "strong-model"}


def test_classifier_unavailable_falls_back_to_cheapest(monkeypatch):
    monkeypatch.setattr(cad, "_build_router_classifier", lambda _m: None)
    cad_router_reset()
    got = cad.route_worker_model(_agent(_TWO), "anything", role="leaf")
    assert got == {"provider": "openrouter", "model": "cheap-flash"}


def test_never_raises_on_classifier_error(monkeypatch):
    def factory(_m):
        def classify(s, u):
            raise RuntimeError("network down")
        return classify
    monkeypatch.setattr(cad, "_build_router_classifier", factory)
    cad_router_reset()
    got = cad.route_worker_model(_agent(_TWO), "anything", role="leaf")
    assert got == {"provider": "openrouter", "model": "cheap-flash"}  # safe fallback


def cad_router_reset():
    """Clear the shared per-task cache so each test scores fresh."""
    from agent import auto_router
    auto_router.reset_cache()
