"""Launch and growth decisions must price against CAPACITY, not live-free
VRAM. Both execute through a server bounce — the outgoing instance's memory
is freed before the new one loads — so a probe that reads the predecessor's
(or the grown model's own) residency as 'gone' vetoes configurations that
genuinely fit. Symptom when this regresses: a model the pane promised
'144K on GPU' launches with its weights pinned to CPU and single-digit
tokens/s while the card sits 60% empty."""

from __future__ import annotations

import ast
import inspect


def _planning_probe_calls(source: str) -> list[bool]:
    """Every probe_budget(...) call's planning= value in the source."""
    tree = ast.parse(source)
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and getattr(node.func, "id", getattr(node.func, "attr", ""))
                == "probe_budget"):
            planning = any(
                kw.arg == "planning"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is True
                for kw in node.keywords)
            out.append(planning)
    return out


def test_bootstrap_presets_price_against_capacity():
    import hermes_cli.local_runtime.bootstrap as bootstrap

    calls = _planning_probe_calls(inspect.getsource(bootstrap))
    assert calls, "bootstrap no longer probes a budget? update this test"
    assert all(calls), (
        "bootstrap prices launch decisions against live-free VRAM; a "
        "restart/refresh probes while the outgoing server still holds the "
        "card, pinning fitting models to CPU")


def test_growth_refit_prices_against_capacity():
    import hermes_cli.local_runtime.growth as growth

    calls = _planning_probe_calls(inspect.getsource(growth))
    assert calls, "growth no longer probes a budget? update this test"
    assert all(calls), (
        "growth re-fits against live-free VRAM; the grown model's own "
        "residency reads as unavailable and vetoes rungs that fit the "
        "post-bounce card")
