"""Regression contract for PR #99053 finding G6: the `codex_app_server` runtime cannot
honour an override model, so the override is dropped explicitly with a warning.

That runtime hands the whole turn to a long-lived subprocess that accepts only its own
models (an arbitrary override can kill it), and its fallback can fall through into the
normal loop, which would let one turn cross the override scope boundary twice. Applying
the override there is worse than refusing it; ignoring it silently is worse than saying
so. (Recommendation of the agy/Gemini full-scale red-team; G1's runtime validation was
reverted because `validate_requested_model` probes providers over the network.)
"""

from __future__ import annotations

from agent.runtime_override import warn_and_drop_unsupported_override
from tests.agent.test_runtime_override import (
    _OVERRIDE_MODEL,
    _SESSION_MODEL,
    _build_model_owned_agent,
    _patch_model_owned_resolution,
)


def test_codex_app_server_drops_the_override_with_a_warning(monkeypatch, caplog):
    _patch_model_owned_resolution(monkeypatch)
    agent = _build_model_owned_agent(model=_SESSION_MODEL)
    agent._runtime_override = {"model": _OVERRIDE_MODEL}

    with caplog.at_level("WARNING"):
        assert warn_and_drop_unsupported_override(agent, "not supported here") is True

    assert not agent._runtime_override, "an unusable override must not stay armed"
    assert any("runtime_override" in r.getMessage() for r in caplog.records), (
        "dropping an override silently would look like it was applied"
    )


def test_dropping_without_an_override_is_a_no_op(monkeypatch):
    _patch_model_owned_resolution(monkeypatch)
    agent = _build_model_owned_agent(model=_SESSION_MODEL)
    agent._runtime_override = {}
    assert warn_and_drop_unsupported_override(agent, "not supported here") is False
