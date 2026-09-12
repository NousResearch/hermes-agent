"""The managed local server owns its picker identity.

A live session on the managed llama-server reports provider "custom"
(the resolution seam's generic label for a raw base_url). The picker
payload used to materialize that as a duplicate "Custom endpoint" group
above the Local row — same staged models listed twice, checkmark on the
wrong group. Contract: when the current session points at the managed
endpoint, the Local row is current and no custom-endpoint duplicate
exists; a user's own external endpoint keeps its row untouched."""

from __future__ import annotations

import dataclasses

import pytest


MANAGED = {"base_url": "http://127.0.0.1:18434/v1", "api_key": "k"}
STAGED = {"Qwen-A-UD-Q4_K_M", "Qwen-B-UD-Q4_K_M"}


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    import hermes_cli.inventory as inv

    monkeypatch.setattr("hermes_cli.local_runtime.bootstrap.staged_model_ids",
                        lambda: set(STAGED))
    monkeypatch.setattr("hermes_cli.local_runtime.endpoint._state_endpoint",
                        lambda: dict(MANAGED))
    context = inv.load_picker_context()
    return inv, context


def _rows(inv, context, *, explicit_only=True, **overrides):
    context = dataclasses.replace(context, **overrides)
    return inv.build_models_payload(
        context, explicit_only=explicit_only
    )["providers"]


@pytest.mark.parametrize("explicit_only", [False, True])
def test_managed_custom_session_shows_only_the_local_row(ctx, explicit_only):
    inv, context = ctx
    rows = _rows(
        inv,
        context,
        explicit_only=explicit_only,
        current_provider="custom",
        current_model="Qwen-A-UD-Q4_K_M",
        current_base_url=MANAGED["base_url"],
    )
    slugs = [r["slug"] for r in rows]
    assert "llamacpp" in slugs
    assert "custom" not in slugs, (
        "managed endpoint leaked a duplicate 'Custom endpoint' group")
    local = next(r for r in rows if r["slug"] == "llamacpp")
    assert local["is_current"] is True
    assert local["name"] == "Local"


def test_external_custom_endpoint_keeps_its_row(ctx):
    inv, context = ctx
    rows = _rows(inv, context,
                 current_provider="custom",
                 current_model="some-model",
                 current_base_url="http://my-vllm-box:8000/v1")
    slugs = [r["slug"] for r in rows]
    assert "custom" in slugs, "a real external endpoint must keep its row"
    custom = next(r for r in rows if r["slug"] == "custom")
    assert custom["is_current"] is True
    local = next(r for r in rows if r["slug"] == "llamacpp")
    assert local["is_current"] is False


def test_remote_provider_session_unaffected(ctx):
    inv, context = ctx
    rows = _rows(inv, context)
    local = next(r for r in rows if r["slug"] == "llamacpp")
    assert local["is_current"] is False
    assert "custom" not in [r["slug"] for r in rows if r.get("is_current")]
