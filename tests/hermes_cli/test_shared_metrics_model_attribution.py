"""Provider/model attribution invariants for the per-model shared metrics: which route a row names,
that user-named models never leave the machine, which profile records it, and how often."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from hermes_cli import lifecycle
from hermes_cli.observability import relay_shared_metrics
from tests.hermes_cli.test_relay_shared_metrics_runtime import (  # noqa: F401 - fixture
    _stored_values,
    direct_runtime,
)

SECRET = "acme-internal-secret-v2"
PUBLIC = "anthropic/claude-sonnet-4"
# The stored install snapshot reflects the fixture's config; the snapshot built from the probed
# config is checked separately.
_PROVIDER_FIELDS = ("provider", "from_provider")


def _flush() -> None:
    for runtime in list(relay_shared_metrics._RUNTIMES.values()):
        runtime.relay.subscribers.flush()


def _all_rows(home) -> list[tuple[str, dict]]:
    from hermes_cli.observability.shared_metrics import SharedMetricsStore

    root = home / "telemetry" / "shared_metrics"
    if not (root / "metrics.sqlite3").exists():
        return []
    store = SharedMetricsStore(root / "metrics.sqlite3", root / "outbox")
    return [(c["metric_name"], c["dimensions"]) for c in store.counter_snapshot()]


def _turn(session_id, task_id, provider, model, *, prompt_tokens=1_000, context_length=200_000, **start):
    base = {"session_id": session_id, "task_id": task_id, "api_request_id": f"{task_id}-r",
            "provider": provider, "model": model}
    lifecycle.invoke_hook("pre_llm_call", **base, platform="cli", **start)
    lifecycle.invoke_hook("pre_api_request", **base)
    lifecycle.invoke_hook("post_api_request", **base, usage={"prompt_tokens": prompt_tokens, "output_tokens": 5},
                          context_length=context_length)
    relay_shared_metrics.finish_task_run(session_id=session_id, task_id=task_id, platform="cli",
                                         result={"completed": True})


def _emit_every_model_metric(provider, model) -> dict:
    """Drive every emitter that carries a provider/model dimension through its real entry point."""
    from hermes_cli.observability import shared_metrics_events as events
    from hermes_cli.observability.shared_metrics_model import record_model_friction, record_tool_call_quality
    from hermes_cli.observability.shared_metrics_snapshot import collect_install_snapshot

    _turn("s1", "t1", provider, model)  # model_route, model_tokens (primary), context_peak
    lifecycle.invoke_hook("post_auxiliary_call", session_id="s1", provider=provider, model=model,
                          aux_task="compression", usage={"input_tokens": 10})
    events.record_fallback(from_provider=provider, to_provider="openrouter", reason="rate_limit")
    events.record_model_switch(from_provider=provider, to_provider="openrouter", surface="cli", from_model=model)
    events.record_setup_completed(surface="cli", provider=provider)
    record_model_friction("retry", session_id="not-seen", provider=provider, model=model)
    agent = SimpleNamespace(provider=provider, model=model, valid_tool_names={"todo"}, tools=[])
    record_tool_call_quality(agent, [SimpleNamespace(function=SimpleNamespace(name="todo", arguments="{}"))], set())
    lifecycle.finalize_session(session_id="s1")
    _flush()
    return collect_install_snapshot({"model": {"provider": provider, "default": model}})


_MODEL_METRICS = {
    "hermes.model_route.count", "hermes.model_tokens.sum", "hermes.context_peak.count",
    "hermes.fallback.count", "hermes.model_switch.count", "hermes.setup.completed",
    "hermes.model_friction.count", "hermes.model_tool_quality.count",
}


@pytest.mark.parametrize("provider", [
    "ollama", "local", "vllm", "llamacpp", "llama.cpp", "custom:acme-lab", None, "", "acme-gpu-box",
])
def test_user_named_models_never_leave_in_any_model_metric(direct_runtime, tmp_path, provider):
    """Local-server aliases of ``custom``, a missing provider and unknown ids all read custom/unknown
    with model ``custom`` in every provider/model-bearing metric; the raw id is nowhere."""
    snapshot = _emit_every_model_metric(provider, SECRET)
    rows = _all_rows(tmp_path / "hermes-home")

    assert SECRET not in json.dumps(rows) + json.dumps(snapshot)
    assert _MODEL_METRICS <= {name for name, _ in rows}
    assert snapshot["main_provider"] in {"custom", "unknown", "none"}
    for name, dims in rows:
        for key in _PROVIDER_FIELDS:
            if key in dims:
                assert dims[key] in {"custom", "unknown", "none"}, (name, dims)
        if "model" in dims:
            assert dims["model"] == "custom", (name, dims)


def test_shipped_provider_and_public_model_are_reported_as_is(direct_runtime, tmp_path):
    snapshot = _emit_every_model_metric("openrouter", PUBLIC)
    rows = _all_rows(tmp_path / "hermes-home")

    assert snapshot["main_provider"] == "openrouter"
    assert _MODEL_METRICS <= {name for name, _ in rows}
    for name, dims in rows:
        for key in _PROVIDER_FIELDS:
            if key in dims:
                assert dims[key] == "openrouter", (name, dims)
        if "model" in dims:
            assert dims["model"] == PUBLIC, (name, dims)


@pytest.mark.parametrize(("configured", "expected"), [
    (("custom", "acme-private-llama", "http://localhost:8080/v1"), ("custom", "custom")),
    (("openrouter", PUBLIC, ""), ("openrouter", PUBLIC)),
])
def test_tui_switch_before_first_prompt_blames_the_configured_route(
    direct_runtime, tmp_path, monkeypatch, configured, expected,
):
    """With no agent yet and ``--provider``, switch_away names the model the session was launched on
    with ITS provider, never the target provider."""
    provider, model, base_url = configured
    home = tmp_path / "hermes-home"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        f"model:\n  provider: {provider}\n  default: {model}\n" + (f"  base_url: {base_url}\n" if base_url else ""))
    import tui_gateway.server as server

    monkeypatch.setattr(server, "_hermes_home", home)  # captured at first import
    result = SimpleNamespace(
        success=True, new_model="gpt-5", target_provider="openai", base_url="https://api.openai.com/v1",
        api_key="k", api_mode="chat_completions", warning_message="", error_message="")
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **kw: result)
    server._apply_model_switch("", {"agent": None}, "gpt-5 --provider openai", confirm_expensive_model=True)
    _flush()

    assert _stored_values(tmp_path, "hermes.model_friction.count") == [
        ({"model": expected[1], "provider": expected[0], "signal": "switch_away"}, 1)]
    assert _stored_values(tmp_path, "hermes.model_switch.count") == [
        ({"execution_surface": "tui", "from_provider": expected[0], "to_provider": "openai"}, 1)]


def _gateway_runner(multiplex_home=None):
    from gateway.slash_commands_model import GatewayModelCommandsMixin

    class Runner(GatewayModelCommandsMixin):
        def __init__(self):
            self.config = SimpleNamespace(multiplex_profiles=multiplex_home is not None)

        def _resolve_profile_home_for_source(self, source):
            return multiplex_home

        def _switch_cached_agent_model(self, result, ctx, picker):
            return None

        async def _record_model_switch(self, *a, **k):
            return None

        async def _model_switch_confirmation(self, *a, **k):
            return "ok"

    return Runner()


def _gateway_switch(runner, config_path):
    from gateway.slash_commands_model import _ModelSwitchContext

    ctx = _ModelSwitchContext(session_key="k", source=None, config_path=config_path, persist_global=False)
    ctx.read_config()
    result = SimpleNamespace(target_provider="anthropic", new_model="claude-opus-4")
    asyncio.run(runner._commit_model_switch_locked(result, ctx, source=None, picker=True))
    _flush()


@pytest.mark.parametrize(("model_block", "expected"), [
    ("  provider: vllm\n  default: acme-private-llama\n", ("custom", "custom")),
    ("  default: acme-private-llama\n  base_url: http://10.0.0.5:8000/v1\n", ("unknown", "custom")),
    (f"  provider: openrouter\n  default: {PUBLIC}\n", ("openrouter", PUBLIC)),
])
def test_gateway_switch_reports_the_configured_route_not_the_switch_default(
    direct_runtime, tmp_path, model_block, expected,
):
    """Gateway /model reads config.yaml: an unset provider must not borrow switch_model's
    ``openrouter`` default, and a local alias must not ship the configured model id."""
    cfg = tmp_path / "gw-config.yaml"
    cfg.write_text("model:\n" + model_block)
    _gateway_switch(_gateway_runner(), cfg)

    assert _stored_values(tmp_path, "hermes.model_friction.count") == [
        ({"model": expected[1], "provider": expected[0], "signal": "switch_away"}, 1)]
    assert _stored_values(tmp_path, "hermes.model_switch.count") == [
        ({"execution_surface": "gateway", "from_provider": expected[0], "to_provider": "anthropic"}, 1)]


def test_multiplexed_gateway_switch_records_in_the_owning_profile(direct_runtime, tmp_path):
    home_b = tmp_path / "profiles" / "b"
    home_b.mkdir(parents=True)
    cfg = home_b / "config.yaml"
    cfg.write_text(f"model:\n  provider: openrouter\n  default: {PUBLIC}\n")
    _gateway_switch(_gateway_runner(multiplex_home=home_b), cfg)

    assert not _all_rows(tmp_path / "hermes-home")
    assert sorted(name for name, _ in _all_rows(home_b)) == ["hermes.model_friction.count", "hermes.model_switch.count"]
