from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import gateway.agent_runtime as agent_runtime
from gateway.agent_runtime import (
    GatewayAgentRuntimeSpec,
    prepare_gateway_sync_turn_runtime,
    reload_gateway_dotenv,
)


def test_reload_gateway_dotenv_retries_with_latin1_on_utf8_decode_error():
    calls: list[tuple[Path, bool, str]] = []

    def fake_load_dotenv(path: Path, *, override: bool, encoding: str) -> None:
        calls.append((path, override, encoding))
        if encoding == "utf-8":
            raise UnicodeDecodeError("utf-8", b"x", 0, 1, "bad bytes")

    reload_gateway_dotenv(
        Path("/tmp/test.env"),
        load_dotenv_fn=fake_load_dotenv,
    )

    assert calls == [
        (Path("/tmp/test.env"), True, "utf-8"),
        (Path("/tmp/test.env"), True, "latin-1"),
    ]


def test_prepare_gateway_sync_turn_runtime_builds_runtime_from_injected_resolvers(
    monkeypatch,
):
    source = SimpleNamespace(platform=SimpleNamespace(value="qq"))
    call_order: list[str] = []
    expected_spec = GatewayAgentRuntimeSpec(
        user_config={"model": {"default": "gpt-test"}},
        source=source,
        platform_key="qq",
        model="gpt-test",
        runtime_kwargs={"api_key": "secret", "provider": "custom"},
        turn_route={"model": "gpt-test", "runtime": {"api_key": "secret"}},
        provider_routing={"order": ["custom"]},
        fallback_model=[{"model": "backup"}],
        reasoning_config={"effort": "high"},
        enabled_toolsets=["hermes-qq"],
        combined_ephemeral="ctx",
        loaded_skills=[],
        missing_skills=[],
        max_iterations=123,
    )

    def fake_load_dotenv(path: Path, *, override: bool, encoding: str) -> None:
        assert path == Path("/tmp/runtime.env")
        assert override is True
        assert encoding == "utf-8"
        call_order.append("dotenv")

    def fake_resolve_runtime() -> dict[str, str]:
        call_order.append("resolve_runtime")
        return {"api_key": "secret", "provider": "custom"}

    def fake_load_reasoning() -> dict[str, str]:
        call_order.append("load_reasoning")
        return {"effort": "high"}

    def fake_build_gateway_agent_runtime(**kwargs):
        call_order.append("build_runtime_spec")
        assert kwargs["source"] is source
        assert kwargs["user_message"] == "hello"
        assert kwargs["context_prompt"] == "ctx"
        assert kwargs["gateway_ephemeral_system_prompt"] == "ephemeral"
        assert kwargs["provider_routing"] == {"order": ["custom"]}
        assert kwargs["fallback_model"] == [{"model": "backup"}]
        assert kwargs["smart_model_routing"] == {"mode": "smart"}
        assert kwargs["reasoning_config"] == {"effort": "high"}
        assert kwargs["preloaded_skills"] is None
        assert kwargs["skill_task_id"] is None
        assert kwargs["user_config"] == {"model": {"default": "gpt-test"}}
        assert kwargs["model"] == "gpt-test"
        assert kwargs["runtime_kwargs"] == {
            "api_key": "secret",
            "provider": "custom",
        }
        assert kwargs["enabled_toolsets"] == ["hermes-qq"]
        return expected_spec

    monkeypatch.setenv("HERMES_MAX_ITERATIONS", "123")
    monkeypatch.setattr(
        agent_runtime,
        "build_gateway_agent_runtime",
        fake_build_gateway_agent_runtime,
    )

    prepared = prepare_gateway_sync_turn_runtime(
        env_path=Path("/tmp/runtime.env"),
        load_dotenv_fn=fake_load_dotenv,
        resolve_runtime_agent_kwargs_fn=fake_resolve_runtime,
        load_reasoning_config_fn=fake_load_reasoning,
        source=source,
        user_message="hello",
        context_prompt="ctx",
        gateway_ephemeral_system_prompt="ephemeral",
        provider_routing={"order": ["custom"]},
        fallback_model=[{"model": "backup"}],
        smart_model_routing={"mode": "smart"},
        user_config={"model": {"default": "gpt-test"}},
        model="gpt-test",
        enabled_toolsets=["hermes-qq"],
    )

    assert prepared.runtime_spec is expected_spec
    assert prepared.reasoning_config == {"effort": "high"}
    assert prepared.max_iterations == 123
    assert call_order == [
        "dotenv",
        "resolve_runtime",
        "load_reasoning",
        "build_runtime_spec",
    ]


def test_prepare_gateway_sync_turn_runtime_passes_preloaded_skills(monkeypatch):
    source = SimpleNamespace(platform=SimpleNamespace(value="qq"))
    captured = {}

    monkeypatch.setenv("HERMES_MAX_ITERATIONS", "90")
    monkeypatch.setattr(
        agent_runtime,
        "build_gateway_agent_runtime",
        lambda **kwargs: captured.update(kwargs) or GatewayAgentRuntimeSpec(
            user_config={},
            source=source,
            platform_key="qq",
            model="gpt-test",
            runtime_kwargs={"api_key": "secret"},
            turn_route={"model": "gpt-test", "runtime": {"api_key": "secret"}},
            provider_routing={},
            fallback_model=None,
            reasoning_config={"effort": "medium"},
            enabled_toolsets=["core"],
            combined_ephemeral="ctx",
            loaded_skills=["ops-skill"],
            missing_skills=[],
            max_iterations=90,
        ),
    )

    prepared = prepare_gateway_sync_turn_runtime(
        env_path=Path("/tmp/runtime.env"),
        load_dotenv_fn=lambda *args, **kwargs: None,
        resolve_runtime_agent_kwargs_fn=lambda: {"api_key": "secret"},
        load_reasoning_config_fn=lambda: {"effort": "medium"},
        source=source,
        user_message="hello",
        preloaded_skills=["ops-skill"],
        skill_task_id="task-1",
    )

    assert captured["preloaded_skills"] == ["ops-skill"]
    assert captured["skill_task_id"] == "task-1"
    assert prepared.runtime_spec.loaded_skills == ["ops-skill"]
