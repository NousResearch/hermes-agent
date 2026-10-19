"""Regression tests for external memory-prefetch spilling and wire transport."""

from __future__ import annotations

from pathlib import Path

from agent.memory_manager import MemoryManager
from agent.memory_provider import (
    MAX_MEMORY_OBSERVATIONS,
    MemoryObservation,
    MemoryPrefetchResult,
    MemoryProvider,
)


class _StructuredProvider(MemoryProvider):
    def __init__(self, name: str, result: MemoryPrefetchResult):
        self._name = name
        self.result = result

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        pass

    def get_tool_schemas(self) -> list[dict]:
        return []

    def prefetch(self, query: str, *, session_id: str = "") -> MemoryPrefetchResult:
        return self.result


def _observation(payload: dict) -> MemoryObservation:
    return MemoryObservation(
        source_kind="fixture_context",
        schema="fixture.context",
        version=1,
        payload=payload,
    )


def _disable_hook(monkeypatch) -> None:
    import agent.plugin_stream_hooks as dispatcher

    monkeypatch.setattr(
        dispatcher,
        "_enqueue_plugin_observer_hook_with_payload_factory",
        lambda _name, _payload_factory: False,
    )


def test_external_spill_preserves_wire_truncation_for_later_providers(
    monkeypatch, tmp_path, caplog
):
    from hermes_cli.config import atomic_config_write
    from hermes_cli.plugin_host_wire import decode, encode

    _disable_hook(monkeypatch)
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    home.mkdir(parents=True)
    atomic_config_write(home / "config.yaml", {"memory": {"prefetch_spill_enabled": True}})
    context = "oversized hosted recall\n" + "x" * 10_100
    raw = MemoryPrefetchResult(
        context=context,
        observations=tuple(
            _observation({"text": "x" * 3910}) for _ in range(MAX_MEMORY_OBSERVATIONS)
        ),
    )
    hosted = decode(encode(raw))
    assert hosted._wire_truncated is True
    hosted_prefix_size = len(hosted.observations)

    manager = MemoryManager()
    manager.add_provider(_StructuredProvider(name="external", result=hosted))
    manager.add_provider(
        _StructuredProvider(
            name="builtin",
            result=MemoryPrefetchResult(
                context="later provider context",
                observations=(_observation({"later": True}),),
            ),
        )
    )

    with caplog.at_level("WARNING", logger="agent.memory_manager"):
        result = manager.prefetch_all_result("question", session_id="spill-wire")

    assert result.context.endswith("later provider context")
    assert "full content saved to " in result.context
    spill_path = Path(
        result.context.split("full content saved to ", 1)[1].split("]", 1)[0]
    )
    assert spill_path.is_relative_to(home)
    assert spill_path.read_text(encoding="utf-8") == context + "\n"
    assert len(result.observations) == hosted_prefix_size
    assert all(item.provider == "external" for item in result.observations)
    assert "plugin-host wire batch bound" in caplog.text
