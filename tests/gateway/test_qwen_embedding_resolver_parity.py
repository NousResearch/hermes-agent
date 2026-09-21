"""Regression contract for the kanban shared-memory writer's embedding space."""
from __future__ import annotations

import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    not all(importlib.util.find_spec(name) is not None for name in ("severian", "sqlalchemy")),
    reason="shared injection requires the Severian integration stack",
)


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        ("qwen", "QwenEmbedding"),
        ("granite", "GraniteR2Embedding"),
        ("hash", "HashEmbedding"),
        (None, "HashEmbedding"),
    ],
)
def test_kanban_shared_injection_uses_central_embedding_resolver(
    monkeypatch: pytest.MonkeyPatch,
    selector: str | None,
    expected: str,
) -> None:
    """The event-driven writer must never turn a Qwen selector into hash."""
    import gateway.kanban_watchers as watchers

    if selector is None:
        monkeypatch.delenv("SEVERIAN_EMBEDDING", raising=False)
    else:
        monkeypatch.setenv("SEVERIAN_EMBEDDING", selector)

    embedding = watchers._resolve_shared_injection_embedding()

    assert embedding.__class__.__name__ == expected
    assert not getattr(embedding, "loaded", False)


def test_kanban_shared_injection_delegates_to_central_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The writer delegates selector parsing; it owns no fallback branch."""
    import gateway.kanban_watchers as watchers
    from severian.infrastructure import embedding_resolver

    sentinel = object()
    monkeypatch.setattr(embedding_resolver, "embedding_from_env", lambda: sentinel)

    assert watchers._resolve_shared_injection_embedding() is sentinel
