"""Curated facts survive automatic recall's shared context budget."""

from types import SimpleNamespace

import pytest

from plugins.memory.honcho import HonchoMemoryProvider
from plugins.memory.honcho.client import HonchoClientConfig


@pytest.mark.parametrize("oversized_section", ["summary", "representation"])
def test_prefetch_keeps_curated_cards_before_generated_context(oversized_section):
    provider = HonchoMemoryProvider()
    provider._config = HonchoClientConfig(context_tokens=100)
    user_card, ai_card = "Human prefers direct answers.", "Assistant identity: deliberate collaborator."
    context = {"card": user_card, "ai_card": ai_card, oversized_section: "generated " * 200}
    provider._manager = SimpleNamespace(pop_context_result=lambda key: context)
    provider._session_key = "synthetic-context"
    provider._session_initialized = True
    provider._base_context_cache = ""
    provider._turn_count = provider._last_dialectic_turn = 2

    injected = provider.prefetch("What is relevant to our current project?")

    assert user_card in injected and ai_card in injected
    assert injected.index(user_card) < injected.index("generated")
    assert injected.index(ai_card) < injected.index("generated")
    assert len(injected) <= provider._config.context_tokens * 4


@pytest.mark.parametrize("tokens", [1, 10, 1000])
@pytest.mark.parametrize("text", ["unbroken" * 1000, "word " * 2000], ids=["unbroken", "words"])
def test_truncation_reserves_space_for_ellipsis(tokens, text):
    provider = HonchoMemoryProvider()
    provider._config = HonchoClientConfig(context_tokens=tokens)

    result = provider._truncate_to_budget(text)

    assert len(result) <= tokens * 4
    assert result.endswith(" …")
