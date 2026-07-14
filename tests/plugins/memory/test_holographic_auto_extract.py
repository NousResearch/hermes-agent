from typing import cast

from plugins.memory.holographic import HolographicMemoryProvider
from plugins.memory.holographic.store import MemoryStore


class _FakeStore:
    def __init__(self):
        self.facts = []

    def add_fact(self, content, category="general", tags=None):
        self.facts.append({"content": content, "category": category, "tags": tags})


def _provider_with_fake_store():
    provider = HolographicMemoryProvider(config={})
    provider._store = cast(MemoryStore, _FakeStore())
    return provider


def test_auto_extract_saves_preference_fact_not_raw_message():
    provider = _provider_with_fake_store()
    message = "Hey, for future work I prefer concise PR comments with no hype."

    provider._auto_extract_facts([{"role": "user", "content": message}])

    assert provider._store.facts == [
        {
            "content": "User prefers concise PR comments with no hype.",
            "category": "user_pref",
            "tags": None,
        }
    ]
    assert provider._store.facts[0]["content"] != message


def test_auto_extract_retains_default_preference_subject():
    provider = _provider_with_fake_store()
    message = "By the way, my default shell is zsh."

    provider._auto_extract_facts([{"role": "user", "content": message}])

    assert provider._store.facts == [
        {
            "content": "User's default shell is zsh.",
            "category": "user_pref",
            "tags": None,
        }
    ]


def test_auto_extract_retains_multiword_favorite_subject():
    provider = _provider_with_fake_store()
    message = "For reference, my favorite text editor is Neovim"

    provider._auto_extract_facts([{"role": "user", "content": message}])

    assert provider._store.facts == [
        {
            "content": "User's favorite text editor is Neovim.",
            "category": "user_pref",
            "tags": None,
        }
    ]


def test_auto_extract_retains_preferred_subject_case():
    provider = _provider_with_fake_store()
    message = "My Preferred IDE is VS Code."

    provider._auto_extract_facts([{"role": "user", "content": message}])

    assert provider._store.facts == [
        {
            "content": "User's preferred IDE is VS Code.",
            "category": "user_pref",
            "tags": None,
        }
    ]


def test_auto_extract_saves_project_fact_not_raw_message():
    provider = _provider_with_fake_store()
    message = "After the spike, we decided to keep the browser worker site-agnostic."

    provider._auto_extract_facts([{"role": "user", "content": message}])

    assert provider._store.facts == [
        {
            "content": "Project decision: keep the browser worker site-agnostic.",
            "category": "project",
            "tags": None,
        }
    ]
    assert provider._store.facts[0]["content"] != message
