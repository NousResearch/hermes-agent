"""Frontmatter YAML is parsed once per distinct text, and callers never share the parsed objects."""
import pytest

from agent import skill_utils
from agent.skill_utils import parse_frontmatter


@pytest.fixture(autouse=True)
def counted_loads(monkeypatch):
    monkeypatch.setattr(skill_utils, "_FRONTMATTER_CACHE", {}, raising=False)
    calls = []
    real = skill_utils.yaml_load

    def counting(text):
        calls.append(text)
        return real(text)

    monkeypatch.setattr(skill_utils, "yaml_load", counting)
    return calls


def skill(name, tags=("a",)):
    return f"---\nname: {name}\ntags: [{', '.join(tags)}]\n---\nbody of {name}\n"


def test_same_frontmatter_is_parsed_once(counted_loads):
    first, body = parse_frontmatter(skill("alpha"))
    second, _ = parse_frontmatter(skill("alpha"))
    assert first == second == {"name": "alpha", "tags": ["a"]}
    assert body == "body of alpha\n"
    assert len(counted_loads) == 1


def test_callers_get_independent_copies():
    first, _ = parse_frontmatter(skill("alpha"))
    first["tags"].append("mutated")
    second, _ = parse_frontmatter(skill("alpha"))
    assert second["tags"] == ["a"]


def test_changed_frontmatter_is_parsed_again(counted_loads):
    parse_frontmatter(skill("alpha"))
    edited, _ = parse_frontmatter(skill("alpha", tags=("a", "b")))
    assert edited["tags"] == ["a", "b"]
    assert len(counted_loads) == 2


def test_malformed_yaml_is_not_cached_and_still_falls_back(counted_loads):
    text = "---\nname: broken\ndescription: [unclosed\n---\nbody\n"
    for _ in range(2):
        frontmatter, _ = parse_frontmatter(text)
        assert frontmatter["name"] == "broken"
    assert len(counted_loads) == 2


def test_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(skill_utils, "_FRONTMATTER_CACHE_MAX", 2)
    for name in ("a", "b", "c"):
        parse_frontmatter(skill(name))
    assert len(skill_utils._FRONTMATTER_CACHE) == 2
