"""Invariant tests for ``hermes config diff`` (hermes_cli/config_diff.py).

Contract, not snapshots: the diff must (1) mark a value that differs from the schema as
``changed`` with both values, (2) mark a key absent from the schema as ``user_only``,
(3) prune keys equal to the default, (4) mask credential-shaped leaves unless ``raw``.
"""

import json

import pytest

from hermes_cli import config_diff


@pytest.fixture
def fake_layers(monkeypatch, tmp_path):
    """Pin both sides of the diff: a tiny schema and a controlled user layer."""
    schema = {
        "agent": {"max_turns": 100, "nested": {"a": 1, "b": 2}},
        "display": {"streaming": False},
        "web": {"backend": ""},
        "same": {"x": 1},
    }
    user = {
        "agent": {"max_turns": 500, "nested": {"a": 1, "b": 99}, "extra": True},
        "display": {"streaming": True},
        "web": {"backend": "ddgs"},
        "same": {"x": 1},
        "api_key": "sk-supersecret-value",
    }
    monkeypatch.setattr(config_diff, "DEFAULT_CONFIG", schema)
    monkeypatch.setattr(
        "hermes_cli.config_effective.load_user_config_effective", lambda *a, **k: user)
    return schema, user


def _by_path(entries):
    return {".".join(p): (m, d, u) for m, p, d, u in entries}


def test_changed_and_user_only_markers(fake_layers):
    got = _by_path(config_diff.diff_entries())
    assert got["agent.max_turns"][0] == "changed"
    assert got["agent.max_turns"][1] == 100 and got["agent.max_turns"][2] == 500
    assert got["agent.nested.b"][0] == "changed"
    assert got["agent.extra"][0] == "user_only"
    assert got["display.streaming"][0] == "changed"
    assert got["web.backend"][0] == "changed"


def test_keys_equal_to_default_are_pruned(fake_layers):
    paths = {".".join(p) for _, p, _, _ in config_diff.diff_entries()}
    assert "agent.nested.a" not in paths  # user == default -> no diff line


def test_credential_leaf_masked_unless_raw(fake_layers):
    masked = _by_path(config_diff.diff_entries(raw=False))["api_key"][2]
    assert "supersecret" not in masked
    raw = _by_path(config_diff.diff_entries(raw=True))["api_key"][2]
    assert raw == "sk-supersecret-value"


def test_tree_renders_nested_sections(fake_layers):
    lines = config_diff.format_tree(config_diff.diff_entries())
    assert "agent:" in lines
    assert any(l.strip().startswith("~ max_turns: 500") for l in lines)
    assert any(l.strip().startswith("+ extra: True") for l in lines)


def test_json_shape(fake_layers):
    payload = json.loads(config_diff.format_json(config_diff.diff_entries()))
    assert set(payload) == {"changed", "user_only"}
    assert payload["changed"]["agent.max_turns"] == {"default": 100, "user": 500}
    assert payload["user_only"]["agent.extra"] is True


def test_subtree_scope(fake_layers):
    got = _by_path(config_diff.diff_entries(key="agent"))
    assert set(got) == {"max_turns", "nested.b", "extra"}  # only the agent subtree
    assert "display.streaming" not in got
    deep = _by_path(config_diff.diff_entries(key="agent.nested"))
    assert set(deep) == {"b"}


def test_scalar_subtree_diff(fake_layers):
    got = _by_path(config_diff.diff_entries(key="agent.max_turns"))
    assert got[("")] == ("changed", 100, 500)


def test_unknown_subtree_raises(fake_layers):
    with pytest.raises(KeyError):
        config_diff.diff_entries(key="nosuch.nope")


def test_subtree_equal_to_default_is_empty(fake_layers):
    assert config_diff.diff_entries(key="same") == []  # user == default -> no diff
