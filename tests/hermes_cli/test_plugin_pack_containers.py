"""Plugin-pack tuple and alias regressions from #85057, after #118838."""

import json
from datetime import date
from textwrap import indent

import pytest

from hermes_cli.plugin_packs import PackError, _strip_forbidden_keys, export_pack, parse_pack


def _pack(config: str) -> str:
    return (
        "name: containers\nplugins:\n  - repo: example/plugin\n"
        f"    ref: {'a' * 40}\nconfig:\n  demo:\n" + indent(config, "    ")
    )


@pytest.mark.parametrize("key", ["api_key", "capabilities_consent", "allow_shell"])
@pytest.mark.parametrize("nested", [False, True])
def test_pack_refuses_forbidden_keys_inside_yaml_pairs(key, nested):
    entry = f"{key}: synthetic-test-value\n"
    if nested:
        entry = "primary:\n" + indent(entry, "  ")
    text = _pack("settings: !!pairs\n  - " + entry.replace("\n", "\n    ").rstrip() + "\n")

    with pytest.raises(PackError, match=key):
        parse_pack(text)


def test_export_filters_pairs_through_real_config_and_preserves_safe_values(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    plugins = tmp_path / "plugins"
    (plugins / "demo").mkdir(parents=True)
    (plugins / ".install-metadata.json").write_text(json.dumps({
        "demo": {"source": "https://github.com/example/plugin.git", "revision": "a" * 40},
    }), encoding="utf-8")
    (tmp_path / "config.yaml").write_text(
        "plugins:\n  entries:\n    demo:\n"
        "      primary: &shared\n        voice: nova\n"
        "      secondary: *shared\n"
        "      settings: !!pairs\n"
        "        - provider:\n            voice: nova\n            api_key: nested-test-value\n"
        "        - password: direct-test-value\n"
        "        - theme: dark\n",
        encoding="utf-8",
    )

    text, warnings = export_pack()

    assert warnings == []
    assert "nested-test-value" not in text
    assert "direct-test-value" not in text
    assert parse_pack(text).config["demo"] == {
        "primary": {"voice": "nova"},
        "secondary": {"voice": "nova"},
        "settings": [["provider", {"voice": "nova"}], ["theme", "dark"]],
    }


@pytest.mark.parametrize("node", ["voice: nova\nloop: *node\n", "- voice: nova\n- *node\n"])
def test_pack_rejects_cyclic_aliases_with_a_pack_error(node):
    text = _pack("node: &node\n" + indent(node, "  "))

    with pytest.raises(PackError, match="cyclic"):
        parse_pack(text)


@pytest.mark.parametrize("container", [dict, list])
def test_export_omits_cycles_without_losing_shared_acyclic_values(container):
    shared = {"voice": "nova"}
    node = container()
    if isinstance(node, dict):
        node.update(safe=shared, loop=node)
        expected_node = {"safe": shared}
    else:
        node.extend([shared, node])
        expected_node = [shared]
    seed = {"node": node, "primary": shared, "secondary": shared}

    assert _strip_forbidden_keys(seed) == {
        "node": expected_node, "primary": shared, "secondary": shared,
    }
    # Sanitizing must not mutate either the caller's cycle or its shared data.
    assert (node["loop"] if isinstance(node, dict) else node[1]) is node
    assert shared == {"voice": "nova"}


@pytest.mark.parametrize("unsupported", [{"set"}, date(2026, 8, 13), b"bytes", object()],
                         ids=["set", "date", "bytes", "object"])
def test_export_retains_the_unsupported_leaf_filter(unsupported):
    # Retain the original PR's follow-up contract when adding tuple traversal.
    assert _strip_forbidden_keys({
        "scalars": ["kept", 7, 1.5, True, None],
        "mapping": {"kept": "yes", "unsupported": unsupported},
        "sequence": ["first", unsupported, {"kept": "last"}],
        "tuple": ("first", unsupported, "last"),
    }) == {
        "scalars": ["kept", 7, 1.5, True, None],
        "mapping": {"kept": "yes"},
        "sequence": ["first", {"kept": "last"}],
        "tuple": ["first", "last"],
    }
