"""Regression tests for the SKIN YAML SCHEMA example in skin_engine's docstring.

The module docstring embeds a ``.. code-block:: yaml`` example that documents
every supported skin field. Users copy this example to author custom skins, so
it must be valid YAML with no duplicate mapping keys: a duplicate key silently
shadows the earlier one and mis-documents which field a comment refers to.
"""

import re

import yaml

import hermes_cli.skin_engine as skin_engine


class _DuplicateKeyError(Exception):
    """Raised when a YAML mapping contains a duplicate key."""


class _DuplicateDetectingLoader(yaml.SafeLoader):
    """SafeLoader that rejects mappings with duplicate keys.

    PyYAML's default behaviour is to silently let a later key overwrite an
    earlier one. For a documentation example that is exactly the bug we want to
    catch, so we override mapping construction to raise instead.
    """


def _construct_mapping_no_dups(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise _DuplicateKeyError(key)
        value = loader.construct_object(value_node, deep=deep)
        mapping[key] = value
    return mapping


_DuplicateDetectingLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping_no_dups,
)


def _extract_schema_yaml() -> str:
    """Return the dedented YAML body of the docstring's code-block example."""
    doc = skin_engine.__doc__
    assert doc, "skin_engine module docstring is missing"

    marker = ".. code-block:: yaml"
    assert marker in doc, "skin_engine docstring lost its YAML code-block example"

    body = doc.split(marker, 1)[1]
    lines = body.splitlines()

    # Skip blank lines immediately after the marker, then collect the indented
    # block until indentation drops back to column 0 (end of the example).
    collected: list[str] = []
    started = False
    for line in lines:
        if not started:
            if line.strip() == "":
                continue
            started = True
        if line.strip() == "":
            collected.append("")
            continue
        if not line.startswith("    "):
            break
        collected.append(line[4:])

    return "\n".join(collected)


def test_schema_example_is_valid_yaml_without_duplicate_keys():
    """The whole documented schema example must parse with no duplicate keys."""
    schema_yaml = _extract_schema_yaml()
    try:
        yaml.load(schema_yaml, Loader=_DuplicateDetectingLoader)
    except _DuplicateKeyError as exc:  # pragma: no cover - failure path
        raise AssertionError(
            f"SKIN YAML SCHEMA example has a duplicate key: {exc.args[0]!r}"
        ) from exc


def test_colors_block_has_no_duplicate_keys():
    """Each color field is documented exactly once in the colors mapping."""
    schema_yaml = _extract_schema_yaml()

    color_keys: list[str] = []
    in_colors = False
    for raw in schema_yaml.splitlines():
        stripped = raw.strip()
        if stripped.startswith("#") or not stripped:
            continue
        # A top-level key (no leading whitespace) other than `colors:` ends the
        # colors mapping.
        if not raw.startswith(" "):
            in_colors = stripped.startswith("colors:")
            continue
        if not in_colors:
            continue
        m = re.match(r'([a-z_]+):\s*"#', stripped)
        if m:
            color_keys.append(m.group(1))

    assert color_keys, "no color keys parsed from the schema example"
    duplicates = sorted({k for k in color_keys if color_keys.count(k) > 1})
    assert not duplicates, (
        "SKIN YAML SCHEMA colors example documents these keys more than once: "
        f"{duplicates}"
    )


def test_documented_color_keys_are_consumed_by_the_engine():
    """voice_status_bg (the key the duplicate was meant to be) is a real field.

    Guards against re-introducing a bogus key: every color key in the example
    must correspond to a field the engine actually reads via get_color().
    """
    schema_yaml = _extract_schema_yaml()
    source = (skin_engine.__file__ or "").replace(".pyc", ".py")
    engine_src = open(source, encoding="utf-8").read()

    in_colors = False
    for raw in schema_yaml.splitlines():
        stripped = raw.strip()
        if stripped.startswith("#") or not stripped:
            continue
        if not raw.startswith(" "):
            in_colors = stripped.startswith("colors:")
            continue
        if not in_colors:
            continue
        m = re.match(r'([a-z_]+):\s*"#', stripped)
        if not m:
            continue
        key = m.group(1)
        assert f'"{key}"' in engine_src, (
            f"documented color key {key!r} is never referenced by skin_engine"
        )
