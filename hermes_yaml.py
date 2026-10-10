"""Shared YAML 1.1 policy for config, manifests, and frontmatter.

Ruamel's native schema includes bare y/n booleans and rejects duplicate keys.
Every operation owns its parser/emitter; instances must not be shared by threads.
"""

from __future__ import annotations

import re
from io import StringIO
from typing import Any, IO, overload

from ruamel.yaml import YAML
from ruamel.yaml.constructor import RoundTripConstructor, SafeConstructor
from ruamel.yaml.error import YAMLError as YAMLError
from ruamel.yaml.representer import RoundTripRepresenter, SafeRepresenter
from ruamel.yaml.resolver import VersionedResolver
from ruamel.yaml.tag import Tag

_FLOAT_TAG = "tag:yaml.org,2002:float"
_STR_TAG = "tag:yaml.org,2002:str"

# YAML 1.1 floats must carry a dot in the mantissa. PyYAML enforced that on both sides — its
# resolver kept dot-less ``123e45`` scalars strings, and its dumper inserted ``.0`` before any
# exponent — so a file PyYAML wrote parsed back cleanly under PyYAML. ruamel's YAML 1.1 resolver
# accepts the dot-less form (warn-and-continue), which silently re-typed legacy session-ID-shaped
# scalars (``20260820_093237_089e44``) into floats on load (#124901). Constructors and
# representers are dispatched by tag through class-level tables, overriding a method in a
# subclass alone does nothing; registering on the base class would rewire ruamel for the whole
# process, so each subclass registers on itself.
_UNSIGNED_DOTLESS_FLOAT = re.compile(r"^[-+]?[0-9][0-9_]*[eE][0-9]+$")


def _legacy_id_scalar(node: Any) -> bool:
    value = node.value
    return isinstance(value, str) and _UNSIGNED_DOTLESS_FLOAT.match(value) is not None


class _SafeConstructor(SafeConstructor):
    def construct_yaml_float(self, node: Any) -> Any:
        if _legacy_id_scalar(node):
            return node.value
        return super().construct_yaml_float(node)


_SafeConstructor.add_constructor(_FLOAT_TAG, _SafeConstructor.construct_yaml_float)


class _RoundTripConstructor(RoundTripConstructor):
    def construct_yaml_float(self, node: Any) -> Any:
        if _legacy_id_scalar(node):
            return node.value
        return super().construct_yaml_float(node)


_RoundTripConstructor.add_constructor(
    _FLOAT_TAG, _RoundTripConstructor.construct_yaml_float
)


def _dotted_mantissa(base: Any) -> Any:
    def _represent_float(representer: Any, data: Any) -> Any:
        node = base.represent_float(representer, data)
        if "e" in node.value:
            mantissa, _, exponent = node.value.partition("e")
            if "." not in mantissa:
                node.value = f"{mantissa}.0e{exponent}"
        return node

    return _represent_float


class _SafeRepresenter(SafeRepresenter):
    pass


_SafeRepresenter.add_representer(float, _dotted_mantissa(SafeRepresenter))


class _RoundTripRepresenter(RoundTripRepresenter):
    pass


_RoundTripRepresenter.add_representer(float, _dotted_mantissa(RoundTripRepresenter))


class _Yaml11Resolver(VersionedResolver):
    # Quote strings like "off" without adding a %YAML directive to every config/snippet.
    @property
    def processing_version(self) -> tuple[int, int]:
        return (1, 1)


def _load(document: str | bytes, *, pure: bool) -> Any:
    yaml = YAML(typ="safe", pure=pure)
    yaml.version = (1, 1)
    yaml.Constructor = _SafeConstructor
    return yaml.load(document)


def safe_load(stream: str | bytes | IO[str] | IO[bytes]) -> Any:
    """Read standard YAML data; existing configs use YAML 1.1 booleans.

    The pure parser defines what parses: Windows ARM64 has no C extension, and libyaml rejects
    documents the pure parser accepts (``[{url: http://h}]``), so a C rejection is re-read pure.
    """
    document = stream if isinstance(stream, (str, bytes)) else stream.read()
    try:
        return _load(document, pure=False)
    except YAMLError:
        return _load(document, pure=True)


@overload
def safe_dump(
    data: Any, stream: None = None, *, default_flow_style: bool = False,
    sort_keys: bool = True, allow_unicode: bool = True, width: int = 80,
) -> str: ...


@overload
def safe_dump(
    data: Any, stream: IO[str], *, default_flow_style: bool = False,
    sort_keys: bool = True, allow_unicode: bool = True, width: int = 80,
) -> None: ...


def safe_dump(
    data: Any,
    stream: IO[str] | None = None,
    *,
    default_flow_style: bool = False,
    sort_keys: bool = True,
    allow_unicode: bool = True,
    width: int = 80,
) -> str | None:
    """Write standard YAML data with readable Unicode and indented block lists."""
    # The C emitter ignores sequence offsets and escapes astral Unicode.
    yaml = YAML(typ="safe", pure=True)
    yaml.Resolver = _Yaml11Resolver
    yaml.Representer = _SafeRepresenter
    yaml.default_flow_style = default_flow_style
    yaml.allow_unicode = allow_unicode
    yaml.width = width
    yaml.sort_base_mapping_type_on_output = sort_keys
    yaml.indent(mapping=2, sequence=4, offset=2)
    if stream is not None:
        yaml.dump(data, stream)
        return None
    output = StringIO()
    yaml.dump(data, output)
    return output.getvalue()


# ruamel's emitter can change a double-quoted value when it folds a long line right after an
# escaped backslash (``D:\\Cent…`` → ``D:\\`` + bare newline): the fold reloads as a literal space
# and a no-op save mutates the stored value (#119844). Config writes must be value-preserving, so
# every round-trip emitter in the tree keeps scalars on one line instead of folding (``None``
# does NOT disable folding on 0.18.x; only a large width does).
ROUNDTRIP_YAML_WIDTH = 2**31 - 1


def roundtrip_yaml() -> YAML:
    """Create a fresh comment/quote-preserving editor for user-authored YAML."""
    yaml = YAML(typ="rt")
    yaml.width = ROUNDTRIP_YAML_WIDTH
    yaml.Resolver = _Yaml11Resolver
    yaml.Representer = _RoundTripRepresenter
    yaml.Constructor = _RoundTripConstructor
    yaml.preserve_quotes = True
    yaml.allow_unicode = True
    yaml.default_flow_style = False
    yaml.indent(mapping=2, sequence=4, offset=2)
    return yaml
