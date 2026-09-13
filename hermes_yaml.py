"""Shared YAML 1.1 policy for config, manifests, and frontmatter.

Ruamel's native schema includes bare y/n booleans and rejects duplicate keys.
Every operation owns its parser/emitter; instances must not be shared by threads.
"""

from io import StringIO
from typing import Any, IO, overload

from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError as YAMLError
from ruamel.yaml.resolver import VersionedResolver


class _Yaml11Resolver(VersionedResolver):
    # Quote strings like "off" without adding a %YAML directive to every config/snippet.
    @property
    def processing_version(self) -> tuple[int, int]:
        return (1, 1)


def safe_load(stream: str | bytes | IO[str] | IO[bytes]) -> Any:
    """Read standard YAML data; existing configs use YAML 1.1 booleans."""
    yaml = YAML(typ="safe")
    yaml.version = (1, 1)
    return yaml.load(stream)


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


def roundtrip_yaml() -> YAML:
    """Create a fresh comment/quote-preserving editor for user-authored YAML."""
    yaml = YAML(typ="rt")
    yaml.Resolver = _Yaml11Resolver
    yaml.preserve_quotes = True
    yaml.allow_unicode = True
    yaml.default_flow_style = False
    yaml.indent(mapping=2, sequence=4, offset=2)
    return yaml
