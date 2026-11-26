"""Shared YAML 1.1 policy for config, manifests, and frontmatter.

Ruamel's native schema includes bare y/n booleans and rejects duplicate keys.
Every operation owns its parser/emitter; instances must not be shared by threads.
"""

import warnings
from io import StringIO
from typing import Any, IO, overload

from ruamel.yaml import YAML
from ruamel.yaml.error import MantissaNoDotYAML1_1Warning, YAMLError
from ruamel.yaml.resolver import VersionedResolver

# YAML 1.1's float type wants a dot in the mantissa, so ruamel lints scalars like ``5e-1`` even
# though it parses them correctly (0.5). Hermes asks for 1.1 to keep bare ``y``/``n`` booleans, so
# that lint fires on every such scalar, on every load, in every process — ~28 lines of stderr each,
# which buries CLI output and breaks callers that parse stderr. Silence that one category: the 1.1
# policy is deliberate here, and real faults still raise (duplicate keys and syntax errors arrive as
# YAMLError). Registered at import so both loaders are covered — ``roundtrip_yaml()`` resolves 1.1
# through ``_Yaml11Resolver`` rather than ``yaml.version``, and it lints too.
warnings.filterwarnings("ignore", category=MantissaNoDotYAML1_1Warning)


class _Yaml11Resolver(VersionedResolver):
    # Quote strings like "off" without adding a %YAML directive to every config/snippet.
    @property
    def processing_version(self) -> tuple[int, int]:
        return (1, 1)


def _load(document: str | bytes, *, pure: bool) -> Any:
    yaml = YAML(typ="safe", pure=pure)
    yaml.version = (1, 1)
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
    yaml.preserve_quotes = True
    yaml.allow_unicode = True
    yaml.default_flow_style = False
    yaml.indent(mapping=2, sequence=4, offset=2)
    return yaml
