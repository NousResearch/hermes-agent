"""Shared YAML 1.1 policy for config, manifests, and frontmatter.

Ruamel's native schema includes bare y/n booleans and rejects duplicate keys.
Every operation owns its parser/emitter; instances must not be shared by threads.
"""

from contextlib import suppress
from io import StringIO
from typing import Any, IO, overload

from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError
from ruamel.yaml.resolver import VersionedResolver


class _Yaml11Resolver(VersionedResolver):
    # Quote strings like "off" without adding a %YAML directive to every config/snippet.
    @property
    def processing_version(self) -> tuple[int, int]:
        return (1, 1)


# Loader attributes that form the self-referencing graph ``YAML.load`` leaves behind (see
# ``_release_loader``): the composer/parser/constructor/resolver aliases are the loader itself, the rest
# are its per-document state. The attributes are deleted, not set to None, so the lazy properties that
# rebuild them still work if a caller keeps the instance. Only names a loader declares are touched.
_LOADER_GRAPH_ATTRS = (
    "_parser", "_composer", "_constructor", "_resolver",  # C parser: aliases of the loader itself
    "loader", "loadumper",                                # pure parser: the same aliases by their public names
    "_reader", "_scanner", "_stream",                     # the stream the loader still shows
    "constructed_objects", "recursive_objects", "state_generators",  # per-document construction state
    "resolver_exact_paths", "resolver_prefix_paths",
)
# The loader is also parked on the ``YAML`` instance itself; delete those too.
_YAML_LOADER_ATTRS = ("_scanner", "_stream", "_reader", "_parser", "_composer", "_constructor", "_resolver")


def _release_loader(yaml: YAML) -> None:
    """Unlink the loader graph ``YAML.load`` leaves behind so refcounting frees it with the call.

    Ruamel parks the loader on ``yaml._scanner`` and gives the loader self-references
    (``loader._parser = loader._composer = loader``, with the constructor and resolver pointing back at it
    too), so no reference count ever reaches zero: the loader, the parsed document text and its libyaml
    buffers all survive until a gen-2 cyclic collection. A process that re-reads configs — the gateway's
    change watcher does ~590 reads/min across served homes — therefore accumulates them (#159451).
    Nothing reads the loader once ``load`` returns (the result is plain dicts/lists), and deleting rather
    than clearing the attributes keeps a reused instance working: the properties rebuild on next access.
    """
    loader = getattr(yaml, "_scanner", None)
    if loader is not None:
        for attr in _LOADER_GRAPH_ATTRS:
            with suppress(AttributeError):  # slots class without that attribute
                delattr(loader, attr)
    for attr in _YAML_LOADER_ATTRS:
        with suppress(AttributeError):
            delattr(yaml, attr)


def _load(document: str | bytes, *, pure: bool) -> Any:
    yaml = YAML(typ="safe", pure=pure)
    yaml.version = (1, 1)
    try:
        return yaml.load(document)
    finally:
        _release_loader(yaml)


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
