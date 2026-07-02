import functools
import typing

from ._meta import PackageMetadata

md_none = functools.partial(typing.cast, PackageMetadata)
"""
Suppress type errors for optional metadata.

Although Distribution.metadata can return None when metadata is corrupt
and thus None, allow callers to assume it's not None and crash if
that's the case.

# python/importlib_metadata#493
"""
