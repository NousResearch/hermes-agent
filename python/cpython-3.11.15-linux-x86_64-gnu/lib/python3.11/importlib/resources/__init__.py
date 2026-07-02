"""Read resources contained within a package."""

from ._common import (
    as_file,
    files,
    Package,
)

from ._legacy import (
    contents,
    open_binary,
    read_binary,
    open_text,
    read_text,
    is_resource,
    path,
    Resource,
)

from .abc import ResourceReader


__all__ = [
    'Package',
    'Resource',
    'ResourceReader',
    'as_file',
    'contents',
    'files',
    'is_resource',
    'open_binary',
    'open_text',
    'path',
    'read_binary',
    'read_text',
]
