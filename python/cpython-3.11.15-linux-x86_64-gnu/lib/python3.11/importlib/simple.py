"""
Compatibility shim for .resources.simple as found on Python 3.10.

Consumers that can rely on Python 3.11 should use the other
module directly.
"""

from .resources.simple import (
    SimpleReader, ResourceHandle, ResourceContainer, TraversableReader,
)

__all__ = [
    'SimpleReader', 'ResourceHandle', 'ResourceContainer', 'TraversableReader',
]
