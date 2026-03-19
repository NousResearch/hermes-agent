import zlib


def compress(data: bytes) -> bytes:
    """Compress data with zlib deflate (level 6)."""
    return zlib.compress(data, level=6)


def decompress(data: bytes) -> bytes:
    """Decompress zlib-compressed data."""
    return zlib.decompress(data)
