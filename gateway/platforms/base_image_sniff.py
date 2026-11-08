"""Magic-byte sniffing for the inbound image cache (``cache_image_from_bytes``)."""

# ISO-BMFF stills: iOS sends uncompressed photos as HEIC documents. Brand detection
# is agent.image_routing's, which transcodes these to PNG before vision sees them.
_ISOBMFF_IMAGE_MIMES = frozenset({"image/heic", "image/avif"})


def _looks_like_image(data: bytes) -> bool:
    """Return True if *data* starts with a known image magic-byte sequence."""
    if len(data) >= 4 and (data[:8] == b"\x89PNG\r\n\x1a\n" or data[:3] == b"\xff\xd8\xff"
                           or data[:6] in {b"GIF87a", b"GIF89a"} or data[:2] == b"BM"
                           or (data[:4] == b"RIFF" and len(data) >= 12 and data[8:12] == b"WEBP")):
        return True
    if data[4:8] != b"ftyp":
        return False
    from agent.image_routing import _sniff_mime_from_bytes
    return _sniff_mime_from_bytes(data[:16]) in _ISOBMFF_IMAGE_MIMES
