"""Inspect Hermes stdio layers without invoking stream attribute delegation."""


def find_stdio_wrapper(stream, wrapper_type):
    """Find an installed layer through cooperating Hermes wrappers only.

    Inspect the class marker, not the stream: ``__getattr__`` would forward
    the marker through routing (possibly to the current thread's sink).
    Unknown streams, including global redirect targets, are opaque boundaries.
    Never unwrap/rebind the actual output path: that would bypass redaction.
    """
    seen = set()
    while stream is not None:
        if id(stream) in seen:
            raise ValueError("Cycle in Hermes stdio wrappers")
        seen.add(id(stream))
        if isinstance(stream, wrapper_type):
            return stream
        inner_attr = getattr(type(stream), "_hermes_stdio_inner_attr", None)
        if inner_attr is None:
            return None
        stream = object.__getattribute__(stream, inner_attr)
    return None
