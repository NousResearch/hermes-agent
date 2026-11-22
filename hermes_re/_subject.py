"""Subject preparation: the buffer RE2 scans for a ``str`` subject, and the two-slot cache of it.

RE2 works on bytes. An ASCII ``str`` is scanned as its ASCII bytes (byte offset == index); any
other ``str`` as UTF-8 with a sparse offset index (``_offsets``); a ``str`` with lone surrogates
as a same-length stand-in text. Encoding costs time proportional to the text, on every call, so
the encoded form of the text in use is kept:

  slot 1  the last subject of ``MIN_CACHED`` to ``MAX_CACHED`` characters. Many patterns usually
          scan one text in a row (redaction, scanners). Kept until another subject of that size
          range arrives, or ``purge()``.
  slot 2  a subject ABOVE ``MAX_CACHED`` characters that is used twice in a row, so a
          ``search(text, pos)`` loop or a pass of many patterns over a huge text does not encode
          it again on every call. A huge text must not stay pinned after its use, so:
          - a single call never pins it: the first call only remembers ``id(text)``; the entry is
            made when the next huge subject is the same object (one extra encoding for a loop,
            nothing kept for a one-shot ``sub`` over a 100 MB text);
          - the entry is paid off by other work: every OTHER ``str`` subject prepared charges its
            length (at least ``MIN_CACHED``) against it, a use of the huge subject resets the
            charge, and when the charge reaches the huge subject's own length it is dropped.
            Re-encoding can therefore never cost more than the other work done in between (no
            thrashing), and the text lives on only until other matching of its own size has
            happened, another huge subject takes the slot, or ``purge()``.

Shorter subjects are not cached: encoding them is cheaper than the bookkeeping.

A search that ``pos`` / ``endpos`` confine to at most ``WINDOW`` characters of a subject that is
NOT cached does not prepare the subject at all: ``window`` encodes that range alone.
"""
from __future__ import annotations

from ._offsets import Offsets

MIN_CACHED, MAX_CACHED = 1024, 4_194_304  # characters
WINDOW = 32768  # a ``pos`` / ``endpos`` range up to this long is encoded by itself (``window``)
_SURROGATES = dict.fromkeys(range(0xD800, 0xE000), 0xFFFD)
# How a subject's match offsets map back to the caller's object.
DIRECT, UTF8, STANDIN = 0, 1, 2  # index == offset | UTF-8 byte offsets | U+FFFD stand-in text
_EMPTY = (None, None, DIRECT, None)
_last_subject = _EMPTY  # slot 1: (str, buffer, mode, offset index or None)
_large = _EMPTY         # slot 2, same shape
_charge = 0             # characters of other subjects prepared since slot 2 was last used
_sighted = 0            # id() of the last huge subject that was NOT kept: a hint, never dereferenced


def _pay(large: tuple, size: int) -> None:
    """Charge ``size`` characters of other work against the slot-2 entry ``large``."""
    global _large, _charge
    _charge += max(MIN_CACHED, size)
    if _charge >= len(large[0]):
        _large = _EMPTY


def subject(string):
    """``(buffer RE2 scans, offset mode, sparse offset index or None)`` for a str subject.
    The index exists for every non-ASCII subject of ``MIN_CACHED`` characters or more, cached
    or not: the matches of one call share it, and for a cached subject ``pos`` conversions are
    O(1) across calls."""
    global _last_subject, _large, _charge, _sighted
    cached, large = _last_subject, _large
    if cached[0] is string:
        if large[0] is not None:
            _pay(large, len(string))
        return cached[1], cached[2], cached[3]
    if large[0] is string:
        _charge = 0
        return large[1], large[2], large[3]
    size = len(string)
    if large[0] is not None:
        _pay(large, size)
    if string.isascii():
        data, mode = string.encode("ascii"), DIRECT
    else:
        try:
            data, mode = string.encode("utf-8"), UTF8
        except UnicodeEncodeError:  # lone surrogates: scan a same-length stand-in
            data, mode = string.translate(_SURROGATES).encode("utf-8"), STANDIN
    offsets = None
    if size >= MIN_CACHED:
        if mode != DIRECT:
            offsets = Offsets(data)
        if size <= MAX_CACHED:
            _last_subject = (string, data, mode, offsets)
        elif _sighted == id(string):  # second use in a row (a stale id only costs a wasted entry)
            _large, _charge = (string, data, mode, offsets), 0
        else:
            _sighted = id(string)
    return data, mode, offsets


def held(string) -> bool:
    """Whether ``string`` is one of the cached subjects (its buffer costs nothing to get)."""
    return _last_subject[0] is string or _large[0] is string


def window(string, first: int, last: int):
    """``(buffer, mode, the text it stands for)`` for ``string[first:last]`` alone: what a search
    confined by ``pos`` / ``endpos`` needs of a subject that is not cached. A stream buffer is a
    NEW ``str`` on every tick, so encoding all of it for a search in its last few hundred
    characters would cost time proportional to the whole buffer, per call. Never cached."""
    large = _large
    if large[0] is not None:
        _pay(large, last - first)
    part = string[first:last]
    if part.isascii():
        return part.encode("ascii"), DIRECT, part
    try:
        return part.encode("utf-8"), UTF8, part
    except UnicodeEncodeError:
        return part.translate(_SURROGATES).encode("utf-8"), STANDIN, part


def flat(string):
    """A bytes-like subject as an object indexed by BYTE, as stdlib reads it: the object itself
    when it already is one (bytes, bytearray, mmap, a one-dimensional view of single bytes), else a
    byte view of its memory (``memoryview(b"abcd").cast("I")`` is four bytes, not one element).
    Raises the TypeError stdlib raises for anything else, a non-contiguous view included."""
    if string.__class__ is bytes or string.__class__ is bytearray:
        return string
    try:
        view = memoryview(string)
    except TypeError:
        view = None
    if view is None or not view.c_contiguous:
        raise TypeError(f"expected string or bytes-like object, got {type(string).__name__!r}")
    if view.ndim == 1 and view.itemsize == 1:
        return string
    return view.cast("B")


def clear() -> None:
    """Drop both cached subjects (called by ``purge()``)."""
    global _last_subject, _large, _charge, _sighted
    _last_subject, _large, _charge, _sighted = _EMPTY, _EMPTY, 0, 0
