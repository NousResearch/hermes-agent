"""Canonical parse/serialize for dotted config-key paths.

One representation, used everywhere a config path is tokenized or rendered:
the CLI helpers in ``hermes_cli/config.py``, the managed-scope key flattening
and comparison in ``hermes_cli/managed_scope.py``, and ``_strip_dotted_keys``.

Why this is its own module: ``managed_scope`` deliberately imports nothing
from ``config`` (``config`` imports ``managed_scope``, not the other way
round), so the shared helper cannot live in ``config`` without inverting that
dependency.  This module is a leaf — stdlib only, no ``hermes_cli`` imports.

The representation is dot-separated segments, where a segment that contains a
dot is wrapped in double quotes::

    platforms.api_server.extra.model_routes."gpt-5.6-sol".model
    -> ['platforms', 'api_server', 'extra', 'model_routes',
        'gpt-5.6-sol', 'model']

Backward compatibility is the load-bearing property: a key with no dot-bearing
segment tokenizes exactly as ``key.split(".")`` did, and serializes back
byte-for-byte unchanged.
"""

from __future__ import annotations

from typing import Iterable, List


def split_dotted_key(dotted_key: str) -> List[str]:
    """Split a dotted config path into segments, honouring double-quoted spans.

    A double-quoted span is taken literally, so a key segment that legitimately
    contains a dot can be addressed.  Every real model id contains one
    (``gpt-5.6-sol``, ``claude-sonnet-4.5``), which is what otherwise makes
    ``platforms.api_server.extra.model_routes.<model-id>`` unreachable::

        platforms.api_server.extra.model_routes."gpt-5.6-sol".model
        -> ['platforms', 'api_server', 'extra', 'model_routes',
            'gpt-5.6-sol', 'model']

    Input without quotes tokenizes byte-for-byte identically to
    ``dotted_key.split(".")``, so this is fully backward compatible.

    Raises ``ValueError`` on an unterminated quote.  Note that the quote
    characters are structural and never appear in the output, so there is no
    way to express a literal ``"`` inside a segment — see ``join_dotted_key``.
    """
    parts: List[str] = []
    buf: List[str] = []
    in_quotes = False
    for ch in dotted_key:
        if ch == '"':
            in_quotes = not in_quotes
        elif ch == "." and not in_quotes:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if in_quotes:
        raise ValueError(f"unterminated quote in config key {dotted_key!r}")
    parts.append("".join(buf))
    return parts


def join_dotted_key(parts: Iterable[object]) -> str:
    """Serialize path segments into the canonical dotted key.

    A segment is double-quoted iff it contains a dot; every other segment is
    emitted bare.  So a path with no dot-bearing segment serializes exactly as
    ``".".join(parts)`` did, which is what keeps every existing managed key
    byte-for-byte stable.

    Segments are coerced with ``str()`` to match what ``managed_scope``'s
    flattening did with non-string YAML keys (ints, bools).  One deliberate
    consequence: a YAML float key (``4.5:``) now serializes as ``"4.5"``
    rather than bare ``4.5``.  That is the same defect this representation
    exists to fix — bare ``4.5`` addresses ``4`` → ``5``, a different path.

    This function never raises.  A segment containing a ``"`` cannot be
    represented (the tokenizer treats quotes as structure and strips them, so
    there is no escape), and is emitted bare — byte-identical to the previous
    behaviour, i.e. still not addressable, but no worse and no new failure.
    ``_flatten_keys`` walks whatever YAML an administrator wrote, and
    ``doctor.py`` calls it outside its ``try``: a raise here would be a new
    crash surface in ``hermes doctor``.
    """
    out: List[str] = []
    for part in parts:
        text = str(part)
        out.append(f'"{text}"' if "." in text else text)
    return ".".join(out)


def normalize_dotted_key(key: str) -> str:
    """Round-trip a dotted key through the canonical representation.

    Lets two spellings of the same path compare equal — e.g. a CLI-supplied
    ``model_routes."gpt-5.6-sol".model`` and the same path flattened out of
    managed YAML.  Redundant quoting on a dot-free segment is dropped.

    Raises ``ValueError`` on an unterminated quote (from ``split_dotted_key``).
    """
    return join_dotted_key(split_dotted_key(key))
