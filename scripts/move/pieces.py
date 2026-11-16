"""Edit the text of a string expression made of literals and f-strings without touching what it evaluates around.

A program or command can be one literal, an implicit concatenation, or an f-string; its text is the
evaluated literal parts with a placeholder for each ``{expression}``. An edit of that text is applied to the
one literal piece it falls in, in source spelling; an edit that crosses pieces, touches a placeholder, or
needs an escape the piece cannot spell gives None.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass

import libcst as cst

_SIMPLE_ESCAPES = {"n": "\n", "t": "\t", "\\": "\\", "'": "'", '"': '"'}


@dataclass
class _Piece:
    node: cst.CSTNode
    """A SimpleString, a FormattedStringText, or a FormattedStringExpression (placeholder)."""
    raw: str
    text: str
    offsets: list[int] | None
    """Source offset of each text character, plus one past the end; None when the piece cannot be edited."""
    quote: str = '"'
    fstring: bool = False


def text_of(node: cst.BaseExpression) -> tuple[str, list[_Piece]] | None:
    """The text ``node`` stands for and its pieces; None for anything but literals and f-strings."""
    pieces: list[_Piece] = []
    if not _collect(node, pieces):
        return None
    return "".join(piece.text for piece in pieces), pieces


def _collect(node: cst.CSTNode, pieces: list[_Piece]) -> bool:
    if isinstance(node, cst.ConcatenatedString):
        return _collect(node.left, pieces) and _collect(node.right, pieces)
    if isinstance(node, cst.SimpleString):
        if "b" in node.prefix.lower():
            return False
        head = len(node.prefix) + len(node.quote)
        raw = node.value[head:-len(node.quote)]
        raw_prefix = "r" in node.prefix.lower()
        text, offsets = (raw, list(range(len(raw) + 1))) if raw_prefix else _decode(raw, fstring=False)
        if text is None:
            text, offsets = node.evaluated_value, None
        pieces.append(_Piece(node, raw, text, offsets, node.quote, False))
        return True
    if isinstance(node, cst.FormattedString):
        raw_prefix = "r" in node.prefix.lower()
        quote = node.start[len(node.prefix):]
        for part in node.parts:
            if isinstance(part, cst.FormattedStringText):
                text, offsets = (part.value, list(range(len(part.value) + 1))) if raw_prefix \
                    else _decode(part.value, fstring=True)
                if text is None:
                    return False
                pieces.append(_Piece(part, part.value, text, offsets, quote, True))
            else:
                marker = f"__hermes_fexpr_{len(pieces)}__"
                pieces.append(_Piece(part, "", marker, None, quote, True))
        return True
    return False


def _decode(raw: str, fstring: bool) -> tuple[str | None, list[int] | None]:
    """Evaluate the simple escapes of ``raw``; (None, None) when it uses any other escape."""
    text, offsets, at = [], [], 0
    while at < len(raw):
        char = raw[at]
        if char == "\\":
            following = raw[at + 1:at + 2]
            if following == "\n":
                at += 2
                continue
            if following not in _SIMPLE_ESCAPES:
                return None, None
            offsets.append(at)
            text.append(_SIMPLE_ESCAPES[following])
            at += 2
        elif fstring and char in "{}" and raw[at + 1:at + 2] == char:
            offsets.append(at)
            text.append(char)
            at += 2
        else:
            offsets.append(at)
            text.append(char)
            at += 1
    offsets.append(len(raw))
    return "".join(text), offsets


def respell(node: cst.BaseExpression, pieces: list[_Piece], new: str) -> cst.BaseExpression | None:
    """``node`` whose text reads ``new``; None when an edit crosses pieces or cannot be spelled."""
    text = "".join(piece.text for piece in pieces)
    starts, at = [], 0
    for piece in pieces:
        starts.append(at)
        at += len(piece.text)
    raws = [piece.raw for piece in pieces]
    edits = [op for op in difflib.SequenceMatcher(None, text, new, autojunk=False).get_opcodes() if op[0] != "equal"]
    for _, i1, i2, j1, j2 in reversed(edits):
        index = _piece_at(starts, pieces, i1, i2)
        if index is None:
            return None
        piece = pieces[index]
        inserted = new[j1:j2]
        if piece.offsets is None or not _spellable(inserted, piece):
            return None
        local1, local2 = i1 - starts[index], i2 - starts[index]
        r1, r2 = piece.offsets[local1], piece.offsets[local2]
        raws[index] = raws[index][:r1] + inserted + raws[index][r2:]
    replaced = {id(piece.node): raw for piece, raw in zip(pieces, raws) if raw != piece.raw}
    return _rebuild(node, replaced)


def _piece_at(starts: list[int], pieces: list[_Piece], i1: int, i2: int) -> int | None:
    """The literal piece holding ``[i1, i2)``; an insertion between two pieces goes to the one that starts there."""
    for index, (start, piece) in enumerate(zip(starts, pieces)):
        end = start + len(piece.text)
        if start <= i1 < end or (i1 == end and index == len(pieces) - 1):
            return index if i2 <= end else None
    return None


def _spellable(inserted: str, piece: _Piece) -> bool:
    if "\\" in inserted or piece.quote[0] in inserted:
        return False
    if len(piece.quote) == 1 and "\n" in inserted:
        return False
    return not (piece.fstring and ("{" in inserted or "}" in inserted))


def _rebuild(node: cst.CSTNode, replaced: dict[int, str]) -> cst.CSTNode:
    if isinstance(node, cst.ConcatenatedString):
        return node.with_changes(left=_rebuild(node.left, replaced), right=_rebuild(node.right, replaced))
    if isinstance(node, cst.SimpleString) and id(node) in replaced:
        head = len(node.prefix) + len(node.quote)
        return node.with_changes(value=node.value[:head] + replaced[id(node)] + node.quote)
    if isinstance(node, cst.FormattedString):
        return node.with_changes(parts=[part.with_changes(value=replaced[id(part)]) if id(part) in replaced else part
                                        for part in node.parts])
    return node
