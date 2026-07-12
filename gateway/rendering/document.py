"""Platform-neutral document model for rendering gateway replies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class ParagraphBlock:
    text: str


@dataclass(frozen=True)
class HeadingBlock:
    level: int
    text: str


@dataclass(frozen=True)
class CodeBlock:
    language: str
    code: str


@dataclass(frozen=True)
class TableBlock:
    headers: list[str]
    rows: list[list[str]]
    raw_markdown: str = ""


@dataclass(frozen=True)
class DividerBlock:
    pass


@dataclass(frozen=True)
class ImageBlock:
    source: str
    alt: str = ""


@dataclass(frozen=True)
class ListBlock:
    ordered: bool
    items: list[str]


Block = Union[
    ParagraphBlock,
    HeadingBlock,
    CodeBlock,
    TableBlock,
    DividerBlock,
    ImageBlock,
    ListBlock,
]


@dataclass(frozen=True)
class MessageDocument:
    blocks: list[Block]
