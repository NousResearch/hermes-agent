"""Injection-safe, deliberately small PostgreSQL/FTS5 lexical grammar bridge.

Only tokens whose FTS5 meaning can be expressed with built-in PostgreSQL text
search are accepted.  The compiler emits static SQL fragments and separately
bound values; user text is never interpolated into SQL or a tsquery expression.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


class PostgreSQLSearchQueryError(ValueError):
    """The caller supplied FTS5 syntax outside the documented compatibility subset."""


_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff\uac00-\ud7af]")
_CJK_LITERAL_RE = re.compile(r"^[A-Za-z0-9_\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff\uac00-\ud7af]+$")
_WORD_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class PostgreSQLSearchExpression:
    sql: str
    params: tuple[str, ...]
    is_cjk_literal: bool = False


@dataclass(frozen=True)
class _Node:
    kind: Literal["term", "prefix", "phrase", "and", "or", "not"]
    value: str | None = None
    left: "_Node | None" = None
    right: "_Node | None" = None


def _syntax_error(detail: str) -> PostgreSQLSearchQueryError:
    return PostgreSQLSearchQueryError(f"unsupported or ambiguous PostgreSQL lexical search syntax: {detail}")


class _Parser:
    def __init__(self, query: str) -> None:
        self.query = query
        self.tokens = self._tokenize(query)
        self.index = 0

    @staticmethod
    def _tokenize(query: str) -> list[tuple[str, str]]:
        tokens: list[tuple[str, str]] = []
        index = 0
        while index < len(query):
            character = query[index]
            if character.isspace():
                index += 1
            elif character in "()":
                raise _syntax_error("parenthesized grouping")
            elif character == '"':
                end = query.find('"', index + 1)
                if end < 0:
                    raise _syntax_error("unclosed quote")
                phrase = query[index + 1:end]
                if not phrase or not all(_WORD_RE.fullmatch(part) for part in phrase.split()):
                    raise _syntax_error("phrase must contain ASCII alphanumeric terms")
                tokens.append(("PHRASE", phrase)); index = end + 1
                if index < len(query) and query[index] == "*":
                    raise _syntax_error("phrase prefix queries")
            else:
                word_match = _WORD_RE.match(query, index)
                if word_match is None:
                    raise _syntax_error(f"character {character!r}")
                word = word_match.group(0)
                index += len(word)
                if index < len(query) and query[index] == "*":
                    tokens.append(("PREFIX", word)); index += 1
                else:
                    tokens.append((word if word in {"AND", "OR", "NOT", "NEAR"} else "TERM", word))
        return tokens

    def _peek(self, kind: str | None = None) -> tuple[str, str] | None:
        if self.index == len(self.tokens):
            return None
        token = self.tokens[self.index]
        return token if kind is None or token[0] == kind else None

    def _take(self, kind: str | None = None) -> tuple[str, str]:
        token = self._peek(kind)
        if token is None:
            raise _syntax_error("missing operand")
        self.index += 1
        return token

    def parse(self) -> _Node:
        if not self.tokens:
            raise _syntax_error("empty query")
        if self._peek("NOT"):
            raise _syntax_error("leading NOT")
        expression = self._or_expression()
        if self._peek() is not None:
            raise _syntax_error("unexpected token")
        return expression

    def _or_expression(self) -> _Node:
        node = self._and_expression()
        while self._peek("OR"):
            self._take("OR")
            node = _Node("or", left=node, right=self._and_expression())
        return node

    def _and_expression(self) -> _Node:
        node = self._primary()
        while True:
            if self._peek("AND"):
                self._take("AND")
                node = _Node("and", left=node, right=self._primary())
            elif self._peek("NOT"):
                self._take("NOT")
                node = _Node("and", left=node, right=_Node("not", right=self._primary()))
            else:
                token = self._peek()
                if token is not None and token[0] in {"TERM", "PREFIX", "PHRASE"}:
                    node = _Node("and", left=node, right=self._primary())
                else:
                    return node

    def _primary(self) -> _Node:
        if self._peek("TERM"):
            return _Node("term", value=self._take("TERM")[1])
        if self._peek("PREFIX"):
            return _Node("prefix", value=self._take("PREFIX")[1])
        if self._peek("PHRASE"):
            return _Node("phrase", value=self._take("PHRASE")[1])
        raise _syntax_error("missing operand")


def _compile(node: _Node) -> PostgreSQLSearchExpression:
    if node.kind == "term":
        return PostgreSQLSearchExpression("plainto_tsquery('simple', %s)", (node.value or "",))
    if node.kind == "prefix":
        # The parser restricts the token to ASCII alphanumerics before adding tsquery syntax.
        return PostgreSQLSearchExpression("to_tsquery('simple', %s)", (f"{node.value}:*",))
    if node.kind == "phrase":
        return PostgreSQLSearchExpression("phraseto_tsquery('simple', %s)", (node.value or "",))
    if node.kind == "not":
        right = _compile(node.right)  # type: ignore[arg-type]
        return PostgreSQLSearchExpression(f"!!({right.sql})", right.params)
    left, right = _compile(node.left), _compile(node.right)  # type: ignore[arg-type]
    operator = "&&" if node.kind == "and" else "||"
    return PostgreSQLSearchExpression(f"({left.sql} {operator} {right.sql})", left.params + right.params)


def compile_postgresql_search_expression(query: str) -> PostgreSQLSearchExpression:
    """Compile supported FTS5-compatible ASCII syntax or CJK literal fallback.

    Accepted ASCII grammar: terms, quoted phrases, ``term*`` prefixes, explicit
    ``AND``/``OR``/binary ``NOT``, and implicit AND. Parenthesized grouping is
    rejected because the SQLite compatibility facade does not preserve it. Operators are
    uppercase, as in FTS5.  CJK has no tokenizer-equivalent route: only literal
    whitespace-separated tokens are accepted and each is canonical-row matched.
    """
    if not isinstance(query, str) or not query.strip():
        raise _syntax_error("empty query")
    query = query.strip()
    if _CJK_RE.search(query):
        terms = query.split()
        if (not terms or any(not _CJK_LITERAL_RE.fullmatch(term) or term in {"AND", "OR", "NOT"} for term in terms)):
            raise _syntax_error("CJK queries support literal whitespace-separated tokens only")
        return PostgreSQLSearchExpression("", tuple(terms), is_cjk_literal=True)
    return _compile(_Parser(query).parse())
