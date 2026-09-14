"""Turning a model's question into a safe FTS5 query, and reading the results back.

This module is **copied verbatim** into each agent's knowledge plugin, exactly as
``nova/policy/decide.py`` is copied into the policy plugin. It therefore imports nothing
from NOVA and nothing from the runtime: standard library only. Two copies of one function
is a worse thing to own than one copy in two places, and the alternative — the plugin
importing NOVA from inside a worker process — is the dependency arrow the architecture
exists to prevent.

The important function here is :func:`build_match_expression`. Everything a model types
arrives as adversarial input to SQLite's FTS5 query parser, which has its own syntax:
``NEAR``, ``*``, ``^``, column filters, boolean operators. Passing a raw question through
gets you, at best, ``OperationalError: fts5: syntax error`` on any question containing an
apostrophe, and at worst a query that reaches past its intended scope. So nothing is passed
through. The question is tokenized into words and phrases, every token is re-quoted, and the
expression is rebuilt from scratch out of parts this module chose.
"""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Any, Optional, Sequence

#: Words carrying no retrieval signal. Dropped only when other terms survive — a search for
#: "who is on call" must not become a search for nothing.
STOPWORDS = frozenset(
    {
        "a", "an", "and", "any", "are", "as", "at", "be", "by", "can", "do", "does", "for",
        "from", "how", "i", "in", "is", "it", "of", "on", "or", "our", "that", "the", "to",
        "was", "we", "what", "when", "where", "which", "who", "why", "with", "you",
    }
)

#: A run of word characters, or anything inside double quotes. Everything else in the
#: question — punctuation, FTS5 operators, stray quotes — is discarded rather than escaped.
_TOKEN = re.compile(r'"([^"]{1,200})"|(\w{1,64})', re.UNICODE)

#: Bounds the expression so a pathological question cannot become a pathological query.
MAX_TERMS = 24

#: Text markers around a matched term in a snippet. Plain brackets rather than markup: the
#: result is read by a model, and inventing a markup dialect invites it to emit one back.
SNIPPET_OPEN = "[["
SNIPPET_CLOSE = "]]"
SNIPPET_ELLIPSIS = " … "
SNIPPET_TOKENS = 28


class EmptyQuery(ValueError):
    """The question contained nothing searchable."""


def build_match_expression(question: str, *, require_all: bool = True) -> str:
    """A safe FTS5 MATCH expression for *question*.

    Quoted runs in the question are preserved as phrases; everything else becomes a quoted
    single term. Terms are joined with ``AND`` for precision, or ``OR`` when the caller is
    widening a search that returned nothing.
    """
    terms: list[str] = []
    for phrase, word in _TOKEN.findall(question or ""):
        token = (phrase or word).strip()
        if not token:
            continue
        terms.append(token if phrase else token.lower())
        if len(terms) >= MAX_TERMS:
            break

    meaningful = [term for term in terms if term.lower() not in STOPWORDS]
    # Falling back to the full list matters: a question that is *entirely* stopwords is
    # still a question, and answering it badly beats raising on it.
    chosen = meaningful or terms
    if not chosen:
        raise EmptyQuery("the question contained no searchable words")

    joiner = " AND " if require_all else " OR "
    return joiner.join('"' + term.replace('"', "") + '"' for term in chosen)


#: Selected columns, in the order :func:`_row_to_hit` reads them.
_COLUMNS = (
    "chunk_id", "doc_id", "source_id", "doc_path", "doc_title",
    "ordinal", "start_line", "end_line", "heading_trail", "text",
)

SEARCH_SQL = """
SELECT {columns},
       snippet(chunks, 0, ?, ?, ?, ?) AS snippet,
       bm25(chunks) AS score
  FROM chunks
 WHERE chunks MATCH ?
   {scope}
 ORDER BY score
 LIMIT ?
"""


def search(
    connection: sqlite3.Connection,
    question: str,
    *,
    source_ids: Sequence[str],
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Search the corpus, restricted to *source_ids*.

    The restriction is a SQL predicate, not an instruction. An agent cannot widen its own
    scope by asking differently, because the scope never reaches the model's side of the
    boundary — it is applied here, to the query the model caused.

    An empty *source_ids* returns no results rather than every result. Silently treating
    "no corpora granted" as "all corpora" is the failure mode that turns a scoping bug into
    a disclosure incident.
    """
    scoped = [str(value) for value in source_ids if str(value).strip()]
    if not scoped:
        return []

    limit = max(1, min(int(limit), 25))
    placeholders = ", ".join("?" for _ in scoped)
    sql = SEARCH_SQL.format(
        columns=", ".join(_COLUMNS),
        scope=f"AND source_id IN ({placeholders})",
    )

    for require_all in (True, False):
        try:
            expression = build_match_expression(question, require_all=require_all)
        except EmptyQuery:
            return []
        parameters = [
            SNIPPET_OPEN, SNIPPET_CLOSE, SNIPPET_ELLIPSIS, SNIPPET_TOKENS,
            expression, *scoped, limit,
        ]
        rows = connection.execute(sql, parameters).fetchall()
        if rows:
            return [_row_to_hit(row) for row in rows]
        # Every term was required and something was missing. Widen once, then stop: a
        # third attempt would be a search for the least specific word in the question.
    return []


def _row_to_hit(row: Sequence[Any]) -> dict[str, Any]:
    values = dict(zip((*_COLUMNS, "snippet", "score"), row))
    try:
        trail = json.loads(values.get("heading_trail") or "[]")
    except (TypeError, ValueError):
        trail = []
    values["heading_trail"] = [str(item) for item in trail] if isinstance(trail, list) else []
    values["citation"] = format_citation(
        values.get("doc_path", ""),
        values.get("start_line") or 0,
        values.get("end_line") or 0,
        values["heading_trail"],
    )
    values["score"] = _relevance(values.get("score"))
    return values


def _relevance(raw: Any) -> float:
    """A bm25 result as a "higher is better" relevance number.

    Two things about it. The sign is flipped, because bm25 returns a negative number and
    every consumer downstream assumes a relevance score sorts the usual way.

    And it is kept to four *significant* figures rather than four decimal places. On a small
    corpus every term is near-universal, so idf approaches zero and real bm25 scores land
    around 1e-06 — rounding those to four decimals reports every hit as exactly 0.0 and
    throws the ranking away at the display layer while the SQL underneath ranked correctly.

    The number is only ever comparable *within one result set*. It is not a percentage, not
    a confidence, and not stable across corpora, so nothing should threshold on it.
    """
    try:
        value = -float(raw or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return float(f"{value:.4g}")


def format_citation(
    doc_path: str,
    start_line: int,
    end_line: int,
    heading_trail: Optional[Sequence[str]] = None,
) -> str:
    """``path:12-34 (Section › Subsection)`` — the one citation format, defined once."""
    where = f"{doc_path}:{start_line}" if start_line else str(doc_path)
    if start_line and end_line and end_line != start_line:
        where += f"-{end_line}"
    section = " › ".join(str(item) for item in (heading_trail or []) if item)
    return f"{where} ({section})" if section else where
