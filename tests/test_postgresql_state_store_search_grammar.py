from __future__ import annotations

import pytest

from state_store_postgresql_search import PostgreSQLSearchQueryError, compile_postgresql_search_expression


@pytest.mark.parametrize(
    ("query", "sql_parts", "params"),
    [
        ("alpha beta", ("&", "plainto_tsquery"), ("alpha", "beta")),
        ('"alpha beta"', ("phraseto_tsquery",), ("alpha beta",)),
        ("alph*", ("to_tsquery",), ("alph:*",)),
        ("alpha OR beta NOT gamma", ("|", "!"), ("alpha", "beta", "gamma")),
    ],
)
def test_compiles_only_static_tsquery_sql(query, sql_parts, params):
    expression = compile_postgresql_search_expression(query)

    assert not expression.is_cjk_literal
    assert expression.params == params
    assert all(part in expression.sql for part in sql_parts)
    assert query not in expression.sql
    assert "%s" in expression.sql


@pytest.mark.parametrize(
    "query",
    [
        "NOT alpha", "alpha AND", "alpha OR OR beta", "(alpha", "alpha)",
        '"alpha', '"alpha beta"*', "alpha:beta", "alpha NEAR beta", "alpha; DROP", "*alpha",
        "中文 AND memory", '"中文"', "中文*", "中文:memory",
    ],
)
def test_rejects_unsupported_or_ambiguous_fts5_syntax(query):
    with pytest.raises(PostgreSQLSearchQueryError, match="unsupported or ambiguous"):
        compile_postgresql_search_expression(query)


def test_cjk_literal_fallback_is_explicit_and_parameterized():
    expression = compile_postgresql_search_expression("中文记忆 断裂")

    assert expression.is_cjk_literal
    assert expression.sql == ""
    assert expression.params == ("中文记忆", "断裂")
