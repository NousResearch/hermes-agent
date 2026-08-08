"""Display formatters for the Hermes CLI — markdown table helpers.

Lazy re-exports of the pure table utilities in ``agent.markdown_tables``,
extracted verbatim from cli.py (wave 1 godfile extraction, shard s1 cluster
c2). The wrappers keep ``agent.markdown_tables`` out of cli.py's module top
so the agent import chain stays cold until a table actually needs it.
"""

def is_table_divider(*args, **kwargs):
    from agent.markdown_tables import is_table_divider as _is_table_divider

    return _is_table_divider(*args, **kwargs)


def looks_like_table_row(*args, **kwargs):
    from agent.markdown_tables import looks_like_table_row as _looks_like_table_row

    return _looks_like_table_row(*args, **kwargs)


def realign_markdown_tables(*args, **kwargs):
    from agent.markdown_tables import realign_markdown_tables as _realign_markdown_tables

    return _realign_markdown_tables(*args, **kwargs)
