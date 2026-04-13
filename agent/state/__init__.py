"""W5.A / F-012: scoped split of `hermes_state.py` (3662 LOC).

A full big-bang split of every method on `SessionDB` into per-domain
files is high-risk against a class with shared state and active feature
work in adjacent areas. This package starts a measured incremental
split — the obsidian-sync cluster (~440 LOC, low coupling) is the first
extraction, providing the pattern for future domain extractions
(routing, cost, search, knowledge core).

Public-API contract: every name that callers used to import from
`hermes_state` must continue to import from there. The shim at
`hermes_state.py` re-exports `SessionDB` (now composed via mixins from
this package) so existing call sites are byte-identical.

Future domains to extract incrementally (in priority order):
- search.py    : search_messages / search_sessions / _sanitize_fts5_query
- routing.py   : log_routing_decision / log_implicit_signal / cron stats
- knowledge.py : save_knowledge_* / search_knowledge / list_knowledge
- cost.py      : update_token_counts / set_token_counts / token+cost log
- sessions.py  : session/message CRUD core (largest, do last)
"""
from agent.state.obsidian import _ObsidianMixin

__all__ = ["_ObsidianMixin"]
