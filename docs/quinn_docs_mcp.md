# Quinn Docs/Notes MCP

Repo-side implementation notes for the scoped Quinn Docs/Notes MCP server.

## Purpose

`quinn_docs` gives Quinn a read-first, proposal-only interface to selected Quinn/Hermes source-of-truth documents. It is designed for safer maintenance workflows where the agent can inspect approved docs, find relevant sections, and prepare text-only patch proposals without browsing arbitrary files or mutating anything.

## Files

- Server: `scripts/mcp/quinn_docs_server.py`
- Tests: `tests/test_quinn_docs_mcp.py`
- Plan: `docs/quinn_docs_mcp_plan.md`

## Security Boundaries

Version 1 is intentionally narrow:

- Exact document IDs only; raw paths are rejected.
- No arbitrary filesystem browsing.
- No private runtime files, transcript directories, logs, or environment files.
- No writes or mutations.
- Excerpts are bounded by line count and character budget.
- Returned text is redacted before it leaves the server.
- Search returns document IDs, titles, line numbers, and snippets only.
- Patch proposals are text templates only and are not applied.

## Document Registry

The server keeps an in-code `DOC_REGISTRY` mapping stable document IDs to exact paths. Public responses expose `doc_id` and `path_alias`, not absolute local paths.

Current registry includes:

- `quinn-hermes-server`
- `quinn-ops-mcp`
- `quinn-ops-snapshot-design`
- `quinn-github-mcp-plan`
- `quinn-observability-mcp-plan`
- `quinn-docs-mcp-plan`
- `quinn-approval-ops-mcp-plan`

## Tools

- `healthcheck()` — readiness, document counts, read-only flag.
- `list_documents()` — metadata for allowlisted docs.
- `get_document_outline(doc_id)` — Markdown headings with line numbers.
- `search_documents(query, limit=20)` — bounded literal search across allowlisted docs.
- `read_document_excerpt(doc_id, start_line=1, limit=80)` — bounded redacted excerpts.
- `get_document_summary(doc_id)` — metadata plus heading outline.
- `check_source_of_truth_freshness()` — missing-doc and required-heading warnings.
- `propose_document_patch(doc_id, change_request)` — approval-required patch template only.

## Non-goals

- Live installation.
- Gateway restart.
- Runtime MCP config edits.
- Editing documents directly.
- General file browsing.
- Private-value inspection.

## Live Promotion

Do not promote automatically. Live use requires explicit operator approval, then a separate controlled flow to copy the server into the live MCP path, edit MCP config, restart the gateway, and verify denied paths plus redacted excerpts.

## Verification

Repo-side verification:

```bash
python3 -m py_compile scripts/mcp/quinn_docs_server.py tests/test_quinn_docs_mcp.py
venv/bin/python -m pytest tests/test_quinn_docs_mcp.py -q
```
