# jMunch Context File Template for Hermes Agent

Copy this content into a `.hermes.md` file at the root of any project to teach Hermes
how to use jMunch tools for that project.

Hermes registers MCP tools as `mcp__<server>__<tool>`, so jMunch tools appear under the
`mcp__jcodemunch__`, `mcp__jdocmunch__`, and `mcp__jdatamunch__` prefixes. The routing
table below uses those prefixed names.

---

## Retrieval Policy

Use jMunch MCP tools for all code, documentation, and data retrieval. Do not read raw
files when an index exists. The jMunch tools return precise, minimal-token responses
that save context window space.

## Tool Routing

| Intent | Tool | Follow-up |
|--------|------|-----------|
| Find a function or class | `mcp__jcodemunch__search_symbols` | `mcp__jcodemunch__get_symbol_source` |
| Understand a file's structure | `mcp__jcodemunch__get_file_outline` | — |
| Browse project layout | `mcp__jcodemunch__get_file_tree` | — |
| Check impact before editing | `mcp__jcodemunch__get_blast_radius` | `mcp__jcodemunch__check_rename_safe` |
| Find who calls a function | `mcp__jcodemunch__get_call_hierarchy` | — |
| Find documentation | `mcp__jdocmunch__search_sections` | `mcp__jdocmunch__get_section` |
| Navigate doc structure | `mcp__jdocmunch__get_toc_tree` | `mcp__jdocmunch__get_document_outline` |
| Profile a data file | `mcp__jdatamunch__describe_dataset` | `mcp__jdatamunch__describe_column` |
| Query data rows | `mcp__jdatamunch__search_data` or `mcp__jdatamunch__get_rows` | `mcp__jdatamunch__aggregate` |

## Indexed Sources

<!-- Update these after indexing your project -->
- Code: `mcp__jcodemunch__index_folder(folder_path=".")` — indexed via jCodeMunch
- Docs: `mcp__jdocmunch__index_local(folder_path="./docs")` — indexed via jDocMunch
- Data: `mcp__jdatamunch__index_local(file_path="./data/records.csv")` — indexed via jDataMunch

## Efficiency

Report `_meta.tokens_saved` from jMunch responses to show retrieval efficiency.
