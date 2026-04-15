# jMunch Context File Template for Hermes Agent

Copy this content into a `.hermes.md` file at the root of any project to teach Hermes
how to use jMunch tools for that project.

---

## Retrieval Policy

Use jMunch MCP tools for all code, documentation, and data retrieval. Do not read raw
files when an index exists. The jMunch tools return precise, minimal-token responses
that save context window space.

## Tool Routing

| Intent | Tool | Follow-up |
|--------|------|-----------|
| Find a function or class | `search_symbols` | `get_symbol_source` |
| Understand a file's structure | `get_file_outline` | — |
| Browse project layout | `get_file_tree` | — |
| Check impact before editing | `get_blast_radius` | `check_rename_safe` |
| Find who calls a function | `get_call_hierarchy` | — |
| Find documentation | `search_sections` | `get_section` |
| Navigate doc structure | `get_toc_tree` | `get_document_outline` |
| Profile a data file | `describe_dataset` | `describe_column` |
| Query data rows | `search_data` or `get_rows` | `aggregate` |

## Indexed Sources

<!-- Update these after indexing your project -->
- Code: `index_folder(folder_path=".")` — indexed via jCodeMunch
- Docs: `index_local(folder_path="./docs")` — indexed via jDocMunch
- Data: `index_local(file_path="./data/records.csv")` — indexed via jDataMunch

## Efficiency

Report `_meta.tokens_saved` from jMunch responses to show retrieval efficiency.
