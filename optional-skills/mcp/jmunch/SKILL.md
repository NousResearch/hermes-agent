---
name: jmunch
description: Token-efficient code, docs, and tabular data retrieval.
version: 1.0.0
author: J. Gravelle (jgravelle)
license: MIT
metadata:
  hermes:
    tags: [MCP, Code Intelligence, Documentation, Data Analysis, Retrieval, AST, Token Efficiency]
    homepage: https://github.com/jgravelle/jcodemunch-mcp
    related_skills: [native-mcp]
prerequisites:
  commands: [uvx]
---

# jMunch MCP Suite

Token-efficient retrieval for code, documentation, and tabular data. The jMunch suite provides three MCP servers that index your sources and serve precise, minimal-token responses instead of dumping entire files into context.

**Benchmarks:** 37x fewer tokens, 19.6x fewer tool calls vs raw file reading. Over 12.4 billion tokens saved across all users.

## When to Use

Use this skill when the task involves:

- navigating or understanding an unfamiliar codebase
- searching for functions, classes, or symbols across a project
- reading documentation without pulling entire files into context
- profiling CSV, Excel, Parquet, or JSONL data files
- checking the impact or blast radius of a code change before editing
- understanding dependency graphs, class hierarchies, or call chains
- finding dead code, untested symbols, or architectural violations

Do **not** use this skill when:

- you need to edit a file (read the file directly for edits)
- the project has fewer than ~5 files (raw reads are fine for tiny projects)

## Prerequisites

Install the jMunch MCP servers. All three are optional -- install only what you need:

```bash
# Code intelligence (70+ languages via tree-sitter AST parsing)
pip install jcodemunch-mcp

# Documentation retrieval (Markdown, RST, AsciiDoc, Jupyter, HTML, OpenAPI)
pip install jdocmunch-mcp

# Tabular data (CSV, Excel, Parquet, JSONL)
pip install jdatamunch-mcp
```

Or use `uvx` to run without installing (configured in Hermes config below).

## Configuration

Add to `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  jcodemunch:
    command: "uvx"
    args: ["jcodemunch-mcp"]
  jdocmunch:
    command: "uvx"
    args: ["jdocmunch-mcp"]
  jdatamunch:
    command: "uvx"
    args: ["jdatamunch-mcp"]
```

Restart Hermes. Hermes registers each server's tools as `mcp__<server>__<tool>`, so the tools appear as `mcp__jcodemunch__*`, `mcp__jdocmunch__*`, and `mcp__jdatamunch__*`. Every example below uses that convention -- call the tools by their prefixed names.

## Core Workflow: Discover, Search, Retrieve

All three servers follow the same pattern from the jMRI (jMunch Retrieval Interface) specification:

1. **Discover** what's indexed: `mcp__jcodemunch__list_repos` / `mcp__jdatamunch__list_datasets`
2. **Search** for what you need: `mcp__jcodemunch__search_symbols` / `mcp__jdocmunch__search_sections` / `mcp__jdatamunch__search_data`
3. **Retrieve** the precise result: `mcp__jcodemunch__get_symbol_source` / `mcp__jdocmunch__get_section` / `mcp__jdatamunch__get_rows`

Never read entire files when an index exists. The whole point is minimal-token, precise retrieval.

---

## jCodeMunch — Code Intelligence

Tools for navigating codebases in 70+ programming languages, registered under the `mcp__jcodemunch__` prefix.

### Index a Project

Before searching, the project must be indexed:

```
# Index a local folder
mcp__jcodemunch__index_folder(folder_path="/path/to/project")

# Index a GitHub repo
mcp__jcodemunch__index_repo(repo_url="https://github.com/owner/repo")

# Check what's already indexed
mcp__jcodemunch__list_repos()
```

### Find and Read Code

```
# Search for symbols by name or description
mcp__jcodemunch__search_symbols(repo="owner/repo", query="authentication middleware")

# Get the exact source code of a symbol
mcp__jcodemunch__get_symbol_source(repo="owner/repo", symbol_id="src/auth.py::AuthMiddleware")

# Get a file's structure without reading the whole file
mcp__jcodemunch__get_file_outline(repo="owner/repo", file_path="src/auth.py")

# Browse the project tree
mcp__jcodemunch__get_file_tree(repo="owner/repo")
```

### Understand Relationships

```
# Who imports this module?
mcp__jcodemunch__find_importers(repo="owner/repo", file_path="src/auth.py")

# Find all references to a symbol
mcp__jcodemunch__find_references(repo="owner/repo", symbol_id="src/auth.py::AuthMiddleware")

# Get the class inheritance hierarchy
mcp__jcodemunch__get_class_hierarchy(repo="owner/repo", symbol_id="src/auth.py::AuthMiddleware")

# Trace call chains
mcp__jcodemunch__get_call_hierarchy(repo="owner/repo", symbol_id="src/auth.py::authenticate")
```

### Before You Edit: Impact Analysis

Always check impact before modifying code:

```
# What breaks if I change this symbol?
mcp__jcodemunch__get_blast_radius(repo="owner/repo", symbol_id="src/auth.py::AuthMiddleware")

# Is it safe to rename this?
mcp__jcodemunch__check_rename_safe(repo="owner/repo", symbol_id="src/auth.py::AuthMiddleware", new_name="AuthHandler")

# Preview the full impact of a change
mcp__jcodemunch__get_impact_preview(repo="owner/repo", symbol_id="src/auth.py::AuthMiddleware")
```

### Code Quality

```
# Find dead code
mcp__jcodemunch__find_dead_code(repo="owner/repo")

# Get complexity metrics for a symbol
mcp__jcodemunch__get_symbol_complexity(repo="owner/repo", symbol_id="src/auth.py::authenticate")

# Find hotspots (high churn + high complexity)
mcp__jcodemunch__get_hotspots(repo="owner/repo")

# Check for dependency cycles
mcp__jcodemunch__get_dependency_cycles(repo="owner/repo")

# Find untested symbols
mcp__jcodemunch__get_untested_symbols(repo="owner/repo")
```

### Budgeted Context Assembly

When you need comprehensive context within a token budget:

```
# Get the most relevant context for a query, ranked and budgeted
mcp__jcodemunch__get_ranked_context(repo="owner/repo", query="how does auth work", max_tokens=4000)

# Get a symbol with all its imports and dependencies as a bundle
mcp__jcodemunch__get_context_bundle(repo="owner/repo", symbol_id="src/auth.py::AuthMiddleware")
```

---

## jDocMunch — Documentation Retrieval

Tools for structured documentation search and retrieval, registered under the `mcp__jdocmunch__` prefix.

### Index Documentation

```
# Index a local docs folder
mcp__jdocmunch__index_local(folder_path="/path/to/docs")

# Index a GitHub repo's docs
mcp__jdocmunch__doc_index_repo(repo_url="https://github.com/owner/repo")

# Check what's indexed
mcp__jdocmunch__doc_list_repos()
```

### Search and Read

```
# Search for documentation sections
mcp__jdocmunch__search_sections(repo="owner/repo", query="authentication setup")

# Get a specific section by ID (minimal tokens)
mcp__jdocmunch__get_section(repo="owner/repo", section_id="docs/auth.md::setup")

# Get multiple related sections
mcp__jdocmunch__get_sections(repo="owner/repo", section_ids=["docs/auth.md::setup", "docs/auth.md::configuration"])

# Get a section with surrounding context
mcp__jdocmunch__get_section_context(repo="owner/repo", section_id="docs/auth.md::setup")
```

### Navigate Document Structure

```
# Table of contents
mcp__jdocmunch__get_toc(repo="owner/repo")

# Full hierarchical TOC tree
mcp__jdocmunch__get_toc_tree(repo="owner/repo")

# Outline of a specific document
mcp__jdocmunch__get_document_outline(repo="owner/repo", file_path="docs/auth.md")
```

### Documentation Quality

```
# Find broken links
mcp__jdocmunch__get_broken_links(repo="owner/repo")

# Check documentation coverage
mcp__jdocmunch__get_doc_coverage(repo="owner/repo")
```

---

## jDataMunch — Tabular Data Analysis

Tools for CSV, Excel, Parquet, and JSONL exploration, registered under the `mcp__jdatamunch__` prefix.

### Index Data Files

```
# Index a local CSV, Excel, or Parquet file
mcp__jdatamunch__index_local(file_path="/path/to/data.csv")

# Check what's indexed
mcp__jdatamunch__list_datasets()
```

### Explore and Profile

```
# Dataset overview (row count, columns, types, size)
mcp__jdatamunch__describe_dataset(dataset="data")

# Deep column profiling (cardinality, nulls, min/max, distribution)
mcp__jdatamunch__describe_column(dataset="data", column="status")

# Sample rows to understand the data
mcp__jdatamunch__sample_rows(dataset="data", n=10)
```

### Query and Analyze

```
# Filter rows with conditions
mcp__jdatamunch__get_rows(dataset="data", where="status = 'active'", limit=20)

# Full-text search across all columns
mcp__jdatamunch__search_data(dataset="data", query="error timeout")

# Server-side aggregations (COUNT, SUM, AVG, MIN, MAX, MEDIAN)
mcp__jdatamunch__aggregate(dataset="data", group_by="category", metrics=["count", "avg:price"])

# Cross-dataset SQL JOINs
mcp__jdatamunch__join_datasets(left="orders", right="customers", on="customer_id", join_type="inner")
```

### Data Quality

```
# Find null hotspots, low-cardinality columns, outlier risks
mcp__jdatamunch__get_data_hotspots(dataset="data")

# Pearson correlations between numeric columns
mcp__jdatamunch__get_correlations(dataset="data")

# Check for schema drift between dataset versions
mcp__jdatamunch__get_schema_drift(dataset_a="data_v1", dataset_b="data_v2")
```

---

## Combined Workflow Example

**Task:** "Understand how the payment pipeline processes transactions"

```
# 1. Search code for payment-related symbols
mcp__jcodemunch__search_symbols(repo="myapp", query="payment pipeline transaction")

# 2. Read the key function
mcp__jcodemunch__get_symbol_source(repo="myapp", symbol_id="src/payments/processor.py::PaymentProcessor.process")

# 3. Check who calls it
mcp__jcodemunch__get_call_hierarchy(repo="myapp", symbol_id="src/payments/processor.py::PaymentProcessor.process")

# 4. Find the documentation
mcp__jdocmunch__search_sections(repo="myapp", query="payment processing architecture")

# 5. Read the relevant doc section
mcp__jdocmunch__get_section(repo="myapp", section_id="docs/architecture.md::payment-flow")

# 6. Profile the transaction data
mcp__jdatamunch__index_local(file_path="/data/transactions.csv")
mcp__jdatamunch__describe_dataset(dataset="transactions")
mcp__jdatamunch__aggregate(dataset="transactions", group_by="status", metrics=["count", "avg:amount"])
```

This gives you a complete picture: code structure, call flow, documentation context, and actual data characteristics -- all with minimal token usage.

## Token Savings

Every jMunch response includes a `_meta` block with `tokens_saved` -- the difference between naive file reading and indexed retrieval. Report this to the user when relevant:

> "Retrieved the `PaymentProcessor` class (342 tokens) instead of reading the full file (18,400 tokens). Saved 18,058 tokens (98% reduction)."

## Troubleshooting

### Tools not appearing after config change

Restart Hermes completely. MCP servers connect at startup.

### "repo not found" errors

The project needs to be indexed first. Run `mcp__jcodemunch__list_repos()` to check what's indexed, then `mcp__jcodemunch__index_folder()` or `mcp__jcodemunch__index_repo()` to index.

### Stale results after code changes

Run `mcp__jcodemunch__index_folder()` again to re-index. jMunch detects changed files and only re-indexes what's different.

## References

- **jCodeMunch:** [PyPI](https://pypi.org/project/jcodemunch-mcp/) | [GitHub](https://github.com/jgravelle/jcodemunch-mcp)
- **jDocMunch:** [PyPI](https://pypi.org/project/jdocmunch-mcp/) | [GitHub](https://github.com/jgravelle/jdocmunch-mcp)
- **jDataMunch:** [PyPI](https://pypi.org/project/jdatamunch-mcp/) | [GitHub](https://github.com/jgravelle/jdatamunch-mcp)
- **jMRI Spec:** [GitHub](https://github.com/jgravelle/mcp-retrieval-spec) (Apache 2.0)
