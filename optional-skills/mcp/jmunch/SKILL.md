---
name: jmunch
description: Token-efficient code, documentation, and tabular data retrieval using the jMunch MCP suite (jCodeMunch, jDocMunch, jDataMunch). Use when navigating unfamiliar codebases, searching documentation, profiling CSV/Excel data, or performing impact analysis before edits. Reduces token consumption by up to 37x compared to raw file reading.
version: 1.0.0
author: jgravelle
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

Restart Hermes. Tools will be registered with prefixes `mcp_jcodemunch_*`, `mcp_jdocmunch_*`, and `mcp_jdatamunch_*`.

## Core Workflow: Discover, Search, Retrieve

All three servers follow the same pattern from the jMRI (jMunch Retrieval Interface) specification:

1. **Discover** what's indexed: `list_repos` / `list_datasets`
2. **Search** for what you need: `search_symbols` / `search_sections` / `search_data`
3. **Retrieve** the precise result: `get_symbol_source` / `get_section` / `get_rows`

Never read entire files when an index exists. The whole point is minimal-token, precise retrieval.

---

## jCodeMunch — Code Intelligence

52 tools for navigating codebases in 70+ programming languages.

### Index a Project

Before searching, the project must be indexed:

```
# Index a local folder
index_folder(folder_path="/path/to/project")

# Index a GitHub repo
index_repo(repo_url="https://github.com/owner/repo")

# Check what's already indexed
list_repos()
```

### Find and Read Code

```
# Search for symbols by name or description
search_symbols(repo="owner/repo", query="authentication middleware")

# Get the exact source code of a symbol
get_symbol_source(repo="owner/repo", symbol_id="src/auth.py::AuthMiddleware")

# Get a file's structure without reading the whole file
get_file_outline(repo="owner/repo", file_path="src/auth.py")

# Browse the project tree
get_file_tree(repo="owner/repo")
```

### Understand Relationships

```
# Who imports this module?
find_importers(repo="owner/repo", file_path="src/auth.py")

# Find all references to a symbol
find_references(repo="owner/repo", symbol_id="src/auth.py::AuthMiddleware")

# Get the class inheritance hierarchy
get_class_hierarchy(repo="owner/repo", symbol_id="src/auth.py::AuthMiddleware")

# Trace call chains
get_call_hierarchy(repo="owner/repo", symbol_id="src/auth.py::authenticate")
```

### Before You Edit: Impact Analysis

Always check impact before modifying code:

```
# What breaks if I change this symbol?
get_blast_radius(repo="owner/repo", symbol_id="src/auth.py::AuthMiddleware")

# Is it safe to rename this?
check_rename_safe(repo="owner/repo", symbol_id="src/auth.py::AuthMiddleware", new_name="AuthHandler")

# Preview the full impact of a change
get_impact_preview(repo="owner/repo", symbol_id="src/auth.py::AuthMiddleware")
```

### Code Quality

```
# Find dead code
find_dead_code(repo="owner/repo")

# Get complexity metrics for a symbol
get_symbol_complexity(repo="owner/repo", symbol_id="src/auth.py::authenticate")

# Find hotspots (high churn + high complexity)
get_hotspots(repo="owner/repo")

# Check for dependency cycles
get_dependency_cycles(repo="owner/repo")

# Find untested symbols
get_untested_symbols(repo="owner/repo")
```

### Budgeted Context Assembly

When you need comprehensive context within a token budget:

```
# Get the most relevant context for a query, ranked and budgeted
get_ranked_context(repo="owner/repo", query="how does auth work", max_tokens=4000)

# Get a symbol with all its imports and dependencies as a bundle
get_context_bundle(repo="owner/repo", symbol_id="src/auth.py::AuthMiddleware")
```

---

## jDocMunch — Documentation Retrieval

13 tools for structured documentation search and retrieval.

### Index Documentation

```
# Index a local docs folder
index_local(folder_path="/path/to/docs")

# Index a GitHub repo's docs
doc_index_repo(repo_url="https://github.com/owner/repo")

# Check what's indexed
doc_list_repos()
```

### Search and Read

```
# Search for documentation sections
search_sections(repo="owner/repo", query="authentication setup")

# Get a specific section by ID (minimal tokens)
get_section(repo="owner/repo", section_id="docs/auth.md::setup")

# Get multiple related sections
get_sections(repo="owner/repo", section_ids=["docs/auth.md::setup", "docs/auth.md::configuration"])

# Get a section with surrounding context
get_section_context(repo="owner/repo", section_id="docs/auth.md::setup")
```

### Navigate Document Structure

```
# Table of contents
get_toc(repo="owner/repo")

# Full hierarchical TOC tree
get_toc_tree(repo="owner/repo")

# Outline of a specific document
get_document_outline(repo="owner/repo", file_path="docs/auth.md")
```

### Documentation Quality

```
# Find broken links
get_broken_links(repo="owner/repo")

# Check documentation coverage
get_doc_coverage(repo="owner/repo")
```

---

## jDataMunch — Tabular Data Analysis

18 tools for CSV, Excel, Parquet, and JSONL exploration.

### Index Data Files

```
# Index a local CSV, Excel, or Parquet file
index_local(file_path="/path/to/data.csv")

# Check what's indexed
list_datasets()
```

### Explore and Profile

```
# Dataset overview (row count, columns, types, size)
describe_dataset(dataset="data")

# Deep column profiling (cardinality, nulls, min/max, distribution)
describe_column(dataset="data", column="status")

# Sample rows to understand the data
sample_rows(dataset="data", n=10)
```

### Query and Analyze

```
# Filter rows with conditions
get_rows(dataset="data", where="status = 'active'", limit=20)

# Full-text search across all columns
search_data(dataset="data", query="error timeout")

# Server-side aggregations (COUNT, SUM, AVG, MIN, MAX, MEDIAN)
aggregate(dataset="data", group_by="category", metrics=["count", "avg:price"])

# Cross-dataset SQL JOINs
join_datasets(left="orders", right="customers", on="customer_id", join_type="inner")
```

### Data Quality

```
# Find null hotspots, low-cardinality columns, outlier risks
get_data_hotspots(dataset="data")

# Pearson correlations between numeric columns
get_correlations(dataset="data")

# Check for schema drift between dataset versions
get_schema_drift(dataset_a="data_v1", dataset_b="data_v2")
```

---

## Combined Workflow Example

**Task:** "Understand how the payment pipeline processes transactions"

```
# 1. Search code for payment-related symbols
search_symbols(repo="myapp", query="payment pipeline transaction")

# 2. Read the key function
get_symbol_source(repo="myapp", symbol_id="src/payments/processor.py::PaymentProcessor.process")

# 3. Check who calls it
get_call_hierarchy(repo="myapp", symbol_id="src/payments/processor.py::PaymentProcessor.process")

# 4. Find the documentation
search_sections(repo="myapp", query="payment processing architecture")

# 5. Read the relevant doc section
get_section(repo="myapp", section_id="docs/architecture.md::payment-flow")

# 6. Profile the transaction data
index_local(file_path="/data/transactions.csv")
describe_dataset(dataset="transactions")
aggregate(dataset="transactions", group_by="status", metrics=["count", "avg:amount"])
```

This gives you a complete picture: code structure, call flow, documentation context, and actual data characteristics -- all with minimal token usage.

## Token Savings

Every jMunch response includes a `_meta` block with `tokens_saved` -- the difference between naive file reading and indexed retrieval. Report this to the user when relevant:

> "Retrieved the `PaymentProcessor` class (342 tokens) instead of reading the full file (18,400 tokens). Saved 18,058 tokens (98% reduction)."

## Troubleshooting

### Tools not appearing after config change

Restart Hermes completely. MCP servers connect at startup.

### "repo not found" errors

The project needs to be indexed first. Run `list_repos()` to check what's indexed, then `index_folder()` or `index_repo()` to index.

### Stale results after code changes

Run `index_folder()` again to re-index. jMunch detects changed files and only re-indexes what's different.

## References

- **jCodeMunch:** [PyPI](https://pypi.org/project/jcodemunch-mcp/) | [GitHub](https://github.com/jgravelle/jcodemunch-mcp)
- **jDocMunch:** [PyPI](https://pypi.org/project/jdocmunch-mcp/) | [GitHub](https://github.com/jgravelle/jdocmunch-mcp)
- **jDataMunch:** [PyPI](https://pypi.org/project/jdatamunch-mcp/) | [GitHub](https://github.com/jgravelle/jdatamunch-mcp)
- **jMRI Spec:** [GitHub](https://github.com/jgravelle/mcp-retrieval-spec) (Apache 2.0)
