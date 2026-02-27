---
name: arxiv-citation-manager
description: High-precision ArXiv metadata extraction and BibTeX generation. Handles ArXiv API quirks, XML namespaces, and ID versioning (v1/v2).
version: 1.0.0
author: community
license: MIT
dependencies: [requests, lxml]
metadata:
  hermes:
    tags: [Research, ArXiv, BibTeX, Citations, API, XML]
---

# ArXiv Citation Manager

This skill encodes **operational knowledge about ArXiv’s API semantics and XML structure** that cannot be reliably inferred through prompting alone. It provides defensive engineering patterns for verifiable metadata extraction, deterministic BibTeX generation, and automated error recovery.

## Why This Skill Must Exist (Failure Mode Comparison)

| Scenario | Default Agent Failure | **ArXiv Citation Manager Guarantee** |
| :--- | :--- | :--- |
| **XML Parsing** | Loses XML namespaces → empty author list or summary. | **Namespace-aware parsing** for Atom and arXiv schemas. |
| **ID Versioning** | Drops version suffix (e.g., `v1`) or confuses it with main paper. | **Version-preserving canonical IDs** for specific branch citations. |
| **BibTeX Format** | Hallucinates keys or misformats author separation (`and`). | **Deterministic BibTeX generation** with verified field mapping. |
| **503 Errors** | Loops on failure or ignores `Retry-After` logic. | **Exponential backoff** for robust recovery during high load. |
| **Hallucination** | Invent IDs or metadata from training memory. | **Verifiable output** sourced directly from the live ArXiv Export API. |

## When to Use This Skill

- **Generating Citations**: When you need a perfect BibTeX entry for a paper.
- **Verifying IDs**: When a user provides an ArXiv ID and you need to confirm its title/authors.
- **Version Tracking**: When you need to know if a paper has multiple versions (e.g., `2401.00001v1` vs `2401.00001v2`).
- **Batch Metadata**: When fetching metadata for multiple papers simultaneously.

## Non-Obvious Knowledge (API Quirks)

### 1. XML Namespaces
The ArXiv API uses multiple XML namespaces (Atom, arXiv, opensearch). Simple regex or string parsing will fail on these. Always use an XML parser with namespace support.
- Atom: `http://www.w3.org/2005/Atom`
- ArXiv: `http://arxiv.org/schemas/atom`

### 2. ID Versioning
- `arxiv.org/abs/1706.03762` always points to the **latest** version.
- `arxiv.org/abs/1706.03762v1` points to a **specific** version.
- When generating BibTeX, it is best practice to cite the version you actually read.

### 3. Rate Limiting
ArXiv strictly enforces a **3-second delay** between API requests. The included script handles this, but manual web searches should also be paced.

## Workflow: Verifying a Paper

1. **Extract ID**: Identify the ArXiv ID from the user's request (e.g., `1706.03762`).
2. **Fetch Metadata**: Use `scripts/verify_arxiv.py` to get structured JSON/BibTeX.
3. **Cross-Check**: If the DOI is available in the metadata, use it to verify the publisher version via CrossRef.

## Usage Examples

- "Get the BibTeX for ArXiv paper 2303.08774."
- "What is the latest version of the 'Attention is All You Need' paper?"
- "Check if ArXiv ID 2402.12345 exists and give me the author list."

## Reference Files

- [api_quirks.md](references/api_quirks.md) - Deep dive into ArXiv API behavior.
- [verify_arxiv.py](scripts/verify_arxiv.py) - Robust Python utility for metadata.
