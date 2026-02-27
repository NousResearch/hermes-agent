# ArXiv API Quirks & Technical Deep Dive

The ArXiv Export API is powerful but has several "gotchas" that can lead to bugs in agentic workflows. This document outlines those quirks and how to handle them.

## 1. The XML Namespace Problem

ArXiv uses a mix of standard Atom and custom ArXiv XML namespaces. A common mistake is using generic XML parsers without namespace awareness, which results in `None` when searching for tags.

**Correct Handling (Python):**
```python
NAMESPACES = {
    'atom': 'http://www.w3.org/2005/Atom',
    'arxiv': 'http://arxiv.org/schemas/atom'
}
# Find the title using the namespace map
title = entry.find('atom:title', NAMESPACES)
```

## 2. Rate Limiting and 503 Handling (Defensive Engineering)

ArXiv strictly requests that users **not make more than one request every 3 seconds**. Failure to respect this leads to `503 Service Unavailable` errors.

**Defensive Implementation:**
The included `verify_arxiv.py` script implements **Exponential Backoff**:
- **Attempt 1**: Initial 4s delay on 503.
- **Attempt 2**: 5s delay on 503.
- **Attempt 3**: 7s delay on 503.
This ensures the agent doesn't enter a "tight loop" of failure, which is a common pitfall in naive agentic integrations.

## 3. ID Versioning and Canonicalization

ArXiv IDs are immutable *per version*, but the base ID is a moving target. 
- **Canonical ID**: `1706.03762` -> Always the latest.
- **Versioned ID**: `1706.03762v1` -> Immutable snapshot.
**Skill Guarantee**: The skill preserves version suffixes during metadata extraction to ensure that citations point to the exact document the agent processed, preventing "citation drift" where a later version might invalidate the agent's summary of an earlier one.

## 4. Withdrawn Papers

When a paper is withdrawn, the summary field in the API often contains a "Withdrawn" notice.
- **Bug**: Some parsers might treat the paper as "active" but find empty metadata fields.
- **Detection**: Check the `arxiv:comment` or `atom:summary` for strings like "withdrawn" or "retracted".

## 5. Submitting for Multi-ID Fetching

Instead of calling the API 10 times for 10 papers:
```
http://export.arxiv.org/api/query?id_list=ID1,ID2,ID3...
```
This is significantly more efficient and respects the rate limit.

## 6. Opensearch Metadata

The API response includes `opensearch:totalResults`. In a search-based query (rather than an ID-based one), this field is critical for pagination.

```xml
<opensearch:totalResults>1240</opensearch:totalResults>
<opensearch:startIndex>0</opensearch:startIndex>
<opensearch:itemsPerPage>10</opensearch:itemsPerPage>
```
Always check `totalResults` to decide if more pages need to be fetched.
