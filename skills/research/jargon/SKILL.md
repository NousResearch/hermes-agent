---
name: jargon
description: "Detect, decode, and track domain-specific jargon (acronyms, technical terms) with a multi-level plainspeak registry."
version: 1.0.0
author: "Hermes Agent"
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Jargon, Registry, Acronyms, Terminology, Domain-Specific, Parsing, Content-Processing]
    related_skills: [youtube-content, structured-digest, hn-brief-digest, blogwatcher]
---

# Jargon Detection & Plainspeak Registry

A generic tool for detecting, decoding, and tracking domain-specific jargon
across any content stream. Uses a JSON registry keyed by term with plainspeak
translations at three sophistication levels, a **saturation filter** for
baseline terms, and per-digest deduplication.

## Concepts

### Registry

A JSON file structured as:

```json
{
  "LLM": {
    "term": "LLM",
    "expanded": "Large Language Model",
    "plainspeak": {
      "doctor": "A neural network with billions of parameters trained on text to generate coherent language and perform reasoning tasks.",
      "high-school": "A computer program that has read a huge amount of text and can write essays, answer questions, and have conversations.",
      "kindergarten": "A super smart computer that can talk and write like a person."
    },
    "category": "ai",
    "saturated": false,
    "since": "2023-01-01"
  }
}
```

Each entry has:

| Field | Description |
|-------|-------------|
| `term` | The acronym or jargon term (canonical casing) |
| `expanded` | The expanded/acronym-free form |
| `plainspeak` | Three translations at `doctor`, `high-school`, and `kindergarten` levels |
| `category` | Domain category (ai, mlops, infra, devops, crypto, bio, etc.) |
| `saturated` | Flag. When true, the term is considered common baseline knowledge and skipped during detection |
| `since` | ISO date when the term was added to the registry |

### Saturation

Mark a term as `"saturated": true` when it has become common knowledge for your
audience (e.g., "API" for a developer audience). Saturated terms are skipped
during scan — they don't trigger explanations and don't clutter output.

### Per-Digest Dedup

During a single digest run, each term is explained at most once. A runtime
set tracks which terms have already been expanded. This prevents the same
acronym being re-explained in every section.

## Workflow

```
load registry → scan content → detect unknown terms → explain + optionally update registry
```

### 1. Load the registry

```bash
# Load into a shell variable
JARGON_REGISTRY=$(cat /path/to/jargon-registry.json)
```

### 2. Scan content for known and unknown terms

Extract acronyms and jargon from text, then cross-reference against the
registry.

```bash
# Extract all-uppercase acronyms (3+ chars) from content
CONTENT="$1"
UNKNOWN_TERMS=$(echo "$CONTENT" | grep -oP '\b[A-Z]{2,}\b' | sort -u)
```

### 3. Filter known (explainable) vs unknown

```python3
import json, sys

with open("jargon-registry.json") as f:
    registry = json.load(f)

content = sys.stdin.read()
words = set(w for w in content.split() if w.isupper() and len(w) >= 2)

# Known = in registry AND not saturated
known = {t for t in words if t in registry and not registry[t].get("saturated", False)}
# Unknown = uppercase term NOT in registry
unknown = {t for t in words if t not in registry}

print("KNOWN:", json.dumps(sorted(known)))
print("UNKNOWN:", json.dumps(sorted(unknown)))
```

### 4. Explain with plainspeak

For each known term, get the proper plainspeak level and append to the output:

```python3
def plainspeak(term, registry, level="high-school"):
    entry = registry.get(term)
    if not entry or entry.get("saturated"):
        return None
    return entry["plainspeak"].get(level, entry["plainspeak"].get("high-school"))
```

### 5. Dedup per digest run

```python3
seen = set()

def explain_once(term, registry, level="high-school", seen=seen):
    if term in seen:
        return None
    seen.add(term)
    return plainspeak(term, registry, level)
```

### 6. Update the registry with new terms

When unknown jargon is detected, add it to the registry:

```python3
def add_term(registry, term, expanded, doctor="", high_school="", kindergarten="", category="general"):
    registry[term] = {
        "term": term,
        "expanded": expanded,
        "plainspeak": {
            "doctor": doctor,
            "high-school": high_school,
            "kindergarten": kindergarten
        },
        "category": category,
        "saturated": False,
        "since": "2025-01-01"
    }
```

Then write back:

```bash
python3 -c "
import json
with open('jargon-registry.json') as f:
    reg = json.load(f)
# ... modify ...
with open('jargon-registry.json', 'w') as f:
    json.dump(reg, f, indent=2)
"
```

## Complete Pipeline Example

This end-to-end pipeline scans a text file, detects known jargon, explains it,
and reports unknown terms:

```bash
#!/usr/bin/env bash
# jargon-pipeline.sh — scan content and explain jargon
set -euo pipefail

CONTENT_FILE="${1:?Usage: $0 <content-file>}"
REGISTRY="${2:-jargon-registry.json}"
LEVEL="${3:-high-school}"
EXPLAINED=$(mktemp)
trap 'rm -f "$EXPLAINED"' EXIT

python3 - "$CONTENT_FILE" "$REGISTRY" "$LEVEL" "$EXPLAINED" << 'PYEOF'
import json, sys, re

content_file, reg_file, level, explained_file = sys.argv[1:]

with open(reg_file) as f:
    registry = json.load(f)

with open(content_file) as f:
    content = f.read()

terms = set(re.findall(r'\b[A-Z]{2,}\b', content))
known = {t for t in terms if t in registry and not registry[t].get("saturated", False)}
unknown = {t for t in terms if t not in registry}

seen = set()
explanations = []

for t in sorted(known):
    if t not in seen:
        seen.add(t)
        ps = registry[t]["plainspeak"].get(level, registry[t]["plainspeak"].get("high-school"))
        expanded = registry[t].get("expanded", t)
        explanations.append(f"**{t}** ({expanded}): {ps}")

print("=== KNOWN TERMS ===")
for e in explanations:
    print(e)
    print()

if unknown:
    print("=== UNKNOWN TERMS (not in registry) ===")
    for t in sorted(unknown):
        print(f"- {t}")

with open(explained_file, 'w') as ef:
    ef.write("\n".join(explanations))
PYEOF
```

## Integration with Digest Skills

This skill pairs naturally with digest-producing skills such as
`hn-brief-digest`, `blogwatcher`, `youtube-content`, and
`structured-digest`. After generating a digest body, pipe it through
the pipeline above to produce a "Jargon Buster" appendix:

> **Jargon Buster**
> - **LLM** (Large Language Model): A computer program that has read
>   a huge amount of text and can write essays and answer questions.
> - **RAG** (Retrieval-Augmented Generation): A technique where the
>   AI looks up relevant documents before answering, like checking your
>   notes before a test.
> - **MCP** (Model Context Protocol): A standard way for AI models to
>   connect to external tools and data sources.

## Skills

- **detect**: Scan text for all uppercase/domain acronyms, cross-reference registry
- **explain**: Return plainspeak definition for a single term at a given level
- **saturate**: Mark a term as saturated so it's no longer flagged
- **unsaturate**: Unmark a term
- **register**: Add a new term to the registry interactively
- **sync**: Merge updates from a shared/multi-source registry
