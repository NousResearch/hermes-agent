---
name: knowledge-graph
description: Build and query a personal knowledge graph. Record typed triples (subject -> predicate -> object), detect knowledge clusters, surface related prior thinking, and generate bounded AI context from curated relationships.
---

# Knowledge Graph Skill

A structured memory layer that records knowledge as **typed relationships** between
entities, then lets you and your AI reason over those relationships.

Core insight: a note says *"A is related to B"*. A triple says
*"A **depends_on** B"* — and that precision is what lets AI follow the chain.

## Quick start

```bash
# Record your first triple
python skills/knowledge-graph/kg_tool.py add \
  --subject "LAB Safe Intake" --subject-type workflow \
  --predicate "redacts" \
  --object "PII" --object-type risk

# See the full graph
python skills/knowledge-graph/kg_tool.py stats

# Get a bounded AI context block for any entity
python skills/knowledge-graph/kg_tool.py context --node "workflow:lab_safe_intake"
```

## Storage

The graph is stored as a JSON file. Default path:

```bash
export KG_PATH="${HOME}/.hermes/knowledge_graph.json"
```

Override for a project-specific graph:

```bash
export KG_PATH="/path/to/project/graph.json"
```

## Recording triples

```bash
# Basic triple
python skills/knowledge-graph/kg_tool.py add \
  --subject "AI Governance" --subject-type policy \
  --predicate "requires" \
  --object "Audit Trail" --object-type workflow

# With provenance (source + confidence)
python skills/knowledge-graph/kg_tool.py add \
  --subject "Audit Trail" --subject-type workflow \
  --predicate "produces" \
  --object "Compliance Report" --object-type document \
  --source "governance_spec_v2.md" \
  --confidence 0.95

# With a summary on the subject node
python skills/knowledge-graph/kg_tool.py add \
  --subject "Document Intelligence" --subject-type tool \
  --subject-summary "Extracts structured data from unstructured documents." \
  --predicate "supports" \
  --object "LAB Safe Intake" --object-type workflow
```

### Valid entity types

`person` `project` `document` `workflow` `claim` `risk` `tool` `policy`
`repository` `topic` `organization`

### Common predicates (use any verb phrase)

| Predicate | Meaning |
|---|---|
| `depends_on` | B must exist for A to work |
| `supports` | A is evidence for B |
| `requires` | A mandates B |
| `produces` | A creates B |
| `redacts` | A removes/hides B |
| `references` | A cites B |
| `contradicts` | A conflicts with B |
| `belongs_to` | A is part of B |
| `leads_to` | A causes B |
| `flagged_with` | A carries risk B |

## Querying

```bash
# Bounded AI context block (for prompt injection)
python skills/knowledge-graph/kg_tool.py context --node "workflow:lab_safe_intake"
python skills/knowledge-graph/kg_tool.py context --node "policy:ai_governance" --hops 2

# Find all clusters (connected knowledge components)
python skills/knowledge-graph/kg_tool.py clusters

# Detect recurring relationship patterns
python skills/knowledge-graph/kg_tool.py patterns

# List all entities of a type
python skills/knowledge-graph/kg_tool.py list --type workflow
python skills/knowledge-graph/kg_tool.py list --type risk

# Shortest path between two entities
python skills/knowledge-graph/kg_tool.py path \
  --from "policy:ai_governance" \
  --to "document:compliance_report"

# Graph statistics
python skills/knowledge-graph/kg_tool.py stats

# Export full graph to JSON
python skills/knowledge-graph/kg_tool.py export
```

## Using context in an AI prompt

```bash
# Capture the context block
CONTEXT=$(python skills/knowledge-graph/kg_tool.py context \
  --node "workflow:lab_safe_intake" --hops 2)

# Include it in a prompt
echo "Use this knowledge graph context:\n$CONTEXT\n\nQuestion: What risks apply to the intake workflow?"
```

## Example: LAB AI knowledge chain

```bash
KG="python skills/knowledge-graph/kg_tool.py"

$KG add --subject "LAB Safe Intake" --subject-type workflow \
         --predicate "redacts" --object "PII" --object-type risk

$KG add --subject "LAB Safe Intake" --subject-type workflow \
         --predicate "requires" --object "Human Review" --object-type workflow

$KG add --subject "Human Review" --subject-type workflow \
         --predicate "precedes" --object "RAG Ingestion" --object-type workflow

$KG add --subject "PII" --subject-type risk \
         --predicate "requires" --object "Audit Log" --object-type document

# Now ask: what does the intake workflow touch?
$KG context --node "workflow:lab_safe_intake" --hops 2
```

Output:
```
Entity: LAB Safe Intake [workflow]
Summary: (no summary)
Relationships:
  LAB Safe Intake -> redacts -> PII
  LAB Safe Intake -> requires -> Human Review
Nearby entities:
  - PII [risk]
  - Human Review [workflow]
  - RAG Ingestion [workflow]
  - Audit Log [document]
```

This context block can be injected directly into any LLM call to ground the
response in your actual documented knowledge rather than general training data.
