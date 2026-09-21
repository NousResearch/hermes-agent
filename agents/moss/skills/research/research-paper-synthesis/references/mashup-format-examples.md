# Mashup Format Examples — Expanded with Remii Validation

Reference file for the expanded mashup format in research-paper-synthesis v2.1.0.

## Template

```
### Mashup: [Paper Short Name] + [GitRadar Repo] = [Build Idea Name]

**Idea:** [1-2 sentences. What is the thing being built? Name the output artifact.]

**How it works:** [2-4 sentences. What does the paper contribute? What does the repo
contribute? How do they integrate? Include technical details — API hooks, data formats,
plugin integration points. Must be specific enough to be falsifiable.]

**Why it matters:** [1-2 sentences. What does this unlock? Context savings, capability
improvement, time saved, competitive advantage for Sahil's stack.]

**Remii's Recommendation:**
- Validation: [Plausibility assessment — paper code status, repo health, technique
  maturity. What would need to be true for this to work?]
- Effort: [Small (<1 day) | Medium (1-3 days) | Large (1-2 weeks)]
- Risk: [Low | Medium | High]
- Action: [Read & Learn | Investigate | Prototype | Escalate | Defer]
- Next step: [Concrete first action — "check X repo for Y capability", "spike plugin
  hook for Z", "ask Octacon about W"]
```

## Full Example

```
### Mashup: LCLMs + Toolaria-Protocol = Latent-Context Blob Store

**Idea:** Replace Toolaria's raw-text blob storage with LCLM-compressed latent
embeddings. Oversized tool results are encoded at 1:16 compression ratio, stored as
vectors, and only decoded to text when specific segments are requested via
rescuer_fetch.

**How it works:** LCLMs provide an encoder-decoder compressor (0.6B encoder + 4B
decoder) that compresses arbitrary text to 1:4, 1:8, or 1:16 ratios. Toolaria's
blobstore.put() currently stores the raw SHA256-addressed text. A new `compress=true`
flag in the blob store config would route content through the LCLM encoder before
storage, storing embeddings instead of text. On retrieval (range/grep/search), the
LCLM decoder reconstructs only the relevant segment(s) from the embedding space,
not the full document. The semantic search index already operates on embeddings —
this just extends the encoding to the blob itself.

**Why it matters:** Reduces Toolaria's storage footprint by 16x for large papers
(currently ~500MB for 7 days of web_extract). Also reduces disk I/O during session
resume (embedding decode is faster than text re-read for sub-100KB blobs).

**Remii's Recommendation:**
- Validation: LCLM paper has no public code yet (pre-print 2026-06-08). 0 citations
  expected at this age. The technique is theoretically sound (encoder-decoder LM is
  well-understood) but needs a reference implementation. Toolaria's plugin hook for
  compress-then-store exists but is undocumented. Spike needed to confirm the hook
  is reachable.
- Effort: Large (1-2 weeks — requires running LCLM inference, which needs GPU or
  quantized CPU)
- Risk: Medium (technique is sound, but dependency on an unreleased model is risky)
- Action: Investigate — check if LCLM weights are released in the next 30 days.
  If released, promote to Prototype.
- Next step: Monitor LCLM repo for weight release. Check if a GGUF quantized version
  exists for CPU inference on this VPS.
```

## Example 2: Lower Confidence

```
### Mashup: AGENTS-K1 + Mnemosyne + GBrain = Agent-Native KG Bridge

**Idea:** Convert KENSEI's LLM wiki (interlinked markdown pages with wikilinks)
into an agent-consumable knowledge graph using AGENTS-K1's reasoning-graph
extraction pipeline, then bridge Mnemosyne's vector search with structured wiki
concepts via GBrain.

**How it works:** AGENTS-K1 introduces a reasoning-graph extraction algorithm that
parses agent decision traces into typed nodes (concept, relationship, decision) and
edges. Applying this to the wiki: parse each .md page's [[wikilinks]] as edges,
YAML frontmatter tags as node types, and entity descriptions as node properties.
The output feeds into Mnemosyne's graph traversal (mnemosyne_graph_query) and
GBrain's structured query layer. The bridge lets semantic AND graph queries operate
on the same knowledge surface — "find all concepts related to agent memory WITH
code implementation AND published in 2026."

**Why it matters:** Current wiki is text-only. A graph index on top unlocks compound
queries (semantic + relational + temporal) that neither vector search nor keyword
search can do alone.

**Remii's Recommendation:**
- Validation: AGENTS-K1 paper has no public code. The extraction algorithm is
  described at a high level — enough for a spike but not a production build.
  Mnemosyne already has graph query. GBrain is integrated. The weakest link is
  AGENTS-K1's extraction pipeline.
- Effort: Large (1-2 weeks — building the extraction pipeline from paper description)
- Risk: High (algorithm is underspecified, no reference impl)
- Action: Read & Learn — file to wiki as concept reference. Revisit if AGENTS-K1
  releases code. Do NOT create a kanban task.
- Next step: Reference the paper in existing wiki concept page on knowledge graphs.
```
