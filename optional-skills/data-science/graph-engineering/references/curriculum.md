# Source Course — Translated Curriculum

Southeast University graduate course《知识图谱》(Knowledge Graphs), Prof. Peng Wang.
Repo: https://github.com/npubird/KnowledgeGraphCourse (4.4K★, running since 2019, updated yearly).
All slide decks are Chinese PDFs in the repo root; 2025 decks are prefixed `2025-pub-`.
This file is the English map of the 2025 edition.

## Lecture 1 — Knowledge Graphs: Theory, Technology, Practice, Challenges
*(2025-pub-1 …A.pdf / …B.pdf)*

- 1.1 The KG view of cognitive intelligence: what "knowledge" adds to machine cognition;
  the essence of a knowledge graph; how KGs evolved (semantic networks → expert systems →
  Semantic Web → Google's 2012 Knowledge Graph); KG vs deep learning; KG vs traditional
  knowledge bases vs databases; application scenarios; core value.
- 1.2 The technology stack, which the rest of the course expands:
  **knowledge extraction → knowledge fusion → representation learning → reasoning → storage.**
- 1.3 Bottlenecks: knowledge acquisition (coverage/cost), knowledge quality
  (noise/consistency), and intelligent application (making the graph actually useful).

## Lecture 2 — Knowledge Representation (2025-pub-2)

Concepts, then the method inventory: semantic networks, production (rule) systems, frame
systems, conceptual graphs, formal concept analysis, description logic, ontologies, ontology
languages (RDF/RDFS/OWL), and KG representation learning (embeddings).
→ Distilled in [modeling.md](modeling.md).

## Lecture 3 — Knowledge Modeling (2025-pub-3)

Ontology deep-dive: ontology engineering methodology, ontology learning (semi-automatic schema
induction), modeling tools (Protégé et al.), and a hands-on modeling practicum.
→ Distilled in [modeling.md](modeling.md).

## Lecture 4 — Knowledge Extraction: Problems & Methods (2025-pub-4)

Problem analysis (scenarios, why extraction is hard) and methods by source type:
structured data (D2R mapping), semi-structured data (wrappers, web tables), and
unstructured text (the NLP pipeline the next three lectures cover).
→ Distilled in [extraction.md](extraction.md).

## Lecture 5 — Entity Recognition (2025-pub-5, plus 2025 frontier deck 5-1)

The full method ladder: rule/dictionary-based → classical ML (HMM/CRF) → deep learning
(BiLSTM-CRF) → semi-supervised → transfer learning → pretrained models (BERT-era) →
LLM-era paradigm. Frontier-progress deck updated yearly.
→ Distilled in [extraction.md](extraction.md).

## Lecture 6 — Relation Extraction (2025-pub-6)

Semantic relations, feature design, benchmark datasets; methods: template/pattern-based,
supervised, weakly supervised, distant supervision, unsupervised (open IE), and
deep/reinforcement-learning approaches.
→ Distilled in [extraction.md](extraction.md).

## Lecture 7 — Event Extraction (2024-pub-7; includes Huawei industry lecture "From Classic
to LLM Paradigms")

Event concepts (trigger, arguments, event types), extraction methods, a finance-domain event
extraction system walkthrough, and event-logic graphs (事理图谱) — graphs whose nodes are
events and edges are causal/temporal/conditional links.
→ Distilled in [extraction.md](extraction.md).

## Lecture 8 — Knowledge Fusion (2024-pub-8, plus frontier-progress deck)

Knowledge heterogeneity; ontology matching; match extraction and tuning; instance matching;
large-scale entity matching (blocking, scaling); real fusion case studies.
→ Distilled in [fusion-and-llm.md](fusion-and-llm.md).

## Lecture 9 — Knowledge Graphs × Large Language Models (2025-pub-9)

Both directions: **KG for LLM** (grounding, retrieval, hallucination reduction, structured
memory) and **LLM for KG** (LLM-powered extraction, schema induction, fusion). The 2024
edition also includes decks on ChatGPT for information extraction, prompt engineering, and
quality evaluation.
→ Distilled in [fusion-and-llm.md](fusion-and-llm.md).

## How the course maps to this skill's 9-stage pipeline

| Skill stage | Course source |
|---|---|
| 1 Scope & value | Lecture 1.1, 1.3 |
| 2 Representation choice | Lecture 2 |
| 3 Ontology modeling | Lecture 3 |
| 4 Entity extraction | Lectures 4-5 |
| 5 Relation extraction | Lecture 6 |
| 6 Event extraction | Lecture 7 |
| 7 Quality gate | Lecture 1.3 + evaluation material in 9 |
| 8 Knowledge fusion | Lecture 8 |
| 9 Serve to LLMs | Lecture 9 |
