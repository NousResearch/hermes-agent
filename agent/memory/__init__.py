"""Memory context relevance framework.

Multi-stage pipeline for relevance-scored memory injection:

  User Message → Intent Extraction → Multi-Provider Recall → Cross-Provider Fusion
      → Relevance Gating → Structured Injection

Phase 1 exposes ScoredMemory, IntentContext, and scoring primitives.
"""
