#!/usr/bin/env python3
"""/findout — craft a bulletproof instruction that makes the agent run the
evidence-prediction verification pipeline via execute_code.

Unlike loading the findout SKILL.md (which the agent may treat as advisory
context), this builds a hard direct instruction that the agent queues as its
next turn. The agent processes it as a command: "import and run SelfVerifyPipeline,
then return the answer." Every step is spelled out explicitly so the agent
does not fall back to answering from its own knowledge.

Usage in CLI::

    /findout how do black holes work

Usage in TUI / gateway — the dispatcher calls ``build_findout_prompt(query)``
and queues the result onto ``_pending_input`` / rewrites ``event.text``::

    event.text = build_findout_prompt(user_query)
    # fall through to agent processing
"""


def build_findout_prompt(query: str) -> str:
    """Build a self-contained instruction that runs the findout pipeline.

    Args:
        query: The user's research / verification question.

    Returns:
        A complete instruction the agent executes as a normal turn. The agent
        uses execute_code to import and run SelfVerifyPipeline, or falls back
        to a manual 5-pass verification workflow if the package is unavailable.
    """
    q = (query or "").strip()
    if not q:
        q = "<no query provided — ask the user what to research>"

    return f"""[/findout] Run the evidence-prediction verification pipeline on this query.

QUERY: {q}

You MUST execute this pipeline programmatically using execute_code. Do NOT answer from your own knowledge. Do NOT describe what you would do — do it.

## STEP 1 — Try the findout package (preferred path)

```python
from findout.config import Config, LLMConfig, SearchConfig, PipelineConfig
from findout.pipeline import SelfVerifyPipeline
from findout.gate import Gate, GateConfig
import os, json

# Configure from env vars with sensible defaults
config = Config(
    llm=LLMConfig(
        model=os.getenv("FINDOUT_MODEL", "qwen3.5:14b"),
        base_url=os.getenv("FINDOUT_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.getenv("FINDOUT_API_KEY", "ollama"),
        max_tokens=4096,
        timeout_seconds=120,
    ),
    search=SearchConfig(
        provider=os.getenv("FINDOUT_SEARCH_PROVIDER", "duckduckgo"),
        max_results_per_query=5,
        max_queries_per_claim=3,
    ),
    pipeline=PipelineConfig(
        default_variant="hybrid",
        gate_enabled=True,
        hybrid_samples=2,
        max_claims_per_answer=12,
        short_circuit_on_agreement=True,
    ),
)

# Gate check first
gate = Gate(GateConfig(), config.llm)
category, reason = gate.classify_with_reason("{q}")
print(f"Gate: {{category}} ({{reason}})")

# Choose variant based on model size from context
pipeline_variant = "hybrid"  # safe default for 3B-14B

# Run pipeline
pipe = SelfVerifyPipeline(config)
result = pipe.run(query="{q}", pipeline=pipeline_variant)

# Print results
print("=" * 60)
print("PIPELINE RESULTS")
print("=" * 60)
print(f"Claims: {{result.total_claims}} total, {{result.verified_claims}} verified, {{result.uncertain_claims}} uncertain, {{result.contradicted_claims}} contradicted")
print(f"Searches: {{result.total_searches}}")
print()
print(result.answer)
```

## STEP 2 — If findout package import fails

Catch ImportError / ModuleNotFoundError and fall back to this manual workflow:

1. **Extract claims** — break the query into atomic factual claims
2. **Web search** — run 3 web searches per claim from different angles
3. **Cross-check** — identify which claims are supported, contradicted, or uncertain
4. **Rewrite** — produce a verified answer with `[verified]` / `[contradicted]` / `[uncertain]` markers
5. **Return** — present the answer with verdict summary

## STEP 3 — If LLM endpoint is unreachable

Catch ConnectionError and network errors. Fall back to a normal web-search-then-
answer workflow using the tools you already have (web_search + your own reasoning).
Tell the user: "Verification pipeline unavailable (model server not running at
${{FINDOUT_BASE_URL:-http://localhost:11434/v1}}). Answered via direct research."

## Output format

Include in your final response:
- The verified answer
- A verdict line: "X claims verified, Y uncertain, Z contradicted"
- Source citations when available
- Which pipeline path was used (findout package / manual fallback / direct research)
"""
