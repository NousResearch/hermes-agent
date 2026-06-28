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

You MUST execute this pipeline programmatically using execute_code. Do NOT answer from your own knowledge. Do NOT describe what you would do — do it. Import the pipeline, run it, and return the result.

## STEP 1 — Try the findout package (preferred path)

Copy-paste this EXACT code into execute_code:

```python
import os, sys, json, requests, textwrap

# === CONFIG ===
BASE_URL = os.getenv("FINDOUT_BASE_URL", "http://localhost:11434/v1")
API_KEY = os.getenv("FINDOUT_API_KEY", "ollama")
MODEL = os.getenv("FINDOUT_MODEL", "qwen3.5:14b")
SEARCH_PROVIDER = os.getenv("FINDOUT_SEARCH_PROVIDER", "duckduckgo")
QUERY = "{q}"

# === CONNECTIVITY PROBE ===
def _probe_llm() -> bool:
    \"\"\"Check if the LLM endpoint is reachable before running the pipeline.\"\"\"
    probe_url = BASE_URL.rstrip("/v1").rstrip("/")
    for endpoint in [probe_url + "/api/tags", BASE_URL + "/models"]:
        try:
            r = requests.get(endpoint, timeout=5, headers={{}})
            if r.status_code == 200:
                return True
        except Exception:
            continue
    # Last resort: try a raw chat completion to see if anything answers
    try:
        r = requests.post(
            BASE_URL + "/chat/completions",
            json={{"model": MODEL, "messages": [{{"role": "user", "content": "ping"}}], "max_tokens": 1}},
            timeout=5,
            headers={{"Authorization": f"Bearer {{API_KEY}}"}} if API_KEY and API_KEY != "ollama" else {{}},
        )
        return r.status_code < 500
    except Exception:
        return False

print("🔍 Checking LLM endpoint...")
llm_ok = _probe_llm()
print(f"   → {{'reachable' if llm_ok else 'UNREACHABLE'}} at {{BASE_URL}}")

if not llm_ok:
    print(f"\\n⚠️  LLM endpoint at {{BASE_URL}} is not reachable.")
    print("   Pipeline skipped — will do manual verification below.")
    raise SystemExit(0)  # exit cleanly, STEP 2 handles it

# === RUN PIPELINE ===
try:
    from findout.config import Config, LLMConfig, SearchConfig, PipelineConfig
    from findout.pipeline import SelfVerifyPipeline
    from findout.gate import Gate, GateConfig

    config = Config(
        llm=LLMConfig(
            model=MODEL, base_url=BASE_URL, api_key=API_KEY,
            max_tokens=4096, timeout_seconds=120,
        ),
        search=SearchConfig(
            provider=SEARCH_PROVIDER,
            max_results_per_query=5, max_queries_per_claim=3,
        ),
        pipeline=PipelineConfig(
            default_variant="hybrid", gate_enabled=True,
            hybrid_samples=2, max_claims_per_answer=12,
            short_circuit_on_agreement=True,
        ),
    )

    # Gate check
    gate = Gate(GateConfig(), config.llm)
    category, reason = gate.classify_with_reason(QUERY)
    print(f"Gate: {{category}} ({{reason}})")

    # Run
    pipe = SelfVerifyPipeline(config)
    result = pipe.run(query=QUERY, pipeline="hybrid")

    # Results
    print("=" * 60)
    print("PIPELINE RESULTS")
    print("=" * 60)
    print(f"Claims: {{result.total_claims}} total, {{result.verified_claims}} verified, {{result.uncertain_claims}} uncertain, {{result.contradicted_claims}} contradicted")
    print(f"Searches: {{result.total_searches}}")
    print()
    print(result.answer)

except ImportError as e:
    print(f"\\n⚠️  findout package not available: {{e}}")
    raise SystemExit(0)  # exit cleanly, STEP 2 handles it

except Exception as e:
    print(f"\\n⚠️  Pipeline error: {{type(e).__name__}}: {{e}}")
    print("   Continuing to manual fallback...")
    raise SystemExit(0)  # exit cleanly, STEP 3 handles it
```

## STEP 2 — Manual fallback (when the package or LLM is unavailable)

If you reach this step (either the SystemExit from STEP 1 or the package isn't importable), do the following manually using your own tools:

1. **Extract claims** — break the query into atomic factual claims (3-8 claims)
2. **Web search** — for each claim, run 2-3 web searches from different angles using your web_search tool
3. **Cross-check** — determine which claims are supported by sources, which are contradicted, and which are uncertain
4. **Synthesize** — write a verified answer with `[verified]` / `[contradicted]` / `[uncertain]` markers
5. **Cite** — include source URLs for each verified or contradicted claim

## Output format

Include in your final response:
- The verified answer (synthesized from pipeline or manual research)
- A verdict line: "X claims verified, Y uncertain, Z contradicted"
- Source citations when available
- Which path was used: pipeline / manual fallback / direct research
"""
