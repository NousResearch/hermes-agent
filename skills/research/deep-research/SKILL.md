---
name: deep-research
description: "Multi-source structured research with citation discipline. Decompose → search → fetch → cross-verify → synthesize a cited report."
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [Research, Multi-Source, Citation, Synthesis, Methodology]
    related_skills: [arxiv, blogwatcher, research-paper-writing, llm-wiki]
prerequisites:
  tools: [web_search, web_extract]
---

# deep-research

Multi-source structured research with strict citation discipline. Use this when one search + one summarization isn't enough — when a question needs 6-12 sources cross-verified into a report you'd actually trust.

The skill is **pure methodology**. It teaches the agent to compose existing tools (`web_search`, `web_extract`, optionally `delegate`) into a research pipeline. No new tools required, but works just as well with the free-tier counterparts (`local_web_search`, `local_web_extract`) for users without paid API keys.

## Quickstart — full local-first agentic stack (5 commands)

Drop-in setup with **zero paid API keys** using:
- ✅ Latest Qwen3.5 / Qwen3.6 (Apache 2.0, Feb-Apr 2026 releases)
- ✅ llama.cpp (no daemon, runs anywhere)
- ✅ SearXNG (self-hosted free search) + ddgr fallback (no key)
- ✅ Hermes Agent's `web_search` / `web_extract` swap-in compatibility

```bash
# 1. Get llama.cpp (Linux x86_64 prebuilt — see release page for macOS/CUDA/Vulkan/ROCm)
curl -fsSL "https://github.com/ggerganov/llama.cpp/releases/download/b9010/llama-b9010-bin-ubuntu-x64.tar.gz" | tar xz
export PATH="$(pwd)/llama-b9010:$PATH"
export LD_LIBRARY_PATH="$(pwd)/llama-b9010:$LD_LIBRARY_PATH"

# 2. Pull a Qwen3.5 GGUF (Q4_K_M — 4B = 2.5 GB, 9B = 5.5 GB)
mkdir -p ~/models && cd ~/models
curl -fLO "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf"

# 3. Boot llama-server with deep-research-friendly defaults
~/.hermes/skills/research/deep-research/scripts/start-llama-server.sh ~/models/Qwen3.5-4B-Q4_K_M.gguf

# 4. (Optional) self-host SearXNG for free internet search
docker run -d --name searxng -p 8888:8080 searxng/searxng

# 5. Configure Hermes to use the local backends
export LLM_BASE_URL=http://127.0.0.1:8088
export LLM_DEFAULT_MODEL=qwen3.5-4b-q4_k_m
export SEARXNG_URL=http://127.0.0.1:8888
# Optional API alternatives if SearXNG isn't an option:
#   export BRAVE_SEARCH_API_KEY=...   (free 2K/mo)
#   export TAVILY_API_KEY=...         (free 1K/mo)
```

That's the whole stack. Now from inside a Hermes session (or any agent calling these tools):

```python
# Use local_web_search instead of web_search; same JSON shape
result = local_web_search(query="UAE brass lighting market 2026", limit=8)

# Use local_web_extract instead of web_extract
pages = local_web_extract(urls=[r["url"] for r in result["data"]["web"][:3]])

# Or invoke the deep-research methodology end-to-end via skill_view
skill_view(name="deep-research")   # reads this file, follows the 5-phase pipeline
```

Performance note: Qwen3.5-4B at Q4 on a 12-thread Intel/AMD CPU = ~12 tok/s. A typical research run (8 fetches + synthesis) finishes in 3-6 min, **at zero per-query cost**.

---

## When to use

| Use case | Reason |
|---|---|
| Market analysis (size, players, drivers, regulation) | needs 6+ sources triangulated |
| Deep competitor / company profile | needs cross-source consistency check |
| Regulatory / compliance research | needs official + secondary sources |
| Trend analysis with named-entity grounding | needs multiple corroborating signals |
| Technical investigation across blogs / papers / docs | needs source-type diversity |

For narrow lookups (single fact, one URL), use simpler primitives directly: `web_search` + one `web_extract` call is enough. deep-research is the right choice when the question genuinely needs 5-15 sources.

## The pipeline (5 phases)

```
PHASE 1 — DECOMPOSE  (you, the agent)
   Break the topic into 4-6 concrete sub-questions.
   Don't search the topic verbatim. Search the sub-questions.

PHASE 2 — SEARCH FAN-OUT  (parallel where possible)
   For each sub-question: call `web_search` (or `local_web_search`).
   Collect 3-5 candidate URLs per sub-question.
   Deduplicate by domain — same content syndicated to 3 sites = 1 source.

PHASE 3 — FETCH  (selectively)
   For the most promising URLs: call `web_extract` (or `local_web_extract`).
   Read pages carefully. Use the LLM-summarization mode for long pages
   to save context budget; use raw mode when fidelity matters.

PHASE 4 — CROSS-VERIFY  (you reasoning)
   For each material claim:
     - which sources support it?
     - confidence:  ★★★ (3+ sources agree)
                    ★★  (2 sources agree)
                    ★   (only 1 source — flag in report)
                    ⚠   (sources disagree — report both sides)
                    ?   (inferred / speculation — name as such)
   Drop or downgrade weakly supported claims.

PHASE 5 — SYNTHESIZE  (you write the report)
   Use the report skeleton below.
   Every quantitative claim has a [n] citation.
   The Open Questions section is mandatory.
```

## Report structure (the canonical skeleton)

```markdown
# Research — <topic> — <date>

## Executive summary
3-5 sentences answering the original question directly. Lead with the answer.

## Method
- Sub-questions decomposed: <list>
- Sources scanned: N URLs across M domains
- Confidence basis: <e.g. "12 sources, 7 with cross-references">

## Key findings (cited)

### Finding 1 — <one-line claim>  ★★★
Evidence: [1] supports X; [3] confirms Y; [7] gives the precise figure of Z.
Implication / next step: <one sentence>

### Finding 2 — <claim>  ★★
...

## Numbers worth knowing
| Metric | Value | Source | Confidence |
|---|---|---|---|
Only include numbers that appear verbatim in fetched content.
Cite [n] for each row.

## Open questions / what couldn't be verified
- <gap 1, named explicitly>
- <gap 2 — sources disagreed; which sides>

## Recommendations / next steps
Optional. Include only if the original task implied actionable advice.

## Sources
[1] <full URL>    — title       (accessed YYYY-MM-DD)
[2] <full URL>    — title       (accessed YYYY-MM-DD)
...
```

Citation discipline is mandatory. Every quantitative claim, every named entity, every "experts say" gets a `[n]` pointing into the Sources section. Unsourced assertions = research junk; don't include them.

## Step-by-step example

Topic: *"Latest developments in small language models for tool use, 2025-2026"*

### 1. Decompose

```
Q1: What are the leading 7-9B param open-source models with native tool calling in 2025-2026?
Q2: Which benchmarks measure tool-use quality, and what scores do current SLMs achieve?
Q3: What architectures dominate (dense vs MoE, what tokenizer/format)?
Q4: What real-world deployments (agents, IDEs) are using SLMs for tool calling at production scale?
Q5: What are the main failure modes — hallucination, schema drift, looping?
Q6: What's the 12-month outlook — RL distillation, longer context, smaller specialized models?
```

### 2. Search fan-out (4-8 parallel calls)

```
web_search("Qwen3 8B tool use benchmark 2026")
web_search("DeepSeek-R1-Distill 8B function calling")
web_search("BFCL Berkeley function calling leaderboard small models")
web_search("Hermes 4 small language model release notes")
web_search("Llama 3.3 8B tool calling production")
web_search("MoE vs dense small language model agent")
```

Or, if you want sub-agent dispatch for parallelism, use `delegate`:

```
delegate(prompt="Run 6 searches in parallel: ...", scope="search-only")
```

### 3. Fetch top candidates

```
web_extract(urls=["https://...berkeley-bfcl-leaderboard...",
                  "https://...qwen3-release-notes...",
                  "https://...hermes-4-paper-arxiv...",
                  "https://...llama-3-3-tool-calling-blog..."],
             use_llm_processing=True)
```

For pages where the LLM summary loses fidelity (e.g. benchmark tables, code listings), re-fetch with `use_llm_processing=False` and read raw.

### 4. Cross-verify

For each candidate finding:
- Does it appear in 1, 2, or 3+ fetched sources?
- Are the numbers consistent across sources, or do they conflict?
- Is the source primary (paper, vendor docs) or secondary (blog summary)?

Mark every claim with the appropriate confidence star. If only one secondary blog claims a number, that's ★, not ★★★.

### 5. Synthesize

Write the report following the skeleton above. Cite every claim. Be specific.


## Recommended models (Qwen3.5 / Qwen3.6 family — first-class support)

The Qwen3.5 (Feb 2026) and Qwen3.6 (Apr 2026) families are the recommended local models for this skill. They have native tool-calling, 200+ language coverage, and a 262K-token native context window that comfortably holds 10+ fetched pages without summarization.

| Model | Params | VRAM @ Q4_K_M | Context (native) | Best for |
|---|---|---|---|---|
| `Qwen/Qwen3.5-4B` | 4B dense | ~2.5 GB | 262K | Cheapest first-class option; CPU-runs at ~12 tok/s on 8-core |
| `Qwen/Qwen3.5-9B` | 9B dense | ~5.5 GB | 262K | Single-GPU sweet spot for 8 GB cards |
| `Qwen/Qwen3.5-27B` | 27B dense | ~16 GB | 262K | High-quality synthesis on a single 24 GB GPU |
| `Qwen/Qwen3.6-27B` | 27B dense | ~16 GB | 262K | Latest dense (Apr 2026) |
| `Qwen/Qwen3.6-35B-A3B` | 35B / 3B active | ~21 GB | 262K | Best speed/quality on bigger hardware (MoE) |
| `Qwen/Qwen3.5-122B-A10B` | 122B / 10B active | ~70 GB | 262K | Multi-GPU; rivals frontier on tool-use benchmarks |

GGUF quants from `unsloth/<model>-GGUF`, `bartowski/Qwen_<model>-GGUF`, or `lmstudio-community/<model>-GGUF`.

### Critical Qwen3.5/3.6 operational notes (read before deploying)

The Qwen3 family had a `/think` and `/no_think` soft-switch convention. **Qwen3.5 and Qwen3.6 explicitly DO NOT honor those directives.** From the official model card:

> "Qwen3.5 does not officially support the soft switch of Qwen3, i.e., `/think` and `/nothink`."

**Thinking mode is on by default** — the model emits `<think>...\n</think>\n\n` blocks before its final answer. To disable for tool-call workflows where reasoning trace would interfere:

```python
# In your /v1/chat/completions request body:
{
  "model": "qwen3.5-4b",
  "messages": [...],
  "tools": [...],
  "chat_template_kwargs": {"enable_thinking": False}   # ← THIS, not /no_think
}
```

This is `chat_template_kwargs` — passed at the request level, picked up by the Jinja chat template that llama-server / vLLM / SGLang load with the model.

### Tool-call parser flags (server-side)

Qwen3.5/3.6 use a specific tool-call format. When starting llama-server / vLLM / SGLang, pass:

```bash
# llama.cpp (with --jinja for chat template support)
llama-server -m Qwen3.5-4B-Q4_K_M.gguf --jinja --port 8088

# vLLM
vllm serve Qwen/Qwen3.5-4B \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder

# SGLang
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-4B \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder
```

The `qwen3_coder` parser is correct for Qwen3.5/3.6 (despite the name — it's the official Qwen3.5+ tool-call format).

### Recommended sampling for this skill

Per the official model card, Qwen3.5/3.6 expect mode-specific sampling:

```python
# Instruct / non-thinking mode (recommended for this skill — predictable tool output)
{"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
 "presence_penalty": 1.5, "repetition_penalty": 1.0}

# Thinking mode (when you DO want chain-of-thought before tool calls)
{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
 "presence_penalty": 1.5, "repetition_penalty": 1.0}
```

The default `temperature: 0.3` we use for stable tool-call generation works fine on Qwen3.5/3.6 too, but if you want max-quality synthesis switch to 0.7 + the presence penalty.

### Multilingual research (a real Qwen3.5/3.6 advantage)

Qwen3.5/3.6 cover **201 languages and dialects** with strong multilingual reasoning benchmarks (MMMLU, MMLU-ProX, NOVA-63, INCLUDE, Global PIQA, PolyMATH, MAXIFE, WMT24++).

For research that should produce reports in non-English (Hindi, Arabic, Spanish, Mandarin, etc.), append a single line to the report skeleton in your system prompt:

> "Output the final report in <language>. Sources may be in any language; preserve named entities and numbers verbatim."

The model handles this cleanly without further prompting.

## Backend choice — paid vs free

This skill is backend-agnostic. The pipeline works the same whether you use:

| Backend tier | Search | Extract | Cost |
|---|---|---|---|
| **Paid (default)** | `web_search` (Parallel/Firecrawl/Tavily/Exa) | `web_extract` (Firecrawl + LLM summarization) | ~$0.05-0.50 per research run |
| **Free-tier** | `local_web_search` (SearXNG/Brave free/Tavily free/ddgr/ddgs) | `local_web_extract` (lynx + optional Ollama summarization) | $0 |

The methodology is the same. To use the free-tier path, just substitute the tool names. See `tools/local_web_tools.py` for setup notes.

## Scaling: sub-agent dispatch

For high-volume research (15+ sub-questions, 30+ sources), use `delegate` to spawn parallel sub-agents — one per sub-question — and gather their findings. Pattern:

```
For each sub-question:
    delegate(
        prompt="Research sub-question: '<sub-q>'. Return at most 3 cited findings + sources.",
        scope="research-leaf"
    )

Then synthesize across all the returned snippets in a single supervisor pass.
```

This trades wall-clock latency (parallel) for orchestration complexity. Use only when serial pipeline takes >10 minutes.

## Anti-patterns

- **Don't trust a single source.** ★ markers are warnings, not endorsements. If only one blog claims it, say so.
- **Don't paraphrase numbers without citing.** "About $2B market" without [n] is fabrication bait.
- **Don't conflate currencies / units.** Always preserve the source's currency. If converting, cite the FX rate and date.
- **Don't claim "experts say" without naming the experts.** Either name them, or say "industry commentary suggests" and mark it ?.
- **Don't include sources you didn't actually fetch.** Every URL in the Sources list must be one you read.
- **Don't synthesize without reading the pages.** The pipeline gathers; YOU read. Skipping the read = generic boilerplate.
- **Don't skip Open Questions.** Empty open-questions section ≈ over-claiming. Real research has gaps.
- **Don't ship a research report directly to a customer / stakeholder without review.** Research is internal. Cherry-pick 2-3 cited points and rewrite in the audience's voice.

## Verification (post-process)

After writing the report, optionally verify citations with this shell helper. It checks every `[n]` in the report points to a fetched source, and flags numbers in the report that don't appear in any source file.

```bash
# In the agent's terminal
python3 -c "
import re, sys
from pathlib import Path

report_path = Path(sys.argv[1])
sources_dir = Path(sys.argv[2])
report = report_path.read_text()

# 1. Check every [n] cites a fetched source
sources = {p.stem.split('_', 1)[0] for p in sources_dir.glob('*.txt')}
cited = set(re.findall(r'\[(\d+)\]', report))
missing = [c for c in cited if c.zfill(2) not in {s.zfill(2) for s in sources} and c not in sources]
if missing:
    print(f'⚠ citations not in sources/: {missing}')

# 2. Flag numbers in report that don't appear verbatim in any source
body = report.split('## Sources')[0] if '## Sources' in report else report
all_text = '\n'.join(p.read_text().lower() for p in sources_dir.glob('*.txt'))
candidates = re.findall(r'\b(\d{1,4}[,.]?\d{0,3}\s*(?:%|USD|EUR|GBP|kg|tonne|year|days?))', body, flags=re.IGNORECASE)
for num in set(candidates):
    digits = re.search(r'\d[\d,.]*', num).group(0)
    if num.lower() not in all_text and digits not in all_text:
        print(f'⚠ unverified number: {num!r}')

print('✓ verification complete')
" report.md sources/
```

## Cross-references

- Search and extraction primitives: see `web_search` and `web_extract` (paid backends), or `local_web_search` and `local_web_extract` (free backends in `tools/local_web_tools.py`)
- For sub-agent dispatch at scale: see `delegate`
- For domain-specific sources: see `arxiv` (academic papers), `blogwatcher` (RSS / blogs), `research-paper-writing` (post-research authoring)
- For the supervisor's review pattern (spot-check + correct sub-agent output): the methodology applies symmetrically when you ARE the sub-agent — synthesize honestly, expect supervision.
