# Search and LLM Routing Reference

## Web Search

### Tavily
- **tavily/search** — AI-generated answers with cited sources
- **tavily/extract** — Clean text extraction from any URL

Best for: research questions, fact-checking, RAG pipelines, news monitoring.

```bash
belt app run tavily/search --input '{"query": "best practices for AI agent deployment 2026", "search_depth": "advanced"}'
belt app run tavily/extract --input '{"url": "https://example.com/research-paper"}'
```

### Exa
- **exa/search** — Semantic (meaning-based) web search
- **exa/answer** — Direct factual answers
- **exa/extract** — Page content analysis

Best for: finding similar content, discovery, semantic matching.

```bash
belt app run exa/search --input '{"query": "open source alternatives to Midjourney", "num_results": 10}'
belt app run exa/answer --input '{"query": "What is the latest version of PyTorch?"}'
```

### When to Use Which
- **Tavily** for "what is the answer to X" (answer generation with sources)
- **Exa** for "find me things like X" (semantic discovery)

## LLM Routing via OpenRouter

All models use the same interface:

```bash
belt app run openrouter/<model> --input '{"prompt": "...", "max_tokens": 1000}'
```

### Model Selection Guide

| Need | Model | Why |
|------|-------|-----|
| Best reasoning | `openrouter/claude-opus-4.6` | Strongest at complex analysis |
| Balanced | `openrouter/claude-sonnet-4.6` | Good quality, reasonable cost |
| Fast/cheap | `openrouter/claude-haiku-4.5` | Quick tasks, high throughput |
| Google ecosystem | `openrouter/gemini-3-pro` | Latest Google model |
| Multi-step agents | `openrouter/kimi-k2` | Thinking agent, 200+ tool calls |
| Open-source | `openrouter/glm-4.6` | Strong coding, open weights |
| Cost-optimized | `openrouter/auto` | Auto-selects cheapest capable |

### Common Patterns

```bash
# Summarization
belt app run openrouter/claude-haiku-4.5 --input '{"prompt": "Summarize in 3 bullet points: <content>"}'

# Translation
belt app run openrouter/claude-sonnet-4.6 --input '{"prompt": "Translate to Japanese: Hello, how are you?"}'

# Code generation
belt app run openrouter/claude-opus-4.6 --input '{"prompt": "Write a Python function that...", "max_tokens": 2000}'
```
