---
name: inference-sh-workflows
description: "Automate AI workflows, web search, LLM routing, and Twitter/X posts via inference.sh CLI (belt). Route prompts to Claude, Gemini, Kimi K2, GLM-4.6 via OpenRouter. Search the web with Tavily and Exa. Post tweets, threads, and DMs. Chain AI operations into batch, sequential, or scheduled pipelines. Use when the user wants AI-powered search, needs to call an LLM from the terminal, wants to post to Twitter/X, or build multi-step AI automation."
version: 1.0.0
author: okaris
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [automation, search, twitter, llm-routing, workflows, tavily, exa, openrouter, inference-sh]
    related_skills: [inference-sh, xurl]
    requires_toolsets: [terminal]
required_environment_variables:
  - name: INFERENCE_API_KEY
    prompt: "inference.sh API Key"
    help: "Sign up at https://inference.sh and get your key from https://inference.sh/settings/api-keys"
    required_for: full functionality
---

# inference.sh — AI Workflows and Automation

Run LLMs, search the web, post to Twitter/X, and chain AI tasks into automated workflows from the terminal. One API key for all providers.

All commands use the `terminal` tool to run `belt` commands.

## When to Use

- User wants to call an LLM from the terminal (Claude, Gemini, Kimi K2, GLM-4.6)
- User asks for AI-powered web search or research (Tavily, Exa)
- User wants to post tweets, threads, DMs, or automate Twitter/X
- User wants to chain multiple AI operations into a pipeline
- User needs batch processing across multiple inputs
- User wants to schedule recurring AI tasks

## Prerequisites

```bash
belt whoami
```

If not installed:

```bash
curl -fsSL cli.inference.sh | sh
belt login
```

## LLM Routing

Access multiple LLM providers through a single CLI via the `terminal` tool.

| Model | App ID | Best For |
|-------|--------|----------|
| Claude Opus 4.6 | `openrouter/claude-opus-4.6` | Complex reasoning, coding |
| Claude Sonnet 4.6 | `openrouter/claude-sonnet-4.6` | Balanced performance |
| Claude Haiku 4.5 | `openrouter/claude-haiku-4.5` | Fast and cheap |
| Gemini 3 Pro | `openrouter/gemini-3-pro` | Google's latest |
| Kimi K2 | `openrouter/kimi-k2` | Multi-step reasoning agent |
| GLM-4.6 | `openrouter/glm-4.6` | Open-source, coding |
| Auto | `openrouter/auto` | Cost-optimized auto-select |

```bash
# Ask Claude
belt app run openrouter/claude-sonnet-4.6 --input '{"prompt": "Summarize this article: ..."}'

# Auto-select cheapest capable model
belt app run openrouter/auto --input '{"prompt": "Translate to Spanish: Hello world"}'
```

## Web Search

| Model | App ID | Best For |
|-------|--------|----------|
| Tavily Search | `tavily/search` | AI-generated answers with sources |
| Tavily Extract | `tavily/extract` | Clean content from URLs |
| Exa Search | `exa/search` | Semantic web discovery |
| Exa Answer | `exa/answer` | Direct factual answers |

```bash
# AI-powered search with sources
belt app run tavily/search --input '{"query": "latest developments in AI video generation 2026"}'

# Extract clean content from a URL
belt app run tavily/extract --input '{"url": "https://example.com/article"}'

# Semantic search (find similar content)
belt app run exa/search --input '{"query": "open source text to video models", "num_results": 5}'
```

## Twitter/X Automation

Connect your X account at https://inference.sh/settings/connections before using Twitter operations.

| Operation | App ID | Input |
|-----------|--------|-------|
| Post tweet | `twitter/post-tweet` | `{"text": "..."}` |
| Post with media | `twitter/post-create` | `{"text": "...", "media": "file.png"}` |
| Like | `twitter/like` | `{"tweet_id": "..."}` |
| Retweet | `twitter/retweet` | `{"tweet_id": "..."}` |
| Delete tweet | `twitter/delete` | `{"tweet_id": "..."}` |
| Get post | `twitter/get-post` | `{"tweet_id": "..."}` |
| Send DM | `twitter/send-dm` | `{"username": "...", "text": "..."}` |
| Follow user | `twitter/follow` | `{"username": "..."}` |

```bash
# Post a tweet
belt app run twitter/post-tweet --input '{"text": "Hello from my AI agent!"}'

# Post with AI-generated image (combine with inference-sh creative skill)
belt app run twitter/post-create --input '{"text": "Generated this with AI", "media": "output.png"}'
```

## Workflow Patterns

**Batch processing:**
```bash
for prompt in "sunset" "mountain" "ocean"; do
  belt app run p-image --input "{\"prompt\": \"$prompt\"}" --no-wait
done
belt task list
```

**Sequential pipeline — research, summarize, tweet:**
```bash
# 1. Search
belt app run tavily/search --input '{"query": "AI news today"}'
# 2. Summarize with LLM
belt app run openrouter/claude-haiku-4.5 --input '{"prompt": "Summarize in 280 chars: <paste search results>"}'
# 3. Post
belt app run twitter/post-tweet --input '{"text": "<paste summary>"}'
```

**Async with polling:**
```bash
belt app run veo/3.1 --input '{"prompt": "..."}' --no-wait
# Returns task ID — poll later
belt task get <task-id>
```

## Pitfalls

1. **Twitter auth** — Twitter operations require connecting your X account at https://inference.sh/settings/connections first.
2. **Async tasks** — use `--no-wait` for batch jobs. Poll with `belt task get <id>`.
3. **Shell escaping** — when piping AI outputs between commands, use proper quoting to handle special characters.
4. **Rate limits** — Tavily and Exa have per-minute rate limits. Space out bulk queries.
5. **LLM context** — OpenRouter models have varying context windows. Check model details with `belt app get openrouter/<model>`.

## Verification

```bash
# Verify CLI
belt whoami

# Verify search works
belt app run tavily/search --input '{"query": "hello world test"}'

# Verify LLM routing
belt app run openrouter/claude-haiku-4.5 --input '{"prompt": "Say hello in one word."}'
```

## Reference Docs

- `references/search-and-llms.md` — Full search and LLM routing reference
- `references/twitter-automation.md` — Complete Twitter/X operations reference
