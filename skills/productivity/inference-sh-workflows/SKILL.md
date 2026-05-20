---
name: inference-sh-workflows
description: Automate AI workflows, web search, and social posts.
version: 1.0.0
author: inference.sh
license: MIT
metadata:
  hermes:
    tags: [automation, search, twitter, llm-routing, workflows]
    related_skills: [inference-sh, xurl]
    requires_toolsets: [terminal]
required_environment_variables:
  - name: INFERENCE_API_KEY
    prompt: "inference.sh API Key"
    help: "Get from https://inference.sh/settings/api-keys"
    required_for: full functionality
---

# inference.sh — AI Workflows and Automation

Run LLMs, search the web, post to Twitter/X, and chain AI tasks into automated workflows from the terminal. One API key for all providers.

## When to Use

- Route prompts to different LLMs (Claude, Gemini, Kimi, GLM) via a single CLI
- Search the web with AI-powered answers (Tavily, Exa)
- Post tweets, threads, or DMs with optional AI-generated media
- Chain multiple AI operations into batch or sequential pipelines
- Schedule recurring AI tasks with cron jobs

## Prerequisites

Install the CLI via the `terminal` tool:

```bash
curl -fsSL cli.inference.sh | sh
belt login
```

## How to Run

```bash
belt app run <model> --input '{"prompt": "..."}'
```

Browse available models:

```bash
belt app store --category search   # or: llm, social
belt app store search "tavily"
```

## Quick Reference

### LLM Routing

Access multiple LLM providers through a single endpoint via the `terminal` tool.

| Model | Best For | Notes |
|-------|----------|-------|
| `openrouter/claude-opus-4.6` | Complex reasoning, coding | Anthropic |
| `openrouter/claude-sonnet-4.6` | Balanced performance | Anthropic |
| `openrouter/claude-haiku-4.5` | Fast and cheap | Anthropic |
| `openrouter/gemini-3-pro` | Google's latest | Google |
| `openrouter/kimi-k2` | Multi-step reasoning | Moonshot AI |
| `openrouter/glm-4.6` | Open-source, coding | Zhipu AI |

```bash
# Ask Claude
belt app run openrouter/claude-sonnet-4.6 --input '{"prompt": "Summarize this article: ..."}'

# Auto-select cheapest capable model
belt app run openrouter/auto --input '{"prompt": "Translate to Spanish: Hello world"}'
```

### Web Search

| Model | Best For | Notes |
|-------|----------|-------|
| `tavily/search` | AI-generated answers with sources | Best for research |
| `tavily/extract` | Clean content from URLs | Page extraction |
| `exa/search` | Semantic web search | Best for discovery |
| `exa/answer` | Direct factual answers | Quick facts |

```bash
# AI-powered search
belt app run tavily/search --input '{"query": "latest developments in AI video generation 2026"}'

# Extract content from a URL
belt app run tavily/extract --input '{"url": "https://example.com/article"}'

# Semantic search
belt app run exa/search --input '{"query": "open source text to video models", "num_results": 5}'
```

### Twitter/X Automation

```bash
# Post a tweet
belt app run twitter/post-tweet --input '{"text": "Hello from my AI agent!"}'

# Post with AI-generated image
belt app run p-image --input '{"prompt": "sunset over mountains"}' 
belt app run twitter/post-create --input '{"text": "Generated this with AI", "media": "output.png"}'

# Like, retweet, follow
belt app run twitter/like --input '{"tweet_id": "123456"}'
belt app run twitter/retweet --input '{"tweet_id": "123456"}'
belt app run twitter/follow --input '{"username": "naborstudio"}'

# Send a DM
belt app run twitter/send-dm --input '{"username": "friend", "text": "Hey!"}'
```

### Workflow Patterns

**Batch processing** — run the same operation on multiple inputs:

```bash
# Generate images for a list of prompts
for prompt in "sunset" "mountain" "ocean"; do
  belt app run p-image --input "{\"prompt\": \"$prompt\"}" --no-wait
done
# Check all tasks
belt task list
```

**Sequential pipeline** — chain outputs:

```bash
# Research → Summarize → Tweet
RESULT=$(belt app run tavily/search --input '{"query": "AI news today"}')
SUMMARY=$(belt app run openrouter/claude-haiku-4.5 --input "{\"prompt\": \"Summarize in 280 chars: $RESULT\"}")
belt app run twitter/post-tweet --input "{\"text\": \"$SUMMARY\"}"
```

**Async with polling** — for long-running tasks:

```bash
# Start a long task
TASK_ID=$(belt app run veo/3.1 --input '{"prompt": "..."}' --no-wait)
# Poll until done
belt task get $TASK_ID
```

**Scheduled automation:**

```bash
# Run a script on a cron schedule
belt cron create "0 9 * * *" "belt app run tavily/search --input '{\"query\": \"AI news\"}'"
```

## Pitfalls

- **Twitter auth:** Twitter operations require connecting your X account at https://inference.sh/settings/connections before use.
- **Async tasks:** Use `--no-wait` for batch jobs. Poll with `belt task get <id>` or `belt task list`.
- **LLM context:** OpenRouter models have varying context windows. Check model details with `belt app get openrouter/<model>`.
- **Search rate limits:** Tavily and Exa have per-minute rate limits. Space out bulk queries.
- **Shell escaping:** When piping outputs between commands, use proper quoting to handle special characters in AI-generated text.

## Verification

```bash
# Verify CLI is installed and authenticated
belt whoami

# Verify search works
belt app run tavily/search --input '{"query": "hello world test"}'

# Verify LLM routing works
belt app run openrouter/claude-haiku-4.5 --input '{"prompt": "Say hello in one word."}'
```
