---
name: belt-productive
description: "Automate AI workflows, web search, LLM routing, Twitter/X posts, video rendering, and browser automation via inference.sh CLI (belt). Route prompts to Claude, Gemini, Kimi K2, GLM-4.6 via OpenRouter. Search the web with Tavily and Exa. Render videos from React/Remotion or HyperFrames compositions. Automate browsers for scraping and testing. Use when the user wants AI-powered search, LLM calls from the terminal, Twitter/X automation, programmatic video rendering, or browser agents."
version: 1.0.0
author: okaris
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [automation, search, twitter, llm-routing, workflows, tavily, exa, openrouter, remotion, hyperframes, browser, inference-sh]
    related_skills: [belt-creative, xurl]
    requires_toolsets: [terminal]
required_environment_variables:
  - name: INFERENCE_API_KEY
    prompt: "inference.sh API Key"
    help: "Sign up at https://inference.sh and get your key from https://inference.sh/settings/api-keys"
    required_for: full functionality
---

# inference.sh — AI Workflows and Automation

Run LLMs, search the web, post to Twitter/X, and chain AI tasks into automated workflows from the terminal using [inference.sh](https://inference.sh). One API key for all providers.

All commands use the `terminal` tool to run `belt` (the inference.sh CLI).

## When to Use

- User wants to call an LLM from the terminal (Claude, Gemini, Kimi K2, GLM-4.6)
- User asks for AI-powered web search or research (Tavily, Exa)
- User wants to post tweets, threads, DMs, or automate Twitter/X
- User wants to render video from React/Remotion components or HyperFrames compositions
- User needs browser automation — scraping, navigation, form filling, testing
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

# Post with AI-generated image (combine with belt-creative skill)
belt app run twitter/post-create --input '{"text": "Generated this with AI", "media": "output.png"}'
```

## Video Rendering

Render videos programmatically from code — no video editor needed.

| Tool | App ID | Best For |
|------|--------|----------|
| Remotion Render | `infsh/remotion-render` | React/Remotion components to MP4 |
| HyperFrames Render | `infsh/hyperframes-render` | HyperFrames compositions to video |
| HTML to Video | `infsh/html-to-video` | HTML/CSS/JS animations to video |

```bash
# Render a Remotion project
belt app run infsh/remotion-render --input '{"repo_url": "https://github.com/user/remotion-project", "composition": "Main"}'

# Render HyperFrames composition
belt app run infsh/hyperframes-render --input '{"composition_url": "https://example.com/comp.json"}'

# Render HTML animation to video
belt app run infsh/html-to-video --input '{"html": "<div style=\"animation: fade 2s\">Hello</div>", "duration": 5}'
```

## Browser Automation

Automate browser interactions — scrape pages, fill forms, take screenshots, run test flows.

| Tool | App ID | Best For |
|------|--------|----------|
| Agent Browser | `infsh/agent-browser` | AI-driven browser navigation and scraping |

```bash
# Navigate and extract content
belt app run infsh/agent-browser --input '{"url": "https://example.com", "task": "extract the main article text and all image URLs"}'

# Take a screenshot
belt app run infsh/agent-browser --input '{"url": "https://example.com", "task": "take a full page screenshot"}'

# Fill a form
belt app run infsh/agent-browser --input '{"url": "https://example.com/signup", "task": "fill the registration form with test data"}'
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
