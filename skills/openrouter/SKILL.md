---
name: openrouter
description: "Monitor and manage OpenRouter: provider health, costs, generation debugging, model discovery."
version: 1.0.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [openrouter, llm, providers, routing, costs, debugging]
    related_skills: []
---

# OpenRouter Management

## Overview

This skill activates the OpenRouter management agent, giving you direct API access to monitor providers, track costs, debug slow generations, and discover models.

## Prerequisites

- `mcp-openrouter-admin` toolset must be active
- OpenRouter API key configured

## Invocation

Delegate to the `openrouter` profile:
```
delegate_task("check provider health for openai/gpt-oss-120b", profile="openrouter")
delegate_task("how much credit do I have left?", profile="openrouter")
delegate_task("debug this generation: gen-XXXX", profile="openrouter")
```

## Common Tasks

### Provider Health Check
"What providers are serving openai/gpt-oss-120b and what are their speeds?"
→ Uses `or_model_endpoints` → returns throughput, uptime, latency per provider

### Cost Dashboard
"Show me today's API spend by model"
→ Uses `or_overview` or `or_activity(aggregation="by_model")`

### Debug Slow Generation
Paste a generation ID (format: gen-XXXXXXXXXX-XXXXX...) 
→ Uses `or_generation(id)` → returns provider used, latency breakdown, tokens, cost

### Credit Balance
"How much credit is left on the OpenRouter account?"
→ Uses `or_credits`

### Model Discovery
"Find the cheapest model with 128K context and tool support"
→ Uses `or_models` with filters

## Tool Reference

| Tool | Purpose |
|------|---------|
| `or_overview` | Dashboard: credits + today's burn by model |
| `or_credits` | Account balance |
| `or_current_key` | API key metadata |
| `or_models` | Model catalog with pricing |
| `or_model_get` | Full details for one model |
| `or_model_endpoints` | Per-provider uptime/throughput/latency |
| `or_generation` | Single generation lookup by ID |
| `or_activity` | Usage analytics (by model/day/provider/key) |
