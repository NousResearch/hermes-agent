---
name: deepagents
description: "Run LangChain's Deep Agents: multi-agent LangGraph harness."
version: 0.1.0
author: Tony Curtis (tc45), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Coding-Agent, LangGraph, DeepAgents, Harness, Python]
    related_skills: [claude-code, codex, opencode, hermes-agent]
---

# Deep Agents

Run [LangChain's Deep Agents](https://github.com/langchain-ai/deepagents) — an
opinionated, model-agnostic agent harness (planning, sub-agents, filesystem,
skills, human-in-the-loop) built on LangGraph — as a scripted Python task, not
an interactive CLI. Use this to prototype/benchmark the harness itself, not as
a drop-in replacement for `claude-code`/`codex` (those wrap real terminal
coding agents with repo access; Deep Agents is a library you compose inside a
Python script).

## When to Use

- User explicitly asks to try/test/benchmark Deep Agents (LangGraph-based
  harness) as opposed to Claude Code, Codex, or OpenHands.
- Prototyping a custom multi-agent LangGraph pipeline (sub-agents, virtual
  filesystem, skills-on-demand) before committing to an architecture.
- Don't use for: routine coding delegation (use `claude-code` or `codex`),
  or when the user just wants a task done fast — Deep Agents needs a Python
  script per run, unlike the CLI-native sibling skills.

## Prerequisites

- Python 3.11+ and `uv` (already on this host).
- A LangChain-compatible chat model. Two model paths:
  1. **Native API key** — `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` etc. in
     `~/.hermes/.env`. Use `init_chat_model("openai:gpt-5")` etc.
  2. **Hermes-authenticated, no separate key** — `hermes proxy start
     --provider nous` runs a local OpenAI-compatible server that forwards to
     Nous Portal using Hermes's own OAuth session. Point `ChatOpenAI` at it.
     This is the path to use when the user has no raw API key configured but
     is logged into Hermes/Nous — check with `hermes proxy status` first.
- Install per-project, not globally: `uv venv .venv && uv pip install
  --python .venv deepagents langchain-openai` (swap `langchain-openai` for
  the relevant `langchain-<provider>` package if using a native key).

## How to Run

### 1. Start the model backend (Nous Portal path)

```
terminal(command="hermes proxy status")   # confirm "nous ... ready"
terminal(command="hermes proxy start --provider nous --port 8645", background=true)
```

List models to pick a slug (paid models 404 with `insufficient_credits_for_paid_model`
if the account balance is too low — filter for `:free` suffixed slugs to avoid that):

```
terminal(command="curl -s http://127.0.0.1:8645/v1/models -H 'Authorization: Bearer x' | python -c \"import json,sys; [print(m['id']) for m in json.load(sys.stdin)['data'] if m.get('pricing',{}).get('prompt') in ('0','0.0000000000')]\"")
```

### 2. Write and run the agent script

```python
# script.py
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent

model = ChatOpenAI(
    model="meituan/longcat-2.0:free",       # any slug from step 1
    base_url="http://127.0.0.1:8645/v1",
    api_key="x",                             # proxy ignores the value, any string works
)

agent = create_deep_agent(
    model=model,
    tools=[],                                # bring your own tools / MCP here
    system_prompt="You are a research assistant.",
)

result = agent.invoke({"messages": [{"role": "user", "content": "..."}]})
print(result["messages"][-1].content)
```

```
terminal(command=".venv/Scripts/python.exe script.py", workdir="<project dir>")
```

## Quick Reference

| Task | Command |
|---|---|
| Check Nous proxy readiness | `hermes proxy status` |
| List proxy upstreams | `hermes proxy providers` |
| Start proxy | `hermes proxy start --provider nous --port 8645` |
| Install harness | `uv pip install --python .venv deepagents langchain-openai` |
| Run script | `.venv/Scripts/python.exe script.py` (Windows) / `.venv/bin/python script.py` (POSIX) |

## Pitfalls

- **Paid-model 404s look like auth failures but aren't.** `Error code: 404 -
  ... insufficient_credits_for_paid_model` means the Nous account balance is
  too low for that specific model, not that the proxy or key is broken. Retry
  with a `:free`-suffixed slug from the `/v1/models` listing.
- **The proxy's `api_key` value is unchecked.** `ChatOpenAI(api_key=...)`
  needs *some* non-empty string to satisfy the client library; the proxy
  authenticates upstream with Hermes's own OAuth session, not that value.
- **`create_deep_agent` pulls in the full LangChain agent stack** (langgraph,
  langsmith, langchain-anthropic middleware for prompt caching, etc.) even
  when the model is OpenAI-compatible — expect a heavier dependency tree than
  `langchain-openai` alone; this is normal, not a broken install.
- **This is a library, not a CLI agent.** Unlike `claude-code`/`codex`, Deep
  Agents has no repo-aware terminal mode out of the box — every run is a
  Python script you write, defining tools/subagents/prompt yourself.
- **Windows paths:** venv Python lives at `.venv\Scripts\python.exe`, not
  `.venv/bin/python`.

## Verification

Run the script above and confirm `result["messages"][-1].content` contains a
real, on-topic answer (not an exception traceback). A working run with the
Nous proxy path prints one line of model output and exits 0.

## Related

- [Deep Agents GitHub](https://github.com/langchain-ai/deepagents)
- [Deep Agents docs](https://docs.langchain.com/oss/python/deepagents/overview)
- Sibling skills: `claude-code` (Anthropic CLI agent), `codex` (OpenAI CLI
  agent), `opencode` (multi-provider CLI agent), `hermes-agent` (native
  Hermes subagents via `delegate_task`).
