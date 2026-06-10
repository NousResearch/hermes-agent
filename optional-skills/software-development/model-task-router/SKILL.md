---
name: model-task-router
description: Automatically route tasks to the optimal model based on task type. Coding → GPT-5.4, architecture → GPT-5.5, orchestration → V4-Pro, search → V4-Pro. Use when you need to classify a user request and dispatch it to the right model without the user manually switching.
version: 1.0.0
author: Sugumaran Balasubramaniyan (https://github.com/Sugumaran-Balasubramaniyan)
license: MIT
metadata:
  hermes:
    tags: [model-routing, task-classification, cost-optimization, delegation, multi-model]
    related_skills: [model-selection]
    requires_toolsets: [terminal]
---

# Model Task Router

Automatic task-to-model routing for Hermes Agent. Classifies incoming user requests and dispatches them to the optimal model — no manual `/model` switches required.

**By [Sugumaran Balasubramaniyan](https://github.com/Sugumaran-Balasubramaniyan)** — AI/ML Engineer | MLOps | LLM Systems  
Portfolio: [sugumaran-balasubramaniyan.com](https://www.sugumaran-balasubramaniyan.com) | LinkedIn: [@sugumaranbalasubramaniyan](https://www.linkedin.com/in/sugumaranbalasubramaniyan/)  
HuggingFace: [sugumaran](https://huggingface.co/sugumaran) | Email: bsugumaran@hotmail.com

## Why This Skill Exists

DeepSWE ([deepswe.datacurve.ai](https://deepswe.datacurve.ai)) — the only contamination-free, real-complexity coding benchmark — reveals a brutal truth:

| Model | SWE-bench Pro | DeepSWE | Collapse |
|-------|:------------:|:-------:|:--------:|
| GPT-5.5 | 58.6% | **70%** | +11 |
| GPT-5.4 | 57.7% | **56%** | −2 |
| DeepSeek V4-Pro | 55.4% | **8%** | **−47** |
| MiniMax M3 | 59.0% | 20% | −39 |

Models that look identical on public benchmarks (55-60% Pro) diverge by **47 points** on real engineering tasks. V4-Pro costs **$52.75 per solved task** vs GPT-5.4 at **$7.82** — despite being 17× cheaper per token — because it solves almost nothing.

But V4-Pro is excellent at tool orchestration (Terminal-Bench 67.9%, $0.87/M). The optimal strategy is **task-based routing**: orchestration on V4-Pro, coding on GPT-5.4+.

Full data: `references/deepswe-routing-data.md`

## When to Use

Load this skill at session start for any session involving mixed task types (coding + orchestration + research). All subsequent turns use automatic routing.

## Routing Table (Backed by DeepSWE + Terminal-Bench Data)

| Task Category | Model | DeepSWE | Why |
|--------------|-------|:-----:|-----|
| **Code Generation** (implement, fix, refactor, write tests, PR) | `openrouter/openai/gpt-5.4` | 56% | 7× V4-Pro on real coding |
| **Hard Architecture** (system design, complex debugging, security audit) | `openrouter/openai/gpt-5.5` | 70% | Best-in-class reasoning + coding |
| **Orchestration** (tools, shell, file ops, navigation, diagnostics) | Current (V4-Pro) | 8% | Terminal-Bench 67.9%, $0.87/M |
| **Research** (web search, documentation, analysis, planning) | Current (V4-Pro) | — | Good reasoning, cheap |
| **Mechanical** (grep, find, list files, run tests, simple edits) | Subagent auto (GPT-5.4-Mini) | 24% | delegate_task default |

## Decision Tree

```
User says: "..."
│
├─ Contains: "implement", "build", "create", "write code", "fix bug",
│           "refactor", "add feature", "PR", "patch", "debug" with code
│  AND task requires writing >20 lines of new code or modifying multiple files
│  → ROUTE: Coding task → dispatch to GPT-5.4
│
├─ Contains: "architecture", "design system", "how would you structure",
│           "hard problem", "complex reasoning", "security review"
│  AND task is high-stakes, multi-system, or requires deep expertise
│  → ROUTE: Architecture task → dispatch to GPT-5.5
│
├─ Contains: "search for", "find all", "grep", "list files",
│           "run tests", "check if", "what's the content of"
│  AND task is purely mechanical
│  → ROUTE: Mechanical → use delegate_task (auto GPT-5.4-Mini)
│
├─ Contains: "research", "summarize", "what is", "explain", "how does",
│           "find documentation", "look up", "search the web for"
│  → HANDLE: Research → use web_search/web_extract with current model
│
└─ Everything else: tool calls, shell commands, file navigation,
   diagnostics, configuration, planning
   → HANDLE: Orchestration → use current model (V4-Pro)
```

## Dispatch Mechanism

### For Coding Tasks → terminal spawn GPT-5.4

```bash
# Dispatch pattern — use terminal with background=true for long tasks
hermes chat -q "<full task description with file paths and context>" \
  --model openrouter/openai/gpt-5.4 \
  --provider openrouter \
  --toolsets terminal,file,web
```

**IMPORTANT:** When dispatching a coding task:
1. Include ALL relevant context in the -q string: file paths, error messages, constraints, existing code snippets
2. Set a generous timeout: coding tasks can take 5-15 minutes
3. If the task is large, spawn in background and poll
4. Verify the subagent's results before reporting to user — subagents can hallucinate completion
5. If the subagent fails, try once more with more specific instructions, then report the blocker

### For Architecture Tasks → terminal spawn GPT-5.5

```bash
hermes chat -q "<full architecture question with context>" \
  --model openrouter/openai/gpt-5.5 \
  --provider openrouter \
  --toolsets terminal,file,web
```

### For Mechanical Tasks → delegate_task

Use `delegate_task` — automatically uses GPT-5.4-Mini (24% DeepSWE). Good for:
- Searching codebase for patterns
- Running test suites
- Simple file modifications
- Finding and listing files

### For Orchestration/Research → Handle Directly

Use the current model (V4-Pro). Do NOT delegate or spawn. Handle tool calls inline.

## Anti-Patterns

- **Do NOT delegate orchestration.** V4-Pro is better at tool-calling than GPT-5.4-Mini. Only delegate actual code generation.
- **Do NOT code inline with V4-Pro.** It scores 8% on DeepSWE. Any non-trivial code change MUST be dispatched to GPT-5.4+.
- **Do NOT use GPT-5.5 for routine coding.** GPT-5.4 is 80% as good at 50% the cost. Reserve GPT-5.5 for architecture and hard reasoning.
- **Do NOT spawn a subagent without context.** The spawned process has no memory of the conversation. Pack ALL relevant info into the -q string.
- **Do NOT trust subagent self-reports.** Subagents say "completed" when they haven't. Always verify: read the file, run the test, check the output.

## Verification Checklist

After any dispatched coding task:
1. Read the modified file(s) to verify changes exist
2. Run relevant tests if available
3. Check for syntax errors
4. Report actual results (not subagent's self-report) to the user

## Cost Awareness

- Orchestration turn (V4-Pro): ~$0.09 per 100K output
- Coding dispatch (GPT-5.4): ~$1.50 per task
- Architecture dispatch (GPT-5.5): ~$3.00 per task
- Mechanical subagent (GPT-5.4-Mini): ~$0.45 per task

Announce the routing decision briefly so the user knows what's happening:
"Routing to GPT-5.4 for implementation..." or "Handling with V4-Pro."

## Hermes Config for This Skill

```yaml
# ~/.hermes/config.yaml
model:
  default: deepseek-v4-pro
  provider: deepseek

fallback_providers:
  - openrouter/openai/gpt-5.4
  - openrouter/qwen/qwen3.7-max
  - openrouter/google/gemini-2.0-flash-001

delegation:
  model: openrouter/openai/gpt-5.4-mini
  provider: openrouter
  reasoning_effort: medium
  max_concurrent_children: 8
  child_timeout_seconds: 900

agent:
  reasoning_effort: high

# All auxiliary models → deepseek-v4-flash + provider: deepseek
```

## Related Hermes Issues

- [#30652](https://github.com/NousResearch/hermes-agent/issues/30652) — Dynamic model routing based on task complexity (open, P3)
- [#16525](https://github.com/NousResearch/hermes-agent/issues/16525) — Expose model_switch as agent-callable tool (open, P3)
- [#32704](https://github.com/NousResearch/hermes-agent/issues/32704) — Capability-Based Multi-Model Routing (closed, duplicate)
- [#18591](https://github.com/NousResearch/hermes-agent/issues/18591) — Per-task model override for delegate_task (open)

## Pitfalls

- **delegate_task model is fixed** — cannot override per-task. Use terminal-spawned hermes for model-specific dispatching.
- **Spawned hermes has no session memory** — pack all context. Be thorough.
- **Background processes need monitoring** — use process(action='poll') or process(action='wait') to track completion.
- **Rate limits** — GPT-5.4 and GPT-5.5 share OpenAI rate limits via OpenRouter. If one fails, try the other or fall back to V4-Pro.
- **V4-Pro for coding is a known failure mode** — 8% DeepSWE means 92% of real coding tasks fail. Never inline code with V4-Pro.
