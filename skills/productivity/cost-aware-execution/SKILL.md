---
name: cost-aware-execution
description: Trigger via slash command to minimize tool use and cost.
version: 0.4.0
author: JackTheGit
license: MIT
metadata:
  tags: [productivity, efficiency, cost-awareness]
---

# Cost-Aware Execution

A meta-skill that enforces **minimum sufficient effort** to reduce cost, latency, and unnecessary tool usage.

---

## Activation

Triggered by the native agent slash command:
- `/cost-aware-execution`

---

## Core Rule

> If a reasonable answer can be given without tools, you MUST NOT use tools.
> Apply this rule per subtask when multiple questions are present.

---

## Tool Suppression

Do NOT invoke native tools like `web_search`, `python_interpreter`, or `code_executor` for:
- simple arithmetic
- general knowledge
- common estimates (prices, ranges, trends)
- explanations or summaries

Native tools are ONLY allowed if:
- real-time or location-specific accuracy strictly requires `web_search` or system execution
- OR the user explicitly requests exact/verified data

If a tool execution is required after activation:
- ask for confirmation before executing the tool
- do not auto-execute

---

## Decision Process

1. Can I answer directly or approximately? → answer immediately  
2. Is the answer sufficient? → STOP  
3. Do I strictly need a tool? → if not, do not use tools  

---

## Multi-Task Handling

If multiple questions are provided in a single message:

- Treat each question independently  
- Answer directly whenever possible  
- Do not escalate the entire task due to one complex subtask  
- Avoid delegation or parallel execution for simple queries  
- Prefer sequential, lightweight handling over parallel delegation  

---

## Anti-Patterns (forbidden)

- using tools like `web_search` or `python_interpreter` for general knowledge
- retrying tool calls unnecessarily
- continuing after a sufficient answer
- over-analyzing simple tasks

---

## Examples

**Arithmetic**  
"What is 25 * 17?" → 425 (no tools)

**General estimate**  
"Price of milk?" → ~$3.50–$4.50 (no tools)

**Real-time query**  
"Current ETH price" → tools (`web_search`) allowed

---

## Guiding Principle

> Solve the task — but never do more work than necessary.
