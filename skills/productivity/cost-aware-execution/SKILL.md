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

Triggered by:
- `Mode: cheap`
- or `Use cost-aware-execution mode: cheap`

---

## Core Rule

> If a reasonable answer can be given without tools, you MUST NOT use tools.
> Apply this rule per subtask when multiple questions are present.

---

## Tool Suppression

Do NOT use tools for:
- simple arithmetic
- general knowledge
- common estimates (prices, ranges, trends)
- explanations or summaries

Tools are ONLY allowed if:
- real-time or location-specific accuracy is required
- OR the user explicitly requests exact/verified data

If a tool is required in cheap mode:
- ask for confirmation before executing
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

## Modes

### cheap (default)
- no tool usage unless strictly required
- prefer approximate answers
- minimal reasoning

### balanced
- allow limited verification if needed

### thorough
- allow deeper reasoning and tool usage

---

## Anti-Patterns (forbidden)

- using tools for general knowledge
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
"Current ETH price" → tools allowed

---

## Guiding Principle

> Solve the task — but never do more work than necessary.
