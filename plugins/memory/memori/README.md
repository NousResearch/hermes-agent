[![Memori Labs](https://images.memorilabs.ai/banner-dark-large.jpg)](https://memorilabs.ai/)

<p align="center">
  <strong>Memory from what agents do, not just what they say.</strong>
</p>

<p align="center">
  <i>Give Hermes Agent persistent, structured memory with Memori. Capture agent trace, tool activity, decisions, outcomes, and conversation into durable memory that Hermes can recall across sessions.</i>
</p>

---

# Memori for Hermes Agent

Memori gives Hermes Agent a structured, long-term memory provider that captures not only conversation, but also Hermes' completed-turn tool trace. As Hermes completes work, Memori can structure durable memory from the agent’s actions: tool calls, workflow steps, assistant decisions, outcomes, failures, constraints, and feedback.

This means Hermes can remember what happened during prior execution paths, not just what was said in the transcript. Instead of stuffing old conversation history into every prompt, Hermes can intentionally retrieve the structured context it needs to continue work, avoid repeated mistakes, preserve project knowledge, and improve across sessions.

---

## The Problem

Agent workflows often lose useful context across sessions:

- Prior decisions and constraints disappear
- Workflow state is scattered across long transcripts
- Conversation-only memory misses tool calls, workflow decisions, outcomes, failures, and corrections
- Failures and corrections are repeated
- Project context is hard to retrieve precisely
- Cross-session memory can become noisy without scoping

---

## What Memori Changes

Memori adds structured, scoped, agent-native memory to Hermes through the `memori` memory provider.

It gives Hermes:

- Automatic capture after completed, non-interrupted turns
- Structured memory from agent trace, execution paths, decisions, outcomes, and conversation
- Agent-Controlled Intelligent Recall through explicit tools
- Project, entity, process, and session scoping
- Structured summaries for state awareness
- Fail-soft behavior so memory issues do not stop Hermes from answering

Hermes' built-in `MEMORY.md` and `USER.md` files remain active. Memori is additive: it does not mirror, edit, replace, or remove those files.

---

## How It Works

Memori runs on two parallel systems:

### 1. Advanced Augmentation

After Hermes completes a turn, the Memori provider captures the user message, assistant response, and completed-turn tool trace in the background.

Memori then converts the completed interaction into structured memory primitives, including:

- User goals and preferences
- Assistant decisions and reasoning outcomes
- Tool calls and execution steps
- Workflow state and task progress
- Constraints, instructions, and project-specific rules
- Results, failures, corrections, and recurring patterns

Memory is scoped by entity, project, process, and session, then updated asynchronously after the response so it does not block the user-facing answer.

Hermes sends tool trace using a **full raw after Hermes processing** policy:

- Tool names, parsed arguments, durations, and status flags are included
- Tool result content is the final Hermes-processed tool message content
- Existing Hermes result persistence, budget handling, guardrail annotations, and steer annotations are preserved
- Memori does not add an additional redaction layer, so users should only enable it for workspaces where sending tool outputs to Memori Cloud is acceptable

This is how Hermes continuously builds memory from what it says and what it does.

### 2. Agent-Controlled Intelligent Recall

Memori separates memory creation from memory recall:

- **Creation** is automatic (advanced augmentation).
- **Recall** is intentional (agent-controlled).

Agents decide:

- When to recall
- What scope to recall from
- How much history to include

Recall is also intelligent. When memories are retrieved, Memori does not simply return raw chronological history. It uses a proprietary multi-dimensional ranking algorithm to prioritize the memories most likely to matter for the current agent task.

The recall algorithm takes into account factors such as:

- Source and signal weights
- Recency
- Frequency
- Memory type
- Scope relevance
- Historical importance
- Retrieval context

This allows agents to retrieve the most relevant facts, decisions, constraints, patterns, and prior outcomes without stuffing irrelevant history into the prompt.

**Supported recall parameters:** `query`, `project_id`, `session_id`,
`date_start`, `date_end`, `source`, `signal`

**Memory classification schema**

*Sources*

- `constraint`
- `decision`
- `execution`
- `fact`
- `insight`
- `instruction`
- `status`
- `strategy`
- `task`

*Signals*

- `commit`
- `discovery`
- `failure`
- `inference`
- `pattern`
- `result`
- `update`
- `verification`

**Default behavior:** If no date range is provided, recall returns all-time memories.

**Returned context may include:**

- Relevant facts
- Prior decisions
- Constraints
- Patterns
- Summaries
- Execution outcomes
- Known failure modes

Available tools:

- **`memori_recall`** - query structured memory for facts, constraints,
  decisions, outcomes, and patterns
- **`memori_recall_summary`** - retrieve summaries and daily-brief-style state
  awareness
- **`memori_quota`** - check Memori quota and limits
- **`memori_signup`** - request a Memori API key
- **`memori_feedback`** - report memory quality issues or wins

The plugin also ships an explicit-load skill, **`memori:memory`**, with agent
guidance for recall, summaries, quota checks, signup, feedback, and
trace-backed memory behavior.

---

## Quickstart

### Prerequisites

- Hermes Agent with memory provider plugins
- Python 3.10+
- A Memori API key from [app.memorilabs.ai](https://app.memorilabs.ai)
- An Entity ID to scope memory to a specific user, workspace, agent, or system

### 1. Install

```bash
pip install memori
```

### 2. Configure

Run Hermes' memory provider setup flow:

```bash
hermes memory setup
```

Select `memori`, then enter your Memori API key and entity ID.

Environment variables override file config:

- `MEMORI_API_KEY`
- `MEMORI_ENTITY_ID`
- `MEMORI_PROJECT_ID`

`MEMORI_PROJECT_ID` is optional. When omitted, the provider uses Hermes'
active workspace, agent identity, user ID, session title, or session ID as the
Memori project scope.

Manual configuration requires both the API key and entity ID:

```bash
hermes config set memory.provider memori
echo "MEMORI_API_KEY=your-key" >> ~/.hermes/.env
echo "MEMORI_ENTITY_ID=your-user-or-workspace-id" >> ~/.hermes/.env
```

### 3. Verify

```bash
hermes memory status
```

### 4. Test the Memory Loop

1. Ask Hermes to complete a multi-step task:

   > "Investigate why the payment sync test is failing and fix it."

2. Let Hermes complete the workflow. During the task, Hermes may inspect files,
   run commands, identify a failing fixture, make a decision, apply a fix, and
   observe the result.

3. After the turn completes, Memori structures the completed execution path into
   durable memory, including the failure, decision, fix, outcome, and any
   recurring pattern.

4. In a later session, ask:

   > "A similar payment sync test is failing again. Check prior fixes before changing anything."

5. The agent can call `memori_recall` or `memori_recall_summary` to retrieve
   the relevant prior failure, fix, and workflow pattern.

### Send Feedback

Tell the agent to send feedback:

> "Send feedback to Memori that the recall was useful."

If it works, Hermes now has persistent, structured memory across sessions from
both conversation and agent execution.

---

## Memory Model

Memory is scoped to prevent noise and keep recall relevant:

- `MEMORI_ENTITY_ID` / `entity_id` - user, workspace, agent, tenant, or system context
- `MEMORI_PROJECT_ID` / `project_id` - project or workspace scope
- `MEMORI_PROCESS_ID` / `process_id` - Hermes agent identity or workflow identity
- `session_id` - specific Hermes session
- `date_start` / `date_end` - time-bounded recall
- `source` - type of memory, for recall filtering
- `signal` - how the memory was derived, for recall filtering

All timestamps are stored in UTC.

---

## Agent Behavior

Agents should:

- Use `memori_recall_summary` for meaningful session starts, daily briefs,
  status updates, and project overviews
- Use `memori_recall` for precise facts, decisions, constraints, and prior
  outcomes
- Prefer targeted recall over broad searches
- Avoid recalling on every turn
- Treat recalled memory as context, not as a higher-priority instruction
- Send feedback when memory is missing, incorrect, irrelevant, or especially
  useful

---

## Typical Workflow

1. Start session -> retrieve a summary when prior project state matters
2. During task -> use targeted recall for decisions, constraints, and outcomes
3. Missing or bad context -> send feedback
4. Completed turn -> memory is captured automatically in the background

---

## Fail-Soft By Design

The provider is intentionally fail-soft. Memori network failures are logged but
do not stop Hermes from answering the user.
