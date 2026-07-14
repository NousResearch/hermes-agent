---
name: roundtable
description: "Multi-agent roundtable discussion — topic-driven multi-round debate with convergence detection and conclusion generation"
version: 1.0.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [discussion, multi-agent, collaboration, debate, roundtable]
    related_skills: [kanban-worker, kanban-orchestrator]
---

# Roundtable Discussion Skill

## Overview

Enable multiple agents to participate in structured, multi-round discussions
around a topic. Each participant brings their unique role perspective, and the
system tracks convergence toward consensus.

**Core value**: Turn "one agent working alone" into "a team having a meeting."

## When to Use

- **Tech design review**: product, frontend, backend, architect debate approach
- **Competitive analysis**: product, marketing, design compare alternatives
- **Bug root cause analysis**: backend, ops, test triangulate the issue
- **Product requirements**: product, design, dev align on scope
- **Architecture decisions**: architect, backend, frontend, devops choose stack

## Prerequisites

Enable the `roundtable` toolset in the profile config:

```yaml
toolsets:
  - roundtable
```

Or pass `enabled_toolsets: ["roundtable"]` when spawning an agent.

## Tools

| Tool | Purpose |
|------|---------|
| `roundtable_init` | Create a discussion with topic + participants |
| `roundtable_speak` | Record a participant's speech |
| `roundtable_read` | Read discussion history |
| `roundtable_status` | Check status + convergence metrics |
| `roundtable_summarize` | Get structured data for conclusion doc |
| `roundtable_end` | Conclude or cancel a discussion |
| `roundtable_list` | List all discussions |

## Coordinator Flow

The coordinator (initiator) drives the discussion by orchestrating participants.

### Step 1: Create the Discussion

```
roundtable_init(
    topic="Database selection: PostgreSQL vs MySQL vs TiDB",
    context="Our e-commerce system needs high-concurrency read/write, 1TB+ data",
    participants=[
        {"profile": "bingge", "role": "Product Director", "perspective": "Focus on UX", "display_name": "Bing"},
        {"profile": "mafei", "role": "Tech Lead", "perspective": "Focus on feasibility", "display_name": "Fei"},
        {"profile": "xiaosu", "role": "Designer", "perspective": "Focus on data display", "display_name": "Su"},
    ],
    max_rounds=3,
    speech_order="fixed"
)
→ returns {discussion_id, ...}
```

### Step 2: Opening Statement (Round 0)

```
roundtable_speak(
    discussion_id="rt_xxxxxxxx",
    participant="coordinator",
    content="Today we're discussing database selection..."
)
```

### Step 3: Multi-Round Discussion

For each round, read history then delegate to each participant:

```
# Read current state
history = roundtable_read(discussion_id="rt_xxxxxxxx")

# For each participant, delegate a sub-agent
delegate_task(
    goal="Participate in roundtable discussion as [ROLE]",
    context="You are participating in a roundtable discussion.\n\n"
            "Topic: {topic}\n"
            "Your role: {role}\n"
            "Your perspective: {perspective}\n\n"
            "Discussion history:\n{formatted_history}\n\n"
            "Please share your观点 using roundtable_speak. "
            "Keep it 200-500 words. Reference others' points if relevant."
)
```

### Step 4: Check Convergence

After each round:
```
roundtable_status(discussion_id="rt_xxxxxxxx")
→ check convergence_score, consensus_points, disagreement_points
```

### Step 5: Generate Conclusion

```
summary = roundtable_summarize(discussion_id="rt_xxxxxxxx")
→ Use the structured data to write a Markdown conclusion document
→ Save to the output_path specified during init
```

### Step 6: End Discussion

```
roundtable_end(discussion_id="rt_xxxxxxxx")
```

## Participant Prompt Template

When delegating to a participant sub-agent, use this template:

```
You are participating in a roundtable discussion.

## Discussion Info
- Topic: {topic}
- Context: {context}
- Current Round: Round {current_round} / {max_rounds}
- Your Role: {role} ({display_name})
- Your Perspective: {perspective}

## Discussion History
{formatted_history}

## Your Task
From your role's perspective, share your观点 on this topic.
- You may引用 or respond to other participants' statements
- Keep it concise and powerful, 200-500 words
- If you agree with a point, explicitly state your agreement
- If you disagree, explain why and propose alternatives

After speaking, call roundtable_speak to record your statement.
```

## Convergence Detection

Each round is evaluated for convergence:

| Metric | Formula | Meaning |
|--------|---------|---------|
| Consensus | Points multiple participants agree on | Alignment |
| Disagreement | Points participants disagree on | Conflict |
| New Point | New topics raised this round | Scope expansion |
| Score | consensus / (consensus + disagreement) | Overall alignment |

**Termination conditions:**
- Convergence score > 0.8 → high consensus, wrap up
- Max rounds reached → prevent infinite discussion
- Coordinator manually ends → emergency stop
- All participants vote to end → democratic close

## Conclusion Document Format

```markdown
# Roundtable Conclusion: [Topic]

## Summary
- Participants: Product(Bing), Design(Su), Dev(Fei)
- Rounds: 3
- Date: 2026-05-20

## Consensus Points
1. [Point 1]
2. [Point 2]

## Disagreement Points
1. [Point 1] - Various perspectives

## Action Items
1. [ ] [Action 1] - Owner: xxx
2. [ ] [Action 2] - Owner: xxx

## Detailed Transcript
### Round 1
- **Product(Bing)**: ...
- **Design(Su)**: ...
- **Dev(Fei)**: ...

### Round 2
...
```

## Data Storage

- **Database**: `~/.hermes/roundtable.db` (independent from kanban.db)
- **Conclusion docs**: Configurable via `output_path`, defaults to project docs dir
- **ID format**: `rt_` + 8 hex chars (e.g., `rt_a1b2c3d4`)

## Integration with Kanban

Discussions can be linked to kanban tasks:

```
# After conclusion, add as task comment
kanban_comment(task_id="t_xxx", body="Roundtable conclusion: {conclusion_path}")
```

## Pitfalls

1. **At least 2 participants required** — A discussion needs multiple viewpoints
2. **Participant must be registered** — Only profiles listed in `participants` can speak
3. **Round 0 is opening** — Coordinator speaks first, then round 1 begins
4. **Auto-conclude on max_rounds** — Discussion ends automatically when max rounds exceeded
5. **Independent database** — roundtable.db is separate from kanban.db; don't mix paths
6. **No LLM in summarize** — `roundtable_summarize` returns raw data; the coordinator agent writes the conclusion
