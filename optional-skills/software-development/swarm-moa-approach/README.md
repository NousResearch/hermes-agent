# Swarm MoA Approach

Optional Hermes skill for using Mixture-of-Agents as a bounded planning/review checkpoint before launching broader agentic work.

## What it adds

- A MoA-worthiness gate so simple tasks stay simple.
- A compact context-pack format for advisor models.
- Schema-bound MoA planning and final-review prompts.
- Clear fallback when MoA is unavailable or too expensive.
- Guardrails that keep execution, side effects, and verification in normal Hermes workflows.

## When to install

Install this if you use Hermes for complex planning, code review, debugging strategy, Kanban/worktree orchestration, or other work where several independent model perspectives can improve the plan before subagents start doing tool-grounded work.

Do not use it for deterministic one-step tasks, source-of-truth operational answers, or sensitive contexts containing raw PII/secrets unless you have redacted the context and explicitly accept model fan-out.

## Usage

After this lands in the official optional catalog:

```bash
hermes skills install official/software-development/swarm-moa-approach
```

Then invoke it naturally, for example:

```text
Use the swarm-moa-approach to plan this refactor.
Run a MoA-backed review before we create the Kanban tasks.
Use a bigger-model planning checkpoint, then execute with normal Hermes workers.
```

The skill assumes Hermes' MoA provider may or may not be configured. If `/moa` is unavailable, it instructs the agent to say so plainly and fall back to a normal planning panel rather than pretending a MoA run happened.

## Safety model

MoA is only the reasoning checkpoint. Real execution still uses the usual Hermes surfaces:

- `delegate_task` for bounded worker analysis
- Kanban for durable/auditable work
- isolated git worktrees for parallel code edits
- parent-owned side effects
- direct verification through tests, diffs, artifacts, and service checks

The skill keeps MoA traces off by default for sensitive contexts and recommends small advisor width/token caps to avoid quiet token explosion.
