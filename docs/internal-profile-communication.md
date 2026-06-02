# Internal profile communication toolbox

Hermes profiles should discuss with each other through direct/internal channels. Kanban is the durable coordination ledger, not a chat transport.

## Rule

Do not create Kanban contact tasks merely to ask, notify, ping, or refine with another profile. Use Kanban for durable work items, dependencies, blockers, evidence, review handoffs, and post-discussion outcomes.

## Direct profile runner path

Use the named profile directly when you need a quick internal answer or instruction:

```bash
hermes -p <profile> chat -q '<question or instruction>' --toolsets safe
```

Notes:

- This starts the target profile directly and does not create a Kanban task.
- Add `--skills <skill-name>` when the target must load a specific policy or domain skill.
- Use a bounded prompt and ask for a concise response so the interaction remains discussion, not a durable work assignment.

## Structured internal message path

When a validated private profile room/channel exists, send a structured message instead of creating a contact task:

```bash
printf '%s\n' '<structured note>' | hermes send --to <target> --file - --subject '[internal:<profile>]'
```

Agents with the messaging tool can use:

```python
send_message(
    target='<target>',
    message='[internal:<profile>]\nContext: ...\nQuestion: ...\nNeeded by: ...',
)
```

Use `hermes send --list` or `send_message(action='list')` before sending to a specific person/channel when the target is ambiguous.

## Matrix escalation and guards

- Human-facing Matrix participation is for decisions or actions that genuinely require Yunuen or another human.
- Before contacting a human from a dedicated Matrix room, validate profile room isolation and gateway activation with the coordination validators.
- If a Matrix profile-room send is blocked, unvalidated, or unavailable, do not create a Kanban contact task as a workaround. Use the direct profile runner path, or block only when a concrete human action is required.

## What to record on Kanban

After direct discussion, write a Kanban comment or follow-up task only for durable outcomes: decisions, evidence, dependencies, blocker status, review feedback, or concrete work that should survive restarts and be audited.
