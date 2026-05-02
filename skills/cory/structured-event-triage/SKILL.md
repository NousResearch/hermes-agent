---
name: structured-event-triage
description: Interpret webhook or structured event payloads in product and project context instead of treating them as self-explanatory.
---

# Structured Event Triage

Use this skill when the source is a webhook or other structured event.

Rules:
- Read the event as a signal that still needs project context.
- Do not assume the downstream workflow solely from event type.
- Differentiate between advisory follow-up, execution preparation, governed change, and KM promotion.
- Escalate ambiguity instead of guessing when repository or project mapping is uncertain.

Output expectations:
- Explain what happened and why it matters.
- Propose the most plausible request type and workflow state transition.
