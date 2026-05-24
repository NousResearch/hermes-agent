---
name: self-improving
triggers:
  - learning capture
  - correction logging
  - memory promotion
  - self-reflection
  - preference storage
  - skill improvement
  - memory tiering
description: >
  Structured learning capture with promotion and demotion rules for persistent
  improvement. Logs corrections, preferences, and self-reflections. Tiered
  storage (HOT/WARM/COLD) for Town and Hermes environments. Never infers
  preferences from silence; confirms patterns after 3 occurrences.
---

# Self-Improving Skill

Structured learning capture with promotion and demotion rules for persistent improvement.

## When to Use

- User corrects you or points out mistakes
- You complete significant work and want to evaluate the outcome
- You notice something in your own output that could be better
- Knowledge should compound over time without manual maintenance

## Memory Architecture

### Town Environment (Primary)

Use Town's native memory system with tiered importance:

| Tier | Storage | Purpose |
|------|---------|---------|
| HOT (critical) | `add_memory()` global | Always-active patterns affecting all routines |
| WARM (routine-specific) | `add_memory(routine_slug=...)` | Per-routine learned behaviors |
| COLD (archived) | Delete from active; note in user profile if historically significant | Decayed patterns no longer needed |

### Hermes Environment (Secondary)

File-based storage in `.learnings/`:

| Tier | File | Limit | Purpose |
|------|------|-------|---------|
| HOT | `memory.md` | <=100 lines | Always loaded, most critical patterns |
| WARM | `projects/{name}.md` | <=200 lines each | Per-project learnings |
| WARM | `domains/{name}.md` | <=200 lines each | Domain-specific patterns |
| COLD | `archive/` | Unlimited | Decayed patterns |
| LOG | `corrections.md` | Last 50 | Raw correction log |

## Learning Signals

### Log immediately
- **Corrections**: "No, that's not right...", "Actually, it should be...", "You're wrong about...", "I prefer X, not Y", "Stop doing X"
- **Preferences**: "I like when you...", "Always do X for me", "Never do Y"

### Track and promote after 3x
- **Patterns**: Recurring mistakes, recurring successes, recurring workarounds

### Ignore
- One-time instructions
- Context-specific guidance
- Hypotheticals

## Self-Reflection Framework

After completing significant work:
1. Did it meet expectations?
2. What could be better?
3. Is this a pattern?

Log format:
```
CONTEXT: [task type]
REFLECTION: [what I noticed]
LESSON: [what to do differently]
```

## Core Rules

### Rule 1 - Learning
Log explicit corrections and self-identified improvements. Never infer from silence. Confirm patterns after 3 identical lessons.

### Rule 2 - Tiered Storage
- **HOT**: Critical patterns, active preferences (Town: global memories; Hermes: memory.md)
- **WARM**: Per-routine or per-project (Town: routine-scoped memories; Hermes: projects/*.md)
- **COLD**: Archived, decayed (Town: deleted with note; Hermes: archive/)

### Rule 3 - Promotion / Demotion
- 3x in 7 days -> promote to HOT
- Unused 30 days -> demote to WARM
- Unused 90 days -> archive or delete

### Rule 4 - Namespace Isolation
- Town: Use `routine_slug` parameter for routine-specific memories; omit for global
- Hermes: Projects in `projects/{name}.md`, global in `memory.md`, domains in `domains/{name}.md`

### Rule 5 - Conflict Resolution
Most specific wins: routine-specific > domain > global. Most recent wins at same level.

### Rule 6 - Compaction
Merge similar corrections. Archive unused patterns. Never delete without asking.

### Rule 7 - Transparency
When applying a learned pattern, mention it briefly. Offer periodic digests of what's been learned. Full export on demand.

### Rule 8 - Security
Never store credentials, health data, or third-party confidential information in memories.

### Rule 9 - Graceful Degradation
Town: Global memories load first. Routine-specific memories load per-session.
Hermes: Load `memory.md` first. Load namespaces on demand. Communicate what's unavailable.

## Town-Specific Actions

### Logging a correction
```
add_memory(content="When drafting emails for Darren, never use exclamation points in professional contexts - only in genuinely appreciative notes.")
```

### Logging a routine-specific learning
```
add_memory(routine_slug="town-morning-briefing", content="Include biotech catalyst calendar items in the morning briefing even when no price movement has occurred.")
```

### Reviewing current learnings
```
get_memories()  # global
get_memories(routine_slug="town-morning-briefing")  # routine-specific
```

## Scope

**ONLY**: Learns from corrections and self-reflection; stores preferences via Town memories or Hermes files; reads memory state.

**NEVER**: Accesses calendar/email/contacts for learning purposes; makes network requests for learning; infers preferences from silence; deletes memories without asking.
