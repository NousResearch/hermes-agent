# Dialog Template Usage

## Quick Start

**Copy to project:**
```bash
cp -r workflows/wds-1-project-brief/templates/project-brief-dialog projects/{{slug}}/dialog
```

**Update as you progress:**
- Complete each file when the corresponding PB step finishes
- Update README.md progress tracker
- Append to decisions.md whenever key decisions are made

---

## What to Capture

### DO:
- Key questions + user responses (not full transcript)
- Signal-based follow-ups that revealed insights
- Reflection checkpoint (synthesis + confirmation + corrections)
- Final outputs (vision, positioning, etc.)
- WHY decisions were made

### DON'T:
- Verbatim transcripts
- Procedural agent actions
- Implementation details
- Repetitive exchanges

---

## Mandatory Checkpoints

**Document EVERY reflection:**
1. Agent's synthesis (2-3 sentences)
2. User confirmed or corrected?
3. What was misunderstood? (if corrected)

---

## Integration with Steps

**Each step file should mandate:**

```markdown
## Design Log Update

Before marking complete:
1. Update `dialog/{{step}}-{{name}}.md`
2. Document reflection checkpoint
3. Record final synthesis
4. Mark complete in `dialog/README.md`
```

---

## File Sizes

All dialog files: 65-86 lines (well under 100-line target)

---

## Design Log (Meta-Level)

**For multi-session work**, agents should use the design log for state tracking and `_progress/agent-experiences/` for session insights.

**Location:** `{{root_folder}}/_progress/00-design-log.md`

**Update Protocol:**
1. Complete current task
2. Update design log with changes
3. Show git diff to user
4. Record session insights in `_progress/agent-experiences/` if needed

---

## Purpose

Create transparent record of discovery conversations so future agents (and humans) understand WHY decisions were made, not just WHAT was decided. The design log provides this continuity across sessions.
