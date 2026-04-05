# Paperclip Bridge Skill

Enables Hermes to create goals and tasks in Paperclip directly from Telegram DMs.

## When to Use

Trigger when Seb says things like:
- "build X for the team"
- "get the engineers working on X"
- "create a task for X"
- "we need to audit X"
- Any request that needs the full engineering org to execute

## How It Works

1. Seb sends request in Telegram DM
2. Hermes asks clarifying questions if needed:
   - What's the goal/objective?
   - What does success look like?
   - Priority: P0, P1, or P2?
3. Hermes creates a goal or issue in Paperclip
4. On next Hermes (CEO) heartbeat, she picks it up and delegates to the team
5. The engineering pipeline runs autonomously

## Examples

**User:** "Audit FalconConnect for dead code"
**Hermes:** "Got it. This is important for quality. Setting priority to P1 (this week). Creating goal now..."
**Result:** Goal appears in Paperclip → Hermes delegates to CTO → Architect reads codebase → findings flow through pipeline

**User:** "Write Facebook ad copy for VGLI"
**Hermes:** "Creating a copywriting task for the team..."
**Result:** Issue created → Copywriter assigned → copy delivered in 5 min

## Configuration

Environment variables needed (auto-injected):
- `PAPERCLIP_API_URL` — base URL of Paperclip instance
- `PAPERCLIP_API_KEY` — agent's API key (Hermes stores this in .env)
- `PAPERCLIP_COMPANY_ID` — company to create goals/issues in (Falcon Financial)
