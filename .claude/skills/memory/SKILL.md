---
name: memory
version: "1.0.0"
description: Session state backend for WDS. Called by wrap, start, and handoff tools — never directly by users. Writes to progress/ in the project repo.
agents: [saga, freya, mimir]
---

# WDS Memory — File Backend

Handles all persistent state for WDS sessions. Two operations: save and load.

State lives in `progress/` at the project root. This folder is project-scoped — not global, not per-machine.

---

## save

**Called by:** wrap (step 3), handoff (step 3)

**Input:**
- `agent_id` — the agent whose state is being saved (saga, freya, mimir)
- `data` — the compiled state block (Wrapped, Context, Plan, Next, Learned, Spec Sync fields)

**Steps:**

1. Ensure `progress/` exists at the project root. Create it if not.
2. Write `progress/[agent_id].md` with the data block exactly as provided.
3. Return: `saved progress/[agent_id].md`

---

## load

**Called by:** start (step 2)

**Input:**
- `agent_id` — the agent whose state is being loaded

**Steps:**

1. Check if `progress/[agent_id].md` exists.
2. If found: return the full file content.
3. If not found: return nothing. The calling tool handles the fresh-start case.

---

## Notes

- `progress/` should be in `.gitignore`. It is machine-local session context, not project content.
- Both save and load are synchronous file operations — no tokens, no async, no IDs.
- This is the canonical storage backend. State is local to the machine and project — not synced, not shared.
