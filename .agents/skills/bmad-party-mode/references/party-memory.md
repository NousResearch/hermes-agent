# Party Memory

The room remembers its past sessions with this user and brings them back to life — in character. Memory is per-party and append-only.

Memory is on when the active party's `memory_enabled` is true — the default room follows `{workflow.party_memory}`, a named group its own `memory` flag (both resolved by `resolve_party.py`); ad-hoc inline casts have none. Read on entry and on any mid-session room switch; write through the session.

## Where it lives

One memlog per party: `{workflow.memory_dir}/{active}/.memlog.md`, where `{active}` is the key `resolve_party.py` already returned — the group id (e.g. `code-review-crew`), or `installed` for the default room. The folder is named after the party.

## Read it on entry — distill, don't dump

The log is append-only and grows every session, so don't pull the raw file into the party. Hand a reader subagent the memlog path (`{workflow.memory_dir}/{active}/.memlog.md`) and have it return a compact brief — a few hundred tokens of *where things stand now*, ready to play in character.

Then let the brief shape the room from the first beat, **in character**: behavioral state resumes (a cold pair opens cold, an alliance opens warm), threads pick up, callbacks land when they fit — organically, not recited on sight. Never break the fourth wall: the room *remembers*; it never announces it loaded anything, and forces nothing that doesn't fit.

## When to write

- **When a memorable beat lands** — a clash that shifts the room's temperature, an alliance forming, a line worth a future callback, a decision, an outcome.
- **A floor.** Once a couple of real exchanges are in from the start, even if nothing dramatic happened, capture what it's about and the opening dynamic.

At wrap-up, if the user does signal done, top up with the final outcome and anything memorable not yet captured.

Writes are silent. The room never announces "noted" or "I'll remember".

## What's worth remembering

The test for every entry: *would this color a future session, or make a callback land, or improve the party?* If not, leave it out. A handful of entries, never a recap, never a transcript. keep each entry as brief as possible but usable by future llm.

## New faces

When a character shows up who isn't in the party's roster — cast from an open-cast scene, or one the user adds on the fly — name them in the entry that captures the moment ("<name> turned up and …") so a recurring face can return next session. At wrap-up these are the faces the room offers to keep, saved into the party's roster through `references/create-party.md` (which writes via `bmad-customize`). Until saved they live only in the memlog, and the room re-conjures them from there.

## Write it

```
uv run {project-root}/_bmad/scripts/memlog.py append \
  --workspace {workflow.memory_dir}/{active} \
  --type <dynamic|moment|callback|outcome> \
  --text "<one succinct line, in the room's own read of it>"
```

Add `--by <persona-code>` when a memory belongs to one character. Choose `init` vs `append` from the existence fact you already hold: the entry-read (and, on a mid-session room switch, that room's read) told you whether the memlog exists — `init --workspace {workflow.memory_dir}/{active}` once before the first append when it doesn't, plain `append` when it does. (`init` errors if the file already exists, so don't call it blind.)

If `memlog.py` is unavailable or a write errors, skip it silently and never stall the party on a failed write.

## Forget

The memlog is append-only by design — no surgical delete. To wipe a party's memory, delete its folder (`{workflow.memory_dir}/{active}/`). To correct a wrong memory, append a new entry that supersedes it; the room reads the latest state.

Keep entries sparse. The distilled read keeps the *room* lean no matter how big the log gets, but the on-disk file still grows append-only.