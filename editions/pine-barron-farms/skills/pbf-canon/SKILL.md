---
name: pbf-canon
description: "Load the Pine Barron Farms studio canon packet into context, with a load receipt (path + sha256) or a plain not-found warning. Run at session start before any canon-dependent work."
version: 1.0.0
author: North Forge
license: MIT
metadata:
  hermes:
    tags: [pine-barron-farms, canon, context, session-start]
---

# Pine Barron Farms — load canon

The studio **canon packet** (characters, locations, props, plates, episode
history, open patches, the full generation method) is deploy-time content. It is
**not** in this repository — an admin drops `PINE_BARRON_FARMS_CANON.md` into the
profile's `canon/` folder on the provisioned drive. This skill loads it.

## Do this first, every session

Run the loader with the terminal tool, before your first substantive reply:

```
python "${HERMES_SKILL_DIR}/../../canon/load_canon.py"
```

(If `python` is not found, try `python3`. The script is standard-library only and
always exits 0 — a missing packet is a state, not an error.)

Take its stdout as-is and treat it as canon for the rest of the session.

## Reading the output

- **`canon loaded: <path>, sha256=<hash>, <n> bytes`** followed by the packet
  between `--- BEGIN PINE BARRON FARMS CANON ---` / `--- END …` markers — that
  block **is the canon**. Follow it exactly (SOUL.md: *"Canon is law"*). Never
  invent, "improve", or quietly patch a fact.
- Ends with **`[canon excerpt truncated …]`** — the packet is larger than the
  budget. Before any canon-dependent answer, read the full file at the path in the
  receipt with the file tool.
- **`WARNING: canon packet NOT FOUND`** — there is no studio canon on this drive.
  Run on the SOUL.md persona and method only. If the user asks anything
  canon-dependent (a character, a plate, an episode, a patch), tell them the
  packet is not loaded and ask them to add it; do not guess.

## Refreshing mid-session

If an admin drops in or updates the packet during a session, run the loader again
— the receipt's `sha256` changes when the packet does, so you can tell.
