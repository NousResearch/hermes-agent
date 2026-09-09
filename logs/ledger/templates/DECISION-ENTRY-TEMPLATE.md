<!--
Add to logs/ledger/decisions/DECISION-LOG.md: a full block under "## Open", and a
row in the "Register (quick scan)" table. On close, move the block under
"## Resolved" with a Decided/Deferred/Dropped status. Rules: logs/ledger/README.md

Use DECISION- (not ERR-) when the item is an open *judgment call* — a choice
between defensible options — rather than a fault to be fixed. "Rebrand vs thin
downstream", "keep or delete the attic clone", "rename the `hermes` command or
not" are decisions. A broken build, a leaked secret, a bad merge are errors.
-->

### DECISION-YYYY-MM-DD-NNN — <area> — <the question, phrased as a question>

- **Opened:** YYYY-MM-DD · **Base:** hermes@<sha> (<N> behind upstream/main)
- **Run:** RUN-YYYY-MM-DD-NNN
- **Source:** <AUDIT- id / CHG- / manual observation>
- **Confidence:** Confirmed Fact · Field-Reasoned · Unverified   <!-- confidence in the framing below: that these are the real options and the tradeoffs are stated correctly. See README. -->
- **Supersedes:** <prior ERR-/DECISION- id this was migrated or split from, or —>
- **The call:** <what actually has to be decided, in one or two sentences>
- **Options:**
  - **A — <name>:** <what it means> · <cost / benefit / who it affects>
  - **B — <name>:** <what it means> · <cost / benefit>
  - **C — <name>:** <…>   <!-- omit if only two -->
- **Leaning:** <recommended option + one line why, or "none — genuinely open">
- **Blocking:** <what cannot proceed until this is decided, or "nothing — informational / can wait">
- **Owner:** <who makes the call>
- **Status:** OPEN

<!-- On resolution, append: -->
- **Decided:** YYYY-MM-DD — chose <option>. <one-line rationale>. Implementing change(s): CHG-YYYY-MM-DD-NNN[, …].
- **Status:** DECIDED   <!-- or DEFERRED (revisit later, with a trigger) / DROPPED (no longer relevant, say why) -->

<!-- Register row:
| DECISION-YYYY-MM-DD-NNN | YYYY-MM-DD | <area> | <question> | OPEN | — |
-->
