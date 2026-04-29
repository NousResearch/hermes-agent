PARALLEL INVESTIGATION SWARM for:

$ARGUMENTS

Deploy subagents to investigate multiple files or areas simultaneously. Each subagent runs in a separate context window, keeping the main thread clean.

STEP 1 — DECOMPOSE THE TASK:
Break $ARGUMENTS into independent investigation areas. Each area should be:
- Self-contained (can be investigated without results from other areas)
- File-scoped (touches a specific set of files that don't overlap with other areas)
- Question-driven (has a specific question to answer)

STEP 2 — SPAWN SUBAGENTS:
For each investigation area, spawn a subagent with a specific mission:

Agent A: "{specific question about area A}"
  Files: {list of files to investigate}
  Report: {what to report back}

Agent B: "{specific question about area B}"
  Files: {list of files to investigate}
  Report: {what to report back}

Agent C: "{specific question about area C}"
  Files: {list of files to investigate}
  Report: {what to report back}

STEP 3 — COLLECT RESULTS:
Wait for all subagents to complete. Collect their findings.

STEP 4 — SYNTHESIZE:
Combine all subagent findings into a unified report:
- Where do findings agree?
- Where do findings conflict? (investigate conflicts)
- What was found that no single-file investigation would have caught?
- What questions remain open?

RULES:
- Maximum 5 subagents per swarm (context budget)
- Each subagent should take < 2 minutes
- If areas overlap in files, merge them into one subagent
- Subagents should use grep and file reads, not edits
- The main thread synthesizes — subagents investigate
