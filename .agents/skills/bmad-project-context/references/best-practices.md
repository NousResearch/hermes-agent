# What belongs in a repo's agent instructions

Rules for deciding what goes in the block, for judging what a repo already has, and for explaining both to the user.

## The test

Not *could an agent derive this* but *what does it cost when it doesn't*: how much exploration finding it takes, how likely the agent is to search the right place in time rather than guess, whether it is available at the point of use or only after the mistake, what a retrieval failure costs — a wasted search, or corrupt data — and whether it is a rule that must hold or a detail the code already shows.

A line that stops the same rediscovery every session earns its place, derivable or not. A stored copy of what the agent reads more accurately first-hand does not — it rots, and it is charged every session.

## Admit

- **Policy the code cannot express** — branch rules, frozen and protected paths, generated files, secrets, security and compliance. Stated by a human or read off an enforcing config, never inferred.
- **What a config file cannot say about running the project** — the root test script does nothing in this workspace, integration tests need a service up first, the suite takes eleven minutes so iterate on single files, the `Makefile` is the real entry point and `package.json` is vestigial, CI runs a typecheck the test script does not. An invocation the obvious guess gets right is already stated in `package.json`, `Makefile`, `pyproject.toml`, or CI config and does not earn a line — the correction, the caveat, and the right command to use do.
- **Conventions that differ from ecosystem defaults.** An agent follows the norm unless told otherwise, so only the divergences earn a line. Command invocations count: when the obvious command is wrong here — a bare-repo prefix, a required wrapper — the exact working invocation earns a line, and no observed mistake is needed to admit it.
- **Pitfalls with observed evidence** — a recorded lesson, the maintainer's recollection, the same mistake fixed repeatedly in history, or one this session made and caught. A repo yields hundreds of trap-looking facts and none of them predict real mistakes; only observed behavior does. A surprising scan finding is a question to ask, not a line to write.
- **Runtime behavior invisible from the repo** — replaying webhooks, lying health endpoints, environment quirks — once a human confirms it.
- **Cross-component rules**, admitted when getting one wrong in one file breaks something elsewhere — what must stay true across parts of the system the agent cannot see from the file it is editing: who owns what, how data must flow, what order a pipeline runs in. "Writes go through the dispatcher; direct store mutation skips the transaction." "The importer is two passes — validate every row, then commit; never write inside the parse loop." A six-line map of who owns what. Never an inventory written for completeness; the exclusions below still bind.
- **Required tool and runtime versions**, read from the project files that declare them, never from this session's environment — which answers faster, and wrongly, so the mistake arrives before the search.
- **Entry points and pointers** to where work lands.

Prefer prohibitions to advice, and name the permitted alternative in the same line.

## Exclude

| | Why |
|---|---|
| Repo overviews, directory trees, stack lists | Derived fresh, more accurately; stored copies rot |
| Anything included for being interesting | Interest is not need |
| Style rules an agent self-enforces | Belongs in a formatter, linter, hook, or CI check — propose the check instead |
| Platitudes | Already the default |
| Transcribed command lists whose obvious invocation is already right | Read from `package.json`, a `Makefile`, or CI config; a copy drifts the moment a script is renamed. The right command to use, and any command the obvious guess gets wrong, are admitted above |
| Pasted code, changelog content, fast-changing facts | Stale immediately |
| Aspirational state | Describe what is; intent belongs in specs |
| History and edit narration | Git holds it; state present truth |

## Retire

A policy or pitfall goes only when the thing it guards is gone, or the user retires it. Nothing failing lately is not evidence — a working rule erases its own evidence. Any other existing instruction goes only on one of the four grounds under "Judging an existing file".

Every line faces one question at each write: would removing it change agent behavior? If no, cut it — but for a line a human wrote, that answer only opens a candidate; a ground still has to carry it.

## Size

Every line is paid in every session, and instruction-following degrades as the loaded set grows. Count what other always-loaded files add. Over budget means cut the weakest lines or move them behind a trigger — never raise the budget. Ten lines of evidence means ten lines.

An adopted file must fit the budget too, but shrinking it works differently. Move the weakest instructions out first — into a child file, a linked doc, or a hook or check that enforces them. Deleting still needs one of the four grounds. If the file is still too big and no ground justifies another deletion, show the user and let them decide — an over-budget file they chose beats a gutted one they didn't. "Keep it small" disciplines what this skill writes, never what the maintainer already wrote.

## Retrieval

An index the agent must choose to fetch gets skipped; one already in context does not. Keep everything load-bearing in the block. A pointer out of it names a trigger the agent can observe — a path, a file type, a named task — never one it must judge ("when the task is complex") or track about itself ("before your first edit").

Rules bounded to a directory can go in a nested `AGENTS.md` there, attached by location rather than by pointer — but only when they are subtree-exclusive and substantial, the split materially reduces the root block, the user approves it, and **loading is verified for every harness in use**. Even with verified loading, keep a rule at the root when it must apply before a session enters that directory or when breaking it can affect work outside the child. Check, never assume: several harnesses build the instruction chain once at session start, root down to the working directory, so a nested file is invisible to the session that later edits into that subtree. Unverified means path-qualified lines at root instead — "in `src/importer/`: ..." — cheaper than a file nobody loads.

Use a linked file only when the trigger is not a path.

## Maintain

- Re-check that caveats still hold — a slow suite that got fast, a workaround for a bug that was fixed.
- Diff deletions and renames since the verified SHA against every line.
- Record provenance in the block so the next run knows what it is diffing from.
- Capture mistakes when they happen, not at review time. One occurrence is a note; recurrence earns a line.
- Route anything mechanically preventable to a hook, lint rule, or CI check. A check that lands deletes its line.

## Repo or home directory

This block belongs committed: shared by the team, consistent across machines, versioned with the code it constrains.

Two things belong in the user's global agent config instead — rules repeating across all their projects, and personal preferences that are theirs rather than the team's.

## Judging an existing file

Every instruction a human wrote is presumed intentional: someone paid for it, usually by watching an agent fail. The file is the baseline being improved, not raw material. Keep its phrasing where it works, and carry each instruction through a ledger entry — `retain | rewrite | relocate | automate | delete`, opened at retain or rewrite — so the user sees where all of it went.

**Deletion needs one of four grounds:**

1. **Stale or incorrect** — the referent is gone, or the instruction was never true; the evidence is named.
2. **Mechanically enforced** — a hook, linter, formatter, or CI check already fails the violation named by the instruction. A tool that only covers the same files or topic does not enforce the instruction.
3. **Harmful or contradictory** — it points agents at the wrong thing, or it contradicts another live instruction and loses the reconciliation.
4. **The user approved this deletion** — asked as a line item, never implied by approving a replacement block.

Grounds 1–3 are evidence the run carries itself, and ride the block approval; ground 4 is the ask-first path everything else takes. Nothing else deletes. Brevity is not grounds, nothing failing lately is not grounds, "the agent could derive it" is not grounds, and **"it is discoverable somewhere in the repository" is never, alone, grounds** — that is the reasoning that empties good files. Content the exclusions table rejects — a directory tree, a stack list, pasted code — has no ground of its own: propose the deletion and let it land under ground 4, asked rather than assumed.

Report, in this order: what is unverifiable or stale, what is missing against the sections above, what is already good, and the ledger, every relocation, automation, and deletion itemized. Recorded lessons are maintainer testimony — kept by default, challenged only with evidence that the thing they name is gone or wrong.
