"""Native combined-review guidance adapted to the employee's three knowledge stores."""
PROMPT = """Review the conversation above and update three things:

**Memory**: durable facts about people and the organization. Pick the right store for each fact:
• Personal memory (memory tool, target='user'): who the named person is — identity, role, preferences, communication and work style, and expectations about how you should behave. Attribute each fact to the correct person using the sender label from the conversation.
• Shared memory (memory tool, target='memory'): facts about the organization and environment that apply regardless of who is speaking — project conventions, paths and endpoints that matter.
One fact goes to ONE store, never both. Use only enabled memory targets; if memory is unavailable, skip it. Procedures belong in connection manuals or responsibilities, not memory. Replace the entry a fact supersedes; never add a contradicting one alongside.

**Connection manuals**: how to operate each connected service ({profile_home}/connections/<service>/manual.md).
• Capture non-trivial techniques, fixes, workarounds, and tool-usage patterns a future session would otherwise rediscover.
• If a manual consulted this session was wrong, missing a step, or outdated, patch it now.
• Keep service mechanics here; the purpose, authority, and state of the work using the service belong in its responsibility.
Prefer the earliest action that fits: 1. update the relevant manual consulted this session; 2. update an existing service manual; 3. add a topical support file — references/ for provider quirks, recipes, and condensed domain notes, scripts/ for re-runnable probes — with a one-line pointer from manual.md. Extend an existing topical file before creating one. Create a missing service manual only when the conversation establishes verified access; a service mention or old listing does not prove access. Otherwise file the lesson under the responsibility whose work used the service, or leave it unfiled.

**Responsibilities**: how to carry out the work you own, including its charter, working knowledge, and current state ({profile_home}/responsibilities/<name>/).
• User corrected your style, tone, format, legibility, verbosity, approach, or sequence of steps for this work. Frustration is a first-class signal: embed the stated correction in the responsibility that governs the task so the next run starts fixed. Do not infer a preference from the person's mood.
• A non-trivial technique, fix, workaround, or debugging path emerged. Capture the reusable method in the responsibility that owns the work; service mechanics belong in the connection manual.
• Knowledge consulted this session was wrong, missing, or outdated. Edit the passage that misled, not an 'UPDATE: actually...' appended underneath it.
• Work moved and STATE.md does not say so yet. Update the handoff for the next run, not a log of this session. State describes what is true now and what the next run needs.
Prefer the earliest action that fits: 1. update the relevant responsibility consulted this session; 2. update an existing responsibility covering the work; 3. add a topical reference or reusable script with a pointer from the owning document. Create a responsibility only when no existing one covers an area of work you own, not for a one-off task. Before creating or reshaping a responsibility beyond state, read {guides_root}/responsibility-authoring/guide.md unless it was already read in the conversation above. Preserve its authority boundaries. A schedule or webhook scope cannot be edited here — record the proposed change in STATE.md.

Be ACTIVE — most sessions produce at least one manual or responsibility update. A pass that does nothing is a missed learning opportunity, not a neutral outcome.

Write reusable knowledge, not a collection of session narratives:
• Procedure first, when writing a procedure: steps in the order they are done, with concrete commands, tool calls, and decision points. Lessons and pitfalls attach to the step they affect. A future session should be able to follow it and produce what the user wants on the first try. Responsibility charters and state instead describe the work, its authority, and what is true now; follow the authoring guide.
• A pitfall is a generalizable rule plus one clause of WHY — the mechanism. Write the instruction that prevents the mistake, not a narrative of what happened this session.
• Reusable rules must stand without the incident behind them: no PR/issue numbers, dates, ticket IDs, or quoted user text unless a short quote is the clearest statement of the rule. Exact-item identifiers and dates belong in state when needed to carry out the work.
• The same lesson learned twice is ONE rule. Search the package and its references for the rule already stated; strengthen or clarify it rather than appending a second copy.
• Do not duplicate what the environment already teaches: repository AGENTS.md files, tool descriptions, or other always-loaded context. Save the workflow and its hard-won pitfalls, not a codebase map or a tool's parameter list.
• Keep standing service instructions in manual.md and responsibility duties and authority in the charter. references/ holds topical depth needed only sometimes — a decision table, recipe, or domain note. Name reusable files by topic, not by date or incident; do not create a references/ file per session.
• Fix wrong knowledge in place. Retire superseded content and consolidate duplicates before adding to a file near its budget; do not create spillover files to evade limits. One fact lives in one place, with pointers where other packages need it.

Read-before-write: this is a replayed conversation and other passes may have reviewed it already. Read each existing target's current text with read_file during this review before changing it; content quoted earlier in the conversation is not a fresh read. Before creating any file, search the folder for an existing artifact covering the same territory. Extend it even if you would have named it differently. Use write_file or patch within the permitted knowledge scope.

User-preference routing: a task-specific preference belongs in the responsibility governing the task; a service-specific operating correction belongs in its connection manual. Personal memory holds cross-cutting preferences no package owns. Save each preference in exactly ONE place, never in both memory and a package.

Do NOT capture as durable rules or procedures (these become persistent self-imposed constraints that bite you when the environment changes):
• Environment-dependent failures: missing binaries, fresh-install errors, post-migration path mismatches, 'command not found', unconfigured credentials, or uninstalled packages. The user can fix these — they are not durable rules.
• Negative claims about tools or features ('browser tools do not work', 'X is broken'). These harden into refusals cited against yourself for months after the actual problem was fixed.
• Session-specific transient errors that resolved before the conversation ended. If retrying worked, the lesson is the retry pattern, not the original failure.
• One-off task narratives. A request to 'summarize today's market' or 'analyze this PR' is not a class of work that warrants a new package.
• Unresolved failures: if the session ended WITHOUT finding a working method, do not write the failed attempts up as a 'reliable workflow' or 'recommended approach'. Either leave them unfiled or, only if independently confident of a real working alternative, capture ONLY that alternative — never guesses or dead ends dressed up as best practice.
• One-off exceptions as standing rules. A release, approval, or defect acceptance for a named item or batch is exact-item state — a STATE.md line naming those items and the approved revision, spent once applied — never a standing rule, narrowed review scope, or relaxed quality gate unless the user said it holds going forward.

If a tool failed because of setup state, capture the FIX (install command, config step, env var) in the relevant service manual, or the responsibility whose work used it — never 'this tool does not work' as a standalone constraint. Current blockers may belong in responsibility state; they are not permanent operating rules.

Act on whichever of the three areas has real signal. End by saying what you filed and where, counting only writes whose tool results succeeded — a failed write was not filed. If genuinely nothing stands out, say 'Nothing to file.' and stop — but do not reach for that conclusion as a default."""
