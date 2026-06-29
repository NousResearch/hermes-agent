# Working-State Patterns

How a skill's work survives across turns and context compaction. This is a design axis of its own, separate from persona, intent modes, and degradation, and it has more than one answer. Load this file when building or revising a multi-turn skill that builds something, or when a skill already carries a `.memlog.md` or a structured working artifact.

## The choice

A multi-turn skill that builds something has to hold state somewhere. Pick by the shape of the work, not by default.

| Strategy | Holds | Choose when |
|---|---|---|
| memlog | the *why* — decisions, directions, rejected alternatives, conflicts | the deliverable is prose or a document and its value includes reasoning that must survive revisits and surface conflicts on update |
| Structured working artifact | the *what* — work-in-progress in a custom schema that transforms into the output | the work decomposes into a natural intermediate the user iterates on directly, which later becomes the deliverable |
| Both | the what and the why | long, revisable creative or engineering work where construction state and rationale both matter |
| Neither | nothing across turns | a one-shot transform, a stateless utility, or a purely conversational skill where the input/output contract or the live conversation is the state |

memlog and the structured artifact are not rivals. memlog is *meta* about the work — a decision trail beside the deliverable. The structured artifact *is* the work — state lives inside it, so continuity comes from re-reading the artifact rather than a side log. A skill uses either, both, or neither.

## memlog: the decision trail

For a skill whose value includes the reasoning behind the deliverable. The memlog carries identity across sessions, keeps the agent from railroading the user, surfaces conflicts on update, and creates an audit trail when the user overrides a past call. A skill that needs it looks fine on the first pass and falls apart on revisit without it.

The memlog is typed, append-only, and written through `{project-root}/_bmad/scripts/memlog.py` to a `.memlog.md` file beside the primary artifact. The model never edits or re-reads it mid-session; it appends one typed entry at a time and trusts the one-line JSON ack. The cycle is capture (append as decisions and directions land), distill (at finalize, account for every entry), and project (read the whole log once on resume or when building a summary).

### Entry types and the CLI

The memlog CLI is runtime-installed at `{project-root}/_bmad/scripts/memlog.py`; a built skill calls it there and bundles no copy of its own. The `{project-root}` token resolves at runtime, so the same invocation works from any skill's root.

- `init --path <file>` creates the log.
- `append --path <file> --type <type> --text <text>` adds one typed entry; `<type>` is one of `decision`, `direction`, `assumption`, `gap`, `note`, `event`.
- `set-complete --path <file>` marks the workflow done.

Each command prints a one-line JSON ack (`{"ok": true, ...}`). The write is atomic (temp file, fsync, rename) so an interrupted run never half-writes an entry, and there is no edit or remove subcommand by design, because history is never rewritten.

### Workspace layout

Files live in a single folder rooted at the primary artifact. When the artifact is a single document, the workspace is the document's containing folder and the log sits as a peer. When the artifact is itself a folder (a built skill, a generated module), the workspace IS that folder and `.memlog.md` sits beside the primary file such as `SKILL.md`. Either way the workspace exists from the moment intent is confirmed, so the user knows the path immediately and state lives on disk rather than in the conversation.

### Resume, update, validate, finalize

- **Resume**: on activation, glob for `.memlog.md` (never `.decision-log.md`). If found, surface it, read it once to rebuild state, and offer to resume. The single read recovers full context regardless of compaction; after that the workflow resumes append-only.
- **Update**: read the memlog first; the change request enters as a signal against the standing record. If it contradicts a prior decision, surface the conflict before applying. Every change gets a new `decision` entry, and an override also records the rejected reasoning.
- **Validate**: read the memlog first; challenge the artifact against the standards the user themselves set, not a generic rubric.
- **Finalize**: distill the memlog — every meaningful entry is either captured in the artifact or explicitly set aside as process noise — then call `set-complete`.

### Treatment style

State the principle once where it first applies, typically inside the Create intent as a single clause ("write the primary skeleton and init `.memlog.md` in the workspace; the memlog is canonical process memory"). Mention reads at the moments that matter: Update reads before changing decisions, Validate before critiquing, Finalize distills at handoff. That is the entire treatment. Do NOT open with a "memlog discipline" enumeration of what to log, write a separate `## Workspace` meta-section, include a tree diagram, or split workspace creation into "for new" and "for existing" sub-sections — "init if absent, append if present" is one sentence. `bmad-product-brief` is the canonical example: about five sentences total, threaded through Create, Update, Validate, Constraints, and Finalize.

## Structured working artifact: the work-in-progress itself

Some skills need no decision trail because the work has a natural intermediate form that carries its own state. The skill builds a custom file with its own schema — story beats, an outline, character sheets, a shot list, a spec kernel, a requirements matrix — that the user reads and edits directly, and that later transforms into the deliverable: beats into prose, an outline into an article, a spec into code, a storyboard into a video.

State lives in the artifact's structure, so cross-turn continuity is just re-reading the file; there is no separate log to keep. Choose this when the work is constructive and decomposes, when the user benefits from seeing and shaping the intermediate, and when the final output is a transformation of it. The artifact's schema is the skill's real contract, so design it deliberately and make each section earn its place the same way a SKILL.md does.

The transform is part of the pattern: name where the intermediate ends and the deliverable begins, and whether the transform is a separate intent ("draft from beats") or the tail of the same run.

## Both, and when

Long or high-stakes work uses both: the structured artifact carries the construction state, and a memlog records the decisions about it ("merged beats 3 and 4 for pacing", "cut the subplot — rejected reasoning here"). Reach for both only when the rationale genuinely needs to outlive the conversation and the artifact alone would not explain why it looks the way it does. For most skills, one or neither is enough.

## When none of this applies

A one-shot transform, a stateless utility, or a purely conversational skill keeps no cross-turn state: the input/output contract or the live conversation is all there is. Do not bolt a memlog or an intermediate artifact onto a skill that does one deterministic thing and returns.
