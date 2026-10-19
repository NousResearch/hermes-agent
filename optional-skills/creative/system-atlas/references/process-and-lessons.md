# Process and lessons

When to load: before your first feedback round, a deep dive, or when deciding docs layout — this is the session-by-session record the skill was distilled from, plus the README table, the subagent deep-dive pattern, cost-model habits, and things that bit.

From the session this skill was distilled from: one agent-architecture atlas, built and then reworked across several rounds of feedback.

## How the session actually went — and what to repeat

| Step | What happened | Repeat / avoid |
|---|---|---|
| Inputs | Fetched the vision doc; looked for a whiteboard photo (not attached — asked, moved on); the user forbade one prior-art branch mid-way | Ask which prior art is allowed; never assume |
| Runtime digest | A subagent read the framework's bundled docs against 13 concrete design questions and returned a ~2.5k-word primer with gotchas and a BYO list | Do this before proposing structure; cite the primer's gotchas in the atlas |
| Discussion | Proposed 7 structures mapped to runtime primitives; 7 sharp questions; user answered with one-liners | Take defaults where the user says "defaults are fine" and say which |
| v1 atlas | Whole system at once, 21 structures, two packet flows | Fine as a first draft — but expect "hard to parse" |
| Feedback 1 | "Make it easier via progressive disclosure"; "better box shapes/labelling" | Chapters + role shapes + labels — now the default |
| Text twin | "Keep a text version in context/ADRs" → CONTEXT.md (glossary only), 7 ADRs, SYSTEM.md generated from atlas data, README | Generate the text from the atlas data from day one |
| Question rounds | The user answered by structure; several "this is not a question", "I don't get this — give a concrete example", "this is a stupid question because…" | Explain before resolving; drop non-questions; thank and move on |
| Deep dive | Scope set by the user (two vendors + a simpler DIY); three researchers on one brief with a shared usage model; synthesis with a normalized $/user/month grid; two different model cost bases | Normalize costs so columns carry the same components; fetch prices live (a cached price was wrong by 33%) |
| Rejected proposal | The synthesis proposed a "truth table in Neon, vendor as index"; the user asked what it was, then rejected it as v0 state ("YAGNI"), and later noticed the doc still described it | After a rejection, sweep every file and rewrite — a banner is not enough |
| Brain swap | The user switched the underlying model choice after an earlier decision had already been written up | Sweep every mention of the old choice; re-run the cost model; note which conclusions flip (on a cheap brain, the memory vendor dominates cost) |
| Sprawl | The user: "you now have a ton of competing docs rather than coordinated" and "the atlas is great for me — but not if it's not up to date" | One source file in the repo, one build script, both views generated, README; rebuild + republish every change |

## Docs-folder table (copy into README.md)

| File | Role | Edit it? |
|---|---|---|
| `atlas/data.mjs` | Single source of truth: structures, flows, chapters, decisions, questions, cost model, prose | Yes |
| `atlas/template.html` + `atlas/build.mjs` | Rendering + generator | Presentation only |
| `atlas.html` | Built atlas; republished at the same URL after every rebuild | No (generated) |
| `SYSTEM.md` | Built text twin | No (generated) |
| `CONTEXT.md` | Glossary (domain-modeling convention) | By hand |
| `adr/` | Hard-to-reverse decisions | By hand |
| `research/` | Evidence | Append-only |

## Subagent pattern for deep dives

- Write one `BRIEF.md`: the port/interface we own, requirements that separate candidates, a usage model (scenarios × cadences × fleet sizes) and a fixed deliverable shape (sections, citations, return only a ≤250-word summary + grid + path).
- One subagent per candidate via `delegate_task`, in parallel, writing reports to files; the main agent synthesizes (fit table, normalized cost grid, verdict, per-question resolutions, "considered and rejected").
- Copy reports into the atlas home's `research/`; fold resolutions into `data.mjs` as `{q, r: '… (from the deep dive, date)'}`.

## Cost-model habits

- Always state: model price with fetch date and source, calls per turn, tokens per call (cached vs not), output tokens incl. thinking, turns/day scenarios, fleet sizes.
- Present at least two brain bases if the choice is open; the memory/vendor share of the bill flips with the brain price.
- A nightly job can use a batch API at 50% off; say so.

## Things that bit

- The published file needs `<meta charset="utf-8">` at the top, or the arrows render as mojibake.
- `fitView` with a zero-size rect produced a negative scale once a paused-tab tween resumed; guard and cancel tweens.
- Some in-app browsers render `file://` as a static snapshot (no scripts, no fonts) — serve the folder with a static server and verify there, not from disk.
- Bash heredocs with large HTML/JS are brittle; write files with `write_file`, patch with small Python/Node scripts, and keep the data block as JSON-serializable objects so scripts can mutate it safely.
- Index badges count only open questions; keep IDs stable by never deleting a question — resolve it or mark it dropped.
