---
name: rnd-execution
description: Evidence-gated R&D execution for agent-led research work.
version: 0.3.0
author: Abdulbaki (TopcuAbdulbaki), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Research, Experiments, Methodology, Documentation]
    category: research
    related_skills: [grounded-citations]
---

# R&D Execution Skill

Project-agnostic research & development execution discipline: evidence-gated,
single-variable, properly recorded. Project knowledge (paths, data, decisions)
stays in the project's own documents; this skill is METHOD ONLY.

## When to Use
- Running an experiment/research program (hypothesis → experiment → measurement → decision)
- Designing a mechanism to lock down; comparative benchmarks
- "This decision broke — what must be touched?" situations (mutation slip)
- Don't use: one-off scripting/tooling, measurement-free product feature work

## The Seven Rules (Kaideler)
1. TALK FIRST, THEN BUILD: never pick design/threshold/data/protocol silently;
   give a concrete proposal (at file/number granularity) + INTUITION (why it
   should work) + reach agreement.
2. SINGLE VARIABLE: results must be attributable; if things must move together,
   label them "combined".
3. FAIR COMPARISON: same conditions, matched budget (write down WHICH budget),
   locked assumptions stated openly; if fairness cannot be achieved, say
   "not yet comparable" — give no numbers.
4. MEASURE, TAG EVIDENCE — DON'T INFER: every claim carries a source tag
   (measured / read-from-code / recalled) and a second look (control run /
   different seed / micro-test). For EXTERNAL source claims (web/literature),
   use the `grounded-citations` skill (optional): its generated [n] citations +
   Sources section + verbatim evidence gate — its '[unverified]' stamp is the
   counterpart of our 'recalled' tag. If it is not installed, apply the same
   discipline by hand: record the source the moment you FETCH it (e.g. with
   `web_extract`), never invent numbers, never hand-write the Sources list.
5. RECORDS: numeric results go into the permanent results archive; "written but
   never run" is its OWN category; the producing script lives in the repo — if
   there is no home for it, the record is kept with a NOT-LANDED stamp + script
   path.
6. REPORT CONFLICTS: if docs and code clash, verify the code first, speak up,
   get agreement, then fix.
7. COST GATE: before a long run — "which decision will this change" + a time
   estimate; if there is no answer, do not run it.

## Noise Floor and Freezing
- A headline claim needs ≥3 seeds (mean + min–max); a single seed is a
  direction indicator, not a claim.
- Don't break locks: an assumption locked behind a micro-test cannot be changed
  silently; an off-switch + deliberate ablation (a labelled run) is required.
- FREEZE: a measured mechanism does not change until its numbers arrive;
  redesign comes AFTER the numbers.

## Decision Ledger + MUTATION SLIP (when a decision breaks)
- Every locked decision gets a CARD: value, rationale, evidence pointer,
  DEPENDENTS (code/config/test/result/doc) + revisions. Kept by hand, verified
  by lint (do the dependency paths exist — `search_files`; is any decision left
  card-less). BEFORE writing to the record store, RE-READ the existing entry
  fresh (`read_file`); copy-paste the match key (never write from memory), and
  read back after the operation.
- When a decision breaks, open a SLIP: (a) the dependents list + a literal
  search for the value (`search_files` — complements hunting unrecorded
  dependencies), (b) one line per dependent: BREAKS (repair + regression test) /
  RE-TEST (contaminated runs are added to the list) / SUPERSEDE (old results
  STAY IN PLACE with a `superseded: <slip>` stamp — never deleted) / UNTOUCHED,
  (c) COMPOUND EFFECTS: second-order derived inferences also go on the slip,
  (d) the card gets revisions + a top-level warning, (e) closure criterion:
  every line except UNTOUCHED is processed.
- The slip is a permanent record (receipts folder); the next session continues
  from the slip.

## Session Ritual and Document Hygiene
- Opening: read status + the living handbook (`read_file`), restate status in
  5 lines, confirm, then enter.
- Closing: APPEND to the handbook; if the work is finished, DIGEST-AND-CLEAN —
  records are digested into their proper homes (results archive / history log /
  idea notebook / slips) and DELETED from the handbook: one handbook, no junk
  drawer.
- Every answer ends with numbered OPEN QUESTIONS; deferred lines fall there.
- Make noise before irreversible work (deletion / overwriting / long runs).
- Postpone side quests (sharing/distribution/logistics) to the END; don't
  branch in the main flow — don't open new workstreams unless the user
  explicitly asks. Genuine agent tricks (locking, orchestration) get a 3–4
  line explanation of what they do the first time they appear; mysterious
  behavior is not wanted.
- VERBATIM records (conversation transcripts, decision quotations) are
  UNTOUCHABLE; pointers are updated only in living documents.

## Pitfalls
<!-- no-tmp: ok — names the anti-pattern this rule tells authors to avoid -->
- Binding evidence to a temp directory (/tmp and kin): it disappears (an
  experienced loss). The results archive is permanent.
- Keeping duplicate files: copies rot and mislead — one source + generated
  views.
- Pipeline/summary commands swallow exit codes (run them with pipefail via
  `terminal`); mind measurement BEFORE/AFTER ordering (measure size after
  rendering). COMMIT GATE: if validation/lint is not truly exit==0, the commit
  step DOES NOT RUN — a commit that passes while red erodes the credibility of
  the records.
- WRITE LOCK: in multi-file operations, run ALL validations (scope,
  classification, loss-equality in ONE unit) before writing anything; on
  mismatch, no file is touched (assertions fire before the write step). A bad
  split/operation can then never corrupt a file; fix one line and re-run.
- A path-changing operation (move/rename/delete) and a write to that path
  cannot happen AT THE SAME TIME: the write targets the old path and silently
  drops. Sequence the steps; don't parallelize dependent work in one stack.
- `git rm` (via `terminal`) aborts the WHOLE command on a single untracked path
  in its list; delete untracked paths directly, tracked ones via git. If you
  work in an ignored directory, measure tracking state first with
  `git ls-files` / `git check-ignore`.
- Mixing measurement units (lines / newlines / bytes) — compare in ONE unit.
- Keep a test file's fixtures separate from its own asserts; surface
  collisions before writing code.

## Verification
- Record-integrity lint is green (manifest↔files, links, budget gates).
- COVER TEST: can a fresh zero-context agent find the right answer in ≤N files
  using the protocol + a short question list? If not, routing is broken — fix
  and repeat. Questions must be single-answer: pin the source/period name in
  the question (WHICH of the day's two runs) — ambiguous questions pollute the
  exam. Run the SAME exam through TWO independent fresh agents (`delegate_task`
  or fresh sessions): acceptance = 2/2 (a single pass may be luck, not
  determinism).
- Source-tag audit: every claim carries a tag (measured / read-from-code /
  recalled); no untagged claim remains.