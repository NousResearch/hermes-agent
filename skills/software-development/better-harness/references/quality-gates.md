# Findings Quality Gates (Hermes adaptation)

Apply while the lead drafts the report. **Any failed gate blocks rendering** —
even when the draft already has five findings. Inspect every reader-visible
string; "do not copy private data" instructions do not make an already embedded
value safe.

## Finding structure

| Field | Requirement |
| --- | --- |
| `title` | Names a specific observed consequence. No numbers to imply severity; no internal detector/maturity jargon |
| `reason` | Separates fact, inference, owner, and uncertainty |
| `severity` | Low / Medium / High, assigned by the lead only |
| `dimensionRefs` | One primary Agent Work Loop dimension; `subdimensionRefs` names the single primary check |
| `owner` | Smallest owner aligned with the repair (file, module, skill, memory, cron job) |
| `aiFixPrompt` | Scoped repair prompt. Every command/tool/owner named in it was discovered in the target. External writes are explicit authorization preconditions |
| `expectedOutput` | 1–3 verifiable outcomes, not prose |
| `verifier` | The check or command that proves the repair landed |

## The eight gates

1. **Evidence eligibility.** Retain only: an observed consequence, an explicit
   governing requirement, or a deterministic present defect (direct secret,
   malformed active config, exact same-scope collision). Counts, search
   absence, similarity, unavailable evidence, and theoretical risk remain
   leads. Resolve aliases to one canonical asset; discard example literals.

2. **Concrete reader value.** Each title names a concrete observed consequence;
   each reason separates fact / inference / owner / uncertainty. Provider-only
   evidence stays provider-labelled — one lane's absence is not a project
   defect.

3. **Fact consistency.** Repeated counts come from the same canonical envelope.
   Different populations name their scope and measurement basis. Overview,
   dimension scores, findings, and coverage rows do not contradict one another.

4. **Asset accountability.** Every authorized nonzero skill/memory/cron/plugin
   surface appears in coverage. `inspectedSurfaces` contains only content
   actually opened. Inventory never proves selection, use, usefulness, or
   outcome. Zero project skills → trace every repeated-session procedure
   candidate through existing coverage; two distinct comparable Episodes with
   no existing owner require a Low skill-coverage finding, or the draft states
   the evidence reason for not promoting one.

5. **Privacy.** No secret value, raw prompt, stable session id, user-home asset
   path, memory title/path, or private cache layout. Long-session rows keep
   only anonymous aliases, role, duration, and aggregate failures.

6. **Executable repair.** Every command, tool, owner, and capability in an
   `aiFixPrompt` was discovered in the target. Credential repair includes
   revocation/rotation but never assumes an env-var syntax, shell profile, or
   restart route before discovery and separate authority. Memory/skill
   collision repair stays metadata review until same scope and bodies are
   verified; merge or deletion always requires separate authority. Never edit
   generated memory files directly.

7. **Score discipline.** Confirmed memory/skill integrity findings stay pending
   in Asset Health / Repair Progress until independent repair review. Counts,
   configuration, or same-window repair never earn Learning Capture or Loop
   Effectiveness credit. Memory credit requires retrieval, relevance,
   application, and a later improved outcome; skill credit requires selection,
   task-relevant invocation, validation, and a later improved outcome. Uncovered
   applicable demand stays ≤ 59; current-task exercise without later comparison
   ≤ 74; 100 requires a later comparable improved outcome.

8. **Candidate promotion.** New reports do not write "suggestions". A candidate
   that passes normal eligibility becomes an ordinary Low finding with the
   standard consequence, owner, expected output, verifier, and dimension links.
   A try-existing, working-pattern, loop, or horizon opportunity without a
   current consequence stays deferred. Skill/memory promotion still requires
   two distinct comparable Task Episodes and the observed → built-in →
   configured → extend → create ladder. Never invent slash-command or
   dollar-command invocation syntax.

## Retain, don't trim

Five findings is a coverage floor for a normal evidence-rich report — never a
target total or deletion threshold. Keep findings beyond five when they have
independent consequences or owners; only the three priority moves are ranked
down. Fewer findings are valid when evidence is sparse. Reject filler, exact
duplicates, and unsupported absence claims rather than deleting eligible
findings for presentation density. Render only after every gate passes and the
working reconciliation accounts for each omitted candidate as an exact merge,
unsupported lead, or explicit defer.

## Support tracks (shapes priority moves only)

| Track | Select when | Bounds |
| --- | --- | --- |
| Bootstrap (0→1) | Initial guidance explicitly requested, or retained findings establish a missing foundational navigation/validation/risk route | |
| Operationalize (1→60) | Relevant mechanisms exist but retained findings show they are not wired into ordinary work or exercised through an outcome | A track shapes at most three priority moves, repair prompts, and reader copy for already-supported findings |
| Optimize (60→100) | Sufficiently complete session evidence contains ≥2 distinct comparable Task Episodes for the repeated goal or friction | Never adds a finding, changes severity, rescales a dimension, or expands authority |
| Undetermined | Evidence required to select a track is unavailable | |
