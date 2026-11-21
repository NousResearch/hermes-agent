# Headless Mode

Load this file ONLY when bmad-brainstorming is invoked headless. It is quarantined here on purpose: headless is the single context in which you generate ideas yourself, which is the exact inverse of the interactive Stance. Loading it in a normal session would corrupt the facilitation. Follow it for the whole run.

## Detection

**If a human is sending messages in this session, you are interactive — no payload shape or phrasing overrides that.** Headless requires the *absence* of an interactive user. It is in effect only when one of these unambiguous machine signals holds:

- the caller sets a `headless: true` flag (or the equivalent argument the harness exposes),
- the invocation comes from another skill or a non-interactive runner (no TTY, no user message stream),
- `{workflow.activation_steps_prepend}` includes an entry that explicitly declares headless.

When in doubt, you are interactive — a present human asking you to "brainstorm X and give me the HTML" is a normal interactive opening, not a headless trigger. Facilitate them; do not brainstorm for them.

## The inversion

There is no user to draw ideas out of, so you become the brainstormer. Run a real divergent session against the supplied topic: discover techniques with `uv run {skill-root}/scripts/brain.py --file {workflow.brain_methods} list --all` (the whole catalog is fine here — you are generating, not pacing a user; add `show "<name>"` for a technique's full method on demand), plus any `{workflow.additional_techniques}`, preferring `{workflow.favorite_techniques}` where they fit; work them, and **shift the creative domain every ~10 ideas** exactly as the interactive Stance demands — technical, then experiential, then business, then failure modes, then wildcards. Push past the obvious; the same quantity ambition (aim past 100) and anti-clustering discipline apply. The only thing that changes is that the ideas are now yours to generate. This relaxation is scoped entirely to this file — it never applies to interactive sessions.

## Inputs the caller is expected to provide

Free-form structured payload in the first message; provide what applies:

- `topic` — what to brainstorm. Required. If absent and uninferable, halt `blocked`.
- `goal` — desired outcome / framing, if any.
- `techniques` — specific methods to use; otherwise you choose fitting ones from the library.
- `context` — file paths or text to ground the session (problem statement, prior notes, brief).
- `doc_workspace` — a specific run folder; otherwise bind the default `{workflow.output_dir}/{workflow.output_folder_name}/`.
- `artifacts` — which outputs to produce: `html`, `intent`, or both. Default: both.

## Run

1. Bind `{doc_workspace}` and create the memlog with `uv run {project-root}/_bmad/scripts/memlog.py init --workspace {doc_workspace} --field topic="<topic>" [--field goal="<goal>"]`. It remains the canonical source every artifact derives from.
2. Run the divergent session per **The inversion**, capturing each idea with `uv run {project-root}/_bmad/scripts/memlog.py append --workspace {doc_workspace} --type idea --text "<idea>"` as it lands, and marking each technique switch with `uv run {project-root}/_bmad/scripts/memlog.py append --workspace {doc_workspace} --type technique --text "started <name>"`.
3. Synthesize: surface the conclusions, connections, and the few directions that matter; record them with `uv run {project-root}/_bmad/scripts/memlog.py append --workspace {doc_workspace} --type insight --text "<insights>"`, then run `uv run {project-root}/_bmad/scripts/memlog.py set --workspace {doc_workspace} --key status --value complete`.
4. Produce the requested artifacts from the log — `brainstorm.html` (the imaginative, self-contained, no-template report) and/or the succinct `brainstorm-intent.md` — the same artifacts `references/finalize.md` describes, delegating each to a subagent that reads the log as its sole source. (Headless produces the `artifacts` payload directly; it does not ask, unlike the interactive opt-in.)
5. Execute each entry in `{workflow.external_handoffs}` (capture returned URLs/IDs into the JSON `external_handoffs` array; skip and flag unavailable tools — local files always exist). Then run `{workflow.on_complete}` if non-empty.

Do not ask questions; do not greet. Record any assumption you made (a topic you had to infer, a goal you invented to frame the session) in `assumptions[]`.

## Return

End with a JSON status block. Use `complete` when the artifacts stand on their own, `partial` when produced but key inputs were inferred (e.g. topic was thin), `blocked` when no artifact was produced (e.g. no topic). Omit keys for artifacts not produced.

```json
{
  "status": "complete",
  "intent": "brainstorm",
  "memlog": "{doc_workspace}/.memlog.md",
  "html": "{doc_workspace}/brainstorm.html",
  "intent_doc": "{doc_workspace}/brainstorm-intent.md",
  "assumptions": [],
  "external_handoffs": []
}
```
